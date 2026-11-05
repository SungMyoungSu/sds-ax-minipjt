"""시간 집계 엔진: 기준 기간 계산, 활동별 상태 판정, 프로필별 집계.

저장소에 의존하지 않는 순수 함수로 둔다. 입력은 프로필, 활동 목록, 규칙 확정 여부, 오늘 날짜다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from models import Activity, Profile

ActivityStatus = Literal["recognized", "pending_proof", "excluded", "rules_unconfirmed"]

STATUS_LABELS: dict[str, str] = {
    "recognized": "인정",
    "pending_proof": "확인서 대기",
    "excluded": "제외",
    "rules_unconfirmed": "규칙 미확정",
}


# ---- 기준 기간 ----

@dataclass(frozen=True)
class Period:
    start: date
    end: date

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


def _add_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 2/29 → 평년은 2/28
        return day.replace(year=day.year + years, day=28)


def period_for(profile: Profile, day: date) -> Period | None:
    """day가 속한 기준 기간. 프로필에 기간이 없으면 None(기간 구분 없이 전체 집계).

    저장된 기간을 1년씩 밀어 day가 들어가는 기간을 찾는다. 어느 기간에도 들어가지 않으면
    앞 기간 끝 다음 날부터 다음 기간 시작 전날까지를 기간으로 본다.
    예: 1/1~12/31은 매년 반복, 학생 3/1~8/31이면 그 사이 9/1~2월 말이 2학기가 된다.
    """
    if profile.period_start is None or profile.period_end is None:
        return None
    base_start, base_end = profile.period_start, profile.period_end
    k0 = day.year - base_start.year
    shifted = [
        Period(_add_years(base_start, k), _add_years(base_end, k)) for k in range(k0 - 2, k0 + 3)
    ]
    for p in shifted:
        if p.contains(day):
            return p
    prev_end = max(p.end for p in shifted if p.end < day)
    next_start = min(p.start for p in shifted if p.start > day)
    return Period(prev_end + timedelta(days=1), next_start - timedelta(days=1))


# ---- 활동 상태 ----

def activity_status(profile: Profile, activity: Activity, rules_confirmed: bool) -> ActivityStatus:
    if profile.source == "company_policy" and not rules_confirmed:
        return "rules_unconfirmed"
    if profile.allowed_type_codes and activity.type_code not in profile.allowed_type_codes:
        return "excluded"
    if profile.certificate_required and not activity.certificate_path:
        return "pending_proof"
    return "recognized"


# ---- 집계 ----

@dataclass
class ActivityLine:
    activity_id: int | None
    start_date: date
    organization: str | None
    hours: float
    type_code: str
    status: ActivityStatus
    school_planned: bool = False
    major_related: bool = False

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]


@dataclass
class Progress:
    profile_id: int | None
    user_type: str
    source: str
    period: Period | None
    rules_confirmed: bool
    goal_hours: float | None
    recognized_hours: float = 0.0
    pending_proof_hours: float = 0.0
    excluded_hours: float = 0.0
    unconfirmed_hours: float = 0.0
    total_hours: float = 0.0  # 상태와 관계없이 이 기간 대상 활동의 합
    remaining_hours: float | None = None
    percent: float | None = None
    # 목표 없는 학생용 (PRD 23)
    hours_by_type: dict[str, float] = field(default_factory=dict)
    major_related_count: int = 0
    school_planned_hours: float = 0.0
    personal_hours: float = 0.0
    lines: list[ActivityLine] = field(default_factory=list)

    @property
    def rules_pending(self) -> bool:
        """회사 프로필인데 규칙이 확정되지 않음 → 남은 시간을 계산하지 않고 '규칙 미확정'으로 표시."""
        return self.source == "company_policy" and not self.rules_confirmed


def _r(x: float) -> float:
    return round(x + 0.0, 2)


def compute_progress(
    profile: Profile,
    activities: list[Activity],
    rules_confirmed: bool,
    today: date,
    period: Period | None = None,
) -> Progress:
    """프로필 하나의 기간 집계. 모든 활동이 모든 활성 프로필의 집계 대상이고(AC3 d), 기간 밖 활동은 뺀다.
    period를 주지 않으면 today가 속한 기간."""
    period = period or period_for(profile, today)
    result = Progress(
        profile_id=profile.id,
        user_type=profile.user_type,
        source=profile.source,
        period=period,
        rules_confirmed=rules_confirmed,
        goal_hours=profile.goal_hours,
    )
    by_type: dict[str, float] = defaultdict(float)
    for a in activities:
        if period is not None and not period.contains(a.start_date):
            continue
        status = activity_status(profile, a, rules_confirmed)
        major_related = bool(a.major_relevance and a.major_relevance.strip())
        result.lines.append(ActivityLine(
            activity_id=a.id, start_date=a.start_date, organization=a.organization, hours=a.hours,
            type_code=a.type_code, status=status, school_planned=a.school_planned,
            major_related=major_related,
        ))
        result.total_hours += a.hours
        if status == "recognized":
            result.recognized_hours += a.hours
            by_type[a.type_code] += a.hours
            if major_related:
                result.major_related_count += 1
            if a.school_planned:
                result.school_planned_hours += a.hours
            else:
                result.personal_hours += a.hours
        elif status == "pending_proof":
            result.pending_proof_hours += a.hours
        elif status == "excluded":
            result.excluded_hours += a.hours
        else:
            result.unconfirmed_hours += a.hours

    for name in ("recognized_hours", "pending_proof_hours", "excluded_hours", "unconfirmed_hours",
                 "total_hours", "school_planned_hours", "personal_hours"):
        setattr(result, name, _r(getattr(result, name)))
    result.hours_by_type = {k: _r(v) for k, v in sorted(by_type.items())}

    if profile.goal_hours and not result.rules_pending:
        result.remaining_hours = _r(max(profile.goal_hours - result.recognized_hours, 0.0))
        result.percent = round(min(result.recognized_hours / profile.goal_hours * 100, 100.0), 1)
    return result
