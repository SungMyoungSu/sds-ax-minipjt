"""기록 수명주기: 입력 검증, 기록 추가·수정·삭제, 확인서 첨부, 프로필.

MCP 도구(TASK 10단계)가 이 함수를 감싸 쓴다. LLM을 부르지 않는 결정적 코드다.
오늘 날짜는 항상 인자로 받는다(`--today` 대응).
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Literal

from volunteer_agent.config import Settings
from volunteer_agent.models import USER_TYPES, Activity, Profile
from volunteer_agent.progress import Progress, compute_progress
from volunteer_agent.store import Store
from volunteer_agent.type_codes import load_type_codes

CERTIFICATE_DIR = "certificates"
CERTIFICATE_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png")
REQUIRED_FIELDS = ("start_date", "organization", "hours", "type_code")
FIELD_LABELS = {"start_date": "날짜", "organization": "기관", "hours": "시간", "type_code": "분야"}


class InputError(ValueError):
    """입력 오류. 메시지는 사용자에게 그대로 보여 줄 수 있는 '입력 오류: …' 문장이다.
    (허위 기록 거절 고정 문구와 다르다. 가드레일이 아니라 입력 검증이다.)"""

    def __init__(self, reason: str):
        super().__init__(f"입력 오류: {reason}")


@dataclass
class ActivityInput:
    start_date: date | None = None
    organization: str | None = None
    hours: float | None = None
    type_code: str | None = None
    end_date: date | None = None
    content: str | None = None
    major_relevance: str | None = None
    lesson: str | None = None
    school_planned: bool = False
    certificate_file: str | Path | None = None  # 함께 첨부할 확인서 파일(선택)


def missing_fields(inp: ActivityInput) -> list[str]:
    """빠진 필수 항목(날짜, 기관, 시간, 분야). 이 항목만 되묻는다 (PRD 16)."""
    missing = []
    for name in REQUIRED_FIELDS:
        value = getattr(inp, name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    return missing


def validate_activity(inp: ActivityInput, today: date) -> None:
    """필수 항목이 다 있는 입력을 검증한다. 문제가 있으면 InputError (PRD 20, AC5, AC23)."""
    hours = inp.hours
    if not isinstance(hours, (int, float)) or isinstance(hours, bool) or not math.isfinite(hours):
        raise InputError("봉사 시간은 숫자로 입력해 주세요.")
    if hours <= 0:
        raise InputError("봉사 시간은 0보다 커야 합니다.")
    if not math.isclose(hours * 2, round(hours * 2), abs_tol=1e-9):
        raise InputError("봉사 시간은 0.5시간(30분) 단위로 입력해 주세요.")

    start, end = inp.start_date, inp.end_date or inp.start_date
    if end < start:
        raise InputError("끝난 날짜가 시작 날짜보다 앞설 수 없습니다.")
    if start > today or end > today:
        raise InputError(
            f"미래 날짜({max(start, end).isoformat()})의 활동은 기록할 수 없습니다. 활동을 마친 뒤 기록해 주세요."
        )
    days = (end - start).days + 1
    if hours > 24 * days:
        raise InputError("하루 24시간을 넘는 시간은 기록할 수 없습니다.")

    table = load_type_codes()
    if not table.is_valid(inp.type_code):
        names = ", ".join(f"{c} {n}" for c, n in table.codes.items())
        raise InputError(f"목록에 없는 분야 코드입니다({inp.type_code}). 사용할 수 있는 분야: {names}")


# ---- 결과 ----

@dataclass
class LogResult:
    status: Literal["saved", "missing_fields", "duplicate"]
    activity: Activity | None = None
    missing: list[str] = field(default_factory=list)
    duplicate_of: int | None = None
    progress: list[Progress] = field(default_factory=list)

    @property
    def missing_labels(self) -> list[str]:
        return [FIELD_LABELS[m] for m in self.missing]


@dataclass
class ChangeResult:
    status: Literal["updated", "deleted", "needs_confirmation", "attached"]
    activity: Activity
    progress: list[Progress] = field(default_factory=list)


# ---- 서비스 ----

class RecordService:
    def __init__(self, store: Store, settings: Settings, today: date):
        self.store = store
        self.settings = settings
        self.today = today

    # -- 공통 --

    def rules_confirmed(self, profile: Profile) -> bool:
        return bool(self.store.list_rules(profile_id=profile.id, status="confirmed"))

    def progress(self, profile_id: int | None = None) -> list[Progress]:
        """활성 프로필별 이번 기간 집계(profile_id를 주면 그 프로필만)."""
        profiles = (
            [self.store.get_profile(profile_id)] if profile_id is not None
            else self.store.list_profiles(active_only=True)
        )
        activities = self.store.list_activities()
        return [
            compute_progress(p, activities, self.rules_confirmed(p), self.today) for p in profiles
        ]

    # -- 프로필 --

    def create_profile(
        self,
        user_type: str,
        source: str | None = None,
        goal_hours: float | None = None,
        period_start: date | None = None,
        period_end: date | None = None,
        major: str | None = None,
    ) -> Profile:
        """기준 프로필 생성. 회사원은 기본 company_policy(목표·기간은 규칙 확정 때 채움).
        학생은 학기 날짜가 없으면 오늘이 속한 학기(3/1~8/31, 9/1~2월 말)로 정한다(미결 결정 3)."""
        if user_type not in USER_TYPES:
            raise InputError(f"사용자 유형은 회사원 또는 학생이어야 합니다({user_type}).")
        if goal_hours is not None and goal_hours <= 0:
            raise InputError("목표 시간은 0보다 커야 합니다.")
        if source is None:
            source = "company_policy" if user_type == "employee" else (
                "self_goal" if goal_hours else "none")
        if user_type == "student" and period_start is None and period_end is None:
            period_start, period_end = default_semester(self.today)
        if (period_start is None) != (period_end is None):
            raise InputError("기간은 시작일과 종료일을 함께 입력해 주세요.")
        if period_start and period_end and period_end < period_start:
            raise InputError("기간 종료일이 시작일보다 앞설 수 없습니다.")
        return self.store.add_profile(Profile(
            user_type=user_type, source=source, goal_hours=goal_hours,
            period_start=period_start, period_end=period_end, major=major,
        ))

    # -- 기록 --

    def find_duplicate(self, start_date: date, organization: str, hours: float,
                       exclude_id: int | None = None) -> Activity | None:
        org = organization.strip()
        for a in self.store.list_activities():
            if (a.id != exclude_id and a.start_date == start_date
                    and (a.organization or "").strip() == org and math.isclose(a.hours, hours)):
                return a
        return None

    def log_activity(self, inp: ActivityInput, confirm_duplicate: bool = False) -> LogResult:
        missing = missing_fields(inp)
        if missing:
            return LogResult(status="missing_fields", missing=missing)
        validate_activity(inp, self.today)
        if inp.certificate_file is not None:
            _check_certificate_file(inp.certificate_file)
        if not confirm_duplicate:
            dup = self.find_duplicate(inp.start_date, inp.organization, inp.hours)
            if dup is not None:
                return LogResult(status="duplicate", duplicate_of=dup.id)

        activity = self.store.add_activity(Activity(
            start_date=inp.start_date, end_date=inp.end_date, organization=inp.organization.strip(),
            content=inp.content, hours=float(inp.hours), type_code=inp.type_code,
            major_relevance=inp.major_relevance, lesson=inp.lesson,
            school_planned=inp.school_planned,
        ))
        if inp.certificate_file is not None:
            activity = self._copy_certificate(activity, inp.certificate_file)
        return LogResult(status="saved", activity=activity, progress=self.progress())

    def update_activity(self, activity_id: int, **changes: object) -> ChangeResult:
        """대상 활동을 확인한 뒤 바로 반영한다 (PRD 21). 바꿀 수 있는 항목만 받는다."""
        allowed = {"start_date", "end_date", "organization", "content", "hours", "type_code",
                   "major_relevance", "lesson", "school_planned"}
        unknown = set(changes) - allowed
        if unknown:
            raise InputError(f"수정할 수 없는 항목입니다({', '.join(sorted(unknown))}).")
        current = self.store.get_activity(activity_id)
        updated = replace(current, **changes)
        inp = ActivityInput(start_date=updated.start_date, organization=updated.organization,
                            hours=updated.hours, type_code=updated.type_code,
                            end_date=updated.end_date)
        missing = missing_fields(inp)
        if missing:
            raise InputError(f"{', '.join(FIELD_LABELS[m] for m in missing)}은(는) 비울 수 없습니다.")
        validate_activity(inp, self.today)
        updated.hours = float(updated.hours)
        activity = self.store.update_activity(updated)
        return ChangeResult(status="updated", activity=activity, progress=self.progress())

    def delete_activity(self, activity_id: int, confirm: bool = False) -> ChangeResult:
        """삭제는 한 번 더 확인한 뒤 지운다 (PRD 21). confirm=False면 지우지 않는다."""
        activity = self.store.get_activity(activity_id)
        if not confirm:
            return ChangeResult(status="needs_confirmation", activity=activity)
        self.store.delete_activity(activity_id)
        return ChangeResult(status="deleted", activity=activity, progress=self.progress())

    def attach_certificate(self, activity_id: int, file_path: str | Path) -> ChangeResult:
        """확인서를 DATA_DIR/certificates로 복사해 붙인다. 확인서 대기였다면 인정으로 바뀐다."""
        activity = self.store.get_activity(activity_id)
        _check_certificate_file(file_path)
        activity = self._copy_certificate(activity, file_path)
        return ChangeResult(status="attached", activity=activity, progress=self.progress())

    def _copy_certificate(self, activity: Activity, file_path: str | Path) -> Activity:
        src = Path(file_path)
        folder = self.settings.ensure_data_dir() / CERTIFICATE_DIR
        folder.mkdir(exist_ok=True)
        dest = folder / f"{activity.id}_{src.name}"
        shutil.copyfile(src, dest)
        activity.certificate_path = f"{CERTIFICATE_DIR}/{dest.name}"  # DATA_DIR 기준 상대 경로
        return self.store.update_activity(activity)


def default_semester(today: date) -> tuple[date, date]:
    """오늘이 속한 기본 학기: 3/1~8/31 또는 9/1~다음 해 2월 말."""
    if 3 <= today.month <= 8:
        return date(today.year, 3, 1), date(today.year, 8, 31)
    start_year = today.year if today.month >= 9 else today.year - 1
    end = date(start_year + 1, 3, 1).toordinal() - 1
    return date(start_year, 9, 1), date.fromordinal(end)


def _check_certificate_file(file_path: str | Path) -> None:
    path = Path(file_path)
    if path.suffix.lower() not in CERTIFICATE_EXTENSIONS:
        raise InputError("확인서는 PDF, JPG, PNG 파일만 첨부할 수 있습니다.")
    if not path.is_file():
        raise InputError(f"확인서 파일을 찾을 수 없습니다({path.name}).")
