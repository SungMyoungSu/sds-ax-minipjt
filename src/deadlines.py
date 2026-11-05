"""마감 관리: 기간 종료 마감 동기화, 다가오는 마감 조회, .ics 내보내기 (결정적 코드).

안내 대상 마감은 두 가지뿐이다(PRD 34):
  (1) 관심 목록 봉사의 모집 마감(recruit_end, 관심 목록에 넣을 때 등록)
  (2) 목표가 있고 아직 미달인 기준 기간 종료일(period_end, 여기서 동기화)
예정 활동 일정과 확인서 대기 독촉은 안내하지 않는다. 규칙 미확정 회사 프로필은 기간 종료 알림이 없다(미결 결정 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from icalendar import Alarm, Calendar, Event

from config import Settings
from models import Deadline, Profile
from progress import Progress, compute_progress
from store import Store

UPCOMING_DAYS = 7
ALARM_DAYS = (30, 7, 1)
CALENDAR_DIR = "calendar"
ICS_FILENAME = "deadlines.ics"
KIND_LABELS = {"recruit_end": "모집 마감", "period_end": "기준 기간 종료"}
EVENT_HOUR = 9  # 마감 일정은 그날 09:00(Asia/Seoul) 30분 일정으로 넣는다


@dataclass
class DeadlineView:
    deadline: Deadline
    days_left: int
    remaining_hours: float | None = None  # period_end일 때 남은 시간

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.deadline.kind, self.deadline.kind)

    def line(self) -> str:
        d = self.deadline
        extra = f", 남은 {self.remaining_hours:g}시간" if self.remaining_hours is not None else ""
        return f"{d.due_date.isoformat()} (D-{self.days_left}) {self.label}: {d.title}{extra}"


@dataclass
class UpcomingResult:
    items: list[DeadlineView] = field(default_factory=list)  # days 이내, 임박순
    period_status: list[DeadlineView] = field(default_factory=list)  # 미달 기간 종료(거리 무관)
    days: int = UPCOMING_DAYS

    @property
    def message(self) -> str:
        lines = [f"{self.days}일 이내 마감:"]
        lines += [f"- {v.line()}" for v in self.items] or ["- 없음"]
        if self.period_status:
            lines.append("이번 기준 기간 진행:")
            lines += [f"- {v.line()}" for v in self.period_status]
        return "\n".join(lines)


def _rules_confirmed(store: Store, profile: Profile) -> bool:
    return bool(store.list_rules(profile_id=profile.id, status="confirmed"))


def period_progress(store: Store, today: date) -> list[tuple[Profile, Progress]]:
    activities = store.list_activities()
    return [
        (p, compute_progress(p, activities, _rules_confirmed(store, p), today))
        for p in store.list_profiles(active_only=True)
    ]


def sync_period_deadlines(store: Store, today: date) -> dict[int, float]:
    """목표 미달인 기간 종료 마감만 남도록 맞춘다. {deadline id: 남은 시간}을 돌려준다.
    기간이 바뀌면 이전 마감은 지우고 새 마감을 만든다(알림 기록도 새로 시작)."""
    wanted: dict[str, tuple[date, str, float]] = {}
    for profile, prog in period_progress(store, today):
        if prog.rules_pending or not prog.goal_hours or prog.period is None:
            continue
        if prog.remaining_hours and prog.remaining_hours > 0:
            title = f"{'회사' if profile.source == 'company_policy' else '개인'} 봉사 목표 {prog.goal_hours:g}시간"
            wanted[str(profile.id)] = (prog.period.end, title, prog.remaining_hours)

    remaining: dict[int, float] = {}
    for d in store.list_deadlines():
        if d.kind != "period_end":
            continue
        want = wanted.get(d.ref_id or "")
        if want is None or want[0] != d.due_date:
            store.delete_deadline(d.id)
        else:
            remaining[d.id] = want[2]
            del wanted[d.ref_id]
    for ref_id, (due, title, hours) in wanted.items():
        d = store.add_deadline(Deadline(kind="period_end", due_date=due, title=title, ref_id=ref_id))
        remaining[d.id] = hours
    return remaining


def active_deadlines(store: Store, today: date) -> list[DeadlineView]:
    """오늘 이후(오늘 포함)의 안내 대상 마감 전체, 임박순."""
    remaining = sync_period_deadlines(store, today)
    views = [
        DeadlineView(d, (d.due_date - today).days, remaining.get(d.id))
        for d in store.list_deadlines()
        if d.due_date >= today and d.kind in KIND_LABELS
    ]
    return sorted(views, key=lambda v: (v.deadline.due_date, v.deadline.id))


def get_upcoming_deadlines(store: Store, today: date, days: int = UPCOMING_DAYS) -> UpcomingResult:
    """days일 이내 마감을 임박순으로(AC6). 미달인 기간 종료는 거리와 관계없이 따로 보여 준다."""
    views = active_deadlines(store, today)
    return UpcomingResult(
        items=[v for v in views if v.days_left <= days],
        period_status=[v for v in views if v.deadline.kind == "period_end"],
        days=days,
    )


# ---- .ics ----

def _uid(d: Deadline) -> str:
    """같은 마감은 다시 내보내도 같은 UID(캘린더에서 중복되지 않게)."""
    return f"deadline-{d.kind}-{d.ref_id or d.id}-{d.due_date:%Y%m%d}@volunteer-agent.local"


def build_calendar(views: list[DeadlineView], settings: Settings, now: datetime | None = None) -> Calendar:
    tz = settings.tz
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cal = Calendar()
    cal.add("prodid", "-//volunteer-agent//deadlines//KO")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-timezone", settings.app_timezone)
    for v in views:
        d = v.deadline
        start = datetime.combine(d.due_date, time(EVENT_HOUR), tzinfo=tz)
        ev = Event()
        ev.add("uid", _uid(d))
        ev.add("dtstamp", stamp)
        ev.add("dtstart", start)
        ev.add("dtend", start + timedelta(minutes=30))
        ev.add("summary", f"[{v.label}] {d.title}")
        desc = f"{v.label}: {d.due_date.isoformat()}"
        if v.remaining_hours is not None:
            desc += f"\n남은 시간: {v.remaining_hours:g}시간"
        ev.add("description", desc)
        for n in ALARM_DAYS:
            alarm = Alarm()
            alarm.add("action", "DISPLAY")
            alarm.add("description", f"{v.label} D-{n}: {d.title}")
            alarm.add("trigger", timedelta(days=-n))
            ev.add_component(alarm)
        cal.add_component(ev)
    cal.add_missing_timezones()
    return cal


def export_ics(store: Store, settings: Settings, today: date, now: datetime | None = None) -> tuple[Path, int]:
    """안내 대상 마감 전체를 DATA_DIR/calendar/deadlines.ics로 쓴다. (경로, 일정 수)."""
    views = active_deadlines(store, today)
    folder = settings.ensure_data_dir() / CALENDAR_DIR
    folder.mkdir(exist_ok=True)
    path = folder / ICS_FILENAME
    path.write_bytes(build_calendar(views, settings, now).to_ical())
    return path, len(views)
