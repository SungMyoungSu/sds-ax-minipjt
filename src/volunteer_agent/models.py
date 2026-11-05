"""저장 개체 (seed `ontology_schema` 기준).

활동의 상태(인정·확인서 대기·제외·규칙 미확정)는 저장하지 않는다. 프로필과 확정 규칙에 따라 달라지므로
집계 엔진(TASK 3단계)이 그때그때 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

UserType = Literal["employee", "student"]
ProfileSource = Literal["company_policy", "school_requirement", "self_goal", "none"]
RuleStatus = Literal["candidate", "confirmed"]
PolicyFormat = Literal["md", "pdf"]
DeadlineKind = Literal["period_end", "recruit_end", "schedule"]
NotificationMark = Literal["D-30", "D-7", "D-1"]
NotificationChannel = Literal["smtp", "eml"]

USER_TYPES: tuple[str, ...] = ("employee", "student")
PROFILE_SOURCES: tuple[str, ...] = ("company_policy", "school_requirement", "self_goal", "none")
RULE_STATUSES: tuple[str, ...] = ("candidate", "confirmed")
POLICY_FORMATS: tuple[str, ...] = ("md", "pdf")
DEADLINE_KINDS: tuple[str, ...] = ("period_end", "recruit_end", "schedule")
NOTIFICATION_MARKS: tuple[str, ...] = ("D-30", "D-7", "D-1")
NOTIFICATION_CHANNELS: tuple[str, ...] = ("smtp", "eml")


@dataclass
class UserSetting:
    user_type: UserType
    created_at: datetime


@dataclass
class Profile:
    user_type: UserType
    source: ProfileSource = "none"
    goal_hours: float | None = None
    period_start: date | None = None
    period_end: date | None = None
    certificate_required: bool = False
    allowed_type_codes: list[str] = field(default_factory=list)  # 비어 있으면 분야 제한 없음
    major: str | None = None
    active: bool = True
    id: int | None = None
    created_at: datetime | None = None
    ended_at: datetime | None = None


@dataclass
class PolicyClause:
    clause_id: str  # 예: "제3조", 없으면 "p2-1" 같은 위치 표기
    text: str  # 원문 그대로
    position: int = 0
    page: int | None = None
    id: int | None = None
    document_id: int | None = None


@dataclass
class PolicyDocument:
    filename: str
    format: PolicyFormat
    clauses: list[PolicyClause] = field(default_factory=list)
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class RuleCandidate:
    profile_id: int
    document_id: int | None = None
    status: RuleStatus = "candidate"
    goal_hours: float | None = None
    period_start: date | None = None
    period_end: date | None = None
    allowed_type_codes: list[str] = field(default_factory=list)
    certificate_required: bool | None = None
    # 항목별 근거: {"goal_hours": {"quote": "...", "filename": "...", "clause_id": "..."}, ...}
    # 규정에 없는 항목은 {"not_in_policy": true}
    source_quotes: dict[str, dict[str, Any]] = field(default_factory=dict)
    id: int | None = None
    created_at: datetime | None = None
    confirmed_at: datetime | None = None


@dataclass
class Activity:
    start_date: date
    hours: float
    type_code: str
    end_date: date | None = None
    organization: str | None = None
    content: str | None = None
    certificate_path: str | None = None
    major_relevance: str | None = None
    lesson: str | None = None
    school_planned: bool = False
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Deadline:
    kind: DeadlineKind
    due_date: date
    title: str = ""
    ref_id: str | None = None  # period_end는 profile id, recruit_end는 program_id, schedule은 activity id
    id: int | None = None


@dataclass
class WatchlistItem:
    program_id: str
    title: str
    url: str | None = None
    recruit_end: date | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class Plan:
    profile_id: int
    content: dict[str, Any]  # VolunteerPlan(target_hours, weeks[], expected_total, shortfall_hours, rationale)
    approved: bool = False
    approved_at: datetime | None = None
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class Notification:
    deadline_id: int
    mark: NotificationMark
    channel: NotificationChannel
    path: str | None = None  # channel이 eml이면 저장한 .eml 경로
    id: int | None = None
    sent_at: datetime | None = None
