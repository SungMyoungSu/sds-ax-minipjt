import sqlite3
from datetime import date

import pytest

from models import (
    Activity,
    Deadline,
    Notification,
    Plan,
    PolicyClause,
    PolicyDocument,
    Profile,
    RuleCandidate,
    WatchlistItem,
)
from store import DB_FILENAME, SCHEMA_VERSION, Store, StoreError

TABLES = {
    "user_settings", "profiles", "policy_documents", "policy_clauses", "rule_candidates",
    "activities", "deadlines", "watchlist", "plans", "notifications",
}


def _profile(store):
    return store.add_profile(Profile(user_type="employee", source="company_policy"))


def test_schema_created_in_data_dir(settings, store):
    assert (settings.data_dir / DB_FILENAME).is_file()
    names = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= names
    assert store.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_reopen_keeps_data(settings, clock):
    with Store.open(settings, now=clock) as s:
        s.init_user_setting("employee")
        _profile(s)
    with Store.open(settings, now=clock) as s:
        assert s.get_user_setting().user_type == "employee"
        assert len(s.list_profiles()) == 1


# ---- 사용자 설정 ----

def test_user_setting_saved_only_once(store, clock):
    assert store.get_user_setting() is None
    us = store.init_user_setting("student")
    assert us.user_type == "student"
    assert us.created_at == clock.t
    with pytest.raises(StoreError):
        store.init_user_setting("employee")
    assert store.get_user_setting().user_type == "student"


@pytest.mark.parametrize("bad", ["admin", "", "Employee"])
def test_invalid_user_type_rejected(store, bad):
    with pytest.raises(sqlite3.IntegrityError):
        store.init_user_setting(bad)


# ---- 프로필 ----

def test_profile_crud_and_end(store, clock):
    p = store.add_profile(Profile(
        user_type="student", source="school_requirement", goal_hours=20,
        period_start=date(2026, 3, 1), period_end=date(2026, 8, 31),
        allowed_type_codes=["0100", "0500"], major="간호학",
    ))
    assert p.id and p.active and p.ended_at is None
    assert p.allowed_type_codes == ["0100", "0500"]
    assert p.period_end == date(2026, 8, 31)

    p.goal_hours = 25
    p.certificate_required = True
    p = store.update_profile(p)
    assert p.goal_hours == 25 and p.certificate_required is True

    clock.tick()
    ended = store.end_profile(p.id)
    assert ended.active is False and ended.ended_at == clock.t
    assert store.list_profiles(active_only=True) == []
    assert len(store.list_profiles()) == 1  # 지우지 않고 남김


def test_invalid_profile_source_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_profile(Profile(user_type="employee", source="hr"))


def test_missing_id_raises_store_error(store):
    with pytest.raises(StoreError):
        store.get_profile(999)
    with pytest.raises(StoreError):
        store.delete_activity(999)


# ---- 규정 원문과 규칙 ----

def test_policy_document_keeps_clause_text_verbatim(store):
    text = "제3조(연간 의무 시간) 임직원은 매년 1월 1일부터 12월 31일까지 8시간 이상 봉사활동을 하여야 한다."
    doc = store.add_policy_document(PolicyDocument(
        filename="volunteer_policy.md", format="md",
        clauses=[PolicyClause("제1조", "제1조(목적) ..."), PolicyClause("제3조", text)],
    ))
    loaded = store.get_policy_document(doc.id)
    assert [c.clause_id for c in loaded.clauses] == ["제1조", "제3조"]
    assert loaded.clauses[1].text == text
    assert loaded.clauses[1].position == 1
    assert len(store.list_policy_documents()) == 1


def test_invalid_policy_format_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_policy_document(PolicyDocument(filename="a.hwp", format="hwp"))


def test_rule_candidate_then_confirmed(store, clock):
    p = _profile(store)
    doc = store.add_policy_document(PolicyDocument(filename="volunteer_policy.md", format="md"))
    quotes = {
        "goal_hours": {"quote": "8시간 이상", "filename": "volunteer_policy.md", "clause_id": "제3조"},
        "period": {"not_in_policy": True},
    }
    r = store.add_rule_candidate(RuleCandidate(
        profile_id=p.id, document_id=doc.id, goal_hours=8,
        period_start=date(2026, 1, 1), period_end=date(2026, 12, 31),
        allowed_type_codes=["0800"], certificate_required=True, source_quotes=quotes,
    ))
    assert r.status == "candidate" and r.confirmed_at is None
    assert r.source_quotes == quotes
    assert store.list_rules(profile_id=p.id, status="confirmed") == []

    clock.tick()
    c = store.mark_rule_confirmed(r.id)
    assert c.status == "confirmed" and c.confirmed_at == clock.t
    assert [x.id for x in store.list_rules(status="confirmed")] == [r.id]


def test_rule_certificate_required_can_be_unknown(store):
    p = _profile(store)
    r = store.add_rule_candidate(RuleCandidate(profile_id=p.id))
    assert r.certificate_required is None


def test_rule_requires_existing_profile(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_rule_candidate(RuleCandidate(profile_id=999))


# ---- 활동 ----

def test_activity_crud(store, clock):
    a = store.add_activity(Activity(
        start_date=date(2026, 11, 3), hours=2, type_code="0800",
        organization="강남구 자원봉사센터", content="환경정화",
    ))
    assert a.id and a.created_at == a.updated_at == clock.t
    assert a.school_planned is False

    clock.tick()
    a.hours = 1.5
    a.certificate_path = "certificates/1.pdf"
    a = store.update_activity(a)
    assert a.hours == 1.5 and a.certificate_path == "certificates/1.pdf"
    assert a.updated_at == clock.t and a.created_at != a.updated_at

    store.delete_activity(a.id)
    assert store.list_activities() == []


def test_activity_student_fields_and_order(store):
    store.add_activity(Activity(start_date=date(2026, 3, 10), hours=3, type_code="0100", school_planned=True,
                                major_relevance="간호학과 관련", lesson="경청"))
    store.add_activity(Activity(start_date=date(2026, 3, 5), hours=2, type_code="0100"))
    acts = store.list_activities()
    assert [a.start_date for a in acts] == [date(2026, 3, 5), date(2026, 3, 10)]
    assert acts[1].school_planned is True
    assert acts[1].lesson == "경청"


# ---- 마감, 관심 목록, 계획 ----

def test_deadlines_sorted_and_kind_checked(store):
    store.add_deadline(Deadline(kind="period_end", due_date=date(2026, 12, 31), ref_id="1"))
    store.add_deadline(Deadline(kind="recruit_end", due_date=date(2026, 11, 20), ref_id="P-1",
                                title="한강 플로깅"))
    assert [d.kind for d in store.list_deadlines()] == ["recruit_end", "period_end"]
    with pytest.raises(sqlite3.IntegrityError):
        store.add_deadline(Deadline(kind="birthday", due_date=date(2026, 1, 1)))


def test_watchlist_add_is_idempotent_and_removable(store):
    w = store.add_watchlist_item(WatchlistItem(program_id="P-1", title="한강 플로깅",
                                               url="https://www.1365.go.kr/x",
                                               recruit_end=date(2026, 11, 20)))
    again = store.add_watchlist_item(WatchlistItem(program_id="P-1", title="다른 제목"))
    assert again.id == w.id and again.title == "한강 플로깅"
    assert len(store.list_watchlist()) == 1
    assert store.remove_watchlist_item("P-1") is True
    assert store.remove_watchlist_item("P-1") is False


def test_plan_saved_then_approved(store, clock):
    p = _profile(store)
    content = {"target_hours": 4, "weeks": [{"week": 1, "target_hours": 2, "items": []}],
               "expected_total": 4, "shortfall_hours": 0, "rationale": "주말 위주"}
    plan = store.add_plan(Plan(profile_id=p.id, content=content))
    assert plan.approved is False and plan.content == content
    clock.tick()
    approved = store.mark_plan_approved(plan.id)
    assert approved.approved is True and approved.approved_at == clock.t
    assert len(store.list_plans(profile_id=p.id)) == 1


# ---- 알림 기록 ----

def test_notification_recorded_once_per_mark(store, tmp_path):
    d = store.add_deadline(Deadline(kind="recruit_end", due_date=date(2026, 11, 11), ref_id="P-1"))
    assert store.was_notified(d.id, "D-7") is False
    n = store.record_notification(Notification(deadline_id=d.id, mark="D-7", channel="eml",
                                               path=str(tmp_path / "a.eml")))
    assert n.id and n.channel == "eml"
    assert store.was_notified(d.id, "D-7") is True
    with pytest.raises(StoreError):
        store.record_notification(Notification(deadline_id=d.id, mark="D-7", channel="smtp"))
    store.record_notification(Notification(deadline_id=d.id, mark="D-1", channel="smtp"))
    assert [x.mark for x in store.list_notifications(d.id)] == ["D-7", "D-1"]


def test_notification_mark_and_channel_checked(store):
    d = store.add_deadline(Deadline(kind="period_end", due_date=date(2026, 12, 31)))
    with pytest.raises(sqlite3.IntegrityError):
        store.record_notification(Notification(deadline_id=d.id, mark="D-3", channel="eml"))
    with pytest.raises(sqlite3.IntegrityError):
        store.record_notification(Notification(deadline_id=d.id, mark="D-1", channel="sms"))


def test_deleting_deadline_removes_its_notifications(store):
    d = store.add_deadline(Deadline(kind="period_end", due_date=date(2026, 12, 31)))
    store.record_notification(Notification(deadline_id=d.id, mark="D-30", channel="eml"))
    store.delete_deadline(d.id)
    assert store.list_notifications() == []
