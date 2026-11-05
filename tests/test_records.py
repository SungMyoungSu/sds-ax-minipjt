from datetime import date

import pytest

from volunteer_agent.models import RuleCandidate
from volunteer_agent.records import (
    ActivityInput,
    InputError,
    RecordService,
    default_semester,
)

TODAY = date(2026, 11, 4)
ENV, LIFE, ETC = "0800", "0100", "1500"


@pytest.fixture
def cert(tmp_path):
    f = tmp_path / "certificate_sample.pdf"
    f.write_bytes(b"%PDF-1.4 fake certificate")
    return f


def employee_service(store, settings, today=TODAY, confirmed=True):
    """회사원 + 확정된 회사 규칙(연 8시간, 1/1~12/31, 확인서 필수, 환경·생활편의만 인정)."""
    store.init_user_setting("employee")
    svc = RecordService(store, settings, today)
    p = svc.create_profile("employee")
    p.goal_hours, p.certificate_required = 8, True
    p.period_start, p.period_end = date(2026, 1, 1), date(2026, 12, 31)
    p.allowed_type_codes = [ENV, LIFE]
    store.update_profile(p)
    if confirmed:
        r = store.add_rule_candidate(RuleCandidate(profile_id=p.id, goal_hours=8))
        store.mark_rule_confirmed(r.id)
    return svc, p


def inp(day=TODAY, org="강남구 복지관", hours=2.0, code=ENV, **kw):
    return ActivityInput(start_date=day, organization=org, hours=hours, type_code=code, **kw)


def company_progress(svc, profile_id):
    return svc.progress(profile_id)[0]


# ---- 회사원 시나리오 시간 흐름 (employee.yaml S4-1 ~ S6-3) ----

def test_employee_scenario_time_flow(store, settings, cert):
    svc, p = employee_service(store, settings)

    # S4-1 유기견 보호소 3시간(인정 분야 아님) → 남은 8
    r = svc.log_activity(inp(org="유기견 보호소", hours=3, code=ETC))
    assert r.status == "saved"
    assert r.progress[0].remaining_hours == 8 and r.progress[0].excluded_hours == 3

    # S4-2 복지관 급식 2.5시간 + 확인서 → 남은 5.5
    r = svc.log_activity(inp(day=date(2026, 10, 31), org="복지관", hours=2.5, code=LIFE,
                             certificate_file=cert))
    assert r.activity.certificate_path.startswith("certificates/")
    assert (settings.data_dir / r.activity.certificate_path).is_file()
    assert r.progress[0].remaining_hours == 5.5

    # S4-3 환경정화 2시간, 확인서 나중 → 확인서 대기 2
    r = svc.log_activity(inp(day=date(2026, 11, 3), org="환경정화", hours=2, code=ENV))
    env_id = r.activity.id
    assert r.progress[0].remaining_hours == 5.5 and r.progress[0].pending_proof_hours == 2

    # S4-3b 1시간 20분 → 입력 오류, 저장 안 됨
    with pytest.raises(InputError, match="0.5시간"):
        svc.log_activity(inp(day=date(2026, 11, 2), org="공원", hours=4 / 3))
    # S4-3c 미래 날짜 → 입력 오류
    with pytest.raises(InputError, match="미래 날짜"):
        svc.log_activity(inp(day=date(2026, 11, 10), org="공원", hours=2))
    assert len(store.list_activities()) == 3
    assert company_progress(svc, p.id).pending_proof_hours == 2

    # S4-4 1.5시간으로 수정 → 확인서 대기 1.5
    r = svc.update_activity(env_id, hours=1.5)
    assert r.progress[0].pending_proof_hours == 1.5

    # S4-5 확인서 첨부 → 대기 0, 남은 4
    r = svc.attach_certificate(env_id, cert)
    assert r.progress[0].pending_proof_hours == 0 and r.progress[0].remaining_hours == 4

    # S6-3 새해 → 남은 8, 확정 규칙 유지 (AC14 d)
    new_year = RecordService(store, settings, date(2027, 1, 5))
    assert company_progress(new_year, p.id).remaining_hours == 8
    assert len(store.list_rules(profile_id=p.id, status="confirmed")) == 1


# ---- AC14 기록 수명주기 ----

def test_ac14a_update_recalculates(store, settings, cert):
    svc, p = employee_service(store, settings)
    a = svc.log_activity(inp(hours=3, certificate_file=cert)).activity
    assert company_progress(svc, p.id).remaining_hours == 5
    assert svc.update_activity(a.id, hours=2).progress[0].remaining_hours == 6


def test_ac14b_delete_needs_confirmation(store, settings, cert):
    svc, p = employee_service(store, settings)
    a = svc.log_activity(inp(hours=3, certificate_file=cert)).activity
    r = svc.delete_activity(a.id)
    assert r.status == "needs_confirmation" and len(store.list_activities()) == 1
    r = svc.delete_activity(a.id, confirm=True)
    assert r.status == "deleted" and r.progress[0].remaining_hours == 8
    assert store.list_activities() == []


def test_ac14c_attach_turns_pending_into_recognized(store, settings, cert):
    svc, p = employee_service(store, settings)
    a = svc.log_activity(inp(hours=2)).activity
    assert company_progress(svc, p.id).pending_proof_hours == 2
    r = svc.attach_certificate(a.id, cert)
    assert r.progress[0].pending_proof_hours == 0 and r.progress[0].recognized_hours == 2


# ---- AC5, AC23 입력 검증 ----

@pytest.mark.parametrize("kwargs, message", [
    (dict(hours=1.2), "0.5시간"),
    (dict(hours=-1), "0보다"),
    (dict(hours=0), "0보다"),
    (dict(hours=25), "24시간"),
    (dict(day=date(2026, 11, 5)), "미래 날짜"),
    (dict(code="ZZZZ"), "목록에 없는 분야"),
])
def test_invalid_input_is_rejected_without_changes(store, settings, kwargs, message):
    svc, _ = employee_service(store, settings)
    svc.log_activity(inp(day=date(2026, 10, 1)))
    with pytest.raises(InputError) as info:
        svc.log_activity(inp(**kwargs))
    assert str(info.value).startswith("입력 오류:") and message in str(info.value)
    assert "실제로 하지 않은 활동" not in str(info.value)  # 허위 기록 거절 문구와 구분
    assert len(store.list_activities()) == 1


def test_multi_day_allows_up_to_24h_per_day(store, settings):
    svc, _ = employee_service(store, settings)
    ok = inp(day=date(2026, 10, 1), end_date=date(2026, 10, 2), hours=30)
    assert svc.log_activity(ok).status == "saved"
    with pytest.raises(InputError, match="24시간"):
        svc.log_activity(inp(day=date(2026, 10, 1), end_date=date(2026, 10, 2), hours=49))
    with pytest.raises(InputError, match="앞설 수 없습니다"):
        svc.log_activity(inp(day=date(2026, 10, 2), end_date=date(2026, 10, 1)))


def test_missing_fields_only_asks_for_those(store, settings):
    svc, _ = employee_service(store, settings)
    r = svc.log_activity(ActivityInput(start_date=TODAY, hours=2, type_code=ENV))
    assert r.status == "missing_fields" and r.missing_labels == ["기관"]
    r = svc.log_activity(ActivityInput(organization="복지관", hours=2, type_code=LIFE))
    assert r.missing_labels == ["날짜"]
    assert store.list_activities() == []


def test_duplicate_requires_confirmation(store, settings):
    svc, _ = employee_service(store, settings)
    first = svc.log_activity(inp()).activity
    r = svc.log_activity(inp(org=" 강남구 복지관 "))
    assert r.status == "duplicate" and r.duplicate_of == first.id
    assert len(store.list_activities()) == 1
    assert svc.log_activity(inp(), confirm_duplicate=True).status == "saved"
    assert len(store.list_activities()) == 2


def test_update_validates_and_rejects_unknown_fields(store, settings):
    svc, _ = employee_service(store, settings)
    a = svc.log_activity(inp(hours=2)).activity
    with pytest.raises(InputError, match="0.5시간"):
        svc.update_activity(a.id, hours=1.3)
    with pytest.raises(InputError, match="수정할 수 없는"):
        svc.update_activity(a.id, certificate_path="x.pdf")
    with pytest.raises(InputError, match="비울 수 없습니다"):
        svc.update_activity(a.id, organization="  ")
    assert store.get_activity(a.id).hours == 2


@pytest.mark.parametrize("name", ["certificate.docx", "certificate.hwp"])
def test_certificate_format_checked(store, settings, tmp_path, name):
    svc, _ = employee_service(store, settings)
    a = svc.log_activity(inp()).activity
    bad = tmp_path / name
    bad.write_bytes(b"x")
    with pytest.raises(InputError, match="PDF, JPG, PNG"):
        svc.attach_certificate(a.id, bad)
    with pytest.raises(InputError, match="찾을 수 없습니다"):
        svc.attach_certificate(a.id, tmp_path / "none.pdf")
    assert store.get_activity(a.id).certificate_path is None


def test_rules_unconfirmed_still_records(store, settings):
    svc, p = employee_service(store, settings, confirmed=False)
    r = svc.log_activity(inp(hours=2))
    assert r.status == "saved"
    prog = r.progress[0]
    assert prog.rules_pending and prog.unconfirmed_hours == 2 and prog.remaining_hours is None


# ---- 프로필 ----

@pytest.mark.parametrize("today, expected", [
    (date(2026, 3, 10), (date(2026, 3, 1), date(2026, 8, 31))),
    (date(2026, 11, 4), (date(2026, 9, 1), date(2027, 2, 28))),
    (date(2028, 1, 10), (date(2027, 9, 1), date(2028, 2, 29))),
])
def test_default_semester(today, expected):
    assert default_semester(today) == expected


def test_create_profile_defaults(store, settings):
    svc = RecordService(store, settings, date(2026, 3, 10))
    emp = svc.create_profile("employee")
    assert emp.source == "company_policy" and emp.period_start is None
    stu = svc.create_profile("student", major="간호학")
    assert stu.source == "none" and stu.period_start == date(2026, 3, 1) and stu.major == "간호학"
    goal = svc.create_profile("student", goal_hours=20)
    assert goal.source == "self_goal"
    with pytest.raises(InputError):
        svc.create_profile("teacher")
    with pytest.raises(InputError):
        svc.create_profile("employee", period_start=date(2026, 1, 1))
