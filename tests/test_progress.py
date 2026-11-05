from datetime import date

import pytest

from volunteer_agent.models import Activity, Profile
from volunteer_agent.progress import Period, activity_status, compute_progress, period_for

TODAY = date(2026, 11, 4)
ENV, EDU, LIFE = "0800", "0400", "0100"


def act(i, day, hours, code, cert=False, **kw):
    return Activity(id=i, start_date=day, hours=hours, type_code=code,
                    certificate_path=f"certificates/{i}.pdf" if cert else None, **kw)


def company(goal=8, codes=(ENV,), cert=True):
    return Profile(id=1, user_type="employee", source="company_policy", goal_hours=goal,
                   period_start=date(2026, 1, 1), period_end=date(2026, 12, 31),
                   certificate_required=cert, allowed_type_codes=list(codes))


# ---- AC3 ----

def test_ac3a_goal_minus_certified_hours():
    p = company(codes=())
    r = compute_progress(p, [act(1, date(2026, 5, 1), 3, ENV, cert=True)], True, TODAY)
    assert r.recognized_hours == 3 and r.remaining_hours == 5 and r.percent == 37.5


def test_ac3b_missing_certificate_is_pending_proof():
    p = company(codes=())
    r = compute_progress(p, [act(1, date(2026, 5, 1), 2, ENV)], True, TODAY)
    assert r.pending_proof_hours == 2 and r.recognized_hours == 0 and r.remaining_hours == 8
    assert r.lines[0].status == "pending_proof" and r.lines[0].status_label == "확인서 대기"


def test_ac3c_source_none_has_no_remaining():
    p = Profile(id=2, user_type="student", source="none",
                period_start=date(2026, 9, 1), period_end=date(2027, 2, 28))
    acts = [act(1, date(2026, 10, 1), 3, LIFE, major_relevance="간호학 관련"),
            act(2, date(2026, 10, 8), 2, EDU, school_planned=True)]
    r = compute_progress(p, acts, False, TODAY)
    assert r.remaining_hours is None and r.percent is None
    assert r.recognized_hours == 5
    assert r.hours_by_type == {LIFE: 3, EDU: 2}
    assert r.major_related_count == 1
    assert r.school_planned_hours == 2 and r.personal_hours == 3


def test_ac3d_company_and_personal_profiles():
    comp = company(goal=8, codes=(ENV,), cert=False)
    personal = Profile(id=3, user_type="employee", source="self_goal", goal_hours=20)
    acts = [act(1, date(2026, 6, 1), 3, ENV), act(2, date(2026, 6, 2), 2, EDU)]
    rc = compute_progress(comp, acts, True, TODAY)
    rp = compute_progress(personal, acts, False, TODAY)
    assert rc.recognized_hours == 3 and rc.excluded_hours == 2 and rc.remaining_hours == 5
    assert rp.recognized_hours == 5 and rp.remaining_hours == 15


def test_ac3e_period_boundary():
    p = company(codes=(), cert=False)
    acts = [act(1, date(2026, 12, 31), 2, ENV), act(2, date(2027, 1, 1), 3, ENV)]
    old = compute_progress(p, acts, True, date(2026, 12, 31))
    new = compute_progress(p, acts, True, date(2027, 1, 1))
    assert old.period == Period(date(2026, 1, 1), date(2026, 12, 31)) and old.recognized_hours == 2
    assert new.period == Period(date(2027, 1, 1), date(2027, 12, 31)) and new.recognized_hours == 3


# ---- 기간 ----

@pytest.mark.parametrize("day, expected", [
    (date(2026, 3, 1), (date(2026, 3, 1), date(2026, 8, 31))),
    (date(2026, 9, 1), (date(2026, 9, 1), date(2027, 2, 28))),
    (date(2027, 2, 15), (date(2026, 9, 1), date(2027, 2, 28))),
    (date(2028, 2, 29), (date(2027, 9, 1), date(2028, 2, 29))),  # 윤년
    (date(2025, 5, 5), (date(2025, 3, 1), date(2025, 8, 31))),  # 과거 기간
])
def test_semester_periods_alternate(day, expected):
    p = Profile(user_type="student", period_start=date(2026, 3, 1), period_end=date(2026, 8, 31))
    assert period_for(p, day) == Period(*expected)


def test_annual_period_repeats_and_no_period_is_none():
    assert period_for(company(), date(2030, 7, 7)) == Period(date(2030, 1, 1), date(2030, 12, 31))
    assert period_for(Profile(user_type="employee"), TODAY) is None


def test_no_period_counts_everything():
    p = Profile(id=3, user_type="employee", source="self_goal", goal_hours=10)
    acts = [act(1, date(2020, 1, 1), 2, ENV), act(2, date(2026, 1, 1), 1, ENV)]
    assert compute_progress(p, acts, False, TODAY).recognized_hours == 3


# ---- 상태와 대상 ----

def test_rules_unconfirmed_company_profile():
    p = company()
    r = compute_progress(p, [act(1, date(2026, 5, 1), 2, ENV, cert=True)], False, TODAY)
    assert r.rules_pending is True
    assert r.unconfirmed_hours == 2 and r.recognized_hours == 0
    assert r.remaining_hours is None and r.percent is None
    assert r.lines[0].status_label == "규칙 미확정"


def test_status_order_excluded_before_pending():
    p = company(codes=(ENV,), cert=True)
    assert activity_status(p, act(1, TODAY, 1, EDU), True) == "excluded"
    assert activity_status(p, act(1, TODAY, 1, ENV), True) == "pending_proof"
    assert activity_status(p, act(1, TODAY, 1, ENV, cert=True), True) == "recognized"


def test_remaining_never_negative_and_percent_capped():
    p = company(goal=8, codes=(), cert=False)
    r = compute_progress(p, [act(1, date(2026, 2, 1), 10, ENV)], True, TODAY)
    assert r.remaining_hours == 0 and r.percent == 100.0


def test_half_hour_sums_are_exact():
    p = company(goal=8, codes=(), cert=False)
    acts = [act(i, date(2026, 2, i), 0.5, ENV) for i in range(1, 8)]
    r = compute_progress(p, acts, True, TODAY)
    assert r.recognized_hours == 3.5 and r.remaining_hours == 4.5
