import email
import socket
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from email import policy

import pytest
from icalendar import Calendar

from deadlines import (
    ICS_FILENAME,
    export_ics,
    get_upcoming_deadlines,
    sync_period_deadlines,
)
from models import RuleCandidate
from notify import due_mark, run_scheduler
from recommend import save_to_watchlist
from records import ActivityInput, RecordService

NOW = datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)
FAKE_PW = "fake-smtp-app-password-123"


def setup_employee(store, settings, today=date(2026, 12, 1), confirmed=True, cert_file=None):
    """회사 규칙(연 8시간, 1/1~12/31, 확인서 필수) + 인정 4시간 → 남은 4시간."""
    store.init_user_setting("employee")
    svc = RecordService(store, settings, today)
    p = svc.create_profile("employee")
    p.goal_hours, p.certificate_required, p.allowed_type_codes = 8, True, ["0800", "0100"]
    p.period_start, p.period_end = date(2026, 1, 1), date(2026, 12, 31)
    store.update_profile(p)
    if confirmed:
        store.mark_rule_confirmed(store.add_rule_candidate(RuleCandidate(profile_id=p.id, goal_hours=8)).id)
    a = svc.log_activity(ActivityInput(start_date=date(2026, 11, 1), organization="복지관", hours=4,
                                       type_code="0100", certificate_file=cert_file)).activity
    return svc, p, a


@pytest.fixture
def cert(tmp_path):
    f = tmp_path / "cert.pdf"
    f.write_bytes(b"%PDF-1.4")
    return f


# ---- 기간 종료 마감 동기화 ----

def test_period_deadline_only_when_goal_unmet(store, settings, cert):
    svc, p, a = setup_employee(store, settings, cert_file=cert)
    remaining = sync_period_deadlines(store, date(2026, 12, 1))
    [d] = store.list_deadlines()
    assert (d.kind, d.due_date, d.ref_id) == ("period_end", date(2026, 12, 31), str(p.id))
    assert remaining == {d.id: 4}

    svc.update_activity(a.id, hours=8)  # 목표 달성 → 마감 사라짐
    sync_period_deadlines(store, date(2026, 12, 1))
    assert store.list_deadlines() == []


def test_no_period_deadline_when_rules_unconfirmed(store, settings):
    setup_employee(store, settings, confirmed=False)
    sync_period_deadlines(store, date(2026, 12, 1))
    assert store.list_deadlines() == []  # 미결 결정 5


def test_new_period_replaces_deadline(store, settings, cert):
    setup_employee(store, settings, cert_file=cert)
    sync_period_deadlines(store, date(2026, 12, 1))
    old = store.list_deadlines()[0]
    sync_period_deadlines(store, date(2027, 1, 5))
    [new] = store.list_deadlines()
    assert new.id != old.id and new.due_date == date(2027, 12, 31)


# ---- 다가오는 마감 (AC6, S5-1) ----

def test_upcoming_within_seven_days_sorted(store, settings, cert):
    setup_employee(store, settings, cert_file=cert)
    save_to_watchlist(store, "P-LATE", "연말 나눔 행사", recruit_end=date(2026, 12, 20))
    save_to_watchlist(store, "P-SOON", "한강 플로깅", recruit_end=date(2026, 12, 5))
    save_to_watchlist(store, "P-TODAY", "급식 봉사", recruit_end=date(2026, 12, 1))
    save_to_watchlist(store, "P-PAST", "지난 봉사", recruit_end=date(2026, 11, 20))
    r = get_upcoming_deadlines(store, date(2026, 12, 1))
    assert [v.deadline.ref_id for v in r.items] == ["P-TODAY", "P-SOON"]
    assert [v.days_left for v in r.items] == [0, 4]
    assert [v.deadline.due_date for v in r.period_status] == [date(2026, 12, 31)]
    assert "남은 4시간" in r.message and "D-30" in r.message


def test_upcoming_empty_message(store, settings):
    r = get_upcoming_deadlines(store, date(2026, 12, 1))
    assert r.items == [] and "- 없음" in r.message


# ---- .ics (AC6, PRD 35) ----

def test_ics_is_valid_with_alarms_and_stable_uid(store, settings, cert):
    setup_employee(store, settings, cert_file=cert)
    save_to_watchlist(store, "P-SOON", "한강 플로깅", recruit_end=date(2026, 12, 5))
    path, n = export_ics(store, settings, date(2026, 12, 1), now=NOW)
    assert path == settings.data_dir / "calendar" / ICS_FILENAME and n == 2
    cal = Calendar.from_ical(path.read_bytes())
    events = cal.walk("VEVENT")
    assert len(events) == 2
    assert cal.walk("VTIMEZONE")[0]["TZID"] == "Asia/Seoul"
    for ev in events:
        assert ev["DTSTART"].dt.tzinfo is not None and str(ev["DTSTART"].dt.tzinfo) == "Asia/Seoul"
        assert ev["DTSTART"].params["TZID"] == "Asia/Seoul"
        assert ev["DTSTAMP"].dt == NOW
        triggers = sorted(a["TRIGGER"].dt for a in ev.walk("VALARM"))
        assert triggers == [timedelta(days=-30), timedelta(days=-7), timedelta(days=-1)]
    raw = path.read_text(encoding="utf-8")
    for t in ("TRIGGER:-P30D", "TRIGGER:-P7D", "TRIGGER:-P1D"):
        assert t in raw

    uids = sorted(str(e["UID"]) for e in events)
    path2, _ = export_ics(store, settings, date(2026, 12, 2), now=NOW + timedelta(days=1))
    assert sorted(str(e["UID"]) for e in Calendar.from_ical(path2.read_bytes()).walk("VEVENT")) == uids


# ---- 스케줄러 ----

@pytest.mark.parametrize("days, mark", [(31, None), (30, "D-30"), (20, "D-30"), (8, "D-30"),
                                        (7, "D-7"), (5, "D-7"), (1, "D-1"), (0, "D-1")])
def test_due_mark(days, mark):
    assert due_mark(days) == mark


def test_no_smtp_saves_eml_once(store, settings, cert):
    """S5-3: SMTP가 없으면 발송 대신 .eml 저장 (AC25, PRD 36)."""
    setup_employee(store, settings, cert_file=cert)
    r = run_scheduler(store, settings, date(2026, 12, 24), now=NOW)
    [path] = r.saved
    assert path.parent == settings.data_dir / "outbox" and path.suffix == ".eml"
    assert ".eml" in r.message and "SMTP가 설정되지 않아" in r.message
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    assert msg["Subject"].startswith("[봉사 D-7] 기준 기간 종료 2026-12-31")
    assert "남은 4시간" in msg["Subject"]
    assert "7일 남았습니다" in msg.get_body(("plain",)).get_content()
    [att] = list(msg.iter_attachments())
    assert att.get_content_type() == "text/calendar" and att.get_filename() == "deadline.ics"
    assert [n.channel for n in store.list_notifications()] == ["eml"]

    again = run_scheduler(store, settings, date(2026, 12, 24), now=NOW)
    assert again.saved == [] and "보낼 알림이 없습니다" in again.message


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def smtp_sink():
    from aiosmtpd.controller import Controller

    class Sink:
        def __init__(self):
            self.messages = []

        async def handle_DATA(self, server, session, envelope):
            self.messages.append(email.message_from_bytes(envelope.content, policy=policy.default))
            return "250 OK"

    handler = Sink()
    port = _free_port()
    controller = Controller(handler, hostname="127.0.0.1", port=port)
    controller.start()
    yield handler, port
    controller.stop()


def test_ac6_smtp_d30_d7_d1_each_once(store, settings, cert, smtp_sink):
    sink, port = smtp_sink
    s = replace(settings, smtp_host="127.0.0.1", smtp_port=port, alert_email_to="me@example.com")
    setup_employee(store, s, cert_file=cert)
    store.set_alert_consent(True)
    for day, mark in [(date(2026, 12, 1), "D-30"), (date(2026, 12, 24), "D-7"), (date(2026, 12, 30), "D-1")]:
        r = run_scheduler(store, s, day, now=NOW)
        assert len(r.sent) == 1 and r.saved == []
        r2 = run_scheduler(store, s, day, now=NOW)  # 다시 돌려도 중복 없음
        assert r2.sent == []
        msg = sink.messages[-1]
        assert msg["To"] == "me@example.com"
        assert msg["Subject"].startswith(f"[봉사 {mark}] 기준 기간 종료 2026-12-31")
        assert "이번 기간 남은 시간: 4시간" in msg.get_body(("plain",)).get_content()
    assert len(sink.messages) == 3
    assert [n.mark for n in store.list_notifications()] == ["D-30", "D-7", "D-1"]


def test_smtp_without_consent_saves_eml(store, settings, cert):
    s = replace(settings, smtp_host="127.0.0.1", smtp_port=2525, alert_email_to="me@example.com")
    setup_employee(store, s, cert_file=cert)
    calls = []
    r = run_scheduler(store, s, date(2026, 12, 24), sender=lambda st, m: calls.append(m), now=NOW)
    assert calls == [] and len(r.saved) == 1 and "수신 동의 전" in r.message


def test_failed_send_is_not_recorded_and_retried(store, settings, cert):
    s = replace(settings, smtp_host="127.0.0.1", smtp_port=587, alert_email_to="me@example.com",
                smtp_user="me@example.com", smtp_password=FAKE_PW)
    setup_employee(store, s, cert_file=cert)
    store.set_alert_consent(True)

    def broken(st, m):
        raise OSError(f"auth failed for {FAKE_PW}")

    r = run_scheduler(store, s, date(2026, 12, 24), sender=broken, now=NOW)
    assert r.sent == [] and len(r.failed) == 1
    assert FAKE_PW not in r.message and "다음 점검 때 다시" in r.message
    assert store.list_notifications() == []

    sent = []
    r = run_scheduler(store, s, date(2026, 12, 24), sender=lambda st, m: sent.append(m), now=NOW)
    assert len(sent) == 1 and len(r.sent) == 1


def test_watchlist_recruit_deadline_alert(store, settings):
    store.init_user_setting("student")
    save_to_watchlist(store, "P-1", "한강 플로깅", recruit_end=date(2026, 12, 2))
    r = run_scheduler(store, settings, date(2026, 12, 1), now=NOW)
    [path] = r.saved
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    assert msg["Subject"] == "[봉사 D-1] 모집 마감 2026-12-02: 한강 플로깅"
    assert "1365 자원봉사포털에서 직접" in msg.get_body(("plain",)).get_content()


def test_unconfirmed_rules_send_nothing(store, settings):
    setup_employee(store, settings, confirmed=False)
    r = run_scheduler(store, settings, date(2026, 12, 24), now=NOW)
    assert r.saved == [] and r.sent == []
