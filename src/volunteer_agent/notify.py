"""마감 알림 스케줄러 진입점 (결정적 코드). Windows 작업 스케줄러가 하루 1회 실행한다.

- 같은 마감의 같은 D-n 알림은 한 번만 보낸다(G11). 발송에 성공했을 때만 기록하므로 실패하면 다음 점검 때 다시 보낸다.
- SMTP가 설정되어 있고 수신 동의가 있으면 메일로 보낸다. SMTP가 없으면(또는 동의 전이면) 보내지 않고
  DATA_DIR/outbox에 .eml(제목, 본문, .ics 첨부)로 저장한 뒤 요약한다(PRD 36).
- 점검을 하루 이상 거르면 그날 해당하는 단계(30·7·1일 중 남은 날 이상인 가장 작은 값)를 아직 안 보냈을 때 보낸다.
"""

from __future__ import annotations

import smtplib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from email.message import EmailMessage
from pathlib import Path

from volunteer_agent.config import Settings
from volunteer_agent.deadlines import ALARM_DAYS, DeadlineView, active_deadlines, build_calendar
from volunteer_agent.masking import mask_exception
from volunteer_agent.models import Notification
from volunteer_agent.store import Store

OUTBOX_DIR = "outbox"
DEFAULT_RECIPIENT = "me@localhost"  # SMTP가 없을 때 .eml 받는 사람 자리

Sender = Callable[[Settings, EmailMessage], None]


def due_mark(days_left: int) -> str | None:
    """오늘 보낼 단계. 남은 날이 30일을 넘으면 None."""
    for n in sorted(ALARM_DAYS):
        if days_left <= n:
            return f"D-{n}"
    return None


def build_message(view: DeadlineView, mark: str, settings: Settings, now: datetime | None = None) -> EmailMessage:
    d = view.deadline
    msg = EmailMessage()
    msg["To"] = settings.alert_email_to or DEFAULT_RECIPIENT
    msg["From"] = settings.smtp_user or settings.alert_email_to or DEFAULT_RECIPIENT
    extra = f" — 남은 {view.remaining_hours:g}시간" if view.remaining_hours is not None else ""
    msg["Subject"] = f"[봉사 {mark}] {view.label} {d.due_date.isoformat()}: {d.title}{extra}"
    body = [
        f"{view.label}이 {view.days_left}일 남았습니다({d.due_date.isoformat()}).",
        f"- 항목: {d.title}",
    ]
    if view.remaining_hours is not None:
        body.append(f"- 이번 기간 남은 시간: {view.remaining_hours:g}시간")
    if d.kind == "recruit_end":
        body.append("- 신청은 1365 자원봉사포털에서 직접 해 주세요.")
    body.append("캘린더에 넣으려면 첨부한 .ics 파일을 여세요.")
    msg.set_content("\n".join(body))
    ics = build_calendar([view], settings, now).to_ical()
    msg.add_attachment(ics, maintype="text", subtype="calendar", filename="deadline.ics")
    return msg


def send_smtp(settings: Settings, msg: EmailMessage) -> None:
    """설정된 SMTP로 보낸다. 587이면 STARTTLS, 465면 SSL, 계정이 있으면 로그인."""
    timeout = 15
    if settings.smtp_port == 465:
        client = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=timeout)
    else:
        client = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=timeout)
    with client:
        if settings.smtp_port == 587:
            client.starttls()
        if settings.smtp_user and settings.smtp_password:
            client.login(settings.smtp_user, settings.smtp_password)
        client.send_message(msg)


@dataclass
class SchedulerResult:
    sent: list[str] = field(default_factory=list)  # 메일로 보낸 알림 제목
    saved: list[Path] = field(default_factory=list)  # 저장한 .eml
    failed: list[str] = field(default_factory=list)  # 실패(마스킹된 이유), 다음 점검 때 재시도
    skipped: int = 0  # 이미 보냈거나 아직 30일 넘게 남은 마감
    notes: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        lines = ["마감 알림 점검 결과:"]
        lines += [f"- 메일 발송: {s}" for s in self.sent]
        lines += [f"- .eml 저장(메일은 보내지 않음): {p}" for p in self.saved]
        lines += [f"- 발송 실패(다음 점검 때 다시 보냄): {f}" for f in self.failed]
        lines += [f"- {n}" for n in self.notes]
        if not (self.sent or self.saved or self.failed):
            lines.append("- 오늘 보낼 알림이 없습니다.")
        return "\n".join(lines)


def run_scheduler(store: Store, settings: Settings, today: date, sender: Sender = send_smtp,
                  now: datetime | None = None) -> SchedulerResult:
    """마감 알림 점검 1회. 오늘 날짜는 주입한다(테스트·--today)."""
    result = SchedulerResult()
    setting = store.get_user_setting()
    consent = bool(setting and setting.alert_consent)
    use_smtp = settings.smtp_configured and consent
    if settings.smtp_configured and not consent:
        result.notes.append("메일 알림 수신 동의 전이라 메일 대신 .eml로 저장했습니다.")
    elif not settings.smtp_configured:
        result.notes.append("SMTP가 설정되지 않아 메일 대신 .eml로 저장했습니다.")

    outbox = settings.ensure_data_dir() / OUTBOX_DIR
    for view in active_deadlines(store, today):
        mark = due_mark(view.days_left)
        d = view.deadline
        if mark is None or store.was_notified(d.id, mark):
            result.skipped += 1
            continue
        msg = build_message(view, mark, settings, now)
        if use_smtp:
            try:
                sender(settings, msg)
            except Exception as exc:  # noqa: BLE001 - 실패는 기록하지 않고 다음에 재시도
                result.failed.append(f"{msg['Subject']} ({mask_exception(exc, settings.secret_values())})")
                continue
            store.record_notification(Notification(deadline_id=d.id, mark=mark, channel="smtp"))
            result.sent.append(str(msg["Subject"]))
        else:
            outbox.mkdir(exist_ok=True)
            path = outbox / f"{today:%Y%m%d}_{d.kind}_{d.id}_{mark}.eml"
            path.write_bytes(msg.as_bytes())
            store.record_notification(Notification(deadline_id=d.id, mark=mark, channel="eml", path=str(path)))
            result.saved.append(path)
    return result
