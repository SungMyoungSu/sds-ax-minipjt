# guards_refusal.py - 거절 응대: 고정 문구 + 내부 기록
import datetime
from langchain_core.messages import AIMessage

REFUSAL_MESSAGES = {
    "injection": (
        "죄송합니다. 해당 요청은 처리할 수 없습니다. "
        "사내 규정 안내, 임직원 정보 조회, 출장 신청 업무를 도와드릴 수 있습니다."
    ),
    "off_topic": (
        "죄송합니다. 그 주제는 제 담당 범위가 아닙니다. "
        "사내 업무와 관련된 질문을 해 주시면 도와드리겠습니다."
    ),
}


def log_block(kind: str, reason: str) -> None:
    """내부 기록: 사유는 여기에만 남깁니다 (사용자에게는 노출 금지)"""
    with open("guard_audit.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()}\t{kind}\t{reason}\n")


def refusal_message(reason: str) -> AIMessage:
    kind = "off_topic" if "금지 주제" in reason else "injection"
    log_block(kind, reason)
    return AIMessage(content=REFUSAL_MESSAGES[kind])
