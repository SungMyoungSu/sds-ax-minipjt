"""비밀 값 마스킹.

로그, trace, 도구 출력, 예외 메시지로 내보내는 모든 문자열은 `mask_secrets`를 거친다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MASK = "***"

# 설정에 값이 없어도 형태만으로 가리는 패턴
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # AWS 액세스 키 ID (장기 AKIA, 임시 ASIA)
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), MASK),
    # Anthropic API 키
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]+"), MASK),
    # data.go.kr 서비스 키 쿼리 파라미터
    (re.compile(r"(?i)(serviceKey=)[^&\s\"']+"), rf"\g<1>{MASK}"),
    # KEY/SECRET/TOKEN/PASSWORD가 들어간 이름의 `이름=값`, `이름: 값`
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*\s*[=:]\s*)"
            r"(?![\s\"']*\*\*\*)[\"']?[^\s\"',&]+[\"']?"
        ),
        rf"\g<1>{MASK}",
    ),
]

# 너무 짧은 값은 일반 단어를 가릴 수 있어 값 기반 마스킹에서 제외한다
_MIN_SECRET_LEN = 4


def mask_secrets(text: object, secrets: Iterable[str | None] = ()) -> str:
    """text에서 비밀 값을 `***`로 바꾼다.

    secrets: 실제 비밀 값 목록(보통 `Settings.secret_values()`). 원문 그대로 찾아 가린다.
    """
    result = str(text)
    for value in sorted({s for s in secrets if s and len(s) >= _MIN_SECRET_LEN}, key=len, reverse=True):
        result = result.replace(value, MASK)
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def mask_exception(exc: BaseException, secrets: Iterable[str | None] = ()) -> str:
    """예외를 `타입: 메시지` 문자열로 만들고 마스킹한다."""
    return mask_secrets(f"{type(exc).__name__}: {exc}", secrets)
