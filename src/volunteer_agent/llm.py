"""LLM 생성. AWS Bedrock ChatBedrockConverse, temperature 0."""

from __future__ import annotations

from typing import Any

from volunteer_agent.config import Settings
from volunteer_agent.errors import InputError


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모은다."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


def make_llm(settings: Settings, model: str | None = None):
    if settings.llm_provider != "bedrock":
        raise InputError("지금은 LLM_PROVIDER=bedrock만 지원합니다.")
    model = model or settings.llm_model
    if not model:
        raise InputError("LLM_MODEL이 설정되지 않았습니다. .env에 Bedrock 모델 ID를 넣어 주세요.")
    from langchain_aws import ChatBedrockConverse

    kwargs: dict[str, Any] = {"model": model, "region_name": settings.aws_region or "us-east-1", "temperature": 0}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token
    return ChatBedrockConverse(**kwargs)


def has_llm_credentials(settings: Settings) -> bool:
    return bool(settings.llm_model and settings.aws_access_key_id and settings.aws_secret_access_key)


QUOTA_ERROR_NAMES = {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException"}
QUOTA_ERROR_TEXT = ("Too many tokens", "Too many requests", "ThrottlingException", "Rate exceeded")


def is_quota_error(exc: BaseException) -> bool:
    """Bedrock 사용 한도 초과(분당·일일 토큰, 요청 수)인가. 감싼 예외(__cause__, __context__)까지 본다.
    botocore가 이미 여러 번 재시도한 뒤라, 평가 실행기는 이 오류를 만나면 남은 문항을 돌리지 않고 멈춘다."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in QUOTA_ERROR_NAMES or any(t in str(current) for t in QUOTA_ERROR_TEXT):
            return True
        current = current.__cause__ or current.__context__
    return False
