"""LLM 생성. AWS Bedrock ChatBedrockConverse, temperature 0."""

from __future__ import annotations

from typing import Any

from config import Settings
from errors import InputError


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모은다."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


CACHE_CONTROL = {"type": "ephemeral"}  # Bedrock cachePoint, TTL 기본 5분


def _cached_class():
    """모든 호출에 cache_control을 넣는 ChatBedrockConverse. langchain-aws가 시스템 프롬프트 끝, 도구 목록 끝,
    마지막·그 앞 메시지에 cachePoint를 넣는다(최대 4개). 비동기 호출은 기본 구현이 _generate를 부른다."""
    from langchain_aws import ChatBedrockConverse

    class CachedChatBedrockConverse(ChatBedrockConverse):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.setdefault("cache_control", CACHE_CONTROL)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.setdefault("cache_control", CACHE_CONTROL)
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    return CachedChatBedrockConverse


def make_llm(settings: Settings, model: str | None = None, *, cache: bool = False):
    """Bedrock 채팅 모델. cache=True이면 프롬프트 캐싱을 켠다.

    에이전트(Supervisor·하위 Agent)만 켠다. 도구 설명·시스템 프롬프트·앞 대화가 매번 같아 재사용된다.
    가드레일·LLM 심사는 끈다: 매번 다른 사용자 입력이 프롬프트 전체라 캐시에 쓰기만 하고(25% 할증) 읽지 못한다."""
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
    cls = _cached_class() if cache and settings.prompt_cache else ChatBedrockConverse
    return cls(**kwargs)


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
