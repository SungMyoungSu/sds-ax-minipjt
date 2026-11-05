"""설정 로더.

`.env`(있을 때)와 환경 변수에서 값을 읽어 `Settings`를 만든다. 변수 이름은 `.env.example`이 기준이다.
우선순위는 환경 변수 > `.env` > 기본값이다. `.env`는 `os.environ`에 주입하지 않고 읽기만 한다.
비밀 값은 `repr`/`str`에 나오지 않는다.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import dotenv_values

DEFAULT_DATA_DIR = "data"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 파일 없이 "회사 규정 등록해줘"라고 하면 쓰는 기본 사내규정(DEFAULT_POLICY_FILE로 바꿀 수 있다)
DEFAULT_POLICY_FILE = PROJECT_ROOT / "data" / "policies" / "사회공헌_봉사활동_규정.md"
DEFAULT_TIMEZONE = "Asia/Seoul"
DEFAULT_API_TIMEOUT = 10.0
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LLM_PROVIDER = "bedrock"
# 1365 봉사참여정보 목록 조회 오퍼레이션(2026-11-05 실제 호출로 확인). VOLUNTEER_1365_LIST_OPERATION으로 바꿀 수 있다.
DEFAULT_1365_LIST_OPERATION = "getVltrSearchWordList"
LLM_PROVIDERS = ("bedrock", "anthropic")


def _policy_path(value: str | None) -> Path:
    """기본 규정 파일 경로. 상대 경로는 프로젝트 폴더 기준."""
    if not value:
        return DEFAULT_POLICY_FILE
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class ConfigError(ValueError):
    """설정 값이 잘못됐을 때. 메시지에는 변수 이름만 넣고 값은 넣지 않는다."""


def _secret() -> object:
    return field(default=None, repr=False, metadata={"secret": True})


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    app_timezone: str = DEFAULT_TIMEZONE
    log_level: str = DEFAULT_LOG_LEVEL

    # LLM
    llm_provider: str = DEFAULT_LLM_PROVIDER
    aws_access_key_id: str | None = _secret()
    aws_secret_access_key: str | None = _secret()
    aws_session_token: str | None = _secret()
    aws_region: str | None = None
    llm_model: str | None = None
    judge_model: str | None = None
    embedding_model: str | None = None
    anthropic_api_key: str | None = _secret()
    anthropic_model: str | None = None
    https_proxy: str | None = _secret()  # 프록시 URL에 계정 정보가 들어갈 수 있음

    # 공공 API
    data_go_kr_key: str | None = _secret()
    data_go_kr_timeout: float = DEFAULT_API_TIMEOUT
    volunteer_1365_base_url: str | None = None
    volunteer_1365_list_operation: str = DEFAULT_1365_LIST_OPERATION

    # 사내규정
    default_policy_file: Path | None = DEFAULT_POLICY_FILE

    # SMTP (비어 있으면 .eml 저장)
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_password: str | None = _secret()
    alert_email_to: str | None = None

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.app_timezone)

    @property
    def smtp_configured(self) -> bool:
        """메일을 실제로 보낼 수 있는지. 받는 사람이 없으면 미설정으로 보고 .eml 저장으로 처리한다.
        계정·비밀번호는 로컬 SMTP sink처럼 인증 없는 서버를 위해 필수로 두지 않는다."""
        return bool(self.smtp_host and self.smtp_port and self.alert_email_to)

    def secret_values(self) -> list[str]:
        """마스킹 대상 비밀 값 목록(비어 있지 않은 것만)."""
        return [
            getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("secret") and getattr(self, f.name)
        ]

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _to_float(name: str, value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{name}는 숫자여야 합니다.") from None
    if number <= 0:
        raise ConfigError(f"{name}는 0보다 커야 합니다.")
    return number


def _to_int(name: str, value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"{name}는 정수여야 합니다.") from None


def load_settings(
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """`.env`와 환경 변수에서 설정을 읽는다.

    env_file: 읽을 `.env` 경로. None이거나 파일이 없으면 건너뛴다.
    environ: 환경 변수(테스트용). None이면 `os.environ`.
    """
    values: dict[str, str | None] = {}
    if env_file is not None and Path(env_file).is_file():
        values.update(dotenv_values(env_file, encoding="utf-8"))
    values.update(os.environ if environ is None else environ)

    def get(name: str) -> str | None:
        return _clean(values.get(name))

    provider = (get("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).lower()
    if provider not in LLM_PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER는 {' 또는 '.join(LLM_PROVIDERS)}여야 합니다.")

    timezone = get("APP_TIMEZONE") or DEFAULT_TIMEZONE
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigError("APP_TIMEZONE이 올바른 시간대 이름이 아닙니다.") from None

    return Settings(
        data_dir=Path(get("DATA_DIR") or DEFAULT_DATA_DIR).resolve(),
        app_timezone=timezone,
        log_level=(get("LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper(),
        llm_provider=provider,
        aws_access_key_id=get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=get("AWS_SESSION_TOKEN"),
        aws_region=get("AWS_REGION"),
        llm_model=get("LLM_MODEL"),
        judge_model=get("JUDGE_MODEL"),
        embedding_model=get("EMBEDDING_MODEL"),
        anthropic_api_key=get("ANTHROPIC_API_KEY"),
        anthropic_model=get("ANTHROPIC_MODEL"),
        https_proxy=get("HTTPS_PROXY"),
        data_go_kr_key=get("DATA_GO_KR_KEY"),
        data_go_kr_timeout=_to_float(
            "DATA_GO_KR_TIMEOUT", get("DATA_GO_KR_TIMEOUT"), DEFAULT_API_TIMEOUT
        ),
        volunteer_1365_base_url=get("VOLUNTEER_1365_BASE_URL"),
        volunteer_1365_list_operation=get("VOLUNTEER_1365_LIST_OPERATION") or DEFAULT_1365_LIST_OPERATION,
        default_policy_file=_policy_path(get("DEFAULT_POLICY_FILE")),
        smtp_host=get("SMTP_HOST"),
        smtp_port=_to_int("SMTP_PORT", get("SMTP_PORT")),
        smtp_user=get("SMTP_USER"),
        smtp_password=get("SMTP_PASSWORD"),
        alert_email_to=get("ALERT_EMAIL_TO"),
    )
