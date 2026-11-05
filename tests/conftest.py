from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from config import load_settings
from store import Store

# .env.example의 변수 이름. 테스트가 개발자 PC의 환경 변수나 .env에 영향을 받지 않게 지운다.
ENV_NAMES = [
    "LLM_PROVIDER", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "LLM_MODEL", "JUDGE_MODEL", "EMBEDDING_MODEL", "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL", "HTTPS_PROXY", "DATA_GO_KR_KEY", "DATA_GO_KR_TIMEOUT",
    "VOLUNTEER_1365_BASE_URL", "VOLUNTEER_1365_LIST_OPERATION",
    "SMTP_HOST", "SMTP_PORT",
    "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO", "APP_TIMEZONE", "DATA_DIR", "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """모든 테스트를 빈 환경 변수와 임시 작업 폴더에서 실행한다(실제 .env를 읽지 않음)."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """임시 DATA_DIR. 이 fixture를 쓰는 테스트의 load_settings()는 이 폴더를 가리킨다."""
    path = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(path))
    return path


@pytest.fixture
def settings(data_dir):
    s = load_settings(env_file=None)
    s.ensure_data_dir()
    return s


class Clock:
    """주입용 시계. tick()으로 시간을 앞으로 보낸다."""

    def __init__(self):
        self.t = datetime(2026, 11, 4, 9, 0, tzinfo=ZoneInfo("Asia/Seoul"))

    def __call__(self):
        return self.t

    def tick(self, minutes=1):
        self.t += timedelta(minutes=minutes)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(settings, clock):
    s = Store.open(settings, now=clock)
    yield s
    s.close()
