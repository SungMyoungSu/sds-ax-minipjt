from pathlib import Path

import pytest

from volunteer_agent.config import ConfigError, load_settings

FAKE_AWS_SECRET = "fakeSecretAccessKey0123456789abcdefXYZ"
FAKE_DATA_KEY = "fakeDataGoKrKey+/=0123456789"


def test_defaults_without_env_file():
    s = load_settings(env_file=".env")  # 임시 작업 폴더에는 .env가 없음
    assert s.data_dir == Path("data").resolve()
    assert s.app_timezone == "Asia/Seoul"
    assert s.data_go_kr_timeout == 10.0
    assert s.log_level == "INFO"
    assert s.llm_provider == "bedrock"
    assert s.smtp_configured is False
    assert s.secret_values() == []


def test_env_file_is_read_without_touching_os_environ(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "DATA_DIR=mydata\nAPP_TIMEZONE=UTC\nDATA_GO_KR_TIMEOUT=5\nSMTP_HOST=localhost\nSMTP_PORT=1025\n",
        encoding="utf-8",
    )
    s = load_settings(env_file=env)
    assert s.data_dir == (tmp_path / "mydata").resolve()
    assert s.app_timezone == "UTC"
    assert s.data_go_kr_timeout == 5.0
    assert s.smtp_port == 1025
    assert s.smtp_configured is False  # 받는 사람이 없으면 미설정
    import os
    assert "SMTP_HOST" not in os.environ


def test_smtp_configured_needs_host_port_and_recipient():
    base = {"SMTP_HOST": "localhost", "SMTP_PORT": "1025"}
    assert load_settings(env_file=None, environ=base).smtp_configured is False
    full = base | {"ALERT_EMAIL_TO": "me@example.com"}
    assert load_settings(env_file=None, environ=full).smtp_configured is True


def test_environment_overrides_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LOG_LEVEL=DEBUG\n", encoding="utf-8")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert load_settings(env_file=env).log_level == "WARNING"


def test_blank_values_fall_back_to_defaults():
    s = load_settings(env_file=None, environ={"DATA_DIR": "  ", "APP_TIMEZONE": "", "SMTP_PORT": ""})
    assert s.data_dir == Path("data").resolve()
    assert s.app_timezone == "Asia/Seoul"
    assert s.smtp_port is None


def test_ensure_data_dir_creates_folder(data_dir):
    s = load_settings(env_file=None)
    assert s.data_dir == data_dir.resolve()
    assert not data_dir.exists()
    s.ensure_data_dir()
    assert data_dir.is_dir()


def test_tz_property():
    assert load_settings(env_file=None).tz.key == "Asia/Seoul"


@pytest.mark.parametrize(
    "environ, name",
    [
        ({"DATA_GO_KR_TIMEOUT": "abc"}, "DATA_GO_KR_TIMEOUT"),
        ({"DATA_GO_KR_TIMEOUT": "-1"}, "DATA_GO_KR_TIMEOUT"),
        ({"SMTP_PORT": "smtp"}, "SMTP_PORT"),
        ({"APP_TIMEZONE": "Mars/Base"}, "APP_TIMEZONE"),
        ({"LLM_PROVIDER": "openai"}, "LLM_PROVIDER"),
    ],
)
def test_invalid_values_raise_config_error(environ, name):
    with pytest.raises(ConfigError, match=name):
        load_settings(env_file=None, environ=environ)


def test_config_error_message_does_not_contain_value():
    with pytest.raises(ConfigError) as info:
        load_settings(env_file=None, environ={"SMTP_PORT": "pw-" + FAKE_AWS_SECRET})
    assert FAKE_AWS_SECRET not in str(info.value)


def test_secrets_hidden_from_repr_and_str():
    s = load_settings(
        env_file=None,
        environ={
            "AWS_SECRET_ACCESS_KEY": FAKE_AWS_SECRET,
            "DATA_GO_KR_KEY": FAKE_DATA_KEY,
            "SMTP_PASSWORD": "fake-smtp-app-password",
            "AWS_REGION": "us-east-1",
        },
    )
    for text in (repr(s), str(s)):
        assert FAKE_AWS_SECRET not in text
        assert FAKE_DATA_KEY not in text
        assert "fake-smtp-app-password" not in text
        assert "us-east-1" in text  # 비밀이 아닌 값은 보임
    assert set(s.secret_values()) == {FAKE_AWS_SECRET, FAKE_DATA_KEY, "fake-smtp-app-password"}
