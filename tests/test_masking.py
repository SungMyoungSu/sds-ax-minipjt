from volunteer_agent.config import load_settings
from volunteer_agent.masking import MASK, mask_exception, mask_secrets

FAKE_AKIA = "AKIAABCDEFGHIJKLMNOP"
FAKE_SECRET = "fakeSecretAccessKey0123456789abcdefXYZ"
FAKE_SERVICE_KEY = "fakeDataGoKrKey+/=0123456789"


def test_aws_access_key_id_pattern():
    assert mask_secrets(f"key id is {FAKE_AKIA} ok") == f"key id is {MASK} ok"
    assert FAKE_AKIA.replace("AKIA", "ASIA") not in mask_secrets(FAKE_AKIA.replace("AKIA", "ASIA"))


def test_anthropic_key_pattern():
    assert "sk-ant-api03-abc_DEF-123" not in mask_secrets("token sk-ant-api03-abc_DEF-123 end")


def test_service_key_in_url():
    url = f"https://apis.data.go.kr/x/getList?serviceKey={FAKE_SERVICE_KEY}&pageNo=1"
    masked = mask_secrets(url)
    assert FAKE_SERVICE_KEY not in masked
    assert f"serviceKey={MASK}&pageNo=1" in masked


def test_named_assignment_pattern():
    text = 'AWS_SECRET_ACCESS_KEY=abc123xyz SMTP_PASSWORD: "hunter22" api_token=t0k3n'
    masked = mask_secrets(text)
    for value in ("abc123xyz", "hunter22", "t0k3n"):
        assert value not in masked
    assert "AWS_SECRET_ACCESS_KEY=" in masked


def test_known_secret_values_masked_anywhere():
    s = load_settings(env_file=None, environ={"AWS_SECRET_ACCESS_KEY": FAKE_SECRET})
    text = f"signature mismatch for {FAKE_SECRET}!"
    assert mask_secrets(text, s.secret_values()) == f"signature mismatch for {MASK}!"


def test_exception_message_masked():
    exc = RuntimeError(f"request failed: serviceKey={FAKE_SERVICE_KEY} id={FAKE_AKIA}")
    masked = mask_exception(exc)
    assert masked.startswith("RuntimeError: request failed")
    assert FAKE_SERVICE_KEY not in masked
    assert FAKE_AKIA not in masked


def test_plain_text_unchanged():
    text = "오늘 요양원에서 3시간 봉사했어. 남은 시간은 5.5시간입니다. keyboard monkey"
    assert mask_secrets(text) == text


def test_short_or_empty_secrets_ignored():
    assert mask_secrets("a b c", ["", None, "ab"]) == "a b c"


def test_already_masked_is_stable():
    once = mask_secrets("SMTP_PASSWORD=hunter22")
    assert mask_secrets(once) == once


def test_keys_glued_to_korean_text_are_masked():
    # \b는 한글을 단어 문자로 봐서 '가AKIA…'를 놓쳤다(trace 200자 요약에서 발견)
    out = mask_secrets("키는AKIAFAKEFAKEFAKEFAKE입니다 sk-ant-abc123 그리고API_KEY=xyz789")
    assert "AKIA" not in out and "sk-ant" not in out and "xyz789" not in out
