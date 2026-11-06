"""프롬프트 캐싱: make_llm(cache=True)이 실제 Bedrock Converse 요청에 cachePoint를 넣는지.

Bedrock은 부르지 않는다. langchain-aws가 만든 요청을 가짜 converse 클라이언트로 받아 본다."""

from dataclasses import replace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from config import load_settings
from llm import make_guard_llm, make_llm
from tracing import _usage

pytest.importorskip("langchain_aws")


class FakeConverse:
    """boto3 bedrock-runtime 클라이언트의 converse만 흉내 낸다."""

    def __init__(self):
        self.requests = []

    def converse(self, **params):
        self.requests.append(params)
        return {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1500, "outputTokens": 2, "totalTokens": 1502,
                          "cacheReadInputTokens": 1200, "cacheWriteInputTokens": 0},
                "stopReason": "end_turn", "metrics": {"latencyMs": 10}}


@pytest.fixture
def settings():
    s = load_settings(env_file=None, environ={"LLM_MODEL": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                                              "AWS_REGION": "us-east-1"})
    return replace(s, aws_access_key_id="AKIAFAKEFAKEFAKE", aws_secret_access_key="fake-secret-for-test")


def ask(llm, tools=None):
    fake = FakeConverse()
    llm.client = fake
    model = llm.bind_tools(tools) if tools else llm
    message = model.invoke([SystemMessage(content="시스템 규칙"), HumanMessage(content="앞 질문"),
                            HumanMessage(content="이번 질문")])
    return fake.requests[-1], message


def cache_points(blocks):
    return [b for b in blocks if isinstance(b, dict) and "cachePoint" in b]


def test_agent_llm_puts_cache_points_on_system_tools_and_messages(settings):
    def get_progress() -> str:
        """남은 시간."""
        return ""

    request, message = ask(make_llm(settings, cache=True), tools=[get_progress])
    assert cache_points(request["system"]) == [{"cachePoint": {"type": "default"}}]
    assert cache_points(request["toolConfig"]["tools"])
    assert cache_points(request["messages"][-1]["content"])
    # 캐시 읽기 토큰이 usage에 들어오고 trace에도 남는다
    assert message.usage_metadata["input_token_details"]["cache_read"] == 1200
    assert _usage(LLMResult(generations=[[ChatGeneration(message=message)]]))["cache_read"] == 1200


def test_guard_llm_uses_guard_model_without_cache(settings):
    haiku = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    guard = make_guard_llm(replace(settings, guard_model=haiku))
    assert guard.model_id == haiku
    request, _ = ask(guard)
    assert request["modelId"] == haiku and not cache_points(request["system"])
    assert make_guard_llm(settings).model_id == settings.llm_model  # GUARD_MODEL이 비면 LLM_MODEL


def test_cache_off_by_default_and_by_switch(settings):
    request, _ = ask(make_llm(settings))  # 가드레일·LLM 심사: cache 인자 없음
    assert not cache_points(request["system"]) and not cache_points(request["messages"][-1]["content"])
    request, _ = ask(make_llm(replace(settings, prompt_cache=False), cache=True))  # PROMPT_CACHE=off
    assert not cache_points(request["system"]) and not cache_points(request["messages"][-1]["content"])
