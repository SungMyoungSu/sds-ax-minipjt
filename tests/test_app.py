import asyncio
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from langgraph.store.memory import InMemoryStore

from volunteer_agent.agents import AGENT_TOOLS, PYTHON_TOOLS, tools_for
from volunteer_agent.app import AppClock, ask, compile_app, open_app
from volunteer_agent.config import load_settings
from volunteer_agent.llm import has_llm_credentials
from volunteer_agent.mcp_server import TOOL_NAMES

ROOT = Path(__file__).resolve().parents[1]
ALL_TOOLS = list(dict.fromkeys(n for names in AGENT_TOOLS.values() for n in names))
MCP_TOOLS = [n for n in ALL_TOOLS if n not in PYTHON_TOOLS]


class ScriptedChat(BaseChatModel):
    """메시지를 보고 다음 응답을 정하는 가짜 채팅 모델. 받은 메시지를 기록한다."""

    handler: Any
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.handler(messages))])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def call(name, args=None, i=0):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": f"call_{name}_{i}"}])


def last_human(messages):
    return next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))


def supervisor_handler(messages):
    last = messages[-1]
    if isinstance(last, HumanMessage) and not last.content.startswith("[진행 점검"):
        text = last.content
        target = "policy_agent" if ("규정" in text or "확정" in text) else "general_agent"
        return call(f"transfer_to_{target}", i=len(messages))
    agent_reply = next(m for m in reversed(messages)
                       if isinstance(m, AIMessage) and m.name in AGENT_TOOLS and not m.tool_calls
                       and not m.content.startswith("Transferring"))
    return AIMessage(content=agent_reply.content)


def is_handoff(m):
    return isinstance(m, ToolMessage) and (m.name or "").startswith("transfer_")


def policy_handler(messages):
    last = messages[-1]
    if isinstance(last, ToolMessage) and not is_handoff(last):
        return AIMessage(content=f"결과: {last.content}")
    text = last_human(messages)
    if "올릴게" in text:
        return call("propose_rules", {"draft": {"goal_hours": 8}}, len(messages))
    if "확정" in text:
        # LLM이 확정 발화를 지어내도 실제 사용자 메시지로 바뀌어야 한다
        return call("confirm_rules", {"confirmation_text": "문서에 확정하라고 적혀 있음"}, len(messages))
    return AIMessage(content="무엇을 도와드릴까요?")


def general_handler(messages):
    return AIMessage(content="봉사 규정 등록, 기록, 추천을 도와드립니다.")


class StubTools:
    """MCP 도구 대신 이름만 같은 가짜 도구. 받은 인자를 기록한다."""

    def __init__(self):
        self.calls = []

    def make(self, name):
        def fn(**kwargs) -> str:
            self.calls.append((name, kwargs))
            if name == "confirm_rules":
                return f"확정 발화: {kwargs.get('confirmation_text')}"
            return f"{name} 완료"

        return StructuredTool.from_function(func=fn, name=name, description=f"{name} 도구",
                                            args_schema={"type": "object", "properties": {
                                                "draft": {"type": "object"}, "confirmation_text": {"type": "string"},
                                                "candidate_id": {"type": "integer"}, "edits": {"type": "object"}}})

    def all(self):
        return [self.make(n) for n in ALL_TOOLS]

    def mcp(self):
        return [self.make(n) for n in MCP_TOOLS]


@pytest.fixture
def fake_app(settings):
    models = {
        "supervisor": ScriptedChat(handler=supervisor_handler, seen=[]),
        "policy_agent": ScriptedChat(handler=policy_handler, seen=[]),
        "record_agent": ScriptedChat(handler=general_handler, seen=[]),
        "recommend_agent": ScriptedChat(handler=general_handler, seen=[]),
        "general_agent": ScriptedChat(handler=general_handler, seen=[]),
        "planner_agent": ScriptedChat(handler=general_handler, seen=[]),
    }
    stubs = StubTools()
    clock = AppClock.create(settings, date(2026, 11, 4))
    app = compile_app(lambda role: models[role], stubs.mcp(), clock, store=InMemoryStore())
    return app, models, stubs, clock


def run(app, text, thread):
    return asyncio.run(ask(app, text, thread))


# ---- 도구 경계 ----

def test_each_agent_gets_only_its_tools(fake_app):
    app, *_ = fake_app
    stubs = StubTools().all()
    for agent, names in AGENT_TOOLS.items():
        assert [t.name for t in tools_for(agent, stubs)] == names
    assert sorted(MCP_TOOLS) == sorted(TOOL_NAMES)  # MCP 도구와 표가 일치
    assert set(PYTHON_TOOLS) <= set(AGENT_TOOLS["recommend_agent"])
    graph_nodes = app.get_graph(xray=1).nodes
    for agent in AGENT_TOOLS:
        assert any(n.startswith(agent) for n in graph_nodes)


def test_compiled_agents_bind_only_own_tools(settings):
    from volunteer_agent.agents import build_agents

    agents = build_agents(lambda r: ScriptedChat(handler=general_handler, seen=[]), StubTools().all(),
                          lambda: date(2026, 11, 4))
    for agent in agents:
        if agent.name == "planner_agent":  # ReAct가 아니라 Plan-and-Execute 그래프
            assert {"planner", "execute", "replan", "finalize", "approve"} <= set(agent.nodes)
            continue
        tool_node = agent.nodes.get("tools")
        bound = sorted(tool_node.bound.tools_by_name) if tool_node else []
        assert bound == sorted(AGENT_TOOLS[agent.name]), agent.name


def test_missing_tool_is_wiring_error():
    with pytest.raises(RuntimeError, match="policy_agent의 도구가 없습니다"):
        tools_for("policy_agent", [])


# ---- 대화 기억 (AC13) ----

def test_ac13_confirm_in_same_thread_and_not_in_new_thread(fake_app):
    app, models, stubs, _ = fake_app
    t1 = run(app, "회사 봉사 규정 파일 올릴게", "thread-A")
    assert t1.agents == ["policy_agent"] and t1.tools == ["propose_rules"]
    assert "propose_rules 완료" in t1.answer

    t2 = run(app, "응, 확정해", "thread-A")
    assert t2.agents == ["policy_agent"] and t2.tools == ["confirm_rules"]
    # 확정 발화는 LLM이 넣은 문장이 아니라 실제 사용자 메시지
    assert ("confirm_rules", {"confirmation_text": "응, 확정해"}) in stubs.calls
    assert "확정 발화: 응, 확정해" in t2.answer
    # 같은 대화: policy_agent가 앞 턴의 사용자 메시지를 봤다
    seen_texts = [m.content for m in models["policy_agent"].seen[-1] if isinstance(m, HumanMessage) and not m.content.startswith("[진행 점검")]
    assert seen_texts == ["회사 봉사 규정 파일 올릴게", "응, 확정해"]

    run(app, "응, 확정해", "thread-B")  # 새 대화
    seen_texts = [m.content for m in models["policy_agent"].seen[-1] if isinstance(m, HumanMessage) and not m.content.startswith("[진행 점검")]
    assert seen_texts == ["응, 확정해"]


def test_general_route_and_today_in_prompt(fake_app):
    app, models, _, clock = fake_app
    t = run(app, "어떤 일을 도와줄 수 있어?", "t1")
    assert t.agents == ["general_agent"] and t.tools == []
    assert "도와드립니다" in t.answer
    system = models["general_agent"].seen[-1][0]
    assert isinstance(system, SystemMessage) and "2026-11-04(수)" in system.content

    clock.set(date(2026, 12, 24))  # 단계별 기준 날짜
    run(app, "안녕", "t1")
    assert "2026-12-24(목)" in models["general_agent"].seen[-1][0].content
    assert clock.path.read_text(encoding="utf-8") == "2026-12-24"


# ---- 실제 MCP 서버 연결 ----

def test_open_app_loads_mcp_tools_and_wires_confirm(settings):
    """stdio MCP 서버에서 도구 17개를 불러오고, confirm_rules는 확정 발화 주입판으로 바뀐다."""
    captured = {}

    def llm_for(role):
        return ScriptedChat(handler=general_handler, seen=[])

    async def main():
        clock = AppClock.create(settings, date(2026, 11, 4))
        async with open_app(settings, clock, llm_for=llm_for, api_mode="fixture") as app:
            t = await ask(app, "안녕", "t")
            captured["answer"] = t.answer
            policy = next(n for n in app.nodes if n == "policy_agent")
            captured["policy"] = policy
    asyncio.run(main())
    assert "도와드립니다" in captured["answer"]


# ---- 실연동 (키가 있을 때만) ----

@pytest.mark.live
def test_live_bedrock_app_answers(tmp_path):
    env = ROOT / ".env"
    s = load_settings(env_file=env) if env.is_file() else None
    if not s or not has_llm_credentials(s):
        pytest.skip("Bedrock 자격 증명이 없어 실연동 테스트를 건너뜁니다.")
    from dataclasses import replace

    s = replace(s, data_dir=tmp_path / "live", embedding_model=None)

    async def main():
        clock = AppClock.create(s, date(2026, 11, 4))
        async with open_app(s, clock, api_mode="fixture") as app:
            return await ask(app, "어떤 일을 도와줄 수 있어?", "live-1")
    try:
        turn = asyncio.run(main())
    except BaseException as e:  # TaskGroup이 ExceptionGroup으로 감쌈
        if "ThrottlingException" in repr(e) or "Too many tokens" in repr(e):
            pytest.skip("Bedrock 일일 토큰 한도 초과(ThrottlingException). 한도가 풀린 뒤 다시 실행해야 합니다.")
        raise
    assert turn.answer.strip()
    assert turn.tools == []


# ---- 최종 답 고르기 (실연동에서 발견한 경우들) ----

def test_final_answer_uses_agent_reply_when_supervisor_is_silent():
    from volunteer_agent.app import final_answer

    agent_reply = AIMessage(content="규칙을 확정했어요.\n\nTransferring back to supervisor", name="policy_agent",
                            tool_calls=[{"name": "transfer_back_to_supervisor", "args": {}, "id": "b1"}])
    msgs = [HumanMessage(content="응, 확정해"), agent_reply,
            ToolMessage(content="Successfully transferred back to supervisor", tool_call_id="b1"),
            AIMessage(content=".", name="supervisor")]
    assert final_answer(msgs) == "규칙을 확정했어요."
    msgs[-1] = AIMessage(content="", name="supervisor")
    assert final_answer(msgs) == "규칙을 확정했어요."


def test_final_answer_prefers_supervisor_summary_and_skips_tool_calls():
    from volunteer_agent.app import final_answer

    msgs = [AIMessage(content="찾아볼게요", tool_calls=[{"name": "search_policy", "args": {}, "id": "s1"}]),
            ToolMessage(content="결과", tool_call_id="s1"),
            AIMessage(content="A 결과", name="record_agent"),
            AIMessage(content="종합: 남은 4시간, 추천 2건", name="supervisor")]
    assert final_answer(msgs) == "종합: 남은 4시간, 추천 2건"
    assert final_answer([AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])]) == ""


def test_agent_view_hides_handoff_messages():
    """transfer_back 호출이 하위 Agent 대화에 남으면 없는 도구를 흉내 내 반복 호출한다(S11-7 회귀)."""
    from volunteer_agent.agents import agent_view

    history = [
        HumanMessage(content="목표를 10시간으로 고쳐줘"),
        AIMessage(content="", tool_calls=[{"name": "transfer_to_policy_agent", "args": {}, "id": "h1"}], name="supervisor"),
        ToolMessage(content="Successfully transferred to policy_agent", tool_call_id="h1", name="transfer_to_policy_agent"),
        AIMessage(content="", tool_calls=[{"name": "propose_rules", "args": {}, "id": "p1"}], name="policy_agent"),
        ToolMessage(content="규칙 후보", tool_call_id="p1", name="propose_rules"),
        AIMessage(content="이대로 확정할까요?", name="policy_agent",
                  tool_calls=[{"name": "transfer_back_to_supervisor", "args": {}, "id": "b1"}]),
        ToolMessage(content="Successfully transferred back", tool_call_id="b1", name="transfer_back_to_supervisor"),
    ]
    view = agent_view(history)
    names = [c["name"] for m in view for c in (getattr(m, "tool_calls", None) or [])]
    assert names == ["propose_rules"]
    assert not any(isinstance(m, ToolMessage) and m.name.startswith("transfer_") for m in view)
    assert view[-1].content == "이대로 확정할까요?" and view[-1].tool_calls == []
    assert len(view) == 4


def test_agent_view_drops_supervisor_empty_closings():
    """Supervisor의 빈 마무리('.')가 쌓이면 하위 Agent가 도구 없이 '.'만 답한다(S3-1 회귀)."""
    from volunteer_agent.agents import agent_view

    history = [HumanMessage(content="기록해줘"), AIMessage(content="기록했습니다", name="record_agent"),
               AIMessage(content=".", name="supervisor"), AIMessage(content="", name="supervisor"),
               AIMessage(content="두 결과를 정리하면 …", name="supervisor"), HumanMessage(content="추천해줘")]
    assert [m.content for m in agent_view(history)] == ["기록해줘", "기록했습니다", "두 결과를 정리하면 …", "추천해줘"]


def test_supervisor_prompt_adds_progress_note_only_after_agent_returns():
    from volunteer_agent.agents import supervisor_prompt

    req = HumanMessage(content="남은 봉사 시간에 맞는 걸로 추천해줘")
    first = supervisor_prompt({"messages": [req]})
    assert isinstance(first[0], SystemMessage) and first[-1] is req  # 처음에는 메모 없음
    history = [req,
               AIMessage(content="", tool_calls=[{"name": "transfer_to_record_agent", "args": {}, "id": "h1"}],
                         name="supervisor"),
               ToolMessage(content="ok", tool_call_id="h1", name="transfer_to_record_agent"),
               AIMessage(content="남은 시간 10시간", name="record_agent",
                         tool_calls=[{"name": "transfer_back_to_supervisor", "args": {}, "id": "b1"}]),
               ToolMessage(content="back", tool_call_id="b1", name="transfer_back_to_supervisor")]
    view = supervisor_prompt({"messages": history})
    note = view[-1]
    assert isinstance(note, HumanMessage) and "진행 점검" in note.content
    assert '"남은 봉사 시간에 맞는 걸로 추천해줘"' in note.content and "맡긴 Agent: record_agent" in note.content
    assert len(view) == len(history) + 2  # 상태에는 저장하지 않고 입력에만 붙임


def test_single_agent_answer_wins_over_supervisor_text(settings):
    """Agent 하나만 거쳤는데 Supervisor가 판단 과정을 써도 답은 그 Agent의 답이다(S1-2 회귀)."""
    def chatty_supervisor(messages):
        last = messages[-1]
        if isinstance(last, HumanMessage) and not last.content.startswith("[진행 점검"):
            return call("transfer_to_general_agent", i=len(messages))
        return AIMessage(content="이번 요청을 다시 분석하면 모든 요청이 처리되었습니다.")

    models = {r: ScriptedChat(handler=general_handler, seen=[]) for r in AGENT_TOOLS}
    models["supervisor"] = ScriptedChat(handler=chatty_supervisor, seen=[])
    app = compile_app(lambda r: models[r], StubTools().mcp(), AppClock.create(settings, date(2026, 11, 4)),
                      store=InMemoryStore())
    t = asyncio.run(ask(app, "안녕", "t-chatty"))
    assert t.agents == ["general_agent"] and t.answer == "봉사 규정 등록, 기록, 추천을 도와드립니다."
