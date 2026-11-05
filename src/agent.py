"""앱 조립: MCP 도구 → 하위 Agent → Supervisor + checkpointer (`build_app`).

    입력 가드레일(14단계) → Supervisor(checkpointer) → policy/record/recommend/general Agent → 로컬 MCP 서버

대화는 thread_id 단위로 기억한다(CLI 실행 1회, Streamlit 세션 1개, 평가 문항 1개가 각각 thread 하나).
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import InjectedState
from langgraph_supervisor import create_supervisor

from agents import AGENT_TOOLS, PROGRESS_NOTE_PREFIX, ForcedToolChat, build_agents, supervisor_prompt
from memory import memory_tools, open_store
from config import Settings
from llm import get_text, make_llm
from rules import RuleEdits

TODAY_FILENAME = ".today"
SRC_DIR = Path(__file__).resolve().parent  # MCP 서버(tools.py)를 -m으로 띄울 때 쓰는 모듈 경로
RECURSION_LIMIT = 40


@dataclass
class AppClock:
    """기준 날짜. 파일에도 써서 MCP 서버가 같은 날짜를 쓰게 한다(--today, 시나리오 단계별 today)."""

    path: Path
    _today: date

    @classmethod
    def create(cls, settings: Settings, today: date | None = None) -> AppClock:
        clock = cls(settings.ensure_data_dir() / TODAY_FILENAME, today or datetime.now(settings.tz).date())
        clock.set(clock._today)
        return clock

    def today(self) -> date:
        return self._today

    def set(self, today: date) -> None:
        self._today = today
        self.path.write_text(today.isoformat(), encoding="utf-8")


def _last_user_text(state: dict) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return get_text(m)
    return ""


def bind_user_confirmation(confirm_tool: BaseTool) -> BaseTool:
    """confirm_rules의 확정 발화를 LLM이 채우지 못하게, 대화 상태의 실제 마지막 사용자 메시지로 넣는다(가드레일 4)."""

    async def confirm_rules(state: Annotated[dict, InjectedState], candidate_id: int | None = None,
                            edits: RuleEdits | None = None) -> str:
        args: dict[str, Any] = {"confirmation_text": _last_user_text(state)}
        if candidate_id is not None:
            args["candidate_id"] = candidate_id
        if edits is not None:
            args["edits"] = edits.model_dump(exclude_none=True)
        result = await confirm_tool.ainvoke(args)
        return result if isinstance(result, str) else get_text(AIMessage(content=result))

    description = (confirm_tool.description or "") + "\n(확정 발화는 사용자의 마지막 메시지가 자동으로 들어간다.)"
    return StructuredTool.from_function(coroutine=confirm_rules, name="confirm_rules", description=description)


def wire_tools(tools: list[BaseTool]) -> list[BaseTool]:
    return [bind_user_confirmation(t) if t.name == "confirm_rules" else t for t in tools]


class RouteFirstChat(ForcedToolChat):
    """Supervisor 모델 감싸개. 턴 시작(마지막 메시지가 사용자 발화)에는 tool_choice="any"로 불러
    반드시 transfer_to_* 하나를 고르게 한다. 하위 Agent가 돌아온 뒤에는 보통대로 불러 빈 응답으로 끝낼 수 있다.

    Supervisor가 앞 대화 내용으로 직접 답해 담당 Agent·도구를 건너뛰는 일이 있었다(시나리오 S2-4).
    프롬프트 메모로 막으려 하니 메모를 새 요청으로 받아 오히려 직접 답해서(17/26), 모델 호출 옵션으로 강제한다."""

    @property
    def _llm_type(self) -> str:
        return "route-first"

    def _force(self, messages: list[Any]) -> bool:
        last = messages[-1] if messages else None
        # 하위 Agent가 돌아온 뒤 붙는 진행 점검 메모(사람 메시지 형태)는 턴 시작이 아니다. 여기서 강제하면 끝낼 수 없다.
        return isinstance(last, HumanMessage) and not get_text(last).startswith(PROGRESS_NOTE_PREFIX)


def compile_app(llm_for: Callable[[str], Any], tools: list[BaseTool], clock: AppClock, checkpointer=None,
                store=None):
    """하위 Agent와 Supervisor를 조립해 컴파일한다. store가 있으면 선호 기억 도구(Python)를 붙인다."""
    all_tools = wire_tools(tools)
    if store is not None:
        names = {t.name for t in all_tools}
        all_tools += [t for t in memory_tools(store) if t.name not in names]
    agents = build_agents(llm_for, all_tools, clock.today)
    supervisor = create_supervisor(agents, model=RouteFirstChat(inner=llm_for("supervisor")), prompt=supervisor_prompt,
                                   add_handoff_back_messages=True, output_mode="full_history")
    return supervisor.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
                              store=store)


def mcp_server_config(settings: Settings, clock: AppClock, api_mode: str = "live",
                      fixtures: Path | None = None) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC_DIR), env.get("PYTHONPATH", "")) if p)
    env.update(DATA_DIR=str(settings.data_dir), VOLUNTEER_TODAY_FILE=str(clock.path),
               VOLUNTEER_API_MODE=api_mode, PYTHONIOENCODING="utf-8")
    if fixtures:
        env["VOLUNTEER_API_FIXTURES"] = str(fixtures)
    return {"command": sys.executable, "args": ["-m", "tools"], "transport": "stdio", "env": env}


@asynccontextmanager
async def open_app(settings: Settings, clock: AppClock, *, llm_for: Callable[[str], Any] | None = None,
                   api_mode: str = "live", fixtures: Path | None = None, checkpointer=None) -> AsyncIterator[Any]:
    """MCP 서버 세션을 열어 둔 채 앱을 돌려준다. 세션은 블록을 나갈 때 닫힌다."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    if llm_for is None:
        def llm_for(_role: str):
            return make_llm(settings)
    client = MultiServerMCPClient({"volunteer": mcp_server_config(settings, clock, api_mode, fixtures)})
    store = open_store(settings)
    try:
        async with client.session("volunteer") as session:
            tools = await load_mcp_tools(session)
            yield compile_app(llm_for, tools, clock, checkpointer, store)
    finally:
        store.conn.close()


async def build_app(settings: Settings, clock: AppClock, **kwargs: Any):
    """세션을 매번 여는 단순 버전(도구 호출마다 MCP 서버를 띄움). 오래 쓰는 화면은 open_app을 쓴다."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    llm_for = kwargs.pop("llm_for", None) or (lambda _role: make_llm(settings))
    client = MultiServerMCPClient({"volunteer": mcp_server_config(settings, clock, kwargs.pop("api_mode", "live"),
                                                                  kwargs.pop("fixtures", None))})
    tools = await client.get_tools()
    return compile_app(llm_for, tools, clock, kwargs.pop("checkpointer", None), open_store(settings))


@dataclass
class Turn:
    answer: str
    messages: list[Any]
    agents: list[str]
    tools: list[str]
    refusal: str | None = None  # 입력 가드레일이 막은 범주(이때 Supervisor는 호출되지 않음)


class ToolRecorder(BaseCallbackHandler):
    """실제로 실행된 도구 이름을 순서대로 모은다(ReAct Agent와 planner 그래프 모두)."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self._tool_runs: set = set()

    def on_tool_start(self, serialized: dict | None, input_str: str, **kwargs: Any) -> None:
        run_id, parent = kwargs.get("run_id"), kwargs.get("parent_run_id")
        self._tool_runs.add(run_id)
        if parent in self._tool_runs:  # 감싼 도구 안에서 다시 부른 같은 도구(confirm_rules 주입판)는 한 번만
            return
        name = (serialized or {}).get("name") or kwargs.get("name") or ""
        if name and not name.startswith(("transfer_to_", "transfer_back_to_")):
            self.names.append(name)


async def ask(app: Any, text: str, thread_id: str, callbacks: list | None = None, guard: Any | None = None,
              tracer: Any | None = None) -> Turn:
    """질문 하나를 thread_id 대화로 실행하고, 이번 턴에서 거친 Agent와 부른 도구를 함께 돌려준다.

    guard(입력 가드레일)가 있으면 Supervisor보다 먼저 검사하고, 막히면 고정 거절 문구 하나만 돌려준다.
    tracer(FileTracer)가 있으면 질문 단위 이벤트(turn_start/guard_block/turn_end)와 모든 LLM·도구 호출을 남긴다.
    """
    started = time.perf_counter()
    callbacks = [*(callbacks or []), *([tracer] if tracer is not None else [])]
    if tracer is not None:
        tracer.write("turn_start", thread_id=thread_id, input=tracer._sum(text))
    if guard is not None:
        checked = await guard.check(text, callbacks=callbacks or None)
        if checked.blocked:
            turn = Turn(answer=checked.message, messages=[], agents=[], tools=[], refusal=checked.category)
            if tracer is not None:
                tracer.write("guard_block", thread_id=thread_id, category=checked.category, layer=checked.layer)
                _trace_turn_end(tracer, thread_id, turn, started)
            return turn
    recorder = ToolRecorder()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT,
              "callbacks": [recorder, *callbacks]}
    before = len((await app.aget_state(config)).values.get("messages", [])) if app.checkpointer else 0
    try:
        result = await app.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
    except Exception as exc:
        if tracer is not None:
            tracer.write("turn_error", thread_id=thread_id, error=tracer._sum(f"{type(exc).__name__}: {exc}"),
                         latency_s=round(time.perf_counter() - started, 3))
        raise
    new = result["messages"][before:]
    agents = []
    for m in new:
        for tc in getattr(m, "tool_calls", None) or []:
            agent = tc["name"].removeprefix("transfer_to_")
            if tc["name"].startswith("transfer_to_") and agent in AGENT_TOOLS and agent not in agents:
                agents.append(agent)
    answer = agents_answer(new, agents)
    turn = Turn(answer=answer or final_answer(new), messages=new, agents=agents, tools=recorder.names)
    if tracer is not None:
        _trace_turn_end(tracer, thread_id, turn, started)
    return turn


def _trace_turn_end(tracer: Any, thread_id: str, turn: Turn, started: float) -> None:
    tracer.write("turn_end", thread_id=thread_id, agents=turn.agents, tools=turn.tools, refusal=turn.refusal,
                 output=tracer._sum(turn.answer), latency_s=round(time.perf_counter() - started, 3))


def agents_answer(messages: list[Any], agents: list[str]) -> str:
    """하위 Agent를 거친 턴의 답: 거친 순서대로 각 Agent의 마지막 답을 잇는다.

    Supervisor의 요약은 쓰지 않는다. 여러 Agent를 거친 턴에서 Supervisor가 실제 결과(추천 목록·링크) 대신
    '모든 처리가 완료되었습니다' 같은 요약으로 덮어쓰는 일이 있어서(시나리오 S3-1) 사실은 Agent의 답 그대로 전한다."""
    parts = []
    for agent in agents:
        text = final_answer([m for m in messages if getattr(m, "name", None) == agent])
        if text and text not in parts:
            parts.append(text)
    return "\n\n".join(parts)


_HANDOFF_NOTE = re.compile(r"\[Called transfer_\w+\]")


def final_answer(messages: list[Any]) -> str:
    """이번 턴의 최종 답.

    Supervisor는 하위 Agent 하나만 거친 경우 빈 응답(또는 '.')으로 끝내도록 되어 있다. 그래서 마지막으로 나온
    '의미 있는' AI 답을 쓴다. Agent의 답은 transfer_back_to_supervisor 호출과 같은 메시지에 들어 있기도 하므로
    그 메시지도 후보로 보고, 끝에 붙은 핸드오프 안내 문구는 지운다."""
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        calls = m.tool_calls or []
        if calls and not all(c["name"].startswith("transfer_back_to_") for c in calls):
            continue
        text = get_text(m)
        for marker in ("Transferring back to supervisor", "Successfully transferred back to supervisor"):
            text = text.replace(marker, "")
        text = _HANDOFF_NOTE.sub("", text)
        text = text.strip().strip("-").strip()
        if len(text) >= 2 and any(ch.isalnum() for ch in text):
            return text
    return ""
