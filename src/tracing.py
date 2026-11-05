"""실행 추적: LLM·도구 호출을 `DATA_DIR/trace.jsonl`에 한 줄씩 남긴다(AC10, `reference/local_tracer.py` 방식).

한 줄(TraceEvent): event, run_id, tool, input, output, latency_s, usage, ts (+ agent, thread_id 등 보조 필드)

이벤트 종류
- turn_start / turn_end: 질문 하나의 시작과 끝(거친 Agent, 부른 도구, 전체 지연시간)
- guard_block: 입력 가드레일이 막은 경우(범주와 판별 층만, 사유는 guard_audit.log에)
- llm_start / llm_end / llm_error: 채팅 모델 호출(토큰 수 포함)
- tool_start / tool_end / tool_error: 도구 호출
- handoff: Agent 이동(Supervisor → Agent는 transfer_to_* 도구, Agent → Supervisor는 최상위 노드 전환)
- embedding_fallback: 규정 검색 임베딩 실패로 BM25만 쓴 경우(MCP 서버 프로세스가 같은 파일에 직접 쓴다)

모든 문자열은 마스킹한 뒤에 200자로 자른다(자르고 나서 가리면 키 앞부분이 남을 수 있다).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from masking import mask_secrets

TRACE_FILENAME = "trace.jsonl"
SUMMARY_LEN = 200

_lock = threading.Lock()


def summarize(value: Any, secrets: Iterable[str | None] = ()) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(mask_secrets(text, secrets).split())
    return text if len(text) <= SUMMARY_LEN else text[:SUMMARY_LEN] + "…"


def append_event(path: Path, record: dict[str, Any], secrets: Iterable[str | None] = ()) -> None:
    """trace.jsonl에 한 줄 추가. 문자열 값은 모두 마스킹한다(다른 프로세스도 같은 함수로 쓴다)."""
    secrets = list(secrets)
    line = {k: (mask_secrets(v, secrets) if isinstance(v, str) else v) for k, v in record.items()}
    line.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


def trace_path(settings: Any) -> Path:
    return settings.ensure_data_dir() / TRACE_FILENAME


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


def _agent_of(metadata: dict | None, tags: list | None = None) -> str | None:
    """LangGraph 메타데이터에서 실행 중인 Agent 이름(supervisor, policy_agent …)을 꺼낸다. 가드레일 판별은 input_guard."""
    if tags and "input_guard" in tags:
        return "input_guard"
    metadata = metadata or {}
    ns = metadata.get("langgraph_checkpoint_ns") or ""
    if ns:
        return ns.split("|")[0].split(":")[0]
    return metadata.get("langgraph_node")


def _usage(response: Any) -> dict[str, Any]:
    try:
        message = response.generations[0][0].message
    except (AttributeError, IndexError):
        return {}
    usage = getattr(message, "usage_metadata", None) or {}
    return {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens") if k in usage}


def _output(response: Any) -> str:
    try:
        gen = response.generations[0][0]
    except (AttributeError, IndexError):
        return ""
    message = getattr(gen, "message", None)
    text = _text(message) if message is not None else (gen.text or "")
    calls = [c["name"] for c in (getattr(message, "tool_calls", None) or [])]
    return f"{text} → tool_calls: {calls}".strip() if calls else text


class FileTracer(BaseCallbackHandler):
    """LLM·도구 콜백을 trace.jsonl로 남긴다. ask(..., tracer=)로 넘기면 질문 단위 이벤트도 쓴다."""

    def __init__(self, path: Path, secrets: Iterable[str | None] = (), thread_id: str | None = None):
        self.path = Path(path)
        self.secrets = [s for s in secrets if s]
        self.thread_id = thread_id
        self._starts: dict[Any, tuple[float, str | None]] = {}
        self._names: dict[Any, str] = {}
        self._node: str | None = None  # 지금 실행 중인 최상위 노드(supervisor 또는 하위 Agent)

    @classmethod
    def for_settings(cls, settings: Any, thread_id: str | None = None) -> FileTracer:
        return cls(trace_path(settings), settings.secret_values(), thread_id)

    def write(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {"event": event}
        if self.thread_id:
            record["thread_id"] = self.thread_id
        record.update({k: v for k, v in fields.items() if v is not None})
        append_event(self.path, record, self.secrets)

    def _sum(self, value: Any) -> str:
        return summarize(value, self.secrets)

    def _start(self, run_id: Any, metadata: dict | None, tags: list | None = None) -> str | None:
        agent = _agent_of(metadata, tags)
        self._starts[run_id] = (time.perf_counter(), agent)
        return agent

    def _stop(self, run_id: Any) -> tuple[float, str | None]:
        started, agent = self._starts.pop(run_id, (time.perf_counter(), None))
        return round(time.perf_counter() - started, 3), agent

    # ---- Agent 이동 ----
    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kw):
        """하위 Agent → Supervisor 복귀는 실제 도구 실행이 아니라(합성 메시지) 최상위 노드가 바뀌는 것으로 잡는다."""
        node = _agent_of(metadata)
        if not node or node == self._node:
            return
        if node == "supervisor" and self._node and self._node.endswith("_agent"):
            self.write("handoff", source=self._node, target="supervisor")
        self._node = node

    # ---- LLM ----
    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, tags=None, **kw):
        agent = self._start(run_id, metadata, tags)
        flat = [m for batch in messages for m in batch]
        last = _text(flat[-1]) if flat else ""
        self.write("llm_start", run_id=str(run_id), agent=agent, input=self._sum(last), n_messages=len(flat))

    def on_llm_end(self, response, *, run_id, **kw):
        latency, agent = self._stop(run_id)
        self.write("llm_end", run_id=str(run_id), agent=agent, output=self._sum(_output(response)),
                   latency_s=latency, usage=_usage(response))

    def on_llm_error(self, error, *, run_id, **kw):
        latency, agent = self._stop(run_id)
        self.write("llm_error", run_id=str(run_id), agent=agent, error=self._sum(f"{type(error).__name__}: {error}"),
                   latency_s=latency)

    # ---- 도구 ----
    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kw):
        agent = self._start(run_id, metadata)
        name = (serialized or {}).get("name") or kw.get("name") or ""
        self.write("tool_start", run_id=str(run_id), agent=agent, tool=name, input=self._sum(input_str))
        if name.startswith("transfer_to_"):
            self.write("handoff", run_id=str(run_id), source=agent, target=name.removeprefix("transfer_to_"))
        self._names[run_id] = name

    def _tool_name(self, run_id: Any) -> str | None:
        return self._names.pop(run_id, None)

    def on_tool_end(self, output, *, run_id, **kw):
        latency, agent = self._stop(run_id)
        content = getattr(output, "content", output)
        self.write("tool_end", run_id=str(run_id), agent=agent, tool=self._tool_name(run_id),
                   output=self._sum(_text(content) if isinstance(content, list) else content), latency_s=latency)

    def on_tool_error(self, error, *, run_id, **kw):
        latency, agent = self._stop(run_id)
        self.write("tool_error", run_id=str(run_id), agent=agent, tool=self._tool_name(run_id),
                   error=self._sum(f"{type(error).__name__}: {error}"), latency_s=latency)


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
