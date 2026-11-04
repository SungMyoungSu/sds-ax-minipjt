"""local_tracer.py - 실행 기록을 JSONL 파일로 남기는 콜백 핸들러"""
import json
import time
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler


def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


class FileTracer(BaseCallbackHandler):
    """LLM 호출, 도구 호출을 JSONL 한 줄씩 기록한다."""

    def __init__(self, path: str = "trace.jsonl"):
        self.path = Path(path)
        self._starts = {}   # run_id -> 시작 시각

    def _write(self, record: dict):
        record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- LLM ----
    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._starts[run_id] = time.time()
        flat = [get_text(m)[:200] for batch in messages for m in batch]
        self._write({"event": "llm_start", "run_id": str(run_id), "input": flat})

    def on_llm_end(self, response, *, run_id, **kw):
        elapsed = time.time() - self._starts.pop(run_id, time.time())
        usage = {}
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        if msg is not None and getattr(msg, "usage_metadata", None):
            usage = msg.usage_metadata     # 입력, 출력 토큰 수
        self._write({
            "event": "llm_end", "run_id": str(run_id),
            "output": gen.text[:200], "latency_s": round(elapsed, 2),
            "usage": usage,
        })

    # ---- 도구 ----
    def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self._starts[run_id] = time.time()
        self._write({"event": "tool_start", "run_id": str(run_id),
                     "tool": serialized.get("name"), "input": input_str[:200]})

    def on_tool_end(self, output, *, run_id, **kw):
        elapsed = time.time() - self._starts.pop(run_id, time.time())
        self._write({"event": "tool_end", "run_id": str(run_id),
                     "output": str(output)[:200], "latency_s": round(elapsed, 2)})

    def on_llm_error(self, error, *, run_id, **kw):
        self._write({"event": "llm_error", "run_id": str(run_id), "error": str(error)})
