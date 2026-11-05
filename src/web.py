"""Streamlit 보조 화면(`app.py`)이 쓰는 백엔드. CLI와 같은 `open_app` + 입력 가드레일 + trace.

Streamlit은 입력마다 스크립트를 처음부터 다시 실행하므로, 앱(MCP 서버 세션 포함)은 세션마다 하나만 만들어
전용 스레드의 이벤트 루프에 열어 두고 질문만 넘긴다. 세션 1개 = 대화(thread) 1개.
"""

from __future__ import annotations

import asyncio
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from config import Settings
from masking import mask_exception

PDF_PATTERN = re.compile(r"[^\s'\"`<>()\[\]]+\.pdf", re.IGNORECASE)


@dataclass
class Reply:
    answer: str
    agents: list[str]
    tools: list[str]
    refusal: str | None
    pdfs: list[Path]


def pdfs_in(answer: str, data_dir: Path) -> list[Path]:
    """답변에 나온 PDF 경로 중 DATA_DIR 안에 실제로 있는 파일만(다운로드 버튼용)."""
    root = data_dir.resolve()
    found: list[Path] = []
    for raw in PDF_PATTERN.findall(answer):
        path = Path(raw.lstrip("*_:;,")).expanduser()  # 마크다운 강조 기호 등 앞쪽 군더더기 제거
        if not path.is_absolute():
            path = root / path
        try:
            path = path.resolve()
        except OSError:
            continue
        if path.is_file() and root in path.parents and path not in found:
            found.append(path)
    return found


def user_type(settings: Settings) -> str | None:
    from store import Store

    with Store.open(settings) as store:
        setting = store.get_user_setting()
    return setting.user_type if setting else None


def save_user_type(settings: Settings, value: str) -> None:
    from store import Store

    with Store.open(settings) as store:
        if store.get_user_setting() is None:
            store.init_user_setting(value)


class ChatBackend:
    """세션 하나의 대화. 앱은 처음 질문할 때 연다."""

    def __init__(self, settings: Settings, runtime: Any, api_mode: str = "live", today: date | None = None):
        self.settings = settings
        self.runtime = runtime
        self.api_mode = api_mode
        self.today = today
        self.thread_id = f"web-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="volunteer-web")
        self._thread.start()
        self._app = None
        self._cm = None
        self._lock = threading.Lock()

    def _run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _open(self) -> None:
        from agent import AppClock
        from tracing import FileTracer

        self.clock = AppClock.create(self.settings, self.today)
        self.guard = self.runtime.make_guard(self.settings)
        self.tracer = FileTracer.for_settings(self.settings, self.thread_id)
        self._cm = self.runtime.open_app(self.settings, self.clock, self.api_mode)
        self._app = await self._cm.__aenter__()

    async def _ask(self, text: str) -> Reply:
        from agent import ask

        if self._app is None:
            await self._open()
        turn = await ask(self._app, text, self.thread_id, guard=self.guard, tracer=self.tracer)
        return Reply(turn.answer, turn.agents, turn.tools, turn.refusal, pdfs_in(turn.answer, self.settings.data_dir))

    def ask(self, text: str) -> Reply:
        with self._lock:
            try:
                return self._run(self._ask(text))
            except Exception as exc:  # noqa: BLE001 - 화면에는 마스킹한 오류만
                return Reply(f"오류가 나서 답하지 못했습니다: {mask_exception(exc, self.settings.secret_values())}",
                             [], [], None, [])

    def close(self) -> None:
        if self._cm is not None:
            try:
                self._run(self._cm.__aexit__(None, None, None))
            except Exception:  # noqa: BLE001
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
