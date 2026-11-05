"""`volunteer-agent` 명령 (CLI가 주 화면이자 평가 대상).

    volunteer-agent [공통 옵션] --once "질문"     질문 하나에 답하고 종료
    volunteer-agent [공통 옵션] chat              대화형(실행 1회 = 대화 1개, '종료'로 끝)
    volunteer-agent export [--output 경로]        DATA_DIR(DB·확인서·보고서) zip 내보내기
    volunteer-agent purge [--confirm 문구]        DATA_DIR 삭제(확인 문구가 정확해야 함)
    volunteer-agent notify [--today 날짜]         마감 알림 점검 1회(Windows 작업 스케줄러용)

공통 옵션: --user-type employee|student(첫 실행 질문 대신), --today YYYY-MM-DD, --show-trace, --api-mode fixture|live,
--attach 파일(--once와 함께, 규정·확인서 파일 첨부)
시나리오 평가(`scenario`)는 17단계에서 붙인다.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sqlite3
import sys
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

from volunteer_agent import __version__
from volunteer_agent.config import Settings, load_settings
from volunteer_agent.masking import mask_exception

USER_TYPES = {"employee": "회사원", "student": "학생"}
_TYPE_ANSWERS = {"1": "employee", "회사원": "employee", "직장인": "employee", "employee": "employee",
                 "2": "student", "학생": "student", "student": "student"}
EXIT_WORDS = {"종료", "exit", "quit", "/exit", "/quit", "/종료", "그만"}
ATTACH_COMMAND = "/첨부"
PURGE_PHRASE = "내 데이터 모두 삭제"
EXPORT_DIRNAME = "exports"
MAX_ASKS = 3


# ---------------------------------------------------------------- 입출력 도우미

@dataclass
class Console:
    stdin: TextIO
    stdout: TextIO

    def say(self, text: str = "") -> None:
        self.stdout.write(text + "\n")
        self.stdout.flush()

    def ask(self, prompt: str) -> str | None:
        """한 줄 입력. 입력이 끝났으면(EOF) None."""
        self.stdout.write(prompt)
        self.stdout.flush()
        line = self.stdin.readline()
        return None if line == "" else line.strip()


def parse_today(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"날짜는 YYYY-MM-DD 형식이어야 합니다: {value}") from None


# ---------------------------------------------------------------- 첫 실행(사용자 유형, 알림 동의)

def ensure_user(settings: Settings, console: Console, user_type: str | None) -> str | None:
    """PC당 사용자 1명. 처음 실행할 때 유형을 저장하고, 이후에는 묻지 않고 바꾸지도 않는다(AC13)."""
    from volunteer_agent.store import Store

    with Store.open(settings) as store:
        setting = store.get_user_setting()
        if setting is not None:
            if user_type and user_type != setting.user_type:
                console.say(f"(이미 {USER_TYPES[setting.user_type]}(으)로 설정되어 있어 유형을 바꾸지 않습니다.)")
            return setting.user_type
        chosen = user_type
        for _ in range(MAX_ASKS if chosen is None else 0):
            answer = console.ask("회사원이신가요, 학생이신가요? (1: 회사원, 2: 학생) > ")
            if answer is None:
                break
            chosen = _TYPE_ANSWERS.get(answer.strip().lower())
            if chosen:
                break
            console.say("1(회사원) 또는 2(학생)로 답해 주세요.")
        if not chosen:
            console.say("사용자 유형을 정하지 못해 종료합니다. --user-type employee|student 로도 정할 수 있습니다.")
            return None
        store.init_user_setting(chosen)
        console.say(f"{USER_TYPES[chosen]}(으)로 저장했습니다. 다음 실행부터는 묻지 않습니다.")
        if settings.smtp_configured:  # 알림 메일은 처음 설정할 때 동의를 받는다(G11)
            answer = console.ask(f"마감 알림 메일을 {settings.alert_email_to}(으)로 받으시겠어요? (y/n) > ") or ""
            consent = answer.strip().lower() in {"y", "yes", "응", "네", "예"}
            store.set_alert_consent(consent)
            console.say("알림 메일 수신에 동의했습니다." if consent else "메일은 보내지 않고 .eml 파일로만 저장합니다.")
        return chosen


def upcoming_notice(settings: Settings, today: date) -> str | None:
    """시작할 때 보여 줄 임박한 마감 안내(LLM 없이 저장소에서 바로)."""
    from volunteer_agent.deadlines import get_upcoming_deadlines
    from volunteer_agent.store import Store

    with Store.open(settings) as store:
        result = get_upcoming_deadlines(store, today)
    if not result.items and not result.period_status:
        return None
    return result.message


# ---------------------------------------------------------------- 대화 실행

def with_attachment(text: str, path: str | Path | None) -> str:
    """첨부 파일 경로를 메시지에 붙인다(Agent는 이 경로를 file_path·certificate_file로 쓴다)."""
    if not path:
        return text
    return f"{text}\n(첨부 파일: {Path(path).resolve()})"


def trace_line(turn: Any) -> str:
    if turn.refusal:
        return f"  [trace] 입력 가드레일 차단: {turn.refusal} (Agent·도구 호출 없음)"
    agents = ", ".join(turn.agents) or "-"
    tools = ", ".join(turn.tools) or "-"
    return f"  [trace] Agent: {agents} | 도구: {tools}"


@dataclass
class Runtime:
    """앱과 가드레일을 만드는 방법. 테스트에서 가짜로 바꾼다."""

    open_app: Callable[..., Any]
    make_guard: Callable[[Settings], Any]


def default_runtime() -> Runtime:
    from volunteer_agent.app import open_app
    from volunteer_agent.guards import InputGuard
    from volunteer_agent.llm import has_llm_credentials, make_llm

    def make_guard(settings: Settings):
        return InputGuard(settings, make_llm(settings) if has_llm_credentials(settings) else None)

    def opener(settings: Settings, clock: Any, api_mode: str):
        return open_app(settings, clock, api_mode=api_mode)

    return Runtime(open_app=opener, make_guard=make_guard)


async def converse(settings: Settings, args: argparse.Namespace, console: Console, runtime: Runtime,
                   questions: list[str] | None) -> int:
    """questions가 있으면 그것만(--once), 없으면 표준 입력에서 한 줄씩(chat). 한 실행 = thread 하나."""
    from volunteer_agent.app import AppClock, ask
    from volunteer_agent.tracing import FileTracer, trace_path

    clock = AppClock.create(settings, args.today)
    thread_id = f"cli-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    tracer = FileTracer.for_settings(settings, thread_id)
    guard = runtime.make_guard(settings)
    secrets = settings.secret_values()

    async def answer(text: str) -> None:
        try:
            turn = await ask(app, text, thread_id, guard=guard, tracer=tracer)
        except Exception as exc:  # noqa: BLE001 - 대화는 이어 간다
            console.say(f"오류가 나서 답하지 못했습니다: {mask_exception(exc, secrets)}")
            return
        console.say(turn.answer or "(답변이 비어 있습니다. 다시 말씀해 주세요.)")
        if args.show_trace:
            console.say(trace_line(turn))

    async with runtime.open_app(settings, clock, args.api_mode) as app:
        if questions is not None:
            for q in questions:
                await answer(q)
            return 0
        console.say(f"봉사활동 도우미입니다. 기준 날짜 {clock.today().isoformat()}. "
                    f"'종료'로 끝내고, '{ATTACH_COMMAND} 파일경로 메시지'로 파일을 함께 보낼 수 있습니다.")
        notice = upcoming_notice(settings, clock.today())
        if notice:
            console.say(notice)
        while True:
            line = console.ask("\n나> ")
            if line is None or line.strip() in EXIT_WORDS:
                break
            if not line.strip():
                continue
            if line.startswith(ATTACH_COMMAND + " "):
                rest = line[len(ATTACH_COMMAND) + 1:].strip()
                path, _, message = rest.partition(" ")
                if not Path(path).is_file():
                    console.say(f"파일을 찾을 수 없습니다: {path}")
                    continue
                line = with_attachment(message or "이 파일 올릴게", path)
            await answer(line)
        console.say(f"대화를 마칩니다. 실행 기록: {trace_path(settings)}")
    return 0


# ---------------------------------------------------------------- export / purge / notify

def export_data(settings: Settings, output: Path | None = None, now: datetime | None = None) -> Path:
    """DATA_DIR 전체(DB·확인서·보고서·규정·기억)를 zip으로. DB는 SQLite 백업 API로 일관된 사본을 담는다(AC15)."""
    data_dir = settings.ensure_data_dir()
    stamp = (now or datetime.now(settings.tz)).strftime("%Y%m%d-%H%M%S")
    target = Path(output) if output else data_dir / EXPORT_DIRNAME / f"volunteer-export-{stamp}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    exports = (data_dir / EXPORT_DIRNAME).resolve()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(data_dir.rglob("*")):
            resolved = path.resolve()
            if not path.is_file() or resolved == target.resolve() or exports in resolved.parents:
                continue
            name = path.relative_to(data_dir).as_posix()
            if path.name.endswith(("-wal", "-shm", "-journal")):
                continue
            if path.suffix in {".db", ".sqlite", ".sqlite3"}:  # 쓰는 중이어도 일관된 사본
                copy = Path(tmp) / f"{uuid.uuid4().hex}.db"
                src, dst = sqlite3.connect(path), sqlite3.connect(copy)
                try:
                    src.backup(dst)
                finally:
                    dst.close()
                    src.close()
                zf.write(copy, name)
            else:
                zf.write(path, name)
    return target


def _unsafe_to_delete(path: Path) -> str | None:
    path = path.resolve()
    if path == Path(path.anchor) or path == Path.home().resolve() or path in Path.home().resolve().parents:
        return "드라이브 루트나 사용자 홈 폴더"
    if path == Path.cwd().resolve() or path in Path.cwd().resolve().parents:
        return "현재 작업 폴더(또는 그 상위)"
    return None


def purge_data(settings: Settings, console: Console, confirm: str | None) -> int:
    data_dir = settings.data_dir
    if not data_dir.exists():
        console.say(f"지울 데이터가 없습니다: {data_dir}")
        return 0
    reason = _unsafe_to_delete(data_dir)
    if reason:
        console.say(f"DATA_DIR이 {reason}라서 지우지 않습니다: {data_dir}")
        return 1
    console.say(f"{data_dir} 안의 모든 데이터(DB, 확인서, 보고서, 규정, 기억, 실행 기록)를 삭제합니다. 되돌릴 수 없습니다.")
    if confirm is None:
        confirm = console.ask(f"계속하려면 '{PURGE_PHRASE}'를 정확히 입력하세요 > ")
    if (confirm or "").strip() != PURGE_PHRASE:
        console.say("확인 문구가 맞지 않아 아무것도 삭제하지 않았습니다.")
        return 1
    shutil.rmtree(data_dir)
    console.say("모든 데이터를 삭제했습니다.")
    return 0


def notify(settings: Settings, today: date | None, console: Console) -> int:
    from volunteer_agent.notify import run_scheduler
    from volunteer_agent.store import Store

    day = today or datetime.now(settings.tz).date()
    with Store.open(settings) as store:
        result = run_scheduler(store, settings, day)
    console.say(result.message)
    return 1 if result.failed else 0


def quiet_library_noise() -> None:
    """대화 화면에 섞이는 라이브러리 경고(langchain-aws의 toolConfig 변환 안내)를 숨긴다."""
    import logging
    import warnings

    warnings.filterwarnings("ignore", message=".*without toolConfig.*")
    logging.getLogger("langchain_aws").setLevel(logging.ERROR)


# ---------------------------------------------------------------- 인자

def _common(parser: argparse.ArgumentParser, suppress: bool) -> None:
    d = {"default": argparse.SUPPRESS} if suppress else {}
    parser.add_argument("--user-type", choices=sorted(USER_TYPES), **d,
                        help="첫 실행의 회사원/학생 질문 대신 지정")
    parser.add_argument("--today", type=parse_today, **d, help="기준 날짜 YYYY-MM-DD (Asia/Seoul)")
    parser.add_argument("--show-trace", action="store_true", **d, help="답변마다 거친 Agent와 도구를 한 줄로 표시")
    parser.add_argument("--api-mode", choices=["fixture", "live"], **d, help="공공 API: 녹화 응답 또는 실제 호출")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="volunteer-agent", description="봉사활동 추천·관리 에이전트")
    parser.add_argument("--version", action="version", version=f"volunteer-agent {__version__}")
    _common(parser, suppress=False)
    parser.add_argument("--once", metavar="질문", help="질문 하나에 답하고 종료")
    parser.add_argument("--attach", metavar="파일", help="--once와 함께 보낼 파일(규정, 확인서)")
    sub = parser.add_subparsers(dest="command")
    chat = sub.add_parser("chat", help="대화형(실행 1회 = 대화 1개)")
    _common(chat, suppress=True)
    exp = sub.add_parser("export", help="DATA_DIR을 zip으로 내보내기")
    exp.add_argument("--output", type=Path, help="zip 경로(기본: DATA_DIR/exports/)")
    purge = sub.add_parser("purge", help="DATA_DIR 삭제(확인 문구 필요)")
    purge.add_argument("--confirm", metavar="문구", help=f"확인 문구('{PURGE_PHRASE}')")
    note = sub.add_parser("notify", help="마감 알림 점검 1회(작업 스케줄러용)")
    note.add_argument("--today", type=parse_today, default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None, *, stdin: TextIO | None = None, stdout: TextIO | None = None,
         settings: Settings | None = None, runtime: Runtime | None = None) -> int:
    if stdout is None:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    if stdin is None and hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    quiet_library_noise()
    console = Console(stdin or sys.stdin, stdout or sys.stdout)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.api_mode = getattr(args, "api_mode", None) or "live"
    try:
        settings = settings or load_settings()
    except Exception as exc:  # noqa: BLE001
        console.say(f"설정을 읽지 못했습니다: {mask_exception(exc)}")
        return 2

    if args.command == "export":
        console.say(f"내보냈습니다: {export_data(settings, args.output)}")
        return 0
    if args.command == "purge":
        return purge_data(settings, console, args.confirm)
    if args.command == "notify":
        return notify(settings, args.today, console)
    if args.command != "chat" and args.once is None:
        parser.print_help(console.stdout)
        return 0
    if args.attach and not Path(args.attach).is_file():
        console.say(f"첨부 파일을 찾을 수 없습니다: {args.attach}")
        return 2

    if ensure_user(settings, console, args.user_type) is None:
        return 2
    if runtime is None:
        from volunteer_agent.llm import has_llm_credentials

        if not has_llm_credentials(settings):
            console.say("Bedrock 설정(LLM 모델, AWS 키)이 .env에 없어 대화를 시작할 수 없습니다.")
            return 2
        runtime = default_runtime()
    questions = None if args.command == "chat" else [with_attachment(args.once, args.attach)]
    return asyncio.run(converse(settings, args, console, runtime, questions))


if __name__ == "__main__":
    sys.exit(main())
