"""`volunteer-agent` 명령 진입점. 실제 명령(chat, --once, scenario 등)은 TASK 16~17단계에서 추가한다."""

from __future__ import annotations

import argparse
import sys

from volunteer_agent import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="volunteer-agent",
        description="봉사활동 추천·관리 에이전트",
    )
    parser.add_argument("--version", action="version", version=f"volunteer-agent {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
