"""인-아웃 세트 자체 평가 한 라운드 (`uv run python evaluation/run_round.py --round 1`).

test_queries.csv를 문항마다 새 데이터·새 대화로 실행하고 LLM 심사와 RAGAS 지표를 계산해
evaluation/round{N}_report.md와 round{N}_results.json으로 저장한다. 이전 라운드 결과가 있으면 비교한다.
판정·지표 정의는 volunteer_agent.round_eval 참고.

    --round N           라운드 번호(필수)
    --changes 파일.md    이번 라운드 전에 바꾼 내용(리포트에 그대로 넣음)
    --only 3,5,7        일부 문항만(개선 확인용, 리포트 파일 이름에 _partial이 붙음)
종료 코드: 모두 통과 0, 실패 있음 1, 설정·형식 오류 2, Bedrock 사용 한도로 중단 3
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from volunteer_agent.cli import default_runtime  # noqa: E402
from volunteer_agent.config import load_settings  # noqa: E402
from volunteer_agent.llm import has_llm_credentials, make_llm  # noqa: E402
from volunteer_agent.rag import make_embeddings  # noqa: E402
from volunteer_agent.round_eval import load_cases, round_meta, run_round, save_round  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="인-아웃 세트 라운드 평가")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--changes", type=Path, help="이번 라운드 전에 바꾼 내용(Markdown)")
    parser.add_argument("--only", help="일부 문항 id(쉼표로 구분)")
    parser.add_argument("--cases", type=Path, default=ROOT / "test_queries.csv")
    args = parser.parse_args()

    try:
        cases = load_cases(args.cases)
    except ValueError as exc:
        print(exc)
        return 2
    if args.only:
        wanted = {int(x) for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c.id in wanted]
    settings = load_settings()
    if not has_llm_credentials(settings):
        print("Bedrock 설정(LLM 모델, AWS 키)이 .env에 없어 평가를 실행할 수 없습니다.")
        return 2
    judge = make_llm(settings, settings.judge_model or settings.llm_model)
    results = asyncio.run(run_round(cases, settings, default_runtime(), judge, make_embeddings(settings)))
    changes = args.changes.read_text(encoding="utf-8") if args.changes else ""
    folder = ROOT / "evaluation"
    if args.only:
        folder = folder / "partial"
    report = save_round(folder, args.round, results, round_meta(settings), changes)
    print(f"\n리포트: {report}")
    if any(r.status == "aborted" for r in results):
        return 3
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
