"""LLM 심사 6문항 (`uv run python eval/llm_judge.py`). 고정 루브릭, 심사 모델(JUDGE_MODEL)과 temperature 0 고정.

문항마다 정해진 출발 상태(seed)의 임시 DATA_DIR에서 실제 앱으로 답을 만들고 심사 모델이 1~5점으로 채점한다.
평균 4점 이상이면 종료 코드 0.
"""

import asyncio
import sys

from volunteer_agent.cli import default_runtime
from volunteer_agent.config import load_settings
from volunteer_agent.evaluation import EVAL_DIR, judge_report, load_json, run_judge, validate_judge_set
from volunteer_agent.llm import has_llm_credentials, make_llm


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    items = load_json(EVAL_DIR / "judge_set.json")
    if problems := validate_judge_set(items):
        print("심사 세트 형식 오류:\n- " + "\n- ".join(problems))
        return 2
    settings = load_settings()
    if not has_llm_credentials(settings):
        print("Bedrock 설정(LLM 모델, AWS 키)이 .env에 없어 심사를 실행할 수 없습니다.")
        return 2
    judge_llm = make_llm(settings, settings.judge_model or settings.llm_model)
    print(f"심사 모델: {settings.judge_model or settings.llm_model} (temperature 0)")
    rows = asyncio.run(run_judge(items, settings, default_runtime(), judge_llm))
    report, code = judge_report(rows)
    print(report)
    return code


if __name__ == "__main__":
    sys.exit(main())
