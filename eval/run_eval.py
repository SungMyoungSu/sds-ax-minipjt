"""인-아웃 평가 13건 (`uv run python eval/run_eval.py`). 판정 로직은 inout_eval.

규칙이 확정된 회사원 상태(임시 DATA_DIR)에서 문항마다 새 대화로 묻고, 입력 가드레일을 앞단에 둔다.
결과는 eval/eval_result.json, 문항별 trace는 DATA_DIR/eval_runs/에 남긴다. 하나라도 실패하면 종료 코드 1, Bedrock 사용 한도로 중단되면 3.
"""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cli import default_runtime
from config import load_settings
from inout_eval import EVAL_DIR, eval_report, load_json, run_eval, validate_cases
from llm import has_llm_credentials


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    cases = load_json(EVAL_DIR / "eval_set.json")
    if problems := validate_cases(cases):
        print("평가셋 형식 오류:\n- " + "\n- ".join(problems))
        return 2
    settings = load_settings()
    if not has_llm_credentials(settings):
        print("Bedrock 설정(LLM 모델, AWS 키)이 .env에 없어 평가를 실행할 수 없습니다.")
        return 2
    results, trace_copy = asyncio.run(run_eval(cases, settings, default_runtime()))
    report, code = eval_report(results, trace_copy)
    print(report)
    out = Path(__file__).with_name("eval_result.json")
    out.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    if any(r.status == "aborted" for r in results):
        return 3  # Bedrock 사용 한도 초과로 끝까지 돌리지 못함
    return code


if __name__ == "__main__":
    sys.exit(main())
