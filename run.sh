#!/usr/bin/env bash
# 실행 스크립트. uv가 있으면 uv로, 없으면 pip(requirements.txt) + PYTHONPATH=src로 실행한다.
#   ./run.sh setup              의존성 설치 (.env가 없으면 .env.example을 복사)
#   ./run.sh chat               대화형 CLI (주 화면, '종료'로 끝)
#   ./run.sh once "질문"         질문 하나
#   ./run.sh web                Streamlit (http://localhost:8501)
#   ./run.sh test               단위 테스트
#   ./run.sh round N            인-아웃 세트 자체 평가 → evaluation/roundN_report.md
#   ./run.sh scenario [파일]     CLI 시나리오 평가 (기본: eval/scenarios/employee.yaml)
#   ./run.sh check              통합 점검
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run)
  PY=(uv run python)
else
  RUN=()
  PY=(python)
fi
cli() { if [ ${#RUN[@]} -gt 0 ]; then "${RUN[@]}" volunteer-agent "$@"; else python -m cli "$@"; fi; }

cmd="${1:-chat}"
shift || true
case "$cmd" in
  setup)
    if command -v uv >/dev/null 2>&1; then uv sync; else python -m pip install -r requirements.txt pytest aiosmtpd; fi
    [ -f .env ] || { cp .env.example .env; echo ".env를 만들었습니다. Bedrock·DATA_GO_KR_KEY 값을 채워 주세요."; }
    ;;
  chat)     cli chat --show-trace "$@" ;;
  once)     cli --once "$@" --show-trace ;;
  web)      "${PY[@]}" -m streamlit run app.py ;;
  test)     "${PY[@]}" -m pytest -q "$@" ;;
  round)    "${PY[@]}" evaluation/run_round.py --round "${1:?라운드 번호를 주세요 (예: ./run.sh round 2)}" "${@:2}" ;;
  scenario) cli scenario "${1:-eval/scenarios/employee.yaml}" ;;
  check)    cli check ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
