[Day7] 공통 checkpoint

7일차에 실행하는 완성 코드가 모두 들어 있습니다.
day7_practice 폴더에 풀어 두고 씁니다.

담긴 것
  plan_and_execute.py    1장 Plan-and-Execute 완성본
  long_term_memory.py    2장 장기 기억(Store) 완성본
  local_tracer.py        3장 실행 기록 콜백 핸들러
  trace_run.py           3장 6일차 그래프에 추적 붙이기
  llm_judge.py           4장 LLM 심사
  run_eval.py            4장 누적 평가셋 실행
  integration_check.py   6장 통합 점검
  final_scenario.py      6일차 Supervisor 종합 시나리오 (RAG + MCP)
  mcp_server.py          4일차 MCP 서버
  make_data.py           임직원·자산·프로젝트 JSON 3종 생성기
  guards_input.py        5일차 입력 가드레일 (run_eval, integration_check 가 import)
  guards_refusal.py      5일차 고정 거절 문구 (위와 같은 이유)
  eval_set.json          2~6일차 누적 평가셋 14문항 (전날 미션을 못 따라온 경우에만 사용)

시작 전에 할 것
  1) python make_data.py 로 JSON 3종 생성
  2) 2일차 chroma_db 폴더를 day6_practice 에서 복사해 옵니다 (final_scenario 가 씁니다)
  3) 6일차 가상환경을 이어서 활성화합니다
       ..\day6_practice\.venv\Scripts\activate
  4) .env 는 레포 루트의 것을 씁니다

주의
  guards_input.py 와 guards_refusal.py 는 5일차에 직접 만든 본인 파일이 있으면
  그것으로 덮어써서 쓰세요. 여기 것은 5일차를 못 따라온 경우의 대체본입니다.
