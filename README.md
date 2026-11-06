# 미니 PJT: 봉사활동 추천·관리 에이전트 (volunteer-agent)

## 무엇을 푸나
회사원(사내 봉사 의무 시간)과 학생(학기 봉사 목표)이 **규정 확인 → 1365 봉사 찾기 → 활동·확인서 기록 → 남은 시간·마감 관리 → 보고서 제출**을 여러 사이트와 엑셀을 오가며 손으로 하던 일을, 대화 한 곳에서 근거(규정 원문·1365 링크)와 함께 처리하게 한다.

## 활용한 패턴 (Day 1~7)
| Day | 패턴 | 이 프로젝트에서 쓰는 곳 |
|---|---|---|
| Day 1 | LLM 호출과 구조화 출력 | `ChatBedrockConverse`(temperature 0) + Pydantic `with_structured_output`: 가드레일 판별(`InjectionCheck`), 계획(`Plan`·`Replan`), 규칙 후보(`RuleDraft`) |
| Day 2 | RAG와 평가셋 | 사내규정을 조항 단위로 나눠 **BM25 + Titan 임베딩 + Chroma 하이브리드(RRF)** 검색, 근거 없으면 고정 문구. 평가셋의 fact/paraphrase/no_answer 문항 |
| Day 3 | 도구 호출 Agent(ReAct) | 하위 Agent 5개를 `create_react_agent`로, 각자 자기 도구만 바인딩 |
| Day 4 | MCP 서버 | fastmcp stdio 서버에 업무 도구 18개(결정적 코드, 채팅 LLM 호출 없음). 모든 도구 docstring에 "쓰는 경우/쓰지 않는 경우" |
| Day 5 | 입력 가드레일·고정 거절 | 1차 정규식 → 2차 LLM 판별(확신도 > 0.7). 거절 5범주 고정 문구, 섞인 요청 전체 거절, `guard_audit.log` |
| Day 6 | Supervisor 다중 에이전트 | `create_supervisor` + policy/record/report/recommend/planner/general Agent, checkpointer로 대화 이어받기 |
| Day 7 | Plan-and-Execute·장기 기억·추적·평가 | planner_agent(planner → execute → replan → finalize, 승인 전 저장 안 함), LangGraph Store 선호 기억, `trace.jsonl`(FileTracer), LLM 심사·인-아웃 평가·통합 점검 |

## 아키텍처
```
 사용자 ── CLI(chat / --once / scenario / check)  ┐
        └ Streamlit(localhost, 채팅·PDF 다운로드) ┘ 같은 build_app() = agent.open_app → agent.compile_app
                         │
            [입력 가드레일] 정규식 → LLM 판별 ── 차단 → 고정 거절 문구 1개 + guard_audit.log
                         │ 통과
   Supervisor (Bedrock Claude, checkpointer=대화 기억, 턴 시작엔 반드시 담당 Agent로 넘김)
     ├ policy_agent    사내규정 등록·검색(하이브리드 RAG)·규칙 후보·확정
     ├ record_agent    프로필·활동 기록·수정·삭제·확인서·남은 시간
     ├ report_agent    마감 일정·캘린더(.ics)·회사 보고서·학생 포트폴리오 PDF
     ├ recommend_agent 1365 공공 API 추천(시도·시군구 필터)·관심 목록·선호 기억(Store)
     ├ planner_agent   Plan-and-Execute 봉사 계획(승인 시에만 저장)
     └ general_agent   인사·기능 안내
                         │ stdio
        로컬 MCP 서버(fastmcp) → SQLite · Chroma · DATA_DIR(확인서·PDF·.ics) · data.go.kr · SMTP(.eml)
   모든 LLM·도구 호출 → DATA_DIR/trace.jsonl (비밀 값 마스킹)
```
- 답변: Agent 하나면 그 Agent의 답, 여럿이면 각 Agent의 답을 순서대로 잇는다(Supervisor 요약은 쓰지 않음).
- 외부 통신은 Bedrock, `apis.data.go.kr`, 설정된 SMTP만 허용(MCP 서버의 네트워크 허용 목록).
- 자세한 설계: [`SERVICE.md`](SERVICE.md), 요구사항: [`PRD.md`](PRD.md), 사용 흐름: [`SCENARIO.md`](SCENARIO.md)

## 실행 방법
```bash
uv sync                                    # Python 3.13, uv (uv가 없으면: pip install -r requirements.txt, PYTHONPATH=src)
./run.sh setup | chat | web | test | round N   # 같은 일을 하는 실행 스크립트
cp .env.example .env                       # Bedrock(LLM_MODEL, AWS 키, EMBEDDING_MODEL), DATA_GO_KR_KEY 입력

uv run volunteer-agent chat --show-trace   # 대화형 (처음 실행 시 회사원/학생 선택, '종료'로 끝)
#   나> /첨부 data/policies/사회공헌_봉사활동_규정.md 회사 봉사 규정 파일 올릴게
#   나> 응, 확정해
#   나> 동작구 주말 봉사 찾아줘
uv run volunteer-agent --once "어떤 일을 도와줄 수 있어?"
uv run streamlit run app.py                # 로컬 화면 (http://localhost:8501)

uv run pytest                              # 단위 테스트 (Bedrock·라이브 API 테스트는 키가 없으면 skip)
uv run volunteer-agent scenario eval/scenarios/   # CLI 시나리오 평가
uv run python evaluation/run_round.py --round 1   # 인-아웃 세트 자체 평가 (LLM 심사 + RAGAS) → evaluation/round1_report.md
uv run volunteer-agent check               # 통합 점검 (복합 질문 + 주입 + 정상 질문)
uv run volunteer-agent notify              # 마감 알림 점검 1회 (Windows 작업 스케줄러용)
uv run volunteer-agent export | purge      # 데이터 백업 / 삭제(확인 문구 필요)
```
- 공통 옵션: `--today YYYY-MM-DD`(기준 날짜), `--api-mode fixture|live`(녹화 응답/실제 1365 API), `--user-type employee|student`
- 기본 사내규정: 파일 없이 "회사 규정 등록해줘"라고 하면 `data/policies/사회공헌_봉사활동_규정.md`를 등록한다.

## RAGAS 평가 결과
인-아웃 세트(`evaluation/test_queries.csv`) 중 규정 검색(`search_policy`)을 쓰는 문항으로 산출한다. 라운드별 상세는 [`evaluation/`](evaluation/)의 리포트를 본다.
- context_recall: 아래 라운드 리포트 참고 (측정 후 갱신)
- context_precision: 〃
- faithfulness: 〃
- answer_relevancy: 〃

## 인-아웃 세트 통과율 (자체 평가)
`evaluation/test_queries.csv` 24건(positive 11 · negative 4 · edge 5 · guardrail 4). 판정: 기대 도구 호출 + LLM 심사(expected_traits 충족, forbidden 없음) + guardrail은 고정 문구 완전 일치.
- 1차: [`evaluation/round1_report.md`](evaluation/round1_report.md) (측정 후 갱신)
- 2차: [`evaluation/round2_report.md`](evaluation/round2_report.md) (개선 후 측정)
- 개선폭: 측정 후 갱신

참고(CLI 시나리오 평가, `eval/scenarios/`, 실제 Bedrock): 첫 전체 실행 37/58 → 개선 후 53/58. employee 시나리오는 이후 수정으로 17/26 → 24/26.

## 트라이앤에러 회고
**시도했지만 실패한 접근**
- **프롬프트 메모로 Supervisor 라우팅 강제:** "직접 답하지 말고 넘겨라" 메모를 사용자 메시지 형태로 붙였더니, Supervisor가 그 메모를 새 요청으로 받아 오히려 직접 답하고 메모 내용까지 답에 섞었다(employee 24/26 → 17/26).
- **Supervisor가 마지막에 결과를 요약:** 여러 Agent를 거친 턴에서 Supervisor가 실제 추천 목록·링크 대신 "모든 처리가 완료되었습니다"로 덮어쓰거나, 없는 봉사 목록과 링크를 지어냈다.
- **하위 Agent에 전체 대화를 그대로 전달:** Supervisor가 넣는 `transfer_back_to_supervisor` 메시지를 하위 Agent가 흉내 내 없는 도구를 18번 반복 호출하다 호출 한도(40)에 걸렸다.
- **"필요하면 get_progress를 불러라" 같은 조건부 지시:** 기록 결과에 남은 시간이 이미 보이면 모델이 도구 호출을 건너뛰었다.
- **지역을 시도 단위로만 필터:** "동작구"를 물어도 서울 전역이 나왔다.

**최종 채택한 접근**
- 프롬프트가 아니라 **코드로 강제**: 턴 시작에 Supervisor 모델을 `tool_choice="any"`로 한 번만 불러 이번 요청의 Agent 순서(`route_plan`)를 받고, 이후 Agent 이동과 종료는 LLM 없이 코드가 처리(`RouteFirstChat`, 재배분·마무리 호출 제거로 LLM 시간의 약 20% 절감 목표), 답변은 Agent 답을 그대로 잇기(`agents_answer`), 하위 Agent가 보는 대화에서 이동 메시지와 Supervisor 발화 제거(`agent_view`).
- **도구가 역할을 나눠 갖게 설계**: 기록 도구는 상태만, 남은 시간은 `get_progress`만, 확인서는 `attach_certificate`만 돌려준다.
- 1365 API의 `schSign1`(시군구)을 실제 호출로 확인하고 서울 25개 구 코드를 검증해 코드표로 둠.
- 평가 실행기가 Bedrock 사용 한도(ThrottlingException)를 만나면 남은 문항을 돌리지 않고 종료 코드 3으로 멈춤.

**남은 한계 · 향후 개선**
- **속도:** 질문 한 번에 LLM을 4~6번 차례로 부른다(측정 예: 13.6초). 가드레일 판별을 가벼운 모델(Haiku)로, Bedrock 프롬프트 캐싱을 검토 중. 도구가 10개로 몰렸던 record_agent는 record_agent(7개)와 report_agent(4개)로 나눴다.
- 규정의 "1일 최대 8시간"은 답변으로만 안내하고 기록 시 자동으로 자르지 않는다.
- 시군구 코드표는 서울만 검증. 다른 시도는 장소·기관 이름으로 거른다.
- 학생 시나리오는 최근 수정 후 재확인하지 않았다(마지막 11/13).

## 핵심 코드 위치
- `src/agent.py:176` — 메인 에이전트 그래프 `compile_app`(Supervisor + 하위 Agent 조립), `:104` `RouteFirstChat`(배분 계획), `:258` `ask`(가드레일 → Supervisor → 답 조립)
- `src/agents.py:19` — Agent별 도구 경계(`AGENT_TOOLS`), `:180` Supervisor 배분 기준, `:255` `build_agents`
- `src/tools.py:56` — 도메인 도구 18개(fastmcp MCP 서버)
- `src/retriever.py:68` — RAG 파이프라인(BM25 + Titan + Chroma 하이브리드), `src/policy.py:77` 조항 분할
- `src/planner.py:179` — Plan-and-Execute 그래프(planner_agent)
- `src/guards.py:130` — 입력 가드레일(정규식 → LLM 판별)
- `src/recommend.py:465` — 1365 추천(`:355` 시도·시군구 해석)
- `src/memory.py:83` — 장기 기억(선호) 도구, `src/tracing.py:94` 실행 추적
- `src/round_eval.py`, `evaluation/run_round.py` — 인-아웃 세트 라운드 평가(LLM 심사 + RAGAS 지표), `src/scenario.py`, `src/inout_eval.py` — 시나리오·인-아웃 13건·통합 점검
