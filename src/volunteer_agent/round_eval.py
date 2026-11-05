"""인-아웃 세트(test_queries.csv) 라운드 평가: 실행 → LLM 심사 → RAGAS 지표 → 라운드 리포트.

판정(문항별 PASS):
- guardrail: 답이 고정 거절 문구 하나와 완전히 일치하고 도구 호출이 없을 것(결정적, LLM 심사 안 함)
- 그 밖: expected_tools가 모두 호출되고, LLM 심사에서 expected_traits를 모두 충족하고 forbidden이 하나도 없을 것

RAGAS 지표(규정 검색 search_policy를 쓴 문항만): RAGAS와 같은 정의로 직접 계산한다.
ragas 라이브러리는 openai·tiktoken 등을 끌어오고 tiktoken이 외부에서 인코딩 파일을 내려받아, 외부 통신을
Bedrock·data.go.kr·SMTP로 제한한 이 프로젝트 정책(CLAUDE.md)에 맞지 않기 때문이다. 심사 LLM은 JUDGE_MODEL(Bedrock),
answer_relevancy의 임베딩은 Titan(EMBEDDING_MODEL)을 쓴다.
- faithfulness: 답변을 단위 주장으로 나눠, 검색 근거로 뒷받침되는 주장의 비율
- answer_relevancy: 답변에서 거꾸로 만든 질문 3개와 원래 질문의 임베딩 코사인 유사도 평균(얼버무린 답이면 0)
- context_precision: 검색 근거(순위대로)가 기대 답에 쓸모 있는지 판정해 계산한 평균 정밀도(AP@k)
- context_recall: 기대 답(expected_traits)의 항목 중 검색 근거에서 찾을 수 있는 비율
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from volunteer_agent.config import Settings
from volunteer_agent.guards import REFUSALS
from volunteer_agent.llm import is_quota_error
from volunteer_agent.masking import mask_exception, mask_secrets

CATEGORIES = ("positive", "negative", "edge", "guardrail")
COLUMNS = ["id", "category", "input", "expected_traits", "forbidden", "expected_tools", "note"]
NO_POLICY_HINT = "규정을 아직 등록하지 않은"  # note에 이 말이 있으면 규정 없는 상태에서 시작
RAG_TOOL = "search_policy"
ROUND_TODAY = date(2026, 11, 4)


# ---------------------------------------------------------------- 인-아웃 세트

@dataclass
class Case:
    id: int
    category: str
    input: str
    traits: list[str]
    forbidden: list[str]
    tools: list[str]
    note: str = ""

    @property
    def seed(self) -> str:
        return "employee_new" if NO_POLICY_HINT in self.note else "default_confirmed"


def _split(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def load_cases(path: Path) -> list[Case]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"컬럼은 {COLUMNS} 순서여야 합니다(실제: {reader.fieldnames}).")
        cases = [Case(int(r["id"]), r["category"].strip(), r["input"].strip(), _split(r["expected_traits"]),
                      _split(r["forbidden"]), _split(r["expected_tools"]), (r["note"] or "").strip())
                 for r in reader]
    problems = validate_cases(cases)
    if problems:
        raise ValueError("test_queries.csv 형식 오류:\n- " + "\n- ".join(problems))
    return cases


def validate_cases(cases: list[Case]) -> list[str]:
    problems = []
    if len(cases) < 10:
        problems.append(f"최소 10건이어야 합니다({len(cases)}건).")
    if [c.id for c in cases] != list(range(1, len(cases) + 1)):
        problems.append("id는 1부터 차례로 매겨야 합니다.")
    for c in cases:
        if c.category not in CATEGORIES:
            problems.append(f"{c.id}번: category는 {', '.join(CATEGORIES)} 중 하나입니다({c.category}).")
        if not c.input or not c.traits:
            problems.append(f"{c.id}번: input과 expected_traits는 필수입니다.")
        if c.category == "guardrail" and not any(p in t for t in c.traits for p in REFUSALS.values()):
            problems.append(f"{c.id}번: guardrail 문항의 expected_traits에는 고정 거절 문구가 있어야 합니다.")
    return problems


# ---------------------------------------------------------------- 실행 중 검색 근거 모으기

class ContextCollector(BaseCallbackHandler):
    """search_policy 도구 출력(검색 근거 원문)을 자르지 않고 모은다(trace는 200자로 자르므로 따로 받는다)."""

    def __init__(self) -> None:
        self._names: dict[Any, str] = {}
        self.outputs: list[str] = []

    def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self._names[run_id] = (serialized or {}).get("name") or kw.get("name") or ""

    def on_tool_end(self, output, *, run_id, **kw):
        if self._names.pop(run_id, "") == RAG_TOOL:
            content = getattr(output, "content", output)
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            self.outputs.append(str(content))


def split_contexts(outputs: list[str]) -> list[str]:
    """search_policy 출력 → 조항별 근거 목록(순위 순서). '[출처: …]'로 시작하는 블록 단위."""
    contexts: list[str] = []
    for out in outputs:
        if "[출처:" not in out:
            continue  # 근거 없음·규정 없음 안내
        contexts += ["[출처: " + part.strip() for part in out.split("[출처:")[1:] if part.strip()]
    return contexts


# ---------------------------------------------------------------- LLM 심사

class TraitCheck(BaseModel):
    item: str
    ok: bool = Field(description="expected_traits면 충족 여부, forbidden이면 답변에 그 내용이 '없으면' true")
    reason: str


class CaseJudgement(BaseModel):
    traits: list[TraitCheck]
    forbidden: list[TraitCheck] = Field(default_factory=list)
    score: int = Field(ge=1, le=5)
    reasoning: str


JUDGE_PROMPT = """너는 봉사활동 추천·관리 에이전트의 답변을 채점하는 심사자다. 정확성만 본다. 길이·말투는 보지 않는다.
각 expected_traits 항목이 답변에 충족됐는지(ok), 각 forbidden 항목이 답변에 나오지 않았는지(나오지 않았으면 ok=true) 판정하라.
항목의 뜻이 충족되면 표현이 달라도 충족이다. 숫자·날짜·고정 문구는 정확해야 한다.
score: 5 = 모든 항목 충족·금지 없음, 3 = 일부 빠짐, 1 = 질문에 답하지 않거나 사실이 틀림.

질문: {input}
expected_traits: {traits}
forbidden: {forbidden}
메모: {note}
에이전트 답변:
<answer>
{answer}
</answer>"""


# ---------------------------------------------------------------- RAGAS 지표 (같은 정의, 자체 구현)

class Claims(BaseModel):
    claims: list[str] = Field(description="답변을 하나의 사실만 담은 짧은 문장들로 나눈 것")


class Verdicts(BaseModel):
    verdicts: list[bool] = Field(description="입력 항목과 같은 순서로, 근거(context)로 뒷받침되면 true")


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="답변만 보고 거꾸로 만든, 그 답변이 답이 될 질문 3개")
    noncommittal: bool = Field(description="답변이 '모르겠다·확인되지 않음'처럼 얼버무린 답이면 true")


async def _structured(llm: Any, schema: type[BaseModel], prompt: str) -> Any:
    return await llm.with_structured_output(schema).ainvoke(prompt)


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


async def faithfulness(llm: Any, answer: str, contexts: list[str]) -> float | None:
    claims = (await _structured(llm, Claims, f"다음 답변을 사실 하나씩 담은 짧은 문장들로 나눠라. 인사·안내 문장은 뺀다.\n답변:\n{answer}")).claims
    if not claims:
        return None
    joined = "\n\n".join(contexts)
    v = await _structured(llm, Verdicts, "각 주장이 아래 근거만으로 뒷받침되는지 순서대로 true/false로 판정하라.\n"
                          f"근거:\n{joined}\n\n주장:\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims)))
    verdicts = (v.verdicts + [False] * len(claims))[:len(claims)]
    return sum(verdicts) / len(claims)


async def answer_relevancy(llm: Any, embeddings: Any, question: str, answer: str) -> float | None:
    if embeddings is None:
        return None
    g = await _structured(llm, GeneratedQuestions, f"다음 답변이 답이 될 만한 질문 3개를 만들어라.\n답변:\n{answer}")
    if g.noncommittal or not g.questions:
        return 0.0
    q = embeddings.embed_query(question)
    sims = [_cosine(q, embeddings.embed_query(x)) for x in g.questions]
    return sum(sims) / len(sims)


async def context_precision(llm: Any, question: str, reference: str, contexts: list[str]) -> float | None:
    if not contexts:
        return None
    v = await _structured(llm, Verdicts, "각 근거가 아래 질문의 기대 답을 만드는 데 쓸모 있는지 순서대로 true/false로 판정하라.\n"
                          f"질문: {question}\n기대 답: {reference}\n\n근거:\n"
                          + "\n\n".join(f"{i + 1}. {c}" for i, c in enumerate(contexts)))
    rel = (v.verdicts + [False] * len(contexts))[:len(contexts)]
    hits, total = 0, 0.0
    for k, r in enumerate(rel, start=1):
        if r:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


async def context_recall(llm: Any, reference_items: list[str], contexts: list[str]) -> float | None:
    if not contexts or not reference_items:
        return None
    joined = "\n\n".join(contexts)
    v = await _structured(llm, Verdicts, "기대 답의 각 항목이 아래 근거에서 찾아지는지(근거로 뒷받침되는지) 순서대로 true/false로 판정하라. "
                          "출처 표기·형식에 관한 항목도 근거에 그 조항이 있으면 true.\n"
                          f"근거:\n{joined}\n\n기대 답 항목:\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reference_items)))
    rel = (v.verdicts + [False] * len(reference_items))[:len(reference_items)]
    return sum(rel) / len(reference_items)


# ---------------------------------------------------------------- 한 문항

@dataclass
class CaseResult:
    id: int
    category: str
    input: str
    passed: bool
    reasons: list[str]
    answer: str = ""
    tools: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    score: int | None = None
    trait_checks: list[dict] = field(default_factory=list)
    ragas: dict[str, float | None] = field(default_factory=dict)
    contexts: int = 0
    latency_s: float = 0.0
    status: str = "ok"  # ok | aborted


async def evaluate_case(case: Case, base: Settings, runtime: Any, judge_llm: Any, embeddings: Any,
                        today: date = ROUND_TODAY) -> CaseResult:
    from volunteer_agent.evaluation import Session

    collector = ContextCollector()
    async with Session(base, runtime, case.seed, today, f"round-case{case.id}") as session:
        asked = await session.ask(case.input, mark={"round_case": case.id}, callbacks=[collector])
    answer = asked.answer
    tools = list(dict.fromkeys(asked.tools))
    result = CaseResult(case.id, case.category, case.input, False, [], mask_secrets(answer, base.secret_values()),
                        tools, asked.agents, latency_s=round(asked.latency_s, 1))

    reasons: list[str] = []
    if missing := [t for t in case.tools if t not in tools]:
        reasons.append(f"호출하지 않은 도구: {missing} (실제: {tools or '없음'})")
    if case.category == "guardrail":
        if tools:
            reasons.append(f"거절 문항인데 도구를 호출함: {tools}")
        if answer.strip() not in REFUSALS.values() or not any(answer.strip() in t for t in case.traits):
            reasons.append("기대한 고정 거절 문구와 완전히 일치하지 않음")
        result.score = 5 if not reasons else 1
    else:
        j: CaseJudgement = await _structured(judge_llm, CaseJudgement, JUDGE_PROMPT.format(
            input=case.input, traits=case.traits, forbidden=case.forbidden or "없음", note=case.note or "-",
            answer=answer))
        result.score = j.score
        result.trait_checks = [c.model_dump() for c in j.traits] + [dict(c.model_dump(), forbidden=True) for c in j.forbidden]
        if miss := [c.item for c in j.traits if not c.ok]:
            reasons.append(f"충족하지 못한 기대 항목: {miss}")
        if bad := [c.item for c in j.forbidden if not c.ok]:
            reasons.append(f"나오면 안 되는 내용이 나옴: {bad}")
        if RAG_TOOL in tools:
            contexts = split_contexts(collector.outputs)
            result.contexts = len(contexts)
            reference = "; ".join(case.traits)
            result.ragas = {
                "context_recall": await context_recall(judge_llm, case.traits, contexts),
                "context_precision": await context_precision(judge_llm, case.input, reference, contexts),
                "faithfulness": await faithfulness(judge_llm, answer, contexts) if contexts else None,
                "answer_relevancy": await answer_relevancy(judge_llm, embeddings, case.input, answer),
            }
    result.reasons = reasons
    result.passed = not reasons
    return result


async def run_round(cases: list[Case], base: Settings, runtime: Any, judge_llm: Any, embeddings: Any,
                    say=print) -> list[CaseResult]:
    results: list[CaseResult] = []
    for i, case in enumerate(cases):
        try:
            r = await evaluate_case(case, base, runtime, judge_llm, embeddings)
        except Exception as exc:  # noqa: BLE001
            if is_quota_error(exc):
                results += [CaseResult(c.id, c.category, c.input, False, ["실행하지 않음: Bedrock 사용 한도 초과"],
                                       status="aborted") for c in cases[i:]]
                say(f"[중단] {case.id}번부터 Bedrock 사용 한도 초과로 실행하지 않았습니다.")
                break
            r = CaseResult(case.id, case.category, case.input, False,
                           [f"실행 오류: {mask_exception(exc, base.secret_values())}"])
        results.append(r)
        say(f"[{'PASS' if r.passed else 'FAIL'}] {r.id} ({r.category}) {r.input} ({r.latency_s}s)"
            + ("" if r.passed else " - " + "; ".join(r.reasons)))
    return results


# ---------------------------------------------------------------- 리포트

RAGAS_KEYS = ("context_recall", "context_precision", "faithfulness", "answer_relevancy")


def ragas_means(results: list[CaseResult]) -> dict[str, tuple[float | None, int]]:
    out = {}
    for k in RAGAS_KEYS:
        vals = [r.ragas[k] for r in results if r.ragas.get(k) is not None]
        out[k] = (sum(vals) / len(vals) if vals else None, len(vals))
    return out


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def render_report(round_no: int, results: list[CaseResult], meta: dict[str, Any],
                  previous: list[dict] | None = None, changes: str = "") -> str:
    scored = [r for r in results if r.status == "ok"]
    passed = sum(r.passed for r in scored)
    lines = [f"# 인-아웃 세트 자체 평가 · Round {round_no}", "",
             f"- 실행: {meta.get('when')} · 커밋 `{meta.get('commit') or '-'}`" + (" (커밋 안 된 변경 포함)" if meta.get("dirty") else ""),
             f"- 평가 대상: `test_queries.csv` {len(results)}건 · 에이전트 모델 `{meta.get('llm_model')}` · 심사 모델 `{meta.get('judge_model')}` (temperature 0)",
             f"- 출발 상태: 회사원·기본 규정(`data/policies/사회공헌_봉사활동_규정.md`) 등록·규칙 확정(목표 16시간, 1/1~12/31, 분야 제한 없음, 확인서 필수)·기록 없음·기준 날짜 {ROUND_TODAY}·1365 녹화 응답. 문항마다 새 데이터와 새 대화",
             "- 판정: guardrail은 고정 거절 문구 완전 일치 + 도구 호출 없음. 그 밖은 expected_tools 호출 + LLM 심사(expected_traits 모두 충족, forbidden 없음)",
             "", "## 요약", "",
             f"**통과 {passed} / {len(scored)}**" + (f" (실행하지 않음 {len(results) - len(scored)}건: Bedrock 사용 한도 초과)" if len(scored) < len(results) else ""),
             "", "| 카테고리 | 통과 | 심사 점수 평균 |", "|---|---|---|"]
    for cat in CATEGORIES:
        rs = [r for r in scored if r.category == cat]
        if rs:
            sc = [r.score for r in rs if r.score is not None]
            lines.append(f"| {cat} | {sum(r.passed for r in rs)} / {len(rs)} | {_fmt(sum(sc) / len(sc) if sc else None)} |")
    means = ragas_means(scored)
    lines += ["", "## RAGAS 지표 (규정 검색 문항)", "",
              "RAGAS와 같은 정의로 직접 계산(ragas 라이브러리는 외부 다운로드가 필요해 프로젝트 통신 정책상 쓰지 않음). 심사 모델과 Titan 임베딩 사용.", "",
              "| 지표 | 값 | 문항 수 |", "|---|---|---|"]
    lines += [f"| {k} | {_fmt(v)} | {n} |" for k, (v, n) in means.items()]
    if previous:
        prev = {p["id"]: p for p in previous}
        prev_pass = sum(1 for p in previous if p.get("passed"))
        fixed = [r.id for r in scored if r.passed and not prev.get(r.id, {}).get("passed")]
        broke = [r.id for r in scored if not r.passed and prev.get(r.id, {}).get("passed")]
        lines += ["", "## 이전 라운드 대비", "",
                  f"- 통과: {prev_pass} → {passed} ({passed - prev_pass:+d}건)",
                  f"- 새로 통과: {fixed or '없음'}", f"- 새로 실패: {broke or '없음'}"]
    if changes:
        lines += ["", "## 이번 라운드 전에 바꾼 것", "", changes.strip()]
    lines += ["", "## 문항별 결과", "", "| id | 카테고리 | 질문 | 결과 | 점수 | 호출 도구 | 지연(s) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        mark = "실행 안 함" if r.status != "ok" else ("PASS" if r.passed else "**FAIL**")
        lines.append(f"| {r.id} | {r.category} | {r.input} | {mark} | {r.score or '-'} | {', '.join(r.tools) or '-'} | {r.latency_s} |")
    fails = [r for r in scored if not r.passed]
    if fails:
        lines += ["", "## 실패 원인", ""]
        for r in fails:
            lines += [f"### {r.id}. {r.input}", f"- 이유: {'; '.join(r.reasons)}",
                      f"- 거친 Agent: {', '.join(r.agents) or '-'}",
                      "- 답변: " + " ".join(r.answer.split())[:400], ""]
    lines += ["", "## 다음 개선 후보", "", "- (리포트를 보고 원인별로 정리한 뒤 다음 라운드 전에 반영)"]
    return "\n".join(lines) + "\n"


def round_meta(settings: Settings) -> dict[str, Any]:
    return {"when": datetime.now().strftime("%Y-%m-%d %H:%M"), "commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")), "llm_model": settings.llm_model,
            "judge_model": settings.judge_model or settings.llm_model}


def save_round(folder: Path, round_no: int, results: list[CaseResult], meta: dict[str, Any],
               changes: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    prev_path = folder / f"round{round_no - 1}_results.json"
    previous = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.is_file() else None
    (folder / f"round{round_no}_results.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    report = folder / f"round{round_no}_report.md"
    report.write_text(render_report(round_no, results, meta, previous, changes), encoding="utf-8")
    return report
