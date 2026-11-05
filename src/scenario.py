"""CLI 시나리오 평가(`volunteer-agent scenario <파일|폴더>`, AC16). 형식은 `eval/scenarios/README.md`.

- 시나리오 파일 하나 = 빈 임시 DATA_DIR 하나 = 대화 하나(`new_thread: true` 단계부터 새 대화, DATA_DIR은 유지).
- 단계는 순서대로 이어서 실행한다. `today`는 그 단계부터 기준 날짜가 된다(날짜는 앞으로만 흐른다).
- Agent·도구 판정은 이번 단계에 trace.jsonl에 새로 쓰인 줄로 한다. 답변 문구는 답변으로, `state`는 저장소를 직접 조회한다.
- 기대값은 평가 기준이므로 여기서 느슨하게 해석하지 않는다(문자열 포함, 고정 문구는 완전 일치).
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from config import Settings
from guards import REFUSALS
from llm import is_quota_error
from masking import mask_exception, mask_secrets
from tracing import FileTracer, read_trace, trace_path

ROOT = Path(__file__).resolve().parents[1]
EXPECT_KEYS = {"agents", "tools", "no_tools", "not_tools", "contains", "contains_any", "not_contains", "refusal",
               "not_contains_env", "state"}
STEP_KEYS = {"id", "input", "attach", "today", "run", "new_thread", "expect"}
QUOTA_REASON = "Bedrock 사용 한도 초과로 중단"
SKIPPED_REASON = "실행하지 않음: Bedrock 사용 한도 초과"
EXIT_QUOTA = 3  # 사용 한도 초과로 평가를 끝까지 돌리지 못함(판정 실패와 구분)

ENV_FIELDS = {  # not_contains_env 이름 → Settings 필드
    "DATA_GO_KR_KEY": "data_go_kr_key", "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
    "AWS_ACCESS_KEY_ID": "aws_access_key_id", "AWS_SESSION_TOKEN": "aws_session_token",
    "SMTP_PASSWORD": "smtp_password", "ANTHROPIC_API_KEY": "anthropic_api_key",
}


class ScenarioError(ValueError):
    """시나리오 파일 형식 오류."""


# ---------------------------------------------------------------- 파일 읽기

@dataclass
class Step:
    id: str
    input: str | None = None
    attach: str | None = None
    today: date | None = None
    run: str | None = None
    new_thread: bool = False
    expect: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    user_type: str
    today: date
    api_mode: str
    steps: list[Step]
    path: Path | None = None


def _date(value: Any, where: str) -> date:
    try:
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError:
        raise ScenarioError(f"{where}: 날짜는 YYYY-MM-DD 형식이어야 합니다({value}).") from None


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    where = Path(path).name
    for key in ("name", "user_type", "today", "steps"):
        if key not in data:
            raise ScenarioError(f"{where}: '{key}'가 없습니다.")
    if data["user_type"] not in ("employee", "student"):
        raise ScenarioError(f"{where}: user_type은 employee 또는 student입니다.")
    steps = []
    for raw in data["steps"]:
        sid = str(raw.get("id", "?"))
        if unknown := set(raw) - STEP_KEYS:
            raise ScenarioError(f"{where} {sid}: 모르는 항목 {sorted(unknown)}")
        expect = raw.get("expect") or {}
        if unknown := set(expect) - EXPECT_KEYS:
            raise ScenarioError(f"{where} {sid}: 모르는 expect 항목 {sorted(unknown)}")
        if bool(raw.get("input")) == bool(raw.get("run")):
            raise ScenarioError(f"{where} {sid}: input과 run 중 하나만 있어야 합니다.")
        if raw.get("run") not in (None, "scheduler"):
            raise ScenarioError(f"{where} {sid}: run은 scheduler만 지원합니다.")
        if (ref := expect.get("refusal")) not in (None, False) and ref not in REFUSALS:
            raise ScenarioError(f"{where} {sid}: refusal 범주가 올바르지 않습니다({ref}).")
        steps.append(Step(id=sid, input=raw.get("input"), attach=raw.get("attach"),
                          today=_date(raw["today"], f"{where} {sid}") if raw.get("today") else None,
                          run=raw.get("run"), new_thread=bool(raw.get("new_thread")), expect=expect))
    return Scenario(name=str(data["name"]), user_type=data["user_type"], today=_date(data["today"], where),
                    api_mode=str(data.get("api_mode") or "fixture"), steps=steps, path=Path(path))


def scenario_files(target: Path) -> list[Path]:
    target = Path(target)
    if target.is_dir():
        return sorted(p for p in target.glob("*.yaml") if p.is_file())
    if target.is_file():
        return [target]
    raise ScenarioError(f"시나리오 파일이나 폴더가 없습니다: {target}")


def resolve_attachment(attach: str, scenario_path: Path | None) -> Path:
    for base in (Path.cwd(), ROOT, *((scenario_path.parent,) if scenario_path else ())):
        candidate = (base / attach).resolve()
        if candidate.is_file():
            return candidate
    raise ScenarioError(f"첨부 파일이 없습니다: {attach}")


# ---------------------------------------------------------------- 관찰과 판정

@dataclass
class Observation:
    answer: str
    agents: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    trace_text: str = ""


def observe_trace(events: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """이번 단계의 trace 줄에서 거친 하위 Agent와 호출한 도구(이동용 transfer_* 제외)를 순서대로 뽑는다."""
    agents: list[str] = []
    tools: list[str] = []
    for e in events:
        if e.get("event") == "handoff" and str(e.get("target", "")).endswith("_agent"):
            if e["target"] not in agents:
                agents.append(e["target"])
        elif e.get("event") == "tool_start":
            name = e.get("tool") or ""
            if name and not name.startswith(("transfer_to_", "transfer_back_to_")):
                tools.append(name)
    return agents, tools


def _num(value: Any) -> float | None:
    return None if value is None else float(value)


def judge(expect: dict[str, Any], obs: Observation, state: dict[str, Any] | None = None,
          env_values: dict[str, str | None] | None = None) -> list[str]:
    """실패 이유 목록. 비어 있으면 PASS."""
    reasons: list[str] = []
    answer = obs.answer or ""
    if missing := [a for a in expect.get("agents") or [] if a not in obs.agents]:
        reasons.append(f"거치지 않은 Agent: {missing} (실제: {obs.agents or '없음'})")
    if missing := [t for t in expect.get("tools") or [] if t not in obs.tools]:
        reasons.append(f"호출하지 않은 도구: {missing} (실제: {obs.tools or '없음'})")
    if expect.get("no_tools") and obs.tools:
        reasons.append(f"도구를 호출하면 안 되는데 호출함: {obs.tools}")
    if called := [t for t in expect.get("not_tools") or [] if t in obs.tools]:
        reasons.append(f"호출하면 안 되는 도구를 호출함: {called}")
    if missing := [s for s in expect.get("contains") or [] if str(s) not in answer]:
        reasons.append(f"답변에 없는 문구: {missing}")
    if (any_of := expect.get("contains_any")) and not any(str(s) in answer for s in any_of):
        reasons.append(f"답변에 다음 중 하나도 없음: {any_of}")
    if found := [s for s in expect.get("not_contains") or [] if str(s) in answer]:
        reasons.append(f"답변에 있으면 안 되는 문구: {found}")
    if "refusal" in expect:
        refusal = expect["refusal"]
        if refusal is False:
            if hit := [k for k, phrase in REFUSALS.items() if phrase in answer]:
                reasons.append(f"거절하면 안 되는데 고정 거절 문구가 나옴: {hit}")
        elif answer.strip() != REFUSALS[refusal]:
            reasons.append(f"고정 거절 문구({refusal})와 완전히 일치하지 않음")
    for name in expect.get("not_contains_env") or []:
        value = (env_values or {}).get(name)
        if value and (value in answer or value in obs.trace_text):
            reasons.append(f"{name} 값이 답변이나 trace에 노출됨")  # 값 자체는 출력하지 않음
    reasons += judge_state(expect.get("state") or {}, state or {})
    return reasons


def judge_state(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    reasons = []
    for key, want in expected.items():
        if key.endswith("_min"):
            got = actual.get(key.removesuffix("_min"))
            if got is None or got < want:
                reasons.append(f"state {key}: {want} 이상이어야 하는데 {got}")
        elif key.endswith("_contains"):
            got = actual.get(key.removesuffix("_contains"))
            if got is None or str(want) not in str(got):
                reasons.append(f"state {key}: '{want}'가 들어 있어야 하는데 {got!r}")
        else:
            got = actual.get(key)
            if isinstance(want, (int, float)) and not isinstance(want, bool):
                ok = got is not None and abs(float(got) - float(want)) < 1e-9
            else:
                ok = str(got) == str(want)
            if not ok:
                reasons.append(f"state {key}: {want}이어야 하는데 {got}")
    return reasons


# ---------------------------------------------------------------- 저장 상태 조회 (LLM 없이)

def read_state(settings: Settings, user_type: str, today: date, keys: set[str]) -> dict[str, Any]:
    """state 판정에 필요한 값만 저장소에서 직접 읽는다."""
    from notify import OUTBOX_DIR
    from records import RecordService
    from store import Store

    base = {k.removesuffix("_min").removesuffix("_contains") for k in keys}
    state: dict[str, Any] = {}
    with Store.open(settings) as store:
        svc = RecordService(store, settings, today)
        progress = [p for p in svc.progress() if p.user_type == user_type]
        prog = progress[0] if progress else None
        profile = store.get_profile(prog.profile_id) if prog and prog.profile_id else None
        acts = store.list_activities()
        state.update(
            remaining_hours=_num(prog.remaining_hours) if prog else None,
            pending_proof_hours=_num(prog.pending_proof_hours) if prog else None,
            goal_hours=_num(profile.goal_hours) if profile else None,
            period_start=profile.period_start.isoformat() if profile and profile.period_start else None,
            period_end=profile.period_end.isoformat() if profile and profile.period_end else None,
            confirmed_rules=len(store.list_rules(status="confirmed")),
            watchlist_count=len(store.list_watchlist()),
            approved_plans=sum(1 for p in store.list_plans() if p.approved),
            activity_count=len(acts),
            school_planned_count=sum(1 for a in acts if a.school_planned),
            latest_activity_lesson=(max(acts, key=lambda a: a.id).lesson if acts else None),
        )
    outbox = settings.data_dir / OUTBOX_DIR
    state["eml_files"] = len(list(outbox.glob("*.eml"))) if outbox.is_dir() else 0
    if base & {"preference_region", "preference_major"}:
        from memory import load_preferences, open_store

        mem = open_store(settings)
        try:
            prefs = load_preferences(mem)
        finally:
            mem.conn.close()
        state["preference_region"] = prefs.get("region")
        state["preference_major"] = prefs.get("major")
    if unknown := base - set(state):
        raise ScenarioError(f"모르는 state 항목: {sorted(unknown)}")
    return state


# ---------------------------------------------------------------- 실행

@dataclass
class StepResult:
    scenario: str
    step_id: str
    reasons: list[str]
    answer: str = ""
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.reasons


@dataclass
class ScenarioReport:
    name: str
    results: list[StepResult]
    trace_copy: Path | None = None
    aborted: bool = False  # 사용 한도 초과로 중간에 멈춤

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)


async def run_scenario(scenario: Scenario, base: Settings, runtime: Any, *, say: Callable[[str], None] = print,
                       work_dir: Path | None = None, keep: bool = False, show_answers: bool = False) -> ScenarioReport:
    """빈 임시 DATA_DIR에서 시나리오 하나를 실행한다. runtime은 cli.Runtime(open_app, make_guard)."""
    from agent import AppClock, ask
    from cli import with_attachment
    from notify import run_scheduler
    from store import Store

    data_dir = Path(work_dir or tempfile.mkdtemp(prefix=f"scenario-{scenario.name}-"))
    settings = replace(base, data_dir=data_dir.resolve())
    settings.ensure_data_dir()
    secrets = settings.secret_values()
    env_values = {name: getattr(settings, attr, None) for name, attr in ENV_FIELDS.items()}
    with Store.open(settings) as store:
        store.init_user_setting(scenario.user_type)
    clock = AppClock.create(settings, scenario.today)
    thread_id = f"{scenario.name}-1"
    guard = runtime.make_guard(settings)
    tpath = trace_path(settings)
    results: list[StepResult] = []
    aborted = False
    say(f"\n=== {scenario.name} ({len(scenario.steps)}단계, DATA_DIR {data_dir}) ===")
    try:
        async with runtime.open_app(settings, clock, scenario.api_mode) as app:
            for n, step in enumerate(scenario.steps, start=1):
                if step.today:
                    clock.set(step.today)
                if step.new_thread:
                    thread_id = f"{scenario.name}-{n}"
                tracer = FileTracer.for_settings(settings, thread_id)
                tracer.write("step", scenario=scenario.name, step=step.id, today=clock.today().isoformat())
                before = len(read_trace(tpath))
                started = datetime.now()
                answer = ""
                try:
                    if step.run == "scheduler":
                        with Store.open(settings) as store:
                            answer = run_scheduler(store, settings, clock.today()).message
                    else:
                        text = step.input or ""
                        if step.attach:
                            text = with_attachment(text, resolve_attachment(step.attach, scenario.path))
                        turn = await ask(app, text, thread_id, guard=guard, tracer=tracer)
                        answer = turn.answer
                    events = read_trace(tpath)[before:]
                    agents, tools = observe_trace(events)
                    obs = Observation(answer, agents, tools, "\n".join(map(str, events)))
                    state_keys = set(step.expect.get("state") or {})
                    state = read_state(settings, scenario.user_type, clock.today(), state_keys) if state_keys else {}
                    reasons = judge(step.expect, obs, state, env_values)
                except Exception as exc:  # noqa: BLE001 - 단계 실패로 기록하고 다음 단계로
                    if is_quota_error(exc):  # 남은 단계도 같은 오류로 재시도만 반복하므로 여기서 멈춘다
                        aborted = True
                        reasons = [QUOTA_REASON]
                    else:
                        reasons = [f"실행 오류: {mask_exception(exc, secrets)}"]
                seconds = (datetime.now() - started).total_seconds()
                result = StepResult(scenario.name, step.id, reasons, mask_secrets(answer, secrets), seconds)
                results.append(result)
                tracer.write("step_result", scenario=scenario.name, step=step.id, passed=result.passed,
                             reasons=reasons, latency_s=round(seconds, 3))
                mark = "PASS" if result.passed else "FAIL"
                say(f"[{mark}] {step.id} ({seconds:.1f}s)" + ("" if result.passed else " - " + "; ".join(reasons)))
                if (show_answers or not result.passed) and not aborted:
                    say("       답변: " + " ".join(result.answer.split())[:300])
                if aborted:
                    rest = scenario.steps[n:]
                    results += [StepResult(scenario.name, st.id, [SKIPPED_REASON]) for st in rest]
                    if rest:
                        say(f"[중단] 남은 {len(rest)}단계는 실행하지 않았습니다({', '.join(st.id for st in rest)}).")
                    break
    finally:
        copy = _save_trace(base, scenario.name, tpath)
        if not keep and work_dir is None:
            shutil.rmtree(data_dir, ignore_errors=True)
    report = ScenarioReport(scenario.name, results, copy, aborted)
    say(f"--- {scenario.name}: {report.passed}/{len(results)} 통과" + (f", trace {copy}" if copy else ""))
    return report


def _save_trace(base: Settings, name: str, tpath: Path) -> Path | None:
    """임시 DATA_DIR은 지우므로 단계별 trace를 실제 DATA_DIR/scenario_runs/ 아래에 남긴다."""
    if not tpath.is_file():
        return None
    target = base.ensure_data_dir() / "scenario_runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{name}-{uuid.uuid4().hex[:4]}.trace.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tpath, target)
    return target


async def run_all(target: Path, base: Settings, runtime: Any, *, say: Callable[[str], None] = print,
                  keep: bool = False, show_answers: bool = False) -> list[ScenarioReport]:
    scenarios = [load_scenario(p) for p in scenario_files(target)]  # 형식 오류는 실행 전에 모두 확인
    reports: list[ScenarioReport] = []
    for s in scenarios:
        if reports and reports[-1].aborted:  # 한도가 풀리기 전에는 다음 시나리오도 실패하므로 돌리지 않는다
            reports.append(ScenarioReport(s.name, [StepResult(s.name, st.id, [SKIPPED_REASON]) for st in s.steps],
                                          aborted=True))
            say(f"\n=== {s.name}: 실행하지 않음(Bedrock 사용 한도 초과) ===")
            continue
        reports.append(await run_scenario(s, base, runtime, say=say, keep=keep, show_answers=show_answers))
    total = sum(len(r.results) for r in reports)
    passed = sum(r.passed for r in reports)
    say("\n=== 결과 ===")
    for r in reports:
        say(f"{r.name}: {r.passed}/{len(r.results)}")
    say(f"전체: {passed}/{total} 통과")
    if any(r.aborted for r in reports):
        skipped = sum(1 for r in reports for x in r.results if x.reasons == [SKIPPED_REASON])
        say(f"중단: Bedrock 사용 한도 초과로 {skipped}단계를 실행하지 않았습니다. 한도가 풀린 뒤 다시 실행하세요.")
    return reports
