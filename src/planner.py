"""planner_agent: Plan-and-Execute 그래프 (planner → execute → replan → finalize).

- planner·replan은 LLM 구조화 출력(Plan, Replan)으로 조회 순서를 정한다.
- execute는 실제 도구(get_progress, get_upcoming_deadlines, get_preferences, search_volunteer)를 부른다.
- finalize는 LLM이 아니라 코드가 계획 숫자를 계산한다(주차별 목표 균등 배분·0.5시간 올림, 선호 요일 우선,
  빈 주는 조건을 바꾸지 않고 빈 주로, 못 채우면 '예상 부족 N시간').
- 계획은 제안만 한다. 저장은 사용자가 승인했을 때 approve_plan으로만 한다(가드레일 4, AC25).
  승인 전 계획 초안은 대화 상태(planner 답 메시지의 additional_kwargs)에만 둔다.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from datetime import date, timedelta
from operator import add
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field, field_validator

from llm import get_text
from rules import is_explicit_confirmation

RECURSION_LIMIT = 40
MAX_STEPS = 8
PLAN_KEY = "volunteer_plan"
ACTIONS = ("get_progress", "get_upcoming_deadlines", "get_preferences", "search_volunteer")
REQUIRED = ("get_progress", "search_volunteer")
WEEKDAYS = "월화수목금토일"


# ---- 구조화 출력 ----

class PlanStep(BaseModel):
    action: Literal["get_progress", "get_upcoming_deadlines", "get_preferences", "search_volunteer"]
    reason: str = Field(description="이 단계가 필요한 이유 한 문장")


class Plan(BaseModel):
    """봉사 계획을 세우려고 조회할 단계들."""

    steps: list[PlanStep] = Field(description="순서대로 실행할 조회 단계")
    target_hours: float | None = Field(None, description="사용자가 이번 요청에서 직접 말한 목표 시간. 말하지 않았으면 null")

    @field_validator("target_hours", mode="before")
    @classmethod
    def _none_like(cls, v: Any) -> Any:
        """LLM이 'null', '없음' 같은 문자열을 넣는 경우를 None으로 받는다."""
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "없음", "n/a"}:
            return None
        return v


class Replan(BaseModel):
    """남은 조회를 계속할지, 바꿀지, 끝낼지 정한다."""

    action: Literal["continue", "revise", "finish"] = Field(description="continue=유지, revise=수정, finish=종료")
    revised_steps: list[PlanStep] = Field(default_factory=list, description="revise일 때 새 남은 단계")
    reason: str = Field(description="판단 근거 한 문장")


class PlannerState(MessagesState):
    request: str
    steps: list[str]
    past_steps: Annotated[list[tuple[str, str]], add]
    target_hours: float | None
    results: dict


# ---- 계획 계산 (결정적) ----

def ceil_half(x: float) -> float:
    return math.ceil(x * 2 - 1e-9) / 2


def build_weeks(today: date, end: date) -> list[tuple[date, date]]:
    weeks, start = [], today
    while start <= end:
        weeks.append((start, min(end, start + timedelta(days=6))))
        start += timedelta(days=7)
    return weeks


def compute_plan(target: float, today: date, end: date, candidates: list[dict], preferred_days: list[int] | None,
                 profile_id: int, sample: bool = False) -> dict:
    """남은 주에 목표를 고르게 나누고(0.5시간 올림) 주마다 후보를 채운다. 조건을 바꿔 다시 찾지 않는다."""
    preferred = set(preferred_days or [])
    pool = []
    for c in candidates:
        if not c.get("program_start") or not c.get("expected_hours"):
            continue
        d = date.fromisoformat(c["program_start"])
        if today <= d <= end:
            pool.append((0 if d.weekday() in preferred else 1, d, c))
    pool.sort(key=lambda x: (x[1], x[0]))

    weeks_out, total, used = [], 0.0, set()
    weeks = build_weeks(today, end)
    week_target = ceil_half(target / len(weeks)) if weeks else 0.0  # 남은 주에 고르게(미결 결정 7)
    for i, (ws, we) in enumerate(weeks):
        need = max(target - total, 0.0)
        items, got = [], 0.0
        in_week = sorted((p for p in pool if ws <= p[1] <= we and p[2]["program_id"] not in used),
                         key=lambda p: (p[0], p[1]))
        for _, d, c in in_week:
            if got >= week_target or total + got >= target:
                break
            used.add(c["program_id"])
            hours = float(c["expected_hours"])
            got += hours
            items.append({"date": d.isoformat(), "program_id": c["program_id"], "title": c["title"],
                          "hours": hours, "url": c["url"], "recruit_end": c.get("recruit_end"),
                          "time": f"{c['start_hour']}시~{c['end_hour']}시" if c.get("start_hour") is not None else ""})
        total += got
        weeks_out.append({"week": i + 1, "start": ws.isoformat(), "end": we.isoformat(),
                          "target_hours": week_target, "items": items, "empty": need > 0 and not items})
    return {"profile_id": profile_id, "target_hours": target, "period_end": end.isoformat(), "weeks": weeks_out,
            "expected_total": total, "shortfall_hours": max(target - total, 0.0), "sample": sample}


def format_plan(plan: dict, rationale: str) -> str:
    lines = [
        "봉사 계획(제안)입니다. 아직 저장하지 않았습니다.",
        f"- 남은 시간: {plan['target_hours']:g}시간",
        f"- 기간 종료일: {plan['period_end']}",
    ]
    for w in plan["weeks"]:
        head = f"{w['week']}주차({w['start']}~{w['end']}) 목표 {w['target_hours']:g}시간"
        if w["empty"]:
            lines.append(f"- {head}: 빈 주(조건에 맞는 봉사 없음)")
        elif not w["items"]:
            lines.append(f"- {head}: 앞 주에서 목표를 채워 추가 활동 필요 없음")
        else:
            lines.append(f"- {head}:")
            for it in w["items"]:
                d = date.fromisoformat(it["date"])
                lines.append(f"  · {it['title']} | {it['date']}({WEEKDAYS[d.weekday()]}) {it['time']} | "
                             f"예상 {it['hours']:g}시간 | 1365 원문: {it['url']}")
    lines.append(f"- 예상 합계: {plan['expected_total']:g}시간")
    if plan["shortfall_hours"] > 0:
        lines.append(f"- 예상 부족 {plan['shortfall_hours']:g}시간 (남은 기간에 조건에 맞는 봉사가 부족합니다)")
    if plan.get("sample"):
        lines.append("- 샘플 데이터로 추천했습니다(공공 API를 쓸 수 없음).")
    lines.append(f"- 계획 근거: {rationale}")
    lines.append("이 계획대로 할까요? 승인하시면 계획을 저장하고 계획의 봉사를 관심 목록에 넣습니다.")
    return "\n".join(lines)


def _parse_prefs(text: str) -> dict:
    out: dict[str, Any] = {}
    m = re.search(r"지역: ([^|]+?)(?: \||$)", text)
    if m:
        out["region"] = m.group(1).strip()
    m = re.search(r"weekdays=\[([\d, ]*)\]", text)
    if m and m.group(1).strip():
        out["weekdays"] = [int(x) for x in m.group(1).split(",")]
    return out


def pending_plan(messages: list) -> dict | None:
    """마지막 사용자 메시지 바로 앞까지에서, 아직 승인되지 않은 최근 계획 초안."""
    for m in reversed(messages[:-1]):
        if isinstance(m, AIMessage) and m.name == "planner_agent":
            plan = (m.additional_kwargs or {}).get(PLAN_KEY)
            if plan:
                return plan
            if "계획 #" in get_text(m) and "저장했습니다" in get_text(m):
                return None  # 이미 승인됨
    return None


# ---- 그래프 ----

def build_planner(llm: Any, tools: list[Any], today_fn: Callable[[], date]):
    by_name = {t.name: t for t in tools}

    async def call_tool(name: str, args: dict, config: RunnableConfig | None = None) -> str:
        # 실행 설정(콜백)을 넘겨야 실행 추적과 시나리오 판정에 도구 호출이 남는다
        result = await by_name[name].ainvoke(args, config=config)
        return result if isinstance(result, str) else get_text(AIMessage(content=result))

    def last_human(state: dict) -> str:
        return next((get_text(m) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    def route(state: PlannerState) -> Literal["approve", "planner"]:
        msgs = state["messages"]
        if pending_plan(msgs) and is_explicit_confirmation(last_human(state)):
            return "approve"
        return "planner"

    async def approve(state: PlannerState, config: RunnableConfig) -> dict:
        plan = pending_plan(state["messages"])
        text = await call_tool("approve_plan", {"plan": plan, "confirmation_text": last_human(state)}, config)
        return {"messages": [AIMessage(content=text, name="planner_agent")]}

    async def planner(state: PlannerState) -> dict:
        request = last_human(state)
        try:
            plan: Plan = await llm.with_structured_output(Plan).ainvoke(
                "너는 봉사 계획 담당이다. 아래 요청에 맞는 봉사 계획을 세우려면 어떤 조회가 필요한지 순서대로 정하라.\n"
                f"쓸 수 있는 조회: {', '.join(ACTIONS)}. 남은 시간(get_progress)과 후보 검색(search_volunteer)은 꼭 필요하다.\n"
                f"오늘: {today_fn().isoformat()}\n요청: {request}")
        except Exception:  # noqa: BLE001 - 구조화 출력 실패 시 기본 조회 순서(계획 숫자는 코드가 계산)
            plan = Plan(steps=[PlanStep(action=a, reason="기본 순서") for a in ACTIONS])
        steps = [s.action for s in plan.steps if s.action in ACTIONS]
        for req in REQUIRED:  # 필수 조회 보강
            if req not in steps:
                steps.append(req)
        if "get_preferences" in steps and steps.index("get_preferences") > steps.index("search_volunteer"):
            steps.remove("get_preferences")
            steps.insert(steps.index("search_volunteer"), "get_preferences")
        steps = [s for s in steps if s != "get_progress"]
        return {"request": request, "steps": ["get_progress", *steps], "target_hours": plan.target_hours,
                "results": {}}

    async def execute(state: PlannerState, config: RunnableConfig) -> dict:
        step, results = state["steps"][0], dict(state.get("results") or {})
        today = today_fn()
        if step == "get_progress":
            out = await call_tool("get_progress", {"output": "json"}, config)
            results["progress"] = json.loads(out) if out.lstrip().startswith("{") else {"error": out}
        elif step == "get_upcoming_deadlines":
            out = await call_tool("get_upcoming_deadlines", {"days": 30}, config)
            results["deadlines"] = out
        elif step == "get_preferences":
            out = await call_tool("get_preferences", {}, config)
            results["preferences"] = _parse_prefs(out)
        else:
            prefs = results.get("preferences", {})
            target = _choose_target(results, state.get("target_hours"), today)
            args: dict[str, Any] = {"region": prefs.get("region") or "상관없음", "start": today.isoformat(),
                                    "limit": 30, "output": "json"}
            if target:
                args["end"] = target["end"].isoformat()
            out = await call_tool("search_volunteer", args, config)
            results["search"] = json.loads(out) if out.lstrip().startswith("{") else {"items": [], "error": out}
        return {"steps": state["steps"][1:], "past_steps": [(step, out[:300])], "results": results}

    async def replan(state: PlannerState) -> dict:
        if not state["steps"] or len(state["past_steps"]) >= MAX_STEPS:
            return {"steps": []}
        try:
            decision: Replan = await llm.with_structured_output(Replan).ainvoke(
                f"요청: {state['request']}\n완료된 조회:\n"
                + "\n".join(f"- {s}: {r[:120]}" for s, r in state["past_steps"])
                + "\n남은 조회: " + ", ".join(state["steps"])
                + "\n남은 조회가 여전히 필요하면 continue, 바꿔야 하면 revise, 계획을 세울 정보가 충분하면 finish. "
                  "단 search_volunteer가 아직 남아 있으면 finish하지 않는다.")
        except Exception:  # noqa: BLE001 - 판단 실패 시 남은 조회를 그대로 계속
            return {}
        if decision.action == "revise":
            steps = [s.action for s in decision.revised_steps if s.action in ACTIONS]
            done = {s for s, _ in state["past_steps"]}
            if "search_volunteer" not in done and "search_volunteer" not in steps:
                steps.append("search_volunteer")
            return {"steps": steps}
        if decision.action == "finish" and "search_volunteer" in state["steps"]:
            return {}  # 필수 검색 전에는 끝내지 않음
        return {"steps": []} if decision.action == "finish" else {}

    def after_replan(state: PlannerState) -> Literal["execute", "finalize"]:
        return "execute" if state["steps"] else "finalize"

    async def finalize(state: PlannerState) -> dict:
        results, today = state.get("results") or {}, today_fn()
        target = _choose_target(results, state.get("target_hours"), today)
        if target is None:
            text = ("계획을 세우려면 목표가 필요합니다. 이번 기간에 몇 시간(또는 몇 회) 봉사하고 싶은지 알려 주세요. "
                    "회사원이라면 회사 규칙이 확정되어 있어야 남은 시간을 계산할 수 있습니다.")
            return {"messages": [AIMessage(content=text, name="planner_agent")]}
        if target["hours"] <= 0:
            return {"messages": [AIMessage(content="이번 기간 목표를 이미 채웠습니다. 추가 계획이 필요하지 않습니다.",
                                           name="planner_agent")]}
        search = results.get("search", {})
        prefs = results.get("preferences", {})
        plan = compute_plan(target["hours"], today, target["end"], search.get("items", []), prefs.get("weekdays"),
                            target["profile_id"], bool(search.get("sample")))
        n_items = sum(len(w["items"]) for w in plan["weeks"])
        rationale = (f"{target['source']} 남은 {target['hours']:g}시간을 {target['end'].isoformat()}까지 "
                     f"{len(plan['weeks'])}주에 고르게 나눴고(0.5시간 단위 올림), 모집 중인 1365 봉사 {n_items}건을 "
                     + ("선호 요일을 먼저 " if prefs.get("weekdays") else "")
                     + "날짜순으로 배치했습니다. 조건에 맞는 봉사가 없는 주는 조건을 바꾸지 않고 빈 주로 두었습니다.")
        return {"messages": [AIMessage(content=format_plan(plan, rationale), name="planner_agent",
                                       additional_kwargs={PLAN_KEY: plan})]}

    g = StateGraph(PlannerState)
    g.add_node("planner", planner)
    g.add_node("execute", execute)
    g.add_node("replan", replan)
    g.add_node("finalize", finalize)
    g.add_node("approve", approve)
    g.add_conditional_edges(START, route, {"approve": "approve", "planner": "planner"})
    g.add_edge("planner", "execute")
    g.add_edge("execute", "replan")
    g.add_conditional_edges("replan", after_replan, {"execute": "execute", "finalize": "finalize"})
    g.add_edge("finalize", END)
    g.add_edge("approve", END)
    return g.compile(name="planner_agent")


def _choose_target(results: dict, user_target: float | None, today: date) -> dict | None:
    """계획 목표: 사용자가 말한 시간 > 목표가 있는 프로필의 남은 시간. 기간 끝은 프로필 기간(없으면 4주 뒤)."""
    profiles = (results.get("progress") or {}).get("profiles") or []
    usable = [p for p in profiles if not p.get("rules_pending")]
    usable.sort(key=lambda p: 0 if p.get("source") == "company_policy" else 1)
    profile = next((p for p in usable if p.get("remaining_hours") is not None), None) or (usable[0] if usable else None)
    if profile is None:
        return None
    end = date.fromisoformat(profile["period_end"]) if profile.get("period_end") else today + timedelta(days=27)
    if user_target:
        return {"hours": float(user_target), "end": end, "profile_id": profile["profile_id"], "source": "요청한 목표"}
    if profile.get("remaining_hours") is None:
        return None  # 목표 없는 학생 → 목표부터 묻기 (PRD 33)
    source = "회사 규칙의" if profile.get("source") == "company_policy" else "목표의"
    return {"hours": float(profile["remaining_hours"]), "end": end, "profile_id": profile["profile_id"], "source": source}
