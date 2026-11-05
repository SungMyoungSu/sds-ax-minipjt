import asyncio
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.store.memory import InMemoryStore

from volunteer_agent import mcp_server as srv
from volunteer_agent.memory import memory_tools, save_preferences
from volunteer_agent.models import RuleCandidate
from volunteer_agent.planner import (
    PLAN_KEY,
    RECURSION_LIMIT,
    Plan,
    PlanStep,
    Replan,
    build_planner,
    ceil_half,
    compute_plan,
    format_plan,
)
from volunteer_agent.records import ActivityInput, RecordService
from volunteer_agent.store import Store

TODAY = date(2026, 12, 1)


def cand(pid, day, hours, start=10):
    return {"program_id": pid, "title": f"활동 {pid}", "url": f"https://1365.go.kr/x?progrmRegistNo={pid}",
            "program_start": day, "expected_hours": hours, "start_hour": start, "end_hour": start + hours,
            "recruit_end": day}


# ---- 계획 계산 ----

@pytest.mark.parametrize("x, y", [(0.8, 1.0), (1.0, 1.0), (1.01, 1.5), (2.4, 2.5), (0.0, 0.0)])
def test_ceil_half(x, y):
    assert ceil_half(x) == y


def test_ac19_plan_covers_target_before_period_end():
    cands = [cand("A", "2026-12-05", 2), cand("B", "2026-12-12", 3), cand("C", "2027-01-02", 5)]
    plan = compute_plan(4, TODAY, date(2026, 12, 31), cands, None, profile_id=1)
    items = [it for w in plan["weeks"] for it in w["items"]]
    assert plan["expected_total"] >= 4 and plan["shortfall_hours"] == 0
    assert all(it["date"] <= "2026-12-31" for it in items)  # C(1월)는 제외
    assert all(it["url"].startswith("https://1365.go.kr") for it in items)
    assert [w["target_hours"] for w in plan["weeks"]][:2] == [1.0, 1.0]  # 4시간/5주 → 0.8 → 1.0
    assert len(plan["weeks"]) == 5 and plan["weeks"][-1]["end"] == "2026-12-31"


def test_empty_week_and_shortfall_without_relaxing():
    plan = compute_plan(6, TODAY, date(2026, 12, 21), [cand("A", "2026-12-05", 2)], None, profile_id=1)
    assert plan["expected_total"] == 2 and plan["shortfall_hours"] == 4
    assert [w["empty"] for w in plan["weeks"]] == [False, True, True]
    text = format_plan(plan, "근거")
    assert "빈 주(조건에 맞는 봉사 없음)" in text and "예상 부족 4시간" in text
    for label in ("남은 시간: 6시간", "기간 종료일: 2026-12-21", "목표 2시간", "1365 원문:", "예상 합계: 2시간", "계획 근거: 근거"):
        assert label in text


def test_preferred_weekdays_first_within_week():
    # 12/1(화) 주차: 12/2(수) 평일 후보와 12/5(토) 주말 후보 → 주말 선호면 토요일 후보를 먼저
    cands = [cand("WED", "2026-12-02", 2), cand("SAT", "2026-12-05", 2)]
    plan = compute_plan(2, TODAY, date(2026, 12, 7), cands, [5, 6], profile_id=1)
    assert [it["program_id"] for it in plan["weeks"][0]["items"]] == ["SAT"]
    plan = compute_plan(2, TODAY, date(2026, 12, 7), cands, None, profile_id=1)
    assert [it["program_id"] for it in plan["weeks"][0]["items"]] == ["WED"]


# ---- 그래프 (실제 MCP 도구 함수 + fixture API, 구조화 출력만 가짜) ----

class FakeStructured:
    """with_structured_output(schema).ainvoke(prompt) 흉내."""

    def __init__(self, plan: Plan, replan=lambda n: Replan(action="continue", reason="계속")):
        self.plan, self.replan, self.calls = plan, replan, []

    def with_structured_output(self, schema):
        outer = self

        class Runner:
            async def ainvoke(self, prompt):
                outer.calls.append(schema.__name__)
                return outer.plan if schema is Plan else outer.replan(len(outer.calls))
        return Runner()


@pytest.fixture
def env(data_dir, monkeypatch):
    monkeypatch.setenv("VOLUNTEER_TODAY", TODAY.isoformat())
    monkeypatch.setenv("VOLUNTEER_API_MODE", "fixture")
    monkeypatch.setenv("DATA_GO_KR_KEY", "fakeKey")
    monkeypatch.setenv("VOLUNTEER_1365_BASE_URL", "https://apis.data.go.kr/1741000/volunteerPartcptnService")
    return data_dir


def employee_with_remaining_4(settings, tmp_path):
    cert = tmp_path / "c.pdf"
    cert.write_bytes(b"%PDF")
    with Store.open(settings) as store:
        store.init_user_setting("employee")
        svc = RecordService(store, settings, TODAY)
        p = svc.create_profile("employee")
        p.goal_hours, p.certificate_required, p.allowed_type_codes = 8, True, ["0800", "0100"]
        p.period_start, p.period_end = date(2026, 1, 1), date(2026, 12, 31)
        store.update_profile(p)
        store.mark_rule_confirmed(store.add_rule_candidate(RuleCandidate(profile_id=p.id)).id)
        svc.log_activity(ActivityInput(start_date=date(2026, 11, 1), organization="복지관", hours=4,
                                       type_code="0100", certificate_file=cert))


def planner_tools(mem):
    tools = [StructuredTool.from_function(func=getattr(srv, n), name=n)
             for n in ("get_progress", "get_upcoming_deadlines", "search_volunteer", "approve_plan")]
    return tools + [t for t in memory_tools(mem) if t.name == "get_preferences"]


def run(graph, messages):
    return asyncio.run(graph.ainvoke({"messages": messages}, {"recursion_limit": RECURSION_LIMIT}))


STEPS = Plan(steps=[PlanStep(action="get_preferences", reason="선호"), PlanStep(action="search_volunteer", reason="후보"),
                    PlanStep(action="get_upcoming_deadlines", reason="마감")])


def test_ac19_ac25_plan_then_approve(env, settings, tmp_path):
    employee_with_remaining_4(settings, tmp_path)
    mem = InMemoryStore()
    save_preferences(mem, region="서울", available_days=[5, 6])
    llm = FakeStructured(STEPS)
    graph = build_planner(llm, planner_tools(mem), lambda: TODAY)

    request = HumanMessage(content="12월까지 남은 시간 채우는 계획 세워줘")
    out = run(graph, [request])
    reply = out["messages"][-1]
    plan = reply.additional_kwargs[PLAN_KEY]
    items = [it for w in plan["weeks"] for it in w["items"]]
    assert plan["target_hours"] == 4 and plan["expected_total"] >= 4
    assert all(it["date"] <= "2026-12-31" and "1365.go.kr" in it["url"] for it in items)
    assert "남은 시간: 4시간" in reply.content and "주차" in reply.content and "1365.go.kr" in reply.content
    assert llm.calls[0] == "Plan" and "Replan" in llm.calls  # planner → execute → replan → finalize
    with Store.open(settings) as store:  # 승인 전에는 저장하지 않음 (AC25)
        assert store.list_plans() == [] and store.list_watchlist() == []

    # 확정이 아닌 말이면 승인하지 않음
    out2 = run(graph, [request, reply, HumanMessage(content="음 다시 생각해 볼게")])
    assert "저장했습니다" not in out2["messages"][-1].content
    with Store.open(settings) as store:
        assert store.list_plans() == []

    out3 = run(graph, [request, reply, HumanMessage(content="응, 이 계획대로 할게")])
    assert "저장했습니다" in out3["messages"][-1].content
    with Store.open(settings) as store:
        [saved] = store.list_plans()
        assert saved.approved and saved.approved_at is not None
        assert {w.program_id for w in store.list_watchlist()} == {it["program_id"] for it in items}
        assert any(d.kind == "recruit_end" for d in store.list_deadlines())


def test_student_without_goal_is_asked_first(env, settings):
    with Store.open(settings) as store:
        store.init_user_setting("student")
        RecordService(store, settings, TODAY).create_profile("student", major="간호학")
    graph = build_planner(FakeStructured(STEPS), planner_tools(InMemoryStore()), lambda: TODAY)
    out = run(graph, [HumanMessage(content="봉사 계획 세워줘")])
    reply = out["messages"][-1]
    assert "몇 시간" in reply.content and PLAN_KEY not in reply.additional_kwargs


def test_student_target_from_request(env, settings):
    with Store.open(settings) as store:
        store.init_user_setting("student")
        RecordService(store, settings, TODAY).create_profile("student", major="간호학")
    plan = Plan(steps=STEPS.steps, target_hours=4)
    graph = build_planner(FakeStructured(plan), planner_tools(InMemoryStore()), lambda: TODAY)
    reply = run(graph, [HumanMessage(content="이번 학기 4시간 하고 싶어. 계획 세워줘")])["messages"][-1]
    assert reply.additional_kwargs[PLAN_KEY]["target_hours"] == 4


def test_replan_loop_is_bounded(env, settings, tmp_path):
    employee_with_remaining_4(settings, tmp_path)
    endless = FakeStructured(STEPS, replan=lambda n: Replan(
        action="revise", revised_steps=[PlanStep(action="get_upcoming_deadlines", reason="또")], reason="반복"))
    graph = build_planner(endless, planner_tools(InMemoryStore()), lambda: TODAY)
    out = run(graph, [HumanMessage(content="계획 세워줘")])  # recursion_limit 안에서 끝나야 함
    assert PLAN_KEY in out["messages"][-1].additional_kwargs


def test_approve_plan_tool_requires_explicit_approval(env, settings, tmp_path):
    employee_with_remaining_4(settings, tmp_path)
    plan = {"profile_id": 1, "weeks": [{"items": [{"program_id": "P1", "title": "t", "url": "u",
                                                    "recruit_end": "2026-12-05"}]}]}
    assert "저장하지 않았습니다" in srv.approve_plan(plan, "잠깐만")
    assert "계획 #1" in srv.approve_plan(plan, "응, 이 계획대로 할게")
    assert "프로필 번호" in srv.approve_plan({"weeks": []}, "응")
