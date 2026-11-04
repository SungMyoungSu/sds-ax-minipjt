"""plan_and_execute.py - Plan-and-Execute 완성본 (checkpoint 배포)"""
from typing import TypedDict, Annotated, Literal
from operator import add

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
)


def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


# ---------- 1) State ----------
class State(TypedDict):
    user_request: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], add]
    final_response: str


# ---------- 2) 계획 수립 노드 ----------
class Plan(BaseModel):
    steps: list[str] = Field(description="순서대로 실행할 단계 리스트")


planner_llm = llm.with_structured_output(Plan)


def make_plan(state: State) -> dict:
    plan = planner_llm.invoke(
        f"다음 요청을 5-7단계의 구체적 계획으로 분해하세요.\n"
        f"각 단계는 한 문장, 실행 가능한 행동이어야 합니다.\n\n"
        f"요청: {state['user_request']}"
    )
    print(f"[planner] 계획 {len(plan.steps)}단계")
    for i, s in enumerate(plan.steps):
        print(f"  {i + 1}. {s}")
    return {"plan": plan.steps}


# ---------- 3) 단계 실행 노드 ----------
def execute_step(state: State) -> dict:
    current = state["plan"][0]
    print(f"\n[execute] {current}")

    context = (
        "전체 계획:\n"
        + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(state["plan"]))
        + "\n\n완료된 단계:\n"
        + "\n".join(f"  - {s}: {r[:60]}" for s, r in state["past_steps"])
        + f"\n\n지금 실행할 단계: {current}\n실행 결과를 한두 문장으로 요약하세요."
    )
    result = get_text(llm.invoke(context))
    return {
        "plan": state["plan"][1:],           # 첫 단계 제거 (덮어쓰기)
        "past_steps": [(current, result)],   # add 리듀서가 누적
    }


# ---------- 4) 재계획 노드 ----------
class Replan(BaseModel):
    """계획을 계속할지, 수정할지, 끝낼지 결정한다."""
    action: Literal["continue", "revise", "finish"] = Field(
        description="continue=남은 계획 유지, revise=계획 수정, finish=종료"
    )
    revised_plan: list[str] = Field(
        default_factory=list, description="revise일 때 새 남은 계획"
    )
    reason: str = Field(description="판단 근거 한 문장")


replanner_llm = llm.with_structured_output(Replan)


def replan(state: State) -> dict:
    decision = replanner_llm.invoke(
        f"원래 요청: {state['user_request']}\n\n"
        f"완료된 단계와 결과:\n"
        + "\n".join(f"  - {s}: {r[:80]}" for s, r in state["past_steps"])
        + "\n\n남은 계획:\n"
        + "\n".join(f"  - {s}" for s in state["plan"])
        + "\n\n판단하세요.\n"
          "- 남은 계획이 여전히 유효하면 continue\n"
          "- 직전 결과 때문에 남은 계획을 바꿔야 하면 revise (새 계획 제시)\n"
          "- 요청에 답하기 충분한 정보가 모였거나 남은 계획이 없으면 finish"
    )
    print(f"[replan] {decision.action} - {decision.reason}")
    if decision.action == "revise":
        return {"plan": decision.revised_plan}
    if decision.action == "finish":
        return {"plan": []}
    return {}


def after_replan(state: State) -> Literal["execute", "finalize"]:
    return "finalize" if not state["plan"] else "execute"


# ---------- 5) 마무리 노드 ----------
def finalize(state: State) -> dict:
    summary = "\n".join(f"- {s}: {r[:100]}" for s, r in state["past_steps"])
    response = llm.invoke(
        f"원래 요청: {state['user_request']}\n\n실행 내역:\n{summary}\n\n"
        f"실행 내역을 근거로 최종 답변을 작성하세요."
    )
    return {"final_response": get_text(response)}


# ---------- 6) 그래프 조립 ----------
builder = StateGraph(State)
builder.add_node("planner", make_plan)
builder.add_node("execute", execute_step)
builder.add_node("replan", replan)
builder.add_node("finalize", finalize)

builder.add_edge(START, "planner")
builder.add_edge("planner", "execute")
builder.add_edge("execute", "replan")
builder.add_conditional_edges("replan", after_replan,
                              {"execute": "execute", "finalize": "finalize"})
builder.add_edge("finalize", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {
            # 교육용 더미 데이터 기반 시나리오입니다
            "user_request": "클라우드운영팀의 이번 분기 클라우드 비용을 분석해서 다음 분기 예산을 추정",
            "plan": [],
            "past_steps": [],
            "final_response": "",
        },
        {"recursion_limit": 40},   # 안전장치: 무한 반복 방지
    )
    print(f"\n=== 최종 답변 ===\n{result['final_response']}")
