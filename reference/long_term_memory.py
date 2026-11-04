"""long_term_memory.py - Store 저장, 회수, 그래프 연결 (checkpoint 배포)"""
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv()
llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-east-1",
    temperature=0,
)

store = InMemoryStore()
user_id = "haneul_kim"   # Store 네임스페이스 라벨에는 점을 쓸 수 없습니다


def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content


# ---------- 1) 저장 ----------
store.put(
    ("users", user_id),          # namespace: 사용자별로 분리
    "profile",                   # key
    {                            # value: 딕셔너리 (교육용 더미 데이터)
        "name": "김하늘",
        "team": "클라우드운영팀",
        "email": "haneul.kim@samsungsds.example.com",
        "preferences": "답변은 짧고 핵심 위주로, 코드 예시는 Python 기준",
        "rules": "외부 발송 작업은 반드시 승인 절차를 거친다",
    },
)

# ---------- 2) 회수 ----------
item = store.get(("users", user_id), "profile")
print("[get]", item.value)

# 검색(search): namespace 안에서 조건이나 유사도로 찾기
results = store.search(("users", user_id))
for r in results:
    print("[search]", r.key, "->", r.value["name"])


# ---------- 3) 그래프에 연결 ----------
def chat_node(state: MessagesState, config, *, store):
    uid = config["configurable"]["user_id"]
    profile = store.get(("users", uid), "profile")
    memory_text = str(profile.value) if profile else "저장된 기억 없음"

    messages = [
        SystemMessage(content=f"[사용자 장기 기억]\n{memory_text}\n"
                              f"기억을 반영해 답하세요."),
        *state["messages"],
    ]
    return {"messages": [llm.invoke(messages)]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# compile에 checkpointer(단기)와 store(장기)를 함께 전달합니다
graph = builder.compile(checkpointer=InMemorySaver(), store=store)

# ---------- 4) 서로 다른 세션에서 기억이 유지되는지 확인 ----------
for thread in ["thread-1", "thread-2"]:   # thread가 달라도 (= 새 세션)
    config = {"configurable": {"thread_id": thread, "user_id": user_id}}
    out = graph.invoke(
        {"messages": [HumanMessage(content="보고서 메일 초안 쓸 때 주의할 점 알려줘")]},
        config,
    )
    print(f"\n[{thread}] {get_text(out['messages'][-1])[:120]}")
