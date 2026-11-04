# final_scenario.py - 6일차 종합: RAG + MCP + Supervisor
import asyncio
import sys
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

load_dotenv()
MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

def get_text(message):
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content

def make_worker_llm():
    return ChatBedrockConverse(model=MODEL, region_name="us-east-1", temperature=0)

# ---------- research_agent: 2일차 RAG ----------
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1",
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="sds_policies",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def retrieve_docs(query: str) -> str:
    """삼성SDS 사내 규정 문서(연차, 재택근무, 출장 등)에서 관련 내용을 검색한다."""
    docs = retriever.invoke(query)
    if not docs:
        return "관련 문서를 찾지 못했습니다."
    return "\n\n".join(
        f"[출처: {d.metadata.get('source', '알 수 없음')}]\n{d.page_content}"
        for d in docs
    )

research_agent = create_react_agent(
    make_worker_llm(),
    [retrieve_docs],
    prompt=(
        "너는 삼성SDS 사내 규정 문서 검색 전문가다.\n"
        "반드시 retrieve_docs 결과에 근거해 답하고 출처를 표기하라.\n"
        "담당 범위 밖 질문에는 '담당 범위가 아닙니다'라고만 답하라."
    ),
    name="research_agent",
)

general_agent = create_react_agent(
    make_worker_llm(),
    [],
    prompt=(
        "너는 일반 응대 담당이다. 인사말과 잡담에 친절히 답하고,\n"
        "업무 질문은 '사내 데이터 조회와 규정 검색을 도와드릴 수 있습니다'로 안내하라."
    ),
    name="general_agent",
)

async def build_app():
    # ---------- data_agent: 4일차 MCP ----------
    client = MultiServerMCPClient(
        {
            "company": {
                "command": sys.executable,
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )
    mcp_tools = await client.get_tools()

    data_agent = create_react_agent(
        make_worker_llm(),
        mcp_tools,
        prompt=(
            "너는 삼성SDS 사내 데이터 조회 전문가다. 임직원, 자산, 프로젝트 조회만 담당한다.\n"
            "개인은 search_employee, 팀 명단과 인원수는 list_team_members를 쓴다.\n"
            "반드시 도구로 조회한 결과에 근거해 답하고, 한 문단으로 정리해 보고하라.\n"
            "담당 범위 밖 질문에는 '담당 범위가 아닙니다'라고만 답하라."
        ),
        name="data_agent",
    )

    # ---------- Supervisor 조립 ----------
    supervisor = create_supervisor(
        [data_agent, research_agent, general_agent],
        model=ChatBedrockConverse(model=MODEL, region_name="us-east-1", temperature=0),
        prompt=(
            "너는 삼성SDS 사내 어시스턴트의 Supervisor다.\n"
            "[배분 기준]\n"
            "- 임직원, 자산, 프로젝트 등 사내 시스템 데이터 조회는 data_agent\n"
            "- 연차, 재택근무, 출장 등 사내 규정 문서 검색은 research_agent\n"
            "- 그 외 인사말, 잡담은 general_agent\n"
            "[규칙]\n"
            "- 직접 답을 지어내지 말고 반드시 담당 Agent를 통해 확인하라.\n"
            "- 여러 요구가 섞인 질문은 각 Agent를 순서대로 호출해 모두 처리하라.\n"
            "- 마지막에는 문서 근거와 데이터를 함께 담아 하나의 보고로 종합하고,\n"
            "  결론(가능, 불가능, 조건부)을 명시하라."
        ),
    )
    return supervisor.compile()

async def main():
    app = await build_app()
    question = (
        "출장 규정에서 식비와 숙박비 근거를 찾고, 물류플랫폼팀 임직원 수와 "
        "진행 중인 프로젝트를 조회해서, 다음 분기 출장 예산 검토 보고를 종합해줘"
    )
    print(f"질문: {question}\n")

    async for event in app.astream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
        config={"recursion_limit": 30},
    ):
        for node, update in event.items():
            for m in (update or {}).get("messages", []):
                if getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        print(f"[{node}] 도구: {tc['name']}({tc.get('args', {})})")
                elif getattr(m, "content", None):
                    label = getattr(m, "name", None) or node
                    print(f"[{label}] {get_text(m)[:200]}")

if __name__ == "__main__":
    asyncio.run(main())
