"""trace_run.py - 6일차 그래프에 추적 붙이기"""
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from local_tracer import FileTracer, get_text
from final_scenario import build_app   # 6일차 checkpoint의 Supervisor 그래프를 만드는 함수

load_dotenv()
tracer = FileTracer("trace.jsonl")


async def main():
    # MCP 도구를 비동기로 가져오므로 그래프도 await로 만들고 ainvoke로 실행합니다
    app = await build_app()
    result = await app.ainvoke(
        {"messages": [HumanMessage(content="김하늘 님의 소속 팀과 지급 자산을 알려줘")]},
        {"configurable": {"thread_id": "trace-demo"}, "callbacks": [tracer]},
    )
    print(get_text(result["messages"][-1]))


asyncio.run(main())
