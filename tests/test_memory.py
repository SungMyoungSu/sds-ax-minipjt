import asyncio
import re
import shutil
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from volunteer_agent.app import AppClock, ask, compile_app
from volunteer_agent.errors import InputError
from volunteer_agent.memory import MEMORY_FILENAME, describe, load_preferences, open_store, save_preferences
from test_app import ScriptedChat, StubTools, call, general_handler, is_handoff, last_human


@pytest.fixture
def mem(settings):
    store = open_store(settings)
    yield store
    store.conn.close()


# ---- 저장과 불러오기 ----

def test_save_merges_and_persists(settings, mem):
    save_preferences(mem, region="서울 강남구", available_days=[6, 5])
    save_preferences(mem, major="간호학", interest_fields=["환경", "보건의료"])
    prefs = load_preferences(mem)
    assert prefs == {"region": "서울 강남구", "available_days": [5, 6], "major": "간호학",
                     "interest_type_codes": ["0800", "0500"]}
    assert (settings.data_dir / MEMORY_FILENAME).is_file()

    other = open_store(settings)  # 새 연결(새 실행)에서도 남아 있음
    try:
        assert load_preferences(other) == prefs
    finally:
        other.conn.close()
    text = describe(prefs)
    assert "지역: 서울 강남구" in text and "주말 (weekdays=[5, 6])" in text and "희망 전공: 간호학" in text
    assert "환경ㆍ생태계보호(0800)" in text


@pytest.mark.parametrize("kwargs, msg", [
    (dict(region="아틀란티스"), "지역을 알 수 없습니다"),
    (dict(available_days=[7]), "요일"),
    (dict(interest_fields=["우주탐사"]), "관심 분야"),
])
def test_invalid_preferences_rejected(mem, kwargs, msg):
    with pytest.raises(InputError, match=msg):
        save_preferences(mem, **kwargs)
    assert load_preferences(mem) == {}


def test_empty_preferences_message(mem):
    assert "저장된 선호가 없습니다" in describe(load_preferences(mem))


# ---- AC18: 새 대화에서 선호 사용, purge 후 없음 ----

def supervisor_handler(messages):
    last = messages[-1]
    if isinstance(last, HumanMessage):
        return call("transfer_to_recommend_agent", i=len(messages))
    reply = next(m for m in reversed(messages) if isinstance(m, AIMessage) and m.name == "recommend_agent"
                 and not m.tool_calls)
    return AIMessage(content=reply.content)


def recommend_handler(messages):
    """LLM 흉내: 선호를 말하면 저장, 추천 요청이면 선호를 불러 그 값으로 검색."""
    last = messages[-1]
    if isinstance(last, ToolMessage) and not is_handoff(last):
        if last.name == "get_preferences":
            region = re.search(r"지역: ([^|]+?)(?: \||$)", last.content)
            major = re.search(r"희망 전공: ([^|]+?)(?: \||$)", last.content)
            args = {"region": region.group(1) if region else None, "major": major.group(1) if major else None}
            return call("search_volunteer", {k: v for k, v in args.items() if v}, len(messages))
        return AIMessage(content=f"결과: {last.content}")
    text = last_human(messages)
    if "살고" in text:
        return call("save_preference", {"region": "서울", "major": "간호학"}, len(messages))
    return call("get_preferences", {}, len(messages))


def test_ac18_preferences_used_in_new_thread_and_gone_after_purge(settings):
    stubs = StubTools()
    models = {"supervisor": ScriptedChat(handler=supervisor_handler, seen=[]),
              "recommend_agent": ScriptedChat(handler=recommend_handler, seen=[])}
    for role in ("policy_agent", "record_agent", "general_agent", "planner_agent"):
        models[role] = ScriptedChat(handler=general_handler, seen=[])
    clock = AppClock.create(settings, date(2026, 11, 4))

    store = open_store(settings)
    app = compile_app(lambda r: models[r], stubs.mcp(), clock, store=store)
    t1 = asyncio.run(ask(app, "고등학생이야. 서울 살고 간호학과 가고 싶어", "thread-1"))
    assert t1.tools == ["save_preference"] and "선호를 저장했습니다" in t1.answer
    store.conn.close()

    # 새 실행·새 대화: 단기 기억(checkpointer)은 비어 있고 장기 기억(Store)만 남아 있다
    store = open_store(settings)
    app = compile_app(lambda r: models[r], stubs.mcp(), clock, store=store)
    t2 = asyncio.run(ask(app, "봉사 추천해줘", "thread-2"))
    assert t2.agents == ["recommend_agent"]
    assert t2.tools == ["get_preferences", "search_volunteer"]
    seen_humans = [m.content for m in models["recommend_agent"].seen[-1] if isinstance(m, HumanMessage)]
    assert seen_humans == ["봉사 추천해줘"]
    assert ("search_volunteer", {"region": "서울", "major": "간호학"}) in stubs.calls
    store.conn.close()

    # purge: DATA_DIR 삭제 → 선호도 사라짐
    shutil.rmtree(settings.data_dir)
    store = open_store(settings)
    try:
        assert load_preferences(store) == {}
    finally:
        store.conn.close()
