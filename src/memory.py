"""장기 기억: LangGraph Store(로컬 SQLite)에 사용자 선호를 저장한다.

namespace ("users", user_id), key "preferences". 값은 지역, 가능한 요일, 관심 분야, 희망 전공.
save_preference·get_preferences는 MCP가 아닌 Python 도구이고 recommend_agent에만 바인딩한다.
Store 파일은 DATA_DIR 안에 있으므로 purge로 함께 지워진다.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from config import Settings
from errors import InputError
from recommend import resolve_region
from type_codes import load_type_codes, map_policy_terms

MEMORY_FILENAME = "memory.sqlite"
USER_ID = "local"  # PC 한 대에 사용자 한 명
NAMESPACE = ("users", USER_ID)
KEY = "preferences"
WEEKDAYS = "월화수목금토일"


def open_store(settings: Settings) -> SqliteStore:
    conn = sqlite3.connect(settings.ensure_data_dir() / MEMORY_FILENAME, check_same_thread=False,
                           isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    return store


def load_preferences(store: BaseStore) -> dict[str, Any]:
    item = store.get(NAMESPACE, KEY)
    return dict(item.value) if item else {}


def save_preferences(store: BaseStore, region: str | None = None, available_days: list[int] | None = None,
                     interest_fields: list[str] | None = None, major: str | None = None) -> dict[str, Any]:
    """주어진 항목만 기존 선호에 합쳐 저장한다. 값은 검증한 뒤 저장한다."""
    prefs = load_preferences(store)
    if region is not None:
        resolve_region(region)  # 모르는 지역이면 InputError(시군구만 말한 '동작구'도 허용)
        prefs["region"] = region.strip()
    if available_days is not None:
        days = sorted({int(d) for d in available_days})
        if any(d < 0 or d > 6 for d in days):
            raise InputError("요일은 0(월)부터 6(일)까지의 숫자로 알려 주세요.")
        prefs["available_days"] = days
    if interest_fields is not None:
        codes, unknown = map_policy_terms(interest_fields)
        if unknown:
            raise InputError(f"1365 분야로 바꿀 수 없는 관심 분야입니다({', '.join(unknown)}).")
        prefs["interest_type_codes"] = codes
    if major is not None:
        prefs["major"] = major.strip()
    store.put(NAMESPACE, KEY, prefs)
    return prefs


def describe(prefs: dict[str, Any]) -> str:
    if not prefs:
        return "저장된 선호가 없습니다. 지역을 모르면 사용자에게 한 번 물어보세요."
    table = load_type_codes()
    parts = []
    if prefs.get("region"):
        parts.append(f"지역: {prefs['region']}")
    if prefs.get("available_days") is not None:
        days = prefs["available_days"]
        label = "주말" if days == [5, 6] else "평일" if days == [0, 1, 2, 3, 4] else ", ".join(WEEKDAYS[d] for d in days)
        parts.append(f"가능한 요일: {label} (weekdays={days})")
    if prefs.get("interest_type_codes"):
        parts.append("관심 분야: " + ", ".join(f"{table.name(c)}({c})" for c in prefs["interest_type_codes"]))
    if prefs.get("major"):
        parts.append(f"희망 전공: {prefs['major']}")
    return "저장된 선호 — " + " | ".join(parts)


def memory_tools(store: BaseStore) -> list[BaseTool]:
    """recommend_agent에 바인딩할 Python 도구 2개."""

    def save_preference(region: str | None = None, available_days: list[int] | None = None,
                        interest_fields: list[str] | None = None, major: str | None = None) -> str:
        """사용자의 봉사 선호(지역, 가능한 요일, 관심 분야, 희망 전공)를 장기 기억에 저장한다. 말한 항목만 넣는다.

        쓰는 경우: "나는 서울 강남구 살고 주말만 가능해", "간호학과 가고 싶어"처럼 다음 대화에서도 쓸 선호를 말할 때.
        쓰지 않는 경우: 이번 한 번만 쓰는 검색 조건(바로 search_volunteer에 넣는다), 활동 기록, 다른 사람의 정보.

        Args:
            region: 사는 지역(예: "서울 강남구")
            available_days: 가능한 요일 숫자 목록(0=월 … 6=일). 주말이면 [5, 6], 평일이면 [0, 1, 2, 3, 4]
            interest_fields: 관심 분야 이름(예: ["환경", "교육"])
            major: 희망 전공(학생)
        """
        try:
            prefs = save_preferences(store, region, available_days, interest_fields, major)
        except InputError as e:
            return str(e)
        return "선호를 저장했습니다. " + describe(prefs)

    def get_preferences() -> str:
        """장기 기억에 저장된 사용자 선호(지역, 가능한 요일, 관심 분야, 희망 전공)를 불러온다.

        쓰는 경우: 봉사를 추천하기 전에 항상 먼저 불러, 사용자가 이번에 말하지 않은 조건의 기본값으로 쓴다.
        쓰지 않는 경우: 활동 기록·집계 조회(record_agent 담당).
        """
        return describe(load_preferences(store))

    return [
        StructuredTool.from_function(func=save_preference, name="save_preference"),
        StructuredTool.from_function(func=get_preferences, name="get_preferences"),
    ]
