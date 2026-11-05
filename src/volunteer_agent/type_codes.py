"""1365 봉사분야 코드표(대분류). 목록에 없는 코드는 입력 오류로 처리한다(검증은 3단계)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources


# 사내규정에 쓰이는 분야 표현 → 1365 대분류 코드. 매핑은 후보일 뿐이고 사용자가 확정한다.
# '사회복지·복지'를 생활편의지원(0100)으로 보는 것은 잠정 매핑이다(7단계 코드 확인 때 함께 점검).
POLICY_TERM_TO_CODES: dict[str, list[str]] = {
    "환경": ["0800"], "환경보호": ["0800"], "환경정화": ["0800"],
    "사회복지": ["0100"], "복지": ["0100"], "생활편의": ["0100"], "생활편의지원": ["0100"],
    "주거": ["0200"], "주거환경": ["0200"], "상담": ["0300"],
    "교육": ["0400"], "멘토링": ["0400"], "보건": ["0500"], "의료": ["0500"], "보건의료": ["0500"],
    "농어촌": ["0600"], "문화": ["0700"], "문화행사": ["0700"], "행정": ["0900"], "행정보조": ["0900"],
    "안전": ["1000"], "방범": ["1000"], "공익": ["1100"], "인권": ["1100"],
    "재난": ["1200"], "재해": ["1200"], "국제": ["1300"], "국제협력": ["1300"], "해외": ["1300"],
}


def map_policy_terms(terms: list[str]) -> tuple[list[str], list[str]]:
    """규정 분야 표현을 코드로 바꾼다. (코드 목록, 매핑하지 못한 표현)을 돌려준다.
    이미 코드(예: "0800")거나 코드표의 분야 이름이면 그대로 받는다."""
    table = load_type_codes()
    codes: list[str] = []
    unknown: list[str] = []
    for raw in terms:
        term = raw.strip().removesuffix("분야").strip()
        found = (
            [term] if table.is_valid(term)
            else [table.code_for_name(term)] if table.code_for_name(term)
            else POLICY_TERM_TO_CODES.get(term.replace(" ", ""))
        )
        if found:
            codes.extend(c for c in found if c not in codes)
        else:
            unknown.append(raw)
    return codes, unknown


@dataclass(frozen=True)
class TypeCodeTable:
    source: str
    as_of: str | None
    verified: bool
    codes: dict[str, str]  # code -> name

    def name(self, code: str) -> str | None:
        return self.codes.get(code)

    def is_valid(self, code: str) -> bool:
        return code in self.codes

    def code_for_name(self, name: str) -> str | None:
        for code, n in self.codes.items():
            if n == name:
                return code
        return None


@lru_cache(maxsize=1)
def load_type_codes() -> TypeCodeTable:
    text = (
        resources.files("volunteer_agent")
        .joinpath("resources", "activity_type_codes.json")
        .read_text(encoding="utf-8")
    )
    data = json.loads(text)
    return TypeCodeTable(
        source=data["source"],
        as_of=data.get("as_of"),
        verified=bool(data.get("verified")),
        codes={c["code"]: c["name"] for c in data["codes"]},
    )
