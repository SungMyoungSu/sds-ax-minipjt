"""1365 봉사분야 코드표(대분류). 목록에 없는 코드는 입력 오류로 처리한다(검증은 3단계)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources


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
