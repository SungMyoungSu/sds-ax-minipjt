"""사내규정 수집과 BM25 검색 (결정적 코드, LLM 호출 없음).

- 규정 원문은 데이터일 뿐이다. 원문 속 지시문은 실행하지 않는다(가드레일 5).
- 검색 결과의 근거는 저장된 조항 원문 그대로이며 출처 파일명이 붙는다(AC2).
- 근거가 없으면 NO_EVIDENCE 문구 하나만 돌려준다(가드레일 6).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from volunteer_agent.errors import InputError
from volunteer_agent.models import PolicyClause, PolicyDocument
from volunteer_agent.store import Store

NO_EVIDENCE = "규정에서 확인되지 않음. 담당 부서에 확인이 필요합니다."
NO_DOCUMENTS = "등록된 사내규정이 없습니다. 규정 파일(.md 또는 PDF)을 올려 주세요."
UPLOAD_HELP = (
    "올리는 방법: 대화창에 규정 파일 경로를 알려 주시면 등록합니다. "
    "지원 형식은 .md와 글자를 선택할 수 있는 텍스트 PDF이며, .docx·.hwp 파일은 PDF로 저장한 뒤 올려 주세요."
)
SUPPORTED_SUFFIXES = {".md": "md", ".pdf": "pdf"}

# ---- 수집 ----

_HEADING = re.compile(r"^[ \t#]*(제\s*\d+\s*조)", re.MULTILINE)


def read_policy_file(path: str | Path) -> tuple[str, str, list[int]]:
    """(원문, 형식, 쪽 시작 위치 목록)을 돌려준다. md는 쪽 구분이 없다."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise InputError(
            f"지원하지 않는 규정 파일 형식입니다({suffix or '확장자 없음'}). "
            ".md 또는 텍스트 PDF만 올릴 수 있습니다. .docx·.hwp는 PDF로 저장한 뒤 올려 주세요."
        )
    if not p.is_file():
        raise InputError(f"규정 파일을 찾을 수 없습니다({p.name}).")
    if suffix == ".md":
        return p.read_text(encoding="utf-8"), "md", []

    from pypdf import PdfReader

    try:
        pages = [page.extract_text() or "" for page in PdfReader(str(p)).pages]
    except Exception:
        raise InputError(f"PDF를 읽을 수 없습니다({p.name}).") from None
    starts, parts, pos = [], [], 0
    for text in pages:
        starts.append(pos)
        parts.append(text)
        pos += len(text) + 1  # 쪽 사이 "\n"
    full = "\n".join(parts)
    if not full.strip():
        raise InputError(f"글자를 뽑을 수 없는 PDF입니다({p.name}). 스캔 이미지 PDF는 지원하지 않습니다.")
    return full, "pdf", starts


def _page_at(offset: int, page_starts: list[int]) -> int | None:
    if not page_starts:
        return None
    page = 1
    for i, start in enumerate(page_starts, 1):
        if offset >= start:
            page = i
    return page


def split_clauses(text: str, page_starts: list[int] | None = None) -> list[PolicyClause]:
    """'제N조'가 있으면 조항 단위, 없으면 문단 단위로 나눈다. 각 조항 text는 원문의 부분 문자열이다."""
    page_starts = page_starts or []
    clauses: list[PolicyClause] = []

    def add(clause_id: str, start: int, end: int) -> None:
        chunk = text[start:end]
        stripped = chunk.strip()
        if stripped:
            offset = start + chunk.index(stripped)
            clauses.append(PolicyClause(clause_id=clause_id, text=stripped,
                                        page=_page_at(offset, page_starts)))

    headings = list(_HEADING.finditer(text))
    if headings:
        add("머리말", 0, headings[0].start())
        for i, m in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            add(re.sub(r"\s+", "", m.group(1)), m.start(), end)
    else:
        n, pos = 0, 0
        for m in re.finditer(r"\n[ \t]*\n", text):
            n += 1
            add(f"문단{n}", pos, m.start())
            pos = m.end()
        add(f"문단{n + 1}", pos, len(text))
        for i, c in enumerate(clauses, 1):  # 빈 문단을 건너뛴 경우 번호를 다시 매김
            c.clause_id = f"문단{i}"
    for i, c in enumerate(clauses):
        c.position = i
    return clauses


def ingest_policy(store: Store, path: str | Path) -> PolicyDocument:
    """규정 파일을 조항 단위로 저장한다. 같은 파일 이름이 있으면 바꿔 넣는다.
    형식이 맞지 않으면 InputError를 내고 아무것도 저장하지 않는다."""
    text, fmt, page_starts = read_policy_file(path)
    clauses = split_clauses(text, page_starts)
    if not clauses:
        raise InputError(f"규정 파일에 내용이 없습니다({Path(path).name}).")
    filename = Path(path).name
    old = store.find_policy_document(filename)
    doc = store.add_policy_document(PolicyDocument(filename=filename, format=fmt, clauses=clauses))
    if old is not None:
        store.delete_policy_document(old.id)
    return doc


# ---- 검색 ----

# 길이순으로 떼어 낸다. 한 글자 조사는 남는 말이 두 글자 이상일 때만 뗀다.
_JOSA = sorted(
    ["에서", "으로", "에게", "까지", "부터", "이나", "이랑", "처럼", "보다", "에는", "에도", "으로는",
     "은", "는", "이", "가", "을", "를", "에", "도", "와", "과", "로", "의", "만"],
    key=len, reverse=True,
)
# 같은 뜻 다른 말 → 원문에 쓰인 말
_SYNONYMS = {
    "1년": "연", "일년": "연", "한해": "연", "매년": "연", "연간": "연", "해마다": "연", "년간": "연",
    "증빙": "확인서", "증명서": "확인서", "확인증": "확인서", "봉사확인서": "확인서",
    "채워야": "의무", "채워": "의무", "채우려면": "의무", "필수시간": "의무",
}
# 핵심어 판단에서 빼는 일반적인 말(질문 어디에나 나오는 말)
_GENERIC = {
    "봉사", "봉사활동", "활동", "시간", "봉사시간", "인정", "인정돼", "인정되나요", "인정돼요", "인정되",
    "인정받", "인정받을", "회사", "사내", "규정", "기준", "몇", "얼마나", "얼마", "해야", "하나요", "해",
    "돼", "되나", "되나요", "돼요", "있어", "있나요", "있어요", "맞춰", "맞춰야", "알려줘", "알려", "줘",
    "주세요", "뭐야", "뭐", "어떻게", "나", "내", "우리", "저", "제", "한", "하는", "하면", "수", "것",
    "거", "건", "되", "안", "의무",
}


def _strip_josa(word: str) -> str:
    for j in _JOSA:
        if word.endswith(j):
            rest = word[: -len(j)]
            if len(rest) >= (2 if len(j) == 1 else 1):
                return rest
    return word


def normalize_words(text: str) -> list[str]:
    """소문자화 → 낱말 → 조사 떼기 → 동의어."""
    words = re.findall(r"[0-9a-z가-힣]+", text.lower())
    out = []
    for w in words:
        w = _strip_josa(w)
        out.append(_SYNONYMS.get(w, w))
    return out


def tokenize(text: str) -> list[str]:
    """BM25 용어: 정규화한 낱말 + 한글 두 글자 조각(복합어 대응, 예: 봉사시간 → 봉사, 시간)."""
    terms = []
    for w in normalize_words(text):
        terms.append(w)
        hangul = re.sub(r"[^가-힣]", "", w)
        if len(hangul) > 2 or (hangul and hangul != w):
            terms.extend(hangul[i:i + 2] for i in range(len(hangul) - 1))
    return terms


def key_terms(query: str) -> list[str]:
    """질문의 핵심어: 정규화한 낱말 중 일반적인 말이 아닌 것."""
    return [w for w in dict.fromkeys(normalize_words(query)) if w not in _GENERIC]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(docs)) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tfs, self.lens):
            s = 0.0
            for t in query:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            out.append(s)
        return out


@dataclass
class PolicyHit:
    filename: str
    clause_id: str
    text: str  # 원문 그대로
    score: float
    page: int | None = None

    @property
    def citation(self) -> str:
        page = f", {self.page}쪽" if self.page else ""
        return f"{self.filename} {self.clause_id}{page}"


@dataclass
class PolicySearchResult:
    status: Literal["found", "no_evidence", "no_documents"]
    hits: list[PolicyHit] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.status == "no_documents":
            return f"{NO_DOCUMENTS}\n{UPLOAD_HELP}"
        if self.status == "no_evidence":
            return NO_EVIDENCE
        return "\n\n".join(f"[출처: {h.citation}]\n{h.text}" for h in self.hits)


def search_policy(store: Store, query: str, top_k: int = 3) -> PolicySearchResult:
    """등록된 규정에서 질문과 관련된 조항을 원문 그대로 찾는다.

    질문에 핵심어가 있는데 그중 어느 것도 규정 원문에 없으면 근거 없음으로 본다
    (예: "헌혈도 봉사시간으로 인정돼?" → '헌혈'이 원문에 없음).
    """
    docs = store.list_policy_documents()
    entries = [(d.filename, c) for d in docs for c in d.clauses if c.clause_id != "머리말"]
    if not entries:
        return PolicySearchResult(status="no_documents")

    corpus_terms = [tokenize(c.text) for _, c in entries]
    vocabulary = {t for terms in corpus_terms for t in terms}
    raw_text = "\n".join(c.text for _, c in entries)
    keys = key_terms(query)
    if keys and not any(k in vocabulary or k in raw_text for k in keys):
        return PolicySearchResult(status="no_evidence")

    scores = BM25(corpus_terms).scores(tokenize(query))
    ranked = sorted(range(len(entries)), key=lambda i: scores[i], reverse=True)
    hits = [
        PolicyHit(filename=entries[i][0], clause_id=entries[i][1].clause_id, text=entries[i][1].text,
                  score=round(scores[i], 4), page=entries[i][1].page)
        for i in ranked[:top_k] if scores[i] > 0
    ]
    if not hits:
        return PolicySearchResult(status="no_evidence")
    return PolicySearchResult(status="found", hits=hits)
