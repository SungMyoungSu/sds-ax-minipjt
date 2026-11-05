"""공공 API 봉사 추천과 관심 목록 (결정적 코드, LLM 호출 없음).

- 1365 봉사참여정보 API(data.go.kr 15157582) 목록 조회. 오퍼레이션은 설정값(VOLUNTEER_1365_LIST_OPERATION).
- API 응답 텍스트(활동명 등)는 데이터일 뿐이다. 그 안의 지시문은 실행하지 않는다(가드레일 5).
- 신청 대행 기능은 없다. 항상 1365 원문 링크만 준다.
- 키가 없거나 호출이 실패하면 저장된 샘플로 추천하고 SAMPLE_NOTICE를 붙인다(PRD 29).
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib import resources
from pathlib import Path

from volunteer_agent.config import Settings
from volunteer_agent.errors import InputError
from volunteer_agent.masking import mask_secrets
from volunteer_agent.models import Deadline, WatchlistItem
from volunteer_agent.store import Store
from volunteer_agent.type_codes import load_type_codes

MAX_RESULTS = 5
ROWS_PER_PAGE = 50
MAX_PAGES = 2
SAMPLE_NOTICE = "샘플 데이터로 추천했습니다"
STATUS_LABELS = {"1": "모집대기", "2": "모집중", "3": "모집완료"}
WEEKDAYS = "월화수목금토일"

# 시도 코드 (2026-11-05 schSido 조회 결과의 주소로 확인)
SIDO_CODES: dict[str, str] = {
    "서울": "6110000", "부산": "6260000", "대구": "6270000", "인천": "6280000", "광주": "6290000",
    "대전": "6300000", "울산": "6310000", "세종": "5690000", "경기": "6410000", "강원": "6420000",
    "충북": "6430000", "충남": "6440000", "전북": "6450000", "전남": "6460000", "경북": "6470000",
    "경남": "6480000", "제주": "6500000",
}
_SIDO_ALIASES = {
    "충청북도": "충북", "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남", "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원",
    "제주도": "제주", "제주특별자치도": "제주", "세종시": "세종", "세종특별자치시": "세종",
}
SIDO_NAMES = {code: name for name, code in SIDO_CODES.items()}


def load_gugun_codes() -> dict[str, dict[str, str]]:
    """시도 코드 → {시군구 이름: 시군구 코드}(resources/gugun_codes.json, 실제 응답으로 확인한 것만)."""
    text = resources.files("volunteer_agent").joinpath("resources", "gugun_codes.json").read_text(encoding="utf-8")
    return json.loads(text)["sido"]


_LOCAL_SUFFIXES = ("구", "시", "군")

# 희망 전공 → 관련 분야 코드(학생 추천 정렬용, PRD 27)
MAJOR_FIELDS: dict[str, list[str]] = {
    "간호": ["0500", "0100"], "의학": ["0500"], "의대": ["0500"], "보건": ["0500"], "약학": ["0500"],
    "물리치료": ["0500"], "사회복지": ["0100", "0300"], "복지": ["0100", "0300"],
    "교육": ["0400", "0300"], "사범": ["0400", "0300"], "유아": ["0400", "0300"], "심리": ["0300"],
    "상담": ["0300"], "환경": ["0800"], "생명": ["0800"], "생태": ["0800"], "체육": ["0700"],
    "예술": ["0700"], "미술": ["0700"], "음악": ["0700"], "관광": ["0700"], "건축": ["0200"],
    "행정": ["0900"], "경찰": ["1000"], "소방": ["1200", "1000"], "법": ["1100"], "국제": ["1300"],
    "외국어": ["1300", "0400"], "농업": ["0600"],
}


class ApiError(Exception):
    """API를 쓸 수 없는 이유. 메시지는 원인과 해결 방법을 담고 비밀 값은 가려져 있다."""


# ---- 데이터 ----

@dataclass
class Program:
    program_id: str
    title: str
    organization: str
    place: str
    sido_code: str
    field_name: str
    recruit_start: date | None
    recruit_end: date | None
    program_start: date | None
    program_end: date | None
    start_hour: int | None
    end_hour: int | None
    status: str
    youth_ok: bool
    url: str
    source_api: str = "1365"
    gugun_code: str = ""  # 시군구 코드(gugunCd)

    @property
    def field_code(self) -> str | None:
        return load_type_codes().code_for_name(self.field_name)

    @property
    def region(self) -> str:
        return SIDO_NAMES.get(self.sido_code, self.sido_code)

    @property
    def expected_hours(self) -> int | None:
        if self.start_hour is None or self.end_hour is None or self.end_hour <= self.start_hour:
            return None
        return self.end_hour - self.start_hour

    def dates(self, limit: int = 31) -> list[date]:
        if not self.program_start:
            return []
        end = self.program_end or self.program_start
        days = min((end - self.program_start).days, limit)
        return [self.program_start + timedelta(days=i) for i in range(days + 1)]


@dataclass
class SearchQuery:
    region: str | None = None  # "서울", "서울 동작구", "동작구" 등. 시도와 시군구 단위로 거른다
    field_codes: list[str] = field(default_factory=list)  # 비어 있으면 분야 제한 없음
    company_rules_applied: bool = False  # field_codes가 확정된 회사 인정 분야에서 왔는지(추천 이유 표시용)
    start: date | None = None  # 활동 기간 조건
    end: date | None = None
    youth_only: bool = False  # 학생은 청소년 참여 가능 활동만 (PRD 25)
    weekdays: list[int] | None = None  # 0=월 … 6=일. 예: 주말 [5, 6]
    keyword: str | None = None
    major: str | None = None  # 학생 희망 전공 → 관련 분야를 먼저 (PRD 27)
    area_label: str | None = None  # 해석한 지역 이름(예: 서울 동작구), 추천 이유·안내용
    rules_unconfirmed: bool = False  # 회사 규칙 미확정: 회사 인정 분야 필터 없이 마감 임박 순, 안내 표시 (미결 결정 4)


@dataclass
class Recommendation:
    program: Program
    reason: str

    def format(self, n: int) -> str:
        p = self.program
        when = _fmt_dates(p.program_start, p.program_end)
        if p.start_hour is not None and p.end_hour is not None:
            when += f" {p.start_hour}시~{p.end_hour}시"
        hours = f"{p.expected_hours}시간" if p.expected_hours else "확인 필요"
        return (
            f"{n}. {p.title} [1365 번호 {p.program_id}]\n"
            f"   기관: {p.organization} | 일시: {when} | 장소: {p.place}\n"
            f"   모집 마감: {_fmt_date(p.recruit_end)} | 예상 시간: {hours} | 분야: {p.field_name}\n"
            f"   추천 이유: {self.reason}\n"
            f"   1365 원문: {p.url}"
        )


@dataclass
class SearchResult:
    items: list[Recommendation] = field(default_factory=list)
    sample: bool = False
    error: str | None = None  # API를 못 쓴 이유(마스킹됨)
    suggestions: list[str] = field(default_factory=list)  # 0건일 때 조건 완화 제안
    notes: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        parts = []
        if self.sample:
            parts.append(f"{SAMPLE_NOTICE}. ({self.error})" if self.error else f"{SAMPLE_NOTICE}.")
        parts += self.notes
        if self.items:
            parts += [r.format(i) for i, r in enumerate(self.items, 1)]
            parts.append("신청은 각 1365 원문 링크에서 직접 해 주세요.")
        else:
            parts.append("조건에 맞는 봉사를 찾지 못했습니다.")
            parts += [f"- {s}" for s in self.suggestions]
        return "\n".join(parts)


# ---- 파싱 ----

def _text(el: ET.Element, tag: str) -> str:
    found = el.find(tag)
    return (found.text or "").strip() if found is not None and found.text else ""


def _date(value: str) -> date | None:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 8:
        return None
    try:
        return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
    except ValueError:
        return None


def _hour(value: str) -> int | None:
    return int(value) if value.isdigit() and 0 <= int(value) <= 24 else None


def _fmt_date(d: date | None) -> str:
    return f"{d.year}-{d.month:02d}-{d.day:02d}({WEEKDAYS[d.weekday()]})" if d else "확인 필요"


def _fmt_dates(a: date | None, b: date | None) -> str:
    return _fmt_date(a) if not b or a == b else f"{_fmt_date(a)}~{_fmt_date(b)}"


def parse_programs(xml_text: str) -> tuple[list[Program], int]:
    """1365 목록 응답을 파싱한다. 오류 응답이면 ApiError."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        raise ApiError("1365 API 응답을 해석할 수 없습니다. 잠시 후 다시 시도해 주세요.") from None
    if root.tag == "OpenAPI_ServiceResponse":  # data.go.kr 게이트웨이 오류
        reason = _text(root, "cmmMsgHeader/returnAuthMsg") or _text(root, "cmmMsgHeader/errMsg")
        code = _text(root, "cmmMsgHeader/returnReasonCode")
        hint = (" .env의 DATA_GO_KR_KEY에 data.go.kr '일반 인증키(Decoding)'를 넣었는지, 활용신청이 승인됐는지 확인해 주세요."
                if code in {"30", "31", "32", "20"} else " 잠시 후 다시 시도해 주세요.")
        raise ApiError(f"1365 API 오류({code} {reason}).{hint}")
    code = _text(root, "header/resultCode")
    if code and code != "00":
        msg = _text(root, "header/resultMsg")
        hint = " 하루 호출 한도를 넘었을 수 있습니다. 내일 다시 시도해 주세요." if code == "22" else " 잠시 후 다시 시도해 주세요."
        raise ApiError(f"1365 API 오류({code} {msg}).{hint}")
    total = int(_text(root, "body/totalCount") or 0)
    programs = []
    for it in root.iter("item"):
        pid = _text(it, "progrmRegistNo")
        if not pid:
            continue
        programs.append(Program(
            program_id=pid, title=_text(it, "progrmSj"), organization=_text(it, "nanmmbyNm"),
            place=_text(it, "actPlace"), sido_code=_text(it, "sidoCd"), field_name=_text(it, "srvcClCode"),
            recruit_start=_date(_text(it, "noticeBgnde")), recruit_end=_date(_text(it, "noticeEndde")),
            program_start=_date(_text(it, "progrmBgnde")), program_end=_date(_text(it, "progrmEndde")),
            start_hour=_hour(_text(it, "actBeginTm")), end_hour=_hour(_text(it, "actEndTm")),
            status=_text(it, "progrmSttusSe"), youth_ok=_text(it, "yngbgsPosblAt") == "Y",
            gugun_code=_text(it, "gugunCd"),
            url=_text(it, "url") or f"https://www.1365.go.kr/vols/P9210/partcptn/timeCptn.do?type=show&progrmRegistNo={pid}",
        ))
    return programs, total


# ---- 상대 날짜 (fixture·샘플) ----

_TOKEN = re.compile(r"\{(today|sat|sun)([+-]\d+)?\}")


def render_relative_dates(text: str, today: date) -> str:
    """{today+N}: 오늘+N일, {sat+N}/{sun+N}: 다가오는 토/일요일 + N주 (미결 결정 8)."""
    def repl(m: re.Match) -> str:
        kind, n = m.group(1), int(m.group(2) or 0)
        if kind == "today":
            d = today + timedelta(days=n)
        else:
            target = 5 if kind == "sat" else 6
            d = today + timedelta(days=(target - today.weekday()) % 7 + 7 * n)
        return d.strftime("%Y%m%d")
    return _TOKEN.sub(repl, text)


def load_sample_programs(today: date) -> list[Program]:
    text = resources.files("volunteer_agent").joinpath("resources", "sample_programs.xml").read_text(encoding="utf-8")
    return parse_programs(render_relative_dates(text, today))[0]


# ---- 클라이언트 ----

Fetcher = Callable[[str, dict[str, str], float], str]


def http_fetch(url: str, params: dict[str, str], timeout: float) -> str:
    full = f"{url}?{urllib.parse.urlencode(params)}"  # 서비스 키는 urlencode가 인코딩
    with urllib.request.urlopen(full, timeout=timeout) as r:  # noqa: S310 - apis.data.go.kr만
        return r.read().decode("utf-8", "replace")


class FixtureFetcher:
    """녹화된 응답(tests/fixtures/api)을 돌려준다. pageNo별 파일, 없으면 빈 응답."""

    def __init__(self, folder: str | Path, today: date, prefix: str = "1365"):
        self.folder, self.today, self.prefix = Path(folder), today, prefix
        self.calls: list[dict[str, str]] = []

    def __call__(self, url: str, params: dict[str, str], timeout: float) -> str:
        self.calls.append(dict(params))
        path = self.folder / f"{self.prefix}_page{params.get('pageNo', '1')}.xml"
        if not path.is_file():
            path = self.folder / f"{self.prefix}_empty.xml"
        return render_relative_dates(path.read_text(encoding="utf-8"), self.today)


class VolunteerClient:
    def __init__(self, settings: Settings, fetcher: Fetcher | None = None):
        self.settings = settings
        self.fetcher = fetcher or http_fetch

    def list_programs(self, page: int = 1, rows: int = ROWS_PER_PAGE, field_code: str | None = None,
                      sido_code: str | None = None, youth_only: bool = False,
                      keyword: str | None = None, gugun_code: str | None = None) -> tuple[list[Program], int]:
        s = self.settings
        if not s.data_go_kr_key:
            raise ApiError("DATA_GO_KR_KEY가 설정되지 않았습니다. data.go.kr에서 1365 봉사참여정보 API 활용신청 후 "
                           "'일반 인증키(Decoding)'를 .env의 DATA_GO_KR_KEY에 넣어 주세요.")
        if not s.volunteer_1365_base_url:
            raise ApiError("VOLUNTEER_1365_BASE_URL이 설정되지 않았습니다. .env.example을 참고해 .env에 넣어 주세요.")
        params = {"serviceKey": s.data_go_kr_key, "numOfRows": str(rows), "pageNo": str(page)}
        if field_code:
            params["upperClCode"] = field_code
        if sido_code:
            params["schSido"] = sido_code
        if gugun_code:
            params["schSign1"] = gugun_code  # 시군구 (2026-11-05 실제 호출로 확인)
        if youth_only:
            params["yngbgsPosblAt"] = "Y"
        if keyword:
            params["keyword"] = keyword
        url = f"{s.volunteer_1365_base_url.rstrip('/')}/{s.volunteer_1365_list_operation}"
        try:
            text = self.fetcher(url, params, s.data_go_kr_timeout)
        except (TimeoutError, socket.timeout):
            raise ApiError(f"1365 API 응답이 {s.data_go_kr_timeout:g}초 안에 오지 않았습니다. 잠시 후 다시 시도해 주세요.") from None
        except urllib.error.HTTPError as e:
            raise ApiError(f"1365 API가 HTTP {e.code} 오류를 돌려줬습니다. 오퍼레이션 경로(VOLUNTEER_1365_LIST_OPERATION)와 "
                           "활용신청 승인 여부를 확인해 주세요.") from None
        except urllib.error.URLError as e:
            reason = e.reason if isinstance(e.reason, str) else type(e.reason).__name__
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise ApiError(f"1365 API 응답이 {s.data_go_kr_timeout:g}초 안에 오지 않았습니다. 잠시 후 다시 시도해 주세요.") from None
            raise ApiError(mask_secrets(f"1365 API에 연결하지 못했습니다({reason}). 인터넷 연결이나 프록시 설정을 확인해 주세요.",
                                        s.secret_values())) from None
        except Exception as e:  # noqa: BLE001
            raise ApiError(mask_secrets(f"1365 API 호출 중 오류가 났습니다({type(e).__name__}: {e}).",
                                        s.secret_values())) from None
        return parse_programs(text)


# ---- 추천 ----

@dataclass
class Region:
    sido: str | None = None  # 시도 코드
    gugun: str | None = None  # 시군구 코드(코드표에 있을 때)
    local_name: str | None = None  # 사용자가 말한 시군구 이름(예: 동작구)

    @property
    def label(self) -> str | None:
        parts = [SIDO_NAMES.get(self.sido, "") if self.sido else "", self.local_name or ""]
        return " ".join(p for p in parts if p) or None


# 여러 시도에 같은 이름이 있는 시군구(시도 없이 말하면 되묻는다)
AMBIGUOUS_LOCAL = {"중구", "동구", "서구", "남구", "북구", "강서구"}


def resolve_region(region: str | None) -> Region:
    """'서울 동작구', '동작구', '부산 해운대구', '서울' → 시도·시군구. 시군구는 코드표에 있으면 코드로,
    없으면 이름만 남겨 활동 장소·기관 이름으로 거른다. 시도를 모르면 InputError."""
    if not region or region.strip() in {"상관없음", "전국", "어디든"}:
        return Region()
    words = re.split(r"\s+", region.strip())
    local = next((w for w in words if len(w) >= 2 and w.endswith(_LOCAL_SUFFIXES) and resolve_sido_word(w) is None),
                 None)
    sido = next((c for c in (resolve_sido_word(w) for w in words) if c), None)
    table = load_gugun_codes()
    if sido is None and local:
        owners = [code for code, gus in table.items() if local in gus]
        if local in AMBIGUOUS_LOCAL or len(owners) != 1:
            raise InputError(f"어느 시도의 {local}인가요? 시도 이름과 함께 알려 주세요(예: 서울 동작구, 경기 수원시).")
        sido = owners[0]
    if sido is None:
        raise InputError(f"지역을 알 수 없습니다({region}). 시도 이름(예: 서울, 경기, 부산)으로 알려 주세요.")
    gugun = table.get(sido, {}).get(local) if local else None
    return Region(sido=sido, gugun=gugun, local_name=local)


def resolve_sido_word(word: str) -> str | None:
    """'서울', '서울시', '서울특별시', '경상남도' → 시도 코드. 시군구 이름('동작구', '수원시')이면 None."""
    key = _SIDO_ALIASES.get(word, word)
    for name, code in SIDO_CODES.items():
        if key.startswith(name) and key[len(name):] in ("", "시", "특별시", "광역시", "특별자치시", "도", "특별자치도"):
            return code
    return None


def resolve_sido(region: str | None) -> str | None:
    """지역 표현 → 시도 코드(시군구만 말해도 코드표로 시도를 찾는다). 모르면 InputError."""
    return resolve_region(region).sido


def _local_stem(name: str) -> str:
    """'동작구' → '동작'(장소 문자열에 '동작구'·'동작동'처럼 나와도 맞도록). 두 글자 이름은 그대로."""
    return name[:-1] if len(name) > 2 else name


def major_codes(major: str | None) -> list[str]:
    if not major:
        return []
    codes: list[str] = []
    for key, cs in MAJOR_FIELDS.items():
        if key in major:
            codes += [c for c in cs if c not in codes]
    return codes


def _eligible(p: Program, q: SearchQuery, sido: str | None, today: date, area: Region | None = None) -> bool:
    if p.status == "3":  # 모집완료 = 정원이 참
        return False
    if p.recruit_end is None or p.recruit_end < today:
        return False
    if sido and p.sido_code != sido:
        return False
    if area and area.gugun and p.gugun_code != area.gugun:
        return False
    if area and area.local_name and not area.gugun and _local_stem(area.local_name) not in f"{p.place} {p.organization} {p.title}":
        return False
    if q.field_codes and p.field_code not in q.field_codes:
        return False
    if q.youth_only and not p.youth_ok:
        return False
    if q.start and p.program_end and p.program_end < q.start:
        return False
    if q.end and p.program_start and p.program_start > q.end:
        return False
    if q.weekdays is not None and not any(d.weekday() in q.weekdays for d in p.dates()):
        return False
    if q.keyword and q.keyword not in p.title:
        return False
    return True


def _reason(p: Program, q: SearchQuery, related: list[str], today: date) -> str:
    bits = []
    if p.field_code in related:
        bits.append(f"희망 전공({q.major})과 관련된 {p.field_name} 분야")
    elif q.company_rules_applied:
        bits.append(f"회사 인정 분야({p.field_name})")
    else:
        bits.append(f"{p.field_name} 분야")
    if q.region:
        bits.append(f"{q.area_label or p.region} 지역")
    if q.weekdays is not None:
        bits.append("가능한 요일에 진행")
    if p.recruit_end:
        bits.append(f"모집 마감 D-{(p.recruit_end - today).days}")
    return ", ".join(bits)


def _suggestions(q: SearchQuery) -> list[str]:
    out = []
    if q.weekdays is not None:
        out.append("요일 조건을 빼고 다시 찾아볼까요?")
    if q.region:
        out.append(f"{q.area_label or q.region} 말고 가까운 다른 지역이나 지역 상관없이 찾아볼까요?")
    if q.start or q.end:
        out.append("활동 기간을 넓혀 볼까요?")
    if q.keyword:
        out.append("검색어를 빼고 찾아볼까요?")
    if q.company_rules_applied:
        out.append("인정 분야 밖의 봉사도 보여 드릴까요? (회사 집계에는 들어가지 않습니다)")
    elif q.field_codes:
        out.append("분야 조건을 빼고 찾아볼까요?")
    return out or ["조건을 조금 넓혀서 다시 찾아볼까요?"]


def search_volunteer(client: VolunteerClient, query: SearchQuery, today: date,
                     limit: int = MAX_RESULTS) -> SearchResult:
    """조건에 맞는 봉사를 최대 limit건 추천한다. API를 못 쓰면 샘플로 추천한다."""
    area = resolve_region(query.region)
    sido = area.sido
    query.area_label = area.label
    related = major_codes(query.major)
    filter_codes = query.field_codes
    result = SearchResult()
    programs: dict[str, Program] = {}
    try:
        for code in filter_codes or [None]:
            fetched = 0
            for page in range(1, MAX_PAGES + 1):
                items, total = client.list_programs(page=page, field_code=code, sido_code=sido,
                                                    youth_only=query.youth_only, keyword=query.keyword,
                                                    gugun_code=area.gugun)
                for p in items:
                    programs.setdefault(p.program_id, p)
                fetched += len(items)
                if not items or fetched >= total:
                    break
    except ApiError as e:
        result.sample, result.error = True, str(e)
        programs = {p.program_id: p for p in load_sample_programs(today)}

    if query.rules_unconfirmed:
        result.notes.append("회사 규칙이 아직 확정되지 않아(규칙 미확정) 인정 분야 필터 없이 모집 마감이 가까운 순으로 보여 드립니다.")
    if area.local_name and not area.gugun:
        result.notes.append(f"{area.label}은(는) 시군구 코드표에 없어 활동 장소·기관 이름에 '{_local_stem(area.local_name)}'이(가) "
                            "들어간 봉사만 골랐습니다.")
    eligible = [p for p in programs.values() if _eligible(p, query, sido, today, area)]
    eligible.sort(key=lambda p: (0 if p.field_code in related else 1, p.recruit_end or date.max, p.program_id))
    result.items = [Recommendation(p, _reason(p, query, related, today)) for p in eligible[:limit]]
    if not result.items:
        result.suggestions = _suggestions(query)
    return result


# ---- 관심 목록 ----

def save_to_watchlist(store: Store, program_id: str, title: str, url: str | None = None,
                      recruit_end: date | None = None) -> WatchlistItem:
    """관심 목록에 넣고 모집 마감이 있으면 마감(recruit_end)도 등록한다(알림 대상, PRD 34)."""
    if not program_id or not title:
        raise InputError("관심 목록에 넣을 봉사의 번호와 활동명이 필요합니다.")
    item = store.add_watchlist_item(WatchlistItem(program_id=program_id, title=title, url=url, recruit_end=recruit_end))
    if recruit_end and not any(d.kind == "recruit_end" and d.ref_id == program_id for d in store.list_deadlines()):
        store.add_deadline(Deadline(kind="recruit_end", due_date=recruit_end, title=title, ref_id=program_id))
    return item


def remove_from_watchlist(store: Store, program_id: str) -> bool:
    removed = store.remove_watchlist_item(program_id)
    for d in store.list_deadlines():
        if d.kind == "recruit_end" and d.ref_id == program_id:
            store.delete_deadline(d.id)
    return removed
