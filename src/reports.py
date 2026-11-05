"""보고서 PDF: 회사 제출용 보고서와 학생 포트폴리오 (결정적 코드, 한글 폰트 임베드).

- 회사 보고서는 규칙이 확정된 뒤에만 만든다(G4). 확정 전에는 만들지 않고 안내 문구를 돌려준다.
- 학생 포트폴리오에는 대입 혜택을 약속하는 문구를 넣지 않는다(가드레일 7).
- 사용자가 입력한 텍스트(기관, 배운 점 등)는 이스케이프해서 PDF 마크업으로 해석되지 않게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import Settings
from errors import InputError
from models import Activity, Profile
from progress import STATUS_LABELS, Progress, compute_progress
from rules import company_profile, company_report_block
from store import Store
from type_codes import load_type_codes

ReportKind = Literal["employee_report", "student_portfolio"]
REPORTS_DIR = "reports"
FONT_NAME = "KoreanFont"
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/NanumGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/Library/Fonts/AppleGothic.ttf"),
]


@dataclass
class ReportResult:
    status: Literal["created", "blocked"]
    path: Path | None = None
    message: str = ""


def find_korean_font() -> Path | None:
    return next((p for p in FONT_CANDIDATES if p.is_file()), None)


def _register_font() -> str:
    if FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return FONT_NAME
    font = find_korean_font()
    if font is None:
        raise InputError("한글 폰트를 찾을 수 없어 PDF를 만들 수 없습니다. 맑은 고딕(malgun.ttf)이나 나눔고딕을 설치해 주세요.")
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(font)))  # TrueType 서브셋으로 PDF에 임베드됨
    return FONT_NAME


def _styles(font: str) -> dict[str, ParagraphStyle]:
    base = dict(fontName=font, wordWrap="CJK")
    return {
        "title": ParagraphStyle("title", fontSize=16, leading=22, spaceAfter=6, **base),
        "h": ParagraphStyle("h", fontSize=11, leading=15, spaceBefore=8, spaceAfter=4, **base),
        "body": ParagraphStyle("body", fontSize=9.5, leading=13, **base),
        "cell": ParagraphStyle("cell", fontSize=8.5, leading=11, **base),
    }


def _p(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text if text not in (None, "") else "-")), style)


def _h(x: float | None) -> str:
    return "-" if x is None else f"{x:g}시간"


def _period(prog: Progress) -> str:
    return f"{prog.period.start.isoformat()} ~ {prog.period.end.isoformat()}" if prog.period else "전체 기간"


def _table(rows: list[list[Paragraph]], widths: list[float], font: str, empty: bool = False) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    extra = [("SPAN", (0, 1), (-1, 1))] if empty else []  # 활동이 없을 때 안내 줄은 표 전체 너비
    t.setStyle(TableStyle(extra + [
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _field(code: str) -> str:
    return load_type_codes().name(code) or code


def _build(path: Path, story: list) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=path.stem, author="volunteer-agent")
    doc.build(story)


def _employee_story(profile: Profile, prog: Progress, acts: dict[int, Activity], today: date, font: str) -> list:
    st = _styles(font)
    story = [
        _p("회사 제출용 봉사활동 보고서", st["title"]),
        _p(f"작성일: {today.isoformat()}", st["body"]),
        _p(f"기준 기간: {_period(prog)}", st["body"]),
        _p(f"연 목표 시간: {_h(profile.goal_hours)} | 확인서 필수: {'예' if profile.certificate_required else '아니오'}", st["body"]),
        _p("요약", st["h"]),
        _p(f"인정 시간 합계: {_h(prog.recognized_hours)}", st["body"]),
        _p(f"확인서 대기 시간: {_h(prog.pending_proof_hours)}", st["body"]),
        _p(f"제외 시간(비인정 분야): {_h(prog.excluded_hours)}", st["body"]),
        _p(f"남은 시간: {_h(prog.remaining_hours)}", st["body"]),
        _p(f"목표 대비: {'-' if prog.percent is None else f'{prog.percent:g}%'}", st["body"]),
        _p("활동 목록", st["h"]),
    ]
    c = st["cell"]
    rows = [[_p(h, c) for h in ("날짜", "기관", "시간", "분야", "구분", "확인서")]]
    for line in prog.lines:
        a = acts[line.activity_id]
        rows.append([_p(line.start_date.isoformat(), c), _p(a.organization, c), _p(_h(line.hours), c),
                     _p(_field(line.type_code), c), _p(STATUS_LABELS[line.status], c),
                     _p("첨부" if a.certificate_path else "없음", c)])
    empty = len(rows) == 1
    if empty:
        rows.append([_p("이 기간에 기록된 활동이 없습니다.", c)] + [""] * 5)
    story.append(_table(rows, [24 * mm, 48 * mm, 18 * mm, 36 * mm, 24 * mm, 18 * mm], font, empty))
    return story


def _student_story(profile: Profile, prog: Progress, acts: dict[int, Activity], today: date, font: str) -> list:
    st = _styles(font)
    by_type = ", ".join(f"{_field(k)} {_h(v)}" for k, v in prog.hours_by_type.items()) or "-"
    story = [
        _p("봉사활동 포트폴리오", st["title"]),
        _p(f"작성일: {today.isoformat()} | 희망 전공: {profile.major or '-'}", st["body"]),
        _p(f"기간: {_period(prog)}", st["body"]),
        _p("요약", st["h"]),
        _p(f"누적 시간: {_h(prog.recognized_hours)}"
           + (f" | 목표 {_h(profile.goal_hours)}, 남은 {_h(prog.remaining_hours)}" if profile.goal_hours else ""), st["body"]),
        _p(f"분야별 시간: {by_type}", st["body"]),
        _p(f"전공 연계 활동: {prog.major_related_count}건", st["body"]),
        _p(f"학교 주관 {_h(prog.school_planned_hours)} / 개인 봉사 {_h(prog.personal_hours)}", st["body"]),
        _p("활동 기록", st["h"]),
    ]
    c = st["cell"]
    rows = [[_p(h, c) for h in ("날짜", "기관", "시간", "분야", "구분", "전공 연관성", "배운 점", "확인서")]]
    for line in prog.lines:
        a = acts[line.activity_id]
        rows.append([_p(line.start_date.isoformat(), c), _p(a.organization, c), _p(_h(line.hours), c),
                     _p(_field(line.type_code), c), _p("학교 주관" if a.school_planned else "개인", c),
                     _p(a.major_relevance, c), _p(a.lesson, c), _p("첨부" if a.certificate_path else "없음", c)])
    empty = len(rows) == 1
    if empty:
        rows.append([_p("이 기간에 기록된 활동이 없습니다.", c)] + [""] * 7)
    story.append(_table(rows, [20 * mm, 24 * mm, 13 * mm, 22 * mm, 18 * mm, 33 * mm, 36 * mm, 14 * mm], font, empty))
    return story


def _student_profile(store: Store) -> Profile | None:
    return next((p for p in store.list_profiles(active_only=True) if p.user_type == "student"), None)


def generate_report(store: Store, settings: Settings, today: date, kind: ReportKind | None = None,
                    now: datetime | None = None) -> ReportResult:
    """보고서 PDF를 DATA_DIR/reports에 만든다. kind가 없으면 사용자 유형으로 정한다."""
    if kind is None:
        setting = store.get_user_setting()
        kind = "student_portfolio" if setting and setting.user_type == "student" else "employee_report"

    if kind == "employee_report":
        block = company_report_block(store)
        if block:
            return ReportResult(status="blocked", message=block)
        profile = company_profile(store)
    else:
        profile = _student_profile(store)
        if profile is None:
            raise InputError("학생 프로필이 없습니다. 먼저 학년·희망 전공으로 프로필을 만들어 주세요.")

    activities = store.list_activities()
    rules_confirmed = bool(store.list_rules(profile_id=profile.id, status="confirmed"))
    prog = compute_progress(profile, activities, rules_confirmed, today)
    acts = {a.id: a for a in activities}
    font = _register_font()
    story = (_employee_story if kind == "employee_report" else _student_story)(profile, prog, acts, today, font)

    folder = settings.ensure_data_dir() / REPORTS_DIR
    folder.mkdir(exist_ok=True)
    stamp = (now or datetime.now(settings.tz)).strftime("%H%M%S")
    path = folder / f"{kind}_{today:%Y%m%d}_{stamp}.pdf"
    _build(path, story + [Spacer(1, 4 * mm)])
    label = "회사 보고서" if kind == "employee_report" else "포트폴리오"
    return ReportResult(status="created", path=path,
                        message=f"{label} PDF를 만들었습니다: {path} (활동 {len(prog.lines)}건, {_period(prog)})")
