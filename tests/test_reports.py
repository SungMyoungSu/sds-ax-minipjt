from datetime import date, datetime

import pytest
from pypdf import PdfReader

from volunteer_agent.errors import InputError
from volunteer_agent.models import RuleCandidate
from volunteer_agent.records import ActivityInput, RecordService
from volunteer_agent.reports import find_korean_font, generate_report
from volunteer_agent.rules import RULES_FIRST_MSG

pytestmark = pytest.mark.skipif(find_korean_font() is None, reason="한글 폰트가 없는 환경")

TODAY = date(2026, 12, 28)
NOW = datetime(2026, 12, 28, 10, 0, 0)


@pytest.fixture
def cert(tmp_path):
    f = tmp_path / "cert.pdf"
    f.write_bytes(b"%PDF-1.4")
    return f


def pdf_text(path):
    """칸 안 줄바꿈과 공백 차이를 없앤 텍스트(비교용)."""
    raw = "".join(page.extract_text() for page in PdfReader(str(path)).pages)
    return "".join(raw.split())


def has(text, s):
    return "".join(s.split()) in text


def embedded_fonts(path):
    found = []
    for page in PdfReader(str(path)).pages:
        for font in page["/Resources"]["/Font"].values():
            font = font.get_object()
            for desc_font in font.get("/DescendantFonts", [font]):
                desc = desc_font.get_object().get("/FontDescriptor")
                if desc is not None:
                    desc = desc.get_object()
                    if any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3")):
                        found.append(str(desc["/FontName"]))
    return found


def employee(store, settings, cert, confirmed=True):
    store.init_user_setting("employee")
    svc = RecordService(store, settings, TODAY)
    p = svc.create_profile("employee")
    p.goal_hours, p.certificate_required, p.allowed_type_codes = 8, True, ["0800", "0100"]
    p.period_start, p.period_end = date(2026, 1, 1), date(2026, 12, 31)
    store.update_profile(p)
    if confirmed:
        store.mark_rule_confirmed(store.add_rule_candidate(RuleCandidate(profile_id=p.id, goal_hours=8)).id)
    svc.log_activity(ActivityInput(start_date=date(2026, 10, 31), organization="강남종합사회복지관", hours=2.5,
                                   type_code="0100", certificate_file=cert))
    svc.log_activity(ActivityInput(start_date=date(2026, 11, 3), organization="한강공원 <환경정화>", hours=1.5,
                                   type_code="0800"))
    svc.log_activity(ActivityInput(start_date=date(2026, 11, 4), organization="유기견 보호소", hours=3,
                                   type_code="1500"))
    svc.log_activity(ActivityInput(start_date=date(2025, 12, 30), organization="지난해 봉사", hours=2,
                                   type_code="0800", certificate_file=cert))
    return p


def test_ac7_employee_report_contents_and_font(store, settings, cert):
    employee(store, settings, cert)
    r = generate_report(store, settings, TODAY, now=NOW)
    assert r.status == "created" and r.path.suffix == ".pdf"
    assert r.path.parent == settings.data_dir / "reports"
    assert ".pdf" in r.message and "활동 3건" in r.message
    text = pdf_text(r.path)
    for s in ["회사 제출용 봉사활동 보고서", "기준 기간: 2026-01-01 ~ 2026-12-31",
              "2026-10-31", "강남종합사회복지관", "2.5시간", "생활편의",
              "2026-11-03", "한강공원 <환경정화>", "1.5시간",  # 이스케이프된 사용자 입력이 그대로 보임
              "2026-11-04", "유기견 보호소", "3시간", "기타",
              "인정", "확인서 대기", "제외", "첨부", "없음",
              "인정 시간 합계: 2.5시간", "확인서 대기 시간: 1.5시간", "남은 시간: 5.5시간", "목표 대비: 31.2%"]:
        assert has(text, s), s
    assert not has(text, "지난해 봉사")  # 기준 기간 밖
    fonts = embedded_fonts(r.path)
    assert fonts and all("KoreanFont" in f or "Malgun" in f or "+" in f for f in fonts)


def test_employee_report_blocked_before_rules_confirmed(store, settings, cert):
    employee(store, settings, cert, confirmed=False)
    r = generate_report(store, settings, TODAY, kind="employee_report")
    assert r.status == "blocked" and r.path is None and r.message == RULES_FIRST_MSG
    assert not (settings.data_dir / "reports").exists()


def test_ac7_ac14e_student_portfolio(store, settings, cert):
    store.init_user_setting("student")
    svc = RecordService(store, settings, date(2026, 6, 30))
    svc.create_profile("student", major="간호학")
    svc.log_activity(ActivityInput(start_date=date(2026, 3, 10), organization="햇살요양원", hours=3, type_code="0500",
                                   major_relevance="어르신 돌봄으로 간호 직무 이해", lesson="경청의 중요성",
                                   certificate_file=cert))
    svc.log_activity(ActivityInput(start_date=date(2026, 4, 2), organization="OO고등학교", hours=2, type_code="0800",
                                   school_planned=True, lesson="환경 보호 실천"))
    r = generate_report(store, settings, date(2026, 6, 30), now=NOW)
    assert r.status == "created" and "포트폴리오" in r.message
    text = pdf_text(r.path)
    for s in ["봉사활동 포트폴리오", "희망 전공: 간호학", "2026-03-01 ~ 2026-08-31",
              "2026-03-10", "햇살요양원", "3시간", "보건ㆍ의료", "어르신 돌봄으로 간호 직무 이해", "경청의 중요성",
              "2026-04-02", "OO고등학교", "환경 보호 실천",
              "학교 주관", "개인", "첨부", "없음",
              "누적 시간: 5시간", "전공 연계 활동: 1건", "학교 주관 2시간 / 개인 봉사 3시간"]:
        assert has(text, s), s
    assert "대입" not in text
    assert embedded_fonts(r.path)


def test_student_without_profile(store, settings):
    store.init_user_setting("student")
    with pytest.raises(InputError, match="학생 프로필이 없습니다"):
        generate_report(store, settings, TODAY)


def test_empty_period_report(store, settings):
    store.init_user_setting("employee")
    svc = RecordService(store, settings, TODAY)
    p = svc.create_profile("employee")
    p.goal_hours, p.period_start, p.period_end = 8, date(2026, 1, 1), date(2026, 12, 31)
    store.update_profile(p)
    store.mark_rule_confirmed(store.add_rule_candidate(RuleCandidate(profile_id=p.id)).id)
    r = generate_report(store, settings, TODAY, now=NOW)
    assert has(pdf_text(r.path), "이 기간에 기록된 활동이 없습니다.")
