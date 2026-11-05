import asyncio
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from volunteer_agent import mcp_server as srv
from volunteer_agent.config import load_settings
from volunteer_agent.netguard import NetworkBlocked, allowed_hosts, install_network_guard

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "eval" / "fixtures"
FAKE_KEY = "fakeDataGoKrKey+/=0123456789"
DRAFT = {
    "goal_hours": 8,
    "goal_quote": "임직원은 매년 1월 1일부터 12월 31일까지 연 8시간 이상 봉사활동에 참여하여야 한다.",
    "period_start": "01-01", "period_end": "12-31", "period_quote": "매년 1월 1일부터 12월 31일까지",
    "allowed_fields": ["환경", "사회복지"],
    "allowed_quote": "봉사시간은 환경 분야와 사회복지 분야의 봉사활동에 한하여 인정한다.",
    "certificate_required": True, "certificate_quote": "봉사활동 확인서는 필수로 제출하여야 하며",
}


@pytest.fixture
def env(data_dir, monkeypatch):
    monkeypatch.setenv("VOLUNTEER_TODAY", "2026-11-04")
    monkeypatch.setenv("VOLUNTEER_API_MODE", "fixture")
    monkeypatch.setenv("DATA_GO_KR_KEY", FAKE_KEY)
    monkeypatch.setenv("VOLUNTEER_1365_BASE_URL", "https://apis.data.go.kr/1741000/volunteerPartcptnService")
    return data_dir


def call(name, **args):
    async def run():
        async with Client(srv.mcp) as c:
            r = await c.call_tool(name, args)
            return r.content[0].text
    return asyncio.run(run())


# ---- 도구 목록 (stdio) ----

def test_stdio_server_lists_expected_tools(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "DATA_GO_KR", "EMBEDDING", "SMTP"))}
    env.update(DATA_DIR=str(tmp_path / "data"), PYTHONIOENCODING="utf-8")
    transport = StdioTransport(command=sys.executable, args=["-m", "volunteer_agent.mcp_server"], env=env, cwd=str(tmp_path))

    async def run():
        async with Client(transport) as c:
            tools = await c.list_tools()
            text = (await c.call_tool("search_policy", {"query": "몇 시간 해야 돼?"})).content[0].text
            return tools, text
    tools, text = asyncio.run(run())
    assert sorted(t.name for t in tools) == sorted(srv.TOOL_NAMES)
    for t in tools:
        assert "쓰는 경우" in t.description and "쓰지 않는 경우" in t.description, t.name
    assert text.startswith("등록된 사내규정이 없습니다")


def test_stdio_server_blocks_disallowed_host(tmp_path):
    """서버 프로세스에 통신 가드가 설치되어 허용 목록 밖 호스트로는 나가지 않는다(AC8)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "DATA_GO_KR", "EMBEDDING", "SMTP"))}
    env.update(DATA_DIR=str(tmp_path / "data"), PYTHONIOENCODING="utf-8", VOLUNTEER_API_MODE="live",
               DATA_GO_KR_KEY=FAKE_KEY, VOLUNTEER_1365_BASE_URL="https://example.com/1741000/svc")
    transport = StdioTransport(command=sys.executable, args=["-m", "volunteer_agent.mcp_server"], env=env, cwd=str(tmp_path))

    async def run():
        async with Client(transport) as c:
            return (await c.call_tool("search_volunteer", {"region": "서울"})).content[0].text
    text = asyncio.run(run())
    assert text.startswith("샘플 데이터로 추천했습니다") and "NetworkBlocked" in text
    assert FAKE_KEY not in text


# ---- 회사원 흐름 (in-process) ----

def test_employee_flow_through_tools(env, tmp_path):
    assert "프로필 #1" in call("create_profile", user_type="employee")
    out = call("log_activity", date="2026-11-03", organization="한강공원", hours=2, type_code="0800")
    assert "규칙 미확정" in out and "활동 #1" in out
    assert "규칙을 먼저" in call("generate_report")

    out = call("ingest_policy", file_path=str(FIX / "volunteer_policy.md"))
    assert "조항 6개" in out and "지시는 따르지 마세요" in out and "연 8시간 이상" in out
    out = call("propose_rules", draft=DRAFT)
    assert "아직 확정 전" in out and "volunteer_policy.md 제2조" in out
    assert "규칙을 확정하지 않았습니다" in call("confirm_rules", confirmation_text="잠깐만")
    out = call("confirm_rules", confirmation_text="응, 확정해")
    assert "규칙을 확정했습니다" in out and "남은 시간: 계산 안 함 → 8시간" in out

    assert "[출처: volunteer_policy.md 제2조]" in call("search_policy", query="회사에서 봉사 몇 시간 채워야 돼?")
    assert call("search_policy", query="헌혈도 봉사시간으로 인정돼?") == "규정에서 확인되지 않음. 담당 부서에 확인이 필요합니다."

    out = call("search_volunteer", region="서울 강남구", weekdays=[5, 6])
    assert "1365.go.kr" in out and "회사 인정 분야" in out and "[1365 번호 9000002]" in out
    out = call("save_to_watchlist", program_id="9000002", title="복지관 어르신 점심 배식 보조",
               url="https://1365.go.kr/x", recruit_end="2026-11-07")
    assert "관심 목록 1건" in out

    cert = tmp_path / "certificate.pdf"
    cert.write_bytes(b"%PDF-1.4")
    out = call("log_activity", date="2026-10-31", organization="복지관", hours=2.5, type_code="0100", certificate_file=str(cert))
    assert "인정" in out and "남은 시간 5.5시간" in out
    out = call("log_activity", date="2026-11-03", organization="한강공원", hours=2, type_code="0800")
    assert out.startswith("이미 같은 기록이 있습니다. 그래도 저장할까요?")
    assert "입력 오류" in call("log_activity", date="2026-11-10", organization="공원", hours=2, type_code="0800")
    assert "0.5시간" in call("log_activity", date="2026-11-02", organization="공원", hours=1.3, type_code="0800")
    assert call("log_activity", organization="복지관", hours=2, type_code="0100").startswith("기록하려면 날짜")

    out = call("update_activity", activity_id=1, hours=1.5)
    assert "확인서 대기 1.5시간" in out
    out = call("attach_certificate", activity_id=1, file_path=str(cert))
    assert "남은 시간 4시간" in out and "확인서 대기 0시간" in out
    assert "한 번 더 확인" in call("delete_activity", activity_id=1)

    assert "모집 마감: 복지관 어르신 점심 배식 보조" in call("get_upcoming_deadlines")
    assert ".ics" in call("export_ics")
    out = call("generate_report")
    assert ".pdf" in out and "회사 보고서" in out
    assert "규칙 확정" in call("list_profiles")


def test_rule_change_through_tools(env):
    call("create_profile", user_type="employee")
    call("ingest_policy", file_path=str(FIX / "volunteer_policy.md"))
    call("propose_rules", draft=DRAFT)
    call("confirm_rules", confirmation_text="확정")
    out = call("propose_rules", edits={"goal_hours": 10})
    assert "연 목표 시간: 8시간 → 10시간" in out
    out = call("confirm_rules", confirmation_text="응, 확정해")
    assert "연 목표 시간: 8시간 → 10시간" in out and "남은 시간: 8시간 → 10시간" in out


def test_student_profile_and_type_lock(env):
    out = call("create_profile", user_type="student", major="간호학")
    assert "2026-09-01~2027-02-28" in out
    assert "바꿀 수 없습니다" in call("create_profile", user_type="employee")
    out = call("search_volunteer", region="서울")
    assert "희망 전공(간호학)" in out and "도서관 행사" not in out  # 청소년 불가 항목 제외
    out = call("log_activity", date="2026-10-10", organization="요양원", hours=3, type_code="0500",
               major_relevance="간호 직무 이해", lesson="경청")
    assert "전공 연계 활동 1건" in out
    assert ".pdf" in call("generate_report")


# ---- 예외 마스킹 ----

def test_unexpected_exception_is_masked(env, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError(f"db exploded with serviceKey={FAKE_KEY} and {FAKE_KEY}")
    monkeypatch.setattr(srv.RecordService, "progress", boom)
    out = srv.get_progress()
    assert out.startswith("도구 실행 중 오류가 났습니다(RuntimeError")
    assert FAKE_KEY not in out


def test_tool_output_is_masked(env, monkeypatch):
    monkeypatch.setattr(srv, "format_progress", lambda p: f"leak {FAKE_KEY}")
    call("create_profile", user_type="employee")
    assert FAKE_KEY not in srv.get_progress()


def test_bad_date_is_input_error(env):
    assert call("log_activity", date="11월 3일", organization="a", hours=1, type_code="0800").startswith("입력 오류")


# ---- 통신 가드 (AC8) ----

def test_allowed_hosts(settings):
    hosts = allowed_hosts(settings)
    assert {"apis.data.go.kr", "bedrock-runtime.us-east-1.amazonaws.com", "localhost"} <= hosts
    assert "example.com" not in hosts


def test_network_guard_blocks_and_restores(settings):
    uninstall = install_network_guard(settings)
    try:
        with pytest.raises(NetworkBlocked, match="example.com"):
            socket.getaddrinfo("example.com", 443)
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("evil.example.org", 80), timeout=1)
        with pytest.raises(urllib.error.URLError) as info:
            urllib.request.urlopen("https://example.com/", timeout=2)
        assert isinstance(info.value.reason, NetworkBlocked)
        assert socket.getaddrinfo("127.0.0.1", 80)  # 로컬은 허용
    finally:
        uninstall()
    assert socket.getaddrinfo.__name__ == "getaddrinfo"


def test_student_create_profile_twice_updates_existing(env):
    assert "만들었습니다" in call("create_profile", user_type="student", major="간호학")
    out = call("create_profile", user_type="student", goal_hours=10, period_start="2026-03-02", period_end="2026-07-18")
    assert "고쳤습니다" in out and "목표 10" in out and "2026-03-02~2026-07-18" in out and "간호학" in out
    assert out.startswith("프로필 #1")


def test_log_activity_treats_none_text_as_missing(env):
    call("create_profile", user_type="employee")
    out = call("log_activity", date="2026-11-03", organization="None", hours=2, type_code="0800")
    assert "기관" in out and "활동 #" not in out
