"""Streamlit 보조 화면 (`uv run streamlit run app.py`, localhost 전용).

CLI와 같은 앱(`open_app`: Supervisor + 하위 Agent + 로컬 MCP 서버)과 입력 가드레일을 쓴다.
범위는 채팅과 PDF 다운로드뿐이다(미결 결정 6). 세션 1개 = 대화 1개.
테스트는 st.session_state["backend"]에 가짜 백엔드를 넣어 LLM 없이 돌린다.
"""

from __future__ import annotations

import streamlit as st

from volunteer_agent.config import load_settings
from volunteer_agent.web import ChatBackend, save_user_type, user_type

USER_TYPES = {"employee": "회사원", "student": "학생"}

st.set_page_config(page_title="봉사활동 도우미", page_icon="🤝")
st.title("봉사활동 도우미")
st.caption("봉사 추천, 기록, 사내 봉사 규정 안내. 신청은 1365 자원봉사포털에서 직접 해 주세요.")


def get_backend():
    backend = st.session_state.get("backend")
    if backend is None:
        from volunteer_agent.cli import default_runtime
        from volunteer_agent.llm import has_llm_credentials

        settings = load_settings()
        if not has_llm_credentials(settings):
            st.error("Bedrock 설정(LLM 모델, AWS 키)이 .env에 없어 시작할 수 없습니다.")
            st.stop()
        backend = ChatBackend(settings, default_runtime())
        st.session_state["backend"] = backend
    return backend


backend = get_backend()

# 처음 실행: 사용자 유형(PC당 1명, 이후 바꾸지 않음)
if user_type(backend.settings) is None:
    st.info("처음 오셨네요. 회사원이신가요, 학생이신가요? 한 번 정하면 바꾸지 않습니다.")
    choice = st.radio("사용자 유형", list(USER_TYPES), format_func=USER_TYPES.get, key="user_type_choice")
    if st.button("저장", key="save_user_type"):
        save_user_type(backend.settings, choice)
        st.rerun()
    st.stop()

history = st.session_state.setdefault("history", [])


def render(entry: dict, index: int) -> None:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        for j, path in enumerate(entry.get("pdfs", [])):
            try:
                data = path.read_bytes()
            except OSError:
                st.caption(f"파일을 찾을 수 없습니다: {path.name}")
                continue
            st.download_button(f"PDF 다운로드: {path.name}", data=data, file_name=path.name,
                               mime="application/pdf", key=f"pdf-{index}-{j}")
        if entry.get("trace"):
            st.caption(entry["trace"])


for i, entry in enumerate(history):
    render(entry, i)

if prompt := st.chat_input("무엇을 도와드릴까요?"):
    history.append({"role": "user", "content": prompt})
    render(history[-1], len(history) - 1)
    with st.spinner("답을 준비하고 있습니다..."):
        reply = backend.ask(prompt)
    if reply.refusal:
        trace = f"입력 가드레일 차단: {reply.refusal}"
    else:
        trace = f"Agent: {', '.join(reply.agents) or '-'} | 도구: {', '.join(reply.tools) or '-'}"
    history.append({"role": "assistant", "content": reply.answer or "(답변이 비어 있습니다.)",
                    "pdfs": reply.pdfs, "trace": trace})
    render(history[-1], len(history) - 1)
