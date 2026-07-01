import json
import os
import streamlit as st
from qa_pipeline import process_question

st.set_page_config(page_title="공공기관 취업 Q&A", page_icon="🏢", layout="wide")

st.title("🏢 공공기관 취업 Q&A")
st.caption("공공기관 채용공고 7만 건 + 취준생 경험 게시글 기반 AI 답변 서비스")

# ── 채팅 기록 파일 경로 ───────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(messages: list):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _render_meta(meta: dict):
    with st.expander("검색 과정 보기", expanded=False):
        route = meta.get("route", "")
        st.markdown(f"**경로:** {'📋 채용공고' if route == 'posting' else '💬 경험 게시글'} — {meta.get('reason', '')}")
        graph = meta.get("graph", [])
        chunks = meta.get("chunks", [])
        exp = meta.get("experience", [])
        if graph:
            st.markdown(f"**구조 검색:** {len(graph)}건")
        if chunks:
            st.markdown(f"**본문 검색:** {len(chunks)}건")
        if exp:
            st.markdown(f"**경험 게시글:** {len(exp)}건")
            for r in exp:
                st.markdown(f"&nbsp;&nbsp;• (유사도 {r.get('score', 0):.3f}) {r.get('제목', '')[:60]}")

# ── 채팅 히스토리 ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            _render_meta(msg["meta"])

# ── 질문 처리 — qa_pipeline.process_question() 사용 ──────────────
# (파이프라인 로직은 qa_pipeline.py 단일 위치에서 관리)


# ── 입력창 ────────────────────────────────────────────────────────
if question := st.chat_input("질문을 입력하세요 (예: 정보처리기사 있으면 지원할 수 있는 공고 뭐 있어?)"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            # 현재 질문(마지막 항목) 제외한 이전 대화를 history로 전달
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            answer, meta = process_question(question, history=history or None)
        st.markdown(answer)
        _render_meta(meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
    save_history(st.session_state.messages)
