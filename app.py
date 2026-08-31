import html
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from core.pipeline import get_transcript, synthesize
from core.rag_engine import build_rag_chain

load_dotenv()

st.set_page_config(page_title="Briefly AI", layout="wide", page_icon="▚")

# ── Stylesheet (scoped: stable data-testid hooks + .bx-* classes only) ────
with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Cached resources ────────────────────────────────────────────────────────
# Transcript (download + STT — the minutes-long stage) is cached per
# (source, language): re-processing the same video skips straight to the
# ~15s LLM stage. Pure function — no UI calls inside, so cache replay is
# safe. Exceptions are never cached.
@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def _cached_transcript(source: str, language: str) -> str:
    return get_transcript(source, language)


# RAG chain cached per-container. Embeddings are a singleton in
# core/vector_store.py, so each entry holds only a ~150KB Qdrant index.
@st.cache_resource(show_spinner="Preparing Q&A index...", max_entries=4)
def _cached_rag_chain(transcript: str):
    return build_rag_chain(transcript)


# ── Session state ───────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ── Sidebar: new meeting ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="bx-brand">Briefly</div>'
        '<div class="bx-brand-sub">Meeting notes from any recording</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload audio or video recording",
        type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
        label_visibility="collapsed",
    )
    language = st.pills(
        "Language",
        ["english", "hinglish"],
        default="english",
        label_visibility="collapsed",
    )
    language = language or "english"
    run_clicked = st.button("Summarize", type="primary", use_container_width=True)

error_msg = None

if run_clicked:
    if uploaded_file is None:
        error_msg = "Please upload an audio or video file first."
    else:
        upload_dir = os.path.join(tempfile.gettempdir(), "briefly_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        input_source = os.path.join(upload_dir, uploaded_file.name)
        with open(input_source, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.rag_chain = None
        try:
            with st.status("Processing meeting…", expanded=True) as status:
                def _stage(name, detail=""):
                    status.update(label=f"{name}…")
                    if detail:
                        status.write(detail)

                status.update(label="Fetching & transcribing…")
                transcript = _cached_transcript(input_source, language)
                # synthesize runs UNCACHED so the callback can drive the
                # status widget live (callbacks inside cache_data are illegal).
                st.session_state.result = synthesize(transcript, language, on_stage=_stage)
                status.update(label="Complete", state="complete", expanded=False)
        except ValueError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"Pipeline failed: {e}"

result = st.session_state.result

# ── Main column ─────────────────────────────────────────────────────────────
if error_msg:
    st.error(error_msg)

if result is None:
    if not error_msg:
        st.markdown(
            '<div class="bx-empty">'
            '<div class="bx-mark">▚ briefly</div>'
            '<div class="bx-title">Turn meetings into actionable notes</div>'
            '<div class="bx-sub">Upload an audio or video file in the sidebar to extract summaries, decisions, and chat with the transcript.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
else:
    lang_used = result.get("language", "english")
    engine = "Sarvam" if lang_used.lower() == "hinglish" else "Whisper"
    st.markdown(
        f'<div class="bx-header"><h1>{html.escape(result["title"])}</h1></div>'
        '<div class="bx-badges">'
        f'<span class="bx-badge"><span class="bx-dot"></span>{engine}</span>'
        f'<span class="bx-badge">{html.escape(lang_used)}</span>'
        f'<span class="bx-badge">{len(result["transcript"].split())} words</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs(
            ["Summary", "Action Items", "Key Decisions", "Open Questions", "Transcript"]
        )
        with tab_summary:
            st.markdown(result["summary"])
        with tab_actions:
            st.markdown(result["action_items"])
        with tab_decisions:
            st.markdown(result["key_decisions"])
        with tab_questions:
            st.markdown(result["open_questions"])
        with tab_transcript:
            st.text_area(
                "Full transcript",
                result["transcript"],
                height=300,
                label_visibility="collapsed",
            )

    st.markdown("## Chat with this meeting")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ask about this meeting…")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                if st.session_state.rag_chain is None:
                    st.session_state.rag_chain = _cached_rag_chain(result["transcript"])
                answer = st.write_stream(st.session_state.rag_chain.stream(question))
                st.session_state.chat_history.append(("assistant", answer))
            except Exception as e:
                st.error(f"Could not answer: {e}")
