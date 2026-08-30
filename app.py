import streamlit as st
from dotenv import load_dotenv
from core.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(page_title="Briefly AI", layout="wide")


def run_pipeline(source: str, language: str = "english") -> dict:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# Session State
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Briefly AI")

with st.sidebar:
    st.header("New video")
    source = st.text_input("YouTube URL or local file path")
    language = st.radio("Language", ["english", "hinglish"], horizontal=True)
    run_clicked = st.button("Process", type="primary", use_container_width=True)

if run_clicked:
    if not source.strip():
        st.sidebar.error("Please enter a URL or file path.")
    else:
        with st.spinner("Processing video..."):
            try:
                st.session_state.result = run_pipeline(source.strip(), language)
                st.session_state.chat_history = []
            except Exception as e:
                st.sidebar.error(f"Pipeline failed: {e}")

result = st.session_state.result

if result:
    st.subheader(result["title"])

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs(
        ["Summary", "Action Items", "Key Decisions", "Open Questions", "Transcript"]
    )
    with tab_summary:
        st.write(result["summary"])
    with tab_actions:
        st.write(result["action_items"])
    with tab_decisions:
        st.write(result["key_decisions"])
    with tab_questions:
        st.write(result["open_questions"])
    with tab_transcript:
        st.text_area("Full transcript", result["transcript"], height=300)

    st.divider()
    st.subheader("Chat with your meeting")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    question = st.chat_input("Ask a question about the video...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(result["rag_chain"], question)
                st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))
else:
    st.info("Enter a YouTube URL or local file path in the sidebar and click Process to get started.")