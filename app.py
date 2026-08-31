import streamlit as st
from dotenv import load_dotenv
from core.pipeline import run_pipeline
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(page_title="Briefly AI", layout="wide")


# Session State
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

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
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.rag_chain = None
        with st.spinner("Processing video..."):
            try:
                st.session_state.result = run_pipeline(source.strip(), language)
            except ValueError as e:
                st.sidebar.error(str(e))
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
            try:
                if st.session_state.rag_chain is None:
                    with st.spinner("Preparing Q&A index..."):
                        st.session_state.rag_chain = build_rag_chain(result["transcript"])
                with st.spinner("Thinking..."):
                    answer = ask_question(st.session_state.rag_chain, question)
                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))
            except Exception as e:
                st.error(f"Error answering question: {e}")
else:
    st.info("Enter a YouTube URL or local file path in the sidebar and click Process to get started.")