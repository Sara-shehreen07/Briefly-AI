import os
import sys
import gradio as gr
from dotenv import load_dotenv

# Ensure module path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ZeroGPU decorator support
try:
    import spaces
    gpu_decorator = spaces.GPU(duration=120)
except Exception:
    def gpu_decorator(func):
        return func

from core.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


@gpu_decorator
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


def handle_process(source_val: str, lang_val: str, current_state: dict):
    if not source_val or not source_val.strip():
        gr.Warning("Please enter a URL or file path.")
        return (
            current_state,
            gr.update(visible=False),
            "",
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            []
        )

    try:
        result = run_pipeline(source_val.strip(), lang_val.lower())
        new_state = {"result": result, "chat_history": []}
        return (
            new_state,
            gr.update(value=f"### {result['title']}", visible=True),
            result["summary"],
            result["action_items"],
            result["key_decisions"],
            result["open_questions"],
            result["transcript"],
            gr.update(visible=True),
            []
        )
    except Exception as e:
        gr.Error(f"Pipeline failed: {e}")
        return (
            current_state,
            gr.update(visible=False),
            "",
            "",
            "",
            "",
            "",
            gr.update(visible=False),
            []
        )


def handle_chat(user_msg: str, history: list, current_state: dict):
    if not user_msg or not user_msg.strip():
        return history, "", current_state

    result = current_state.get("result") if current_state else None
    if not result or not result.get("rag_chain"):
        gr.Warning("Please process a video first.")
        return history, "", current_state

    history = history or []
    history.append({"role": "user", "content": user_msg})

    try:
        answer = ask_question(result["rag_chain"], user_msg.strip())
    except Exception as e:
        answer = f"Error querying transcript: {e}"

    history.append({"role": "assistant", "content": answer})
    current_state["chat_history"] = history

    return history, "", current_state


with gr.Blocks(title="Briefly AI") as demo:
    state = gr.State(value={"result": None, "chat_history": []})

    gr.Markdown("# Briefly AI")

    with gr.Row():
        # Sidebar Column
        with gr.Column(scale=1):
            gr.Markdown("### New video")
            source_input = gr.Textbox(
                label="YouTube URL or local file path",
                placeholder="Enter YouTube URL or file path"
            )
            language_radio = gr.Radio(
                choices=["english", "hinglish"],
                value="english",
                label="Language"
            )
            process_btn = gr.Button("Process", variant="primary")

        # Main Content Column
        with gr.Column(scale=3):
            title_display = gr.Markdown(visible=False)

            with gr.Tabs():
                with gr.TabItem("Summary"):
                    summary_box = gr.Markdown()
                with gr.TabItem("Action Items"):
                    actions_box = gr.Markdown()
                with gr.TabItem("Key Decisions"):
                    decisions_box = gr.Markdown()
                with gr.TabItem("Open Questions"):
                    questions_box = gr.Markdown()
                with gr.TabItem("Transcript"):
                    transcript_box = gr.TextArea(
                        label="Full transcript",
                        lines=12,
                        interactive=False
                    )

            with gr.Group(visible=False) as chat_section:
                gr.Markdown("---")
                gr.Markdown("### Chat with your meeting")
                chatbot = gr.Chatbot(type="messages", height=350)
                chat_input = gr.Textbox(
                    placeholder="Ask a question about the video...",
                    show_label=False
                )

    # Event Connections
    process_btn.click(
        fn=handle_process,
        inputs=[source_input, language_radio, state],
        outputs=[
            state,
            title_display,
            summary_box,
            actions_box,
            decisions_box,
            questions_box,
            transcript_box,
            chat_section,
            chatbot
        ]
    )

    chat_input.submit(
        fn=handle_chat,
        inputs=[chat_input, chatbot, state],
        outputs=[chatbot, chat_input, state]
    )

if __name__ == "__main__":
    demo.launch()