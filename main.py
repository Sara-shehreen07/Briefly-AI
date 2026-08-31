import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI video assistant...")

    # ignore_cleanup_errors: on Windows, ffmpeg file handles can linger briefly
    # after chunk export; without this a successful run can die on tempdir cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        chunks = process_input(source, temp_dir=temp_dir)
        transcript = transcribe_all(chunks, language)

    if not transcript.strip():
        raise ValueError("No speech detected in the audio.")

    print(f"raw transcription (first 300 characters): {transcript[:300]}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            "title": executor.submit(generate_title, transcript),
            "summary": executor.submit(summarize, transcript),
            "action_items": executor.submit(extract_action_items, transcript),
            "key_decisions": executor.submit(extract_key_decisions, transcript),
            "open_questions": executor.submit(extract_questions, transcript),
        }
        results = {name: future.result() for name, future in futures.items()}

    return {
        "title": results["title"],
        "transcript": transcript,
        "summary": results["summary"],
        "action_items": results["action_items"],
        "key_decisions": results["key_decisions"],
        "open_questions": results["open_questions"],
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG (built lazily on first question)
    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = None
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        if rag_chain is None:
            print("Preparing Q&A index...")
            rag_chain = build_rag_chain(result["transcript"])
        answer = ask_question(rag_chain, question)
        print(f"\n Assistant: {answer}\n")

