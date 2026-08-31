import sys
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.pipeline import run_pipeline
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


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

