import os
import sys
import tempfile
from dotenv import load_dotenv

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()  # MUST be before any core/ imports

from core.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not set in environment/.env")

source = "https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s"
language = "english"  # "english" → Groq Whisper, "hinglish" → Sarvam

with tempfile.TemporaryDirectory() as temp_dir:
    chunks = process_input(source, temp_dir=temp_dir)
    transcript = transcribe_all(chunks, language=language)

print("\n" + "=" * 60)
print("[TRANSCRIPT]")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)

title = generate_title(transcript)
summary = summarize(transcript)

print("\n" + "=" * 60)
print(f"[TITLE]: {title}")
print("=" * 60)
print("\n[SUMMARY]")
print("-" * 60)
print(summary)

action_items = extract_action_items(transcript)
decisions = extract_key_decisions(transcript)
questions = extract_questions(transcript)

print("\n" + "=" * 60)
print("[ACTION ITEMS]")
print("=" * 60)
print(action_items)

print("\n" + "=" * 60)
print("[KEY DECISIONS]")
print("=" * 60)
print(decisions)

print("\n" + "=" * 60)
print("[OPEN QUESTIONS]")
print("=" * 60)
print(questions)

print("\n" + "=" * 60)
print("[RAG Q&A TEST]")
print("=" * 60)
rag_chain = build_rag_chain(transcript)
test_answer = ask_question(rag_chain, "What was the main topic discussed in this meeting?")
print(f"Answer: {test_answer}")