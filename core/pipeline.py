import tempfile
from concurrent.futures import ThreadPoolExecutor

from core.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions


def run_pipeline(source: str, language: str = "english") -> dict:
    # ignore_cleanup_errors: on Windows, ffmpeg file handles can linger briefly
    # after chunk export; without this a successful run can die on tempdir cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        chunks = process_input(source, temp_dir=temp_dir)
        transcript = transcribe_all(chunks, language)

    if not transcript.strip():
        raise ValueError("No speech detected in the audio.")

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
