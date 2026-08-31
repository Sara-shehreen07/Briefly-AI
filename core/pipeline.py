import tempfile
from concurrent.futures import ThreadPoolExecutor

from core.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions


def get_transcript(source: str, language: str = "english", on_stage=None) -> str:
    """Stage 1: download + convert + transcribe. Pure function of
    (source, language) — safe to cache (no UI, no callbacks required)."""
    def _emit(stage: str, detail: str = ""):
        if on_stage:
            on_stage(stage, detail)

    # ignore_cleanup_errors: on Windows, ffmpeg file handles can linger briefly
    # after chunk export; without this a successful run can die on tempdir cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        _emit("download", "Fetching audio…")
        chunks = process_input(source, temp_dir=temp_dir)
        _emit("transcribe", f"{len(chunks)} chunk(s)")
        return transcribe_all(chunks, language)


def synthesize(transcript: str, language: str = "english", on_stage=None) -> dict:
    """Stage 2: the 5 parallel LLM calls. ~10-25s; cheap enough to re-run
    uncached, which allows live stage callbacks from the UI layer."""
    def _emit(stage: str, detail: str = ""):
        if on_stage:
            on_stage(stage, detail)

    if not transcript.strip():
        raise ValueError("No speech detected in the audio.")

    _emit("synthesize", "Summary, action items, decisions, questions")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            "title": executor.submit(generate_title, transcript),
            "summary": executor.submit(summarize, transcript),
            "action_items": executor.submit(extract_action_items, transcript),
            "key_decisions": executor.submit(extract_key_decisions, transcript),
            "open_questions": executor.submit(extract_questions, transcript),
        }
        results = {name: future.result() for name, future in futures.items()}

    _emit("done", "")
    return {
        "title": results["title"],
        "transcript": transcript,
        "summary": results["summary"],
        "action_items": results["action_items"],
        "key_decisions": results["key_decisions"],
        "open_questions": results["open_questions"],
        "language": language,
    }


def run_pipeline(source: str, language: str = "english", on_stage=None) -> dict:
    """Full pipeline: get_transcript + synthesize (composed for CLI use)."""
    transcript = get_transcript(source, language, on_stage)
    return synthesize(transcript, language, on_stage)
