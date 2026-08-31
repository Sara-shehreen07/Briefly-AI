import os
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
load_dotenv()

SARVAM_PIECE_SECONDS = 25
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

SARVAM_MAX_CONCURRENCY = 6   # global cap on in-flight Sarvam requests
GROQ_MAX_CONCURRENCY = 6     # Groq free tier ~30 RPM; 6 workers with retry is safe

_sarvam_slots = BoundedSemaphore(SARVAM_MAX_CONCURRENCY)


def _retryable(e: BaseException) -> bool:
    """Retry only transient failures (429/5xx/timeout); fail fast on 4xx like 401/413."""
    if isinstance(e, (requests.ConnectionError, requests.Timeout)):
        return True
    return (
        isinstance(e, requests.HTTPError)
        and e.response is not None
        and e.response.status_code in (429, 500, 502, 503, 504)
    )


def _wait(retry_state):
    """Honor Retry-After header when present, else exponential backoff (2s..15s)."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
    return wait_exponential(multiplier=1, min=2, max=15)(retry_state)


_RETRY = dict(
    retry=retry_if_exception(_retryable),
    stop=stop_after_attempt(4),
    wait=_wait,
)


@retry(**_RETRY)
def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / secrets.")
    headers = {"api-subscription-key": api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": os.getenv("SARVAM_STT_MODEL", "saaras:v2.5"), "with_diarization": "false"}
        with _sarvam_slots:
            response = requests.post(
                SARVAM_STT_TRANSLATE_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

    if not response.ok:
        print(f"Sarvam returned {response.status_code}")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces, send them concurrently (order-preserving), and join.
    """
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    piece_paths = []
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        if len(piece) == 0:
            continue
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        piece_paths.append(piece_path)

    try:
        with ThreadPoolExecutor(max_workers=SARVAM_MAX_CONCURRENCY) as ex:
            texts = list(ex.map(_send_to_sarvam, piece_paths))
    finally:
        for p in piece_paths:
            if os.path.exists(p):
                os.remove(p)

    return " ".join(t for t in texts if t).strip()


@retry(**_RETRY)
def transcribe_chunk_groq(chunk_path: str) -> str:
    """Send audio chunk to Groq's fast cloud Whisper API (whisper-large-v3)."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment / .env.")

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {groq_key}"}

    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
        data = {
            "model": "whisper-large-v3",
            "language": "en",
            "response_format": "json",
        }
        response = requests.post(url, headers=headers, files=files, data=data, timeout=120)

    if not response.ok:
        print(f"Groq API error: {response.status_code}")
        response.raise_for_status()

    return response.json().get("text", "")


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Groq Whisper or Sarvam depending on language selection.
    - hinglish -> Sarvam (translates to English while transcribing)
    - english  -> Groq Cloud Whisper (requires GROQ_API_KEY)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set in environment / .env. Please set GROQ_API_KEY to enable speech-to-text.")

    return transcribe_chunk_groq(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Groq Whisper"
    print(f"Using {engine} for transcription.")

    if not chunks:
        return ""

    # Sarvam path: pieces already run 6-wide per chunk, so limit chunk-level
    # workers to 2 (semaphore still caps total in-flight requests at 6).
    workers = 2 if language.lower() == "hinglish" else GROQ_MAX_CONCURRENCY
    with ThreadPoolExecutor(max_workers=workers) as ex:
        texts = list(ex.map(lambda c: transcribe_chunk(c, language=language), chunks))

    done = sum(1 for t in texts if t)
    print(f"Transcription complete ({done}/{len(chunks)} chunks produced text).")
    return " ".join(t for t in texts if t).strip()
