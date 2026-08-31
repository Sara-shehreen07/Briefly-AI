import os
import glob
import subprocess
import tempfile
import yt_dlp


def download_youtube_audio(url: str, output_dir: str = None) -> str:
    """Download the best available audio stream as-is (no WAV postprocessing;
    conversion + chunking happen in one streaming ffmpeg pass later)."""
    target_dir = output_dir or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as e:
            raise ValueError(
                f"Could not retrieve video info (private/age-gated/live/unsupported URL): {e}"
            ) from e
        if info.get("_type") == "playlist" or "entries" in info:
            raise ValueError("Playlist URL detected. Please provide a single video URL.")
        if info.get("is_live"):
            raise ValueError("Live streams are not supported. Please provide a completed video.")
        try:
            info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raise ValueError(
                f"Could not download audio (private/age-gated/unsupported URL): {e}"
            ) from e
        filename = ydl.prepare_filename(info)
    if not os.path.exists(filename):
        raise ValueError(f"Downloaded file not found on disk: {filename}")
    return filename


def convert_to_wav(input_path: str, output_dir: str = None) -> str:
    """Convert any audio/video file to 16kHz mono WAV using ffmpeg directly (no full-file RAM decode)."""
    target_dir = output_dir or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(target_dir, f"{base_name}_converted.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", output_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"Could not decode '{input_path}' (corrupted or unsupported file): "
            f"{e.stderr.decode(errors='replace')[-300:] if e.stderr else 'unknown ffmpeg error'}"
        ) from e
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5, output_dir: str = None) -> list:
    """Slice a WAV file into N-minute chunks in one streaming ffmpeg pass (disk-based, no RAM decode)."""
    target_dir = output_dir or os.path.dirname(wav_path) or tempfile.gettempdir()
    os.makedirs(target_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    pattern = os.path.join(target_dir, f"{base_name}_chunk_%03d.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-f", "segment", "-segment_time", str(chunk_minutes * 60),
                "-c", "copy", pattern,
            ],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"Audio chunking failed for '{wav_path}': "
            f"{e.stderr.decode(errors='replace')[-300:] if e.stderr else 'unknown ffmpeg error'}"
        ) from e
    # NOTE: glob treats '%' literally, so match ffmpeg's %03d output with '*'
    return sorted(glob.glob(os.path.join(target_dir, f"{base_name}_chunk_*.wav")))


def process_input(source: str, temp_dir: str = None) -> list:
    target_dir = temp_dir or tempfile.gettempdir()
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        raw_path = download_youtube_audio(source, output_dir=target_dir)
    else:
        if not os.path.exists(source):
            raise ValueError(f"Local file not found: {source}")
        print("Detected local file.")
        raw_path = source

    print("Converting to 16kHz mono WAV...")
    wav_path = convert_to_wav(raw_path, output_dir=target_dir)

    # 10-min chunks: 16kHz mono WAV = 19.2MB per chunk, under Groq's 25MB limit,
    # halving request count. Sarvam re-slices to 25s pieces anyway.
    print("Chunking audio...")
    chunks = chunk_audio(wav_path, chunk_minutes=10, output_dir=target_dir)
    if not chunks:
        raise ValueError("No audio stream found in the source (empty or corrupted file).")
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks
