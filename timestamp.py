import shutil
from pathlib import Path

import whisper


# -----------------------------
# SETTINGS
# -----------------------------
MODEL_NAME = "base"
MIN_DURATION = 0.50
MAX_DURATION = 5.00
PAUSE_SPLIT_SECONDS = 0.70
PUNCTUATION = ".,?!؛؟"
SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mov",
    ".aac",
    ".flac",
    ".ogg",
    ".webm",
}


def find_media_files(audio_folder):
    """Find supported media files directly inside the audio folder."""
    if not audio_folder.is_dir():
        return []

    return sorted(
        (
            path
            for path in audio_folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS.xx, including durations over one hour."""
    total_centiseconds = int(round(seconds * 100))
    total_seconds, centiseconds = divmod(total_centiseconds, 100)
    total_minutes, whole_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{whole_seconds:02d}.{centiseconds:02d}"
    )


def format_srt_timestamp(seconds):
    """Format seconds as the SRT timestamp HH:MM:SS,mmm."""
    total_milliseconds = int(round(seconds * 1000))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    total_minutes, whole_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{whole_seconds:02d},{milliseconds:03d}"
    )


def collect_words(result):
    """Return all word timestamp dictionaries in Whisper's original order."""
    words = []
    for segment in result["segments"]:
        words.extend(segment.get("words", []))
    return words


def save_txt(segments, output_path):
    """Save timestamped segment text as UTF-8."""
    blocks = [
        f"[{format_timestamp(segment['start'])} --> "
        f"{format_timestamp(segment['end'])}]\n{segment['text']}"
        for segment in segments
        if segment["text"]
    ]
    output_path.write_text("\n\n".join(blocks), encoding="utf-8")


def save_srt(segments, output_path):
    """Save numbered standard SRT subtitles as UTF-8."""
    blocks = [
        f"{number}\n"
        f"{format_srt_timestamp(segment['start'])} --> "
        f"{format_srt_timestamp(segment['end'])}\n"
        f"{segment['text']}"
        for number, segment in enumerate(segments, start=1)
        if segment["text"]
    ]
    output_path.write_text("\n\n".join(blocks), encoding="utf-8")


def reserve_output_paths(audio_path, output_dir):
    """Reserve a matching TXT/SRT name without overwriting an older result."""
    version = 1
    while True:
        suffix = "" if version == 1 else f"_{version}"
        txt_path = output_dir / f"{audio_path.stem}_timestamps{suffix}.txt"
        srt_path = output_dir / f"{audio_path.stem}_subtitles{suffix}.srt"

        try:
            with txt_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            version += 1
            continue

        try:
            with srt_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            txt_path.unlink()
            version += 1
            continue
        except OSError:
            txt_path.unlink()
            raise

        return txt_path, srt_path


def _make_segment(words, overlong_word=False):
    parts = []
    for word in words:
        text = word["word"]
        if parts and text and not text[0].isspace():
            previous_text = "".join(parts)
            if (
                previous_text
                and not previous_text[-1].isspace()
                and text[0] not in PUNCTUATION
            ):
                parts.append(" ")
        parts.append(text)

    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": "".join(parts).strip(),
        "overlong_word": overlong_word,
    }


def create_segments(
    words,
    min_duration=MIN_DURATION,
    max_duration=MAX_DURATION,
):
    """Group whole Whisper words into short segments using their timestamps."""
    segments = []
    current_words = []

    def finish_current():
        if current_words:
            segments.append(_make_segment(current_words))
            current_words.clear()

    for word in words:
        word_duration = word["end"] - word["start"]

        if word_duration > max_duration:
            finish_current()
            segments.append(_make_segment([word], overlong_word=True))
            print(
                "WARNING: Keeping this word intact even though it is longer "
                f"than {max_duration:.2f}s: {word['word'].strip()}"
            )
            continue

        if current_words:
            current_duration = word["end"] - current_words[0]["start"]
            pause = word["start"] - current_words[-1]["end"]
            candidate_duration = word["end"] - current_words[0]["start"]

            if (
                candidate_duration > max_duration
                or (
                    pause >= PAUSE_SPLIT_SECONDS
                    and current_duration >= min_duration
                )
            ):
                finish_current()

        current_words.append(word)
        current_duration = word["end"] - current_words[0]["start"]

        if (
            current_duration >= min_duration
            and word["word"].rstrip()
            and word["word"].rstrip()[-1] in PUNCTUATION
        ):
            finish_current()

    finish_current()
    return segments


def process_media_file(audio_path, model, output_dir):
    """Transcribe one media file and save its separate TXT and SRT files."""
    print("🎧 Transcribing...")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        fp16=False,
    )

    print("✂️ Creating timestamps...")
    segments = create_segments(collect_words(result))
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if not segment["overlong_word"] and duration > MAX_DURATION:
            raise ValueError(
                f"Generated segment is too long: {duration:.2f}s"
            )

    txt_path, srt_path = reserve_output_paths(audio_path, output_dir)
    try:
        print("💾 Saving TXT...")
        save_txt(segments, txt_path)

        print("💾 Saving SRT...")
        save_srt(segments, srt_path)
    except Exception:
        txt_path.unlink(missing_ok=True)
        srt_path.unlink(missing_ok=True)
        raise

    long_word_count = sum(
        1 for segment in segments if segment["overlong_word"]
    )
    return len(segments), txt_path, srt_path, long_word_count


def main():
    project_dir = Path(__file__).resolve().parent
    audio_folder = project_dir / "audio"
    output_dir = project_dir / "output"
    media_files = find_media_files(audio_folder)

    if not media_files:
        print("❌ No audio/video files found in audio/.")
        print(
            "Put an MP3, WAV, MP4, M4A or other supported media file "
            "inside the audio folder."
        )
        return

    if shutil.which("ffmpeg") is None:
        print("❌ FFmpeg was not found. Install FFmpeg and make sure it is in PATH.")
        return

    print(f"Files found: {len(media_files)}")
    print("⏳ Loading Whisper...")
    try:
        model = whisper.load_model(MODEL_NAME)
    except Exception as error:
        print(f"❌ Could not load Whisper model: {error}")
        print(f"Completed: 0\nFailed: {len(media_files)}")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"❌ Could not create output folder: {error}")
        print(f"Completed: 0\nFailed: {len(media_files)}")
        return

    completed = 0
    failed = 0

    for index, audio_path in enumerate(media_files, start=1):
        print(f"\n{index}/{len(media_files)}")
        print(f"🎵 File detected: {audio_path.name}")

        try:
            _, txt_path, srt_path, _ = process_media_file(
                audio_path,
                model,
                output_dir,
            )
        except Exception as error:
            failed += 1
            print(f"❌ Failed to process {audio_path.name}: {error}")
            continue

        completed += 1
        print("✅ COMPLETE")
        print(f"\nTXT:\noutput/{txt_path.name}")
        print(f"\nSRT:\noutput/{srt_path.name}")

    print(f"\nFiles found: {len(media_files)}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()