# Video and Audio Timestamp Generator

Transcribe audio and video files with OpenAI Whisper and save short timestamped
segments as both plain text and SRT subtitle files.

## Requirements

- Python 3.9 or newer
- [FFmpeg](https://ffmpeg.org/)

Install the Python dependency:

```bash
python -m pip install -r requirements.txt
```

## Usage

Put supported media files in `audio/`, then run:

```bash
python timestamp.py
```

The first run downloads the Whisper `base` model. Generated `.txt` and `.srt`
files are written to `output/`; existing output files are not overwritten.

Supported formats include MP3, WAV, M4A, MP4, MOV, AAC, FLAC, OGG, and WEBM.
