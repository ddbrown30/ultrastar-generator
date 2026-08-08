"""ffmpeg-based media extraction: decoding a video container's audio track
for internal analysis, or extracting it into a real standalone mp3 for
output. Generalizes video_sync.py's own ffmpeg subprocess pattern (the
only ffmpeg call site in the repo before this module) into one shared
place, rather than duplicating the subprocess invocation a second time.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import config


def has_audio_stream(path: Path) -> bool:
    """ffprobe-based check for whether a container has any audio stream at
    all. Returns False (never raises) if ffprobe is missing OR the file
    can't be probed for any reason -- callers must treat 'unknown' the
    same as 'no audio', never proceed on an unverified assumption (this is
    specifically what feature 5's "abort if the avi has no audio" needs:
    a clean, confident answer before committing to an extraction attempt)."""
    if shutil.which("ffprobe") is None:
        return False
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and proc.stdout.strip() != ""


def extract_audio_track(src: Path, dst: Path, *, as_mp3: bool = False, sr: Optional[int] = None) -> bool:
    """Extracts src's audio track to dst via ffmpeg.

    as_mp3=False (default): decodes to mono PCM wav at `sr` Hz (16000 if
    not given) -- the format this project's own analysis code (librosa/
    Demucs) expects, used for feeding a video-derived source into pass 1/
    Demucs/WhisperX internally.

    as_mp3=True: encodes to a real standalone mp3 (libmp3lame, VBR quality
    config.AVI_EXTRACTED_MP3_QUALITY) -- an actual output-facing audio
    file, used when a video's audio track needs to become a real #MP3
    companion (feature 5's avi-with-no-matching-audio case), not just an
    internal analysis feed.

    Returns False (never raises) on any failure -- missing ffmpeg, no
    audio stream, encoder unavailable, etc. -- same graceful-degrade
    convention video_sync.py's own ffmpeg helper already used.
    """
    if shutil.which("ffmpeg") is None:
        return False

    if as_mp3:
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-codec:a", "libmp3lame", "-q:a", str(config.AVI_EXTRACTED_MP3_QUALITY),
            str(dst),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(sr or 16000), "-f", "wav", str(dst),
        ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0
