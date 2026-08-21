"""ffmpeg-based media extraction: pulls a video container's audio track out for internal analysis or as a standalone mp3."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import config

# Suppresses console-window flashing when launched from the GUI (pythonw.exe); no-op on non-Windows.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def has_audio_stream(path: Path) -> bool:
    """Whether the container has any audio stream. Returns False (never raises) if ffprobe is missing or the file can't be probed."""
    if shutil.which("ffprobe") is None:
        return False
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and proc.stdout.strip() != ""


def probe_duration_sec(path: Path) -> Optional[float]:
    """Container duration in seconds (audio or video). Returns None (never raises) if ffprobe is missing or duration can't be determined."""
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def extract_audio_track(src: Path, dst: Path, *, as_mp3: bool = False, sr: Optional[int] = None) -> bool:
    """Extracts src's audio track to dst via ffmpeg: mono PCM wav (default) or a standalone mp3 (as_mp3=True). Returns False (never raises) on failure."""
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
        proc = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def strip_audio_track(src: Path, dst: Path) -> bool:
    """Copies src's video stream to dst with the audio track dropped (stream copy, no re-encode). Returns False (never raises) on failure."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "copy", "-an", str(dst),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0
