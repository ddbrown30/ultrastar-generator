"""ffmpeg-based media extraction: decoding a video container's audio track
for internal analysis, or extracting it into a real standalone mp3 for
output. Generalizes video_sync.py's own ffmpeg subprocess pattern (the
only ffmpeg call site in the repo before this module) into one shared
place, rather than duplicating the subprocess invocation a second time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import config

# Suppresses the console window Windows otherwise pops up for a real
# console-subsystem child process (ffmpeg/ffprobe) when the PARENT has no
# console of its own (e.g. the GUI, launched via pythonw.exe -- see
# run_gui.bat) -- confirmed real user-visible symptom ("console windows
# popping up during the pipeline run"). A no-op (flag 0) on non-Windows,
# where `subprocess.CREATE_NO_WINDOW` doesn't exist and this concept
# doesn't apply. Harmless when a console DOES already exist (CLI run from
# a terminal): the child already shares that console either way, and all
# output is captured via capture_output=True regardless of window
# visibility, so nothing is lost by suppressing a window that wasn't
# going to newly appear in that case anyway.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and proc.stdout.strip() != ""


def probe_duration_sec(path: Path) -> Optional[float]:
    """ffprobe-based container duration in seconds -- works uniformly
    across real audio files AND video containers (mp4/mpg/avi/etc, same
    ones resolve_primary_source can return), so a caller never needs to
    know which kind of file it's probing. Returns None (never raises) if
    ffprobe is missing, the file can't be probed, or the duration field
    is missing/unparseable -- callers must treat 'unknown' as a real
    possibility, never assume a number came back."""
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
        proc = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def strip_audio_track(src: Path, dst: Path) -> bool:
    """Copies src's video stream to dst with the audio track dropped (a
    stream copy, no re-encode -- fast, lossless). Used when a video
    companion's own audio track is redundant with the real #MP3 (a
    genuinely separate audio file, or extracted directly from this same
    video) and would only waste output size. Returns False (never
    raises) on any failure -- missing ffmpeg, no video stream, etc. --
    same graceful-degrade convention as extract_audio_track."""
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
