"""Estimates #VIDEOGAP by cross-correlating the video's own audio track
(if it has one) against the song's audio file.

Per the format spec, #VIDEOGAP is the number of SECONDS to delay video
playback, i.e. positive means the video should start later relative to
the audio. If the video's audio leads the song audio by `offset` seconds,
the video needs to be delayed by that same `offset` to line back up, so
VIDEOGAP = offset.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


def _extract_audio_wav(src: Path, dst: Path, sr: int = 16000) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def estimate_videogap(
    video_path: Path,
    audio_path: Path,
    max_lag_sec: float = 20.0,
    compare_window_sec: float = 60.0,
) -> Optional[float]:
    """Returns the estimated VIDEOGAP in seconds, or None if the video has
    no audio track / extraction fails."""
    import librosa
    from scipy.signal import correlate

    sr = 16000
    with tempfile.TemporaryDirectory() as tmp:
        video_wav = Path(tmp) / "video_audio.wav"
        if not _extract_audio_wav(video_path, video_wav, sr=sr):
            return None  # no audio track in the video, or ffmpeg unavailable

        try:
            v, _ = librosa.load(str(video_wav), sr=sr, mono=True, duration=compare_window_sec)
            a, _ = librosa.load(str(audio_path), sr=sr, mono=True, duration=compare_window_sec)
        except Exception:
            return None

        if len(v) == 0 or len(a) == 0:
            return None

        # Normalize to avoid amplitude differences dominating correlation.
        v = (v - np.mean(v)) / (np.std(v) + 1e-9)
        a = (a - np.mean(a)) / (np.std(a) + 1e-9)

        corr = correlate(a, v, mode="full")
        lag_samples = np.arange(-len(v) + 1, len(a))
        max_lag_samples = int(max_lag_sec * sr)

        mask = np.abs(lag_samples) <= max_lag_samples
        best_lag = lag_samples[mask][np.argmax(corr[mask])]

        offset_sec = best_lag / sr  # positive: video's audio starts AFTER song audio
        return round(float(offset_sec), 2)
