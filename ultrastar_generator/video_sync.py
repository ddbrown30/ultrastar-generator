"""Estimates #VIDEOGAP (seconds to delay video playback) by cross-correlating video and song audio."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .media_extract import extract_audio_track


def estimate_videogap(
    video_path: Path,
    audio_path: Path,
    max_lag_sec: float = 20.0,
    compare_window_sec: float = 60.0,
) -> Optional[float]:
    """Returns the estimated VIDEOGAP in seconds, or None on failure."""
    import librosa
    from scipy.signal import correlate

    sr = 16000
    with tempfile.TemporaryDirectory() as tmp:
        video_wav = Path(tmp) / "video_audio.wav"
        if not extract_audio_track(video_path, video_wav, as_mp3=False, sr=sr):
            return None  # no audio track, or ffmpeg unavailable

        try:
            v, _ = librosa.load(str(video_wav), sr=sr, mono=True, duration=compare_window_sec)
            a, _ = librosa.load(str(audio_path), sr=sr, mono=True, duration=compare_window_sec)
        except Exception:
            return None

        if len(v) == 0 or len(a) == 0:
            return None

        # Normalize so amplitude differences don't dominate correlation.
        v = (v - np.mean(v)) / (np.std(v) + 1e-9)
        a = (a - np.mean(a)) / (np.std(a) + 1e-9)

        corr = correlate(a, v, mode="full")
        lag_samples = np.arange(-len(v) + 1, len(a))
        max_lag_samples = int(max_lag_sec * sr)

        mask = np.abs(lag_samples) <= max_lag_samples
        best_lag = lag_samples[mask][np.argmax(corr[mask])]

        offset_sec = best_lag / sr  # positive: video audio starts after song audio
        return round(float(offset_sec), 2)
