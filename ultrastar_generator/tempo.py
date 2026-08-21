"""Tempo (BPM) and vocal-onset (GAP) estimation. UltraStar's #BPM is multiplied by 4 internally for the note-beat grid, so the written value must be a real musical tempo, not the x4'd rate; octave errors are folded into config.MIN_BPM..MAX_BPM."""

from __future__ import annotations

import math

import numpy as np

from . import config


def _fold_into_range(bpm: float) -> float:
    while bpm < config.MIN_BPM:
        bpm *= 2
    while bpm > config.MAX_BPM:
        bpm /= 2
    return bpm


def detect_bpm(y: np.ndarray, sr: int) -> float:
    import librosa
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo <= 0:
        return config.FALLBACK_BPM
    return round(_fold_into_range(tempo), 2)


def beat_duration_ms(bpm_as_written: float) -> float:
    """Duration of one UltraStar note-beat, in ms (#BPM is multiplied by 4 for the real note-grid BPM)."""
    grid_bpm = bpm_as_written * 4
    return 60000.0 / grid_bpm


def seconds_to_beat(t_sec: float, gap_ms: int, bpm_as_written: float) -> int:
    ms = t_sec * 1000.0
    return int(round((ms - gap_ms) / beat_duration_ms(bpm_as_written)))


def seconds_to_beat_length(duration_sec: float, bpm_as_written: float) -> int:
    """Floors (never rounds up) -- a too-short note is preferable to a too-long one; always at least 1 beat."""
    ms = duration_sec * 1000.0
    length = int(math.floor(ms / beat_duration_ms(bpm_as_written)))
    return max(1, length)


def beat_to_seconds(beat: int, gap_ms: int, bpm_as_written: float) -> float:
    """Inverse of seconds_to_beat -- parses an existing .txt's beat numbers back into seconds using its own header GAP/BPM."""
    return (gap_ms + beat * beat_duration_ms(bpm_as_written)) / 1000.0
