"""Tempo (BPM) and vocal-onset (GAP) estimation.

UltraStar's #BPM is multiplied by 4 internally to get the note-beat grid
(see the format spec), so the value we write should be a plausible *real*
musical tempo, not the x4'd grid rate. We fold octave errors from the
detector into config.MIN_BPM..MAX_BPM.
"""

from __future__ import annotations

import math
from typing import Tuple

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
    """Duration of one UltraStar note-beat, in ms, per the format spec:
    the txt #BPM value is multiplied by 4 to get the real note-grid BPM."""
    grid_bpm = bpm_as_written * 4
    return 60000.0 / grid_bpm


def seconds_to_beat(t_sec: float, gap_ms: int, bpm_as_written: float) -> int:
    ms = t_sec * 1000.0
    return int(round((ms - gap_ms) / beat_duration_ms(bpm_as_written)))


def seconds_to_beat_length(duration_sec: float, bpm_as_written: float) -> int:
    """FLOORS (never rounds up) -- user's explicit request, 2026-08-10: a
    note that's a bit too SHORT is preferable to one that's a bit too
    LONG. round() would overshoot on any duration whose fractional beat
    count is >= 0.5; floor() always undershoots (or lands exact), never
    overshoots, at the cost of a slightly shorter note on average.
    `max(1, ...)` still guarantees every note gets written (never 0
    beats), same as before."""
    ms = duration_sec * 1000.0
    length = int(math.floor(ms / beat_duration_ms(bpm_as_written)))
    return max(1, length)


def beat_to_seconds(beat: int, gap_ms: int, bpm_as_written: float) -> float:
    """Inverse of seconds_to_beat -- used to parse an EXISTING .txt file's
    own beat numbers back into seconds, using the GAP/BPM values read from
    THAT file's own header (not necessarily this run's own detected
    values). See usdx_parser.py."""
    return (gap_ms + beat * beat_duration_ms(bpm_as_written)) / 1000.0
