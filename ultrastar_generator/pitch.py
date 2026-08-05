"""Small pitch-related helpers shared by note_detection.py and
lyric_alignment.py. The actual note segmentation lives in
note_detection.py -- this module is just utility functions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def hz_to_ultrastar_pitch(hz: float) -> int:
    midi = 69 + 12 * np.log2(hz / 440.0)
    return int(round(midi)) - 60


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def ultrastar_pitch_to_note_name(pitch: int) -> str:
    """Converts an UltraStar pitch value (MIDI - 60) back into a
    human-readable note name like "G#3", for diagnostics/debug output."""
    midi = pitch + 60
    name = _NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def median_pitch_in_span(
    y: np.ndarray, sr: int, start: float, end: float,
    fmin: float = 65.0, fmax: float = 1046.5,
) -> Optional[float]:
    """Best-effort pYIN median pitch (Hz) over a short, specific time span.

    Used only as a fallback for words that the primary, whole-track note
    detector didn't find any note for (e.g. very short/quiet function
    words). Less reliable than the whole-track pass since pYIN has less
    context to work with on a short clip.
    """
    import librosa

    i0 = max(0, int(start * sr))
    i1 = min(len(y), int(end * sr))
    if i1 - i0 < int(0.02 * sr):
        return None
    segment = y[i0:i1]
    try:
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            segment, sr=sr, fmin=fmin, fmax=fmax, frame_length=1024
        )
    except Exception:
        return None
    voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    voiced = voiced[~np.isnan(voiced)]
    if len(voiced) == 0:
        return None
    return float(np.median(voiced))
