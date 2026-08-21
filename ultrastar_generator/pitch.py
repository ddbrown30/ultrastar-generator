"""Pitch helpers shared by note_detection.py and lyric_alignment.py."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import config


def hz_to_ultrastar_pitch(hz: float) -> int:
    midi = 69 + 12 * np.log2(hz / 440.0)
    return int(round(midi)) - 60


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def ultrastar_pitch_to_note_name(pitch: int) -> str:
    """Converts an UltraStar pitch value (MIDI - 60) into a note name like "G#3"."""
    midi = pitch + 60
    name = _NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def median_pitch_in_span(
    y: np.ndarray, sr: int, start: float, end: float,
    fmin: float = 65.0, fmax: float = 1046.5,
    pitch_source: Optional[str] = None,
) -> Optional[float]:
    """Best-effort median pitch (Hz) over a short span; last-resort fallback when no note or neighboring pitch is available."""
    from .note_detection import PITCH_SOURCES

    if pitch_source is None:
        pitch_source = config.DEFAULT_PITCH_SOURCE

    i0 = max(0, int(start * sr))
    i1 = min(len(y), int(end * sr))
    if i1 - i0 < int(0.02 * sr):
        return None
    segment = y[i0:i1]
    hop_length = 256
    n_frames = 1 + len(segment) // hop_length
    try:
        midi, _conf, voiced = PITCH_SOURCES[pitch_source](
            segment, sr, hop_length, 1024, fmin, fmax, n_frames,
        )
    except Exception:
        return None
    voiced = np.asarray(voiced, dtype=bool)
    voiced_midi = np.asarray(midi, dtype=float)[voiced]
    voiced_midi = voiced_midi[~np.isnan(voiced_midi)]
    if len(voiced_midi) == 0:
        return None
    median_midi = float(np.median(voiced_midi))
    return 440.0 * 2 ** ((median_midi - 69) / 12)
