"""Small pitch-related helpers shared by note_detection.py and
lyric_alignment.py. The actual note segmentation lives in
note_detection.py -- this module is just utility functions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import config


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
    pitch_source: Optional[str] = None,
) -> Optional[float]:
    """Best-effort median pitch (Hz) over a short, specific time span, using
    whichever pitch source pass 1 itself uses (see
    note_detection.PITCH_SOURCES; defaults to config.DEFAULT_PITCH_SOURCE).

    Used only as a fallback for words that the primary, whole-track note
    detector didn't find any note for (e.g. very short/quiet function
    words) AND that have no neighboring note anywhere in the whole song to
    borrow a pitch from instead (see lyric_alignment.py's
    _nearest_note_pitch, which is tried first). Less reliable than the
    whole-track pass since the pitch source has far less context to work
    with on a short, isolated clip (see CLAUDE.md's "never run inference
    on a tiny isolated clip" lesson) -- this is a last-resort fallback,
    not a general-purpose short-clip pitch detector.
    """
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
