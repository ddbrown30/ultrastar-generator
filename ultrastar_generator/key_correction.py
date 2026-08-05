"""Optional final polish pass: detect the song's most likely musical key
from its pitch-class distribution, and nudge notes that don't fit that
key to the nearest in-key neighbor.

This is inspired by the key-correction idea in the "ultrastar_pitch"
project (github: paradigm/ultrastar_pitch), which re-classifies pitch for
an existing, already-correctly-timed note file and snaps outliers using a
circle-of-fifths probability table. Our version is a from-scratch,
simpler implementation of the same underlying idea (best-fit diatonic
scale + nearest-neighbor snap), used here as a light cleanup pass on top
of our own pitch detection rather than as the primary pitch source.

Deliberately conservative: only snaps a note when doing so moves it by
exactly one semitone onto a scale tone, and only when the note isn't
already in the detected key. Larger disagreements are left alone, since
those are more likely to be genuine chromatic notes than tracking noise.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .models import Syllable

# Semitone offsets (0-11, root = 0) for natural major and natural minor.
_MAJOR = {0, 2, 4, 5, 7, 9, 11}
_MINOR = {0, 2, 3, 5, 7, 8, 10}


def _scale_for(root: int, mode: str) -> set:
    intervals = _MAJOR if mode == "major" else _MINOR
    return {(root + i) % 12 for i in intervals}


def detect_key(pitch_classes: List[int]) -> tuple:
    """Returns (root 0-11, mode) that best fits the given pitch-class
    histogram (each value already reduced mod 12)."""
    counts = np.zeros(12)
    for pc in pitch_classes:
        counts[pc % 12] += 1

    best = (0, "major")
    best_score = -1
    for root in range(12):
        for mode in ("major", "minor"):
            scale = _scale_for(root, mode)
            score = sum(counts[pc] for pc in scale)
            if score > best_score:
                best_score = score
                best = (root, mode)
    return best


def snap_to_key(syllables: List[Syllable]) -> List[Syllable]:
    if not syllables:
        return syllables

    pitch_classes = [s.midi_note % 12 for s in syllables]
    root, mode = detect_key(pitch_classes)
    scale = _scale_for(root, mode)

    # Pitch-class frequency across the whole song, used to break ties when
    # an out-of-scale note sits exactly one semitone from two in-scale
    # neighbors (the common case for a standard 7-note diatonic scale,
    # where every non-scale tone falls in a whole-tone gap): prefer
    # snapping toward whichever neighbor pitch class the song actually
    # uses more, rather than leaving it un-snapped.
    counts = np.zeros(12)
    for pc in pitch_classes:
        counts[pc] += 1

    out = []
    for s in syllables:
        pc = s.midi_note % 12
        if pc in scale:
            out.append(s)
            continue
        up = (pc + 1) % 12
        down = (pc - 1) % 12
        up_in = up in scale
        down_in = down in scale

        if up_in and not down_in:
            new_pitch = s.midi_note + 1
        elif down_in and not up_in:
            new_pitch = s.midi_note - 1
        elif up_in and down_in:
            # Equidistant (the common case in a diatonic scale): snap
            # toward whichever neighbor the song uses more often.
            new_pitch = s.midi_note + 1 if counts[up] >= counts[down] else s.midi_note - 1
        else:
            # Neither neighbor is in-key either (double chromatic) --
            # leave it alone rather than compounding a guess.
            out.append(s)
            continue
        out.append(Syllable(
            text=s.text, start=s.start, end=s.end,
            midi_note=new_pitch, is_word_start=s.is_word_start,
            note_type=s.note_type,
        ))
    return out
