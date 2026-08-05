"""Glue module for pass 2 + phrasing. Pass 1 (note_detection.detect_notes)
is now called directly by main.py, not here -- that keeps the two passes
clearly separated (per the pipeline's own design principle: pass 1 builds
pitch/timing from audio alone, pass 2 only ever assigns words onto it,
never changes it) and lets main.py write the pass-1-only debug file
before pass 2 touches anything.

  (notes, words) -> lyric_alignment.align_words_to_notes()  (pass 2: fit lyrics onto notes)
  -> key_correction.snap_to_key()   (optional polish, off by default)
  -> phrasing.build_lines()
"""

from __future__ import annotations

from typing import List

import numpy as np

from .models import Word
from . import config
from .note_detection import NoteEvent
from .lyric_alignment import align_words_to_notes, AlignmentStats
from .phrasing import build_lines
from .key_correction import snap_to_key


def build_entries(
    words: List[Word],
    notes: List[NoteEvent],
    y: np.ndarray,
    sr: int,
    key_correction: bool = config.ENABLE_KEY_CORRECTION,
) -> tuple:
    """Returns (entries, stats) -- entries ready for usdx_writer, and an
    AlignmentStats for diagnostics/logging."""
    syllables, stats = align_words_to_notes(words, notes, y, sr)
    if key_correction:
        syllables = snap_to_key(syllables)
    return build_lines(syllables), stats
