"""Glue module for pass 3 (lyric/word alignment): fits words onto the pass-1 note grid (lyric_alignment.align_words_to_notes), then applies reference-text corrections (verification.apply_reference_text), re-fitting if text changed."""

from __future__ import annotations

from typing import List

import numpy as np

from .models import Word
from .note_detection import NoteEvent
from .lyric_alignment import align_words_to_notes
from .verification import apply_reference_text


def align_words(
    words: List[Word],
    notes: List[NoteEvent],
    y: np.ndarray,
    sr: int,
    debug_log=None,
) -> tuple:
    """Pass 3 entry point: fits words onto the pass-1 note grid. Returns (syllables, stats) -- a flat, time-ordered Syllable list (not yet phrased into lines) and an AlignmentStats for diagnostics."""
    syllables, stats = align_words_to_notes(words, notes, y, sr, debug_log=debug_log)
    indices = [i for i in range(len(words)) if not words[i].dropped]  # dropped words never reach output
    if indices:
        corrected_words, override_results = apply_reference_text(words, indices)
        if override_results:
            words = corrected_words
            if debug_log is not None:
                debug_log.section("RE-RUNNING PASS 3 -- reference-text override corrected at least one word")
            syllables, stats = align_words_to_notes(words, notes, y, sr, debug_log=debug_log)
        stats.verification_results = override_results

    return syllables, stats
