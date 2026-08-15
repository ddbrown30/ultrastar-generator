"""Glue module for pass 3 (lyric/word alignment). Pipeline order (see
main.py):

  pass 1 (note_detection.detect_notes) -> transcription/lyrics lookup ->
  pass 3 (this module) -> phrasing.build_lines().

This module's own steps:

  (notes, words) -> lyric_alignment.align_words_to_notes()  (fit lyrics onto
     the pass-1 note grid)
  -> verification.apply_reference_text()  (always on, free: forces a word's
     text to match its reference_text wherever they disagree, and re-runs
     word-to-note fitting if any text changed)

Note: this module used to also offer a `verify_placement` re-check (cropped
a window at each word's FINAL note-assigned position and cross-checked it).
Removed 2026-08-10 -- real ground-truth comparison confirmed it a net
regression on every pitch/timing metric (see CLAUDE.md's "Removed /
rejected approaches"), even after further pipeline improvements; not worth
keeping around. Also used to offer an opt-out/opt-in re-TRANSCRIPTION
recheck (`verify_words`/`verify_all_words`/`verify_whisper_model`) gating
this same text-override -- removed 2026-08-15, see verification.py's own
docstring for why (the recheck never actually gated anything).
"""

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
    """Pass 3 entry point. Takes the ALREADY key-corrected note grid (pass
    2) and fits words onto it. Returns (syllables, stats) -- a flat,
    time-ordered list of Syllable objects (not yet phrased into lines --
    see main.py for phrasing), and an AlignmentStats for diagnostics/
    logging.

    `debug_log` (see debug_log.DebugLog) records the reference-line
    grouping, note-zone boundary math, and syllable-proportional split
    decisions -- pass None to skip (no-op either way if the DebugLog
    itself was constructed with path=None).
    """
    syllables, stats = align_words_to_notes(words, notes, y, sr, debug_log=debug_log)
    # Dropped words (lyrics_lookup.align_words_to_reference found no real
    # reference correspondence at all) are skipped -- their text never
    # reaches the final output regardless.
    indices = [i for i in range(len(words)) if not words[i].dropped]
    if indices:
        corrected_words, override_results = apply_reference_text(words, indices)
        if override_results:
            words = corrected_words
            if debug_log is not None:
                debug_log.section("RE-RUNNING PASS 3 -- reference-text override corrected at least one word")
            syllables, stats = align_words_to_notes(words, notes, y, sr, debug_log=debug_log)
        stats.verification_results = override_results

    return syllables, stats
