"""Glue module for pass 3 (lyric/word alignment). Pipeline order (see
main.py):

  pass 1 (note_detection.detect_notes) -> transcription/lyrics lookup ->
  pass 3 (this module) -> phrasing.build_lines().

This module's own steps:

  (notes, words) -> lyric_alignment.align_words_to_notes()  (fit lyrics onto
     the pass-1 note grid)
  -> verification.verify_words()  (optional, on by default: re-transcribes words
     in isolation, cross-checks against reference lyrics where available, and
     re-runs word-to-note fitting if any text changed)

Note: this module used to also offer a `verify_placement` re-check (cropped
a window at each word's FINAL note-assigned position and cross-checked it).
Removed 2026-08-10 -- real ground-truth comparison confirmed it a net
regression on every pitch/timing metric (see CLAUDE.md's "Removed /
rejected approaches"), even after further pipeline improvements; not worth
keeping around.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .models import Word
from . import config
from .note_detection import NoteEvent
from .lyric_alignment import align_words_to_notes
from .verification import verify_words as _verify_words_check


def align_words(
    words: List[Word],
    notes: List[NoteEvent],
    y: np.ndarray,
    sr: int,
    verify_words: bool = config.ENABLE_WORD_VERIFICATION,
    verify_all_words: bool = config.VERIFY_ALL_WORDS,
    verify_whisper_model: str = config.DEFAULT_WHISPER_MODEL,
    snap_boundaries: bool = config.ENABLE_ZONE_BOUNDARY_SNAP,
    snap_radius_sec: float = config.ZONE_BOUNDARY_SNAP_RADIUS_SEC,
    debug_log=None,
    verbose: bool = True,
) -> tuple:
    """Pass 3 entry point. Takes the ALREADY key-corrected note grid (pass
    2) and fits words onto it. Returns (syllables, stats) -- a flat,
    time-ordered list of Syllable objects (not yet phrased into lines --
    see main.py for phrasing), and an AlignmentStats for diagnostics/
    logging.

    `verify_whisper_model` is deliberately independent of whichever model
    drove the main transcription pass (main.py's --whisper-model, already
    baked into `words`' timestamps by this point) -- verify_words calls
    the model hundreds of times on tiny clips, where a big model's fixed
    per-call overhead dominates. Confirmed in practice that using a large
    model made the verify pass alone take ~10x longer than a small one;
    whether the re-check model's size actually matters much for accuracy
    (as opposed to the main pass's own timestamps, which this doesn't
    touch) hasn't been cleanly measured yet.

    `debug_log` (see debug_log.DebugLog) records the reference-line
    grouping, note-zone boundary math, and syllable-proportional split
    decisions -- pass None to skip (no-op either way if the DebugLog
    itself was constructed with path=None).
    """
    syllables, stats = align_words_to_notes(
        words, notes, y, sr, debug_log=debug_log,
        snap_boundaries=snap_boundaries, snap_radius_sec=snap_radius_sec,
    )
    if verify_words:
        # Dropped words (lyrics_lookup.align_words_to_reference found no
        # real reference correspondence at all) are never re-transcribed
        # here -- their text will never reach the final output regardless
        # of what a recheck says, so verifying them only wastes an ASR
        # call per word.
        candidate_indices = range(len(words)) if verify_all_words else stats.suspicious_word_indices
        indices = [i for i in candidate_indices if not words[i].dropped]
        if indices:
            corrected_words, verify_results = _verify_words_check(
                words, indices, y, sr, verify_whisper_model, verbose=verbose,
            )
            if any(r.replaced for r in verify_results):
                words = corrected_words
                if debug_log is not None:
                    debug_log.section("RE-RUNNING PASS 3 -- verify_words corrected at least one word")
                syllables, stats = align_words_to_notes(
                    words, notes, y, sr, debug_log=debug_log,
                    snap_boundaries=snap_boundaries, snap_radius_sec=snap_radius_sec,
                )
            stats.verification_results = verify_results

    return syllables, stats
