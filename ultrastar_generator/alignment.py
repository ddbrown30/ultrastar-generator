"""Glue module for pass 3 (lyric/word alignment). Pipeline order (see
main.py):

  pass 1 (note_detection.detect_notes) -> pass 2 (key_correction.snap_to_key,
  optional, on by default -- operates on pass 1's raw NoteEvent list,
  BEFORE any lyrics exist) -> transcription/lyrics lookup -> pass 3 (this
  module) -> phrasing.build_lines().

This module's own steps:

  (notes, words) -> lyric_alignment.align_words_to_notes()  (fit lyrics onto
     the pass-2 note grid)
  -> verification.verify_words()  (optional, on by default: re-transcribes words
     in isolation, cross-checks against reference lyrics where available, and
     re-runs word-to-note fitting if any text changed)
  -> verification.verify_placement()  (optional, OFF by default: for each
     word, checks whether its FINAL note-assigned position actually
     matches what's sung there -- catches pass 3 putting a
     correctly-transcribed word on the wrong notes, which verify_words
     can't see since it only checks the word's original ASR timestamp.
     Crops a small window at the assigned position, transcribes it, and
     expands the window until the expected word is found (or gives up);
     once found, forced-alignment over that confirmed window pins down
     the exact position. When that position is precisely located (not
     just "somewhere in this window"), the word's own (start, end) is
     corrected to it and pass 3 re-runs with the corrected word list --
     same pattern as verify_words re-running pass 3 after a text
     correction, just for timing instead of text. Off by default: it's an
     expensive expand-search re-transcription loop over every word --
     see config.ENABLE_PLACEMENT_VERIFICATION.)

Key correction was deliberately moved OUT of this module (it used to run
here, on Syllable objects, as a final bundled step) and into its own pass
2, running on notes alone before word-fitting -- so it can never depend on
or affect which word gets which note, and so this module's own output can
be inspected or written out completely independently of whatever key
correction changes.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .models import Word
from . import config
from .note_detection import NoteEvent
from .lyric_alignment import align_words_to_notes, AlignmentStats
from .verification import verify_words as _verify_words_check, verify_placement as _verify_placement_check


def align_words(
    words: List[Word],
    notes: List[NoteEvent],
    y: np.ndarray,
    sr: int,
    verify_words: bool = config.ENABLE_WORD_VERIFICATION,
    verify_placement: bool = config.ENABLE_PLACEMENT_VERIFICATION,
    verify_all_words: bool = config.VERIFY_ALL_WORDS,
    verify_whisper_model: str = config.DEFAULT_WHISPER_MODEL,
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
    baked into `words`' timestamps by this point) -- verify_words/
    verify_placement call the model hundreds of times on tiny clips, where
    a big model's fixed per-call overhead dominates. Confirmed in practice
    that using a large model for both made the verify passes alone take
    ~10x longer than a small one; whether the re-check model's size
    actually matters much for accuracy (as opposed to the main pass's own
    timestamps, which this doesn't touch) hasn't been cleanly measured yet.

    `debug_log` (see debug_log.DebugLog) records the reference-line
    grouping, note-zone boundary math, and syllable-proportional split
    decisions -- pass None to skip (no-op either way if the DebugLog
    itself was constructed with path=None).
    """
    syllables, stats = align_words_to_notes(words, notes, y, sr, debug_log=debug_log)
    if verify_words:
        indices = list(range(len(words))) if verify_all_words else stats.suspicious_word_indices
        if indices:
            corrected_words, verify_results = _verify_words_check(
                words, indices, y, sr, verify_whisper_model, verbose=verbose,
            )
            if any(r.replaced for r in verify_results):
                words = corrected_words
                if debug_log is not None:
                    debug_log.section("RE-RUNNING PASS 2 -- verify_words corrected at least one word")
                syllables, stats = align_words_to_notes(words, notes, y, sr, debug_log=debug_log)
            stats.verification_results = verify_results

    if verify_placement:
        placement_indices = list(range(len(words))) if verify_all_words else stats.suspicious_word_indices
        if placement_indices:
            corrected_words, placement_corrections, placement_warnings = _verify_placement_check(
                words, syllables, placement_indices, y, sr, verify_whisper_model, verbose=verbose,
            )
            if placement_corrections:
                prior_verification_results = stats.verification_results
                words = corrected_words
                if debug_log is not None:
                    debug_log.section("RE-RUNNING PASS 2 -- verify_placement corrected at least one word's position")
                syllables, stats = align_words_to_notes(words, notes, y, sr, debug_log=debug_log)
                # align_words_to_notes always returns a fresh AlignmentStats --
                # carry forward verify_words' results from before this re-run.
                stats.verification_results = prior_verification_results
            stats.placement_corrections = placement_corrections
            stats.placement_warnings = placement_warnings
    return syllables, stats
