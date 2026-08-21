"""Final safety net: pushes forward any note starting before the previous one ended.
Never sorts by timestamp -- the given (reading) order is the correct word order and must be preserved."""

from __future__ import annotations

from typing import List

from . import config
from .models import Syllable


def enforce_monotonic(syllables: List[Syllable]) -> List[Syllable]:
    min_gap = config.MIN_NOTE_GAP_SEC
    min_dur = config.MIN_NOTE_DURATION_SEC

    fixed: List[Syllable] = []
    prev_end = None
    for syl in syllables:
        start = syl.start
        end = syl.end

        if prev_end is not None and start < prev_end + min_gap:
            start = prev_end + min_gap

        if end <= start:
            end = start + min_dur

        fixed.append(Syllable(
            text=syl.text,
            start=start,
            end=end,
            midi_note=syl.midi_note,
            is_word_start=syl.is_word_start,
            note_type=syl.note_type,
            line_id=syl.line_id,
            confidence=syl.confidence,
        ))
        prev_end = end

    return fixed
