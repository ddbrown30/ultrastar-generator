"""Final safety net: guarantees that a sequence of Syllable entries never
has a note starting before the previous note has ended.

IMPORTANT: this does NOT sort by timestamp. The input order (word/reading
order from lyric_alignment.py) IS the correct semantic order -- that's
what determines the text you read left to right. Timestamps only affect
*where on the beat grid* each note lands and how long it holds, never
which order the words appear in.

An earlier version sorted by `start` here, on the theory that this was a
purely defensive, order-agnostic cleanup pass. In practice that was a
real bug: lyric_alignment.py can occasionally hand back a fallback note
(built from a word's own imprecise ASR timestamp, for a word that got no
notes of its own) whose timestamp doesn't quite line up with a
neighboring word's note-derived timestamp. Sorting by that timestamp
then silently reordered the WORDS themselves -- exactly the "words are
jumbled" bug reported in practice. Trusting the given order and only
pushing later entries forward when needed fixes this by construction.
"""

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
            line_id=syl.line_id,  # was silently dropped here (defaulted
                                    # to None on every call) before this
                                    # fix -- found while threading
                                    # confidence through the same
                                    # reconstruction; unrelated to pass 4
                                    # itself but a real bug worth fixing
                                    # since it's the same line of code.
            confidence=syl.confidence,
        ))
        prev_end = end

    return fixed
