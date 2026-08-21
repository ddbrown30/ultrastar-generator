"""Groups a flat syllable stream into lines, inserting LineBreak markers.

A known, matching reference line_id (from lyrics_lookup) wins over the gap/length heuristics below, even across a long pause. `strict_reference_lines=True` breaks only on a line_id change, no heuristics. Non-strict fallback (for syllables with no known line_id, or as a safety net): break on a silence gap >= MIN_LINE_GAP_SEC; else at the next word boundary past MAX_SYLLABLES_PER_LINE; else, once 1.5x over length, break near a known segment's own middle at trailing punctuation, or mechanically if no line_id."""

from __future__ import annotations

from typing import List, Optional, Set

from . import config
from .models import Syllable, LineBreak

_OVERFLOW_BREAK_PUNCTUATION = (",", ".")


def _segment_overflow_break_before(seg: List[Syllable]) -> Optional[int]:
    """Returns the segment-local syllable index to break before: the word-start nearest the segment's own middle whose preceding word ends in comma/period. None if no such interior punctuation exists (excludes the first and last words)."""
    word_start_indices = [i for i, s in enumerate(seg) if s.is_word_start]
    if len(word_start_indices) < 3:
        return None
    mid = len(seg) / 2.0
    candidates = []
    for wi in word_start_indices[1:-1]:
        prev_text = (seg[wi - 1].text or "").rstrip()
        if prev_text and prev_text[-1] in _OVERFLOW_BREAK_PUNCTUATION:
            candidates.append(wi)
    if not candidates:
        return None
    return min(candidates, key=lambda wi: abs(wi - mid))


def _find_reference_overflow_breaks(syllables: List[Syllable]) -> Set[int]:
    """Precomputes indices where an extra LineBreak is forced for reference-line segments that overflow `MAX_SYLLABLES_PER_LINE * 1.5`."""
    breaks: Set[int] = set()
    n = len(syllables)
    i = 0
    while i < n:
        lid = syllables[i].line_id
        if lid is None:
            i += 1
            continue
        j = i
        while j < n and syllables[j].line_id == lid:
            j += 1
        seg = syllables[i:j]
        if len(seg) >= int(config.MAX_SYLLABLES_PER_LINE * 1.5):
            local = _segment_overflow_break_before(seg)
            if local is not None:
                breaks.add(i + local)
        i = j
    return breaks


def build_lines(syllables: List[Syllable], strict_reference_lines: bool = False) -> List[object]:
    extra_breaks_before: Set[int] = (
        set() if strict_reference_lines else _find_reference_overflow_breaks(syllables)
    )

    entries: List[object] = []
    current_line_len = 0
    prev_end = None
    current_line_id = None

    for idx, syl in enumerate(syllables):
        if syl.is_word_start and prev_end is not None:
            line_id_changed = (
                current_line_id is not None
                and syl.line_id is not None
                and syl.line_id != current_line_id
            )

            if strict_reference_lines:
                force_break = line_id_changed
                soft_break = False
                hard_overflow = False
            else:
                known_same_line = (
                    current_line_id is not None
                    and syl.line_id is not None
                    and syl.line_id == current_line_id
                )
                gap = syl.start - prev_end

                if known_same_line:
                    # Still inside the same reference line -- only the length safety net can break it.
                    force_break = idx in extra_breaks_before
                    soft_break = False
                    hard_overflow = False
                else:
                    force_break = line_id_changed or gap >= config.MIN_LINE_GAP_SEC
                    soft_break = (
                        not line_id_changed
                        and current_line_len >= config.MAX_SYLLABLES_PER_LINE
                        and gap >= config.PREFERRED_LINE_GAP_SEC
                    )
                    hard_overflow = (
                        not line_id_changed
                        and current_line_len >= int(config.MAX_SYLLABLES_PER_LINE * 1.5)
                    )

            if force_break or soft_break or hard_overflow:
                entries.append(LineBreak(start=prev_end, end=syl.start))
                current_line_len = 0

        if syl.is_word_start and syl.line_id is not None:
            current_line_id = syl.line_id

        entries.append(syl)
        current_line_len += 1
        prev_end = syl.end

    return entries
