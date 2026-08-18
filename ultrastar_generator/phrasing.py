"""Groups a flat syllable stream into lines (phrases), inserting LineBreak
markers between them.

If reference-lyrics line info is available (Syllable.line_id, set by
lyrics_lookup.align_words_to_reference), a break is forced exactly where
the reference lyrics have one -- every '\\n' in the source becomes a '-'
in the output. This takes priority over the heuristics below: as long as
BOTH the current line and the next word have known, matching line_ids,
NOTHING (not even a long silence gap) breaks the line early -- a real
mid-line pause a singer takes is not a phrase boundary just because it's
long. The gap/length heuristics only apply to words with no line_id
(lyrics lookup disabled/failed, or this specific word went unmatched),
or (in the default, non-strict mode) as a safety net for a single
reference line that's implausibly long.

`strict_reference_lines=True` (see `build_lines`'s own parameter):
removes that safety net and every other heuristic entirely -- used when
every word's line_id came from confident, calibrated LRC line tracking
(see `lyrics_lookup.assign_lrc_line_ids_sequentially`). User's explicit directive
(2026-08-14): when using LRC, match it 100%, no exceptions. Real
motivating case: a long, melisma-heavy reference line (many held notes
per word) was getting split by the old MAX_SYLLABLES_PER_LINE*1.5 safety
net even though its line_id was confidently, correctly tracked
throughout -- real-audio validated (Trixie Mattel - Video Games) that
removing the safety net specifically for LRC-confident songs is a clear
net improvement (line-break agreement against ground truth 81.7%->87.8%,
spurious breaks 29->7), not just a wash.

Heuristic fallback (non-strict mode only), in priority order:
  1. A silence gap >= MIN_LINE_GAP_SEC before a word-start syllable always
     forces a break (it's audibly a new phrase).
  2. Otherwise, once a line has reached MAX_SYLLABLES_PER_LINE syllables,
     break at the next word boundary that has at least a small
     (PREFERRED_LINE_GAP_SEC) gap, to avoid mid-word splits.
  3. If neither applies but the line is *way* over length (1.5x max),
     the length safety net fires -- but ONLY for a KNOWN reference-line
     segment does it get to choose WHERE (see `_segment_overflow_break_
     before`'s own docstring): the whole segment (a contiguous run of
     syllables sharing one known line_id) is scanned up front for a
     punctuation-marked word near its own middle to break after; an
     overlong segment with no such punctuation is left whole, unbroken
     (2026-08-19, user's explicit request after a real case broke
     "...it's just an ordinary day," right before the trailing "day,"
     purely because that's where the running syllable count happened to
     cross the threshold, not because of anything about the text there).
     Content with no known line_id at all still breaks mechanically at
     the next word boundary once over-length -- there's no bounded
     segment to scan ahead of time for that content.
"""

from __future__ import annotations

from typing import List, Optional, Set

from . import config
from .models import Syllable, LineBreak

_OVERFLOW_BREAK_PUNCTUATION = (",", ".")


def _segment_overflow_break_before(seg: List[Syllable]) -> Optional[int]:
    """Given one reference-line segment (a contiguous run of Syllable
    sharing the same known line_id, already confirmed to overflow
    `MAX_SYLLABLES_PER_LINE * 1.5`), returns the segment-local syllable
    index to break BEFORE -- the word-start nearest the segment's own
    middle whose PRECEDING word ends in trailing comma/period
    punctuation. Returns None (don't break at all) when no such
    punctuation exists in the segment's INTERIOR.

    Deliberately excludes both the first word (breaking there would
    leave an empty first line) and the LAST word's own trailing
    punctuation specifically -- a reference line's own final word
    almost always ends in terminal punctuation, and that's the line's
    own sentence-ending mark, not a real interior clause boundary (real
    case: "I've got a smile on my face and I've got four walls around
    me," has exactly one comma, on its own final word -- the correct
    outcome is leaving this whole line unbroken, not splitting off
    "me," as its own one-word line)."""
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
    """Precomputes indices (into `syllables`) where an extra LineBreak
    should be forced for the length safety net, scoped to confidently-
    tracked reference-line segments (contiguous runs sharing the same
    known Syllable.line_id) that overflow `MAX_SYLLABLES_PER_LINE * 1.5`
    -- see `_segment_overflow_break_before`'s own docstring for the
    split-point rule. Content with no known line_id is left entirely to
    `build_lines`'s own streaming gap/length heuristics, unaffected by
    this."""
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
                # Every word's line_id is trusted completely -- break IFF
                # the reference line actually changed. No gap/length
                # heuristic may add OR suppress a break here.
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
                    # Confirmed still inside the same reference line -- the
                    # reference wins outright, even over a long silence gap
                    # (real case: "Just a little change" has an audible
                    # pause before "change" that used to force a spurious
                    # break here). Only the implausible-length safety net
                    # still applies, and only at the precomputed segment-
                    # level punctuation split point (if any) -- never
                    # mechanically at whichever word boundary the running
                    # count happens to cross.
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
