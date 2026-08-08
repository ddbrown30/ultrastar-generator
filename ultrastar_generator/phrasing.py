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
or as a safety net for a single reference line that's implausibly long.

Heuristic fallback, in priority order:
  1. A silence gap >= MIN_LINE_GAP_SEC before a word-start syllable always
     forces a break (it's audibly a new phrase).
  2. Otherwise, once a line has reached MAX_SYLLABLES_PER_LINE syllables,
     break at the next word boundary that has at least a small
     (PREFERRED_LINE_GAP_SEC) gap, to avoid mid-word splits.
  3. If neither applies but the line is *way* over length (1.5x max),
     force a break at the next word boundary anyway, so no single line
     runs on indefinitely.
"""

from __future__ import annotations

from typing import List

from . import config
from .models import Syllable, LineBreak


def build_lines(syllables: List[Syllable]) -> List[object]:
    entries: List[object] = []
    current_line_len = 0
    prev_end = None
    current_line_id = None

    for syl in syllables:
        gap = (syl.start - prev_end) if (prev_end is not None) else 0.0

        if syl.is_word_start and prev_end is not None:
            known_same_line = (
                current_line_id is not None
                and syl.line_id is not None
                and syl.line_id == current_line_id
            )
            line_id_changed = (
                current_line_id is not None
                and syl.line_id is not None
                and syl.line_id != current_line_id
            )

            if known_same_line:
                # Confirmed still inside the same reference line -- the
                # reference wins outright, even over a long silence gap
                # (real case: "Just a little change" has an audible pause
                # before "change" that used to force a spurious break
                # here). Only the implausible-length safety net still
                # applies.
                force_break = False
                soft_break = False
                hard_overflow = current_line_len >= int(config.MAX_SYLLABLES_PER_LINE * 1.5)
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
