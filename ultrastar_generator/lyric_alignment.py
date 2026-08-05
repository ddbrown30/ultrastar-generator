"""Pass 2 of the pipeline: fits transcribed words onto the note grid that
note_detection.py already built from the audio alone.

This is deliberately note-driven, not word-driven: pitch/timing accuracy
was the top priority in the reported bugs, so we treat the acoustically
detected notes as ground truth and fit lyric text onto them, rather than
deriving note boundaries from (less reliable) ASR word timestamps.

Algorithm:
  1. Words are grouped into consecutive runs sharing the same reference-
     lyrics line id (see lyrics_lookup.align_words_to_reference). A word
     with no line id (lyrics lookup unavailable/failed for it) is its own
     singleton group.
  2. The timeline is partitioned into one contiguous zone per GROUP, using
     boundaries at the midpoint between each group's ASR span and the
     next. Each detected note is assigned to whichever zone its midpoint
     falls in. This is deliberately NOT "each note independently picks
     whichever word overlaps it best" -- that let ASR timing imprecision
     assign a note to the wrong word/line, which then read as scrambled
     order once anything got re-sorted by time (a bug reported in
     practice). A zone partition is monotonic by construction.
  3. Within a MULTI-word group (a real matched reference line), the
     group's notes are NOT further subdivided using each interior word's
     own ASR timestamp -- individual in-line word timestamps turned out
     to be unreliable enough that a single bad one could swallow a large
     stretch of real, musically-distinct notes into one word (reported in
     practice: a whole passage collapsed into one giant melisma on the
     wrong word). Instead, the group's notes are split across its words
     PROPORTIONALLY BY SYLLABLE COUNT, in reading order -- e.g. a
     3-syllable word gets roughly 3x the notes of a 1-syllable word in
     the same line, regardless of what ASR thought each word's individual
     timing was. This only requires the coarser, more reliable group-level
     (line-level) timing to be right, not every interior word boundary.
     A singleton group (no line match) falls back to using that one
     word's own ASR span directly, same as before.
  4. Within a word's final note allocation, syllable count is reconciled
     against notes actually received (1:1, merged if the word has more
     syllables than notes, "~" melisma continuation if fewer).
  5. Words that end up with zero notes (short/quiet function words, or a
     line whose syllable count exceeded its note count) get a fallback
     note. Critically, this does NOT run a fresh, isolated pitch analysis
     on that word's own (often very short, e.g. <0.2s) ASR clip -- a real
     case of exactly that produced a wildly wrong note (confirmed against
     the pass-1 debug file: the bad note didn't exist in pass 1's output
     at all, it was fabricated here, from a noisy, context-starved
     re-analysis of a tiny clip). Instead, the fallback borrows the pitch
     of whichever pass-1 note (from the FULL, already-verified note list)
     is nearest in time -- reusing known-good information instead of
     manufacturing new, unverified pitch data.
  6. Everything is finally run through postprocess.enforce_monotonic, so
     no matter what happened above, the output can never contain
     overlapping notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import config
from .models import Word, Syllable
from .note_detection import NoteEvent
from .syllables import hyphenate
from .pitch import median_pitch_in_span, hz_to_ultrastar_pitch
from .postprocess import enforce_monotonic


@dataclass
class AlignmentStats:
    """Diagnostic summary of pass 2, so it's easy to tell how much of the
    output actually came from pass-1 note detection vs. fallback guesses."""
    total_words: int = 0
    words_with_notes: int = 0
    words_with_fallback: int = 0             # got zero pass-1 notes
    fallback_words: List[str] = field(default_factory=list)  # "text @ start"
    fallback_used_neighbor: int = 0            # fallback pitch borrowed from nearest pass-1 note
    fallback_used_fresh_analysis: int = 0      # fallback pitch from isolated re-analysis (no notes existed at all)
    words_with_melisma: int = 0               # fewer syllables than notes
    words_with_syllable_merge: int = 0        # more syllables than notes
    total_notes_consumed: int = 0
    lines_syllable_distributed: int = 0        # multi-word lines redistributed
                                                 # by syllable count, not ASR timing
    words_in_syllable_distributed_lines: int = 0


def _group_words_by_line(words: List[Word]) -> List[Tuple[Optional[int], List[int]]]:
    """Groups consecutive words sharing the same non-None line_id into
    line-groups, preserving order. A word with line_id None is always its
    own singleton group (never merged with a neighboring None -- those
    are unrelated, not "the same missing line")."""
    groups: List[Tuple[Optional[int], List[int]]] = []
    cur_line_id = None
    cur_indices: List[int] = []
    for i, w in enumerate(words):
        if w.line_id is not None and cur_indices and w.line_id == cur_line_id:
            cur_indices.append(i)
        else:
            if cur_indices:
                groups.append((cur_line_id, cur_indices))
            cur_line_id = w.line_id
            cur_indices = [i]
    if cur_indices:
        groups.append((cur_line_id, cur_indices))
    return groups


def _assign_notes_to_groups(
    group_spans: List[Tuple[float, float]], notes: List[NoteEvent]
) -> List[List[NoteEvent]]:
    """Same monotonic zone-partition technique as before, generalized to
    operate on (start, end) spans that may represent a single word OR a
    whole multi-word line."""
    n = len(group_spans)
    assigned: List[List[NoteEvent]] = [[] for _ in range(n)]
    if n == 0 or not notes:
        return assigned

    boundaries: List[float] = []
    running_max = float("-inf")
    for i in range(n - 1):
        raw = (group_spans[i][1] + group_spans[i + 1][0]) / 2.0
        running_max = max(running_max, raw, group_spans[i][0])
        boundaries.append(running_max)

    zone_idx = 0
    for note in notes:
        mid = (note.start + note.end) / 2.0
        while zone_idx < len(boundaries) and mid >= boundaries[zone_idx]:
            zone_idx += 1
        assigned[zone_idx].append(note)

    for group in assigned:
        group.sort(key=lambda note: note.start)
    return assigned


def _split_proportionally_by_syllables(
    notes: List[NoteEvent], syllable_counts: List[int]
) -> List[List[NoteEvent]]:
    """Splits a contiguous, time-ordered note list into len(syllable_counts)
    contiguous chunks, sized proportionally to each word's syllable count
    (cumulative rounding, so proportions stay accurate across the whole
    line rather than drifting from per-word rounding error). A word can
    legitimately end up with zero notes if the line has more words than
    notes -- that's handled by the normal per-word fallback afterward."""
    n_notes = len(notes)
    n_words = len(syllable_counts)
    if n_notes == 0 or n_words == 0:
        return [[] for _ in range(n_words)]

    total_syllables = sum(syllable_counts)
    weights = syllable_counts if total_syllables > 0 else [1] * n_words
    total_weight = sum(weights)

    boundaries = []
    cum = 0.0
    for w in weights:
        cum += w
        boundaries.append(round(cum / total_weight * n_notes))
    boundaries[-1] = n_notes

    chunks = []
    prev = 0
    for b in boundaries:
        b = max(b, prev)
        b = min(b, n_notes)
        chunks.append(notes[prev:b])
        prev = b
    return chunks


def _note_type_for(duration: float) -> str:
    return config.NOTE_GOLDEN if duration >= config.GOLDEN_NOTE_MIN_DURATION_SEC else config.NOTE_NORMAL


def _nearest_note_pitch(word: Word, all_notes: List[NoteEvent]) -> Optional[int]:
    """Finds the pass-1 note nearest in time to this word (by distance
    from the word's midpoint to the note's start or end, whichever is
    closer) and returns its pitch. Used for the fallback path so an
    unmatched word borrows already-verified pitch information instead of
    a fresh, isolated re-analysis of its own (often very short) clip."""
    if not all_notes:
        return None
    mid = (word.start + word.end) / 2.0
    nearest = min(all_notes, key=lambda n: min(abs(mid - n.start), abs(mid - n.end)))
    return nearest.pitch


def _chunk_syllables(parts: List[str], n_chunks: int) -> List[str]:
    """Merges a syllable list down to exactly n_chunks contiguous text
    chunks (used when a word has more syllables than the audio resolved
    distinct notes for)."""
    n_chunks = max(1, n_chunks)
    if n_chunks >= len(parts):
        return parts
    # Distribute parts across n_chunks as evenly as possible, in order.
    chunks = []
    base = len(parts) // n_chunks
    extra = len(parts) % n_chunks
    idx = 0
    for c in range(n_chunks):
        take = base + (1 if c < extra else 0)
        take = max(1, take)
        chunks.append("".join(parts[idx:idx + take]))
        idx += take
    return chunks


def _syllables_for_word(word: Word, notes: List[NoteEvent], all_notes: List[NoteEvent],
                         y: np.ndarray, sr: int, stats: AlignmentStats) -> List[Syllable]:
    if not notes:
        # Fallback: no acoustically-detected note overlapped this word at
        # all. Prefer borrowing the pitch of the nearest already-verified
        # pass-1 note over running a fresh, isolated pitch analysis on
        # this word's own (often very short) clip -- see module
        # docstring point 5 for why: the isolated analysis is a real
        # source of bad notes in practice.
        stats.words_with_fallback += 1
        stats.fallback_words.append(f'"{word.text}" @ {word.start:.2f}s')
        neighbor_pitch = _nearest_note_pitch(word, all_notes)
        if neighbor_pitch is not None:
            stats.fallback_used_neighbor += 1
            pitch = neighbor_pitch
        else:
            stats.fallback_used_fresh_analysis += 1
            hz = median_pitch_in_span(y, sr, word.start, word.end)
            pitch = hz_to_ultrastar_pitch(hz) if hz else 0
        end = max(word.end, word.start + config.MIN_NOTE_DURATION_SEC)
        return [Syllable(
            text=word.text, start=word.start, end=end,
            midi_note=pitch, is_word_start=True,
            note_type=_note_type_for(end - word.start),
            line_id=word.line_id,
        )]

    stats.words_with_notes += 1
    stats.total_notes_consumed += len(notes)
    parts = hyphenate(word.text)
    n_notes = len(notes)

    if len(parts) == n_notes:
        text_for_note = parts
    elif len(parts) > n_notes:
        stats.words_with_syllable_merge += 1
        text_for_note = _chunk_syllables(parts, n_notes)
    else:
        stats.words_with_melisma += 1
        # Melisma: fewer syllables than notes. Real text goes on the
        # notes belonging to each syllable in turn, then any leftover
        # trailing notes get the continuation marker.
        text_for_note = list(parts) + [config.MELISMA_CONTINUATION_TEXT] * (n_notes - len(parts))

    out: List[Syllable] = []
    for i, (text, note) in enumerate(zip(text_for_note, notes)):
        out.append(Syllable(
            text=text,
            start=note.start,
            end=note.end,
            midi_note=note.pitch,
            is_word_start=(i == 0),
            note_type=_note_type_for(note.end - note.start),
            line_id=word.line_id,
        ))
    return out


def align_words_to_notes(
    words: List[Word],
    notes: List[NoteEvent],
    y: np.ndarray,
    sr: int,
) -> tuple:
    """Top-level entry point for pass 2. Returns (syllables, stats):
    a flat, time-ordered, non-overlapping list of Syllable objects ready
    for phrasing.build_lines, plus an AlignmentStats summary for
    diagnostics/logging.
    """
    stats = AlignmentStats(total_words=len(words))
    if not words:
        return [], stats

    line_groups = _group_words_by_line(words)
    group_spans = [
        (words[indices[0]].start, words[indices[-1]].end)
        for _line_id, indices in line_groups
    ]
    group_notes = _assign_notes_to_groups(group_spans, notes)

    all_syllables: List[Syllable] = []
    for (line_id, indices), g_notes in zip(line_groups, group_notes):
        if len(indices) == 1:
            # Singleton group (no line match, or a one-word line): same as
            # the original per-word behavior, using that word's own notes.
            word = words[indices[0]]
            all_syllables.extend(_syllables_for_word(word, g_notes, notes, y, sr, stats))
            continue

        # Multi-word matched line: redistribute this line's notes across
        # its words by syllable count instead of trusting each interior
        # word's individual (unreliable) ASR timestamp.
        stats.lines_syllable_distributed += 1
        stats.words_in_syllable_distributed_lines += len(indices)
        syllable_counts = [len(hyphenate(words[i].text)) for i in indices]
        per_word_notes = _split_proportionally_by_syllables(g_notes, syllable_counts)
        for i, w_notes in zip(indices, per_word_notes):
            all_syllables.extend(_syllables_for_word(words[i], w_notes, notes, y, sr, stats))

    return enforce_monotonic(all_syllables), stats
