"""Pass 3 of the pipeline: fits transcribed words onto the note grid that
note_detection.py (pass 1) and key_correction.py (pass 2) already built
from the audio alone, with no lyrics involved.

This is deliberately note-driven, not word-driven: pitch/timing accuracy
was the top priority in the reported bugs, so we treat the acoustically
detected notes as ground truth and fit lyric text onto them, rather than
deriving note boundaries from (less reliable) ASR word timestamps.

Algorithm:
  1. Words are grouped into consecutive runs with no significant ASR gap
     between them (see _group_words_by_gap) -- deliberately based on
     AUDIO TIMING ALONE, never on reference-lyrics line breaks. Reference
     line breaks (Word.line_id, from lyrics_lookup.align_words_to_reference)
     control ONLY where a '-' appears in the output, independently, in
     phrasing.py -- they must never influence which notes a word gets.
     This was a real, confirmed bug: the SAME sung phrase can come back
     from lyrics.ovh split into a different number of lines in different
     verses of the same song, and forcing a note-assignment zone boundary
     at a line break with no real pause behind it swallowed one word
     several beats too long and pushed everything after it late.
  2. The timeline is partitioned into one contiguous zone per GROUP, using
     boundaries at the midpoint between each group's ASR span and the
     next. Each detected note is assigned to whichever zone its midpoint
     falls in. This is deliberately NOT "each note independently picks
     whichever word overlaps it best" -- that let ASR timing imprecision
     assign a note to the wrong word/line, which then read as scrambled
     order once anything got re-sorted by time (a bug reported in
     practice). A zone partition is monotonic by construction.
  3. Within a MULTI-word group (a real matched reference line), the
     group's notes are distributed across its words using each word's own
     ASR start/end time as a SPLIT point, not just a selection criterion:
     a single pass-1 note that spans more than one word's zone gets cut
     into multiple contiguous pieces (same pitch, same confidence) at the
     word boundary, one piece per word it overlaps. Notes are never
     added, removed, or moved -- only split. This is deliberately
     different from an earlier, rejected design that instead distributed
     EXISTING notes across words proportionally by syllable count without
     ever splitting one: that couldn't handle a real, confirmed case
     where "fall as Lucifer" was sung legato on one held pitch and pass 1
     (correctly) detected it as a single long note -- count-proportional
     distribution always gave that whole note to just one word, no matter
     how the syllable weights were tuned, because there was only ever one
     note to hand out. Splitting fixes this: the long note gets cut at
     each word's boundary and each word gets its own (correctly-pitched)
     slice. Word boundaries are computed the same clamped-monotonic
     midpoint technique as the group/zone boundaries above, so splits
     can't land out of order even if ASR word timestamps are imprecise.
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
    """Diagnostic summary of pass 3, so it's easy to tell how much of the
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
    lines_word_boundary_split: int = 0         # multi-word lines whose notes were
                                                 # split by each word's own ASR
                                                 # start/end time, not distributed
                                                 # by syllable count
    words_in_word_boundary_split_lines: int = 0
    suspicious_word_indices: List[int] = field(default_factory=list)  # into the
                                                 # `words` list passed to
                                                 # align_words_to_notes -- see
                                                 # verification.py
    verification_results: List = field(default_factory=list)  # filled in by
                                                 # alignment.build_entries after
                                                 # running verification.py
    placement_warnings: List = field(default_factory=list)  # filled in by
                                                 # alignment.align_words after running
                                                 # verification.verify_placement -- a
                                                 # word whose text is fine but whose FINAL
                                                 # note-assigned position doesn't match
                                                 # what's actually sung there, and where
                                                 # the mismatch couldn't be confidently
                                                 # auto-corrected (see placement_corrections)
    placement_corrections: List = field(default_factory=list)  # filled in by
                                                 # alignment.align_words after running
                                                 # verification.verify_placement -- a word
                                                 # whose FINAL note-assigned position was
                                                 # confirmed wrong AND precisely re-located,
                                                 # so its (start, end) was corrected and
                                                 # pass 3 was re-run with the fix applied


def _group_words_by_gap(words: List[Word], max_gap_sec: float) -> List[List[int]]:
    """Groups consecutive words into one NOTE-ASSIGNMENT group whenever the
    ASR gap between them is small enough to be the same sung phrase --
    deliberately based on AUDIO TIMING ALONE, never on reference-lyrics
    line breaks (Word.line_id).

    This used to group by line_id instead (every line_id change forced a
    new group, and therefore a new note-assignment zone boundary). That
    was a real, confirmed bug: reference lyrics text can split the SAME
    sung phrase into two lines inconsistently -- e.g. one occurrence of
    "And if they fall as Lucifer fell" arrived as two separate lines
    ("And if they fall" / "as Lucifer fell,") while a later, otherwise
    identical repetition arrived as one -- and forcing a zone boundary at
    that artificial line break (with no real pause in the audio to
    justify it) swallowed "fall" a full ~7 beats too long and pushed
    every word after it late. Reference line breaks are ONLY ever meant
    to control where a `-` appears in the output for display
    (phrasing.build_lines reads Syllable.line_id independently and is
    untouched by this function) -- never how notes/words get assigned.
    A word's own line_id is preserved untouched per-word by
    _syllables_for_word regardless of which group it ends up in here, so
    phrasing still breaks in exactly the right places even when this
    function merges words from two different reference lines into one
    zone-assignment group.
    """
    if not words:
        return []
    groups: List[List[int]] = [[0]]
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap <= max_gap_sec:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _assign_notes_to_groups(
    group_spans: List[Tuple[float, float]], notes: List[NoteEvent], debug_log=None,
    group_labels: Optional[List[str]] = None,
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

    if debug_log is not None:
        debug_log.section("NOTE-ZONE ASSIGNMENT (group spans -> boundaries -> notes per group)")
        debug_log.line("Zone boundary between group i and i+1 = midpoint of "
                        "(group[i]'s ASR end, group[i+1]'s ASR start), clamped monotonic.")
        for i, (span, notes_in_group) in enumerate(zip(group_spans, assigned)):
            label = group_labels[i] if group_labels else f"group {i}"
            b_before = boundaries[i - 1] if i > 0 else None
            b_after = boundaries[i] if i < len(boundaries) else None
            debug_log.line(f"  [{label}] ASR span=({span[0]:.3f}, {span[1]:.3f})  "
                            f"boundary_before={b_before}  boundary_after={b_after}  "
                            f"-> {len(notes_in_group)} note(s)"
                            + (f", spanning ({notes_in_group[0].start:.3f}, {notes_in_group[-1].end:.3f})"
                               if notes_in_group else ""))

    return assigned


def _split_notes_by_word_boundaries(
    notes: List[NoteEvent], word_spans: List[Tuple[float, float]]
) -> List[List[NoteEvent]]:
    """Distributes a contiguous, time-ordered note list across len(word_spans)
    words using each word's own ASR (start, end) as the cut point, SPLITTING
    a note into multiple same-pitch pieces whenever it spans more than one
    word's zone -- never adding, removing, or moving a note, only cutting
    one into contiguous pieces that still cover exactly the same total time.
    Word boundaries use the same clamped-monotonic midpoint technique as
    _assign_notes_to_groups, so an imprecise ASR timestamp can push a
    boundary later/earlier but never out of order. A word can legitimately
    end up with zero note pieces if it has no notes anywhere near it --
    handled by the normal per-word fallback afterward."""
    n_words = len(word_spans)
    if n_words == 0:
        return []
    if not notes:
        return [[] for _ in range(n_words)]

    boundaries: List[float] = []
    running_max = float("-inf")
    for i in range(n_words - 1):
        raw = (word_spans[i][1] + word_spans[i + 1][0]) / 2.0
        running_max = max(running_max, raw, word_spans[i][0])
        boundaries.append(running_max)

    buckets: List[List[NoteEvent]] = [[] for _ in range(n_words)]
    word_idx = 0
    for note in notes:
        while word_idx < len(boundaries) and boundaries[word_idx] <= note.start:
            word_idx += 1
        seg_start = note.start
        is_first_piece = True
        while word_idx < len(boundaries) and boundaries[word_idx] < note.end:
            b = boundaries[word_idx]
            buckets[word_idx].append(NoteEvent(
                start=seg_start, end=b, pitch=note.pitch,
                confidence=note.confidence,
                protected_start=note.protected_start if is_first_piece else False,
            ))
            seg_start = b
            is_first_piece = False
            word_idx += 1
        buckets[word_idx].append(NoteEvent(
            start=seg_start, end=note.end, pitch=note.pitch,
            confidence=note.confidence,
            protected_start=note.protected_start if is_first_piece else False,
        ))
    return [_drop_leading_slivers(bucket) for bucket in buckets]


def _drop_leading_slivers(word_notes: List[NoteEvent]) -> List[NoteEvent]:
    """Drops a word's leading note piece(s) whenever they're shorter than
    config.SLIVER_DROP_MAX_DURATION_SEC, instead of stretching a note to
    cover them.

    A tiny leading piece here is usually the ASR word-start timestamp
    landing a beat early on an unvoiced STRIDENT CONSONANT (s, f, etc.):
    there's no real pitch there at all, only noise, yet the word boundary
    still carves off a sliver of whatever pass-1 note happened to be
    playing just before the real singing starts. An earlier version of
    this function instead STRETCHED the surviving note's own start
    backward to cover that sliver -- exactly backwards for UltraStar,
    where a note's start tells the player when to begin vocalizing the
    pitch: it would make the player start singing during a consonant that
    has no pitch at all. So the sliver is simply DROPPED (that dead time
    becomes a gap, not folded into anything), and the next piece keeps
    its own pass-1-detected (start, end, pitch) completely unchanged --
    pass 1's real onset is trusted over the ASR word-start guess.

    Never drops a piece whose OWN protected_start is True, even if it's
    short enough to otherwise qualify -- that flag is pass-1's explicit
    signal that this note's start is a deliberate, confirmed
    re-articulation (see NoteEvent's docstring), not an artifact."""
    notes = list(word_notes)
    while (len(notes) > 1
           and (notes[0].end - notes[0].start) < config.SLIVER_DROP_MAX_DURATION_SEC
           and not notes[0].protected_start):
        notes.pop(0)
    return notes


def _note_type_for(duration: float) -> str:
    return config.NOTE_GOLDEN if duration >= config.GOLDEN_NOTE_MIN_DURATION_SEC else config.NOTE_NORMAL


def _nearest_note_pitch(word: Word, all_notes: List[NoteEvent]) -> Optional[Tuple[int, float]]:
    """Finds the pass-1 note nearest in time to this word (by distance
    from the word's midpoint to the note's start or end, whichever is
    closer) and returns (pitch, confidence). Used for the fallback path
    so an unmatched word borrows already-verified pitch information
    instead of a fresh, isolated re-analysis of its own (often very
    short) clip -- the borrowed note's own confidence comes along too,
    since a borrowed value is inherently less certain than a directly
    matched one and callers (musicxml_reference.py) should be able to
    tell the difference."""
    if not all_notes:
        return None
    mid = (word.start + word.end) / 2.0
    nearest = min(all_notes, key=lambda n: min(abs(mid - n.start), abs(mid - n.end)))
    return nearest.pitch, nearest.confidence


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
        neighbor = _nearest_note_pitch(word, all_notes)
        if neighbor is not None:
            stats.fallback_used_neighbor += 1
            pitch, confidence = neighbor
        else:
            stats.fallback_used_fresh_analysis += 1
            hz = median_pitch_in_span(y, sr, word.start, word.end)
            pitch = hz_to_ultrastar_pitch(hz) if hz else 0
            confidence = 0.0  # no real pass-1 evidence at all -- the least reliable case
        end = max(word.end, word.start + config.MIN_NOTE_DURATION_SEC)
        return [Syllable(
            text=word.text, start=word.start, end=end,
            midi_note=pitch, is_word_start=True,
            note_type=_note_type_for(end - word.start),
            line_id=word.line_id, confidence=confidence,
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
            confidence=note.confidence,
        ))
    return out


def align_words_to_notes(
    words: List[Word],
    notes: List[NoteEvent],
    y: np.ndarray,
    sr: int,
    debug_log=None,
) -> tuple:
    """Top-level entry point for pass 3. Returns (syllables, stats):
    a flat, time-ordered, non-overlapping list of Syllable objects ready
    for phrasing.build_lines, plus an AlignmentStats summary for
    diagnostics/logging.
    """
    stats = AlignmentStats(total_words=len(words))
    if not words:
        return [], stats

    word_groups = _group_words_by_gap(words, config.NOTE_ASSIGNMENT_MAX_GAP_SEC)
    group_spans = [
        (words[indices[0]].start, words[indices[-1]].end)
        for indices in word_groups
    ]

    if debug_log is not None:
        debug_log.section(f"NOTE-ASSIGNMENT GROUPING (gap-based, <= {config.NOTE_ASSIGNMENT_MAX_GAP_SEC}s -- "
                           f"purely audio timing, NOT reference-lyrics line breaks; those only ever "
                           f"control display '-' breaks, independently, in phrasing.py)")
        for indices, span in zip(word_groups, group_spans):
            text = " ".join(words[i].text for i in indices)
            line_ids = sorted({words[i].line_id for i in indices if words[i].line_id is not None})
            debug_log.line(f"  line_id(s)={line_ids}  words[{indices[0]}:{indices[-1]+1}]  "
                            f"ASR span=({span[0]:.3f}, {span[1]:.3f})  {text!r}")

    group_labels = [" ".join(words[i].text for i in indices) for indices in word_groups]
    group_notes = _assign_notes_to_groups(group_spans, notes, debug_log=debug_log, group_labels=group_labels)

    all_syllables: List[Syllable] = []
    for indices, g_notes in zip(word_groups, group_notes):
        if len(indices) == 1:
            # Singleton group (isolated by gaps on both sides): same as the
            # original per-word behavior, using that word's own notes.
            word = words[indices[0]]
            if not g_notes:
                # Zero pass-1 notes in this word's own zone -- a fallback,
                # and therefore suspicious (see verification.py).
                stats.suspicious_word_indices.append(indices[0])
            all_syllables.extend(_syllables_for_word(word, g_notes, notes, y, sr, stats))
            continue

        # Multi-word group (no significant gap between these words):
        # split the group's notes across its words using each word's own
        # ASR start/end time as the cut point -- a note spanning several
        # words gets divided into same-pitch pieces, never merged/moved.
        stats.lines_word_boundary_split += 1
        stats.words_in_word_boundary_split_lines += len(indices)
        word_spans = [(words[i].start, words[i].end) for i in indices]
        n_notes = len(g_notes)
        per_word_notes = _split_notes_by_word_boundaries(g_notes, word_spans)
        for i, w_notes in zip(indices, per_word_notes):
            if not w_notes:
                # No note overlapped this word's own ASR span -- it's
                # about to fall back, same as a singleton fallback word.
                stats.suspicious_word_indices.append(i)
        if debug_log is not None:
            debug_log.section(f"WORD-BOUNDARY SPLIT: {' '.join(words[i].text for i in indices)!r}")
            debug_log.line(f"  {n_notes} note(s) in this group's zone, split by ASR word boundaries")
            for i, w_notes in zip(indices, per_word_notes):
                span_str = f"({w_notes[0].start:.3f}, {w_notes[-1].end:.3f})" if w_notes else "(none -- fallback)"
                debug_log.line(f"    {words[i].text!r}: ASR=({words[i].start:.3f}, {words[i].end:.3f}) "
                                f"-> {len(w_notes)} note piece(s) {span_str}")
        for i, w_notes in zip(indices, per_word_notes):
            all_syllables.extend(_syllables_for_word(words[i], w_notes, notes, y, sr, stats))

    final = enforce_monotonic(all_syllables)

    if debug_log is not None:
        debug_log.section("FINAL SYLLABLES (post pass-2, this is what gets written to the .txt)")
        debug_log.line("Columns: start, end, duration, pitch, word_start, text")
        for s in final:
            debug_log.line(f"  {s.start:8.3f} - {s.end:8.3f}  ({s.end - s.start:6.3f}s)  "
                            f"pitch={s.midi_note:+3d}  word_start={str(s.is_word_start):5}  {s.text!r}")

    return final, stats
