"""Pass 3: fits transcribed words onto pass 1's already-detected note grid.

Note-driven, not word-driven: notes are ground truth, lyric text is fit onto them.

Algorithm:
  1. Words are grouped into consecutive runs with no significant ASR gap between
     them (_group_words_by_gap) -- based on audio timing only, never reference
     line breaks (those only affect display '-' breaks, in phrasing.py).
  2. The timeline is partitioned into one monotonic zone per group (boundary =
     midpoint between consecutive groups' ASR spans); each note goes to whichever
     zone its midpoint falls in.
  3. Within a multi-word group, notes are split (never merged/moved) at each
     word's own ASR boundary, so a note spanning multiple words gets one
     same-pitch piece per word.
  4. Syllable count is reconciled against notes received (merged if fewer notes
     than syllables, "~" melisma continuation if more).
  5. A word left with zero notes borrows the nearest pass-1 note's pitch rather
     than running pitch analysis on its own short, unreliable clip.
  6. Everything runs through postprocess.enforce_monotonic as a final guarantee
     against overlapping notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from . import config
from .models import Word, Syllable
from .note_detection import NoteEvent
from .syllables import hyphenate, chunk_to_count
from .pitch import median_pitch_in_span, hz_to_ultrastar_pitch
from .postprocess import enforce_monotonic


def cap_word_durations_to_note_gaps(
    words: List[Word], notes: List[NoteEvent], max_gap_sec: float = config.NOTE_GROUP_REACH_MAX_GAP_SEC,
    debug_log=None,
) -> List[Word]:
    """Truncates a word's own END when pass-1's independently-detected notes show a genuine
    silence gap inside the word's claimed ASR span -- forced alignment can misattribute a real
    pause to the preceding word (real case: David Bowie - "I'm Afraid of Americans", "pretend"
    reported spanning 7.1 real seconds when pass-1's own notes show it actually ends after
    ~0.6s, followed by a genuine 3.2s silence gap). Left uncorrected, this doesn't just corrupt
    the word's own placement in pass 3 -- lyrics_lookup.recover_dropped_reference_words trusts
    an adjacent word's own timestamp as a hard window bound when force-aligning reference text
    ASR dropped entirely, so one bad word can also corrupt a whole separate phrase's own
    recovery window (real case, same song: the next phrase, "Johnny's in America", got force-
    aligned into a 1.16s window instead of its own real ~2s+ span, bounded by "pretend"'s own
    corrupted end on one side).

    Called once, early -- BEFORE recover_dropped_reference_words and any reference-lyrics
    correction -- so every downstream consumer sees the corrected timing, not just pass 3.

    Only ever shrinks a word's own end, never its start or any other word's timing; a word with
    no pass-1 note anywhere in its own span is left untouched (no independent evidence either
    way). `notes` must be sorted by start (pass-1's own output already is).

    A note is only counted as real evidence for THIS word if it actually STARTS at or after the
    word's own claimed start -- a note that merely trails in from before (the tail end of a
    neighboring word's own, legitimately-adjacent note) doesn't count, even if it technically
    overlaps by a sliver. Without this, two back-to-back words sharing an exact boundary (real
    case: a word recovered by force-aligning a dropped reference gap starting exactly where the
    PRECEDING word's own end was just capped to) can have that shared boundary point itself
    misread as "real evidence starting right at this word's own start," found immediately
    followed by a real gap, collapsing the word to zero length instead of correctly finding no
    reliable evidence at all and leaving it untouched."""
    if not words or not notes:
        return words
    out: List[Word] = []
    for w in words:
        if w.end <= w.start:
            out.append(w)
            continue
        prev_end: Optional[float] = None
        new_end = w.end
        for note in notes:
            if note.start >= w.end:
                break
            if note.start < w.start:
                continue
            if prev_end is not None and note.start - prev_end > max_gap_sec:
                new_end = prev_end
                break
            prev_end = note.end if prev_end is None else max(prev_end, note.end)
        if new_end < w.end:
            if debug_log is not None:
                debug_log.line(
                    f"[word-duration-capped] {w.text!r} {w.start:.3f}-{w.end:.3f}s "
                    f"({w.end - w.start:.2f}s) -> {w.start:.3f}-{new_end:.3f}s: pass-1's own notes show a "
                    f"real silence gap inside this word's claimed ASR span"
                )
            w = replace(w, end=new_end)
        out.append(w)
    return out


@dataclass
class AlignmentStats:
    """Diagnostic summary of pass 3: how much output came from pass-1 notes vs. fallback."""
    total_words: int = 0
    words_with_notes: int = 0
    words_with_fallback: int = 0             # got zero pass-1 notes
    fallback_words: List[str] = field(default_factory=list)  # "text @ start"
    fallback_used_neighbor: int = 0            # pitch borrowed from nearest pass-1 note
    fallback_used_fresh_analysis: int = 0      # pitch from isolated re-analysis
    words_with_melisma: int = 0               # fewer syllables than notes
    words_with_syllable_merge: int = 0        # more syllables than notes
    total_notes_consumed: int = 0
    lines_word_boundary_split: int = 0         # multi-word lines split by word boundary
    words_in_word_boundary_split_lines: int = 0
    suspicious_word_indices: List[int] = field(default_factory=list)  # fallback words, by index
    verification_results: List = field(default_factory=list)  # filled by alignment.align_words


def _group_words_by_gap(words: List[Word], max_gap_sec: float) -> List[List[int]]:
    """Groups consecutive words into one note-assignment group when the ASR gap
    between them is small enough to be the same phrase -- based on audio timing
    alone, never reference-lyrics line breaks (those only affect display '-'
    breaks in phrasing.py, independently of this grouping)."""
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


def _trim_notes_to_reachable_span(
    notes: List[NoteEvent], asr_start: float, asr_end: float, max_gap_sec: float,
) -> List[NoteEvent]:
    """Keeps only the notes reachable from [asr_start, asr_end] by a chain of notes each within
    max_gap_sec of its neighbor -- a genuine melisma tail (one continuous run of detected pitch,
    no real silence) is kept even far past an imprecise ASR timestamp, but a note on the far
    side of an actual silence gap this wide is dropped, since it more likely belongs to a
    separate, unrelated musical event (real case: see config.NOTE_GROUP_REACH_MAX_GAP_SEC's own
    docstring). `notes` must already be sorted by start time."""
    if not notes:
        return notes
    core = [i for i, n in enumerate(notes) if n.end > asr_start and n.start < asr_end]
    if not core:
        mid = (asr_start + asr_end) / 2.0
        core = [min(range(len(notes)), key=lambda i: min(abs(mid - notes[i].start), abs(mid - notes[i].end)))]
    lo, hi = min(core), max(core)
    while lo > 0 and notes[lo].start - notes[lo - 1].end <= max_gap_sec:
        lo -= 1
    while hi < len(notes) - 1 and notes[hi + 1].start - notes[hi].end <= max_gap_sec:
        hi += 1
    return notes[lo:hi + 1]


def _assign_notes_to_groups(
    group_spans: List[Tuple[float, float]], notes: List[NoteEvent], debug_log=None,
    group_labels: Optional[List[str]] = None,
) -> List[List[NoteEvent]]:
    """Monotonic zone-partition of notes across group spans (each may be a single
    word or a whole multi-word line)."""
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

    assigned = [
        _trim_notes_to_reachable_span(group, span[0], span[1], config.NOTE_GROUP_REACH_MAX_GAP_SEC)
        for group, span in zip(assigned, group_spans)
    ]

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
    notes: List[NoteEvent], word_spans: List[Tuple[float, float]],
) -> List[List[NoteEvent]]:
    """Distributes notes across word_spans, splitting (never adding/removing/moving)
    a note into same-pitch pieces wherever it crosses a word's ASR boundary. A word
    can end up with zero pieces, handled by the normal per-word fallback."""
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
    """Drops a word's leading note piece(s) shorter than
    config.SLIVER_DROP_MAX_DURATION_SEC (usually an ASR word-start landing early on
    an unvoiced consonant) rather than stretching the surviving note backward to
    cover them. Never drops a piece with protected_start=True (a confirmed
    re-articulation, not an artifact).

    Two tiers, since a genuine artifact and a genuinely-short-but-real fragment need different
    tests to tell apart:

    Tier 1 -- a leading piece under config.MIN_NOTE_GAP_SEC (10ms -- far below any real syllable
    fragment, but comfortably above the sub-millisecond slivers this catches) is always dropped,
    unconditionally: it's too short to be anything but a floating-point boundary artifact, not a
    real (if brief) note. Typically a sliver from an adjacent word's own note ending right at
    this word's own start (real case: a word recovered by force-aligning a dropped reference
    gap, starting exactly where the PRECEDING word's own end was capped to elsewhere -- the
    preceding word's last real note can leave a sub-millisecond trailing fragment right on that
    shared boundary).

    Tier 2 -- a leading piece under SLIVER_DROP_MAX_DURATION_SEC is dropped only while at least
    one piece AFTER it is itself substantial (>= the same threshold) -- otherwise there's no
    "real" note to protect by dropping the artifact, and popping anyway can strip a word made
    entirely of several genuinely brief fragments (a quickly-spoken word, not an ASR artifact)
    down to just its last, equally-brief piece. Real case, same song: "pretend" -- 3 real
    ~65-95ms fragments, none individually meeting the threshold, used to collapse to just the
    last one (75ms, ~1 beat) once an earlier fix correctly shrank the word's own claimed ASR
    span to its true short duration (before that fix, the word's own inflated 7s+ span gave this
    function plenty of later, longer pieces to fall back on, masking the bug)."""
    notes = list(word_notes)
    while (len(notes) > 1
           and (notes[0].end - notes[0].start) < config.MIN_NOTE_GAP_SEC
           and not notes[0].protected_start):
        notes.pop(0)
    while (len(notes) > 1
           and (notes[0].end - notes[0].start) < config.SLIVER_DROP_MAX_DURATION_SEC
           and not notes[0].protected_start
           and any((n.end - n.start) >= config.SLIVER_DROP_MAX_DURATION_SEC for n in notes[1:])):
        notes.pop(0)
    return notes


def _note_type_for(duration: float) -> str:
    return config.NOTE_GOLDEN if duration >= config.GOLDEN_NOTE_MIN_DURATION_SEC else config.NOTE_NORMAL


def _nearest_note_pitch(word: Word, all_notes: List[NoteEvent]) -> Optional[Tuple[int, float]]:
    """Finds the pass-1 note nearest in time to this word and returns (pitch,
    confidence), for a fallback word to borrow instead of re-analyzing its own
    short clip."""
    if not all_notes:
        return None
    mid = (word.start + word.end) / 2.0
    nearest = min(all_notes, key=lambda n: min(abs(mid - n.start), abs(mid - n.end)))
    return nearest.pitch, nearest.confidence



def _syllables_for_word(word: Word, notes: List[NoteEvent], all_notes: List[NoteEvent],
                         y: np.ndarray, sr: int, stats: AlignmentStats) -> List[Syllable]:
    if not notes:
        # No pass-1 note overlapped this word; borrow the nearest note's pitch
        # rather than analyzing this word's own short clip in isolation.
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
            confidence = 0.0  # no pass-1 evidence at all
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
        text_for_note = chunk_to_count(parts, n_notes)
    else:
        stats.words_with_melisma += 1
        # Melisma: fewer syllables than notes; leftover notes get the continuation marker.
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
    """Top-level entry point for pass 3. Returns (syllables, stats): a flat,
    non-overlapping list of Syllable objects for phrasing.build_lines, plus a
    diagnostics summary."""
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
    group_notes = _assign_notes_to_groups(
        group_spans, notes, debug_log=debug_log, group_labels=group_labels,
    )

    all_syllables: List[Syllable] = []
    for indices, g_notes in zip(word_groups, group_notes):
        if len(indices) == 1:
            # Singleton group: use that word's own notes directly.
            word = words[indices[0]]
            if word.dropped:
                # Hallucinated word: bounded its neighbors' zones but emits no
                # syllables; its claimed notes are discarded, not reassigned.
                continue
            if not g_notes:
                stats.suspicious_word_indices.append(indices[0])  # fallback word
            all_syllables.extend(_syllables_for_word(word, g_notes, notes, y, sr, stats))
            continue

        # Multi-word group: split notes across words at each word's own ASR boundary.
        stats.lines_word_boundary_split += 1
        stats.words_in_word_boundary_split_lines += len(indices)
        word_spans = [(words[i].start, words[i].end) for i in indices]
        n_notes = len(g_notes)
        per_word_notes = _split_notes_by_word_boundaries(g_notes, word_spans)
        for i, w_notes in zip(indices, per_word_notes):
            if not w_notes and not words[i].dropped:
                stats.suspicious_word_indices.append(i)  # about to fall back
        if debug_log is not None:
            debug_log.section(f"WORD-BOUNDARY SPLIT: {' '.join(words[i].text for i in indices)!r}")
            debug_log.line(f"  {n_notes} note(s) in this group's zone, split by ASR word boundaries")
            for i, w_notes in zip(indices, per_word_notes):
                span_str = f"({w_notes[0].start:.3f}, {w_notes[-1].end:.3f})" if w_notes else "(none -- fallback)"
                debug_log.line(f"    {words[i].text!r}: ASR=({words[i].start:.3f}, {words[i].end:.3f}) "
                                f"-> {len(w_notes)} note piece(s) {span_str}"
                                + (" [DROPPED -- no syllables emitted]" if words[i].dropped else ""))
        for i, w_notes in zip(indices, per_word_notes):
            if words[i].dropped:
                continue
            all_syllables.extend(_syllables_for_word(words[i], w_notes, notes, y, sr, stats))

    final = enforce_monotonic(all_syllables)

    if debug_log is not None:
        debug_log.section("FINAL SYLLABLES (this is what gets written to the .txt)")
        debug_log.line("Columns: start, end, duration, pitch, word_start, text")
        for s in final:
            debug_log.line(f"  {s.start:8.3f} - {s.end:8.3f}  ({s.end - s.start:6.3f}s)  "
                            f"pitch={s.midi_note:+3d}  word_start={str(s.is_word_start):5}  {s.text!r}")

    return final, stats
