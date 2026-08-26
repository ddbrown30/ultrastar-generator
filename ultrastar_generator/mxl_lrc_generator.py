"""Primary generation path: MXL supplies pitch/rhythm, LRC line starts anchor
real time, ASR of our own audio places words within each line (falling back
to proportional placement from MXL's own relative offsets when ASR can't
confidently match). Candidate validity is judged downstream by ASR/MXL
match rate (`MxlLrcQuality.asr_placement_rate`), not by upfront filtering.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

from . import config
from .lyrics_lookup import LrcLibCandidate, search_lrclib, effective_lrc_duration
from .lrc_timing import (parse_lrc, two_tier_time_calibration, match_asr_to_lrc_lines, lrc_line_window,
                          match_block_to_candidates, words_in_time_window, find_cursor_window_match)
from .models import Syllable, Word
from .syllables import hyphenate, chunk_to_count
from .text_normalize import (normalize_word as _normalize,
                              normalize_for_fuzzy_match as _normalize_fuzzy, is_all_filler)


@dataclass
class MxlWord:
    text: str
    norm: str
    offset: float  # quarter-note offset of first syllable
    syllables: List[Tuple[float, float, int, str]]  # (offset, quarterLength, midi, text)


def load_mxl_vocal_words(mxl_path: str, preferred_part_name: Optional[str] = None) -> Tuple[List[MxlWord], List[str]]:
    """Parses a MusicXML/.mxl file into whole words (syllables merged via each
    note's `syllabic` marker), unlike `musicxml_reference.load_vocal_notes`
    which stays at single-note granularity. Part selection: `preferred_part_name`
    if it names a real lyric-bearing part, else whichever has the most
    lyric-bearing notes; multiple lyric parts are never merged."""
    import music21

    score = music21.converter.parse(mxl_path)
    lyric_parts = []
    for part in score.parts:
        notes = list(part.flatten().notes)
        n_with_lyrics = sum(1 for n in notes if n.lyrics and not n.isChord)
        if n_with_lyrics > 0:
            lyric_parts.append((part, n_with_lyrics))

    if not lyric_parts:
        return [], []

    chosen_part = None
    if preferred_part_name is not None:
        chosen_part = next((p for p, _ in lyric_parts if p.partName == preferred_part_name), None)
    if chosen_part is None:
        chosen_part = max(lyric_parts, key=lambda t: t[1])[0]

    # Materialize notes up front so the lookahead OCR-repair pass below can inspect neighbors.
    raw_notes = []
    for n in chosen_part.flatten().notes:
        if n.isChord:
            continue
        text = syl = None
        for ly in n.lyrics:
            if ly.text:
                text, syl = ly.text, ly.syllabic
            break  # one lyric verse only
        raw_notes.append({
            "offset": float(n.offset), "dur": float(n.quarterLength),
            "midi": int(n.pitch.midi), "text": text, "syl": syl,
            "tied": n.tie is not None,
        })

    # A pure-punctuation note (e.g. stray ellipsis) isn't a real word: absorb its text
    # onto an immediately-contiguous preceding word, and make it a melisma continuation.
    for i, entry in enumerate(raw_notes):
        if not entry["text"] or _normalize(entry["text"]):
            continue
        if i == 0:
            continue
        prev = raw_notes[i - 1]
        if not prev["text"]:
            continue
        if abs(entry["offset"] - (prev["offset"] + prev["dur"])) > 1e-6:
            continue
        prev["text"] += entry["text"]
        entry["text"] = None
        entry["syl"] = None

    # Two words merged onto one note's lyric (e.g. "right,it's") via a missing engraver
    # split: recover by moving text after the internal comma/period/space onto the next
    # note, but only if that note has no lyric of its own already.
    merge_re = re.compile(r"^(.+?[,.\s])([A-Za-z0-9].*)$")
    for i, entry in enumerate(raw_notes):
        if not entry["text"]:
            continue
        m = merge_re.match(entry["text"])
        if not m:
            continue
        if i + 1 >= len(raw_notes) or raw_notes[i + 1]["text"]:
            continue
        entry["text"] = m.group(1)
        raw_notes[i + 1]["text"] = m.group(2)
        raw_notes[i + 1]["syl"] = "single"

    words: List[MxlWord] = []
    cur_syllables: List[Tuple[float, float, int, str]] = []
    cur_text = ""
    cur_offset = None

    def flush():
        nonlocal cur_syllables, cur_text, cur_offset
        if cur_syllables:
            words.append(MxlWord(text=cur_text, norm=_normalize(cur_text),
                                  offset=cur_offset, syllables=cur_syllables))
        cur_syllables = []
        cur_text = ""
        cur_offset = None

    for entry in raw_notes:
        if not entry["text"]:
            # No lyric on this note: if it's contiguous with the word in progress (no rest
            # between), it's a tied hold or slurred pitch change, not silence to discard.
            if cur_syllables:
                off, dur, prev_midi, text = cur_syllables[-1]
                if abs(entry["offset"] - (off + dur)) < 1e-6:
                    midi = entry["midi"]
                    if entry["tied"] and midi == prev_midi:
                        cur_syllables[-1] = (off, dur + entry["dur"], prev_midi, text)
                    else:
                        cur_syllables.append((entry["offset"], entry["dur"], midi, ""))
            continue
        syl = entry["syl"]
        if syl in (None, "single", "begin"):
            flush()
            cur_text = entry["text"]
            cur_offset = entry["offset"]
        else:
            cur_text += entry["text"]
        cur_syllables.append((entry["offset"], entry["dur"], entry["midi"], entry["text"]))
    flush()

    return words, [chosen_part.partName]


@dataclass
class LrcMatch:
    candidate: LrcLibCandidate
    lrc_lines: List[Tuple[float, str]]
    content_match_ratio: float
    duration_delta: Optional[float]


def _artist_matches(our_artist: str, candidate_artist: str) -> bool:
    """Whether the candidate's credited artist plausibly matches ours (substring match after
    normalization, either direction; empty on either side never matches)."""
    a = _normalize(our_artist)
    b = _normalize(candidate_artist)
    if not a or not b:
        return False
    return a in b or b in a


def select_lrc_candidate(artist: str, title: str, mxl_words: List[MxlWord], audio_duration: float,
                          forced: Optional[LrcLibCandidate] = None) -> Optional[LrcMatch]:
    """Picks an LRC candidate for timing. `forced` (user-pinned/--lrclib-id) is used
    directly, unfiltered. Otherwise searches LRCLIB (artist/title + free-text, deduped),
    requiring synced lyrics and content match >= `MXL_LRC_MIN_CONTENT_MATCH_RATIO`
    (deliberately permissive -- real validity is gated downstream, see module docstring).
    Duration is NEVER a hard filter here (real case: a SingStar-ripped clip can legitimately
    run 30+ seconds shorter than an OST candidate's own full-length recording while still
    being the correct, best-content-matching candidate) -- it only breaks ties in ranking,
    computed via `effective_lrc_duration`, not the raw LRCLIB field. Ranked by (1)
    `_artist_matches` decisively -- same-artist always beats different-artist regardless of
    ratio/duration, (2) a COARSE content-ratio bucket -- a meaningfully better content match
    always wins outright, never traded away for a closer duration (real case: our own artist
    tag can be a show/movie TITLE that's a substring of many different cast recordings'
    artist strings, so `_artist_matches` alone can't separate the real matching arrangement
    from a structurally different one), (3) duration proximity as the tiebreaker within that
    bucket, (4) exact ratio as the final tiebreaker. No artist match anywhere falls back to
    the same ratio-bucket-then-duration ranking."""
    mxl_norm_words = [w.norm for w in mxl_words if w.norm]

    if forced is not None:
        if not forced.synced_lyrics:
            return None
        lrc_lines = parse_lrc(forced.synced_lyrics)
        if not lrc_lines:
            return None
        lrc_norm = [_normalize(t) for t in (forced.plain_lyrics or "").split()]
        lrc_norm = [w for w in lrc_norm if w]
        ratio = difflib.SequenceMatcher(None, mxl_norm_words, lrc_norm, autojunk=False).ratio() if lrc_norm else 0.0
        forced_duration = effective_lrc_duration(forced)
        delta = abs(forced_duration - audio_duration) if forced_duration is not None else None
        return LrcMatch(candidate=forced, lrc_lines=lrc_lines, content_match_ratio=ratio, duration_delta=delta)

    candidates = search_lrclib(artist, title) + search_lrclib(q=title)
    seen = set()
    deduped = []
    for c in candidates:
        key = (c.track_name, c.artist_name, c.duration)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    scored = []
    for c in deduped:
        if c.instrumental or not c.synced_lyrics or c.duration is None:
            continue
        c_duration = effective_lrc_duration(c)
        delta = abs(c_duration - audio_duration)
        lrc_norm = [_normalize(t) for t in (c.plain_lyrics or "").split()]
        lrc_norm = [w for w in lrc_norm if w]
        if not lrc_norm:
            continue
        ratio = difflib.SequenceMatcher(None, mxl_norm_words, lrc_norm, autojunk=False).ratio()
        if ratio < config.MXL_LRC_MIN_CONTENT_MATCH_RATIO:
            continue
        scored.append((_artist_matches(artist, c.artist_name), ratio, delta, c))

    if not scored:
        return None
    # Ranked by (1) decisive artist match, (2) a COARSE content-ratio bucket (a large ratio gap,
    # e.g. a structurally different arrangement, must win outright -- see
    # MXL_LRC_CANDIDATE_RATIO_TIE_BUCKET's own docstring), (3) duration proximity as the
    # tiebreaker within that bucket, (4) exact ratio as the final tiebreaker.
    scored.sort(key=lambda t: (
        not t[0], -round(t[1] / config.MXL_LRC_CANDIDATE_RATIO_TIE_BUCKET), t[2], -t[1]))
    _artist_match, ratio, delta, best = scored[0]
    lrc_lines = parse_lrc(best.synced_lyrics)
    if not lrc_lines:
        return None
    return LrcMatch(candidate=best, lrc_lines=lrc_lines, content_match_ratio=ratio, duration_delta=delta)


def _merge_words_to_count(words: List[str], n_chunks: int) -> List[str]:
    """Merges `words` down to exactly n_chunks contiguous, space-joined chunks (word-level
    analogue of `syllables.chunk_to_count`, which joins with "" instead)."""
    n_chunks = max(1, n_chunks)
    if n_chunks >= len(words):
        return list(words)
    chunks = []
    base = len(words) // n_chunks
    extra = len(words) % n_chunks
    idx = 0
    for c in range(n_chunks):
        take = base + (1 if c < extra else 0)
        take = max(1, take)
        chunks.append(" ".join(words[idx:idx + take]))
        idx += take
    return chunks


def _slice_by_weights(text: str, weights: List[int]) -> Optional[List[str]]:
    """Slices `text` into len(weights) contiguous pieces sized proportionally to `weights`
    (each MXL syllable's own character length) -- recovers the notated split without a
    linguistic hyphenation guess. Returns None if `text` has fewer characters than slots
    (a genuine melisma, not a recoverable split)."""
    n = len(weights)
    if n <= 0 or len(text) < n:
        return None
    total = sum(weights) or n
    boundaries = [0]
    acc = 0
    for i, w in enumerate(weights[:-1]):
        acc += w
        remaining_slots = n - (i + 1)
        b = round(acc / total * len(text))
        b = max(b, boundaries[-1] + 1)
        b = min(b, len(text) - remaining_slots)
        boundaries.append(b)
    boundaries.append(len(text))
    return [text[boundaries[i]:boundaries[i + 1]] for i in range(n)]


def _distribute_words_to_slots(lrc_words_raw: List[str], n_slots: int,
                                mxl_slot_texts: Optional[List[str]] = None) -> List[str]:
    """Distributes a recovered stretch of raw LRC words across n_slots MXL word display
    slots (OCR word segmentation doesn't always match real word boundaries). Always
    returns exactly n_slots items: fewer real words are split via a literal hyphen first,
    then via `_slice_by_weights` against `mxl_slot_texts` (falling back to
    `syllables.hyphenate`), then melisma-padded if still short; more real words are merged
    evenly (`_merge_words_to_count`); equal counts assign 1:1."""
    words = list(lrc_words_raw)
    if len(words) < n_slots:
        expanded = []
        for w in words:
            pieces = [p for p in w.split("-") if p] if "-" in w else None
            expanded.extend(pieces if pieces else [w])
        words = expanded
    if len(words) == 1 and n_slots > 1:
        sliced = None
        if mxl_slot_texts and len(mxl_slot_texts) == n_slots:
            sliced = _slice_by_weights(words[0], [max(1, len(t)) for t in mxl_slot_texts])
        if sliced is not None:
            words = sliced
        else:
            parts = hyphenate(words[0])
            if len(parts) > 1:
                words = chunk_to_count(parts, n_slots) if len(parts) > n_slots else parts
    if len(words) > n_slots:
        words = _merge_words_to_count(words, n_slots)
    elif len(words) < n_slots:
        words = words + [config.MELISMA_CONTINUATION_TEXT] * (n_slots - len(words))
    return words


def _slice_time_by_weights(offset: float, dur: float, weights: List[int]) -> List[Tuple[float, float]]:
    """Splits one note's (offset, duration) span into len(weights) proportionally-sized
    sub-spans -- synthesizes note slots for a single note that OCR-merged multiple real
    words, using each target syllable's character length as the weight (time analogue of
    `_slice_by_weights`)."""
    total = sum(weights) or len(weights)
    spans = []
    acc = 0.0
    start = offset
    for i, w in enumerate(weights):
        acc += w
        end = offset + dur if i == len(weights) - 1 else offset + dur * (acc / total)
        end = max(end, start + 1e-6)
        spans.append((start, end - start))
        start = end
    return spans


def _flatten_real_syllables(words_slice: List[MxlWord]) -> List[Tuple[int, int]]:
    """Returns (word_local_idx, syllable_idx) for every real (non-empty) syllable across a
    slice of MXL words, in order."""
    flat = []
    for wi, w in enumerate(words_slice):
        for si, (_, _, _, text) in enumerate(w.syllables):
            if text:
                flat.append((wi, si))
    return flat


def _hyphenate_token(tok: str) -> List[str]:
    """Splits one recovered LRC token into syllable-shaped pieces: a literal hyphen wins
    over guessing, else falls back to `syllables.hyphenate`."""
    if "-" in tok:
        pieces = [p for p in tok.split("-") if p]
        if pieces:
            return pieces
    return hyphenate(tok)


MAX_PENDING_MXL_WORDS = 60  # mirrors lrc_timing.match_asr_to_lrc_lines's own cap


@dataclass
class MxlLineReconciliation:
    """Result of `reconcile_mxl_to_lrc_lines`."""
    line_mxl_range: Dict[int, Tuple[int, int]]   # lrc_line_index -> inclusive (start, end) mxl_words range
    dropped_lrc_lines: List[int]                 # lrc line indices with no confident MXL match
    orphan_mxl_runs: List[Tuple[int, int]]       # contiguous mxl_words ranges (inclusive) claimed by no line
    n_lrc_lines: int                             # non-empty lrc lines considered
    match_ratio: float                           # len(line_mxl_range) / n_lrc_lines
    longest_dropped_run: int                     # longest consecutive run of dropped lrc lines


def _best_edge_k(
    mxl_words: List[MxlWord], cursor: int, a1: int, a2: int, target_text: str, from_start: bool,
) -> Optional[int]:
    """Within MXL words [cursor+a1, cursor+a2), finds how many of them (counted from the START,
    from_start=True, or from the END, from_start=False) to include so their concatenated
    normalized text is the best available match for `target_text`. Grows the count one word at a
    time and stops as soon as another word makes the ratio WORSE (or reaches a perfect match) --
    a real notation-split word's own concatenation peaks at (or very near) an exact match once
    all its real pieces are included; further, unrelated adjacent content only ever degrades it.
    Returns None if even the best count never clears `MXL_LRC_FUZZY_TEXT_MIN_RATIO` at all."""
    max_k = min(a2 - a1, config.MXL_LRC_BLOCK_MAX_WORDS)
    best_k, best_ratio = None, -1.0
    for k in range(1, max_k + 1):
        idxs = range(a1, a1 + k) if from_start else range(a2 - k, a2)
        candidate = "".join(_normalize(mxl_words[cursor + i].text) for i in idxs)
        ratio = difflib.SequenceMatcher(None, candidate, target_text).ratio()
        if ratio < best_ratio:
            break
        best_k, best_ratio = k, ratio
        if ratio >= 0.999:
            break
    if best_k is not None and best_ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO:
        return best_k
    return None


def reconcile_mxl_to_lrc_lines(mxl_words: List[MxlWord], lrc_lines: List[Tuple[float, str]]) -> MxlLineReconciliation:
    """Assigns each LRC line a contiguous MXL-word range via a forward-only cursor over the MXL
    word stream (`lrc_timing.find_cursor_window_match`, the same mechanism `match_asr_to_lrc_lines`
    uses for ASR-vs-LRC), instead of one global whole-song diff -- a repeated phrase later in the
    MXL can never be confused with an earlier occurrence, since the cursor has already advanced
    past it. A line with no confident match is dropped (temporarily discarded, not force-matched);
    MXL words never claimed by any line become an "orphan run", recovered separately by
    `recover_orphan_mxl_runs`. This is the fix for a real, measured bug: the MXL score and a fetched
    LRC candidate can differ slightly in a repeated section (an extra/missing repeat, a dropped
    ad-lib), and a single global diff has no way to avoid matching the wrong occurrence once that
    happens -- the same "repeated-phrase disambiguation" failure class documented elsewhere in this
    project, just never fixed here before.

    Matching tolerates filler/ad-lib vocalise variation (`text_normalize.
    normalize_for_fuzzy_match` -- "ah-ah-ah" vs "na na na" transcribing the same real sound
    shouldn't count as a mismatch), except a line that's ENTIRELY filler is skipped outright
    (`is_all_filler`) -- no real content to safely anchor a match on."""
    mxl_norm = [_normalize_fuzzy(w.text) for w in mxl_words]
    n_mxl = len(mxl_words)
    cursor = 0
    pending_word_count = 0
    line_mxl_range: Dict[int, Tuple[int, int]] = {}
    dropped_lrc_lines: List[int] = []
    considered: List[int] = []

    for li, (_, text) in enumerate(lrc_lines):
        line_tokens_plain = [t for t in (_normalize(tok) for tok in text.split()) if t]
        if not line_tokens_plain or is_all_filler(line_tokens_plain):
            continue
        line_tokens = [_normalize_fuzzy(tok) for tok in text.split() if _normalize(tok)]
        considered.append(li)
        pending_word_count = min(pending_word_count + len(line_tokens), MAX_PENDING_MXL_WORDS)
        found = find_cursor_window_match(cursor, mxl_norm, line_tokens, pending_word_count)
        if found is None:
            # No match -- don't advance the cursor; next line's window grows to cover this one too.
            dropped_lrc_lines.append(li)
            continue
        opcodes, _window = found
        first_offset = last_offset = None
        for tag, a1, a2, _b1, _b2 in opcodes:
            if tag != "equal":
                continue
            if first_offset is None:
                first_offset = a1
            last_offset = a2 - 1
        # Extend the matched range to include a small leading/trailing "replace" block that
        # covers the line's own first/last target token -- e.g. the MXL notates "coming" as two
        # separate single-syllable "words" ("com"+"ing", a real score-engraving quirk), which a
        # plain "equal"-opcode check can't see on its own, but which clearly belongs to this line
        # since it's the line's own leading/trailing content. Mirrors the same "N MXL words
        # recover to 1 LRC word" shape `_reconcile_line_text_block` already handles WITHIN an
        # already-confirmed range -- this just lets the range include it in the first place.
        #
        # A "replace" opcode's own span can ALSO include real content that belongs to a
        # DIFFERENT, adjacent (often dropped) line -- e.g. a dropped "In 3D" line's own "3"/"d"
        # sitting immediately before a real "na"+"ture"="Nature" notation-split. A bare ratio (or
        # ratio+length) check on the WHOLE block isn't enough to reject this: "3"+"d"+"na"+"ture"
        # ("3dnature") still scores a deceptively high ratio against "nature" (a long shared
        # suffix), and even trimming to "d"+"na"+"ture" ("dnature") still does. `_best_edge_k`
        # instead finds the extension size that MAXIMIZES the match ratio and stops as soon as
        # adding another word makes it WORSE -- a real notation-split word's own concatenation
        # peaks at (or very near) a perfect match; genuinely unrelated adjacent content only ever
        # degrades it further, so the peak search naturally excludes it.
        for tag, a1, a2, b1, b2 in opcodes:
            if tag != "replace" or (a2 - a1) > config.MXL_LRC_BLOCK_MAX_WORDS:
                continue
            lrc_block = "".join(line_tokens_plain[b1:b2])
            if b1 == 0 and a1 < first_offset:
                k = _best_edge_k(mxl_words, cursor, a1, a2, lrc_block, from_start=False)
                if k is not None:
                    first_offset = a2 - k
            if b2 == len(line_tokens) and a2 - 1 > last_offset:
                k = _best_edge_k(mxl_words, cursor, a1, a2, lrc_block, from_start=True)
                if k is not None:
                    last_offset = a1 + k - 1
        start_i, end_i = cursor + first_offset, cursor + last_offset
        line_mxl_range[li] = (start_i, end_i)
        cursor = end_i + 1
        pending_word_count = 0

    claimed = [False] * n_mxl
    for start_i, end_i in line_mxl_range.values():
        for i in range(start_i, end_i + 1):
            claimed[i] = True
    orphan_mxl_runs: List[Tuple[int, int]] = []
    run_start = None
    for i in range(n_mxl):
        if not claimed[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            orphan_mxl_runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        orphan_mxl_runs.append((run_start, n_mxl - 1))

    dropped_set = set(dropped_lrc_lines)
    longest_run = cur_run = 0
    for li in considered:
        if li in dropped_set:
            cur_run += 1
            longest_run = max(longest_run, cur_run)
        else:
            cur_run = 0

    return MxlLineReconciliation(
        line_mxl_range=line_mxl_range, dropped_lrc_lines=dropped_lrc_lines,
        orphan_mxl_runs=orphan_mxl_runs, n_lrc_lines=len(considered),
        match_ratio=(len(line_mxl_range) / len(considered)) if considered else 1.0,
        longest_dropped_run=longest_run,
    )


def _neighbor_lines_for_orphan(
    orphan_range: Tuple[int, int], line_mxl_range: Dict[int, Tuple[int, int]],
) -> Tuple[Optional[int], Optional[int]]:
    """The confirmed line indices immediately before/after an orphan MXL range, by MXL word
    position (None on either side if the orphan is before the first / after the last match)."""
    start_i, end_i = orphan_range
    li_prev = li_next = None
    best_prev_end, best_next_start = -1, None
    for li, (s, e) in line_mxl_range.items():
        if e < start_i and e > best_prev_end:
            best_prev_end, li_prev = e, li
        if s > end_i and (best_next_start is None or s < best_next_start):
            best_next_start, li_next = s, li
    return li_prev, li_next


def recover_orphan_mxl_runs(
    mxl_words: List[MxlWord],
    lrc_lines: List[Tuple[float, str]],
    reconciliation: MxlLineReconciliation,
    asr_words: List[Word],
) -> Dict[int, Tuple[float, float]]:
    """Phase 2: tries to place each orphan MXL run (a stretch `reconcile_mxl_to_lrc_lines` couldn't
    tie to any LRC line -- e.g. the MXL notates a repeat/ad-lib the fetched LRC lacks, or vice versa)
    directly against real ASR, bounded to the real-time window its own neighboring CONFIRMED lines
    imply (`lrc_lines` is already time-calibrated by the time this runs, so even a DROPPED line's own
    timestamp is a trustworthy bound). This is the "go back to the skipped lines and look at the ASR
    between the two lines we did assign" recovery pass.

    An orphan run that can't be confidently recovered (no ASR words in its window, or too few of its
    own words confirmed) is simply left out of the returned dict -- the caller's existing nearest-
    line/interpolation fallback handles it exactly as it did before this function existed, so this is
    a pure add-on, never worse than not running it at all.

    Returns `{mxl_word_index: (start, end)}` for recovered real (non-empty) words only."""
    # Fallback width when no usable forward bound exists (see the two cases below) -- a modest
    # fixed pad, not a stretch to a distant/absent anchor. Real bug this fixed (Nature Trail to
    # Hell): li_next=None used to fall back to the LRC's own LAST line, which can be 100+ seconds
    # and 20+ dropped repeat-lines away, inviting the exact repeated-phrase mismatch risk a wide
    # window creates everywhere else in this project; li_next immediately adjacent to li_prev (no
    # dropped line between them) collapsed t0==t1, rejecting the orphan outright via `t1 <= t0`.
    FALLBACK_WINDOW_PAD_SEC = 5.0
    orphan_starts: Dict[int, Tuple[float, float]] = {}
    for orphan_range in reconciliation.orphan_mxl_runs:
        li_prev, li_next = _neighbor_lines_for_orphan(orphan_range, reconciliation.line_mxl_range)
        if li_prev is None and li_next is None:
            continue
        t0 = lrc_lines[li_prev + 1][0] if li_prev is not None and li_prev + 1 < len(lrc_lines) else lrc_lines[0][0]
        t1 = lrc_lines[li_next][0] if li_next is not None else None
        if t1 is None:
            # MXL itself has run out of notated content here -- every remaining LRC line is
            # dropped, so there's no real forward anchor to bound against at all.
            t1 = t0 + FALLBACK_WINDOW_PAD_SEC
        elif t1 <= t0:
            # li_next sits immediately adjacent to li_prev (no dropped line between them) --
            # search a small window straddling the boundary point instead of the collapsed span.
            t0 = max(0.0, t0 - FALLBACK_WINDOW_PAD_SEC)
            t1 = t0 + 2 * FALLBACK_WINDOW_PAD_SEC
        asr_in_window = words_in_time_window(asr_words, t0, t1)
        if not asr_in_window:
            continue
        start_i, end_i = orphan_range
        run_words = mxl_words[start_i:end_i + 1]
        real_positions = [i for i, w in enumerate(run_words) if w.norm]
        target_norm = [run_words[i].norm for i in real_positions]
        if not target_norm:
            continue
        # Exclude ASR words a neighboring, already-line-matched MXL word is itself expected to
        # claim -- an orphan's fuzzy match can otherwise steal an ASR word that merely resembles
        # it (e.g. "two" vs "to") but rightfully belongs to that other, already-confirmed word
        # (real case: Nature Trail to Hell's "two"/MXL vs "Part II"/LRC mismatch orphaned "two",
        # whose fuzzy match then stole "to"'s own ASR word instead of the real, unheard-of-by-
        # text-similarity "2" sitting right next to it -- see project memory).
        reserved_norms: set = set()
        for boundary_li in (li_prev, li_next):
            if boundary_li is not None and boundary_li in reconciliation.line_mxl_range:
                lo, hi = reconciliation.line_mxl_range[boundary_li]
                reserved_norms.update(mxl_words[j].norm for j in range(lo, hi + 1) if mxl_words[j].norm)
        reserved_norms -= set(target_norm)
        asr_candidates = [a for a in asr_in_window if _normalize(a.text) not in reserved_norms]
        if not asr_candidates:
            continue
        matched = match_block_to_candidates(target_norm, asr_candidates)
        if len(matched) < max(1, (len(target_norm) + 1) // 2):
            continue  # too sparse to trust -- leave this run to the existing fallback
        for local_idx, ridx in enumerate(real_positions):
            if local_idx not in matched:
                continue
            asr_w = matched[local_idx]
            orphan_starts[start_i + ridx] = (asr_w.start, asr_w.end)
    return orphan_starts


def _reconcile_line_text_block(
    mxl_words: List[MxlWord], start_i: int, end_i: int, lrc_raw_tokens: List[str],
    word_clean_text: dict, word_group: List[int], word_group_text: dict,
    word_syllable_override: dict, word_lrc_candidate: dict,
) -> None:
    """Within one already-confirmed MXL-range <-> LRC-line pairing (from
    `reconcile_mxl_to_lrc_lines`), recovers per-word/per-syllable clean display text -- MXL only
    supplies pitch/rhythm, never display text, when clean LRC text exists -- via a single diff
    bounded to just this line's own tokens, mutating the shared dicts in place. Earned via: an
    exact normalized match; a 1:1 "replace" pair clearing `MXL_LRC_FUZZY_TEXT_MIN_RATIO`; or a
    multi-word "replace" block (up to `MXL_LRC_BLOCK_MAX_WORDS`) whose whole concatenated text
    clears that ratio. Within a confirmed block, syllable-level reconciliation is tried first:
    flatten the block's real syllables (`_flatten_real_syllables`) and pair them positionally
    against target syllables (sliced via `_slice_by_weights` for one recovered word, or hyphenated
    per-token for several), each pair also gated on the fuzzy ratio. If the block is exactly one
    MXL word recovering multiple LRC words with no spare note slot, `_slice_time_by_weights`
    synthesizes new note slots from the one note's span (sharing its pitch) and the word's
    `syllables` are replaced wholesale. Otherwise falls through to word-level
    `_distribute_words_to_slots`.

    Also fills `word_group`/`word_group_text`: when several separate MXL words recover to exactly
    one real LRC word, they're grouped (`word_group[i]` = group's first index) so
    `place_words_via_asr` can search ASR for the whole word once instead of per-fragment.

    Also fills `word_lrc_candidate`: `{mxl_word_index: rejected LRC token}` for a 1:1 replace pair
    whose fuzzy ratio fell short. `place_words_via_asr` tries it against ASR; an independent ASR
    confirmation upgrades the word's display text to it.

    Being bounded to one already-line-confirmed MXL range means this diff can no longer reach
    across into a different repeat of the same line elsewhere in the song -- unlike the single
    whole-song diff this replaced."""
    mxl_slice = mxl_words[start_i:end_i + 1]
    mxl_norm = [w.norm for w in mxl_slice]
    lrc_flat_raw = [tok for tok in lrc_raw_tokens if _normalize(tok)]
    lrc_flat = [_normalize(tok) for tok in lrc_flat_raw]
    sm = difflib.SequenceMatcher(None, mxl_norm, lrc_flat, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        gi1 = start_i + i1
        if tag == "equal":
            for k in range(i2 - i1):
                word_clean_text[gi1 + k] = lrc_flat_raw[j1 + k]
        elif tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
            ratio = difflib.SequenceMatcher(None, mxl_norm[i1], lrc_flat[j1]).ratio()
            if ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO:
                word_clean_text[gi1] = lrc_flat_raw[j1]
            else:
                # Too different to trust directly; kept as an unconfirmed candidate for
                # place_words_via_asr to try against ASR.
                word_lrc_candidate[gi1] = lrc_flat_raw[j1]
        elif (tag == "replace" and (i2 - i1) <= config.MXL_LRC_BLOCK_MAX_WORDS
              and (j2 - j1) <= config.MXL_LRC_BLOCK_MAX_WORDS):
            mxl_block_norm = "".join(mxl_norm[i1:i2])
            lrc_block_norm = "".join(lrc_flat[j1:j2])
            ratio = difflib.SequenceMatcher(None, mxl_block_norm, lrc_block_norm).ratio()
            if ratio < config.MXL_LRC_FUZZY_TEXT_MIN_RATIO:
                continue

            # Syllable-level reconciliation first: a clean flattened-count match,
            # each pair fuzzy-verified too.
            block_words = mxl_slice[i1:i2]
            flat_real = _flatten_real_syllables(block_words)
            if j2 - j1 == 1:
                # One recovered LRC word spanning several MXL syllable slots
                # (e.g. "ne"+"ver" -> "never") -- slice by real per-syllable letter
                # lengths, not a linguistic hyphenation guess.
                real_weights = [max(1, len(block_words[wi].syllables[si][3])) for wi, si in flat_real]
                sliced = _slice_by_weights(lrc_flat_raw[j1], real_weights) if flat_real else None
                target_syllables: List[str] = sliced if sliced is not None else []
                # Only the first piece is a real word start; the rest continue it.
                target_is_new_word = [k == 0 for k in range(len(target_syllables))]
            else:
                target_syllables = []
                target_is_new_word = []
                for tok in lrc_flat_raw[j1:j2]:
                    pieces = _hyphenate_token(tok)
                    target_syllables.extend(pieces)
                    target_is_new_word.extend(k == 0 for k in range(len(pieces)))
            applied_syllable_level = False
            if flat_real and len(flat_real) == len(target_syllables):
                pair_ratios = [
                    difflib.SequenceMatcher(
                        None, _normalize(block_words[wi].syllables[si][3]), _normalize(piece)
                    ).ratio()
                    for (wi, si), piece in zip(flat_real, target_syllables)
                ]
                if all(r >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO for r in pair_ratios):
                    for (wi, si), piece, is_new in zip(flat_real, target_syllables, target_is_new_word):
                        gi = start_i + i1 + wi
                        if gi not in word_syllable_override:
                            word_syllable_override[gi] = [
                                (config.MELISMA_CONTINUATION_TEXT, False)] * len(block_words[wi].syllables)
                        word_syllable_override[gi][si] = (piece, is_new)
                    applied_syllable_level = True

            if not applied_syllable_level and (i2 - i1) == 1 and (j2 - j1) > 1:
                # One note OCR-merged multiple words with no spare slot to borrow:
                # synthesize new note slots from the note's own span, proportional to
                # each target syllable's character length, sharing the original pitch.
                only_word = block_words[0]
                o0 = only_word.syllables[0][0]
                d0 = (only_word.syllables[-1][0] + only_word.syllables[-1][1]) - o0
                midi0 = only_word.syllables[0][2]
                weights = [max(1, len(s)) for s in target_syllables]
                spans = _slice_time_by_weights(o0, d0, weights)
                only_word.syllables = [(off, dur, midi0, piece)
                                        for (off, dur), piece in zip(spans, target_syllables)]
                word_syllable_override[gi1] = list(zip(target_syllables, target_is_new_word))
                applied_syllable_level = True

            if not applied_syllable_level:
                assigned = _distribute_words_to_slots(
                    lrc_flat_raw[j1:j2], i2 - i1,
                    mxl_slot_texts=[w.text for w in block_words])
                for k in range(i2 - i1):
                    word_clean_text[gi1 + k] = assigned[k]

            if (i2 - i1) > 1 and (j2 - j1) == 1:
                # Several MXL words recovered to one real LRC word -- group them
                # so place_words_via_asr searches ASR for the whole word once.
                for k in range(i1, i2):
                    word_group[start_i + k] = gi1
                word_group_text[gi1] = lrc_flat_raw[j1]


def assign_words_to_lines(
        mxl_words: List[MxlWord],
        lrc_lines: List[Tuple[float, str]],
        reconciliation: Optional[MxlLineReconciliation] = None,
) -> Tuple[List[int], List[Optional[str]], List[int], dict, dict, dict]:
    """Assigns each MXL word to an LRC line index via `reconcile_mxl_to_lrc_lines`'s forward-cursor
    line-boundary matching (computed internally if `reconciliation` isn't given -- callers that also
    need it for orphan recovery, i.e. `generate_from_mxl_and_lrc`, compute it once and pass it in
    here to avoid a redundant pass). Words in a dropped/orphan region inherit the nearest preceding
    confirmed line (or the first confirmed line, if before it) -- same fallback this function always
    had, now only reached for a genuine MXL/LRC structural difference instead of any ordinary
    repeated-phrase ambiguity.

    Also returns a per-word clean-text replacement for display, `word_group`/`word_group_text`,
    `word_syllable_override`, and `word_lrc_candidate` -- see `_reconcile_line_text_block`, which
    does the actual per-line text reconciliation this function now delegates to (bounded to one
    confirmed line's own MXL range at a time, instead of one whole-song diff)."""
    n = len(mxl_words)
    if reconciliation is None:
        reconciliation = reconcile_mxl_to_lrc_lines(mxl_words, lrc_lines)

    word_line: dict = {}
    for li, (start_i, end_i) in reconciliation.line_mxl_range.items():
        for i in range(start_i, end_i + 1):
            word_line[i] = li

    word_clean_text: dict = {}
    word_group = list(range(n))  # identity by default -- each word its own group
    word_group_text: dict = {}
    word_syllable_override: dict = {}
    word_lrc_candidate: dict = {}
    for li, (start_i, end_i) in reconciliation.line_mxl_range.items():
        _reconcile_line_text_block(
            mxl_words, start_i, end_i, lrc_lines[li][1].split(),
            word_clean_text, word_group, word_group_text, word_syllable_override, word_lrc_candidate,
        )

    filled: List[Optional[int]] = [None] * n
    last = None
    for i in range(n):
        if i in word_line:
            filled[i] = word_line[i]
            last = word_line[i]
        else:
            filled[i] = last
    first_known = next((v for v in filled if v is not None), None)
    lines = [v if v is not None else first_known for v in filled]
    clean_text = [word_clean_text.get(i) for i in range(n)]
    return lines, clean_text, word_group, word_group_text, word_syllable_override, word_lrc_candidate


@dataclass
class MxlLrcQuality:
    n_words: int = 0
    n_asr_placed: int = 0
    n_fallback: int = 0
    non_monotonic_fix_count: int = 0
    # MXL-vs-LRC line reconciliation signal (reconcile_mxl_to_lrc_lines) -- detects when the MXL
    # score and the fetched LRC candidate disagree on structure (an extra/missing repeat, a dropped
    # ad-lib), independent of the ASR-vs-placement gates above. Diagnostic only, no hard gate.
    n_lrc_lines: int = 0
    n_lrc_lines_dropped: int = 0
    mxl_lrc_match_ratio: float = 1.0
    longest_dropped_lrc_run: int = 0

    @property
    def asr_placement_rate(self) -> float:
        return self.n_asr_placed / self.n_words if self.n_words else 0.0


def place_words_via_asr(mxl_words: List[MxlWord], word_lines: List[int], lrc_lines: List[Tuple[float, str]],
                         asr_words: List[Word],
                         word_clean_text: Optional[List[Optional[str]]] = None,
                         word_group: Optional[List[int]] = None,
                         word_group_text: Optional[dict] = None,
                         word_lrc_candidate: Optional[dict] = None,
                         preplaced: Optional[Dict[int, Tuple[float, float]]] = None,
                         ) -> Tuple[List[float], List[float], MxlLrcQuality]:
    """Pass -1: any word already placed by `recover_orphan_mxl_runs` (real ASR timing found for an
    MXL run that had no LRC line of its own at all) is taken as-is and confident -- Pass 0/1/2 never
    reconsider it.

    Pass 0: for each grouped block (`word_group`/`word_group_text`, several MXL words
    notated for one real word), searches the line window once for the whole recovered word
    via `match_block_to_candidates`, then distributes the matched span across member notes
    proportionally by quarterLength. Ungrouped/unmatched groups fall through to Pass 1/2.

    Pass 1: per LRC line, matches remaining MXL words against nearby-in-time ASR words
    (order-preserving difflib). Matches on `word_clean_text` when available, else the MXL's
    raw OCR norm; also accepts a close (not just exact) 1:1 "replace" pairing via fuzzy
    ratio, since ASR can mishear independently of MXL OCR. A match is trusted only if it
    also clears `MXL_LRC_MIN_ASR_WORD_CONFIDENCE`; start/end then come directly from the ASR
    word. A word with no clean text instead tries `word_lrc_candidate` (a previously-rejected
    LRC token) against ASR; if ASR confirms it, display text is upgraded too.

    Pass 2: every remaining word is placed by interpolating from its nearest CONFIDENT
    neighbors by MXL offset (not proportionally across the whole line, which trailing
    silence would distort). One-sided anchors extrapolate from that anchor's own nearest
    pair; no anchor anywhere falls back to whole-line-proportional placement. Result is
    clamped into the word's own LRC line window.

    Starts are then clamped non-decreasing; ends are clamped to never exceed the next
    word's start but may end earlier, preserving a real rest."""
    line_word_idxs: dict = {}
    for i, li in enumerate(word_lines):
        line_word_idxs.setdefault(li, []).append(i)

    n = len(mxl_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    confident: List[bool] = [False] * n
    quality = MxlLrcQuality(n_words=n)

    # --- Pass -1: orphan-run recovery already placed these directly; take as-is. ---
    grouped_handled: set = set()
    if preplaced:
        for i, (s, e) in preplaced.items():
            starts[i] = s
            ends[i] = e
            confident[i] = True
            quality.n_asr_placed += 1
            grouped_handled.add(i)

    # --- Pass 0: grouped multi-MXL-word blocks. ---
    if word_group and word_group_text:
        groups: dict = {}
        for i, gid in enumerate(word_group):
            groups.setdefault(gid, []).append(i)
        for gid, members in groups.items():
            members = [m for m in members if m not in grouped_handled]
            if len(members) <= 1 or gid not in word_group_text:
                continue
            li = word_lines[members[0]]
            t0, t1 = lrc_line_window(lrc_lines, li)
            asr_in_window = words_in_time_window(asr_words, t0, t1)
            target_norm = _normalize(word_group_text[gid])
            if not target_norm:
                continue
            matched = match_block_to_candidates([target_norm], asr_in_window)
            asr_w = matched.get(0)
            if asr_w is None:
                continue
            total_qtr = sum(sum(s[1] for s in mxl_words[m].syllables) for m in members) or len(members)
            asr_dur = asr_w.end - asr_w.start
            span = asr_dur if asr_dur > 0 else total_qtr * config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC
            cursor = asr_w.start
            for m in members:
                m_qtr = sum(s[1] for s in mxl_words[m].syllables) or 1.0
                frac = m_qtr / total_qtr
                starts[m] = cursor
                ends[m] = cursor + span * frac
                cursor = ends[m]
                confident[m] = True
                quality.n_asr_placed += 1
                grouped_handled.add(m)

    # --- Pass 1: confident ASR matches only. ---
    for li, idxs in line_word_idxs.items():
        idxs = sorted(i for i in idxs if i not in grouped_handled)
        t0, t1 = lrc_line_window(lrc_lines, li)
        asr_in_window = words_in_time_window(asr_words, t0, t1)
        mxl_norm_line = []
        used_candidate = []
        for i in idxs:
            if word_clean_text and word_clean_text[i]:
                mxl_norm_line.append(_normalize(word_clean_text[i]))
                used_candidate.append(False)
            elif word_lrc_candidate and word_lrc_candidate.get(i):
                mxl_norm_line.append(_normalize(word_lrc_candidate[i]))
                used_candidate.append(True)
            else:
                mxl_norm_line.append(mxl_words[i].norm)
                used_candidate.append(False)
        matched_local = match_block_to_candidates(mxl_norm_line, asr_in_window)

        for local_i, global_i in enumerate(idxs):
            if local_i not in matched_local:
                continue
            asr_w = matched_local[local_i]
            w = mxl_words[global_i]
            word_qtr_dur = sum(s[1] for s in w.syllables)
            starts[global_i] = asr_w.start
            asr_dur = asr_w.end - asr_w.start
            ends[global_i] = (asr_w.start + asr_dur if asr_dur > 0
                               else asr_w.start + word_qtr_dur * config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC)
            confident[global_i] = True
            quality.n_asr_placed += 1
            if used_candidate[local_i] and word_clean_text is not None:
                # ASR confirmed the rejected LRC candidate -- upgrade display text too.
                word_clean_text[global_i] = word_lrc_candidate[global_i]

    # Pass -1 (orphan-run recovery) is a fuzzy, potentially wide-window match -- if what it
    # placed conflicts in ORDER with a neighboring word Pass 0/1 just matched directly and
    # confidently, the orphan's match is more likely a wrong/stolen ASR word than the genuinely-
    # matched neighbor is (real case: Nature Trail to Hell's orphaned "two" fuzzy-matched an
    # earlier neighbor's own "to", landing before it -- see project memory). Demote it back to
    # unconfident so Pass 2's own nearest-anchor interpolation places it instead of corrupting
    # into a zero-length note via the later monotonic-order clamp.
    if preplaced:
        confident_order = sorted(i for i in range(n) if confident[i])
        for pos, i in enumerate(confident_order):
            if i not in preplaced:
                continue
            prev_i = confident_order[pos - 1] if pos > 0 else None
            next_i = confident_order[pos + 1] if pos + 1 < len(confident_order) else None
            if (prev_i is not None and starts[i] < starts[prev_i]) or \
                    (next_i is not None and starts[i] > starts[next_i]):
                confident[i] = False
                starts[i] = None
                ends[i] = None
                quality.n_asr_placed -= 1

    # --- Pass 2: nearest-confident-anchor interpolation for everything else. ---
    confident_idxs = [i for i in range(n) if confident[i]]

    def nearest_before(i: int) -> Optional[int]:
        best = None
        for ci in confident_idxs:
            if ci < i:
                best = ci
            else:
                break
        return best

    def nearest_after(i: int) -> Optional[int]:
        for ci in confident_idxs:
            if ci > i:
                return ci
        return None

    for i in range(n):
        if confident[i]:
            continue
        li = word_lines[i]
        t0, t1 = lrc_line_window(lrc_lines, li)
        w = mxl_words[i]
        word_qtr_dur = sum(s[1] for s in w.syllables)

        pb = nearest_before(i)
        pa = nearest_after(i)
        rate = None
        base_idx = None
        if pb is not None and pa is not None:
            off_delta = mxl_words[pa].offset - mxl_words[pb].offset
            if off_delta > 0:
                rate = (starts[pa] - starts[pb]) / off_delta
            base_idx = pb
        elif pb is not None:
            pbb = nearest_before(pb)
            if pbb is not None:
                off_delta = mxl_words[pb].offset - mxl_words[pbb].offset
                if off_delta > 0:
                    rate = (starts[pb] - starts[pbb]) / off_delta
            base_idx = pb
        elif pa is not None:
            paa = nearest_after(pa)
            if paa is not None:
                off_delta = mxl_words[paa].offset - mxl_words[pa].offset
                if off_delta > 0:
                    rate = (starts[paa] - starts[pa]) / off_delta
            base_idx = pa

        if rate is not None and base_idx is not None:
            base = mxl_words[base_idx]
            est_start = starts[base_idx] + (w.offset - base.offset) * rate
        else:
            # No anchor anywhere: fall back to whole-line-proportional placement.
            idxs = sorted(line_word_idxs[li])
            offs = [mxl_words[j].offset for j in idxs]
            lo_off, hi_off = min(offs), max(offs)
            span = hi_off - lo_off
            frac = (w.offset - lo_off) / span if span > 0 else 0.0
            est_start = t0 + frac * (t1 - t0)
            rate = (t1 - t0) / span if span > 0 else config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC

        # Backstop: never escape this word's own LRC line window.
        est_start = max(t0, min(est_start, t1))
        starts[i] = est_start
        ends[i] = est_start + word_qtr_dur * (rate if rate and rate > 0 else config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC)
        quality.n_fallback += 1

    for i in range(1, n):
        if starts[i] < starts[i - 1]:
            starts[i] = starts[i - 1]
            quality.non_monotonic_fix_count += 1

    # Ends must never overlap the next word's start, but may end earlier (a real rest).
    for i in range(n):
        if i + 1 < n and ends[i] > starts[i + 1]:
            ends[i] = starts[i + 1]
        if ends[i] < starts[i]:
            ends[i] = starts[i]

    return starts, ends, quality


def _text_for_mxl_syllables(clean_text: Optional[str], mxl_syllable_texts: List[str]) -> List[str]:
    """Returns the display text for each of an MXL word's syllable slots. Uses the MXL's
    own notated syllable split directly when it matches `clean_text` after normalization
    (more musically correct than `hyphenate`'s print-hyphenation dictionary). If not, slices
    `clean_text` by the MXL's own per-syllable letter-length weights (`_slice_by_weights`),
    falling back to `hyphenate` if that's not possible, or to the MXL's raw syllable text if
    no clean match exists. A melisma (<=1 real syllable slot) always keeps the whole clean
    word unsliced on its one real slot; other slots melisma-pad. Slicing is gated on
    `clean_text` plausibly being the same word (no internal whitespace, similarity clearing
    `MXL_LRC_FUZZY_TEXT_MIN_RATIO`) -- otherwise falls back to the MXL's raw text."""
    def _as_display(texts: List[str]) -> List[str]:
        # An untexted note ("") displays as the melisma-continuation marker, never blank.
        return [t if t else config.MELISMA_CONTINUATION_TEXT for t in texts]

    def _place_at_real_slots(pieces: List[str], real_idxs: List[int]) -> List[str]:
        result = [config.MELISMA_CONTINUATION_TEXT] * n
        for idx, piece in zip(real_idxs, pieces):
            result[idx] = piece
        return result

    n = len(mxl_syllable_texts)
    if clean_text is None:
        return _as_display(mxl_syllable_texts)

    mxl_joined = _normalize("".join(mxl_syllable_texts))
    clean_norm = _normalize(clean_text)
    if mxl_joined and mxl_joined == clean_norm:
        return _as_display(mxl_syllable_texts)

    real_idxs = [i for i, t in enumerate(mxl_syllable_texts) if t]
    n_real = len(real_idxs)
    if n_real <= 1:
        real_idx = real_idxs[0] if real_idxs else 0
        result = [config.MELISMA_CONTINUATION_TEXT] * n
        result[real_idx] = clean_text
        return result

    is_multi_word = any(c.isspace() for c in clean_text)
    ratio = difflib.SequenceMatcher(None, mxl_joined, clean_norm).ratio() if mxl_joined else 0.0
    if is_multi_word or ratio < config.MXL_LRC_FUZZY_TEXT_MIN_RATIO:
        return _as_display(mxl_syllable_texts)

    real_weights = [max(1, len(mxl_syllable_texts[i])) for i in real_idxs]
    sliced = _slice_by_weights(clean_text, real_weights)
    if sliced is not None:
        return _place_at_real_slots(sliced, real_idxs)

    parts = hyphenate(clean_text)
    if len(parts) > n_real:
        parts = chunk_to_count(parts, n_real)
    return _place_at_real_slots(parts, real_idxs)


def build_syllables(mxl_words: List[MxlWord], word_starts: List[float], word_ends: List[float],
                     word_lines: List[int], word_clean_text: List[Optional[str]],
                     word_syllable_override: Optional[dict] = None) -> List[Syllable]:
    """Splits each word's syllables proportionally within [word_start, word_end) using the
    MXL's own reliable relative sub-word offsets. Display text comes from `word_clean_text`
    via `_text_for_mxl_syllables`, unless `word_syllable_override` has an explicit per-slot
    override for that word (used directly instead, since it carries the correct cross-word
    split). `line_id` comes from `assign_words_to_lines`.

    Word-start marking normally comes from slot position (`syl_i == 0`), but that's wrong
    once `word_syllable_override` has moved syllables across a word boundary -- its entries
    are `(text, is_word_start)` pairs, used directly instead whenever present."""
    syllables: List[Syllable] = []
    for i, w in enumerate(mxl_words):
        t0 = word_starts[i]
        t1 = word_ends[i]
        if t1 <= t0:
            # Zero-width word: usdx_writer.py already has its own minimum-display-length
            # mechanism for this.
            t1 = t0
        lo = w.offset
        hi = w.offset + sum(s[1] for s in w.syllables)
        mxl_syllable_texts = [s[3] for s in w.syllables]
        if word_syllable_override and i in word_syllable_override:
            display_pairs = word_syllable_override[i]
        else:
            display_pairs = [(t, syl_i == 0)
                              for syl_i, t in enumerate(_text_for_mxl_syllables(word_clean_text[i], mxl_syllable_texts))]
        for (off, dur, midi, _orig_text), (text, is_start) in zip(w.syllables, display_pairs):
            frac0 = (off - lo) / (hi - lo) if hi > lo else 0.0
            frac1 = (off + dur - lo) / (hi - lo) if hi > lo else 1.0
            syllables.append(Syllable(
                text=text, start=t0 + frac0 * (t1 - t0), end=t0 + frac1 * (t1 - t0),
                midi_note=midi - 60, is_word_start=is_start, line_id=word_lines[i],
            ))
    return syllables


def calibrate_mxl_syllable_pitch(
    syllables: List[Syllable],
    vocals_path: Optional[str],
    min_calibration_samples: int = config.MUSICXML_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.MUSICXML_MIN_CALIBRATION_CONFIDENCE,
    force_calibration: bool = config.ENABLE_MUSICXML_FORCE_CALIBRATION,
    verbose: bool = True,
    debug_log=None,
) -> Tuple[List[Syllable], Optional[int], float]:
    """Corrects each syllable's pitch class (never octave or timing) against a per-song
    calibration offset, via the same shared logic as pass 4 and `pitch_refresh`'s MXL
    correction. This path skips pass 1, so it has no independent audio pitch to calibrate
    against without this step. Runs one whole-track pitch-class pass
    (`pitch_refresh.compute_pitch_class_predictions`) over the already-placed syllables'
    spans (never a per-word isolated clip) to get an independent reading; only `midi_note`
    is touched, never timing. No-op (unchanged, offset None) if `vocals_path`/`syllables`
    is missing."""
    if not vocals_path or not syllables:
        return syllables, None, 0.0

    import librosa

    from .musicxml_reference import _calibrate_pitch_class, nearest_pitch_for_class
    from .pitch_refresh import compute_pitch_class_predictions

    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)
    predicted_pc = compute_pitch_class_predictions(syllables, y, sr)

    calibration_samples: List[Tuple[int, int, float]] = []
    for syl, pred_pc in zip(syllables, predicted_pc):
        if pred_pc is None:
            continue
        mxl_absolute_midi = syl.midi_note + 60  # raw MXL pitch, pre-correction
        our_fake_pitch = pred_pc - 60  # (our_fake_pitch + 60) % 12 == pred_pc
        calibration_samples.append((our_fake_pitch, mxl_absolute_midi, syl.confidence))

    calibration, confidence, _, skipped_reason = _calibrate_pitch_class(
        calibration_samples, min_calibration_samples, min_calibration_confidence,
        force_calibration, verbose, log_prefix="[mxl-lrc]",
    )
    if calibration is None:
        if verbose:
            print(f"[mxl-lrc] pitch-class calibration skipped: {skipped_reason}")
        return syllables, None, 0.0

    new_syllables = list(syllables)
    n_corrected = 0
    for i, syl in enumerate(new_syllables):
        mxl_p = syl.midi_note + 60
        target_pc = (mxl_p - calibration) % 12
        new_pitch = nearest_pitch_for_class(syl.midi_note, target_pc)
        if new_pitch == syl.midi_note:
            continue
        n_corrected += 1
        new_syllables[i] = replace(
            syl, midi_note=new_pitch,
            confidence=max(syl.confidence, config.MUSICXML_CORRECTED_CONFIDENCE),
        )
        if debug_log is not None:
            debug_log.line(f"[mxl-lrc] {syl.text!r} @ {syl.start:.2f}s: pitch class corrected "
                            f"{syl.midi_note:+d} -> {new_pitch:+d} (target class {target_pc}, "
                            f"calibration {calibration:+d})")

    if verbose:
        print(f"[mxl-lrc] pitch-class calibration: {calibration:+d} semitone(s), {confidence:.0%} "
              f"agreement over {len(calibration_samples)} sample(s), {n_corrected} note(s) corrected")

    return new_syllables, calibration, confidence


@dataclass
class TimeCalibration:
    offset_sec: Optional[float] = None
    slope: float = 0.0
    confidence: float = 0.0
    kind: Optional[str] = None   # "constant", "drift", "piecewise", "isotonic", or None if uncalibrated
    skipped_reason: Optional[str] = None
    correction_fn: Optional[Callable[[float], float]] = None  # lrc_start -> corrected real time
    holdout_residual_sec: Optional[float] = None  # diagnostic-only, tier 3 ("piecewise"/"isotonic") only


@dataclass
class MxlLrcResult:
    success: bool
    reason: str
    syllables: List[Syllable] = field(default_factory=list)
    quality: Optional[MxlLrcQuality] = None
    lrc_match: Optional[LrcMatch] = None
    mxl_path: Optional[str] = None
    part_names_used: List[str] = field(default_factory=list)
    time_calibration: Optional[TimeCalibration] = None
    pitch_calibration_offset: Optional[int] = None  # semitones; None if skipped, see calibrate_mxl_syllable_pitch
    pitch_calibration_confidence: float = 0.0


def apply_gap_anchor_override(
    corrected_lines: List[Tuple[float, str]],
    time_candidates: List[Tuple[int, float, float]],
    tolerance_sec: float = config.GAP_ANCHOR_OVERRIDE_TOLERANCE_SEC,
    debug_log=None,
) -> List[Tuple[float, str]]:
    """GAP anchor safety net. Line 0 determines #GAP for the whole file, so an error there
    has a much larger blast radius than on any other line. If a direct real-ASR anchor for
    line 0 exists (`time_candidates[0][0] == 0`) and disagrees with the globally-calibrated
    version by more than `tolerance_sec`, the direct anchor wins; every other line keeps
    the normal global calibration untouched."""
    if not (time_candidates and time_candidates[0][0] == 0):
        return corrected_lines
    _, direct_lrc_start, direct_delta = time_candidates[0]
    direct_anchor = direct_lrc_start + direct_delta
    calibrated_start = corrected_lines[0][0]
    if abs(calibrated_start - direct_anchor) <= tolerance_sec:
        return corrected_lines
    msg = (f"[mxl-lrc] GAP anchor override: line 0's calibrated start "
           f"({calibrated_start:.3f}) disagrees with its own direct ASR anchor "
           f"({direct_anchor:.3f}) by {abs(calibrated_start - direct_anchor):.2f}s "
           f"(> {tolerance_sec}s) -- using the direct anchor instead, since it corrupts "
           f"#GAP for the whole file otherwise")
    print(msg)
    if debug_log is not None:
        debug_log.line(msg)
    return [(direct_anchor, corrected_lines[0][1])] + corrected_lines[1:]


def generate_from_mxl_and_lrc(mxl_path: str, artist: str, title: str, audio_duration: float,
                               asr_words: List[Word], forced_candidate: Optional[LrcLibCandidate] = None,
                               preferred_part_name: Optional[str] = None,
                               vocals_path: Optional[str] = None, debug_log=None) -> MxlLrcResult:
    """Orchestrates the full MXL+LRC generation for one MusicXML file and applies the
    quality gate. Never raises on expected failure modes -- returns a failed `MxlLrcResult`
    with a human-readable `reason` instead."""
    mxl_words, part_names = load_mxl_vocal_words(mxl_path, preferred_part_name)
    if not mxl_words:
        return MxlLrcResult(success=False, reason=f"{mxl_path}: no lyric-bearing part found", mxl_path=mxl_path)

    lrc_match = select_lrc_candidate(artist, title, mxl_words, audio_duration, forced=forced_candidate)
    if lrc_match is None:
        return MxlLrcResult(success=False, reason="no matching synced lyrics found on LRCLIB",
                             mxl_path=mxl_path, part_names_used=part_names)

    # Calibrate away a systematic time offset between LRC line timestamps and our audio's
    # real timing (e.g. extra lead-in silence); a null/near-zero calibration is a no-op.
    # No `structural_check` is passed (no native line grouping to build one from), so tier
    # 3's "rescue" case always declines here -- the ASR-placement-rate gate below is the
    # backstop.
    time_candidates = match_asr_to_lrc_lines(asr_words, lrc_match.lrc_lines)
    offset, slope, confidence, kind, skipped_reason, correction_fn, holdout = two_tier_time_calibration(
        time_candidates)
    time_cal = TimeCalibration(offset_sec=offset, slope=slope, confidence=confidence,
                                kind=kind, skipped_reason=skipped_reason, correction_fn=correction_fn,
                                holdout_residual_sec=holdout)
    if offset is not None:
        corrected_lines = [(correction_fn(t), text) for t, text in lrc_match.lrc_lines]
        lrc_match.lrc_lines = apply_gap_anchor_override(corrected_lines, time_candidates, debug_log=debug_log)

    # Line-boundary reconciliation (forward-cursor, disambiguation-safe) computed once and shared
    # between word/line assignment and orphan-run recovery below.
    reconciliation = reconcile_mxl_to_lrc_lines(mxl_words, lrc_match.lrc_lines)
    print(f"[mxl-lrc] MXL/LRC line reconciliation: {len(reconciliation.line_mxl_range)}/"
          f"{reconciliation.n_lrc_lines} line(s) matched ({reconciliation.match_ratio:.0%}), "
          f"{len(reconciliation.dropped_lrc_lines)} dropped, longest dropped run "
          f"{reconciliation.longest_dropped_run} line(s), {len(reconciliation.orphan_mxl_runs)} "
          f"orphan MXL word run(s)")
    orphan_starts = recover_orphan_mxl_runs(mxl_words, lrc_match.lrc_lines, reconciliation, asr_words)
    if orphan_starts:
        print(f"[mxl-lrc] orphan-run recovery: {len(orphan_starts)} word(s) placed directly against "
              f"ASR in their own neighboring-line time window")

    word_lines, word_clean_text, word_group, word_group_text, word_syllable_override, word_lrc_candidate = \
        assign_words_to_lines(mxl_words, lrc_match.lrc_lines, reconciliation=reconciliation)
    word_starts, word_ends, quality = place_words_via_asr(mxl_words, word_lines, lrc_match.lrc_lines, asr_words,
                                                            word_clean_text=word_clean_text,
                                                            word_group=word_group,
                                                            word_group_text=word_group_text,
                                                            word_lrc_candidate=word_lrc_candidate,
                                                            preplaced=orphan_starts)
    quality.n_lrc_lines = reconciliation.n_lrc_lines
    quality.n_lrc_lines_dropped = len(reconciliation.dropped_lrc_lines)
    quality.mxl_lrc_match_ratio = reconciliation.match_ratio
    quality.longest_dropped_lrc_run = reconciliation.longest_dropped_run
    syllables = build_syllables(mxl_words, word_starts, word_ends, word_lines, word_clean_text,
                                 word_syllable_override=word_syllable_override)

    nonmonotonic_rate = quality.non_monotonic_fix_count / quality.n_words if quality.n_words else 1.0
    if quality.asr_placement_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:
        return MxlLrcResult(
            success=False,
            reason=(f"ASR/MXL word match rate too low ({quality.asr_placement_rate:.0%}, need "
                     f"{config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:.0%}) -- the matched lyrics likely don't "
                     f"correspond to this recording"),
            syllables=syllables, quality=quality, lrc_match=lrc_match,
            mxl_path=mxl_path, part_names_used=part_names, time_calibration=time_cal,
        )
    if nonmonotonic_rate > config.MXL_LRC_MAX_NONMONOTONIC_RATE:
        return MxlLrcResult(
            success=False,
            reason=f"too many out-of-order word placements ({nonmonotonic_rate:.0%})",
            syllables=syllables, quality=quality, lrc_match=lrc_match,
            mxl_path=mxl_path, part_names_used=part_names, time_calibration=time_cal,
        )

    # Only calibrated once both quality gates pass -- not worth the pitch-inference cost
    # on a result about to be discarded.
    syllables, pitch_cal_offset, pitch_cal_confidence = calibrate_mxl_syllable_pitch(
        syllables, vocals_path, debug_log=debug_log,
    )

    return MxlLrcResult(success=True, reason="", syllables=syllables, quality=quality,
                         lrc_match=lrc_match, mxl_path=mxl_path, part_names_used=part_names,
                         time_calibration=time_cal, pitch_calibration_offset=pitch_cal_offset,
                         pitch_calibration_confidence=pitch_cal_confidence)


def try_mxl_lrc_primary(mxl_paths: List[str], artist: str, title: str, audio_duration: float,
                         asr_words: List[Word], forced_candidate: Optional[LrcLibCandidate] = None,
                         preferred_part_name: Optional[str] = None,
                         vocals_path: Optional[str] = None, debug_log=None) -> Optional[MxlLrcResult]:
    """Tries each MXL path in order, returning the first that clears the quality gate. If
    none succeed, returns the last (failed) result so the caller has a concrete reason.
    Returns None only if `mxl_paths` is empty."""
    last_result = None
    for mxl_path in mxl_paths:
        result = generate_from_mxl_and_lrc(
            mxl_path, artist, title, audio_duration, asr_words,
            forced_candidate=forced_candidate, preferred_part_name=preferred_part_name,
            vocals_path=vocals_path, debug_log=debug_log,
        )
        if result.success:
            return result
        last_result = result
    return last_result
