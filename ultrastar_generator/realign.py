"""Alignment-only mode: given a FINISHED UltraStar .txt and its audio (or a
video that stands in for it, same as the main pipeline), re-times the
file's own notes to better match the audio -- GAP, note start time, and
note length -- WITHOUT touching pitch and without adding, removing, or
reordering a single note. Assumes the input file's notes are already in
the right order and its lyric TEXT is correct; makes no other assumption
about the file's own timing (it could be hand-authored, machine-generated
by a different tool, or -- the degenerate case this is explicitly designed
to survive -- a flat list of equal-length placeholder notes that don't
correspond to the audio at all).

The existing file being realigned is treated as READ-ONLY, unconditionally
and permanently -- `run_realign_pipeline` never writes to it, defaults to
a separate "<name> [REALIGNED].txt" output, and hard-refuses to run at all
if an explicit output path is ever given that resolves to the SAME path as
the existing file (see its own `output_path` check). No override exists
for this on purpose -- don't add one.

Design mirrors `mxl_lrc_generator.py`'s proven shape (real transcription of
OUR OWN audio as the primary signal, LRCLIB synced-lyrics line starts as a
secondary anchor when available, nearest-confident-anchor interpolation
for everything else) but adapted for a fundamentally different situation:
mxl_lrc_generator trusts the MXL's own relative offsets/durations as a
reliable rhythm template and only needs LRC to anchor per-LINE windows,
because MXL data is professionally authored. Here the INPUT FILE's own
timing is exactly what's being corrected, so it can't be trusted as a
rhythm template OR used to window the ASR search the way MXL+LRC windows
each line by +-0.5s -- a badly-off input file would just miss the correct
ASR words entirely under that scheme. Instead:

  - ASR matching is a single WHOLE-SONG, order-preserving text alignment
    (existing words vs. ASR words), not time-windowed at all -- text order
    is the only thing this mode can trust the input file for.
  - The existing word's own ORIGINAL start time is used purely as a
    proportional "offset" for interpolating between confident anchors
    (exactly the role MXL's quarter-note offset plays in
    mxl_lrc_generator.place_words_via_asr) -- if the original timing
    already roughly tracks the audio, this recovers real local tempo
    variation; if the original timing is degenerate (e.g. uniform
    single-beat notes), it degrades gracefully to even spacing between
    confident anchors, no worse.
  - Unlike mxl_lrc_generator (which has no "original" to fall back to and
    so must always guess something), this mode always has the original
    timing as a safe default -- a word with NO confident anchor anywhere
    in the whole song keeps its original timing untouched rather than
    extrapolating from nothing.
"""

from __future__ import annotations

import copy
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from . import config
from .models import LineBreak, Song, Syllable, Word
from .usdx_parser import ParsedSong, UsdxParseError, parse_usdx_file
from .usdx_writer import write_song
from .lyrics_lookup import LrcLibCandidate, fetch_lrclib_by_id
from .lrc_timing import match_asr_to_lrc_lines, two_tier_time_calibration
from .mxl_lrc_generator import MxlWord, select_lrc_candidate, assign_words_to_lines


def _normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", s)


@dataclass
class ExistingWord:
    """One word from the EXISTING file, reconstructed from its own
    word-start-tagged syllable run (see `usdx_parser.parse_usdx_file` --
    `is_word_start` now correctly handles both the leading-space
    convention this project's own writer uses AND the trailing-space
    convention real hand/SingStar-authored files use)."""
    entry_indices: List[int]   # indices into ParsedSong.entries for this word's syllables
    text: str
    norm: str
    orig_start: float
    orig_end: float


def extract_words(entries: List[Union[Syllable, LineBreak]]) -> List[ExistingWord]:
    """Groups the parsed entries into words, in order. LineBreaks are not
    part of any word -- they're repositioned separately once every word's
    new timing is known (see `_reposition_line_breaks`)."""
    words: List[ExistingWord] = []
    cur_indices: List[int] = []
    cur_text = ""

    def flush():
        if cur_indices:
            first = entries[cur_indices[0]]
            last = entries[cur_indices[-1]]
            words.append(ExistingWord(
                entry_indices=list(cur_indices), text=cur_text, norm=_normalize(cur_text),
                orig_start=first.start, orig_end=last.end,
            ))

    for i, e in enumerate(entries):
        if isinstance(e, LineBreak):
            continue
        if e.is_word_start and cur_indices:
            flush()
            cur_indices = []
            cur_text = ""
        cur_indices.append(i)
        cur_text += e.text
    flush()

    return words


@dataclass
class RealignQuality:
    n_words: int = 0
    n_asr_matched: int = 0
    n_lrc_seeded: int = 0
    n_interpolated: int = 0
    n_kept_original: int = 0

    @property
    def anchor_rate(self) -> float:
        return (self.n_asr_matched + self.n_lrc_seeded) / self.n_words if self.n_words else 0.0


def match_words_to_asr(existing_words: List[ExistingWord], asr_words: List[Word]
                        ) -> Tuple[List[Optional[float]], List[Optional[float]], List[bool]]:
    """Whole-song, order-preserving text match of the existing file's own
    words against real ASR words -- deliberately NOT time-windowed (see
    module docstring for why: this mode can't trust the input file's own
    timing enough to window a search with it). Same matching technique as
    `mxl_lrc_generator.place_words_via_asr` (exact match, plus a
    fuzzy-ratio "replace" fallback for ASR's own mishearing of a word),
    just applied once across the whole word sequence instead of per-line.

    Returns (starts, ends, confident), parallel to `existing_words` --
    unmatched/low-confidence words are None/False."""
    n = len(existing_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    confident: List[bool] = [False] * n

    a = [w.norm for w in existing_words]
    b = [_normalize(w.text) for w in asr_words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(a2 - a1):
                asr_w = asr_words[b1 + k]
                if asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                    starts[a1 + k] = asr_w.start
                    ends[a1 + k] = asr_w.end
                    confident[a1 + k] = True
        elif tag == "replace" and (a2 - a1) == 1:
            # A single unmatched existing word against one or more ASR
            # words in this block (the ASR side isn't always exactly one
            # word -- a neighboring word can ride along) -- try every
            # candidate, keep the best fuzzy match, same technique
            # mxl_lrc_generator.place_words_via_asr already validated for
            # exactly this failure mode (ASR mishearing a word
            # differently than the reference text).
            best_ratio = 0.0
            best_asr_w = None
            for bk in range(b1, b2):
                ratio = difflib.SequenceMatcher(None, a[a1], b[bk]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_asr_w = asr_words[bk]
            if best_ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO and best_asr_w is not None:
                if best_asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                    starts[a1] = best_asr_w.start
                    ends[a1] = best_asr_w.end
                    confident[a1] = True

    return starts, ends, confident


@dataclass
class LrcSeedResult:
    lrc_match: object
    calibration_offset: Optional[float]
    calibration_kind: Optional[str]
    calibration_confidence: float
    n_seeded: int = 0


@dataclass
class LrcPrep:
    """Result of candidate selection + time calibration + per-word line
    assignment -- the part of LRC integration that's IDENTICAL regardless
    of how the lines get used afterward (seeding a single anchor per line
    vs. windowing the ASR search for every word in the line). Factored
    out so both strategies share one candidate-selection/calibration
    path, not two."""
    lrc_match: object
    lrc_lines: List[Tuple[float, str]]   # calibrated (or raw, if calibration failed)
    word_lines: List[Optional[int]]      # per existing_word, its assigned LRC line index
    calibration_offset: Optional[float]
    calibration_kind: Optional[str]
    calibration_confidence: float


def prepare_lrc(existing_words: List[ExistingWord], asr_words: List[Word],
                 artist: str, title: str, audio_duration: float,
                 forced_candidate: Optional[LrcLibCandidate] = None) -> Optional[LrcPrep]:
    """Selects an LRC candidate, calibrates its line timestamps against
    OUR audio's real ASR transcription, and assigns each existing word to
    a line -- shared setup for both `seed_lrc_anchors` (LRC seeds ONE
    anchor per line, ASR is primary everywhere else) and
    `match_words_to_asr_windowed` (LRC lines WINDOW the ASR search for
    EVERY word, ASR only resolves position within a line -- mirrors
    mxl_lrc_generator.place_words_via_asr's design). Returns None if no
    usable candidate exists at all (real confirmed case: BATB has none --
    see CLAUDE.md) -- both callers then fall through to whole-song,
    LRC-free matching.

    `select_lrc_candidate`/`assign_words_to_lines` are reused as-is from
    mxl_lrc_generator.py -- they only ever read `MxlWord.norm`, so
    existing words are wrapped in real `MxlWord` instances with harmless
    placeholder `offset`/`syllables` rather than duplicating that
    candidate-selection/line-assignment logic a third time."""
    fake_words = [MxlWord(text=w.text, norm=w.norm, offset=float(i), syllables=[])
                  for i, w in enumerate(existing_words)]

    lrc_match = select_lrc_candidate(artist, title, fake_words, audio_duration, forced=forced_candidate)
    if lrc_match is None:
        return None

    # Calibrate away a systematic offset between LRC's own line timestamps
    # and OUR audio's real timing (e.g. different lead-in silence) BEFORE
    # trusting a line start as an anchor -- same technique/function
    # mxl_lrc_generator uses, factored into lrc_timing.py for exactly this
    # kind of reuse.
    time_candidates = match_asr_to_lrc_lines(asr_words, lrc_match.lrc_lines)
    offset, slope, confidence, kind, _skipped = two_tier_time_calibration(time_candidates)
    lrc_lines = lrc_match.lrc_lines
    if offset is not None:
        lrc_lines = [(t + offset + slope * t, text) for t, text in lrc_lines]

    word_lines, _clean_text = assign_words_to_lines(fake_words, lrc_lines)

    return LrcPrep(lrc_match=lrc_match, lrc_lines=lrc_lines, word_lines=word_lines,
                    calibration_offset=offset, calibration_kind=kind, calibration_confidence=confidence)


def seed_lrc_anchors(existing_words: List[ExistingWord], asr_words: List[Word],
                      starts: List[Optional[float]], ends: List[Optional[float]], confident: List[bool],
                      artist: str, title: str, audio_duration: float,
                      forced_candidate: Optional[LrcLibCandidate] = None) -> Optional[LrcSeedResult]:
    """"seed" strategy: fills in a real-time anchor, from LRCLIB's
    synced-lyrics LINE starts, for the first not-yet-confident word of
    each line the candidate's lyrics can be matched to -- mutates
    `starts`/`ends`/`confident` in place. ASR (whole-song, see
    `match_words_to_asr`) is the PRIMARY signal in this strategy; LRC
    only fills in the residual gaps ASR couldn't reach. Returns None (no
    mutation) if no usable LRC candidate exists for this song at all.

    See `match_words_to_asr_windowed` for the alternative "windowed"
    strategy (LRC lines are primary, ASR only resolves position within a
    line) -- kept as a separate code path pending real end-to-end
    comparison of the two (see CLAUDE.md)."""
    prep = prepare_lrc(existing_words, asr_words, artist, title, audio_duration, forced_candidate)
    if prep is None:
        return None
    return seed_from_prep(existing_words, prep, starts, ends, confident)


def seed_from_prep(existing_words: List[ExistingWord], prep: "LrcPrep",
                    starts: List[Optional[float]], ends: List[Optional[float]],
                    confident: List[bool]) -> LrcSeedResult:
    """The actual seeding loop from `seed_lrc_anchors`, taking an
    already-prepared `LrcPrep` -- split out so `realign_song` can reuse a
    single `prepare_lrc` call (one candidate search, one calibration)
    across BOTH deciding whether "windowed" mode is usable AND falling
    back to "seed" mode, instead of preparing LRC data twice."""
    result = LrcSeedResult(lrc_match=prep.lrc_match, calibration_offset=prep.calibration_offset,
                            calibration_kind=prep.calibration_kind,
                            calibration_confidence=prep.calibration_confidence)
    prev_line = None
    for i, li in enumerate(prep.word_lines):
        is_first_of_line = li is not None and li != prev_line
        prev_line = li
        if not is_first_of_line or confident[i]:
            continue
        line_start = prep.lrc_lines[li][0]
        orig_dur = max(0.0, existing_words[i].orig_end - existing_words[i].orig_start)
        starts[i] = line_start
        ends[i] = line_start + orig_dur
        confident[i] = True
        result.n_seeded += 1

    return result


def _line_window(lrc_lines: List[Tuple[float, str]], li: int) -> Tuple[float, float]:
    t0 = lrc_lines[li][0]
    t1 = lrc_lines[li + 1][0] if li + 1 < len(lrc_lines) else t0 + 5.0
    return t0, t1


def match_words_to_asr_windowed(existing_words: List[ExistingWord], word_lines: List[Optional[int]],
                                 lrc_lines: List[Tuple[float, str]], asr_words: List[Word]
                                 ) -> Tuple[List[Optional[float]], List[Optional[float]], List[bool]]:
    """"windowed" strategy (PROTOTYPE, see CLAUDE.md for the real-audio
    comparison against "seed"): LRC line starts are trusted PRIMARY
    anchors -- for each line, only ASR words whose own timestamp falls
    near that line's real-time window are candidates for ITS words,
    mirroring mxl_lrc_generator.place_words_via_asr's Pass 1 exactly
    (same matching technique: exact match, plus a fuzzy-ratio "replace"
    fallback for ASR's own mishearing), just matching against the
    EXISTING file's own already-trusted text instead of MXL's OCR'd text
    (so there's no "clean text" substitution step needed here -- the
    existing words ARE the clean text).

    Unlike `match_words_to_asr` (whole-song, no time information used at
    all), this can mis-place a word if the LRC candidate's calibration is
    wrong/unreliable -- the whole point of comparing this against "seed"
    is to measure whether windowing's improved local disambiguation is
    worth that added dependency on LRC quality. Returns (starts, ends,
    confident) parallel to `existing_words`, same shape as
    `match_words_to_asr` -- unwindowed/unmatched words are left
    None/False for `interpolate_fallback` to handle exactly as before."""
    n = len(existing_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    confident: List[bool] = [False] * n

    line_word_idxs: dict = {}
    for i, li in enumerate(word_lines):
        if li is None:
            continue
        line_word_idxs.setdefault(li, []).append(i)

    for li, idxs in line_word_idxs.items():
        idxs = sorted(idxs)
        t0, t1 = _line_window(lrc_lines, li)
        asr_in_window = [w for w in asr_words if t0 - 0.5 <= w.start <= t1 + 0.5]
        asr_norm = [_normalize(w.text) for w in asr_in_window]
        existing_norm_line = [existing_words[i].norm for i in idxs]

        sm = difflib.SequenceMatcher(None, existing_norm_line, asr_norm, autojunk=False)
        matched_local: dict = {}
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(a2 - a1):
                    asr_w = asr_in_window[b1 + k]
                    if asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                        matched_local[a1 + k] = asr_w
            elif tag == "replace" and (a2 - a1) == 1:
                best_ratio, best_asr_w = 0.0, None
                for bk in range(b1, b2):
                    ratio = difflib.SequenceMatcher(None, existing_norm_line[a1], asr_norm[bk]).ratio()
                    if ratio > best_ratio:
                        best_ratio, best_asr_w = ratio, asr_in_window[bk]
                if best_ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO and best_asr_w is not None:
                    if best_asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                        matched_local[a1] = best_asr_w

        for local_i, global_i in enumerate(idxs):
            if local_i not in matched_local:
                continue
            asr_w = matched_local[local_i]
            starts[global_i] = asr_w.start
            ends[global_i] = asr_w.end
            confident[global_i] = True

    return starts, ends, confident


def interpolate_fallback(existing_words: List[ExistingWord],
                          starts: List[Optional[float]], ends: List[Optional[float]],
                          confident: List[bool]) -> Tuple[int, int]:
    """PASS 2: every word still without a confident anchor is placed
    relative to its nearest confident neighbor(s) -- mutates `starts`/
    `ends` in place. Returns (n_interpolated, n_kept_original).

    Two confident anchors on both sides: the existing word's own ORIGINAL
    start (never trusted as absolute truth elsewhere in this module) is
    used purely as a proportional offset to interpolate BETWEEN the two
    real anchors -- recovers real local tempo variation the original file
    got wrong, while degrading gracefully (even spacing) if the original
    offsets carry no real information at all (e.g. uniform single-beat
    placeholder notes).

    Only one side has an anchor: interpolation has nothing to rate against,
    so this instead applies that anchor's own constant (start - orig_start)
    SHIFT to this word -- safer than extrapolating a rate from a single
    data point out into open territory.

    No anchor anywhere in the whole song: nothing here can be verified
    against the audio at all, so the word's ORIGINAL timing is kept
    completely unchanged rather than guessed -- this mode always has a
    safe fallback available, unlike mxl_lrc_generator's equivalent pass,
    which has no "original" to fall back to."""
    n = len(existing_words)
    confident_idxs = [i for i in range(n) if confident[i]]
    n_interpolated = 0
    n_kept_original = 0

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
        w = existing_words[i]
        orig_dur = max(0.0, w.orig_end - w.orig_start)

        pb = nearest_before(i)
        pa = nearest_after(i)

        if pb is None and pa is None:
            starts[i] = w.orig_start
            ends[i] = w.orig_end
            n_kept_original += 1
            continue

        if pb is not None and pa is not None:
            wb, wa = existing_words[pb], existing_words[pa]
            off_delta = wa.orig_start - wb.orig_start
            rate = (starts[pa] - starts[pb]) / off_delta if off_delta > 1e-6 else None
            if rate is not None and rate > 0:
                est_start = starts[pb] + (w.orig_start - wb.orig_start) * rate
                est_end = est_start + orig_dur * rate
            else:
                # Degenerate (identical/out-of-order original offsets) --
                # fall back to a constant shift from the nearer anchor
                # rather than trust a zero/negative rate.
                shift = starts[pb] - wb.orig_start
                est_start = w.orig_start + shift
                est_end = est_start + orig_dur
        elif pb is not None:
            wb = existing_words[pb]
            shift = starts[pb] - wb.orig_start
            est_start = w.orig_start + shift
            est_end = est_start + orig_dur
        else:
            wa = existing_words[pa]
            shift = starts[pa] - wa.orig_start
            est_start = w.orig_start + shift
            est_end = est_start + orig_dur

        starts[i] = est_start
        ends[i] = est_end
        n_interpolated += 1

    # Non-decreasing starts, and an end never overlapping the next word's
    # own (already-finalized) start -- same clamp shape as
    # mxl_lrc_generator.place_words_via_asr's final safety net.
    for i in range(1, n):
        if starts[i] < starts[i - 1]:
            starts[i] = starts[i - 1]
    for i in range(n):
        if i + 1 < n and ends[i] > starts[i + 1]:
            ends[i] = starts[i + 1]
        if ends[i] < starts[i]:
            ends[i] = starts[i]

    return n_interpolated, n_kept_original


def _redistribute_syllables(entries: List[Union[Syllable, LineBreak]], word: ExistingWord,
                             new_start: float, new_end: float) -> List[Tuple[int, float, float]]:
    """Splits a word's new [new_start, new_end) span across its own
    syllables using their ORIGINAL relative sub-word timing (pitch is
    never touched anywhere in this module -- only start/end). Falls back
    to an even split if the word's original span was zero-width (e.g.
    every syllable crammed onto one beat) rather than collapsing every
    syllable onto the same point."""
    idxs = word.entry_indices
    lo, hi = word.orig_start, word.orig_end
    out: List[Tuple[int, float, float]] = []
    if hi > lo:
        for idx in idxs:
            syl = entries[idx]
            frac0 = (syl.start - lo) / (hi - lo)
            frac1 = (syl.end - lo) / (hi - lo)
            out.append((idx, new_start + frac0 * (new_end - new_start),
                        new_start + frac1 * (new_end - new_start)))
    else:
        n = len(idxs)
        step = (new_end - new_start) / n if n else 0.0
        for k, idx in enumerate(idxs):
            out.append((idx, new_start + k * step, new_start + (k + 1) * step))
    return out


def _reposition_line_breaks(entries: List[Union[Syllable, LineBreak]],
                             new_start_by_idx: dict, new_end_by_idx: dict) -> None:
    """A LineBreak's own start/end aren't independently meaningful -- they
    just mark the gap between the syllable before it and the syllable
    after (see phrasing.build_lines, which sets them the same way). Once
    every syllable has a new time, line breaks are simply re-anchored to
    their new neighbors -- never guessed independently."""
    n = len(entries)
    for i, e in enumerate(entries):
        if not isinstance(e, LineBreak):
            continue
        prev_end = None
        for j in range(i - 1, -1, -1):
            if j in new_end_by_idx:
                prev_end = new_end_by_idx[j]
                break
        next_start = None
        for j in range(i + 1, n):
            if j in new_start_by_idx:
                next_start = new_start_by_idx[j]
                break
        e.start = prev_end if prev_end is not None else (next_start if next_start is not None else e.start)
        if e.end is not None:
            e.end = next_start if next_start is not None else e.start


@dataclass
class RealignResult:
    success: bool
    error: Optional[str] = None
    song: Optional[Song] = None
    quality: Optional[RealignQuality] = None
    lrc_seed: Optional[LrcSeedResult] = None


def realign_song(existing: ParsedSong, asr_words: List[Word], *,
                  artist: Optional[str] = None, title: Optional[str] = None,
                  audio_duration: Optional[float] = None,
                  use_lrc: bool = True, lrc_mode: str = "windowed",
                  forced_lrc_candidate: Optional[LrcLibCandidate] = None,
                  log: Callable[[str], None] = print) -> RealignResult:
    """Core, audio-loading-free realignment step: given an already-parsed
    existing song and already-transcribed ASR words, returns a new `Song`
    with the SAME notes (never added/removed/reordered, pitch untouched)
    but corrected start/end times. See module docstring for the overall
    approach.

    `lrc_mode` picks between two LRC integration strategies (no-op if
    `use_lrc` is False or no usable candidate exists -- both fall through
    to whole-song ASR matching identically either way):
      - "windowed" (DEFAULT as of 2026-08-09): LRC line starts are
        PRIMARY anchors; ASR only resolves word POSITION within each
        line's own real-time window, mirroring mxl_lrc_generator's
        design -- ONLY actually used when the LRC candidate's time
        calibration confidently succeeds (`LrcPrep.calibration_offset is
        not None`); an unconfident/failed calibration transparently
        falls back to the exact same whole-song-ASR-plus-seed behavior
        "seed" (below) always uses -- i.e. this mode is ALREADY an
        auto-select ("use windowing when the LRC timing can be trusted,
        otherwise fall back"), not a strict alternative to "seed". Real
        end-to-end comparison (BATB/Stars/Chicago, see CLAUDE.md) found
        it never worse than "seed" once this gate was added, and a clear
        win when calibration was confident even at just 54% agreement
        (BATB: 63%->84% of words landing within 100ms of ground truth) --
        made the shipped default on that evidence.
      - "seed": ASR (whole-song, no time information) is PRIMARY always,
        even when a confidently-calibrated LRC candidate exists; LRC only
        seeds a single anchor for the first not-yet-confident word of
        each matched line. Kept as an explicit opt-out / for A-B testing
        against "windowed" -- not needed for normal use now that
        "windowed" already degrades to equivalent behavior on its own
        whenever LRC can't be trusted."""
    # A deep copy, not just a new list -- the Syllable/LineBreak objects
    # themselves get mutated in place below (start/end reassigned), and
    # `existing` is a caller-owned ParsedSong that must come back out
    # untouched (e.g. so realign_song can be called twice on the SAME
    # parsed object to compare lrc_mode strategies against each other,
    # which a shallow `list(existing.entries)` would silently corrupt --
    # the second call would then be realigning the FIRST call's output,
    # not the original file).
    entries = copy.deepcopy(existing.entries)
    words = extract_words(entries)
    if not words:
        return RealignResult(success=False, error="Existing file has no words to realign.")

    lrc_prep = None
    if use_lrc and audio_duration is not None:
        lrc_prep = prepare_lrc(words, asr_words, artist or existing.artist, title or existing.title,
                                audio_duration, forced_candidate=forced_lrc_candidate)

    # "windowed" additionally requires a CONFIDENT time calibration before it's
    # trusted -- real comparison (see CLAUDE.md, BATB/Stars/Chicago) found
    # windowing raw, UNCALIBRATED LRC lines is actively harmful: every word's
    # match is gated through the same untrusted signal, so an uncorrected
    # drift corrupts the whole song (confirmed real case: Chicago's
    # auto-picked candidate had no confident calibration and windowed mode's
    # within-100ms accuracy dropped from 65% to 41% as a result). "seed" mode
    # doesn't need this same gate -- LRC there only ever seeds a handful of
    # residual words ASR couldn't reach, so a bad candidate's blast radius is
    # already small by construction; windowed's is not, since it decides
    # EVERY word's search window.
    lrc_seed = None
    use_windowed = (lrc_mode == "windowed" and lrc_prep is not None
                     and lrc_prep.calibration_offset is not None)
    if lrc_mode == "windowed" and lrc_prep is not None and not use_windowed:
        log(f"  LRC candidate found but no confident time calibration -- 'windowed' mode isn't safe to use "
            f"here (would window the ENTIRE match against an uncalibrated signal), falling back to "
            f"whole-song ASR matching instead.")
    if use_windowed:
        starts, ends, confident = match_words_to_asr_windowed(words, lrc_prep.word_lines, lrc_prep.lrc_lines,
                                                                asr_words)
        quality = RealignQuality(n_words=len(words), n_asr_matched=sum(confident))
        c = lrc_prep.lrc_match.candidate
        log(f"  LRC-windowed matching: {c.track_name!r}/{c.artist_name!r} (lrclib id={c.id}), "
            f"time calibration ({lrc_prep.calibration_kind}) offset {lrc_prep.calibration_offset:+.1f}s "
            f"({lrc_prep.calibration_confidence:.0%} agreement)")
    else:
        starts, ends, confident = match_words_to_asr(words, asr_words)
        quality = RealignQuality(n_words=len(words), n_asr_matched=sum(confident))
        if use_lrc and audio_duration is not None:
            if lrc_prep is not None:
                lrc_seed = seed_from_prep(words, lrc_prep, starts, ends, confident)
                if lrc_seed is not None:
                    quality.n_lrc_seeded = lrc_seed.n_seeded
                    log(f"  LRC anchors: {lrc_seed.lrc_match.candidate.track_name!r}/"
                        f"{lrc_seed.lrc_match.candidate.artist_name!r} (lrclib id={lrc_seed.lrc_match.candidate.id}) -- "
                        f"seeded {lrc_seed.n_seeded} additional line-start anchor(s)"
                        + (f", time calibration ({lrc_seed.calibration_kind}) offset "
                           f"{lrc_seed.calibration_offset:+.1f}s ({lrc_seed.calibration_confidence:.0%} agreement)"
                           if lrc_seed.calibration_offset is not None else ", no time calibration found"))
            else:
                log("  No usable LRCLIB synced-lyrics candidate for this song -- ASR matching only.")

    quality.n_interpolated, quality.n_kept_original = interpolate_fallback(words, starts, ends, confident)

    log(f"  {quality.n_asr_matched}/{quality.n_words} word(s) matched directly to real ASR transcription, "
        f"{quality.n_lrc_seeded} from LRC line anchors, {quality.n_interpolated} interpolated between "
        f"anchors, {quality.n_kept_original} kept at their original timing (no anchor found nearby).")
    if quality.anchor_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:
        log(f"  WARNING: only {quality.anchor_rate:.0%} of words got a real anchor from the audio -- "
            f"this file's lyrics may not match this audio at all. Review the output carefully.")

    new_start_by_idx: dict = {}
    new_end_by_idx: dict = {}
    for i, w in enumerate(words):
        for idx, s, e in _redistribute_syllables(entries, w, starts[i], ends[i]):
            new_start_by_idx[idx] = s
            new_end_by_idx[idx] = e

    for idx, e in enumerate(entries):
        if isinstance(e, Syllable):
            e.start = new_start_by_idx[idx]
            e.end = new_end_by_idx[idx]

    _reposition_line_breaks(entries, new_start_by_idx, new_end_by_idx)

    first_syllable = next((e for e in entries if isinstance(e, Syllable)), None)
    gap_ms = int(round(first_syllable.start * 1000)) if first_syllable else existing.gap_ms

    song = _song_from_existing(existing, entries, gap_ms)
    return RealignResult(success=True, song=song, quality=quality, lrc_seed=lrc_seed)


def _song_from_existing(existing: ParsedSong, entries: List[object], gap_ms: int) -> Song:
    """Builds the output Song, carrying every OTHER header tag from the
    existing file through completely untouched -- this mode only ever
    changes GAP and note start/length."""
    tags = existing.raw_tags

    def _float(name):
        v = tags.get(name)
        return float(v.replace(",", ".")) if v else None

    def _int(name):
        v = tags.get(name)
        return int(float(v.replace(",", "."))) if v else None

    return Song(
        title=existing.title, artist=existing.artist,
        language=tags.get("LANGUAGE") or config.DEFAULT_LANGUAGE,
        mp3=tags.get("MP3") or "",
        cover=tags.get("COVER"), background=tags.get("BACKGROUND"), video=tags.get("VIDEO"),
        videogap=_float("VIDEOGAP"),
        bpm=existing.bpm, gap_ms=gap_ms,
        preview_start=_float("PREVIEWSTART"),
        genre=tags.get("GENRE"), year=_int("YEAR"),
        edition=tags.get("EDITION"), creator=tags.get("CREATOR"),
        entries=entries,
    )


class AmbiguousExistingTxtError(ValueError):
    """Raised by `find_existing_txt_in_folder` when the folder doesn't
    have exactly one obvious existing .txt to realign -- fails closed,
    same convention as `file_discovery.AmbiguousInputError` (never
    silently guesses which file the user meant)."""


def find_existing_txt_in_folder(folder: Path) -> Path:
    """Auto-detects the single existing UltraStar .txt to realign within
    a folder -- used when no explicit --existing-txt is given (required
    for batch mode, where a single explicit path can't apply across
    multiple subfolders; optional convenience in single-song mode too).

    Excludes this module's OWN "[REALIGNED]" output naming convention --
    otherwise re-running on a folder that already has a previous run's
    output would either see two candidates (falsely "ambiguous") or,
    worse, pick the REALIGNED file itself as this run's new INPUT,
    compounding drift across repeated runs instead of always realigning
    the same original file.

    When more than one real candidate remains, tries ONE further
    disambiguation before giving up: a file named exactly "<folder
    name>.txt" (case-insensitive) -- the common convention this project's
    own output/companion files already follow elsewhere (e.g. `#COVER`/
    `#BACKGROUND` matching by basename in file_discovery.py). Only trusted
    when it narrows the field to EXACTLY one match; still fails closed
    (never guesses) otherwise."""
    folder = Path(folder)
    candidates = sorted(p for p in folder.glob("*.txt") if "[REALIGNED]" not in p.stem)
    if not candidates:
        raise AmbiguousExistingTxtError(f"No .txt file found in {folder} to realign.")
    if len(candidates) == 1:
        return candidates[0]

    expected_name = f"{folder.name}.txt".lower()
    name_matches = [p for p in candidates if p.name.lower() == expected_name]
    if len(name_matches) == 1:
        return name_matches[0]

    names = ", ".join(p.name for p in candidates)
    raise AmbiguousExistingTxtError(
        f"Multiple .txt files found in {folder} ({names}), and none (or more than one) match the "
        f"folder's own name ({folder.name}.txt) -- pass --existing-txt explicitly to disambiguate.")


def resolve_realign_output_path(existing_txt_path: Path, output_path_override: Optional[str]) -> Path:
    """Where a realigned .txt gets written -- an explicit override if given,
    else a NEW file alongside the existing one, never the existing file
    itself. Pure path logic, factored out from `run_realign_pipeline` so it
    (and `check_output_not_existing_file` below) can be unit-tested without
    needing real audio/ASR."""
    existing_txt_path = Path(existing_txt_path)
    return (Path(output_path_override).resolve() if output_path_override
            else existing_txt_path.with_name(existing_txt_path.stem + " [REALIGNED].txt"))


def check_output_not_existing_file(output_path: Path, existing_txt_path: Path) -> Optional[str]:
    """HARD guarantee, not just a default: the existing file being realigned
    is treated as read-only, unconditionally -- never overwritten, even if
    an explicit output path is given that resolves to the same path (e.g. a
    typo, or a future caller assuming "in place" is safe). No override
    exists for this on purpose -- don't add one. Returns an error message if
    `output_path` would overwrite `existing_txt_path`, else None."""
    if Path(output_path).resolve() == Path(existing_txt_path).resolve():
        return (f"Refusing to overwrite the existing file being realigned ({existing_txt_path}) -- "
                f"the input file is always treated as read-only. Pass a different --output path.")
    return None


# --- CLI ---------------------------------------------------------------

def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Alignment-only mode: re-time an EXISTING UltraStar .txt's notes against its "
                    "audio (GAP, note start/length only -- pitch and the note sequence are never "
                    "touched, changed, added, or removed)."
    )
    p.add_argument("input", help="Path to the song's folder (containing the audio, or a video that "
                                  "stands in for it -- same folder-resolution rules as the main pipeline). "
                                  "With --batch, this is a PARENT folder whose immediate subdirectories are "
                                  "each realigned the same way.")
    p.add_argument("--batch", action="store_true",
                    help="Treat the positional argument as a PARENT folder: run realignment on each of its "
                         "immediate subdirectories (not the parent itself), auto-detecting each subfolder's "
                         "own existing .txt (see --existing-txt). One song failing does not abort the rest. "
                         "Not allowed together with --existing-txt/--audio-file/--work-dir/--artist/--title/"
                         "--lrclib-id (none of which make sense as a single override across multiple songs).")
    p.add_argument("--existing-txt", dest="existing_txt_path", default=None,
                    help="Path to the existing UltraStar .txt to realign. Optional in single-song mode -- "
                         "auto-detects the single .txt file in the input folder if omitted. Not allowed "
                         "with --batch (each subfolder is always auto-detected).")
    p.add_argument("--output", dest="output_path", default=None,
                    help="Where to write the realigned .txt (default: alongside the existing file, "
                         "named '<name> [REALIGNED].txt'). The existing file (--existing-txt) is always "
                         "treated as read-only -- this refuses to run if --output resolves to that same "
                         "path, no override exists.")
    p.add_argument("--audio-file", default=None, help="Same as the main pipeline's --audio-file.")
    p.add_argument("--work-dir", default=None, help="Same as the main pipeline's --work-dir.")
    p.add_argument("--whisper-model", default=config.DEFAULT_WHISPER_MODEL)
    p.add_argument("--demucs-model", default=config.DEFAULT_DEMUCS_MODEL)
    p.add_argument("--skip-separation", action="store_true")
    p.add_argument("--vocals-path", default=None)
    p.add_argument("--no-whisperx", action="store_true")
    p.add_argument("--whisperx-no-vad", dest="whisperx_no_vad", action="store_true",
                    default=config.ENABLE_WHISPERX_NO_VAD)
    p.add_argument("--whisperx-vad", dest="whisperx_no_vad", action="store_false")
    p.add_argument("--artist", default=None, help="Override the artist used for LRCLIB lookup "
                                                    "(default: the existing file's own #ARTIST tag).")
    p.add_argument("--title", default=None, help="Override the title used for LRCLIB lookup "
                                                   "(default: the existing file's own #TITLE tag).")
    p.add_argument("--lrclib-id", dest="lrclib_id", type=int, default=None)
    p.add_argument("--no-lrc", dest="use_lrc", action="store_false", default=True,
                    help="Don't use LRCLIB synced lyrics even if available -- ASR matching only.")
    p.add_argument("--lrc-mode", choices=["seed", "windowed"], default="windowed",
                    help="How LRC lines get used when available (default: windowed). 'windowed': LRC line "
                         "starts window the ASR search, mirroring mxl_lrc_generator.py's design -- but ONLY "
                         "when the candidate's time calibration is confident, otherwise transparently falls "
                         "back to whole-song ASR matching (same as 'seed'), so this is already an "
                         "auto-select, not a strict alternative. Real comparison (see CLAUDE.md) found it "
                         "never worse than 'seed' and a clear win when calibration was confident. 'seed': "
                         "always whole-song-ASR-primary, even with a confidently-calibrated candidate -- "
                         "kept for A-B comparison, not needed for normal use.")
    return p


@dataclass
class RealignPipelineOptions:
    """Every knob `run_realign_pipeline` needs, decoupled from argparse --
    same shape/purpose as `config.PipelineOptions` for the main pipeline,
    letting the GUI and the CLI build this the same way and call the exact
    same pipeline code (see gui.py)."""
    audio_file: Optional[str] = None
    work_dir: Optional[str] = None
    whisper_model: str = config.DEFAULT_WHISPER_MODEL
    demucs_model: str = config.DEFAULT_DEMUCS_MODEL
    skip_separation: bool = False
    vocals_path: Optional[str] = None
    no_whisperx: bool = False
    whisperx_no_vad: bool = config.ENABLE_WHISPERX_NO_VAD
    artist: Optional[str] = None
    title: Optional[str] = None
    lrclib_id: Optional[int] = None
    use_lrc: bool = True
    lrc_mode: str = "windowed"
    output_path: Optional[str] = None


@dataclass
class RealignPipelineResult:
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None


def run_realign_pipeline(input_dir: Path, existing_txt_path: Optional[Path], opts: RealignPipelineOptions,
                          *, log: Callable[[str], None] = print) -> RealignPipelineResult:
    """Runs the full realign CLI/GUI flow for one song: resolves the audio,
    isolates vocals, transcribes, realigns, and writes the output file.
    Never raises on an "expected" failure (bad existing file, ambiguous
    folder, no words transcribed, etc.) -- those come back as
    `RealignPipelineResult(success=False, error=...)`, same convention as
    `main.run_pipeline`, so the CLI wrapper and gui.py can both handle
    failure uniformly.

    `existing_txt_path=None` auto-detects the single .txt in `input_dir`
    (see `find_existing_txt_in_folder`) -- required for `run_realign_batch`
    (a single explicit path can't apply across multiple subfolders), and
    a convenience in single-song mode too."""
    from .song_input import resolve_song_folder
    from .file_discovery import AmbiguousInputError, NoAudioSourceFoundError
    from .separation import isolate_vocals, SeparationError
    from .transcription import transcribe_words

    input_dir = Path(input_dir)
    if existing_txt_path is None:
        try:
            existing_txt_path = find_existing_txt_in_folder(input_dir)
        except AmbiguousExistingTxtError as e:
            return RealignPipelineResult(success=False, error=str(e))
        log(f"Auto-detected existing file: {existing_txt_path}")
    else:
        existing_txt_path = Path(existing_txt_path)
    work_dir = Path(opts.work_dir).resolve() if opts.work_dir else (input_dir / ".ultrastar_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        existing = parse_usdx_file(existing_txt_path)
    except UsdxParseError as e:
        return RealignPipelineResult(success=False, error=f"Could not parse {existing_txt_path}: {e}")

    try:
        resolved = resolve_song_folder(input_dir, work_dir, audio_file_override=opts.audio_file)
    except (AmbiguousInputError, NoAudioSourceFoundError) as e:
        return RealignPipelineResult(success=False, error=str(e))

    if opts.skip_separation:
        if not opts.vocals_path:
            return RealignPipelineResult(success=False, error="skip_separation requires vocals_path")
        vocals_path = Path(opts.vocals_path).resolve()
    else:
        log("Isolating vocals with Demucs...")
        try:
            vocals_path = isolate_vocals(resolved.analysis_audio, work_dir, model=opts.demucs_model)
        except SeparationError as e:
            return RealignPipelineResult(success=False, error=f"Vocal isolation failed: {e}")

    import librosa
    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)
    audio_duration = len(y) / sr

    log(f"Transcribing with {'whisperx' if not opts.no_whisperx else 'faster-whisper'} "
        f"({opts.whisper_model})...")
    asr_words = transcribe_words(
        vocals_path, opts.whisper_model, prefer_whisperx=not opts.no_whisperx,
        whisperx_vad_options=config.WHISPERX_NO_VAD_OPTIONS if opts.whisperx_no_vad else None,
    )
    if not asr_words:
        return RealignPipelineResult(
            success=False, error="No words were transcribed -- check the audio / vocal isolation quality.")
    log(f"Transcribed {len(asr_words)} words.")

    forced_candidate = None
    if opts.lrclib_id is not None:
        forced_candidate = fetch_lrclib_by_id(opts.lrclib_id)
        if forced_candidate is None:
            log(f"Could not fetch LRCLIB id {opts.lrclib_id} -- ignoring.")

    log("Realigning...")
    result = realign_song(
        existing, asr_words, artist=opts.artist, title=opts.title, audio_duration=audio_duration,
        use_lrc=opts.use_lrc, lrc_mode=opts.lrc_mode, forced_lrc_candidate=forced_candidate, log=log,
    )
    if not result.success:
        return RealignPipelineResult(success=False, error=result.error)

    output_path = resolve_realign_output_path(existing_txt_path, opts.output_path)
    guard_error = check_output_not_existing_file(output_path, existing_txt_path)
    if guard_error is not None:
        return RealignPipelineResult(success=False, error=guard_error)
    write_song(result.song, output_path)
    log(f"Wrote {output_path}")
    return RealignPipelineResult(success=True, output_path=output_path)


def run_realign_batch(parent_dir: Path, opts: RealignPipelineOptions,
                       *, log: Callable[[str], None] = print) -> List[Tuple[str, RealignPipelineResult]]:
    """Runs `run_realign_pipeline` once per immediate subdirectory of
    `parent_dir` (mirrors `batch.run_batch`'s shape exactly), auto-
    detecting each subfolder's own existing .txt to realign (see
    `find_existing_txt_in_folder`) -- a single explicit path can't apply
    across multiple subfolders, so `opts.output_path`/a single existing-
    txt override are never meaningful here; the caller is expected to
    leave those None (see gui.py/`run`'s own incompatibility checks).
    Unlike `run_batch`, there's no output-folder mirroring to set up --
    each result is always written next to ITS OWN subfolder's existing
    file, so no `output_parent_dir` parameter exists here at all.

    Any exception from a single song -- even one `run_realign_pipeline`
    itself didn't already catch -- is caught HERE, logged, and recorded
    as a failed result; one bad song must never abort the rest of the
    batch, same reasoning as `run_batch`."""
    parent_dir = Path(parent_dir)
    subdirs = sorted(p for p in parent_dir.iterdir() if p.is_dir())
    results: List[Tuple[str, RealignPipelineResult]] = []

    for i, sub in enumerate(subdirs, 1):
        log(f"== Batch {i}/{len(subdirs)}: {sub.name} ==")
        try:
            result = run_realign_pipeline(sub, None, opts, log=log)
        except Exception as e:
            log(f"  FAILED (unexpected error): {e}")
            result = RealignPipelineResult(success=False, error=str(e))
        if not result.success:
            log(f"  FAILED: {result.error}")
        results.append((sub.name, result))

    n_ok = sum(1 for _, r in results if r.success)
    log(f"\nBatch complete: {n_ok}/{len(results)} song(s) succeeded.")
    for name, result in results:
        status = "OK" if result.success else f"FAILED ({result.error})"
        log(f"  {name}: {status}")

    return results


def _opts_from_args(args) -> RealignPipelineOptions:
    return RealignPipelineOptions(
        audio_file=args.audio_file, work_dir=args.work_dir, whisper_model=args.whisper_model,
        demucs_model=args.demucs_model, skip_separation=args.skip_separation, vocals_path=args.vocals_path,
        no_whisperx=args.no_whisperx, whisperx_no_vad=args.whisperx_no_vad,
        artist=args.artist, title=args.title, lrclib_id=args.lrclib_id,
        use_lrc=args.use_lrc, lrc_mode=args.lrc_mode, output_path=args.output_path,
    )


def run(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    args = build_arg_parser().parse_args(argv)

    if args.batch:
        incompatible = []
        if args.existing_txt_path:
            incompatible.append("--existing-txt")
        if args.audio_file:
            incompatible.append("--audio-file")
        if args.work_dir:
            incompatible.append("--work-dir")
        if args.artist or args.title:
            incompatible.append("--artist/--title")
        if args.lrclib_id:
            incompatible.append("--lrclib-id")
        if incompatible:
            print(f"--batch is not allowed together with {', '.join(incompatible)} "
                  f"(a single override doesn't make sense across multiple songs).", file=sys.stderr)
            return 1

    from .main import check_cuda_available
    cuda_error = check_cuda_available()
    if cuda_error:
        print(cuda_error, file=sys.stderr)
        return 1

    input_dir = Path(args.input).resolve()
    opts = _opts_from_args(args)

    if args.batch:
        results = run_realign_batch(input_dir, opts)
        return 0 if all(r.success for _, r in results) else 2

    existing_txt_path = Path(args.existing_txt_path).resolve() if args.existing_txt_path else None
    result = run_realign_pipeline(input_dir, existing_txt_path, opts)
    if not result.success:
        print(result.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run())
