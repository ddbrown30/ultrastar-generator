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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from . import config
from .debug_log import DebugLog
from .file_discovery import sanitize_filename
from .models import LineBreak, Song, Syllable, Word
from .usdx_parser import ParsedSong, UsdxParseError, parse_usdx_file
from .usdx_writer import write_song
from .lyrics_lookup import LrcLibCandidate, fetch_lrclib_by_id, load_lrc_file
from .lrc_timing import (match_asr_to_lrc_lines, two_tier_time_calibration, check_repeat_structure,
                          reconcile_line_structure, find_cursor_window_match, lrc_line_window,
                          match_block_to_candidates, words_in_time_window)
from .mxl_lrc_generator import MxlWord, select_lrc_candidate, assign_words_to_lines
from .text_normalize import normalize_word as _normalize
from .worker_process import run_cancellable, WorkerError


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
    n_force_aligned: int = 0  # see _force_align_unconfident_runs's own docstring
    n_interpolated: int = 0
    n_kept_original: int = 0
    # Longest CONSECUTIVE run of words with no real anchor (asr/lrc/
    # force-align) at all -- see config.RETRY_ASR_MIN_UNCONFIDENT_RUN.
    # A per-PASSAGE signal, independent of anchor_rate: real case (David
    # Bowie - Magic Dance, small.en) had a song-wide anchor rate of 58%
    # (well above the retry bar) while one hallucinated decoder segment
    # still dragged a real, contiguous passage ~12-14s off in the final
    # output -- an aggregate rate spread across an otherwise well-matched
    # song hid it completely.
    longest_unconfident_run: int = 0

    @property
    def anchor_rate(self) -> float:
        return ((self.n_asr_matched + self.n_lrc_seeded + self.n_force_aligned)
                 / self.n_words if self.n_words else 0.0)


def _longest_false_run(confident: List[bool]) -> int:
    """Longest consecutive run of `False` in `confident` -- see
    `RealignQuality.longest_unconfident_run`'s docstring."""
    best = cur = 0
    for c in confident:
        cur = 0 if c else cur + 1
        best = max(best, cur)
    return best


def _force_align_unconfident_runs(words_text: List[str], starts: List[Optional[float]],
                                   ends: List[Optional[float]], confident: List[bool],
                                   vocals_path: Path, *, debug_log: Optional[DebugLog] = None
                                   ) -> int:
    """For each
    contiguous run of `confident[i] == False`, forces the EXISTING file's
    OWN text for that run onto the audio window between the nearest
    CONFIDENT neighbors (or 0.0/audio-end at the edges), via
    `transcription.force_align_words_in_window` -- a real, measured
    wav2vec2 CTC forced alignment, not a guess. Unlike a local text-search
    rematch (searches ASR's own already-decoded words in the window --
    useless if the decoder produced none there; tried as `realign.
    rematch_local_gaps`, removed 2026-08-15 as a net regression, see
    CLAUDE.md), this doesn't depend on ASR having transcribed anything at
    all in the gap.

    Mutates `starts`/`ends`/`confident` in place for any run that
    succeeds; leaves a run untouched (falls through to
    `interpolate_fallback` as before) if the window's too short or the
    alignment result doesn't validate. Returns how many words were
    recovered this way. Lazily loads the audio/align model at most once,
    and only if there's actually an unconfident run to try -- a clean
    file with no gaps never pays this cost.

    Takes plain `words_text` (not the existing file's own ExistingWord
    objects) -- the only thing this ever needed from each word was its
    own text -- so a cancellable-subprocess caller (see
    _force_align_unconfident_runs_cancellable/worker_process.py) can
    serialize the input as plain strings, no reconstruction needed."""
    if not any(not c for c in confident):
        return 0
    from .transcription import force_align_words_in_window
    import whisperx
    from . import model_cache

    audio = None
    align_model = metadata = None
    audio_duration = 0.0
    promoted = 0

    n = len(words_text)
    idx = 0
    while idx < n:
        if confident[idx]:
            idx += 1
            continue
        run_start = idx
        while idx < n and not confident[idx]:
            idx += 1
        run = list(range(run_start, idx))

        if audio is None:
            audio = whisperx.load_audio(str(vocals_path))
            align_model, metadata = model_cache.get_whisperx_align_model()
            audio_duration = len(audio) / 16000.0

        win_start = ends[run_start - 1] if run_start > 0 and ends[run_start - 1] is not None else 0.0
        win_end = starts[idx] if idx < n and starts[idx] is not None else audio_duration
        win_start = max(0.0, min(win_start, audio_duration))
        win_end = max(0.0, min(win_end, audio_duration))

        run_text = [words_text[k].strip() for k in run]
        aligned = force_align_words_in_window(run_text, win_start, win_end, align_model, metadata, audio)
        if aligned is None:
            if debug_log is not None:
                debug_log.line(f"  force-align: gap [{win_start:8.3f}-{win_end:8.3f}] "
                                f"({len(run)} word(s) {' '.join(run_text)!r}) -- no usable result, "
                                f"left for interpolation")
            continue

        for k, (ws, we, score) in zip(run, aligned):
            starts[k] = ws
            ends[k] = we
            confident[k] = True
            promoted += 1
        if debug_log is not None:
            debug_log.line(f"  force-align: gap [{win_start:8.3f}-{win_end:8.3f}] "
                            f"({len(run)} word(s) {' '.join(run_text)!r}) -- recovered via forced alignment")

    return promoted


def _force_align_unconfident_runs_cancellable(words_text: List[str], starts: List[Optional[float]],
                                               ends: List[Optional[float]], confident: List[bool],
                                               vocals_path: Path, *,
                                               cancel_requested: Optional[Callable[[], bool]],
                                               debug_log: Optional[DebugLog] = None,
                                               log: Callable[[str], None] = print) -> int:
    """Same contract as `_force_align_unconfident_runs` (mutates
    starts/ends/confident in place, returns the promoted count) -- routes
    through a killable child process (worker_process.run_cancellable)
    when `cancel_requested` is given, so the GUI's Stop button can kill
    this mid-call instead of waiting for it to finish; falls straight
    through to the in-process call otherwise (the CLI, and any other
    caller with no cancel_requested at all)."""
    if cancel_requested is None:
        return _force_align_unconfident_runs(words_text, starts, ends, confident, vocals_path,
                                               debug_log=debug_log)
    result = run_cancellable(
        "force_align_unconfident_runs",
        {"words_text": words_text, "starts": starts, "ends": ends, "confident": confident,
         "vocals_path": str(vocals_path)},
        cancel_requested=cancel_requested, debug_log=debug_log, log=log,
    )
    starts[:] = result["starts"]
    ends[:] = result["ends"]
    confident[:] = result["confident"]
    return result["promoted"]


def _line_chunks(line_of_word: List[int]) -> List[Tuple[int, int]]:
    """Groups consecutive existing-word indices sharing the same real
    line index into (start, end) chunk bounds, in order -- the natural
    unit `match_words_to_asr` chunks by when real line info is given
    (see its own docstring for why a real line beats an arbitrary
    fixed-size chunk)."""
    chunks: List[Tuple[int, int]] = []
    if not line_of_word:
        return chunks
    start = 0
    cur = line_of_word[0]
    for idx in range(1, len(line_of_word)):
        if line_of_word[idx] != cur:
            chunks.append((start, idx))
            start = idx
            cur = line_of_word[idx]
    chunks.append((start, len(line_of_word)))
    return chunks


def match_words_to_asr(existing_words: List[ExistingWord], asr_words: List[Word],
                        line_of_word: Optional[List[int]] = None
                        ) -> Tuple[List[Optional[float]], List[Optional[float]], List[bool]]:
    """Whole-song, order-preserving text match of the existing file's own
    words against real ASR words -- deliberately NOT TIME-windowed (see
    module docstring for why: this mode can't trust the input file's own
    timing enough to window a search with it).

    The window search itself (forward-only cursor, tight-vs-wide, quality
    gates -- the part responsible for this project's own recurring
    "repeated-phrase disambiguation" failure class, see CLAUDE.md's
    "Lessons learned") is `lrc_timing.find_cursor_window_match` -- see its
    own docstring for the full real-bug history behind it, and CLAUDE.md's
    "Shared cursor-window matching" note for the full list of callers to
    re-check whenever that shared function changes. This function's own
    remaining job is CHUNKING existing words and interpreting a winning
    match for this shape: mark every individually-matched word's own
    start/end/confidence, with a fuzzy-ratio fallback for a `"replace"`
    block (ASR mishearing a word differently than the reference text,
    same technique `mxl_lrc_generator.place_words_via_asr` already
    validated) -- neither of which lives in the shared search itself,
    since interpreting a match is genuinely caller-specific (this wants
    every individual word's own position; `match_asr_to_lrc_lines` wants
    just the line's own single earliest-word anchor).

    Processes existing words in CHUNKS, not one at a time -- a single
    common word (e.g. "the"/"a") has almost no power to disambiguate ITS
    OWN position in a repeat-heavy song on its own; a several-word chunk
    does. `line_of_word` (`_word_line_indices`, when the caller has real
    entries to derive it from), when given, chunks by REAL LYRIC LINES
    -- the same unit `reconcile_line_structure`/`match_asr_to_lrc_lines`
    already use successfully for this exact failure class -- rather than
    an arbitrary fixed word count. This mattered in practice: an EARLIER
    version of this fix used fixed-size 6-word chunks and, even after
    several rounds of tightening the shared search's own guards, still
    occasionally accepted a wrong match on a real repeated-chorus stretch
    -- an arbitrary chunk boundary can slice a real line in half, leaving
    neither half enough distinctive content to reliably self-disambiguate.
    A real line varies its own length to match real content and rarely
    straddles a phrase boundary, which made the difference. Falls back
    to a fixed-size chunk (`CHUNK_SIZE`) when `line_of_word` isn't given
    (e.g. a caller/test with no entries to derive real lines from) --
    still real cursor-based matching, just without the line-boundary
    quality improvement.

    Returns (starts, ends, confident), parallel to `existing_words` --
    unmatched/low-confidence words are None/False."""
    CHUNK_SIZE = 6
    MAX_PENDING_WORDS = 60

    n = len(existing_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    confident: List[bool] = [False] * n

    a_norm = [w.norm for w in existing_words]
    b_norm = [_normalize(w.text) for w in asr_words]

    if line_of_word is not None and len(line_of_word) == n:
        chunks = _line_chunks(line_of_word)
    else:
        chunks = [(i, min(i + CHUNK_SIZE, n)) for i in range(0, n, CHUNK_SIZE)]

    def _apply_match(opcodes, window: List[str], chunk_tokens: List[str], chunk_start: int, cursor: int) -> Optional[int]:
        """Marks every confirmed word from `opcodes` as matched. Returns
        the offset (within `window`) of the last ACTUALLY CONFIRMED
        match -- never a raw, unconfirmed opcode boundary (a `"replace"`
        block's own span is just wherever difflib decided the mismatched
        region ends, not a real match; real bug found via Pink Pony Club
        validation: using that raw boundary to advance the cursor could
        jump it all the way to the edge of a large window even though the
        real match sat much earlier)."""
        last_confirmed_offset: Optional[int] = None
        for tag, wa1, wa2, ca1, ca2 in opcodes:
            if tag == "equal":
                for k in range(wa2 - wa1):
                    asr_w = asr_words[cursor + wa1 + k]
                    ex_idx = chunk_start + ca1 + k
                    if asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                        starts[ex_idx] = asr_w.start
                        ends[ex_idx] = asr_w.end
                        confident[ex_idx] = True
                        last_confirmed_offset = max(last_confirmed_offset or 0, wa1 + k)
            elif tag == "replace" and (ca2 - ca1) == 1:
                # A single unmatched existing word against one or more ASR
                # words in this window slice -- try every candidate, keep
                # the best fuzzy match, same technique `mxl_lrc_generator.
                # place_words_via_asr` already validated for exactly this
                # failure mode (ASR mishearing a word differently than the
                # reference text).
                best_ratio = 0.0
                best_asr_w = None
                best_wk = None
                for wk in range(wa1, wa2):
                    ratio = difflib.SequenceMatcher(None, chunk_tokens[ca1], window[wk]).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_asr_w = asr_words[cursor + wk]
                        best_wk = wk
                if best_ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO and best_asr_w is not None:
                    if best_asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                        ex_idx = chunk_start + ca1
                        starts[ex_idx] = best_asr_w.start
                        ends[ex_idx] = best_asr_w.end
                        confident[ex_idx] = True
                        last_confirmed_offset = max(last_confirmed_offset or 0, best_wk)
        return last_confirmed_offset

    cursor = 0
    pending_word_count = 0
    for chunk_start, chunk_end in chunks:
        chunk_tokens = a_norm[chunk_start:chunk_end]
        pending_word_count = min(pending_word_count + len(chunk_tokens), MAX_PENDING_WORDS)

        found = find_cursor_window_match(cursor, b_norm, chunk_tokens, pending_word_count,
                                          lenient_min_matches=1)
        if found is None:
            # Not confirmable in either window -- don't advance the
            # cursor; the NEXT chunk's own search starts from the same
            # position, its window growing to cover this chunk's skipped
            # content too. Every word in this chunk stays None/False for
            # interpolate_fallback.
            continue
        opcodes, window = found

        last_confirmed_offset = _apply_match(opcodes, window, chunk_tokens, chunk_start, cursor)
        if last_confirmed_offset is not None:
            cursor += last_confirmed_offset + 1
        pending_word_count = 0

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


def _reconstruct_our_lines(entries: List[Union[Syllable, LineBreak]]) -> List[str]:
    """Rebuilds the existing file's own display lines (delimited by its own
    LineBreak markers) -- used to compare repeat STRUCTURE against an LRC
    candidate's own lines (see `check_repeat_structure`), the same "-"
    boundaries phrasing.build_lines writes for a freshly-generated song."""
    lines: List[str] = []
    cur: List[str] = []
    for e in entries:
        if isinstance(e, LineBreak):
            if cur:
                lines.append("".join(cur))
            cur = []
            continue
        prefix = " " if e.is_word_start and cur else ""
        cur.append(prefix + e.text)
    if cur:
        lines.append("".join(cur))
    return lines


def _word_line_indices(entries: List[Union[Syllable, LineBreak]],
                        existing_words: List[ExistingWord]) -> List[int]:
    """Parallel to `existing_words` -- which `_reconstruct_our_lines`
    display-line index (SAME numbering, built from the SAME walk over
    `entries`) each word belongs to. Exists so `prepare_lrc` can turn
    `reconcile_line_structure`'s own `our_line_index` (which of OUR
    lines each reconciled LRC entry came from) directly into a per-WORD
    LRC-line mapping -- reusing the correspondence reconcile_line_
    structure already established at the (more reliable) LINE level,
    instead of re-deriving it via `assign_words_to_lines`'s own
    independent whole-song WORD-level diff, which has no line-boundary
    information at all and can disagree with reconcile_line_structure's
    own answer (real confirmed case, 2026-08-15: Trixie Mattel - Video
    Games -- see LineReconciliation's own docstring)."""
    entry_line: List[int] = []
    line_idx = 0
    cur_has_content = False
    for e in entries:
        if isinstance(e, LineBreak):
            if cur_has_content:
                line_idx += 1
                cur_has_content = False
            entry_line.append(-1)  # never looked up -- a word's own
                                     # entry_indices never include a
                                     # LineBreak (extract_words skips them)
            continue
        entry_line.append(line_idx)
        cur_has_content = True
    return [entry_line[w.entry_indices[0]] for w in existing_words]


# `check_repeat_structure` moved to lrc_timing.py, 2026-08-11 (re-exported
# here unchanged so `from .realign import check_repeat_structure` --
# including test_dry_run.py's own import -- keeps working) -- lrc_timing.
# py's own tier-3 rescue gate (see `two_tier_time_calibration`'s
# `structural_check` param) needed the exact same check and can't import
# FROM realign.py (this module already imports FROM lrc_timing.py), so it
# moved to the shared module instead of being duplicated a second time.


def prepare_lrc(existing_words: List[ExistingWord], asr_words: List[Word],
                 artist: str, title: str, audio_duration: float,
                 forced_candidate: Optional[LrcLibCandidate] = None,
                 our_lines: Optional[List[str]] = None,
                 our_line_of_word: Optional[List[int]] = None,
                 log: Optional[Callable[[str], None]] = None) -> Optional[LrcPrep]:
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
    candidate-selection/line-assignment logic a third time.

    `our_lines` (the existing file's own display lines, see
    `_reconstruct_our_lines`) -- when given -- additionally RECONCILES a
    candidate whose REPEAT STRUCTURE doesn't match ours (see
    `reconcile_line_structure`; real confirmed case: David Bowie - "I'm
    Afraid of Americans", CLAUDE.md) instead of rejecting it outright: a
    localized repeat-count difference (an edition with a few extra/
    missing chorus repeats) gets the extra lines on whichever side
    dropped, and the candidate's own lines used from here on are the
    reconciled subset, not its raw lines. Still rejects outright (returns
    None) when reconciliation can't find a real correspondence for most
    of our own lines at all (see `reconcile_line_structure`'s own
    `min_match_ratio`) -- a genuinely different recording/arrangement,
    not just a differing repeat count. Optional and skipped (no
    reconciliation/rejection) when not given, since not every caller has
    the original entries handy (e.g. `seed_lrc_anchors`'s own standalone/
    test-only call path)."""
    fake_words = [MxlWord(text=w.text, norm=w.norm, offset=float(i), syllables=[])
                  for i, w in enumerate(existing_words)]

    lrc_match = select_lrc_candidate(artist, title, fake_words, audio_duration, forced=forced_candidate)
    if lrc_match is None:
        return None

    candidate_lrc_lines = lrc_match.lrc_lines
    reconciliation = None
    if our_lines is not None:
        lrc_line_texts = [text for _t, text in candidate_lrc_lines]
        reconciliation = reconcile_line_structure(our_lines, candidate_lrc_lines)
        if reconciliation is None:
            if log is not None:
                log(f"  Warning: LRC candidate {lrc_match.candidate.track_name!r}/{lrc_match.candidate.artist_name!r} "
                    f"rejected: repeat structure doesn't reconcile against our own lines "
                    f"({len(our_lines)} of our own lines, {len(lrc_line_texts)} candidate lines)")
            return None
        if reconciliation.n_lrc_dropped or reconciliation.n_our_unmatched:
            if log is not None:
                log(f"  LRC candidate {lrc_match.candidate.track_name!r}/{lrc_match.candidate.artist_name!r} "
                    f"repeat structure reconciled: {reconciliation.n_matched}/{reconciliation.n_our_lines} of "
                    f"our own lines matched, {reconciliation.n_lrc_dropped} candidate line(s) dropped as extras, "
                    f"{reconciliation.n_our_unmatched} of our own line(s) have no LRC counterpart")
        candidate_lrc_lines = reconciliation.lrc_lines

    # Calibrate away a systematic offset between LRC's own line timestamps
    # and OUR audio's real timing (e.g. different lead-in silence) BEFORE
    # trusting a line start as an anchor -- same technique/function
    # mxl_lrc_generator uses, factored into lrc_timing.py for exactly this
    # kind of reuse. `structural_check` deliberately reuses check_repeat_
    # structure against the candidate's RAW (un-reconciled) lines, not the
    # filtered ones above -- it's answering a different question (is this
    # fundamentally the WRONG recording, e.g. a choral cover) than the
    # reconciliation above (is this the right recording with a differing
    # repeat count) -- gates tier 3's own "rescue" case (see
    # two_tier_time_calibration's own docstring).
    time_candidates = match_asr_to_lrc_lines(asr_words, candidate_lrc_lines)
    structural_check = None
    if our_lines is not None:
        lrc_line_texts_for_check = [text for _t, text in lrc_match.lrc_lines]
        structural_check = lambda: check_repeat_structure(our_lines, lrc_line_texts_for_check)
    offset, slope, confidence, kind, _skipped, correction_fn, _holdout = two_tier_time_calibration(
        time_candidates, structural_check=structural_check)
    lrc_lines = candidate_lrc_lines
    if offset is not None:
        lrc_lines = [(correction_fn(t), text) for t, text in lrc_lines]

    # Word-to-line assignment: when reconciliation ran AND the caller gave
    # us a per-word line mapping (`our_line_of_word`, see
    # `_word_line_indices`), build word_lines DIRECTLY from reconcile_
    # line_structure's own `our_line_index` -- reusing the correspondence
    # already established at the LINE level -- instead of calling
    # assign_words_to_lines, whose own independent whole-song WORD-level
    # diff has no line-boundary information at all and can disagree with
    # reconcile_line_structure's own answer (real confirmed regression,
    # 2026-08-15: Trixie Mattel - Video Games -- a 3x-repeated chorus
    # correctly resolved by reconcile_line_structure was still placed
    # ~30s off the real audio by assign_words_to_lines's own re-diff; see
    # LineReconciliation's own docstring for the full case). Falls back
    # to assign_words_to_lines when reconciliation wasn't used at all
    # (`our_lines`/`our_line_of_word` not given -- e.g. seed_lrc_anchors's
    # own standalone/test-only call path).
    if reconciliation is not None and our_line_of_word is not None:
        our_idx_to_li = {our_idx: li for li, our_idx in enumerate(reconciliation.our_line_index)}
        word_lines = [our_idx_to_li.get(our_line) for our_line in our_line_of_word]
    else:
        word_lines, _clean_text, _group, _group_text, _syl_override, _lrc_candidate = assign_words_to_lines(
            fake_words, lrc_lines)

    return LrcPrep(lrc_match=lrc_match, lrc_lines=lrc_lines, word_lines=word_lines,
                    calibration_offset=offset, calibration_kind=kind, calibration_confidence=confidence)


def seed_lrc_anchors(existing_words: List[ExistingWord], asr_words: List[Word],
                      starts: List[Optional[float]], ends: List[Optional[float]], confident: List[bool],
                      artist: str, title: str, audio_duration: float,
                      forced_candidate: Optional[LrcLibCandidate] = None,
                      our_lines: Optional[List[str]] = None,
                      log: Optional[Callable[[str], None]] = None) -> Optional[LrcSeedResult]:
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
    prep = prepare_lrc(existing_words, asr_words, artist, title, audio_duration, forced_candidate,
                        our_lines=our_lines, log=log)
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
        t0, t1 = lrc_line_window(lrc_lines, li)
        asr_in_window = words_in_time_window(asr_words, t0, t1)
        existing_norm_line = [existing_words[i].norm for i in idxs]

        matched_local = match_block_to_candidates(existing_norm_line, asr_in_window)

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

    # Non-decreasing starts -- but a CONFIDENT word's own real match is a
    # FIXED POINT, never moved by a neighboring fallback word's estimate.
    # Real bug found (David Bowie - "I'm Afraid of Americans", a song with
    # many near-identical repeated short lines): one fallback word's
    # over-extrapolated estimate exceeded FOURTEEN subsequent, genuinely
    # confident ASR matches -- the old forward-only clamp (which treated
    # every index identically regardless of `confident`) dragged all
    # fourteen of them up to that single wrong value instead of trusting
    # them, destroying real information, not just filling a gap. Only
    # fallback words are adjusted in either direction here: pass 1 pushes
    # a fallback word forward past a larger PRECEDING value; pass 2 (in
    # reverse) pulls a fallback word back if it exceeds the next
    # CONFIDENT word's own value. Confident words themselves are never
    # rewritten by either pass.
    for i in range(1, n):
        if confident[i]:
            continue
        if starts[i] < starts[i - 1]:
            starts[i] = starts[i - 1]
    next_confident_start = None
    for i in range(n - 1, -1, -1):
        if confident[i]:
            next_confident_start = starts[i]
            continue
        if next_confident_start is not None and starts[i] > next_confident_start:
            starts[i] = next_confident_start
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
                  vocals_path: Optional[Path] = None,
                  log: Callable[[str], None] = print,
                  debug_log: Optional[DebugLog] = None,
                  cancel_requested: Optional[Callable[[], bool]] = None) -> RealignResult:
    """Core realignment step: given an already-parsed existing song and
    already-transcribed ASR words, returns a new `Song`
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
        whenever LRC can't be trusted.

    If "windowed" mode's own result still ends up with fewer than
    `config.MXL_LRC_MIN_ASR_PLACEMENT_RATE` of words getting a real anchor
    (a low anchor rate under a CONFIDENTLY-calibrated candidate means the
    per-line window itself is probably mis-targeted to this file's own line
    structure, not that ASR can't find the words at all -- same class of
    problem as `realign_song_validate`'s own fallback, just triggered by a
    different mechanism), this transparently retries once with
    `lrc_mode="seed"` and returns THAT result instead. Never retries a
    second time -- "seed" mode's own low anchor rate (or "windowed" already
    having degraded to "seed" behavior because calibration wasn't
    confident) is left as the final, if disappointing, answer.

    Force-aligns the existing file's OWN text for any STILL-unconfident run (after the
    above) onto the audio window between its nearest confident neighbors
    via a real wav2vec2 CTC forced alignment (`transcription.
    force_align_words_in_window`) -- unlike a text-search-based rematch,
    this doesn't need ASR to have transcribed anything in the gap at all,
    since it isn't searching a transcript, it's measuring where the GIVEN
    text fits. Requires `vocals_path` (the loaded audio + align model are
    only paid for if there's an actual gap to try). Adapted from
    UltraStarKaraokeMaker's own `realign_gap_windows` -- see CLAUDE.md."""
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
    our_line_of_word = _word_line_indices(entries, words)

    lrc_prep = None
    if use_lrc and audio_duration is not None:
        our_lines = _reconstruct_our_lines(entries)
        lrc_prep = prepare_lrc(words, asr_words, artist or existing.artist, title or existing.title,
                                audio_duration, forced_candidate=forced_lrc_candidate,
                                our_lines=our_lines, our_line_of_word=our_line_of_word, log=log)

    # BOTH strategies require a CONFIDENT time calibration before an LRC
    # candidate is trusted at all -- real comparison (see CLAUDE.md, BATB/
    # Stars/Chicago) already found windowing raw, UNCALIBRATED LRC lines
    # actively harmful for "windowed" (every word's match routes through the
    # same untrusted signal). A SECOND real case (David Bowie - "Heroes")
    # showed "seed" is NOT safe either, despite it only ever seeding a
    # handful of words directly: the LRC candidate found was a Kolacny
    # Brothers CHORAL COVER (confirmed via calibration_confidence=0.0, a
    # real "zero agreement" case, not just "not enough samples") -- ONE bad
    # seed anchor landed later in time than several of its own already-
    # correctly-ASR-matched NEIGHBORS, and `interpolate_fallback`'s
    # monotonic clamp (forward-push only, no notion of "this anchor might
    # itself be wrong") then dragged every one of those correct neighbors
    # forward to match the bad anchor -- corrupting real, independently-
    # confirmed-correct ASR matches, not just filling a genuine gap. Once
    # confirmed, this obsoleted the earlier "seed's blast radius is small
    # by construction" reasoning: a single bad anchor's blast radius is
    # actually unbounded once the monotonic clamp propagates it forward,
    # regardless of strategy. Both strategies now use the exact same gate.
    lrc_seed = None
    lrc_confident = lrc_prep is not None and lrc_prep.calibration_offset is not None
    if lrc_mode == "windowed" and lrc_prep is not None and not lrc_confident:
        log("  LRC candidate found but no confident time calibration -- 'windowed' mode isn't safe to use "
            "here (would window the ENTIRE match against an uncalibrated signal), falling back to "
            "whole-song ASR matching instead.")
    if lrc_mode == "windowed" and lrc_confident:
        starts, ends, confident = match_words_to_asr_windowed(words, lrc_prep.word_lines, lrc_prep.lrc_lines,
                                                                asr_words)
        quality = RealignQuality(n_words=len(words), n_asr_matched=sum(confident))
        c = lrc_prep.lrc_match.candidate
        log(f"  LRC-windowed matching: {c.track_name!r}/{c.artist_name!r} (lrclib id={c.id}), "
            f"time calibration ({lrc_prep.calibration_kind}) offset {lrc_prep.calibration_offset:+.1f}s "
            f"({lrc_prep.calibration_confidence:.0%} agreement)")
    else:
        starts, ends, confident = match_words_to_asr(words, asr_words, line_of_word=our_line_of_word)
        quality = RealignQuality(n_words=len(words), n_asr_matched=sum(confident))
        if use_lrc and audio_duration is not None:
            if lrc_prep is not None and lrc_confident:
                lrc_seed = seed_from_prep(words, lrc_prep, starts, ends, confident)
                if lrc_seed is not None:
                    quality.n_lrc_seeded = lrc_seed.n_seeded
                    log(f"  LRC anchors: {lrc_seed.lrc_match.candidate.track_name!r}/"
                        f"{lrc_seed.lrc_match.candidate.artist_name!r} (lrclib id={lrc_seed.lrc_match.candidate.id}) -- "
                        f"seeded {lrc_seed.n_seeded} additional line-start anchor(s)"
                        + (f", time calibration ({lrc_seed.calibration_kind}) offset "
                           f"{lrc_seed.calibration_offset:+.1f}s ({lrc_seed.calibration_confidence:.0%} agreement)"
                           if lrc_seed.calibration_offset is not None else ", no time calibration found"))
            elif lrc_prep is not None:
                log(f"  LRC candidate found ({lrc_prep.lrc_match.candidate.track_name!r}/"
                    f"{lrc_prep.lrc_match.candidate.artist_name!r}) but no confident time calibration -- "
                    f"NOT trusted as a seed anchor either (a wrong-recording candidate's uncalibrated line "
                    f"timestamp can corrupt correctly-matched neighbors via the monotonic clamp -- confirmed "
                    f"real case, see CLAUDE.md), falling back to ASR-only interpolation.")
            else:
                log("  No usable LRCLIB synced-lyrics candidate for this song -- ASR matching only.")

    source = ["asr" if c else "" for c in confident]

    if vocals_path is not None:
        quality.n_force_aligned = _force_align_unconfident_runs_cancellable(
            [w.text for w in words], starts, ends, confident, vocals_path,
            cancel_requested=cancel_requested, debug_log=debug_log, log=log)
        if quality.n_force_aligned:
            log(f"  Force-alignment of known gaps: recovered {quality.n_force_aligned} word(s) via a real "
                f"wav2vec2 CTC forced alignment of the existing file's own text.")
        for i in range(len(words)):
            if confident[i] and not source[i]:
                source[i] = "force_align"

    quality.longest_unconfident_run = _longest_false_run(confident)

    had_any_anchor = any(confident)
    quality.n_interpolated, quality.n_kept_original = interpolate_fallback(words, starts, ends, confident)
    for i in range(len(words)):
        if not source[i]:
            source[i] = "interpolated" if had_any_anchor else "kept_original"

    log(f"  {quality.n_asr_matched}/{quality.n_words} word(s) matched directly to real ASR transcription, "
        f"{quality.n_lrc_seeded} from LRC line anchors, "
        f"{quality.n_force_aligned} from forced alignment of known gaps, "
        f"{quality.n_interpolated} interpolated between anchors, {quality.n_kept_original} kept "
        f"at their original timing (no anchor found nearby).")
    if quality.anchor_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:
        log(f"  WARNING: only {quality.anchor_rate:.0%} of words got a real anchor from the audio -- "
            f"this file's lyrics may not match this audio at all. Review the output carefully.")
        if lrc_mode == "windowed" and lrc_confident:
            log("  Falling back to 'seed' mode (whole-song ASR primary) -- this low an anchor rate under "
                "per-line LRC windowing may mean the windowed search is mis-targeted to this file's own "
                "line structure, not that ASR itself can't find these words at all.")
            return realign_song(
                existing, asr_words, artist=artist, title=title, audio_duration=audio_duration,
                use_lrc=use_lrc, lrc_mode="seed", forced_lrc_candidate=forced_lrc_candidate,
                vocals_path=vocals_path, log=log, debug_log=debug_log,
            )

    if debug_log is not None:
        debug_log.section("REALIGN SUMMARY (strategy=replace)")
        debug_log.line(f"  {quality.n_asr_matched}/{quality.n_words} matched directly to ASR, "
                        f"{quality.n_lrc_seeded} from LRC anchors, "
                        f"{quality.n_force_aligned} from forced alignment of known gaps, "
                        f"{quality.n_interpolated} interpolated, {quality.n_kept_original} kept original.")
        if lrc_prep is not None:
            c = lrc_prep.lrc_match.candidate
            debug_log.line(f"  LRC candidate: {c.track_name!r}/{c.artist_name!r} (id={c.id}), "
                            f"calibration={lrc_prep.calibration_kind} offset={lrc_prep.calibration_offset} "
                            f"confidence={lrc_prep.calibration_confidence}")
        debug_log.section("PER-WORD TRACE (text, orig_start->orig_end, source, final_start->final_end)")
        for i, w in enumerate(words):
            debug_log.line(f"  {i:4d} {w.text:20s} orig={w.orig_start:9.3f}-{w.orig_end:9.3f}  "
                            f"[{source[i]:14s}]  final={starts[i]:9.3f}-{ends[i]:9.3f}  "
                            f"delta={starts[i] - w.orig_start:+8.3f}")
        debug_log.section(f"RAW ASR TRANSCRIPT ({len(asr_words)} word(s), real timestamps + confidence -- "
                           f"WhisperX/faster-whisper is NOT deterministic run-to-run, so this is the only record "
                           f"of what THIS specific run's ASR actually heard)")
        for aw in asr_words:
            debug_log.line(f"  {aw.start:9.3f}-{aw.end:9.3f}  conf={aw.confidence:.3f}  {aw.text!r}")

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


# --- "validate" strategy (PROTOTYPE, see CLAUDE.md) -------------------------
#
# `realign_song` above always REPLACES a matched word's timing with ASR's
# own value, even when the existing file was already correct there. This
# alternative strategy instead VALIDATES first: a note whose original
# position (after a single global GAP/drift correction -- see
# `compute_gap_calibration`, user's own framing: "sometimes the song file
# is nearly perfect but the GAP is wrong so everything is offset") is
# independently confirmed by a confident ASR match is left COMPLETELY
# UNTOUCHED, position and length both -- never overwritten by ASR's own,
# less precise, value. Only words that don't validate get repositioned, the
# same way `realign_song`'s fallback words already do (reusing
# `interpolate_fallback` unchanged -- a validated word is exactly as
# trustworthy an anchor as a directly-ASR-matched one, so no new
# interpolation logic is needed). NOT YET wired into the CLI/GUI as a real
# selectable strategy -- prototype only, pending real-audio comparison
# against `realign_song`'s existing "replace" behavior.


@dataclass
class GapCalibration:
    offset: Optional[float]
    slope: float
    confidence: float
    kind: Optional[str]           # "constant", "drift", "piecewise", "isotonic", or None if uncalibrated
    skipped_reason: Optional[str]
    asr_starts: List[Optional[float]]   # whole-song match_words_to_asr's own raw results,
    asr_ends: List[Optional[float]]     # reused directly by validate_words_against_asr
    asr_confident: List[bool]           # so it's never recomputed a second time
    # `correction_fn(orig_start) -> corrected_real_time`, populated for
    # every successful `kind` (incl. constant/drift, for API uniformity)
    # -- see two_tier_time_calibration's own docstring. Defaults to None
    # so a manually-constructed GapCalibration (e.g. in tests) still
    # works via the offset/slope fallback in validate_words_against_asr.
    correction_fn: Optional[Callable[[float], float]] = None


def compute_gap_calibration(existing_words: List[ExistingWord], asr_words: List[Word],
                             line_of_word: Optional[List[int]] = None) -> GapCalibration:
    """FIRST PASS: checks whether the whole file can be explained by a
    single constant real-time offset (or slow linear drift) from its OWN
    original timing -- i.e. #GAP (or a slow drift) is the only real
    problem, and the file's own relative note timing/rhythm is already
    correct.

    Reuses `match_words_to_asr` (the same whole-song, order-preserving,
    repeat-safe ASR matching used elsewhere in this module) to find
    confident (original_start, real_asr_start) pairs, then feeds them into
    `two_tier_time_calibration` (the SAME robust, repeat-safe two-tier fit
    already validated for LRC line calibration -- don't reimplement a
    third time) using each word's own original start (seconds) as the "x"
    axis, exactly mirroring how LRC line timestamps are used there.

    No `structural_check` passed -- unlike LRC candidate calibration,
    there's no separate "candidate identity" to verify here (both sides
    are the SAME existing file's own words vs. the SAME audio's own
    ASR, just two different timing views of one source, never a
    different recording). Tier 3's "rescue" case therefore never fires
    for GAP calibration -- only "refine" (tier 1/2 already found some
    real support) -- which is the safe, correct default for this call
    site, not a missing feature."""
    starts, ends, confident = match_words_to_asr(existing_words, asr_words, line_of_word=line_of_word)
    candidates = [(i, w.orig_start, starts[i] - w.orig_start)
                  for i, w in enumerate(existing_words) if confident[i]]
    offset, slope, confidence, kind, skipped_reason, correction_fn, _holdout = two_tier_time_calibration(candidates)
    return GapCalibration(offset=offset, slope=slope, confidence=confidence, kind=kind,
                           skipped_reason=skipped_reason, asr_starts=starts, asr_ends=ends,
                           asr_confident=confident, correction_fn=correction_fn)


def validate_words_against_asr(existing_words: List[ExistingWord], gap: GapCalibration,
                                tolerance_sec: float = config.REALIGN_VALIDATE_TOLERANCE_SEC
                                ) -> Tuple[List[Optional[float]], List[Optional[float]], List[bool]]:
    """For each word: computes its expected start after applying the global
    GAP correction found above (a no-op shift of 0 if no confident
    calibration was found -- validates against the ORIGINAL timing
    directly in that case). If a confident whole-song ASR match ALSO
    exists for this word and its own real START lands within
    `tolerance_sec` of that expected position, the word VALIDATES -- but
    its final position is ASR's OWN start (not the original/expected
    one), on the same original duration. Length is never touched by ASR
    either way (see below).

    Real bug found via real hand-timed ground truth (2026-08-15, Our Lady
    Peace - "Somewhere Out There", user's own `truth.txt`): the ORIGINAL
    version of this function kept the GAP-corrected ORIGINAL start
    EXACTLY, discarding ASR's own (already-known-close, since it just
    passed this exact check) position entirely. When the original file's
    own timing carried a small systematic bias (~0.12s here, well inside
    the default 0.3s tolerance), every validated word inherited that same
    bias -- and so did every INTERPOLATED word between validated anchors,
    since interpolation is relative to them. Real comparison against
    `truth.txt`: only 27.6%/51.2% of words landed within 100ms/150ms
    tolerance, vs. 51.9%/74.2% for `--strategy replace` on the identical
    audio -- confirming ASR's own reading is measurably more precise here
    once it clears the tolerance bar at all.

    This is NOT equivalent to `replace` with extra steps: `replace`
    (`match_words_to_asr`) trusts ANY confident ASR match unconditionally,
    with no comparison to the word's own original position at all. This
    function still requires ASR to already AGREE with the (GAP-corrected)
    original within `tolerance_sec` before trusting it at all -- a word
    where ASR found something wildly different (a real mismatch, e.g. the
    wrong occurrence of a repeated phrase) is still correctly rejected
    and left for `interpolate_fallback`, exactly as before. Only within
    that already-validated agreement zone does this now prefer ASR's
    reading over the original's for the small extra precision -- the
    safety net `validate` exists for is unchanged.

    Only ASR's START is ever used, never its END: ASR is known-unreliable
    at detecting the end of a sustained/held note (see CLAUDE.md's
    "Lessons learned" -- "Don't trust individual ASR word timestamps for
    fine-grained boundaries"), so a word's own already-correct LENGTH is
    always trusted from the input file, applied on top of ASR's start,
    rather than second-guessed by ASR's own, less reliable, end
    timestamp.

    Words that don't validate are left None/False for `interpolate_fallback`
    to resolve using validated neighbors (and, optionally, LRC anchors) as
    anchors."""
    n = len(existing_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    validated: List[bool] = [False] * n
    offset = gap.offset if gap.offset is not None else 0.0
    for i, w in enumerate(existing_words):
        if gap.correction_fn is not None:
            expected_start = gap.correction_fn(w.orig_start)
        else:
            expected_start = w.orig_start + offset + gap.slope * w.orig_start
        if gap.asr_confident[i] and abs(gap.asr_starts[i] - expected_start) <= tolerance_sec:
            orig_dur = w.orig_end - w.orig_start
            starts[i] = gap.asr_starts[i]
            ends[i] = gap.asr_starts[i] + orig_dur
            validated[i] = True
    return starts, ends, validated


@dataclass
class ValidateQuality:
    n_words: int = 0
    n_validated: int = 0
    n_lrc_seeded: int = 0
    n_interpolated: int = 0
    n_kept_original: int = 0

    @property
    def anchor_rate(self) -> float:
        return (self.n_validated + self.n_lrc_seeded) / self.n_words if self.n_words else 0.0


def realign_song_validate(existing: ParsedSong, asr_words: List[Word], *,
                           artist: Optional[str] = None, title: Optional[str] = None,
                           audio_duration: Optional[float] = None,
                           use_lrc: bool = True, forced_lrc_candidate: Optional[LrcLibCandidate] = None,
                           validate_tolerance_sec: float = config.REALIGN_VALIDATE_TOLERANCE_SEC,
                           log: Callable[[str], None] = print,
                           debug_log: Optional[DebugLog] = None) -> RealignResult:
    """PROTOTYPE strategy -- see the module comment above `GapCalibration`.
    Same overall shape/contract as `realign_song` (never raises, returns
    `RealignResult`, never touches pitch/note count/order), but each word
    is either VALIDATED (kept exactly as-is beyond a single global GAP
    correction) or repositioned via `interpolate_fallback` using validated
    (+ optional LRC) anchors -- never simply overwritten with ASR's own
    value the way `realign_song` does.

    If fewer than `config.MXL_LRC_MIN_ASR_PLACEMENT_RATE` of words end up
    validated or LRC-anchored, that's a sign the file's own original timing
    can't be trusted -- exactly the case this strategy is bad at (it only
    ever repositions relative to that untrusted timing). Falls back to
    `realign_song(lrc_mode="seed")` instead, which replaces every word's
    timing from ASR/LRC directly rather than trusting the input at all."""
    entries = copy.deepcopy(existing.entries)
    words = extract_words(entries)
    if not words:
        return RealignResult(success=False, error="Existing file has no words to realign.")
    our_line_of_word = _word_line_indices(entries, words)

    quality = ValidateQuality(n_words=len(words))

    gap = compute_gap_calibration(words, asr_words, line_of_word=our_line_of_word)
    if gap.offset is not None:
        drift_desc = f", drift {gap.slope:+.4f}s/s" if gap.kind == "drift" else ""
        rep_desc = " (representative only, see correction_fn)" if gap.kind in ("piecewise", "isotonic") else ""
        log(f"  GAP check ({gap.kind}): offset {gap.offset:+.2f}s{drift_desc}{rep_desc} "
            f"({gap.confidence:.0%} agreement) -- applying as a single correction before validating "
            f"individual words.")
    else:
        log(f"  GAP check: no confident single offset found ({gap.skipped_reason}) -- validating "
            f"word-by-word against the file's own original timing directly.")

    starts, ends, validated = validate_words_against_asr(words, gap, tolerance_sec=validate_tolerance_sec)
    quality.n_validated = sum(validated)
    source = ["validated" if v else "" for v in validated]

    lrc_seed = None
    if use_lrc and audio_duration is not None:
        our_lines = _reconstruct_our_lines(entries)
        prep = prepare_lrc(words, asr_words, artist or existing.artist, title or existing.title,
                            audio_duration, forced_candidate=forced_lrc_candidate,
                            our_lines=our_lines, our_line_of_word=our_line_of_word, log=log)
        if prep is not None and prep.calibration_offset is not None:
            lrc_seed = seed_from_prep(words, prep, starts, ends, validated)
            quality.n_lrc_seeded = lrc_seed.n_seeded
            log(f"  LRC anchors: {lrc_seed.lrc_match.candidate.track_name!r}/"
                f"{lrc_seed.lrc_match.candidate.artist_name!r} -- seeded {lrc_seed.n_seeded} "
                f"additional line-start anchor(s) for words that didn't validate.")
        elif prep is not None:
            log("  LRC candidate found but not confidently calibrated -- not used as an anchor source.")
        else:
            log("  No usable LRCLIB synced-lyrics candidate for this song.")
        for i in range(len(words)):
            if validated[i] and not source[i]:
                source[i] = "lrc"

    had_any_anchor = any(validated)
    quality.n_interpolated, quality.n_kept_original = interpolate_fallback(words, starts, ends, validated)
    for i in range(len(words)):
        if not source[i]:
            source[i] = "interpolated" if had_any_anchor else "kept_original"

    log(f"  {quality.n_validated}/{quality.n_words} word(s) validated (ASR confirmed the original position "
        f"within tolerance -- using ASR's own, more precise start on the word's own original length), "
        f"{quality.n_lrc_seeded} from LRC anchors, {quality.n_interpolated} repositioned by "
        f"interpolation, {quality.n_kept_original} kept at their original timing (no anchor nearby).")
    if quality.anchor_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:
        log(f"  WARNING: only {quality.anchor_rate:.0%} of words validated or got an LRC anchor -- "
            f"this file's lyrics may not match this audio at all. Review the output carefully.")
        log("  Falling back to 'seed' mode (whole-song ASR primary, replacing every word's timing) -- "
            "'validate' trusts the file's own original timing, which this low an anchor rate means "
            "isn't safe to trust here.")
        return realign_song(
            existing, asr_words, artist=artist, title=title, audio_duration=audio_duration,
            use_lrc=use_lrc, lrc_mode="seed", forced_lrc_candidate=forced_lrc_candidate, log=log,
            debug_log=debug_log,
        )

    if debug_log is not None:
        debug_log.section("REALIGN SUMMARY (strategy=validate)")
        debug_log.line(f"  GAP: offset={gap.offset} slope={gap.slope} confidence={gap.confidence} "
                        f"kind={gap.kind} skipped_reason={gap.skipped_reason}")
        debug_log.line(f"  {quality.n_validated}/{quality.n_words} validated (kept as-is), "
                        f"{quality.n_lrc_seeded} from LRC anchors, {quality.n_interpolated} interpolated, "
                        f"{quality.n_kept_original} kept original.")
        debug_log.section("PER-WORD TRACE (text, orig_start->orig_end, source, final_start->final_end)")
        for i, w in enumerate(words):
            debug_log.line(f"  {i:4d} {w.text:20s} orig={w.orig_start:9.3f}-{w.orig_end:9.3f}  "
                            f"[{source[i]:14s}]  final={starts[i]:9.3f}-{ends[i]:9.3f}  "
                            f"delta={starts[i] - w.orig_start:+8.3f}")
        debug_log.section(f"RAW ASR TRANSCRIPT ({len(asr_words)} word(s), real timestamps + confidence -- "
                           f"WhisperX/faster-whisper is NOT deterministic run-to-run, so this is the only record "
                           f"of what THIS specific run's ASR actually heard)")
        for aw in asr_words:
            debug_log.line(f"  {aw.start:9.3f}-{aw.end:9.3f}  conf={aw.confidence:.3f}  {aw.text!r}")

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


def find_existing_txt_in_folder(folder: Path, *, exclude_markers: Tuple[str, ...] = ("[REALIGNED]",)) -> Path:
    """Auto-detects the single existing UltraStar .txt to realign within
    a folder -- used when no explicit --existing-txt is given (required
    for batch mode, where a single explicit path can't apply across
    multiple subfolders; optional convenience in single-song mode too).

    Excludes this module's OWN "[REALIGNED]" output naming convention --
    otherwise re-running on a folder that already has a previous run's
    output would either see two candidates (falsely "ambiguous") or,
    worse, pick the REALIGNED file itself as this run's new INPUT,
    compounding drift across repeated runs instead of always realigning
    the same original file. `exclude_markers` is overridable so a sibling
    module with its own output naming convention (e.g. `pitch_refresh.py`'s
    "[PITCH_REFRESHED]") can reuse this same auto-detection logic and
    exclude its OWN prior output too, without either module needing to
    know about the other's marker by default.

    When more than one real candidate remains, tries ONE further
    disambiguation before giving up: a file named exactly "<folder
    name>.txt" (case-insensitive) -- the common convention this project's
    own output/companion files already follow elsewhere (e.g. `#COVER`/
    `#BACKGROUND` matching by basename in file_discovery.py). Only trusted
    when it narrows the field to EXACTLY one match; still fails closed
    (never guesses) otherwise."""
    folder = Path(folder)
    candidates = sorted(
        p for p in folder.glob("*.txt") if not any(marker in p.stem for marker in exclude_markers)
    )
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
    p.add_argument("--lrc-file", dest="lrc_file", default=None,
                    help="Path to a LOCAL .lrc synced-lyrics file to use directly, bypassing LRCLIB "
                         "search/fetch entirely -- same override tier as --lrclib-id (wins over it if "
                         "both are given). For a user who already has a synced-lyrics file on disk, "
                         "including a deliberately imprecise one for testing how the LRC-seeding/"
                         "windowing logic handles an off-but-close candidate.")
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
    p.add_argument("--delete-work-files", action="store_true",
                    help="Delete the large, fully-regeneratable work files under "
                         "<input-folder>/.ultrastar_work after realigning. Default: OFF (keeps them so "
                         "re-runs reuse the cached separation) -- same convention as the main pipeline's "
                         "own --delete-work-files.")
    p.add_argument("--no-debug-log", action="store_true",
                    help="Skip writing '<Artist> - <Title> [DEBUG LOG].txt' (per-word match/interpolation "
                         "trace) to the work dir. Default: ON, same convention as the main pipeline's own "
                         "debug log.")
    p.add_argument("--retry-low-quality-asr", dest="retry_low_quality_asr", action="store_true",
                    default=config.RETRY_LOW_QUALITY_ASR,
                    # NOTE: literal '%' in argparse help text must be escaped as '%%' -- argparse
                    # interpolates the stored help string with %-formatting when rendering --help.
                    help=(f"PROTOTYPE, default: ON. Retries with '{config.RETRY_ASR_MODEL}' if either: fewer "
                          f"than {config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:.0%} of words get a real anchor from "
                          f"ASR/LRC (whole-song), or {config.RETRY_ASR_MIN_UNCONFIDENT_RUN}+ consecutive words "
                          f"have no real anchor at all (one bad passage, even if the rest of the song is fine). "
                          f"No-op if --whisper-model is already '{config.RETRY_ASR_MODEL}'; only kept if it "
                          f"actually scores better. The actual re-transcription only runs in --batch mode "
                          f"(roughly doubles this run's ASR time) -- outside --batch, a triggered condition "
                          f"logs a WARNING suggesting --batch or a manual --whisper-model large-v3 instead. "
                          f"Only applies to --strategy replace. See CLAUDE.md."
                          ).replace("%", "%%"))
    p.add_argument("--no-retry-low-quality-asr", dest="retry_low_quality_asr", action="store_false")
    p.add_argument("--strategy", choices=["replace", "validate"], default="validate",
                    help="(default: validate). 'validate': first checks whether the whole file is explained "
                         "by a single global GAP/drift correction, then a word whose (GAP-corrected) original "
                         "position is independently CONFIRMED by ASR is left completely untouched -- position "
                         "and length both -- instead of being overwritten; only words that don't validate are "
                         "repositioned. Automatically falls back to whole-song ASR-primary matching (the same "
                         "mechanism 'replace' uses) when too few words validate (see "
                         "MXL_LRC_MIN_ASR_PLACEMENT_RATE), since that means the file's own timing can't be "
                         "trusted enough for 'validate' to be safe. 'replace': a word confidently matched to "
                         "ASR has its timing REPLACED with ASR's own value directly. See CLAUDE.md.")
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
    lrc_file: Optional[str] = None
    use_lrc: bool = True
    lrc_mode: str = "windowed"
    output_path: Optional[str] = None
    delete_work_files: bool = False
    no_debug_log: bool = False
    retry_low_quality_asr: bool = config.RETRY_LOW_QUALITY_ASR
    # Whether this run is part of a --batch invocation -- see
    # _retry_asr_if_low_quality's own docstring for why the large-v3
    # retry itself is gated on this (set by run() from args.batch, and by
    # gui.py's _build_realign_opts from its own _is_batch()).
    batch: bool = False
    # "validate" (DEFAULT as of 2026-08-15, see CLAUDE.md): realign_song_
    # validate -- a word whose original position (after a single global
    # GAP/drift correction) is independently confirmed by ASR is left
    # completely untouched instead of being overwritten; automatically
    # falls back to whole-song ASR-primary matching (the same mechanism
    # "replace" uses) when too few words validate, so it's safe even on
    # a file whose own timing turns out not to be trustworthy at all.
    # "replace": realign_song, a matched word's timing is REPLACED with
    # ASR's own value directly -- selectable via CLI --strategy or the
    # GUI's Strategy dropdown.
    strategy: str = "validate"
    # GUI-only, never set by the CLI -- see config.PipelineOptions.
    # cancel_requested's own docstring for the full explanation (same
    # stage-boundary-only convention, shared config.check_cancelled/
    # PipelineCancelled).
    cancel_requested: Optional[Callable[[], bool]] = None


@dataclass
class RealignPipelineResult:
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None


def _transcribe_words_cancellable(vocals_path: Path, model_name: str, opts: RealignPipelineOptions,
                                   debug_log: Optional[DebugLog], log: Callable[[str], None]) -> List[Word]:
    """Same contract as `transcription.transcribe_words`, routed through a
    killable child process (worker_process.run_cancellable) whenever
    `opts.cancel_requested` is set -- see that module's own docstring.
    Falls straight through to the in-process call otherwise (the CLI)."""
    whisperx_vad_options = config.WHISPERX_NO_VAD_OPTIONS if opts.whisperx_no_vad else None
    if opts.cancel_requested is None:
        from .transcription import transcribe_words
        return transcribe_words(
            vocals_path, model_name, prefer_whisperx=not opts.no_whisperx,
            whisperx_vad_options=whisperx_vad_options,
            debug_log=debug_log,
        )
    result = run_cancellable(
        "transcribe_words",
        {"vocals_path": str(vocals_path), "model_name": model_name,
         "prefer_whisperx": not opts.no_whisperx, "whisperx_vad_options": whisperx_vad_options},
        cancel_requested=opts.cancel_requested, debug_log=debug_log, log=log,
    )
    from .models import Word as _Word
    return [_Word(**d) for d in result]


def _retry_asr_if_low_quality(result: RealignResult, *, existing: ParsedSong, vocals_path: Path,
                               opts: RealignPipelineOptions, audio_duration: float,
                               forced_candidate: Optional[LrcLibCandidate],
                               debug_log: Optional[DebugLog],
                               log: Callable[[str], None]) -> RealignResult:
    """PROTOTYPE, see config.RETRY_LOW_QUALITY_ASR's docstring. Only wired
    up for `realign_song` (strategy="replace") -- `realign_song_validate`
    uses a differently-shaped `ValidateQuality` with no `anchor_rate`, not
    addressed here.

    Two INDEPENDENT triggers, either is enough:
      - `RealignQuality.anchor_rate` below `config.MXL_LRC_MIN_ASR_PLACEMENT_RATE`
        (the SAME metric/bar `realign_song` already uses for its own "this
        file's lyrics may not match this audio at all" warning) -- a
        whole-song signal.
      - `RealignQuality.longest_unconfident_run` at or above
        `config.RETRY_ASR_MIN_UNCONFIDENT_RUN` -- a per-PASSAGE signal
        (real case: David Bowie - Magic Dance with `small.en`, song-wide
        anchor rate 58% -- well above the bar -- while one hallucinated
        decoder segment still dragged a real, contiguous passage ~12-14s
        off in the final output).

    Either way, re-transcribes the WHOLE song with `config.RETRY_ASR_MODEL`
    and re-runs `realign_song` against the new transcription, keeping
    whichever of the two results is better on WHICHEVER signal actually
    triggered the retry (comparing only anchor_rate could miss a retry
    that fixes one bad passage -- a handful of words out of the whole
    song barely moves the aggregate rate -- without making it worse
    either). Never retries a second time, and never fires at all if
    `opts.whisper_model` is already the retry model.

    `opts.batch` gates whether a triggered retry actually RUNS (2026-08-
    10, user's explicit request): a whole-song re-transcription roughly
    doubles this run's ASR time, which is a reasonable cost to pay
    automatically in an unattended `--batch` run, but a surprising one to
    silently eat in a single-song run where the user is presumably
    watching and would rather decide for themselves. Outside `--batch`,
    a triggered condition logs a WARNING (same explanation, both console
    and debug log) suggesting `--batch` or a manual `--whisper-model
    large-v3` instead of retrying automatically."""
    # Mirrors every top-level decision-narration message into the debug log FILE too, not just the
    # console/GUI `log` callback -- otherwise the file only shows the retry's raw ASR/realign data (via
    # `debug_log` threaded into the calls below) with no record of WHY it fired or what was decided,
    # forcing anyone auditing the file alone to go dig up separately-captured console output instead.
    def dlog(msg: str) -> None:
        log(msg)
        if debug_log is not None:
            debug_log.line(msg)

    if not (opts.retry_low_quality_asr and result.success and result.quality is not None):
        return result
    low_rate_trigger = result.quality.anchor_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE
    long_run_trigger = result.quality.longest_unconfident_run >= config.RETRY_ASR_MIN_UNCONFIDENT_RUN
    if not (low_rate_trigger or long_run_trigger):
        return result
    if opts.whisper_model == config.RETRY_ASR_MODEL:
        return result

    if long_run_trigger and not low_rate_trigger:
        reason = (f"anchor rate is fine ({result.quality.anchor_rate:.0%}) but "
                  f"{result.quality.longest_unconfident_run} consecutive word(s) have no real anchor at all "
                  f"(a passage the whole-song rate doesn't catch)")
    else:
        reason = f"only {result.quality.anchor_rate:.0%} of words got a real anchor from the audio"

    if not opts.batch:
        dlog(f"  WARNING: {reason} -- retrying with '{config.RETRY_ASR_MODEL}' would likely help, but "
             f"automatic retries only run in --batch mode (a whole-song re-transcription roughly doubles "
             f"this run's ASR time, which --batch runs unattended but a single run doesn't spend without "
             f"asking). Re-run with --batch, or pass --whisper-model {config.RETRY_ASR_MODEL} directly, to "
             f"get this fixed.")
        return result

    config.check_cancelled(opts.cancel_requested)
    if debug_log is not None:
        debug_log.section(f"ASR QUALITY RETRY -- re-transcribing with '{config.RETRY_ASR_MODEL}'")
    dlog(f"  {reason[0].upper()}{reason[1:]} -- retrying transcription with '{config.RETRY_ASR_MODEL}' "
         f"before accepting this result...")
    retry_asr_words = _transcribe_words_cancellable(vocals_path, config.RETRY_ASR_MODEL, opts, debug_log, log)
    if not retry_asr_words:
        dlog(f"  Retry transcription with '{config.RETRY_ASR_MODEL}' produced no words -- keeping the "
             f"original result.")
        return result

    retry_result = realign_song(
        existing, retry_asr_words, artist=opts.artist, title=opts.title, audio_duration=audio_duration,
        use_lrc=opts.use_lrc, lrc_mode=opts.lrc_mode, forced_lrc_candidate=forced_candidate,
        vocals_path=vocals_path, log=log,
        debug_log=debug_log, cancel_requested=opts.cancel_requested,
    )
    if retry_result.success and retry_result.quality is not None:
        improved = (retry_result.quality.anchor_rate > result.quality.anchor_rate
                    or retry_result.quality.longest_unconfident_run < result.quality.longest_unconfident_run)
        if improved:
            dlog(f"  Retry with '{config.RETRY_ASR_MODEL}' improved: anchor rate {result.quality.anchor_rate:.0%} "
                 f"-> {retry_result.quality.anchor_rate:.0%}, longest unconfident run "
                 f"{result.quality.longest_unconfident_run} -> {retry_result.quality.longest_unconfident_run} "
                 f"word(s) -- using the retry.")
            return retry_result

    got = (f"anchor rate {retry_result.quality.anchor_rate:.0%}, longest unconfident run "
           f"{retry_result.quality.longest_unconfident_run} word(s)"
           if retry_result.success and retry_result.quality is not None else "failed")
    dlog(f"  Retry with '{config.RETRY_ASR_MODEL}' did not improve on the original ({got}) -- keeping the "
         f"original result.")
    return result


def run_realign_pipeline(input_dir: Path, existing_txt_path: Optional[Path], opts: RealignPipelineOptions,
                          *, log: Callable[[str], None] = print) -> RealignPipelineResult:
    """Thin wrapper around `_run_realign_pipeline_body` so `opts.
    delete_work_files` is honored via `finally`, regardless of which of
    the body's several early-return failure paths was taken -- work_dir
    may be partially populated (e.g. separation already ran) even on a
    failed run. Mirrors `main.run_pipeline`'s own wrapper exactly (see
    its docstring); work_dir's own location is recomputed here rather
    than threaded back out of the body, since it's a pure function of
    (input_dir, opts.work_dir) that can never diverge from what the body
    itself used."""
    try:
        return _run_realign_pipeline_body(input_dir, existing_txt_path, opts, log=log)
    except config.PipelineCancelled:
        log("Cancelled by user.")
        return RealignPipelineResult(success=False, error="Cancelled by user.")
    except WorkerError as e:
        return RealignPipelineResult(success=False, error=str(e))
    finally:
        if opts.delete_work_files:
            from .main import delete_work_files as _delete_work_files
            wd = Path(opts.work_dir).resolve() if opts.work_dir else (Path(input_dir) / ".ultrastar_work")
            _delete_work_files(wd)


def _run_realign_pipeline_body(input_dir: Path, existing_txt_path: Optional[Path], opts: RealignPipelineOptions,
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

    artist = opts.artist or existing.artist
    title = opts.title or existing.title
    debug_log_path = (None if opts.no_debug_log else
                       (work_dir / sanitize_filename(f"{artist} - {title} [DEBUG LOG].txt")))
    debug_log = DebugLog(debug_log_path)
    if debug_log_path is not None:
        log(f"Writing debug log to: {debug_log_path}")

    try:
        resolved = resolve_song_folder(input_dir, work_dir, audio_file_override=opts.audio_file)
    except (AmbiguousInputError, NoAudioSourceFoundError) as e:
        return RealignPipelineResult(success=False, error=str(e))

    config.check_cancelled(opts.cancel_requested)
    if opts.skip_separation:
        if not opts.vocals_path:
            return RealignPipelineResult(success=False, error="skip_separation requires vocals_path")
        vocals_path = Path(opts.vocals_path).resolve()
    else:
        log("Isolating vocals with Demucs...")
        try:
            vocals_path = isolate_vocals(resolved.analysis_audio, work_dir, model=opts.demucs_model,
                                          cancel_requested=opts.cancel_requested)
        except SeparationError as e:
            return RealignPipelineResult(success=False, error=f"Vocal isolation failed: {e}")

    import librosa
    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)
    audio_duration = len(y) / sr

    config.check_cancelled(opts.cancel_requested)
    log(f"Transcribing with {'whisperx' if not opts.no_whisperx else 'faster-whisper'} "
        f"({opts.whisper_model})...")
    asr_words = _transcribe_words_cancellable(vocals_path, opts.whisper_model, opts, debug_log, log)
    if not asr_words:
        return RealignPipelineResult(
            success=False, error="No words were transcribed -- check the audio / vocal isolation quality.")
    log(f"Transcribed {len(asr_words)} words.")

    forced_candidate = None
    if opts.lrc_file is not None:
        forced_candidate = load_lrc_file(opts.lrc_file, artist=opts.artist or "", title=opts.title or "")
        if forced_candidate is None:
            log(f"Could not read/parse --lrc-file {opts.lrc_file!r} -- ignoring.")
    elif opts.lrclib_id is not None:
        forced_candidate = fetch_lrclib_by_id(opts.lrclib_id)
        if forced_candidate is None:
            log(f"Could not fetch LRCLIB id {opts.lrclib_id} -- ignoring.")

    log("Realigning...")
    if opts.strategy == "validate":
        result = realign_song_validate(
            existing, asr_words, artist=opts.artist, title=opts.title, audio_duration=audio_duration,
            use_lrc=opts.use_lrc, forced_lrc_candidate=forced_candidate, log=log,
            debug_log=debug_log,
        )
    else:
        result = realign_song(
            existing, asr_words, artist=opts.artist, title=opts.title, audio_duration=audio_duration,
            use_lrc=opts.use_lrc, lrc_mode=opts.lrc_mode, forced_lrc_candidate=forced_candidate,
            vocals_path=vocals_path, log=log,
            debug_log=debug_log, cancel_requested=opts.cancel_requested,
        )
        result = _retry_asr_if_low_quality(
            result, existing=existing, vocals_path=vocals_path, opts=opts, audio_duration=audio_duration,
            forced_candidate=forced_candidate, debug_log=debug_log, log=log,
        )
    if not result.success:
        return RealignPipelineResult(success=False, error=result.error)

    output_path = resolve_realign_output_path(existing_txt_path, opts.output_path)
    guard_error = check_output_not_existing_file(output_path, existing_txt_path)
    if guard_error is not None:
        return RealignPipelineResult(success=False, error=guard_error)
    write_song(result.song, output_path)
    log(f"Wrote {output_path}")
    debug_log.close()
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
        if opts.cancel_requested is not None and opts.cancel_requested():
            log(f"Cancelled by user before {sub.name} ({i}/{len(subdirs)}).")
            break
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
        artist=args.artist, title=args.title, lrclib_id=args.lrclib_id, lrc_file=args.lrc_file,
        use_lrc=args.use_lrc, lrc_mode=args.lrc_mode, output_path=args.output_path,
        delete_work_files=args.delete_work_files, no_debug_log=args.no_debug_log,
        retry_low_quality_asr=args.retry_low_quality_asr,
        batch=args.batch, strategy=args.strategy,
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
        if args.lrc_file:
            incompatible.append("--lrc-file")
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
