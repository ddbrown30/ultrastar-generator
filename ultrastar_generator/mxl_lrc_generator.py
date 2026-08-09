"""Primary generation path: MusicXML for pitch, LRCLIB synced-lyrics LINE
starts as real-world-time anchors, real transcription (ASR) of our own
audio to place words WITHIN each line, falling back to proportional
placement (using the MXL's own relative offsets) only where ASR doesn't
confidently match a word.

Real, ground-truth-validated origin (2026-08-08/09 session): three
progressively better designs were tried and measured against real
SingStar-style ground truth (Chicago - "When You're Good to Mama"):
  1. A single global linear fit (MXL offset -> real seconds, calibrated
     against the LRC candidate's own timestamps): 39.5% of words landed
     within 500ms of ground truth. Root cause found: the MXL score has
     real, human-marked tempo-region changes ("Lower Tempo" / "Rubato con
     moto" / "Moderato, in 2") that a single constant tempo assumption
     can't capture -- confirmed directly from the MXL's own raw
     `<direction><words>` text, not guessed.
  2. Per-LRC-line proportional placement (each line's own MXL words
     distributed proportionally between that line's LRC start and the
     next line's LRC start): 56.0% within 500ms. Better -- LRC line
     starts ARE reliable anchors -- but individual word-level pacing
     WITHIN a line still doesn't track a real singer's local
     push-and-pull against the MXL's own fixed relative note durations.
  3. **This module's design**: trust LRC line starts as hard anchors,
     but place words WITHIN a line using REAL transcription of our own
     audio (order-preserving match against ASR words whose own timestamp
     falls inside that line's real-time window), falling back to
     proportional-by-MXL-offset only for words ASR doesn't confidently
     catch: 99.0% within 500ms, mean error 92ms -- on par with or better
     than this project's own best real full-pipeline numbers on other
     songs, achieved with ZERO audio-only pitch detection (no CREPE/pYIN
     pass 1 at all).

The same session also found two real candidate-selection failures (BATB,
Les Miserables - Stars): both had a matching-duration, matching-content
LRC candidate that was nonetheless timed to a DIFFERENT recording/
performance than the user's own audio (confirmed independently for
both). Neither is fixable by tightening the upfront duration/content
filter -- the wrong candidates passed those checks cleanly. The real,
reliable signal is downstream: a wrong-recording candidate's LRC line
timestamps don't actually correspond to what our own audio says at those
moments, so the ASR-vs-MXL word-level match rate inside `place_words_via_asr`
collapses. `generate_from_mxl_and_lrc`'s quality gate uses exactly this
signal (`MxlLrcQuality.asr_placement_rate`) rather than trying to perfect
candidate selection -- see CLAUDE.md for the real validation of this
claim on BATB/Stars.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from . import config
from .lyrics_lookup import LrcLibCandidate, search_lrclib
from .lrc_timing import parse_lrc
from .models import Syllable, Word


def _normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", s)


@dataclass
class MxlWord:
    text: str
    norm: str
    offset: float  # quarter-note offset of this word's first syllable
    syllables: List[Tuple[float, float, int, str]]  # (offset, quarterLength, midi, syllable_text)


def load_mxl_vocal_words(mxl_path: str, preferred_part_name: Optional[str] = None) -> Tuple[List[MxlWord], List[str]]:
    """Parses a MusicXML/.mxl file into whole WORDS (syllables merged via
    each note's own `syllabic` marker -- begin/middle/end/single), unlike
    `musicxml_reference.load_vocal_notes` which deliberately stays at the
    single-note/single-syllable-fragment level (right for pitch-class
    correction, wrong for word-level ASR/LRC-line matching).

    Part selection: `preferred_part_name` if it names a real lyric-bearing
    part; otherwise the single lyric-bearing part with the most
    lyric-bearing notes. Deliberately does NOT reproduce
    `load_vocal_notes`' multi-part MERGE (filling gaps across several
    lyric-bearing parts) -- none of this feature's validated songs needed
    it, and merging while preserving per-note syllabic markers is real
    added complexity; a multi-voice arrangement falls back to whichever
    single part has the most lyrics, same as this file's own
    `_scan_lyrics_mxl_candidates`-style prototype used throughout
    validation.
    """
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

    for n in chosen_part.flatten().notes:
        if n.isChord or not n.lyrics:
            continue
        for ly in n.lyrics:
            if not ly.text:
                continue
            syl = ly.syllabic
            if syl in (None, "single", "begin"):
                flush()
                cur_text = ly.text
                cur_offset = float(n.offset)
            else:
                cur_text += ly.text
            cur_syllables.append((float(n.offset), float(n.quarterLength), int(n.pitch.midi), ly.text))
            break  # one lyric verse only
    flush()

    return words, [chosen_part.partName]


@dataclass
class LrcMatch:
    candidate: LrcLibCandidate
    lrc_lines: List[Tuple[float, str]]
    content_match_ratio: float
    duration_delta: Optional[float]


def select_lrc_candidate(artist: str, title: str, mxl_words: List[MxlWord], audio_duration: float,
                          forced: Optional[LrcLibCandidate] = None) -> Optional[LrcMatch]:
    """Picks an LRC candidate to use for timing. If `forced` is given (a
    user-pinned or --lrclib-id-resolved candidate), it's used directly,
    no filtering -- the user already vetted it. Otherwise searches LRCLIB
    (both artist/title and free-text `q`, deduped -- the free-text search
    was found necessary this session: LRCLIB's artist/title search alone
    can miss a candidate its own free-text search finds), requires
    `synced_lyrics`, requires duration within
    `config.MXL_LRC_DURATION_TOLERANCE_SEC`, and picks the best
    content-match (difflib ratio of MXL words vs the candidate's plain
    lyrics) among those clearing `config.MXL_LRC_MIN_CONTENT_MATCH_RATIO`.
    This bar is intentionally permissive -- see this module's docstring
    for why the real validity gate is downstream, not here."""
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
        delta = abs(forced.duration - audio_duration) if forced.duration is not None else None
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
        delta = abs(c.duration - audio_duration)
        if delta > config.MXL_LRC_DURATION_TOLERANCE_SEC:
            continue
        lrc_norm = [_normalize(t) for t in (c.plain_lyrics or "").split()]
        lrc_norm = [w for w in lrc_norm if w]
        if not lrc_norm:
            continue
        ratio = difflib.SequenceMatcher(None, mxl_norm_words, lrc_norm, autojunk=False).ratio()
        if ratio < config.MXL_LRC_MIN_CONTENT_MATCH_RATIO:
            continue
        scored.append((ratio, delta, c))

    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    ratio, delta, best = scored[0]
    lrc_lines = parse_lrc(best.synced_lyrics)
    if not lrc_lines:
        return None
    return LrcMatch(candidate=best, lrc_lines=lrc_lines, content_match_ratio=ratio, duration_delta=delta)


def assign_words_to_lines(mxl_words: List[MxlWord], lrc_lines: List[Tuple[float, str]]) -> List[int]:
    """Assigns each MXL word to an LRC line index via word-level
    whole-sequence matching (order-preserving, resistant to picking a
    wrong repeated-phrase instance the same way this project's other
    whole-sequence alignments are). Words that don't directly match any
    LRC token (OCR-garbled MXL text, minor wording differences) inherit
    the nearest PRECEDING confirmed match's line -- falling back to the
    first confirmed line for any words before the first match."""
    lrc_flat: List[str] = []
    lrc_line_idx: List[int] = []
    for li, (_, text) in enumerate(lrc_lines):
        for tok in text.split():
            n = _normalize(tok)
            if n:
                lrc_flat.append(n)
                lrc_line_idx.append(li)

    mxl_norm = [w.norm for w in mxl_words]
    sm = difflib.SequenceMatcher(None, mxl_norm, lrc_flat, autojunk=False)
    word_line = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            word_line[i1 + k] = lrc_line_idx[j1 + k]

    n = len(mxl_words)
    filled: List[Optional[int]] = [None] * n
    last = None
    for i in range(n):
        if i in word_line:
            filled[i] = word_line[i]
            last = word_line[i]
        else:
            filled[i] = last
    first_known = next((v for v in filled if v is not None), None)
    return [v if v is not None else first_known for v in filled]


@dataclass
class MxlLrcQuality:
    n_words: int = 0
    n_asr_placed: int = 0
    n_fallback: int = 0
    non_monotonic_fix_count: int = 0

    @property
    def asr_placement_rate(self) -> float:
        return self.n_asr_placed / self.n_words if self.n_words else 0.0


def _line_window(lrc_lines: List[Tuple[float, str]], li: int) -> Tuple[float, float]:
    t0 = lrc_lines[li][0]
    t1 = lrc_lines[li + 1][0] if li + 1 < len(lrc_lines) else t0 + 5.0
    return t0, t1


def place_words_via_asr(mxl_words: List[MxlWord], word_lines: List[int], lrc_lines: List[Tuple[float, str]],
                         asr_words: List[Word]) -> Tuple[List[float], List[float], MxlLrcQuality]:
    """For each LRC line, matches that line's own MXL words against real
    ASR words whose own timestamp falls near the line's real-time window
    (order-preserving difflib, same technique used throughout this
    project for text alignment).

    A matched word is only trusted if the ASR match ALSO clears
    `config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE` -- confirmed real case: a
    text match with confidence 0.003 had a genuinely wrong (0.77s off)
    timestamp, independent of anything else in this pipeline. A
    low-confidence "match" is treated as no match at all.

    START: a trusted match uses the ASR word's own start; an untrusted/
    unmatched word falls back to proportional placement using the MXL's
    own relative offsets within the line's window (unchanged from before).

    END (real duration): a trusted match uses the ASR word's own reported
    end directly. An untrusted/unmatched word's end is ESTIMATED from its
    own MXL note value (how many quarter notes it spans) times a locally-
    calibrated real-seconds-per-quarter-note rate, derived from this
    line's own (t0, t1) window and MXL offset span -- i.e. "how long
    nearby notes are actually taking, applied to this word's own notated
    length" rather than blindly stretching to the next word's start
    (confirmed real bug: that produced e.g. a single word held for 3.1s
    or 7.1s, swallowing what should have been a real pause).

    Non-decreasing order on START is then enforced (clamp) -- ASR can
    occasionally produce a slightly out-of-order local match (e.g. a
    repeated/garbled word within one line). ENDs are then clamped to
    never exceed the NEXT word's own start (no overlap) but are free to
    end EARLIER, leaving a real rest -- this is the actual fix for the
    swallowed-pause bug."""
    line_word_idxs: dict = {}
    for i, li in enumerate(word_lines):
        line_word_idxs.setdefault(li, []).append(i)

    n = len(mxl_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    quality = MxlLrcQuality(n_words=n)

    for li, idxs in line_word_idxs.items():
        idxs = sorted(idxs)
        t0, t1 = _line_window(lrc_lines, li)
        asr_in_window = [w for w in asr_words if t0 - 0.5 <= w.start <= t1 + 0.5]
        asr_norm = [_normalize(w.text) for w in asr_in_window]
        mxl_norm_line = [mxl_words[i].norm for i in idxs]
        sm = difflib.SequenceMatcher(None, mxl_norm_line, asr_norm, autojunk=False)
        matched_local = {}
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if tag != "equal":
                continue
            for k in range(a2 - a1):
                asr_w = asr_in_window[b1 + k]
                if asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                    matched_local[a1 + k] = asr_w
                # else: leave unmatched -- falls through to the estimated
                # fallback placement/duration below, same as a real miss.

        offs = [mxl_words[i].offset for i in idxs]
        lo, hi = min(offs), max(offs)
        span = hi - lo
        # Real-seconds-per-quarter-note for this line, used to estimate
        # duration for any word that doesn't have a trusted ASR (start, end).
        line_rate = (t1 - t0) / span if span > 0 else config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC
        for local_i, global_i in enumerate(idxs):
            w = mxl_words[global_i]
            word_qtr_dur = sum(s[1] for s in w.syllables)
            if local_i in matched_local:
                asr_w = matched_local[local_i]
                starts[global_i] = asr_w.start
                asr_dur = asr_w.end - asr_w.start
                ends[global_i] = asr_w.start + asr_dur if asr_dur > 0 else asr_w.start + word_qtr_dur * line_rate
                quality.n_asr_placed += 1
            else:
                off = w.offset
                frac = (off - lo) / span if span > 0 else 0.0
                start = t0 + frac * (t1 - t0)
                starts[global_i] = start
                ends[global_i] = start + word_qtr_dur * line_rate
                quality.n_fallback += 1

    for i in range(1, n):
        if starts[i] < starts[i - 1]:
            starts[i] = starts[i - 1]
            quality.non_monotonic_fix_count += 1

    # ENDs must never overlap the next word's own (already-finalized) start
    # -- but are otherwise free to be shorter, leaving a real rest between
    # words rather than always filling the whole gap.
    for i in range(n):
        if i + 1 < n and ends[i] > starts[i + 1]:
            ends[i] = starts[i + 1]
        if ends[i] < starts[i]:
            ends[i] = starts[i]

    return starts, ends, quality


def build_syllables(mxl_words: List[MxlWord], word_starts: List[float], word_ends: List[float],
                     word_lines: List[int]) -> List[Syllable]:
    """Splits each word's own syllables proportionally within
    [word_start, word_end) (see `place_words_via_asr` for how those are
    derived from ASR and/or MXL note values -- NOT simply "until the next
    word starts", which used to swallow real pauses between words) using
    the MXL's own relative sub-word offsets -- that part of the MXL data
    (syllable-to-syllable ratios within one word) is reliable, so there's
    no need to guess those from ASR too. `line_id` is set from
    `assign_words_to_lines` so `phrasing.build_lines` gets accurate,
    LRC-native line breaks."""
    syllables: List[Syllable] = []
    for i, w in enumerate(mxl_words):
        t0 = word_starts[i]
        t1 = word_ends[i]
        if t1 <= t0:
            # Zero-width word (e.g. its own estimated duration rounded to
            # nothing, or it was clamped flush against the next word with
            # no room at all) -- usdx_writer.py already has a well-tested
            # minimum-display-length mechanism for exactly this case; don't
            # guess a local padding value here.
            t1 = t0
        lo = w.offset
        hi = w.offset + sum(s[1] for s in w.syllables)
        for syl_i, (off, dur, midi, text) in enumerate(w.syllables):
            frac0 = (off - lo) / (hi - lo) if hi > lo else 0.0
            frac1 = (off + dur - lo) / (hi - lo) if hi > lo else 1.0
            syllables.append(Syllable(
                text=text, start=t0 + frac0 * (t1 - t0), end=t0 + frac1 * (t1 - t0),
                midi_note=midi - 60, is_word_start=(syl_i == 0), line_id=word_lines[i],
            ))
    return syllables


@dataclass
class MxlLrcResult:
    success: bool
    reason: str
    syllables: List[Syllable] = field(default_factory=list)
    quality: Optional[MxlLrcQuality] = None
    lrc_match: Optional[LrcMatch] = None
    mxl_path: Optional[str] = None
    part_names_used: List[str] = field(default_factory=list)


def generate_from_mxl_and_lrc(mxl_path: str, artist: str, title: str, audio_duration: float,
                               asr_words: List[Word], forced_candidate: Optional[LrcLibCandidate] = None,
                               preferred_part_name: Optional[str] = None) -> MxlLrcResult:
    """Orchestrates the full MXL+LRC generation for one MusicXML file and
    applies the quality gate. Never raises on expected failure modes (no
    lyric-bearing part, no candidate found) -- returns a failed
    `MxlLrcResult` with a human-readable `reason` instead, for the caller
    to log/prompt with."""
    mxl_words, part_names = load_mxl_vocal_words(mxl_path, preferred_part_name)
    if not mxl_words:
        return MxlLrcResult(success=False, reason=f"{mxl_path}: no lyric-bearing part found", mxl_path=mxl_path)

    lrc_match = select_lrc_candidate(artist, title, mxl_words, audio_duration, forced=forced_candidate)
    if lrc_match is None:
        return MxlLrcResult(success=False, reason="no matching synced lyrics found on LRCLIB",
                             mxl_path=mxl_path, part_names_used=part_names)

    word_lines = assign_words_to_lines(mxl_words, lrc_match.lrc_lines)
    word_starts, word_ends, quality = place_words_via_asr(mxl_words, word_lines, lrc_match.lrc_lines, asr_words)
    syllables = build_syllables(mxl_words, word_starts, word_ends, word_lines)

    nonmonotonic_rate = quality.non_monotonic_fix_count / quality.n_words if quality.n_words else 1.0
    if quality.asr_placement_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:
        return MxlLrcResult(
            success=False,
            reason=(f"ASR/MXL word match rate too low ({quality.asr_placement_rate:.0%}, need "
                     f"{config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:.0%}) -- the matched lyrics likely don't "
                     f"correspond to this recording"),
            syllables=syllables, quality=quality, lrc_match=lrc_match,
            mxl_path=mxl_path, part_names_used=part_names,
        )
    if nonmonotonic_rate > config.MXL_LRC_MAX_NONMONOTONIC_RATE:
        return MxlLrcResult(
            success=False,
            reason=f"too many out-of-order word placements ({nonmonotonic_rate:.0%})",
            syllables=syllables, quality=quality, lrc_match=lrc_match,
            mxl_path=mxl_path, part_names_used=part_names,
        )

    return MxlLrcResult(success=True, reason="", syllables=syllables, quality=quality,
                         lrc_match=lrc_match, mxl_path=mxl_path, part_names_used=part_names)


def try_mxl_lrc_primary(mxl_paths: List[str], artist: str, title: str, audio_duration: float,
                         asr_words: List[Word], forced_candidate: Optional[LrcLibCandidate] = None,
                         preferred_part_name: Optional[str] = None) -> Optional[MxlLrcResult]:
    """Tries each MXL path in order (mirrors `apply_musicxml_references`'
    multi-file convention), returning the first one that clears the
    quality gate. If every path was attempted but none succeeded, returns
    the LAST attempted (failed) result so the caller has a concrete
    reason to log/prompt with, rather than a bare None. Returns None only
    if `mxl_paths` is empty."""
    last_result = None
    for mxl_path in mxl_paths:
        result = generate_from_mxl_and_lrc(
            mxl_path, artist, title, audio_duration, asr_words,
            forced_candidate=forced_candidate, preferred_part_name=preferred_part_name,
        )
        if result.success:
            return result
        last_result = result
    return last_result
