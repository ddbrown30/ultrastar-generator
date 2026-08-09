"""Cross-checks pass-3 LINE placement against LRCLIB's synced (per-line
timestamped) lyrics, when available (see lyrics_lookup.LyricsResult.
synced_lyrics).

DIAGNOSTIC ONLY as of its first version -- flags lines whose assigned
start time disagrees with a confidently-calibrated LRC-derived estimate,
but never moves anything. Deliberately NOT auto-correcting yet: this
project's own `verify_placement` was built with the same good intentions
(catch a class of real note/word-placement error) and, when validated for
real end-to-end this session, produced a NET REGRESSION on every pitch
and timing metric on both songs it was tested against, despite correctly
fixing some individual real problems -- see CLAUDE.md. Shipping an
auto-correction here without first confirming the underlying signal is
trustworthy would risk repeating that mistake. The intended path: run
this as a flag-only diagnostic, cross-reference its flagged lines against
real ground-truth timing error (now measurable -- see
compare_full_pipeline_output.py's compare_timing()) to confirm the signal
actually correlates with real problems, and only THEN consider building
an actual correction step.

Calibration mirrors musicxml_reference.py's approach but for TIME instead
of PITCH CLASS: LRC line timestamps and our own per-song timing are
first assumed to differ by a roughly constant amount (e.g. a different
silence-trim/lead-in in whichever recording LRCLIB's synced version was
made from), calibrated as the mode of per-line deltas at 1-second
resolution (not mean/median -- see compare_full_pipeline_output.py's own
"Lessons learned" this session on why a straight median isn't robust
against a song with many closely-matching false candidates). Real audio
testing found this constant-offset assumption doesn't always hold,
though (see CLAUDE.md's 0k-0m) -- several songs showed a real, roughly
LINEAR drift instead (the offset grows smoothly over the song, up to
~9%/s of elapsed time on some songs, most likely because whichever
recording LRCLIB's synced lyrics were timed against isn't quite the same
edit/tempo as ours). When the constant-offset check fails, a second,
stricter attempt fits offset+slope with a robust (Theil-Sen) estimator,
tolerant of the wrong-repeated-line-instance and different-arrangement
outliers real songs showed (see `_robust_linear_fit`). Only lines that
still disagree with whichever calibration won get flagged.
"""

from __future__ import annotations

import re
import difflib
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple

from . import config
from .models import Syllable, Word

_LRC_TAG_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9']", "", s.lower())


def parse_lrc(synced_lyrics: str) -> List[Tuple[float, str]]:
    """Parses LRC-format synced lyrics ("[mm:ss.xx]line text" per line)
    into a time-ordered list of (start_sec, raw_text) tuples. Lines with
    no timestamp tag or no text after it are skipped."""
    entries = []
    for line in synced_lyrics.splitlines():
        m = _LRC_TAG_RE.match(line)
        if not m:
            continue
        minutes, seconds = m.groups()
        t = int(minutes) * 60 + float(seconds)
        text = line[m.end():].strip()
        if text:
            entries.append((t, text))
    entries.sort(key=lambda e: e[0])
    return entries


@dataclass
class LineTimingFlag:
    """One line whose assigned start time disagrees with LRC's
    calibrated expectation by more than the tolerance."""
    syllable_index: int   # index into the syllables list of the line's first word
    text: str
    assigned_start: float
    lrc_expected_start: float   # LRC's own timestamp, with calibration offset applied
    delta_sec: float            # assigned_start - lrc_expected_start


@dataclass
class LRCTimingStats:
    n_lrc_lines: int = 0
    n_our_lines: int = 0
    n_matched_lines: int = 0
    calibration_offset_sec: Optional[float] = None
    calibration_slope: float = 0.0   # 0.0 for a constant-offset calibration
    calibration_kind: Optional[str] = None   # "constant" or "drift"
    calibration_confidence: float = 0.0
    flags: List[LineTimingFlag] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


def _reconstruct_lines(syllables: List[Syllable]) -> List[Tuple[int, float, List[str]]]:
    """Groups syllables into lines by Syllable.line_id (set by
    lyrics_lookup.align_words_to_reference during reference-lyric
    alignment -- the exact same grouping phrasing.py uses to place '-'
    breaks). Returns (first_syllable_index, line_start_sec,
    normalized_word_tokens) per line, in line order. Syllables with no
    line_id (lyrics lookup unavailable/didn't cover them) are excluded --
    there's no line to anchor them to."""
    lines: dict = {}  # line_id -> [first_syl_idx, start_sec, [tokens]]
    order: List[int] = []
    for i, s in enumerate(syllables):
        if s.line_id is None or not s.is_word_start:
            continue
        n = _normalize(s.text)
        if not n:
            continue
        if s.line_id not in lines:
            lines[s.line_id] = [i, s.start, []]
            order.append(s.line_id)
        lines[s.line_id][2].append(n)
    return [tuple(lines[lid]) for lid in order]


def _match_lines_word_level(
    our_lines: List[Tuple[int, float, List[str]]],
    lrc_lines: List[Tuple[float, str]],
) -> List[Tuple[int, float, float]]:
    """Matches our_lines to lrc_lines at the WORD level via one whole-
    sequence alignment (order-preserving -- still resistant to pairing a
    repeated line against the wrong occurrence, unlike an independent
    per-line nearest-neighbor search), then re-derives a per-LINE
    correspondence by majority vote of each our-line's own matched words.

    Recovers correspondences a whole-line exact match misses whenever a
    line differs by only a word or two -- confirmed the common case on
    real audio: ASR words that survived reference-lyric correction only
    partially (e.g. "the hu world is a mess" vs LRC's "the human world
    is a mess"). Requiring only a MAJORITY (not all) of a line's own
    words to agree tolerates this without requiring exact text equality.

    Returns (our_line_idx, lrc_start_sec, delta_sec) triples, at most one
    per our-line (its single best-voted LRC line).
    """
    our_words: List[Tuple[int, str]] = []
    for i, (_, _, tokens) in enumerate(our_lines):
        our_words.extend((i, t) for t in tokens)
    lrc_words: List[Tuple[int, str]] = []
    for j, (_, text) in enumerate(lrc_lines):
        for w in text.split():
            n = _normalize(w)
            if n:
                lrc_words.append((j, n))

    a = [w for _, w in our_words]
    b = [w for _, w in lrc_words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pair_votes: Dict[Tuple[int, int], int] = {}
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(a1 - a0):
            key = (our_words[a0 + k][0], lrc_words[b0 + k][0])
            pair_votes[key] = pair_votes.get(key, 0) + 1

    our_wordcount = Counter(i for i, _ in our_words)
    best_for_our: Dict[int, Tuple[int, int]] = {}
    for (oi, lj), v in pair_votes.items():
        if oi not in best_for_our or v > best_for_our[oi][1]:
            best_for_our[oi] = (lj, v)

    candidates = []
    for oi, (lj, v) in best_for_our.items():
        if v >= max(1, (our_wordcount[oi] + 1) // 2):
            our_start = our_lines[oi][1]
            lrc_start = lrc_lines[lj][0]
            candidates.append((oi, lrc_start, our_start - lrc_start))
    candidates.sort(key=lambda c: c[0])
    return candidates


def _robust_linear_fit(
    candidates: List[Tuple[int, float, float]],
    inlier_tolerance_sec: float,
) -> Optional[Tuple[float, float, float, int]]:
    """Theil-Sen (median-of-pairwise-slopes) robust fit of
    delta = offset + slope*lrc_start. Robust to outliers by construction
    -- a single wrong-repeated-line-instance match, or a whole cluster of
    lines from a differently-arranged passage, can't drag the median slope
    far the way it would an ordinary least-squares fit.

    Returns (offset, slope, confidence, n_inliers), or None if there are
    fewer than 2 distinct lrc_start values to compute a slope from.
    confidence is the fraction of candidates within inlier_tolerance_sec
    of the fitted line.
    """
    pts = [(lrc_start, delta) for _, lrc_start, delta in candidates]
    slopes = []
    for i in range(len(pts)):
        ti, di = pts[i]
        for j in range(i + 1, len(pts)):
            tj, dj = pts[j]
            if abs(tj - ti) < 1e-6:
                continue
            slopes.append((dj - di) / (tj - ti))
    if not slopes:
        return None
    slope = median(slopes)
    intercept = median(d - slope * t for t, d in pts)
    residuals = [d - (intercept + slope * t) for t, d in pts]
    n_inliers = sum(1 for r in residuals if abs(r) <= inlier_tolerance_sec)
    confidence = n_inliers / len(pts)
    return intercept, slope, confidence, n_inliers


def match_asr_to_lrc_lines(asr_words: List[Word], lrc_lines: List[Tuple[float, str]]
                            ) -> List[Tuple[int, float, float]]:
    """Matches ASR's own flat, time-ordered word stream against the LRC
    lines' text (one whole-sequence, order-preserving alignment -- same
    technique used throughout this project) to find, per LRC line, the
    EARLIEST real ASR word confidently belonging to it. Returns
    (lrc_line_index, lrc_start, delta) candidates, delta = that word's
    real ASR start time minus the LRC line's own declared start --
    exactly the shape `two_tier_time_calibration` (below) expects.

    This gives a real-time anchor per LRC line straight from OUR OWN
    audio's transcription, independent of any other reference data --
    used to calibrate away a systematic offset (e.g. extra lead-in
    silence in our recording vs. whichever recording LRCLIB's synced
    lyrics were timed against) BEFORE those timestamps are trusted as
    placement anchors, rather than only diagnosing the mismatch after the
    fact the way `apply_lrc_timing_check` does. Originally built for
    `mxl_lrc_generator.py`; factored out here (its data shape never
    depended on MXL at all) once `realign.py` needed the exact same
    ASR-vs-LRC-line calibration step -- don't reimplement this a third
    time."""
    lrc_flat: List[Tuple[int, str]] = []
    for li, (_, text) in enumerate(lrc_lines):
        for tok in text.split():
            n = _normalize(tok)
            if n:
                lrc_flat.append((li, n))
    lrc_norm = [n for _, n in lrc_flat]
    asr_norm = [_normalize(w.text) for w in asr_words]
    sm = difflib.SequenceMatcher(None, asr_norm, lrc_norm, autojunk=False)
    first_asr_for_line: dict = {}
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(a2 - a1):
            li = lrc_flat[b1 + k][0]
            if li not in first_asr_for_line:
                first_asr_for_line[li] = a1 + k

    candidates = []
    for li, asr_idx in first_asr_for_line.items():
        lrc_start = lrc_lines[li][0]
        candidates.append((li, lrc_start, asr_words[asr_idx].start - lrc_start))
    candidates.sort(key=lambda c: c[0])
    return candidates


def two_tier_time_calibration(
    candidates: List[Tuple[int, float, float]],
    min_calibration_samples: int = config.LRC_TIMING_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.LRC_TIMING_MIN_CALIBRATION_CONFIDENCE,
    min_drift_samples: int = config.LRC_TIMING_MIN_DRIFT_SAMPLES,
    min_drift_confidence: float = config.LRC_TIMING_MIN_DRIFT_CONFIDENCE,
    drift_inlier_tolerance_sec: float = config.LRC_TIMING_DRIFT_INLIER_TOLERANCE_SEC,
) -> Tuple[Optional[float], float, float, Optional[str], Optional[str]]:
    """Shared two-tier time calibration: given (key, lrc_start, delta)
    candidates, first tries a single constant offset (mode of deltas at
    1s resolution); if that isn't confident enough, falls back to a
    robust (Theil-Sen) offset+slope fit tolerant of a real per-song
    timing drift (see module docstring for why both tiers exist).

    Returns (offset, slope, confidence, kind, skipped_reason) -- offset is
    None (with skipped_reason set, kind None) if no confident calibration
    could be established. Factored out of `apply_lrc_timing_check` so
    `mxl_lrc_generator.py` can reuse the exact same technique to calibrate
    away a systematic offset between LRC line timestamps and OUR OWN
    audio (e.g. different lead-in silence) BEFORE trusting those
    timestamps as placement anchors, rather than only diagnosing the
    mismatch after the fact -- don't reimplement this a third time."""
    if len(candidates) < min_calibration_samples:
        return None, 0.0, 0.0, None, (
            f"only {len(candidates)} matched line(s) (< {min_calibration_samples} required) -- "
            f"not enough to trust a calibration offset"
        )

    # Tier 1: constant offset -- mode at coarse (1s) resolution, since
    # line-level timestamps are far less precise than word-level, so a
    # fine bucket would just split a real single cluster across several
    # adjacent buckets.
    BUCKET_SEC = 1.0
    bucket_counts = Counter(round(delta / BUCKET_SEC) for _, _, delta in candidates)
    best_bucket, n_agree = bucket_counts.most_common(1)[0]
    offset, slope, confidence, kind = best_bucket * BUCKET_SEC, 0.0, n_agree / len(candidates), "constant"

    if confidence < min_calibration_confidence:
        # Tier 2: a real per-song drift, not just noise around one
        # constant -- confirmed on real audio (stars, tarzan,
        # little_mermaid), see module docstring. Stricter gate than tier
        # 1: a 2-parameter fit can trivially match a handful of points
        # exactly, so this needs both more samples and a higher inlier
        # fraction before it's trusted.
        fit = _robust_linear_fit(candidates, drift_inlier_tolerance_sec) if len(candidates) >= min_drift_samples else None
        if fit is None or fit[2] < min_drift_confidence:
            return None, 0.0, 0.0, None, (
                f"no clear per-song time calibration -- constant-offset best candidate {offset:+.1f}s "
                f"covers {confidence:.0%} of {len(candidates)} matched lines (need {min_calibration_confidence:.0%}), "
                f"and drift fit " + (
                    f"only reached {fit[2]:.0%} inliers (need {min_drift_confidence:.0%})" if fit
                    else f"needs >= {min_drift_samples} matched lines"
                )
            )
        offset, slope, confidence, _ = fit
        kind = "drift"

    return offset, slope, confidence, kind, None


def apply_lrc_timing_check(
    syllables: List[Syllable],
    synced_lyrics: str,
    min_calibration_samples: int = config.LRC_TIMING_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.LRC_TIMING_MIN_CALIBRATION_CONFIDENCE,
    min_drift_samples: int = config.LRC_TIMING_MIN_DRIFT_SAMPLES,
    min_drift_confidence: float = config.LRC_TIMING_MIN_DRIFT_CONFIDENCE,
    drift_inlier_tolerance_sec: float = config.LRC_TIMING_DRIFT_INLIER_TOLERANCE_SEC,
    flag_tolerance_sec: float = config.LRC_TIMING_FLAG_TOLERANCE_SEC,
    verbose: bool = True,
    debug_log=None,
) -> LRCTimingStats:
    """Aligns pass-3's own lines (grouped by Syllable.line_id) against
    LRCLIB's synced-lyrics lines by TEXT (word-level whole-sequence
    match, see `_match_lines_word_level`), calibrates a per-song time
    offset, then flags any line whose delta from the calibrated
    expectation exceeds flag_tolerance_sec.

    Calibration is two-tiered (see module docstring): first tries a
    single constant offset (mode of per-line deltas at 1s resolution --
    the original, higher-precision technique); if that isn't confident
    enough, falls back to a robust offset+slope fit tolerant of a real
    per-song timing drift. Whichever tier succeeds is what lines get
    flagged against; if neither does, this returns with skipped_reason
    set and no flags.

    Returns stats only -- `syllables` is never modified. See module
    docstring for why this doesn't auto-correct yet.
    """
    stats = LRCTimingStats()

    lrc_lines = parse_lrc(synced_lyrics)
    stats.n_lrc_lines = len(lrc_lines)
    if not lrc_lines:
        stats.skipped_reason = "no synced-lyrics lines to check against"
        return stats

    our_lines = _reconstruct_lines(syllables)
    stats.n_our_lines = len(our_lines)
    if not our_lines:
        stats.skipped_reason = "no line_id-tagged syllables to check (reference lyrics unavailable/didn't cover this song)"
        return stats

    candidates = _match_lines_word_level(our_lines, lrc_lines)
    stats.n_matched_lines = len(candidates)

    if verbose:
        print(f"[lrc-timing] {len(lrc_lines)} synced lines, {len(our_lines)} of our own lines, "
              f"{len(candidates)} matched by text")

    offset, slope, confidence, kind, skipped_reason = two_tier_time_calibration(
        candidates, min_calibration_samples, min_calibration_confidence,
        min_drift_samples, min_drift_confidence, drift_inlier_tolerance_sec,
    )
    if offset is None:
        stats.skipped_reason = skipped_reason
        if verbose:
            print(f"[lrc-timing] skipping: {stats.skipped_reason}")
        return stats

    stats.calibration_offset_sec = offset
    stats.calibration_slope = slope
    stats.calibration_kind = kind
    stats.calibration_confidence = confidence

    if verbose:
        drift_desc = f", drift {slope:+.4f}s per LRC-second" if kind == "drift" else ""
        print(f"[lrc-timing] calibration ({kind}): offset {offset:+.1f}s{drift_desc}, "
              f"{confidence:.0%} agreement over {len(candidates)} matched line(s)")

    for our_idx, lrc_start, delta in candidates:
        expected = lrc_start + offset + slope * lrc_start
        residual = delta - (offset + slope * lrc_start)
        if abs(residual) <= flag_tolerance_sec:
            continue
        first_syl_idx, our_start, _ = our_lines[our_idx]
        flag = LineTimingFlag(
            syllable_index=first_syl_idx, text=syllables[first_syl_idx].text,
            assigned_start=our_start, lrc_expected_start=expected, delta_sec=residual,
        )
        stats.flags.append(flag)
        if verbose:
            print(f"    [lrc-timing] {syllables[first_syl_idx].text!r} line assigned to start at "
                  f"{our_start:.2f}s, but LRC (calibrated) expects ~{expected:.2f}s "
                  f"(off by {residual:+.2f}s) -- flagged, not corrected")
        if debug_log is not None:
            debug_log.line(f"[lrc-timing] {syllables[first_syl_idx].text!r} @ {our_start:.2f}s: "
                            f"LRC expects ~{expected:.2f}s (off by {residual:+.2f}s)")

    if verbose:
        print(f"[lrc-timing] {len(stats.flags)}/{len(candidates)} matched line(s) flagged as "
              f"disagreeing with LRC timing by more than {flag_tolerance_sec:.1f}s")

    return stats
