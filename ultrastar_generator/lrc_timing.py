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
of PITCH CLASS: LRC line timestamps and our own per-song timing can be
offset by a roughly constant amount (e.g. a different silence-trim/lead-in
in whichever recording LRCLIB's synced version was made from), so a
per-song calibration offset is established first (mode of per-line
deltas, not mean/median -- see compare_full_pipeline_output.py's own
"Lessons learned" this session on why a straight median isn't robust
against a song with many closely-matching false candidates), then only
lines that still disagree with the calibrated expectation by more than
a tolerance get flagged.
"""

from __future__ import annotations

import re
import difflib
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import config
from .models import Syllable

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


def apply_lrc_timing_check(
    syllables: List[Syllable],
    synced_lyrics: str,
    min_calibration_samples: int = config.LRC_TIMING_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.LRC_TIMING_MIN_CALIBRATION_CONFIDENCE,
    flag_tolerance_sec: float = config.LRC_TIMING_FLAG_TOLERANCE_SEC,
    verbose: bool = True,
    debug_log=None,
) -> LRCTimingStats:
    """Aligns pass-3's own lines (grouped by Syllable.line_id) against
    LRCLIB's synced-lyrics lines by TEXT (whole-sequence difflib, same
    technique used throughout this project), calibrates a per-song time
    offset once enough lines agree closely on one (mode of per-line
    deltas, see module docstring), then flags any line whose delta from
    that calibrated offset exceeds flag_tolerance_sec.

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

    our_tokens = [" ".join(tokens) for _, _, tokens in our_lines]
    lrc_tokens = [" ".join(_normalize(w) for w in text.split() if _normalize(w)) for _, text in lrc_lines]

    sm = difflib.SequenceMatcher(None, our_tokens, lrc_tokens, autojunk=False)
    candidates = []  # (our_line_idx, lrc_line_idx, delta_sec)
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(a1 - a0):
            our_idx, lrc_idx = a0 + k, b0 + k
            _, our_start, _ = our_lines[our_idx]
            lrc_start, _ = lrc_lines[lrc_idx]
            candidates.append((our_idx, lrc_idx, our_start - lrc_start))
    stats.n_matched_lines = len(candidates)

    if verbose:
        print(f"[lrc-timing] {len(lrc_lines)} synced lines, {len(our_lines)} of our own lines, "
              f"{len(candidates)} matched by text")

    if len(candidates) < min_calibration_samples:
        stats.skipped_reason = (
            f"only {len(candidates)} matched line(s) (< {min_calibration_samples} required) -- "
            f"not enough to trust a calibration offset"
        )
        if verbose:
            print(f"[lrc-timing] skipping: {stats.skipped_reason}")
        return stats

    # Mode at coarse (1s) resolution -- line-level timestamps are far
    # less precise than word-level, so a fine bucket would just split a
    # real single cluster across several adjacent buckets.
    BUCKET_SEC = 1.0
    bucket_counts = Counter(round(d / BUCKET_SEC) for _, _, d in candidates)
    best_bucket, n_agree = bucket_counts.most_common(1)[0]
    confidence = n_agree / len(candidates)
    offset = best_bucket * BUCKET_SEC

    stats.calibration_offset_sec = offset
    stats.calibration_confidence = confidence

    if confidence < min_calibration_confidence:
        stats.skipped_reason = (
            f"no clear per-song time calibration (best candidate {offset:+.1f}s covers "
            f"{confidence:.0%} of {len(candidates)} matched lines -- below the required bar)"
        )
        if verbose:
            print(f"[lrc-timing] skipping: {stats.skipped_reason}")
        return stats

    if verbose:
        print(f"[lrc-timing] calibration offset: {offset:+.1f}s, {confidence:.0%} agreement "
              f"over {len(candidates)} matched line(s)")

    for our_idx, lrc_idx, delta in candidates:
        expected = lrc_lines[lrc_idx][0] + offset
        residual = delta - offset
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
