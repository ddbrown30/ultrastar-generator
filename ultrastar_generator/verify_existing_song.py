"""Verifies an EXISTING UltraStar .txt file's pitch/timing against a
FRESH pipeline run of the same song, before deciding whether to overwrite
it (feature 6). Shaped like musicxml_reference.py's calibrate-then-compare
pattern (same kind of external-reference-vs-our-own-output comparison),
but for TIME as well as pitch, and calibration-free (this compares two
ABSOLUTE timelines of the SAME audio, not two independently-timed
recordings the way lrc_timing.py's LRCLIB comparison does -- no offset to
find).

Reuses techniques this session already validated on real ground-truth
data (see CLAUDE.md's 0i/0o for the numbers) rather than reinventing them:
  - Word-level whole-sequence text alignment (same technique as
    lyrics_lookup.py/musicxml_reference.py), not naive index pairing.
  - Pitch compared at PITCH CLASS (mod 12), matching how UltraStar Deluxe
    itself scores and how pass 4 already treats reference data -- never
    exact-MIDI/octave.
  - Timing compared with the repeat-instance bucketing guard confirmed
    necessary in CLAUDE.md's 0i (a repeated chorus/line can otherwise get
    matched against the wrong sung instance, corrupting the comparison):
    bucket candidate deltas at 200ms resolution, keep only the dominant
    cluster before scoring.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import config
from .models import Syllable
from .usdx_parser import ParsedSong


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9']", "", s.lower())


@dataclass
class ExistingSongVerification:
    n_matched: int = 0
    pitch_class_accuracy: float = 0.0
    timing_within_tolerance_pct: float = 0.0
    verdict: str = "COULD_NOT_VERIFY"   # "PASS" | "PROBLEMS_FOUND" | "COULD_NOT_VERIFY"
    reason: Optional[str] = None
    pitch_mismatches: List[Tuple[str, int, int]] = field(default_factory=list)       # text, existing_pc, fresh_pc
    timing_mismatches: List[Tuple[str, float, float]] = field(default_factory=list)  # text, existing_start, fresh_start


def _word_start_words(syllables: List[Syllable]) -> List[Tuple[str, float, int]]:
    """(normalized_text, start_sec, midi_note) for each WORD-START
    syllable -- word-level granularity, matching the technique this
    project's other cross-checks (lrc_timing, the ground-truth
    comparisons in CLAUDE.md) already use."""
    out = []
    for s in syllables:
        if not s.is_word_start:
            continue
        n = _normalize(s.text)
        if n:
            out.append((n, s.start, s.midi_note))
    return out


def verify_existing_song(
    existing: ParsedSong,
    fresh_syllables: List[Syllable],
    *,
    min_matched: int = config.EXISTING_TXT_MIN_MATCHED,
    min_pitch_accuracy: float = config.EXISTING_TXT_MIN_PITCH_ACCURACY,
    timing_tolerance_sec: float = config.EXISTING_TXT_TIMING_TOLERANCE_SEC,
    min_timing_agreement: float = config.EXISTING_TXT_MIN_TIMING_AGREEMENT,
    verbose: bool = True,
    debug_log=None,
) -> ExistingSongVerification:
    """Compares `existing` (parsed from an on-disk .txt, see usdx_parser.py)
    against `fresh_syllables` (this run's own freshly-computed pass-3/4
    output, BEFORE phrasing.build_lines -- comparison only needs the
    syllable sequence). Returns stats only -- never modifies either input.
    `verdict` is "COULD_NOT_VERIFY" (never "PASS") whenever too few words
    matched to trust the comparison at all -- an inconclusive check must
    never read as confirmation.
    """
    existing_words = _word_start_words([
        s for s in existing.entries if isinstance(s, Syllable)
    ])
    fresh_words = _word_start_words(fresh_syllables)

    a = [w for w, _, _ in existing_words]
    b = [w for w, _, _ in fresh_words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    candidates = []  # (text, existing_start, fresh_start, existing_pc, fresh_pc)
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(a1 - a0):
            ei, fi = a0 + k, b0 + k
            text, e_start, e_pitch = existing_words[ei]
            _, f_start, f_pitch = fresh_words[fi]
            candidates.append((text, e_start, f_start, e_pitch % 12, f_pitch % 12))

    stats = ExistingSongVerification(n_matched=len(candidates))

    if len(candidates) < min_matched:
        stats.reason = (f"only {len(candidates)} word(s) matched by text (< {min_matched} required) -- "
                         f"not enough to trust a comparison")
        stats.verdict = "COULD_NOT_VERIFY"
        if verbose:
            print(f"[existing-txt] {stats.reason}")
        return stats

    # Repeat-instance guard (see module docstring): bucket timing deltas
    # at 200ms resolution, keep only candidates near the dominant cluster.
    # Unlike lrc_timing.py's cross-recording comparison, these two
    # timelines describe the SAME audio, so the true cluster should sit
    # very close to 0 -- but a repeated line/chorus can still pair a
    # word against the wrong occurrence of itself, so the guard still
    # matters here, just with the dominant offset expected near zero
    # rather than an arbitrary per-song value.
    BUCKET_SEC = 0.2
    deltas = [e_start - f_start for _, e_start, f_start, _, _ in candidates]
    bucket_counts = Counter(round(d / BUCKET_SEC) for d in deltas)
    best_bucket, _ = bucket_counts.most_common(1)[0]
    center = best_bucket * BUCKET_SEC
    guarded = [c for c, d in zip(candidates, deltas) if abs(d - center) <= 3.0]

    n_pitch_ok = sum(1 for _, _, _, e_pc, f_pc in guarded if e_pc == f_pc)
    n_timing_ok = sum(1 for _, e_start, f_start, _, _ in guarded if abs(e_start - f_start) <= timing_tolerance_sec)

    stats.pitch_class_accuracy = n_pitch_ok / len(guarded)
    stats.timing_within_tolerance_pct = n_timing_ok / len(guarded)
    stats.pitch_mismatches = [(text, e_pc, f_pc) for text, _, _, e_pc, f_pc in guarded if e_pc != f_pc]
    stats.timing_mismatches = [(text, e_start, f_start) for text, e_start, f_start, _, _ in guarded
                                if abs(e_start - f_start) > timing_tolerance_sec]

    if stats.pitch_class_accuracy < min_pitch_accuracy or stats.timing_within_tolerance_pct < min_timing_agreement:
        stats.verdict = "PROBLEMS_FOUND"
        stats.reason = (f"pitch-class accuracy {stats.pitch_class_accuracy:.0%} "
                         f"(need {min_pitch_accuracy:.0%}), timing agreement "
                         f"{stats.timing_within_tolerance_pct:.0%} (need {min_timing_agreement:.0%})")
    else:
        stats.verdict = "PASS"

    if verbose:
        print(f"[existing-txt] {len(candidates)} word(s) matched by text, {len(guarded)} after repeat-instance guard: "
              f"pitch-class accuracy {stats.pitch_class_accuracy:.0%}, "
              f"timing agreement {stats.timing_within_tolerance_pct:.0%} -> {stats.verdict}")
        if stats.verdict != "PASS":
            for text, e_pc, f_pc in stats.pitch_mismatches[:10]:
                print(f"    pitch mismatch: {text!r} existing_pc={e_pc} fresh_pc={f_pc}")
            for text, e_start, f_start in stats.timing_mismatches[:10]:
                print(f"    timing mismatch: {text!r} existing={e_start:.2f}s fresh={f_start:.2f}s")

    if debug_log is not None:
        debug_log.line(f"[existing-txt] verdict={stats.verdict} "
                        f"pitch_class_accuracy={stats.pitch_class_accuracy:.2f} "
                        f"timing_agreement={stats.timing_within_tolerance_pct:.2f} "
                        f"n_matched={len(candidates)}")

    return stats
