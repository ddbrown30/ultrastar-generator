"""Verifies an EXISTING UltraStar .txt file's pitch/timing against a FRESH
pipeline run of the same song. Word-level sequence alignment; pitch compared
at PITCH CLASS (mod 12); nothing excluded from scoring denominators."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import config
from .models import Syllable
from .usdx_parser import ParsedSong
from .text_normalize import normalize_word as _normalize


@dataclass
class ExistingSongVerification:
    n_matched: int = 0
    pitch_class_accuracy: float = 0.0
    timing_within_tolerance_pct: float = 0.0
    verdict: str = "COULD_NOT_VERIFY"   # "PASS" | "PROBLEMS_FOUND" | "COULD_NOT_VERIFY"
    reason: Optional[str] = None
    pitch_mismatches: List[Tuple[str, int, int]] = field(default_factory=list)       # text, existing_pc, fresh_pc
    timing_mismatches: List[Tuple[str, float, float]] = field(default_factory=list)  # text, existing_start, fresh_start
    coverage_fresh: float = 0.0     # n_matched / len(fresh words)
    coverage_existing: float = 0.0  # n_matched / len(existing words)
    unmatched_fresh: List[str] = field(default_factory=list)     # fresh words that matched nothing in existing
    unmatched_existing: List[str] = field(default_factory=list)  # existing words that matched nothing in fresh
    recall: float = 0.0     # n_correct / len(existing words)
    precision: float = 0.0  # n_correct / len(fresh words)
    f1: float = 0.0


def _word_start_words(syllables: List[Syllable]) -> List[Tuple[str, float, int]]:
    """(normalized_text, start_sec, midi_note) for each word-start syllable."""
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
    min_coverage: float = config.EXISTING_TXT_MIN_COVERAGE,
    verbose: bool = True,
    debug_log=None,
) -> ExistingSongVerification:
    """Compares `existing` (parsed .txt) against `fresh_syllables` (this run's
    fresh output). Returns stats only, never modifies either input. Too few
    matched words yields verdict "COULD_NOT_VERIFY", never "PASS"."""
    existing_words = _word_start_words([
        s for s in existing.entries if isinstance(s, Syllable)
    ])
    fresh_words = _word_start_words(fresh_syllables)

    a = [w for w, _, _ in existing_words]
    b = [w for w, _, _ in fresh_words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    candidates = []  # (text, existing_start, fresh_start, existing_pc, fresh_pc)
    matched_existing_idxs = set()
    matched_fresh_idxs = set()
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(a1 - a0):
            ei, fi = a0 + k, b0 + k
            text, e_start, e_pitch = existing_words[ei]
            _, f_start, f_pitch = fresh_words[fi]
            candidates.append((text, e_start, f_start, e_pitch % 12, f_pitch % 12))
            matched_existing_idxs.add(ei)
            matched_fresh_idxs.add(fi)

    stats = ExistingSongVerification(n_matched=len(candidates))
    stats.coverage_existing = len(candidates) / len(existing_words) if existing_words else 0.0
    stats.coverage_fresh = len(candidates) / len(fresh_words) if fresh_words else 0.0
    stats.unmatched_existing = [w for i, (w, _, _) in enumerate(existing_words) if i not in matched_existing_idxs]
    stats.unmatched_fresh = [w for i, (w, _, _) in enumerate(fresh_words) if i not in matched_fresh_idxs]

    if len(candidates) < min_matched:
        stats.reason = (f"only {len(candidates)} word(s) matched by text (< {min_matched} required) -- "
                         f"not enough to trust a comparison")
        stats.verdict = "COULD_NOT_VERIFY"
        if verbose:
            print(f"[existing-txt] {stats.reason}")
        return stats

    n_timing_ok = sum(1 for _, e_start, f_start, _, _ in candidates if abs(e_start - f_start) <= timing_tolerance_sec)
    stats.timing_within_tolerance_pct = n_timing_ok / len(candidates)
    stats.timing_mismatches = [(text, e_start, f_start) for text, e_start, f_start, _, _ in candidates
                                if abs(e_start - f_start) > timing_tolerance_sec]

    # Pitch scored only among pairs that also have correct timing.
    correct = [c for c in candidates if abs(c[1] - c[2]) <= timing_tolerance_sec]
    n_pitch_ok = sum(1 for _, _, _, e_pc, f_pc in correct if e_pc == f_pc)
    stats.pitch_class_accuracy = n_pitch_ok / len(correct) if correct else 0.0
    stats.pitch_mismatches = [(text, e_pc, f_pc) for text, _, _, e_pc, f_pc in correct if e_pc != f_pc]

    n_correct = len(correct)
    stats.recall = n_correct / len(existing_words) if existing_words else 0.0
    stats.precision = n_correct / len(fresh_words) if fresh_words else 0.0
    stats.f1 = (2 * stats.precision * stats.recall / (stats.precision + stats.recall)
                if (stats.precision + stats.recall) else 0.0)

    problems = []
    if stats.pitch_class_accuracy < min_pitch_accuracy:
        problems.append(f"pitch-class accuracy {stats.pitch_class_accuracy:.0%} (need {min_pitch_accuracy:.0%})")
    if stats.timing_within_tolerance_pct < min_timing_agreement:
        problems.append(f"timing agreement {stats.timing_within_tolerance_pct:.0%} "
                         f"(need {min_timing_agreement:.0%})")
    if stats.coverage_fresh < min_coverage:
        problems.append(f"fresh coverage {stats.coverage_fresh:.0%} (need {min_coverage:.0%}) -- "
                         f"{len(stats.unmatched_fresh)} fresh word(s) never matched the existing file at all")
    if stats.coverage_existing < min_coverage:
        problems.append(f"existing coverage {stats.coverage_existing:.0%} (need {min_coverage:.0%}) -- "
                         f"{len(stats.unmatched_existing)} existing word(s) never matched the fresh output at all")

    if problems:
        stats.verdict = "PROBLEMS_FOUND"
        stats.reason = "; ".join(problems)
    else:
        stats.verdict = "PASS"

    if verbose:
        print(f"[existing-txt] {len(candidates)} word(s) matched by text: "
              f"pitch-class accuracy {stats.pitch_class_accuracy:.0%}, "
              f"timing agreement {stats.timing_within_tolerance_pct:.0%}, "
              f"coverage fresh={stats.coverage_fresh:.0%}/existing={stats.coverage_existing:.0%} -> {stats.verdict}")
        print(f"    real match (text+timing correct, nothing excluded from either denominator): "
              f"recall={stats.recall:.0%} ({n_correct}/{len(existing_words)} existing/GT word(s) reproduced), "
              f"precision={stats.precision:.0%} ({n_correct}/{len(fresh_words)} fresh word(s) are real matches), "
              f"F1={stats.f1:.0%} -- a 100% match requires BOTH at 100% (no missing words AND no extras)")
        if stats.verdict != "PASS":
            if stats.unmatched_fresh:
                shown = stats.unmatched_fresh[:15]
                print(f"    unmatched fresh word(s): {', '.join(shown)}"
                      + (" ..." if len(stats.unmatched_fresh) > 15 else ""))
            if stats.unmatched_existing:
                shown = stats.unmatched_existing[:15]
                print(f"    unmatched existing word(s): {', '.join(shown)}"
                      + (" ..." if len(stats.unmatched_existing) > 15 else ""))
            for text, e_pc, f_pc in stats.pitch_mismatches[:10]:
                print(f"    pitch mismatch: {text!r} existing_pc={e_pc} fresh_pc={f_pc}")
            for text, e_start, f_start in stats.timing_mismatches[:10]:
                print(f"    timing mismatch: {text!r} existing={e_start:.2f}s fresh={f_start:.2f}s")

    if debug_log is not None:
        debug_log.line(f"[existing-txt] verdict={stats.verdict} "
                        f"pitch_class_accuracy={stats.pitch_class_accuracy:.2f} "
                        f"timing_agreement={stats.timing_within_tolerance_pct:.2f} "
                        f"coverage_fresh={stats.coverage_fresh:.2f} "
                        f"coverage_existing={stats.coverage_existing:.2f} "
                        f"recall={stats.recall:.2f} precision={stats.precision:.2f} f1={stats.f1:.2f} "
                        f"n_matched={len(candidates)}")

    return stats
