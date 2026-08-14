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
  - Nothing is excluded from scoring: every text-matched pair counts
    toward pitch_class_accuracy/timing_within_tolerance_pct's
    denominators, and recall/precision/f1 are computed bidirectionally
    over the FULL existing/fresh word counts, same design as
    scratchpad/compare_video_games.py. An EARLIER version of this module
    bucketed candidate timing deltas at 200ms resolution and silently
    DROPPED any candidate more than 3.0s from the dominant cluster before
    scoring at all (meant as a repeat-instance guard, since a repeated
    chorus/line can otherwise get matched against the wrong sung
    instance) -- real confirmed bug (2026-08-14, user's own catch): a
    repeat-instance mixup is a REAL timing failure, and excluding it from
    the denominator instead of counting it wrong inflated every reported
    accuracy number, most visibly when comparing against USKM's own
    output. A wildly-wrong delta already fails the plain timing-tolerance
    check on its own -- no separate exclusion step was ever needed.
"""

from __future__ import annotations

import difflib
import re
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
    # Coverage: what fraction of EACH side's own word-start syllables text-
    # matched the other side AT ALL. A word that never matches (e.g.
    # garbled/wrong text on either side) never appears in pitch_mismatches/
    # timing_mismatches -- it just isn't scored -- so these are the only
    # signal that catches that failure mode.
    coverage_fresh: float = 0.0     # n_matched / len(fresh words)
    coverage_existing: float = 0.0  # n_matched / len(existing words)
    unmatched_fresh: List[str] = field(default_factory=list)     # fresh words that matched nothing in existing
    unmatched_existing: List[str] = field(default_factory=list)  # existing words that matched nothing in fresh
    # Real bidirectional "is this a 100% match" numbers, same design as
    # scratchpad/compare_video_games.py (2026-08-13/14): a word only
    # counts as CORRECT if it text-matched AND landed within timing
    # tolerance -- nothing is excluded from either denominator, so a
    # missing GT word (recall) or an extra/hallucinated candidate word
    # (precision) both directly bring the score down. For a true 100%
    # match, recall == precision == 1.0 is required -- matching every GT
    # word is not enough on its own if extra notes are also present.
    recall: float = 0.0     # n_correct / len(existing words) -- every real word reproduced, in the right place
    precision: float = 0.0  # n_correct / len(fresh words) -- every produced word is a real, correctly-placed one
    f1: float = 0.0


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
    min_coverage: float = config.EXISTING_TXT_MIN_COVERAGE,
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

    # NOTHING is excluded from scoring here -- every text-matched pair in
    # `candidates` counts toward the denominator below. This module used to
    # bucket timing deltas at 200ms resolution and DROP any candidate more
    # than 3.0s from the dominant cluster before scoring pitch/timing
    # accuracy at all (a "repeat-instance guard"), on the theory that a
    # repeated line/chorus pairing against the wrong occurrence of itself
    # was noise to filter out. Real confirmed bug (2026-08-14, user's own
    # catch): that's exactly backwards -- a repeat-occurrence mixup is a
    # REAL timing failure, and silently excluding it from the denominator
    # (rather than counting it as wrong) inflated every reported accuracy
    # number, most visibly when comparing against USKM's own output. Same
    # fix already validated in scratchpad/compare_video_games.py's own
    # design: nothing is excluded, an outlier's own displacement already
    # fails the timing-tolerance check on its own, no separate exclusion
    # step needed.
    n_timing_ok = sum(1 for _, e_start, f_start, _, _ in candidates if abs(e_start - f_start) <= timing_tolerance_sec)
    stats.timing_within_tolerance_pct = n_timing_ok / len(candidates)
    stats.timing_mismatches = [(text, e_start, f_start) for text, e_start, f_start, _, _ in candidates
                                if abs(e_start - f_start) > timing_tolerance_sec]

    # Pitch is scored only among pairs that ALSO have correct timing --
    # comparing pitch class across a repeat-mismatched pair would be
    # comparing two different sung instances, not a meaningful check of
    # this one. Same "pitch conditional on correct recall" design as
    # compare_video_games.py.
    correct = [c for c in candidates if abs(c[1] - c[2]) <= timing_tolerance_sec]
    n_pitch_ok = sum(1 for _, _, _, e_pc, f_pc in correct if e_pc == f_pc)
    stats.pitch_class_accuracy = n_pitch_ok / len(correct) if correct else 0.0
    stats.pitch_mismatches = [(text, e_pc, f_pc) for text, _, _, e_pc, f_pc in correct if e_pc != f_pc]

    # Real bidirectional "100% match" numbers (see the dataclass field
    # docstrings): a word only counts as correct if BOTH text and timing
    # match, over the FULL existing/fresh word counts -- a missing GT word
    # or an extra/hallucinated candidate word both directly cost score.
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
    # Coverage gate: a word that never text-matches at all (e.g. garbled or
    # wrong output text) never shows up in pitch_mismatches/timing_mismatches
    # above -- it's simply absent from `candidates` -- so pitch/timing
    # accuracy alone can look perfect while a real chunk of the song was
    # silently never scored. Confirmed real case: 12/116 (10%) of a real
    # MXL+LRC output's words never matched ground truth at all and were
    # invisible to the accuracy numbers until this gate was added.
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
