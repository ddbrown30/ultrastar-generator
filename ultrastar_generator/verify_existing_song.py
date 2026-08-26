"""Verifies an EXISTING UltraStar .txt file's pitch/timing against a FRESH
pipeline run of the same song. Word-level sequence alignment; pitch compared
at PITCH CLASS (mod 12). Filler/ad-lib "noise" (na na na, ah ah ah, mmm, ...) is
discarded from both sides entirely before scoring -- everything else counts,
nothing else is excluded from scoring denominators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import config
from .lrc_timing import find_cursor_window_match
from .models import LineBreak, Syllable
from .usdx_parser import ParsedSong
from .text_normalize import normalize_word as _normalize, is_filler_token


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


@dataclass
class _WordSpan:
    text: str
    start: float
    midi_note: int


def _line_chunks(entries: List[object]) -> List[List[_WordSpan]]:
    """Groups a syllable stream into whole real (non-empty, non-filler) WORDS, chunked into
    lines split at `LineBreak` markers (one chunk covering everything if the entry list has no
    LineBreak at all, e.g. a synthetically-built syllable list, or is used flat for the "fresh"
    side).

    A word is reconstructed by concatenating a word-start syllable's own text with all of its
    own trailing continuation syllables (skipping melisma-continuation markers,
    `config.MELISMA_CONTINUATION_TEXT`) -- comparing only the word-start syllable's OWN isolated
    fragment (e.g. "John") instead of the whole word (e.g. "Johnny's") was a real bug: two
    independently-generated files can split the very same sung word across a different number of
    syllables (rhythm/melisma reasons unrelated to the word itself), which then could never
    text-match its counterpart at all even though the real word is identical.

    Ad-lib "noise" (na na na, ah ah ah, mmm, ...) is discarded entirely here, not just tolerated,
    so it never counts toward either side's denominators (coverage/recall/precision) and never
    needs to text-match its counterpart's own choice of nonsense syllable."""
    chunks: List[List[_WordSpan]] = []
    cur: List[_WordSpan] = []
    pending: Optional[Syllable] = None
    parts: List[str] = []

    def flush() -> None:
        nonlocal pending, parts
        if pending is not None:
            text = "".join(parts)
            n = _normalize(text)
            if n and not is_filler_token(n):
                cur.append(_WordSpan(text=text, start=pending.start, midi_note=pending.midi_note))
        pending, parts = None, []

    for e in entries:
        if isinstance(e, LineBreak):
            flush()
            if cur:
                chunks.append(cur)
            cur = []
        elif isinstance(e, Syllable):
            piece = e.text.strip()
            if e.is_word_start:
                flush()
                pending = e
                parts = [piece] if piece != config.MELISMA_CONTINUATION_TEXT else []
            elif pending is not None and piece and piece != config.MELISMA_CONTINUATION_TEXT:
                parts.append(piece)
    flush()
    if cur:
        chunks.append(cur)
    return chunks


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
    matched words yields verdict "COULD_NOT_VERIFY", never "PASS".

    Matching is a forward-only cursor over fresh's word stream, one EXISTING line (split at
    `LineBreak`) at a time (`lrc_timing.find_cursor_window_match` -- the same mechanism already
    used for MXL-vs-LRC and LRC-vs-ASR line reconciliation elsewhere in this project), not one
    whole-song `difflib` diff -- a repeated phrase later in the song can no longer be confused
    with an earlier occurrence. Real bug this fixed: a whole-song diff on a heavily-repeated song
    could match an early existing-file line against a much later fresh occurrence of the same
    text, reporting a spurious 60+ second "timing mismatch" that was actually just the wrong
    pairing, not a real placement error."""
    existing_chunks = _line_chunks(existing.entries)
    existing_words: List[_WordSpan] = [s for chunk in existing_chunks for s in chunk]
    fresh_words: List[_WordSpan] = [w for chunk in _line_chunks(fresh_syllables) for w in chunk]
    fresh_norm = [_normalize(w.text) for w in fresh_words]

    MAX_PENDING_WORDS = 60  # mirrors lrc_timing.match_asr_to_lrc_lines's own cap
    cursor = 0
    pending_word_count = 0
    candidates = []  # (text, existing_start, fresh_start, existing_pc, fresh_pc)
    matched_existing_ids = set()
    matched_fresh_ids = set()
    for chunk in existing_chunks:
        tokens = [_normalize(s.text) for s in chunk]
        pending_word_count = min(pending_word_count + len(tokens), MAX_PENDING_WORDS)
        found = find_cursor_window_match(cursor, fresh_norm, tokens, pending_word_count)
        if found is None:
            # No match anywhere in range -- don't advance the cursor, next chunk's window grows.
            continue
        opcodes, _window = found
        last_offset = None
        for tag, a1, a2, b1, b2 in opcodes:
            if tag != "equal":
                continue
            for k in range(a2 - a1):
                f_syl = fresh_words[cursor + a1 + k]
                e_syl = chunk[b1 + k]
                candidates.append((e_syl.text, e_syl.start, f_syl.start,
                                    e_syl.midi_note % 12, f_syl.midi_note % 12))
                matched_existing_ids.add(id(e_syl))
                matched_fresh_ids.add(id(f_syl))
            last_offset = a2 - 1 if last_offset is None else max(last_offset, a2 - 1)
        if last_offset is not None:
            cursor += last_offset + 1
            pending_word_count = 0

    stats = ExistingSongVerification(n_matched=len(candidates))
    stats.coverage_existing = len(candidates) / len(existing_words) if existing_words else 0.0
    stats.coverage_fresh = len(candidates) / len(fresh_words) if fresh_words else 0.0
    stats.unmatched_existing = [s.text for s in existing_words if id(s) not in matched_existing_ids]
    stats.unmatched_fresh = [w.text for w in fresh_words if id(w) not in matched_fresh_ids]

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
