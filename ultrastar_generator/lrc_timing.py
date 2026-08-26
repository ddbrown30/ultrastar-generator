"""Cross-checks pass-3 line placement against LRCLIB's synced lyrics. Diagnostic only -- flags disagreements, never corrects.

Three-tier time calibration, increasingly flexible: (1) constant offset (mode of per-line deltas at 1s resolution), (2) linear drift (robust Theil-Sen offset+slope fit), (3) piecewise/isotonic correction for discontinuous drift (a real edit difference between recordings). Tier 3 proceeds unconditionally if tier 1/2 already found partial support ("refine"); otherwise it needs an independent `structural_check` to pass ("rescue").

Known limit: `check_repeat_structure` can't always distinguish a genuinely different recording from a legitimate different edition of the same song.
"""

from __future__ import annotations

import re
import difflib
from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Dict, List, Optional, Tuple

from . import config
from .models import Syllable, Word
from .text_normalize import (normalize_word as _normalize, is_filler_token,
                              normalize_for_fuzzy_match as _normalize_fuzzy, is_all_filler)

_LRC_TAG_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def _normalize_line(text: str) -> str:
    return " ".join(n for n in (_normalize(tok) for tok in text.split()) if n)


def lrc_line_window(lrc_lines: List[Tuple[float, str]], li: int) -> Tuple[float, float]:
    """Real-time window [line's own start, next line's own start) an LRC line's words should fall within; last line gets a flat +5.0s fallback width."""
    t0 = lrc_lines[li][0]
    t1 = lrc_lines[li + 1][0] if li + 1 < len(lrc_lines) else t0 + 5.0
    return t0, t1


def words_in_time_window(words: List[Word], t0: float, t1: float, slack: float = 0.5) -> List[Word]:
    """Every `Word` whose own start falls in `[t0 - slack, t1 + slack]`, order preserved."""
    return [w for w in words if t0 - slack <= w.start <= t1 + slack]


def match_block_to_candidates(
    target_norm: List[str],
    candidate_words: List[Word],
    *,
    fuzzy_min_ratio: Optional[float] = None,
    min_candidate_confidence: Optional[float] = None,
) -> Dict[int, Word]:
    """Matches target block's normalized words against a candidate `Word` list via one whole-block diff: "equal" opcodes match directly (gated on candidate confidence); a single-token "replace" tries every candidate token in that slice for the best fuzzy-ratio match (handles ASR mishearing, e.g. "favors" for "favorites"). Returns `{target_local_index: matched Word}`."""
    if fuzzy_min_ratio is None:
        fuzzy_min_ratio = config.MXL_LRC_FUZZY_TEXT_MIN_RATIO
    if min_candidate_confidence is None:
        min_candidate_confidence = config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE
    candidate_norm = [_normalize(w.text) for w in candidate_words]
    sm = difflib.SequenceMatcher(None, target_norm, candidate_norm, autojunk=False)
    matched: Dict[int, Word] = {}
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(a2 - a1):
                w = candidate_words[b1 + k]
                if w.confidence >= min_candidate_confidence:
                    matched[a1 + k] = w
        elif tag == "replace" and (a2 - a1) == 1:
            best_ratio, best_w = 0.0, None
            for bk in range(b1, b2):
                ratio = difflib.SequenceMatcher(None, target_norm[a1], candidate_norm[bk]).ratio()
                if ratio > best_ratio:
                    best_ratio, best_w = ratio, candidate_words[bk]
            if best_ratio >= fuzzy_min_ratio and best_w is not None \
                    and best_w.confidence >= min_candidate_confidence:
                matched[a1] = best_w
    return matched


def _strip_filler_flat(normalized_line: str) -> str:
    """Whitespace-flattened `normalized_line` with filler/ad-lib tokens (`text_normalize.
    is_filler_token` -- FILLER_WORDS plus a bare/repeated vocalise syllable like "na"/"ahahah")
    removed; used only as a fallback alongside the raw flattened form."""
    return "".join(tok for tok in normalized_line.split() if not is_filler_token(tok))


def check_repeat_structure(our_lines: List[str], lrc_line_texts: List[str],
                            min_repeat: int = 3, min_word_len: int = 4) -> Optional[str]:
    """Rejects an LRC candidate whose repeat structure (e.g. extra chorus repeats from a different edition) doesn't match ours -- global time calibration alone can't catch this.

    Finds our own most-repeated normalized line (skipped if nothing repeats >= `min_repeat` times), then counts WORD occurrences of its content words (>= `min_word_len` chars) across the whole song on each side -- a repeated chorus is often split across several near-duplicate line variants, so counting a distinctive word is more robust than counting exact line repeats. Tolerance +-15% (min +-1) absorbs ordinary per-song noise.

    Returns a rejection reason, or None if consistent enough to trust. Re-exported from realign.py for backward compat."""
    our_normalized_lines = [_normalize_line(t) for t in our_lines]
    line_counts = Counter(nl for nl in our_normalized_lines if nl)
    if not line_counts:
        return None
    most_line, line_n = line_counts.most_common(1)[0]
    if line_n < min_repeat:
        return None

    our_word_counts = Counter(_normalize(tok) for line in our_lines for tok in line.split())
    lrc_word_counts = Counter(_normalize(tok) for line in lrc_line_texts for tok in line.split())
    candidate_words = {w for w in most_line.split() if len(w) >= min_word_len}
    if not candidate_words:
        return None
    fingerprint_word, our_n = max(
        ((w, our_word_counts.get(w, 0)) for w in candidate_words), key=lambda t: t[1])
    if our_n < min_repeat:
        return None
    lrc_n = lrc_word_counts.get(fingerprint_word, 0)
    tolerance = max(1, round(0.15 * our_n))
    if abs(our_n - lrc_n) > tolerance:
        return (f"our most-repeated line ({most_line!r}, {line_n}x) has a distinctive word "
                f"({fingerprint_word!r}) appearing {our_n}x total in the existing file but {lrc_n}x in "
                f"the LRC candidate's own lyrics (tolerance +/-{tolerance}) -- likely a different "
                f"edition/arrangement with a different repeat structure, not just timing noise")
    return None


@dataclass
class LineReconciliation:
    """Result of `reconcile_line_structure`."""
    lrc_lines: List[Tuple[float, str]]   # candidate lines matched to our_lines, in LRC order; unmatched ones dropped.
    our_line_index: List[int]            # parallel to lrc_lines: which our_lines[i] each entry came from (lets a caller build a word-to-LRC-line mapping directly instead of re-deriving one via a separate word-level diff).
    n_our_lines: int
    n_matched: int
    n_lrc_dropped: int    # candidate lines with no match in our_lines
    n_our_unmatched: int  # our own lines with no match in the candidate
    match_ratio: float    # n_matched / n_our_lines


FUZZY_LINE_MIN_RATIO = 0.85  # see _flat_fuzzy_equal


def _flat_fuzzy_equal(a: str, b: str, min_ratio: float = FUZZY_LINE_MIN_RATIO) -> bool:
    """Whether two flattened normalized strings match modulo a dropped/elided letter (e.g. "Ev'rything"/"Everything"). Exact match checked first; fuzzy is only a fallback for an already-fixed pair, never used to widen a search."""
    if a == b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= min_ratio


def _consume_as_merge(target_flat: str, piece_flats: List[str]) -> Optional[int]:
    """Whether `target_flat` is the exact back-to-back concatenation of `piece_flats[0]`, `[1]`, ... (each piece's length fixes its slice; `_flat_fuzzy_equal` tolerance applies per piece). Returns pieces consumed (>=2), or None if no such split exists."""
    if not target_flat:
        return None
    pos = 0
    for k, piece in enumerate(piece_flats, start=1):
        if not piece:
            return None
        end = pos + len(piece)
        if end > len(target_flat) or not _flat_fuzzy_equal(target_flat[pos:end], piece):
            return None
        pos = end
        if pos == len(target_flat):
            return k if k >= 2 else None
    return None


def reconcile_line_structure(
    our_lines: List[str],
    lrc_lines: List[Tuple[float, str]],
    max_skip: int = 8,
    min_match_ratio: float = 0.5,
    max_merge_lines: int = 4,
) -> Optional[LineReconciliation]:
    """Reconciles an LRC candidate's lines against ours when repeat structure doesn't match, as a localized alternative to `check_repeat_structure`'s outright rejection.

    Two cursors walk our_lines/lrc_lines forward only, never backward, so a repeated phrase can't be confused with an earlier occurrence. On a mismatch, a joint (p, q)-offset search (bounded by `max_skip` per side) finds the next plain-line or merged-line (`_consume_as_merge`) match, trying a filler-word-tolerant fallback (`FILLER_WORDS`) too; unresolved lines are dropped.

    Returns None if the matched fraction of our lines falls below `min_match_ratio`, else a `LineReconciliation`.
    """
    our_norm = [_normalize_line(t) for t in our_lines]
    lrc_norm = [_normalize_line(t) for _t, t in lrc_lines]
    # Flattened (no inter-word space) form so a stray/missing space at a word boundary can't cause a false mismatch; still an exact match otherwise.
    our_flat = ["".join(t.split()) for t in our_norm]
    lrc_flat = ["".join(t.split()) for t in lrc_norm]
    # Filler-stripped versions, fallback only -- see _match_kind.
    our_flat_nofill = [_strip_filler_flat(t) for t in our_norm]
    lrc_flat_nofill = [_strip_filler_flat(t) for t in lrc_norm]
    n_i, n_j = len(our_norm), len(lrc_norm)

    def _match_kind(i2: int, j2: int) -> Optional[Tuple[str, int]]:
        """(kind, k) if our_flat[i2]/lrc_flat[j2] align directly (a plain
        match or a merge in either direction), else None. k is always 1
        for a plain match, the piece-count for either merge kind."""
        if i2 >= n_i or j2 >= n_j:
            return None
        if our_flat[i2] and _flat_fuzzy_equal(our_flat[i2], lrc_flat[j2]):
            return ("match", 1)
        # Filler-tolerant fallback; guarded so two all-filler lines can't match via an empty-string tie.
        if our_flat_nofill[i2] and lrc_flat_nofill[j2] and \
                _flat_fuzzy_equal(our_flat_nofill[i2], lrc_flat_nofill[j2]):
            return ("match", 1)
        lrc_k = _consume_as_merge(lrc_flat[j2], our_flat[i2:i2 + max_merge_lines])
        if lrc_k is None:
            lrc_k = _consume_as_merge(lrc_flat_nofill[j2], our_flat_nofill[i2:i2 + max_merge_lines])
        if lrc_k is not None:
            return ("lrc_merge", lrc_k)
        our_k = _consume_as_merge(our_flat[i2], lrc_flat[j2:j2 + max_merge_lines])
        if our_k is None:
            our_k = _consume_as_merge(our_flat_nofill[i2], lrc_flat_nofill[j2:j2 + max_merge_lines])
        if our_k is not None:
            return ("our_merge", our_k)
        return None

    i = j = 0
    kept: List[Tuple[float, str]] = []
    our_line_index: List[int] = []
    our_matched = [False] * n_i
    lrc_used: set = set()
    while i < n_i and j < n_j:
        found = None  # (p, q, kind, k)
        for total in range(0, 2 * max_skip + 1):
            for p in range(max(0, total - max_skip), min(total, max_skip) + 1):
                q = total - p
                if q > max_skip:
                    continue
                m = _match_kind(i + p, j + q)
                if m is not None:
                    found = (p, q, m[0], m[1])
                    break
            if found is not None:
                break

        if found is None:
            j += 1
            continue

        p, q, kind, k = found
        i, j = i + p, j + q
        if kind == "match":
            kept.append(lrc_lines[j])
            our_line_index.append(i)
            our_matched[i] = True
            lrc_used.add(j)
            i += 1
            j += 1
        elif kind == "lrc_merge":
            t_start, t_next = lrc_line_window(lrc_lines, j)
            word_counts = [max(1, len(our_norm[i + m2].split())) for m2 in range(k)]
            total_words = sum(word_counts) or 1
            cumulative = 0
            for m2 in range(k):
                t_piece = t_start + (t_next - t_start) * (cumulative / total_words)
                kept.append((t_piece, our_lines[i + m2]))
                our_line_index.append(i + m2)
                our_matched[i + m2] = True
                cumulative += word_counts[m2]
            lrc_used.add(j)
            i += k
            j += 1
        else:  # our_merge
            kept.append((lrc_lines[j][0], our_lines[i]))
            our_line_index.append(i)
            our_matched[i] = True
            lrc_used.update(range(j, j + k))
            i += 1
            j += k

    n_matched = sum(our_matched)
    match_ratio = n_matched / n_i if n_i else 0.0
    if match_ratio < min_match_ratio:
        return None
    return LineReconciliation(
        lrc_lines=kept, our_line_index=our_line_index, n_our_lines=n_i, n_matched=n_matched,
        n_lrc_dropped=n_j - len(lrc_used), n_our_unmatched=n_i - n_matched,
        match_ratio=match_ratio,
    )


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
    calibration_kind: Optional[str] = None   # "constant", "drift", "piecewise", or "isotonic"
    correction_fn: Optional[Callable[[float], float]] = None  # raw_key_time -> corrected_real_time, populated for every successful kind
    calibration_confidence: float = 0.0
    holdout_residual_sec: Optional[float] = None  # diagnostic only, tier 3 only -- see _holdout_residual_sec
    flags: List[LineTimingFlag] = field(default_factory=list)
    skipped_reason: Optional[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


def _reconstruct_lines(syllables: List[Syllable]) -> List[Tuple[int, float, List[str]]]:
    """Groups syllables into lines by Syllable.line_id. Returns (first_syllable_index, line_start_sec, normalized_word_tokens) per line; syllables with no line_id are excluded."""
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
    """Matches our_lines to lrc_lines at the word level via one whole-sequence alignment, then re-derives a per-line correspondence by majority vote of each line's matched words -- recovers lines that differ by only a word or two, unlike a whole-line exact match.

    Returns (our_line_idx, lrc_start_sec, delta_sec) triples, at most one per our-line.
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
    """Theil-Sen (median-of-pairwise-slopes) robust fit of delta = offset + slope*lrc_start -- outliers can't drag the median slope the way they would an ordinary least-squares fit.

    Returns (offset, slope, confidence, n_inliers), or None if fewer than 2 distinct lrc_start values exist. confidence is the fraction of candidates within inlier_tolerance_sec of the fitted line.
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


def _filter_theilsen_inliers(
    candidates: List[Tuple[int, float, float]],
    fit: Tuple[float, float, float, int],
    outlier_tolerance_sec: float,
) -> List[Tuple[int, float, float]]:
    """Tier 3's noise filter: keeps only candidates within `outlier_tolerance_sec` of tier 2's own (possibly too-imprecise-to-trust) Theil-Sen fit, to separate real signal from raw text-matching noise."""
    intercept, slope, _confidence, _n = fit
    return [c for c in candidates if abs(c[2] - (intercept + slope * c[1])) <= outlier_tolerance_sec]


def _enforce_monotonic_anchors(
    candidates: List[Tuple[int, float, float]],
) -> List[Tuple[float, float]]:
    """"piecewise" tier-3 strategy: greedily drops any candidate whose implied real time would be earlier than the running max already kept (a line can't be timed before an earlier one), keeping the first of any conflicting pair.

    Returns (lrc_start, real_time) anchor pairs, sorted by lrc_start."""
    kept: List[Tuple[float, float]] = []
    running_max: Optional[float] = None
    for _key, lrc_start, delta in sorted(candidates, key=lambda c: c[1]):
        real_time = lrc_start + delta
        if running_max is not None and real_time < running_max:
            continue
        kept.append((lrc_start, real_time))
        running_max = real_time
    return kept


def _pava_isotonic(candidates: List[Tuple[int, float, float]]) -> List[Tuple[float, float]]:
    """"isotonic" tier-3 strategy: Pool-Adjacent-Violators (PAVA) fits the candidates' implied real times as a monotonic non-decreasing step function, pooling (averaging) a violating run with its predecessor instead of dropping it like `_enforce_monotonic_anchors` does.

    Returns one (mean_lrc_start, fitted_real_time) anchor per pooled block, in lrc_start order."""
    pts = sorted(((lrc_start, lrc_start + delta) for _key, lrc_start, delta in candidates),
                 key=lambda p: p[0])
    # Each block: [sum_x, sum_y, count, mean_y].
    blocks: List[List[float]] = []
    for x, y in pts:
        blocks.append([x, y, 1, y])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            count = b1[2] + b2[2]
            merged = [b1[0] + b2[0], b1[1] + b2[1], count, 0.0]
            merged[3] = merged[1] / count
            blocks.append(merged)
    return [(b[0] / b[2], b[3]) for b in blocks]


def _correction_from_anchors(anchors: List[Tuple[float, float]]) -> Callable[[float], float]:
    """Builds `correction_fn(lrc_start) -> corrected_real_time` by linear interpolation between consecutive (already sorted, monotonic) anchors, extrapolating past the ends using the boundary segment's own slope. Shared by both tier-3 strategies."""
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    n = len(xs)

    def fn(t: float) -> float:
        if n == 1:
            return ys[0] + (t - xs[0])  # only one anchor: hold its own local (no-op) correction
        if t <= xs[0]:
            i = 0
        elif t >= xs[-1]:
            i = n - 2
        else:
            i = 0
            while i + 1 < n - 1 and xs[i + 1] < t:
                i += 1
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (t - x0) / (x1 - x0)

    return fn


def _max_adjacent_gap(anchors: List[Tuple[float, float]]) -> float:
    """Largest lrc_start gap between two consecutive anchors."""
    if len(anchors) < 2:
        return 0.0
    return max(b[0] - a[0] for a, b in zip(anchors, anchors[1:]))


def _holdout_residual_sec(
    anchors: List[Tuple[float, float]],
    min_anchors: int = config.LRC_TIMING_HOLDOUT_MIN_ANCHORS,
) -> Optional[float]:
    """Odd/even-anchor holdout check: fits a correction on odd-indexed anchors, scores it against even-indexed anchors' real times -- a fit tracking noise typically predicts held-out anchors worse than one tracking real structure.

    Returns mean absolute holdout residual in seconds, or None if not enough anchors on both sides. Diagnostic only, not a hard gate."""
    if len(anchors) < min_anchors:
        return None
    odd = anchors[1::2]
    even = anchors[0::2]
    if len(odd) < 2 or len(even) < 2:
        return None
    fit_fn = _correction_from_anchors(odd)
    residuals = [abs(y - fit_fn(x)) for x, y in even]
    return sum(residuals) / len(residuals)


def _piecewise_or_isotonic_calibration(
    candidates: List[Tuple[int, float, float]],
    theilsen_fit: Tuple[float, float, float, int],
    drift_model: str = config.LRC_TIMING_DRIFT_MODEL,
    outlier_tolerance_sec: float = config.LRC_TIMING_PIECEWISE_OUTLIER_TOLERANCE_SEC,
    min_anchors: int = config.LRC_TIMING_PIECEWISE_MIN_ANCHORS,
    max_anchor_gap_sec: float = config.LRC_TIMING_PIECEWISE_MAX_ANCHOR_GAP_SEC,
) -> Optional[Tuple[float, float, float, str, Callable[[float], float], Optional[float]]]:
    """Tier 3: builds a piecewise correction for a discontinuous drift neither tier 1 nor tier 2 can fit. Filters to Theil-Sen inliers, derives a monotonic anchor set via `drift_model`'s strategy, gates on minimum anchor count + max adjacent-anchor gap, then interpolates a correction function.

    Returns (offset, slope, confidence, kind, correction_fn, holdout_residual_sec) on success -- offset/slope are only representative (the real correction is always `correction_fn`); confidence is the fraction of original candidates that survived the inlier filter. Returns None if the anchor-count/gap gates aren't cleared.
    """
    inliers = _filter_theilsen_inliers(candidates, theilsen_fit, outlier_tolerance_sec)
    if not inliers:
        return None

    if drift_model == "isotonic":
        anchors = _pava_isotonic(inliers)
    else:
        anchors = _enforce_monotonic_anchors(inliers)

    if len(anchors) < min_anchors:
        return None
    if _max_adjacent_gap(anchors) > max_anchor_gap_sec:
        return None

    correction_fn = _correction_from_anchors(anchors)
    confidence = len(inliers) / len(candidates)
    holdout_residual_sec = _holdout_residual_sec(anchors)

    # Representative offset/slope from the first segment, solved so `t + rep_offset + rep_slope*t` reproduces its two anchors exactly.
    x0, y0 = anchors[0]
    if len(anchors) >= 2 and anchors[1][0] != x0:
        x1, y1 = anchors[1]
        rep_slope = (y1 - y0) / (x1 - x0) - 1.0
    else:
        rep_slope = 0.0
    rep_offset = y0 - x0 - rep_slope * x0

    kind = "isotonic" if drift_model == "isotonic" else "piecewise"
    return rep_offset, rep_slope, confidence, kind, correction_fn, holdout_residual_sec


def find_cursor_window_match(
    cursor: int,
    haystack_norm: List[str],
    target_tokens: List[str],
    pending_word_count: int,
    *,
    tight_slack: int = 4,
    wide_multiplier: int = 3,
    wide_slack: int = 10,
    min_match_fraction: float = 0.5,
    lenient_min_matches: int = 2,
) -> Optional[Tuple[List[Tuple[str, int, int, int, int]], List[str]]]:
    """Forward-only cursor-based window search: find `target_tokens` somewhere after `cursor` in `haystack_norm`. Returns the winning window's `(opcodes, window)` (raw `difflib` opcodes, or None if neither window confirms) -- interpreting opcodes is left to the caller, since callers want different things from a match.

    Tries both a TIGHT window (barely more than `target_tokens`' size) and a WIDE window (grown by `pending_word_count`, so a garbled/ad-lib stretch can recover once real content resumes); tight wins whenever it covers at least half of `target_tokens` (`min_match_fraction`), regardless of what wide finds -- otherwise even a wide window on a fresh search can jump past the nearest correct occurrence into a later repeat of the same content, since `difflib` doesn't prefer the nearest match.

    Both windows share two guards: a minimum-match-count floor (rejects a coincidentally-shared common word) and a span guard bounded by the evidence actually found (rejects a technically-passing match whose few real tokens are scattered across the whole window, which would strand the cursor at the window's far edge)."""
    n = len(haystack_norm)

    def _try(window: List[str], require_strict_fraction: bool) -> Tuple[Optional[List], int]:
        if not window:
            return None, 0
        sm = difflib.SequenceMatcher(None, window, target_tokens, autojunk=False)
        opcodes = sm.get_opcodes()
        equal_wa = [(wa1, wa2) for tag, wa1, wa2, _ca1, _ca2 in opcodes if tag == "equal"]
        matched = sum(wa2 - wa1 for wa1, wa2 in equal_wa)
        # A 1-word target just needs its single token. Longer targets need >= lenient_min_matches even in lenient mode, so one coincidental word can't validate a multi-word target; strict mode additionally requires the full fraction.
        if len(target_tokens) == 1:
            min_needed = 1
        elif require_strict_fraction:
            min_needed = max(lenient_min_matches, round(len(target_tokens) * min_match_fraction))
        else:
            min_needed = lenient_min_matches
        span = (max(wa2 for _wa1, wa2 in equal_wa) - min(wa1 for wa1, _wa2 in equal_wa)) if equal_wa else 0
        max_span = matched * 3 + 3
        if matched < min_needed or span > max_span:
            return None, matched
        return opcodes, matched

    tight_end = min(cursor + len(target_tokens) + tight_slack, n)
    tight_window = haystack_norm[cursor:tight_end]
    tight_opcodes, tight_matched = _try(tight_window, require_strict_fraction=False)

    wide_end = min(cursor + pending_word_count * wide_multiplier + wide_slack, n)
    wide_window = haystack_norm[cursor:wide_end]
    window_is_inflated = pending_word_count > len(target_tokens)
    wide_opcodes, _wide_matched = _try(wide_window, require_strict_fraction=window_is_inflated)

    tight_preference_threshold = max(1, round(len(target_tokens) * min_match_fraction))
    if tight_opcodes is not None and tight_matched >= tight_preference_threshold:
        return tight_opcodes, tight_window
    if wide_opcodes is not None:
        return wide_opcodes, wide_window
    if tight_opcodes is not None:
        return tight_opcodes, tight_window
    return None


def match_asr_to_lrc_lines(asr_words: List[Word], lrc_lines: List[Tuple[float, str]]
                            ) -> List[Tuple[int, float, float]]:
    """Matches ASR's word stream against LRC lines' text to find, per LRC line, the earliest real ASR word confidently belonging to it -- a real-time anchor per line for calibrating away a systematic LRC-vs-audio offset before trusting LRC timestamps as placement anchors. Returns (lrc_line_index, lrc_start, delta) candidates, delta = ASR start minus LRC's declared start.

    Uses `find_cursor_window_match` for the window search; advances the cursor to just past the last matched word, never a raw unconfirmed opcode boundary. Shared by `mxl_lrc_generator.py` and `realign.py`.

    Matching tolerates filler/ad-lib vocalise variation (`text_normalize.normalize_for_fuzzy_match`
    -- "ah-ah-ah" vs "na na na" transcribing the same real sound shouldn't count as a mismatch),
    except a line that's ENTIRELY filler is skipped outright (`is_all_filler`) -- no real content
    to safely anchor a match on."""
    MAX_PENDING_WORDS = 60

    asr_norm = [_normalize_fuzzy(w.text) for w in asr_words]
    cursor = 0
    pending_word_count = 0
    candidates = []
    for li, (lrc_start, text) in enumerate(lrc_lines):
        line_tokens = [t for t in (_normalize(tok) for tok in text.split()) if t]
        if not line_tokens or is_all_filler(line_tokens):
            continue
        line_tokens = [_normalize_fuzzy(tok) for tok in text.split() if _normalize(tok)]
        pending_word_count = min(pending_word_count + len(line_tokens), MAX_PENDING_WORDS)
        found = find_cursor_window_match(cursor, asr_norm, line_tokens, pending_word_count)
        if found is None:
            # No match -- don't advance the cursor; next line's window grows to cover this one too.
            continue
        opcodes, _window = found
        first_offset = None
        last_offset = None
        for tag, a1, a2, _b1, _b2 in opcodes:
            if tag != "equal":
                continue
            if first_offset is None:
                first_offset = a1
            last_offset = a2 - 1
        asr_idx = cursor + first_offset
        candidates.append((li, lrc_start, asr_words[asr_idx].start - lrc_start))
        cursor += last_offset + 1
        pending_word_count = 0
    return candidates


def two_tier_time_calibration(
    candidates: List[Tuple[int, float, float]],
    min_calibration_samples: int = config.LRC_TIMING_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.LRC_TIMING_MIN_CALIBRATION_CONFIDENCE,
    min_drift_samples: int = config.LRC_TIMING_MIN_DRIFT_SAMPLES,
    min_drift_confidence: float = config.LRC_TIMING_MIN_DRIFT_CONFIDENCE,
    drift_inlier_tolerance_sec: float = config.LRC_TIMING_DRIFT_INLIER_TOLERANCE_SEC,
    drift_model: str = config.LRC_TIMING_DRIFT_MODEL,
    piecewise_outlier_tolerance_sec: float = config.LRC_TIMING_PIECEWISE_OUTLIER_TOLERANCE_SEC,
    piecewise_min_anchors: int = config.LRC_TIMING_PIECEWISE_MIN_ANCHORS,
    piecewise_max_anchor_gap_sec: float = config.LRC_TIMING_PIECEWISE_MAX_ANCHOR_GAP_SEC,
    rescue_min_prior_confidence: float = config.LRC_TIMING_RESCUE_MIN_PRIOR_CONFIDENCE,
    structural_check: Optional[Callable[[], Optional[str]]] = None,
) -> Tuple[Optional[float], float, float, Optional[str], Optional[str],
           Optional[Callable[[float], float]], Optional[float]]:
    """Three-tier time calibration (name kept despite now trying three, for backward compat): given (key, lrc_start, delta) candidates, tries a constant offset, then a Theil-Sen linear-drift fit, then a piecewise/isotonic correction for discontinuous drift.

    Returns (offset, slope, confidence, kind, skipped_reason, correction_fn, holdout_residual_sec). `offset` is None (skipped_reason set) if no tier calibrated confidently. `correction_fn` is populated for every successful kind, so callers apply it uniformly without special-casing piecewise/isotonic (whose offset/slope are only representative). `holdout_residual_sec` is diagnostic only, tier 3 only.

    Tier 3's refine-vs-rescue split: if tier 1/2 already found partial support ("refine", `max(tier1_confidence, tier2_confidence) >= rescue_min_prior_confidence`), tier 3 proceeds unconditionally. Otherwise ("rescue"), tier 3 would be supplying both the hypothesis and its own validation with zero independent support -- requires `structural_check` to be given and pass, else declines (the safe default when no check is wired up).

    Shared by `mxl_lrc_generator.py` and `realign.py` (the latter for GAP calibration) via `apply_lrc_timing_check`."""
    if len(candidates) < min_calibration_samples:
        return None, 0.0, 0.0, None, (
            f"only {len(candidates)} matched line(s) (< {min_calibration_samples} required) -- "
            f"not enough to trust a calibration offset"
        ), None, None

    # Tier 1: constant offset, mode at coarse (1s) resolution -- line-level timestamps are too imprecise for a finer bucket.
    BUCKET_SEC = 1.0
    bucket_counts = Counter(round(delta / BUCKET_SEC) for _, _, delta in candidates)
    best_bucket, n_agree = bucket_counts.most_common(1)[0]
    offset, slope, confidence, kind = best_bucket * BUCKET_SEC, 0.0, n_agree / len(candidates), "constant"
    tier1_confidence = confidence

    if confidence < min_calibration_confidence:
        # Tier 2: linear drift fit. Stricter gate than tier 1 -- a 2-parameter fit can trivially match a handful of points, so needs more samples and a higher inlier fraction.
        fit = _robust_linear_fit(candidates, drift_inlier_tolerance_sec) if len(candidates) >= min_drift_samples else None
        tier2_confidence = fit[2] if fit is not None else 0.0
        if fit is None or fit[2] < min_drift_confidence:
            # Tier 3: discontinuous drift. Needs at least a rough Theil-Sen fit as its own noise filter, so unreachable when fit is None.
            tier3 = _piecewise_or_isotonic_calibration(
                candidates, fit, drift_model, piecewise_outlier_tolerance_sec,
                piecewise_min_anchors, piecewise_max_anchor_gap_sec,
            ) if fit is not None else None
            if tier3 is not None:
                t3_offset, t3_slope, t3_confidence, t3_kind, t3_correction_fn, t3_holdout = tier3
                prior_confidence = max(tier1_confidence, tier2_confidence)
                is_rescue = prior_confidence < rescue_min_prior_confidence
                if not is_rescue:
                    return t3_offset, t3_slope, t3_confidence, t3_kind, None, t3_correction_fn, t3_holdout
                # Rescue case: require a structural check to pass before trusting tier 3's fit.
                structural_rejection = structural_check() if structural_check is not None else (
                    "no structural_check provided -- a 'rescue' (tier 1/2 found no independent support at "
                    "all) is never accepted without one, see two_tier_time_calibration's own docstring")
                if structural_rejection is None:
                    return t3_offset, t3_slope, t3_confidence, t3_kind, None, t3_correction_fn, t3_holdout
                return None, 0.0, 0.0, None, (
                    f"tier 3 ({t3_kind}) found a fit ({t3_confidence:.0%} confidence) but tier 1/2 found NO "
                    f"independent support for it first (prior confidence {prior_confidence:.0%} < "
                    f"{rescue_min_prior_confidence:.0%}) -- declined as an unverified 'rescue': "
                    f"{structural_rejection}"
                ), None, None
            return None, 0.0, 0.0, None, (
                f"no clear per-song time calibration -- constant-offset best candidate {offset:+.1f}s "
                f"covers {confidence:.0%} of {len(candidates)} matched lines (need {min_calibration_confidence:.0%}), "
                f"drift fit " + (
                    f"only reached {fit[2]:.0%} inliers (need {min_drift_confidence:.0%})" if fit
                    else f"needs >= {min_drift_samples} matched lines"
                ) + ", and piecewise/isotonic tier 3 also failed its own anchor-count/spacing gate"
            ), None, None
        offset, slope, confidence, _ = fit
        kind = "drift"

    correction_fn: Callable[[float], float] = (lambda t, _o=offset, _s=slope: t + _o + _s * t)
    return offset, slope, confidence, kind, None, correction_fn, None


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
    """Aligns pass-3's lines (grouped by Syllable.line_id) against LRCLIB's synced-lyrics lines by text, calibrates a per-song time offset (three-tiered, see module docstring), then flags any line whose delta from the calibrated expectation exceeds flag_tolerance_sec.

    Returns stats only -- `syllables` is never modified.
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

    # Structural check for tier 3's "rescue" case, built from data already in hand.
    our_line_texts = [" ".join(tokens) for _, _, tokens in our_lines]
    lrc_line_texts = [text for _, text in lrc_lines]

    def _structural_check() -> Optional[str]:
        return check_repeat_structure(our_line_texts, lrc_line_texts)

    offset, slope, confidence, kind, skipped_reason, correction_fn, holdout_residual_sec = two_tier_time_calibration(
        candidates, min_calibration_samples, min_calibration_confidence,
        min_drift_samples, min_drift_confidence, drift_inlier_tolerance_sec,
        structural_check=_structural_check,
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
    stats.correction_fn = correction_fn
    stats.holdout_residual_sec = holdout_residual_sec

    if verbose:
        drift_desc = f", drift {slope:+.4f}s per LRC-second" if kind == "drift" else ""
        rep_desc = " (offset/slope shown are representative only -- see correction_fn)" if kind in (
            "piecewise", "isotonic") else ""
        holdout_desc = f", odd/even holdout residual {holdout_residual_sec:.2f}s" if holdout_residual_sec is not None else ""
        print(f"[lrc-timing] calibration ({kind}): offset {offset:+.1f}s{drift_desc}{rep_desc}, "
              f"{confidence:.0%} agreement over {len(candidates)} matched line(s){holdout_desc}")

    for our_idx, lrc_start, delta in candidates:
        expected = correction_fn(lrc_start)
        residual = (lrc_start + delta) - expected
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
