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
real ground-truth timing error to confirm the signal actually correlates
with real problems, and only THEN consider building an actual correction
step.

(CORRECTION, 2026-08-11: this docstring originally pointed at
`compare_full_pipeline_output.py`'s `compare_timing()` as "now
measurable" -- checked via `git log -S` and that file was NEVER
committed to this repo, in any commit, including the one that first
wrote this docstring. It was a real but untracked/ad-hoc scratch script
from that original session (same throwaway-script pattern this project
later explicitly moved away from -- see `verify_existing_song.py`'s own
docstring and CLAUDE.md's "Use verify_existing_song.verify_existing_song
directly for any future real-output-vs-ground-truth comparison -- don't
write another ad hoc script", added in a LATER commit than this file).
`verify_existing_song.verify_existing_song` is the current, git-tracked
equivalent -- it takes a trusted ground-truth `ParsedSong` and this run's
own fresh syllables and returns real `timing_within_tolerance_pct` /
`pitch_class_accuracy` / coverage stats via the same word-level,
repeat-instance-guarded alignment technique `compare_timing()` would
have used. Use IT for validating this module's own flags/tiers against
ground truth -- don't recreate the old script.)

Calibration mirrors musicxml_reference.py's approach but for TIME instead
of PITCH CLASS: LRC line timestamps and our own per-song timing are
first assumed to differ by a roughly constant amount (e.g. a different
silence-trim/lead-in in whichever recording LRCLIB's synced version was
made from), calibrated as the mode of per-line deltas at 1-second
resolution (not mean/median -- a straight median isn't robust against a
song with many closely-matching false candidates). Real audio
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

A THIRD tier exists for a case neither of the above can fit: whichever
recording LRCLIB's synced lyrics were timed against was EDITED
differently from ours -- a repeated chorus removed, a bridge shortened,
etc. -- producing a DISCONTINUOUS drift (the delta jumps, or the slope
itself changes, partway through the song) instead of one smooth global
trend. Tried only when tiers 1 and 2 both fail (see `two_tier_time_
calibration`): filters candidates to inliers of tier 2's OWN Theil-Sen
fit first (even a fit too imprecise to trust as a single global slope is
still useful purely as a noise filter -- see `_filter_theilsen_inliers`),
then builds a correction directly from the surviving anchors --
`config.LRC_TIMING_DRIFT_MODEL` selects between "isotonic" (PAVA
monotonic regression, `_pava_isotonic`) and "piecewise" (greedy
monotonic-anchor filtering + linear interpolation,
`_enforce_monotonic_anchors`) -- both feed the same `_correction_from_
anchors` interpolator, so there's exactly one production code path for
"turn anchors into a correction function" regardless of which tier-3
strategy produced them. Gated on both a minimum anchor count and a
maximum gap between adjacent anchors (see config comments) -- 2 anchors
spanning a huge gap is just tier 2's linear drift again, with less
evidence behind it, and should fail closed rather than fabricate a local
guess across a gap that wide.

REAL VALIDATION, 2026-08-11 (reused cached RAW WHISPERX OUTPUT from past
real debug logs -- Stars, Chicago, David Bowie - Heroes -- no fresh GPU
pass needed, see scratchpad/validate_tier3.py): two clear, opposite
results, both important.

GENUINE WIN: Chicago's auto-picked candidate (lrclib id 34321033) was
previously undocumented as `kind=None` ("FAILED -- no offset found",
see CLAUDE.md's `lrc_mode="windowed"` real comparison) -- tier 1 and 2
both genuinely can't fit it. Tier 3 (isotonic) now calibrates it at 91%
confidence. Checked against Chicago's own real ground-truth `.txt`
(`sandbox/Chicago - When You're Good to Mama/Chicago - When You're Good
to Mama.txt`, timing-only, see CLAUDE.md): the RAW (uncalibrated)
candidate's own line starts land within 1s of ground truth only 33% of
the time (mean error 3.42s); the SAME candidate's tier-3-corrected line
starts land within 1s 87% of the time (mean error 0.76s). A real,
substantial, verified improvement -- exactly the case this tier was
built for.

REAL RISK, found on the SAME validation pass, and the REFINE-VS-RESCUE
FIX built in response (both same day, 2026-08-11): David Bowie -
"Heroes"'s own previously-investigated wrong-recording candidate
(lrclib id 37517902, "Heroes" by Kolacny Brothers -- a CHORAL COVER,
confirmed `calibration_confidence=0.0` under the old 2-tier system, see
this project's own earlier investigation) was ALSO confidently
calibrated by tier 3 -- 84% confidence, not 0%. Checked the corrected
line time for the exact passage that earlier investigation used as its
own ground truth ("Just for one day", real ASR-confirmed at
175.0-177.9s in our audio): the tier-3-corrected candidate lines landed
at 170.8s and 183.5s for the two candidate lines straddling that
content -- 4-8s off, NOT meaningfully more correct than declining to
calibrate at all would have been.

Root cause: a stricter residual threshold on tier 3 alone can't tell
"genuinely complex real drift" from "overfit to a wrong recording" --
both look like a flexible model finding SOME fit. Fixed by requiring
INDEPENDENT evidence (see `two_tier_time_calibration`'s own docstring
for the full refine-vs-rescue design and
`config.LRC_TIMING_RESCUE_MIN_PRIOR_CONFIDENCE`): tier 3 only proceeds
unconditionally when tier 1/2's OWN best (still-rejected) fit already
found SOME real support (>= 30% prior confidence, "refine"); when
neither rigid model found ANY support (Heroes: 26%; Chicago's own
genuinely-rescuable candidate: 38%), tier 3 requires an independent
`structural_check` to pass, and the SAFE DEFAULT (no check provided) is
to decline -- confirmed this alone correctly declines Heroes' rescue
while Chicago's stays available (both re-tested against the same real
cached data, `scratchpad/validate_tier3_v2.py`).

**HONEST LIMIT, found while validating the gate itself**: `check_
repeat_structure` -- the closest existing structural check, wired into
`realign.py`'s `prepare_lrc` and this module's own `apply_lrc_timing_
check` whenever real line-structured text is available -- does NOT
reject Heroes' choral-cover candidate (`check_repeat_structure` returns
`None`, i.e. passes, when compared against a real trusted Rock-Band-
chart copy of this song's own lyrics). So in `realign.py`'s own typical
real usage (an existing file IS usually available, meaning `our_lines`
IS usually provided), this specific Heroes rescue is STILL accepted end
to end -- the gap is only closed for callers with no structural check
wired at all. Also tried and ALSO found NOT to separate this real pair:
LRCLIB-declared duration vs. our real audio duration (Heroes 200s vs.
208.7s actual, ~4% off; Chicago's own good candidate similarly ~1% off
-- both plausible), LRC-line-span vs. audio duration (Heroes 13% short,
Chicago's good candidate 9% short -- same order of magnitude), a bare
`candidate.artist_name` string comparison (Chicago's OWN genuinely-good
candidates are ALSO credited to different names -- "Marcia Lewis -
Topic", "Queen Latifah, Taye Diggs" -- real cast-recording artists for
a musical-theatre song, not the same string as our own "Chicago" artist
tag either, so this would reject a real win, not just Heroes), and the
odd/even holdout residual (Heroes 0.62s vs. Chicago 0.45s -- same order
of magnitude, not a clean separator). **This specific real pair (Heroes
vs. Chicago) is not cleanly separable by any cheap signal tried so
far** -- both are legitimately "different credited performer of the
same song" situations; the actual difference is CONTENT/arrangement,
which none of these proxies measure directly. Left open rather than
force-fit a threshold to 2 data points a second time (see this same
docstring's earlier `LRC_TIMING_RESCUE_MIN_PRIOR_CONFIDENCE` caveat
about exactly that risk). `check_repeat_structure` stays wired anyway
(catches OTHER real cases, e.g. Americans' 31-vs-40 repeat-count
mismatch, and provides real, if partial, coverage) -- Heroes' own
specific residual risk is reported here, not silently declared fixed.
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

_LRC_TAG_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def _normalize(s: str) -> str:
    # Real bug found via reconcile_line_structure's own real-audio
    # validation (2026-08-15, David Bowie - I'm Afraid of Americans):
    # our own file's apostrophes are curly (U+2019, e.g. "Johnny's"),
    # LRCLIB's are straight ASCII ('). The old regex below only ever kept
    # straight apostrophes, silently DELETING every curly one -- "Johnny's"
    # normalized to "johnnys" on our side but "johnny's" on the LRC side,
    # so EVERY line containing an apostrophe failed to match at all, even
    # an otherwise byte-identical line. realign.py/mxl_lrc_generator.py's
    # own _normalize already had this exact fix; this module's copy (and
    # lyrics_lookup.py's) had drifted out of sync with it.
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", s)


def _normalize_line(text: str) -> str:
    return " ".join(n for n in (_normalize(tok) for tok in text.split()) if n)


def check_repeat_structure(our_lines: List[str], lrc_line_texts: List[str],
                            min_repeat: int = 3, min_word_len: int = 4) -> Optional[str]:
    """Rejects an LRC candidate whose REPEAT STRUCTURE doesn't match ours --
    real confirmed case (David Bowie - "I'm Afraid of Americans", see
    CLAUDE.md): our own file's most-repeated line ("I'm afraid of
    Americans") appears 31 times total across its repeated-word family, but
    the auto-picked LRCLIB candidate (a different edition/box-set mix of
    the same song) has 40 -- 9 EXTRA chorus repeats, a real structural
    arrangement difference between recordings, not just per-line timing
    noise. Global time CALIBRATION alone can't reliably catch this: a
    genuinely different-but-similar-length edition can still clear the
    confidence bar by chance on the non-repeated portions of the song
    (confirmed: this exact candidate calibrated at 48%, in the same range
    as OTHER real candidates' validated-good calibrations, e.g. Ordinary
    Day's 46% -- the confidence NUMBER alone can't tell these apart).

    First finds OUR OWN most-repeated normalized LINE (skipped entirely if
    nothing repeats at least `min_repeat` times -- most songs have no such
    line, and this check has nothing to say about them). A repeated
    CHORUS is rarely one single exact-duplicate line, though -- real
    confirmed case: "I'm afraid of Americans"/"...of the world"/"...I
    can't help it"/"...I can't" are four DIFFERENT lines, so comparing
    only the single most-repeated exact line (10x in our file, 12x in the
    candidate -- within tolerance on its own) completely misses the real
    mismatch, since the true repeat count is split across all four
    variants. Instead, this counts WORD occurrences (not exact-line
    matches) for the most-repeated line's own content words (short/filler
    words under `min_word_len` chars excluded, since a common word like
    "of"/"the" repeating a lot doesn't indicate a repeated CHORUS the way
    a shared distinctive word across every variant does) across the WHOLE
    song on each side, and uses whichever qualifying word has the HIGHEST
    count in our own file as the comparison signal -- confirmed this
    picks "afraid" (appears in all four variants, 31 vs 40 in the real
    case, a 29% difference) over a less complete signal like "Americans"
    (present in only one variant).

    A real edition/arrangement difference tends to differ by MORE than a
    small fraction of the true repeat count (tolerance +-15%, minimum
    +-1, absorbs ordinary per-song noise like an intro/outro repeat some
    editions add or drop) -- confirmed this tolerance passes Ordinary
    Day's genuinely-matching candidate (its own most-repeated line count:
    4 vs 4) while rejecting Americans' (31 vs 40).

    Returns a human-readable rejection reason, or None if the candidate's
    repeat structure is consistent enough to trust (including whenever
    there's no repeated line in our own file to check at all).

    MOVED here from realign.py, 2026-08-11 (re-exported from there
    unchanged for backward compat -- `from .realign import
    check_repeat_structure` still works) -- `apply_lrc_timing_check`'s own
    tier-3 rescue gate (see `two_tier_time_calibration`) needed the exact
    same check and lrc_timing.py can't import FROM realign.py (realign.py
    already imports FROM lrc_timing.py -- would be circular), so this
    moved to the shared module instead of being duplicated a second time,
    matching this file's own established "factor out, don't reimplement"
    pattern (see `match_asr_to_lrc_lines`'s docstring)."""
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
    """Result of `reconcile_line_structure` -- see its own docstring."""
    lrc_lines: List[Tuple[float, str]]   # the candidate's own lines that
                                          # matched something in our_lines,
                                          # in original LRC order/timing --
                                          # everything else was dropped.
    n_our_lines: int
    n_matched: int
    n_lrc_dropped: int    # candidate lines with no match in our_lines
    n_our_unmatched: int  # our own lines with no match in the candidate
    match_ratio: float    # n_matched / n_our_lines


def reconcile_line_structure(
    our_lines: List[str],
    lrc_lines: List[Tuple[float, str]],
    max_skip: int = 8,
    min_match_ratio: float = 0.5,
) -> Optional[LineReconciliation]:
    """Reconciles an LRC candidate's own lines against OUR OWN file's
    lines when their REPEAT STRUCTURE doesn't match, instead of
    `check_repeat_structure`'s outright reject-the-whole-candidate
    response to the same situation (real confirmed case: David Bowie -
    "I'm Afraid of Americans", where an alternate edition's LRC has
    extra chorus repeats ours doesn't -- see `check_repeat_structure`'s
    own docstring). A real, LOCALIZED repeat-count difference gets
    resolved here (the extra lines on whichever side dropped) instead of
    discarding an otherwise-good candidate outright.

    User's own design (2026-08-14): walk both line sequences forward
    TOGETHER, one cursor per side, each only ever advancing forward --
    same cursor-based principle as `lyrics_lookup.
    assign_lrc_line_ids_sequentially` (there: ASR words vs. one LRC
    line's text; here: our own file's lines vs. the LRC candidate's own
    lines), so a repeated phrase later in either sequence can never be
    confused with an earlier occurrence -- by the time a later position
    is being resolved, the cursor has already moved past the earlier
    occurrence's own line.

    Algorithm: at each step, compare our_lines[i] to lrc_lines[j] (exact
    match after normalization, see `_normalize_line`). On a match, keep
    the LRC line and advance both cursors. On a mismatch, look ahead up
    to `max_skip` lines on BOTH sides for the next real match:
      - our_lines[i] found later in lrc_lines (at j+k) -> lrc_lines[j:j+k]
        are extras only the candidate has (a different edition's added
        chorus repeat, an extra bridge, etc.) -- drop them, advance j.
      - lrc_lines[j] found later in our_lines (at i+k) -> our_lines[i:i+k]
        have no LRC counterpart at all -- leave them unmatched (no LRC
        anchor for those lines), advance i.
      - Both found -> whichever needs the SMALLER skip wins (ties go to
        dropping the LRC side, the less-trusted external source) -- keeps
        the walk from jumping further than necessary on either side.
        Confirmed against the "Americans" case: after a run of genuine
        matches, our_lines[i] is "...the world" while lrc_lines[j] is 3
        lines into a run of extra "...Americans" repeats the candidate
        alone has -- both sides find their match exactly 3 lines ahead
        (tying), so the LRC side's 3 extra lines get dropped, landing
        both cursors back in sync at the shared "...the world" line.
      - Neither found within the window -> this LRC line has no
        plausible match nearby at all; drop it (advance j by 1) and keep
        walking, rather than aborting the whole reconciliation over one
        bad line.

    Returns None (caller should fall back to the old outright-rejection
    behavior) if the fraction of OUR OWN lines that found a real match
    falls below `min_match_ratio` -- a genuinely different
    recording/arrangement (not just a differing repeat count) should
    still show up as a low match rate here, not get silently patched
    together from whatever happens to align. Otherwise returns a
    `LineReconciliation` whose `lrc_lines` is the candidate's own lines
    filtered down to only the ones that matched something in our_lines --
    everything downstream (time calibration, per-word line assignment)
    should use THIS list instead of the candidate's raw, un-reconciled
    lines.
    """
    our_norm = [_normalize_line(t) for t in our_lines]
    lrc_norm = [_normalize_line(t) for _t, t in lrc_lines]
    n_i, n_j = len(our_norm), len(lrc_norm)

    i = j = 0
    kept: List[Tuple[float, str]] = []
    n_matched = 0
    while i < n_i and j < n_j:
        if our_norm[i] and our_norm[i] == lrc_norm[j]:
            kept.append(lrc_lines[j])
            n_matched += 1
            i += 1
            j += 1
            continue

        lrc_skip = next(
            (k for k in range(1, max_skip + 1)
             if j + k < n_j and our_norm[i] and our_norm[i] == lrc_norm[j + k]),
            None,
        )
        our_skip = next(
            (k for k in range(1, max_skip + 1)
             if i + k < n_i and lrc_norm[j] and our_norm[i + k] == lrc_norm[j]),
            None,
        )
        if lrc_skip is not None and (our_skip is None or lrc_skip <= our_skip):
            j += lrc_skip
        elif our_skip is not None:
            i += our_skip
        else:
            j += 1

    match_ratio = n_matched / n_i if n_i else 0.0
    if match_ratio < min_match_ratio:
        return None
    return LineReconciliation(
        lrc_lines=kept, n_our_lines=n_i, n_matched=n_matched,
        n_lrc_dropped=n_j - n_matched, n_our_unmatched=n_i - n_matched,
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
    # Corrected-time function, `correction_fn(raw_key_time) -> corrected_real_time`
    # -- populated for every successful kind, incl. constant/drift (as
    # `lambda t: t + offset + slope*t`, purely for API uniformity so a
    # caller never has to special-case "piecewise"/"isotonic", which have
    # no single global offset/slope to apply directly). None only when
    # calibration_offset is also None (nothing to calibrate with).
    correction_fn: Optional[Callable[[float], float]] = None
    calibration_confidence: float = 0.0
    # Diagnostic-only odd/even-anchor holdout residual (seconds), tier
    # 3 ("piecewise"/"isotonic") only -- see _holdout_residual_sec. None
    # for "constant"/"drift" (not computed there) or when uncalibrated.
    holdout_residual_sec: Optional[float] = None
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


def _filter_theilsen_inliers(
    candidates: List[Tuple[int, float, float]],
    fit: Tuple[float, float, float, int],
    outlier_tolerance_sec: float,
) -> List[Tuple[int, float, float]]:
    """Tier 3's own noise filter: keeps only candidates within
    `outlier_tolerance_sec` of tier 2's OWN Theil-Sen fit -- even a fit
    too imprecise to trust as a single global drift (that's WHY tier 3 is
    being tried at all) still separates real signal from raw text-
    matching noise / wrong-repeated-line-instance mismatches, since a
    genuine per-segment regime change only differs from the GLOBAL fit by
    the edit's own duration, not by an arbitrary amount the way a bad
    text match can. Deliberately looser than `LRC_TIMING_DRIFT_INLIER_
    TOLERANCE_SEC` (tier 2's own inlier bar) for exactly this reason --
    see config.py's own comment."""
    intercept, slope, _confidence, _n = fit
    return [c for c in candidates if abs(c[2] - (intercept + slope * c[1])) <= outlier_tolerance_sec]


def _enforce_monotonic_anchors(
    candidates: List[Tuple[int, float, float]],
) -> List[Tuple[float, float]]:
    """"piecewise" tier-3 strategy: greedily drops any candidate whose
    implied real time (lrc_start + delta) is EARLIER than the running max
    real time already kept -- a line literally can't be timed before an
    earlier line once corrected, so an anchor that would imply that is
    itself untrustworthy (more likely a residual bad text match than a
    real discontinuity). Processes candidates in lrc_start order and
    keeps the FIRST of any conflicting pair, consistent with how ASR/LRC
    matching elsewhere in this project always trusts earlier evidence
    over later when the two irreconcilably conflict.

    Returns (lrc_start, real_time) anchor pairs, sorted by lrc_start,
    ready for `_correction_from_anchors`."""
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
    """"isotonic" tier-3 strategy: Pool-Adjacent-Violators (PAVA),
    unweighted, O(n) amortized via a merge stack. Fits the candidates'
    implied real times (lrc_start + delta, in lrc_start order) as a
    monotonic NON-DECREASING step function minimizing squared error --
    the same monotonicity constraint `_enforce_monotonic_anchors` enforces
    by dropping violators, but PAVA instead POOLS (averages) a violating
    run with its predecessor rather than discarding it outright, so it
    doesn't need `_enforce_monotonic_anchors`'s own "keep the first,
    drop the rest" heuristic -- real per-song noise gets smoothed rather
    than arbitrarily thrown away, which is why the module docstring/
    config.py recommend trying this tier first.

    Returns one (mean_lrc_start, fitted_real_time) anchor per resulting
    pooled block, in lrc_start order -- every original point inside a
    block shares the same fitted value by construction (a flat segment
    there IS the correct fit, not a degenerate one), so one anchor per
    block is exactly the right granularity for `_correction_from_anchors`
    to interpolate between."""
    pts = sorted(((lrc_start, lrc_start + delta) for _key, lrc_start, delta in candidates),
                 key=lambda p: p[0])
    # Each block: [sum_x, sum_y, count, mean_y] -- mean_y is what
    # monotonicity is checked/pooled against; mean_x is recovered as
    # sum_x/count only when a block is finalized (below).
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
    """Builds `correction_fn(lrc_start) -> corrected_real_time` by linear
    interpolation between consecutive (lrc_start, real_time) anchors
    (already sorted, already monotonic -- both tier-3 strategies
    guarantee this before calling here), extrapolating past the first/
    last anchor using that boundary segment's own local slope (a flat
    isotonic boundary block naturally extrapolates as a flat/slope-0
    hold, which is the safe default there). The SAME interpolator serves
    both "piecewise" and "isotonic" -- see module docstring for why
    there's only one production code path for this step regardless of
    which strategy produced the anchors."""
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
    """Largest lrc_start gap between two CONSECUTIVE anchors -- see
    `LRC_TIMING_PIECEWISE_MAX_ANCHOR_GAP_SEC`'s own comment for why a
    single huge gap invalidates the whole tier-3 attempt rather than just
    that one segment."""
    if len(anchors) < 2:
        return 0.0
    return max(b[0] - a[0] for a, b in zip(anchors, anchors[1:]))


def _holdout_residual_sec(
    anchors: List[Tuple[float, float]],
    min_anchors: int = config.LRC_TIMING_HOLDOUT_MIN_ANCHORS,
) -> Optional[float]:
    """Odd/even-anchor holdout check: fits a correction on the ODD-indexed
    anchors only (`_correction_from_anchors`, the same interpolator tier 3
    uses for real), scores it against the EVEN-indexed anchors' own real
    times. Targets the general "genuine drift vs. fit to noise" question
    -- broader than the refine/rescue split below, which only looks at
    whether tier 1/2 independently found SOME support; this instead asks
    whether tier 3's own fit predicts anchors it didn't see, which a fit
    that's just tracking noise typically won't do as well as a fit
    tracking a real underlying shape. Cheap -- the anchor set already
    exists, this just re-runs the same interpolator on half of it.

    Returns the mean absolute holdout residual in seconds, or None if
    there aren't enough anchors on both sides to run this at all (real-
    tested 2026-08-11: on the Heroes wrong-recording case this did NOT
    by itself read as dramatically worse than a genuine-rescue case --
    see config.LRC_TIMING_HOLDOUT_MIN_ANCHORS's own comment -- so this is
    a diagnostic value, not currently a hard gate)."""
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
    """Tier 3: builds a piecewise correction for a DISCONTINUOUS drift
    neither tier 1 (constant) nor tier 2 (linear) can fit (see module
    docstring). Filters to Theil-Sen inliers, derives a monotonic anchor
    set via whichever strategy `drift_model` selects, gates on minimum
    anchor count + maximum adjacent-anchor gap, then builds a correction
    function by interpolation.

    Returns (offset, slope, confidence, kind, correction_fn,
    holdout_residual_sec) on success -- `offset`/`slope` are
    REPRESENTATIVE only (the first segment's own delta-at-start / local
    slope), kept so every existing "is this calibrated at all" check
    (`offset is not None`) still works uniformly across all three tiers;
    the REAL correction is always `correction_fn`. `confidence` is the
    fraction of the ORIGINAL candidates that survived the Theil-Sen
    inlier filter (consistent with tier 1/2's own "fraction that agree"
    semantics -- NOT the same as final anchor count, since "isotonic"
    pools inliers rather than dropping them further). `holdout_residual_
    sec` is a DIAGNOSTIC value only (see `_holdout_residual_sec`), never
    gates acceptance here -- the caller (`two_tier_time_calibration`)
    owns the actual accept/reject decision, including the refine-vs-
    rescue structural-check gate.
    Returns None if the min-anchor/max-gap GATES aren't cleared -- caller
    falls through to reporting no confident calibration, same as if
    tier 3 didn't exist.
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

    # Representative offset/slope from the FIRST segment, purely for
    # display/backward-compat with code that inspects offset/slope
    # directly instead of calling correction_fn -- see docstring above.
    # Solved so `t + rep_offset + rep_slope*t` reproduces the first
    # segment's own two anchors exactly (the same `t + offset + slope*t`
    # convention every other call site already uses).
    x0, y0 = anchors[0]
    if len(anchors) >= 2 and anchors[1][0] != x0:
        x1, y1 = anchors[1]
        rep_slope = (y1 - y0) / (x1 - x0) - 1.0
    else:
        rep_slope = 0.0
    rep_offset = y0 - x0 - rep_slope * x0

    kind = "isotonic" if drift_model == "isotonic" else "piecewise"
    return rep_offset, rep_slope, confidence, kind, correction_fn, holdout_residual_sec


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
    drift_model: str = config.LRC_TIMING_DRIFT_MODEL,
    piecewise_outlier_tolerance_sec: float = config.LRC_TIMING_PIECEWISE_OUTLIER_TOLERANCE_SEC,
    piecewise_min_anchors: int = config.LRC_TIMING_PIECEWISE_MIN_ANCHORS,
    piecewise_max_anchor_gap_sec: float = config.LRC_TIMING_PIECEWISE_MAX_ANCHOR_GAP_SEC,
    rescue_min_prior_confidence: float = config.LRC_TIMING_RESCUE_MIN_PRIOR_CONFIDENCE,
    structural_check: Optional[Callable[[], Optional[str]]] = None,
) -> Tuple[Optional[float], float, float, Optional[str], Optional[str],
           Optional[Callable[[float], float]], Optional[float]]:
    """Shared (despite the name -- see below) THREE-tier time
    calibration: given (key, lrc_start, delta) candidates, tries a single
    constant offset (mode of deltas at 1s resolution); if that isn't
    confident enough, a robust (Theil-Sen) offset+slope fit tolerant of a
    real per-song LINEAR timing drift; if THAT isn't confident enough
    either, a piecewise/isotonic correction tolerant of a DISCONTINUOUS
    drift (a real edit difference between recordings -- see module
    docstring for why all three tiers exist).

    Name kept as `two_tier_time_calibration` despite now trying three --
    every call site already imports it by this name, and the function's
    OWN job (calibrate away an LRC-vs-our-audio time mismatch, trying
    progressively more flexible models) hasn't changed, just how far it's
    willing to go before giving up.

    Returns (offset, slope, confidence, kind, skipped_reason,
    correction_fn, holdout_residual_sec). `offset` is None (with
    skipped_reason set, kind/correction_fn/holdout_residual_sec None) if
    no confident calibration could be established by ANY tier.
    `correction_fn(raw_key_time) -> corrected_real_time` is populated for
    every successful kind -- including "constant"/"drift", as `lambda t:
    t + offset + slope*t` -- so every caller can apply the calibration the
    SAME way regardless of which tier produced it, without special-casing
    "piecewise"/"isotonic" (which have no single global offset/slope to
    apply directly; `offset`/`slope` are only representative there, see
    `_piecewise_or_isotonic_calibration`). `holdout_residual_sec` is
    always None for "constant"/"drift" (not computed there -- see
    `_holdout_residual_sec`); a DIAGNOSTIC value for "piecewise"/
    "isotonic", never itself a gate.

    TIER 3's REFINE-VS-RESCUE SPLIT (added 2026-08-11 after a real
    validation finding -- see module docstring and config.
    LRC_TIMING_RESCUE_MIN_PRIOR_CONFIDENCE's own comment for the full
    story): a stricter residual threshold on tier 3 alone can't tell
    "genuinely complex real drift" from "overfit to a wrong recording" --
    both look like a flexible model finding SOME fit; the distinguishing
    signal has to come from somewhere INDEPENDENT of tier 3's own
    residuals. That signal is whether tiers 1/2 -- two RIGID models,
    incapable of overfitting the way tier 3 can -- already found SOME
    real support before being rejected for insufficient COVERAGE (not
    insufficient AGREEMENT where they did apply):
      - "refine" (`max(tier1_confidence, tier2_confidence) >=
        rescue_min_prior_confidence`): two independent rigid models
        already partially agree with this candidate -- tier 3 is
        sharpening a shape they were already circling. Proceeds exactly
        as before, no structural check needed.
      - "rescue" (neither rigid model cleared the floor): tier 3 would be
        supplying both the hypothesis (a discontinuous drift exists) AND
        its own validation (a good-looking fit to it) with zero
        independent support. Requires `structural_check` to be given AND
        to return None (pass) before being accepted; declines (falls
        through to "uncalibrated", same as if tier 3 didn't exist)
        otherwise -- including when `structural_check` isn't provided at
        all, which is the SAFE default for a caller that hasn't wired one
        up yet (e.g. `compute_gap_calibration`, which has no separate
        "candidate identity" to check in the first place -- see its own
        docstring).

    Factored out of `apply_lrc_timing_check` so `mxl_lrc_generator.py`
    and `realign.py` can reuse the exact same technique to calibrate away
    a systematic offset between LRC line timestamps (or, for `realign.py`'s
    GAP calibration, an existing file's own original timing) and OUR OWN
    audio BEFORE trusting those timestamps as placement anchors, rather
    than only diagnosing the mismatch after the fact -- don't reimplement
    this a third time."""
    if len(candidates) < min_calibration_samples:
        return None, 0.0, 0.0, None, (
            f"only {len(candidates)} matched line(s) (< {min_calibration_samples} required) -- "
            f"not enough to trust a calibration offset"
        ), None, None

    # Tier 1: constant offset -- mode at coarse (1s) resolution, since
    # line-level timestamps are far less precise than word-level, so a
    # fine bucket would just split a real single cluster across several
    # adjacent buckets.
    BUCKET_SEC = 1.0
    bucket_counts = Counter(round(delta / BUCKET_SEC) for _, _, delta in candidates)
    best_bucket, n_agree = bucket_counts.most_common(1)[0]
    offset, slope, confidence, kind = best_bucket * BUCKET_SEC, 0.0, n_agree / len(candidates), "constant"
    tier1_confidence = confidence

    if confidence < min_calibration_confidence:
        # Tier 2: a real per-song drift, not just noise around one
        # constant -- confirmed on real audio (stars, tarzan,
        # little_mermaid), see module docstring. Stricter gate than tier
        # 1: a 2-parameter fit can trivially match a handful of points
        # exactly, so this needs both more samples and a higher inlier
        # fraction before it's trusted.
        fit = _robust_linear_fit(candidates, drift_inlier_tolerance_sec) if len(candidates) >= min_drift_samples else None
        tier2_confidence = fit[2] if fit is not None else 0.0
        if fit is None or fit[2] < min_drift_confidence:
            # Tier 3: a DISCONTINUOUS drift (real edit difference between
            # recordings) neither tier above can fit -- see module
            # docstring. Needs at least a rough Theil-Sen fit to use as
            # its own noise filter (`_filter_theilsen_inliers`), so this
            # is unreachable when `fit is None` (fewer than 2 distinct
            # lrc_start values -- nothing to filter against either way).
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
                # Rescue case: neither rigid model found independent
                # support -- require a structural check to pass before
                # trusting tier 3's own fit at all (see docstring above).
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
                ) + f", and piecewise/isotonic tier 3 also failed its own anchor-count/spacing gate"
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
    """Aligns pass-3's own lines (grouped by Syllable.line_id) against
    LRCLIB's synced-lyrics lines by TEXT (word-level whole-sequence
    match, see `_match_lines_word_level`), calibrates a per-song time
    offset, then flags any line whose delta from the calibrated
    expectation exceeds flag_tolerance_sec.

    Calibration is three-tiered (see module docstring): first tries a
    single constant offset (mode of per-line deltas at 1s resolution --
    the original, higher-precision technique); if that isn't confident
    enough, a robust offset+slope fit tolerant of a real per-song LINEAR
    timing drift; if that isn't either, a piecewise/isotonic correction
    tolerant of a DISCONTINUOUS drift (a real edit difference between
    recordings). Whichever tier succeeds is what lines get flagged
    against; if none does, this returns with skipped_reason set and no
    flags.

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

    # Tier 3's "rescue" case (see two_tier_time_calibration's own
    # docstring) needs independent structural evidence before trusting a
    # fit built on candidates tier 1/2 found no real support for at all --
    # `our_lines`'s own reconstructed line text (already have it above)
    # vs. the LRC candidate's own line text is exactly what
    # `check_repeat_structure` compares, so build the check straight from
    # data already in hand here rather than needing a caller to pass one.
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
