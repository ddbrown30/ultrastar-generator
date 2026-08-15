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
from .text_normalize import normalize_word as _normalize

_LRC_TAG_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def _normalize_line(text: str) -> str:
    return " ".join(n for n in (_normalize(tok) for tok in text.split()) if n)


def lrc_line_window(lrc_lines: List[Tuple[float, str]], li: int) -> Tuple[float, float]:
    """The real-time window [line's own start, next line's own start) an
    LRC line's own words are expected to fall within -- the LAST line has
    no next line to bound it, so it gets a flat +5.0s fallback width
    instead. Shared by every mechanism that needs to bound a per-LRC-line
    search/estimate by real time (`mxl_lrc_generator.place_words_via_asr`,
    `realign.match_words_to_asr_windowed`, and `reconcile_line_structure`
    above, all identical 3-line copies before this was extracted
    2026-08-16)."""
    t0 = lrc_lines[li][0]
    t1 = lrc_lines[li + 1][0] if li + 1 < len(lrc_lines) else t0 + 5.0
    return t0, t1


def words_in_time_window(words: List[Word], t0: float, t1: float, slack: float = 0.5) -> List[Word]:
    """Every `Word` whose own start falls in `[t0 - slack, t1 + slack]`,
    order preserved. Trivial (a single filtered list comprehension), but
    was 3 separately-written copies (`mxl_lrc_generator.
    place_words_via_asr`, `realign.match_words_to_asr_windowed`, `realign.
    rematch_local_gaps`) before this was extracted 2026-08-16 -- kept as
    its own function so all 3 stay in sync if the slop convention ever
    changes, not because the logic itself is complex."""
    return [w for w in words if t0 - slack <= w.start <= t1 + slack]


def match_block_to_candidates(
    target_norm: List[str],
    candidate_words: List[Word],
    *,
    fuzzy_min_ratio: Optional[float] = None,
    min_candidate_confidence: Optional[float] = None,
) -> Dict[int, Word]:
    """Matches a target block's own normalized words against a caller-
    bounded candidate `Word` list (already narrowed by time window or
    proximity -- this function has no time/window concept of its own) via
    ONE whole-block `difflib.SequenceMatcher` call: an "equal" opcode
    matches directly (gated on the candidate's own confidence, so a
    low-confidence ASR word never anchors anything); a "replace" opcode
    where the TARGET side is exactly one token tries every candidate
    token in that slice for the best fuzzy-ratio match (real, necessary
    case: ASR mishears a word differently than its true text -- e.g.
    "favors" transcribed as "favorites" -- independent of any upstream
    OCR/text issue), gated on BOTH the fuzzy ratio and the candidate's own
    confidence. Returns `{target_local_index: matched Word}` -- keyed by
    the TARGET's own local index (not the candidate's), so a caller maps
    local -> global however it needs to (a contiguous offset, or an
    explicit non-contiguous index list) -- that mapping is caller-
    specific and deliberately not part of this function's job.

    Extracted 2026-08-16: `mxl_lrc_generator.place_words_via_asr`'s own
    Pass 1, `realign.match_words_to_asr_windowed`, and `realign.
    rematch_local_gaps` had all independently implemented this exact
    same opcode-walk (confirmed by `match_words_to_asr_windowed`'s own
    docstring: "mirroring mxl_lrc_generator.place_words_via_asr's Pass 1
    exactly"). Deliberately NOT shared with `realign.match_words_to_asr`'s
    own `_apply_match` -- that one uses a REVERSED SequenceMatcher
    argument order (candidate first, target second, since it's built on
    top of the caller-agnostic `find_cursor_window_match` above, which
    searches a haystack for a target) and returns a candidate-side offset
    for the caller's own forward cursor to advance by, neither of which
    apply to these 3 callers (each an independent, non-cursored, per-
    block/per-line match)."""
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


# Ad-lib/filler and connector words that different transcribers (an LRC's
# own author vs. whoever wrote our existing .txt) commonly add or drop
# without it meaning the LINE is actually different content -- user's own
# explicit request, 2026-08-15, specifically motivated by realignment:
# "the choice of filler words may differ between the author of the LRC and
# the author of the usdx file." Deliberately a short, conservative list
# (ad-libs + a few connector words the user named directly) rather than a
# broad stopword list -- a real content word incorrectly on this list
# would silently make two genuinely different lines match.
FILLER_WORDS = frozenset({
    "ooh", "ooo", "oh", "ohh", "mmm", "mm",
    "yeah", "and", "but",
})


def _strip_filler_flat(normalized_line: str) -> str:
    """Whitespace-flattened form of `normalized_line` (itself the output
    of `_normalize_line`) with any whole FILLER_WORDS token removed --
    used as a FALLBACK comparison alongside the raw flattened form (see
    `reconcile_line_structure`), never in place of it, so two lines still
    have to be identical content-wise once ad-libs/connectors are set
    aside."""
    return "".join(tok for tok in normalized_line.split() if tok not in FILLER_WORDS)


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
    our_line_index: List[int]            # parallel to lrc_lines -- which
                                          # our_lines[i] each entry came
                                          # from (a merge split contributes
                                          # several consecutive entries,
                                          # each its own our_lines index).
                                          # This is the whole reason this
                                          # dataclass exists rather than
                                          # just returning lrc_lines alone:
                                          # a caller with its own per-word
                                          # line membership (e.g. realign.
                                          # py's prepare_lrc) can build a
                                          # word-to-LRC-line mapping
                                          # DIRECTLY from this -- reusing
                                          # the correspondence this walk
                                          # already established at the
                                          # (more reliable) LINE level --
                                          # instead of re-deriving it via
                                          # a separate whole-song WORD-
                                          # level diff (assign_words_to_
                                          # lines) that has no line-
                                          # boundary information at all
                                          # and can disagree with this
                                          # walk's own answer. Real
                                          # confirmed case (2026-08-15,
                                          # Trixie Mattel - Video Games):
                                          # this walk correctly resolved
                                          # a 3x-repeated chorus, but
                                          # assign_words_to_lines's own
                                          # independent word-level diff
                                          # still placed the third
                                          # repeat ~30s off the real
                                          # audio -- exactly the
                                          # "repeated-phrase disambiguation"
                                          # failure class this project has
                                          # been burned by before.
    n_our_lines: int
    n_matched: int
    n_lrc_dropped: int    # candidate lines with no match in our_lines
    n_our_unmatched: int  # our own lines with no match in the candidate
    match_ratio: float    # n_matched / n_our_lines


FUZZY_LINE_MIN_RATIO = 0.85  # see _flat_fuzzy_equal


def _flat_fuzzy_equal(a: str, b: str, min_ratio: float = FUZZY_LINE_MIN_RATIO) -> bool:
    """Whether two whitespace-flattened normalized strings are the SAME
    text modulo a common lyric-transcription variant -- a dropped/elided
    letter marked with an apostrophe (real cases, user's own request,
    2026-08-15: "Ev'rything" for "Everything", "livin'" for "living") --
    not a genuinely different word. Exact equality is checked first and
    is always preferred; this is only ever a FALLBACK for a candidate
    pair the exact-structure walk already identified as the position to
    compare (a fixed (i, j) or a merge-piece slice), never used to WIDEN
    a search across many competing positions -- so this doesn't reopen
    the "repeated-phrase disambiguation" risk class other fuzzy-matching
    attempts in this codebase were rejected over (see CLAUDE.md): it's
    comparing two already-fixed candidates, not picking one out of many.
    `min_ratio` is deliberately high (character-level, not word-level) --
    this should catch a single dropped/added letter on an otherwise
    long, exact line, not a genuinely different line that merely shares
    some words."""
    if a == b:
        return True
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= min_ratio


def _consume_as_merge(target_flat: str, piece_flats: List[str]) -> Optional[int]:
    """Whether `target_flat` is the back-to-back concatenation of
    `piece_flats[0]`, `[1]`, ... in order, ALLOWING `_flat_fuzzy_equal`'s
    fallback tolerance per piece (not just exact equality) -- each
    piece's OWN length still determines exactly how many characters of
    `target_flat` it's compared against (a length-preserving letter
    SUBSTITUTION, like both real cases above, keeps piece boundaries
    aligned; this is not a general fuzzy alignment that could re-shuffle
    boundaries on an insertion/deletion). Both sides are whitespace-
    FLATTENED normalized text (see the call site, not just
    `_normalize_line`'s single-space-joined form) -- deliberately, so a
    stray missing/extra space at a line boundary (real confirmed case,
    2026-08-15: our own file's own typo "livin'if" for "livin' if")
    can't make an otherwise byte-identical line fail to match. Returns
    how many pieces were consumed (>=2, since exactly 1 piece is just a
    normal single-line match, not a merge) once `target_flat` is used up
    EXACTLY (by length), or None if no such split exists."""
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

    Before any of the above, a mismatch is first checked against a MERGE
    in either direction (user's own design, 2026-08-15, real motivating
    case: Trixie Mattel - "Video Games", where the candidate writes "It's
    you, it's you, it's all for you," + "Everything I do" -- two of our
    own lines -- as one combined LRC line) via `_consume_as_merge`: does
    lrc_lines[j]'s own text EXACTLY equal our_lines[i], [i+1], ...
    concatenated word-for-word (up to `max_merge_lines` pieces), or does
    our_lines[i] EXACTLY equal lrc_lines[j], [j+1], ... concatenated the
    same way? This is intentionally an EXACT-consumption check, not a
    fuzzy one -- same "never guess across a real ambiguity" caution as
    the rest of this module.
      - LRC merged K of our lines into one: split lrc_lines[j]'s single
        real timestamp across K synthetic entries (own text = each of
        our_lines[i:i+K], own time = proportionally interpolated by word
        count across [lrc_lines[j]'s own time, lrc_lines[j+1]'s own time)
        -- the first piece keeps the EXACT real timestamp, later pieces
        are estimates bounded within that real window, never claiming
        precision the data doesn't have). Necessary, not cosmetic:
        downstream `assign_words_to_lines` buckets every one of our own
        words to whichever ONE lrc_lines entry its own whole-song word-
        diff lands on, so without a distinct entry per piece, only the
        FIRST of the K lines could ever get a real anchor -- the rest
        would silently stay anchor-less, exactly the original problem.
      - Our own line was written as K separate LRC lines: use
        lrc_lines[j]'s own (first piece's, already exactly real, no
        interpolation needed) timestamp directly for our_lines[i]'s one
        entry, then advance j past all K real lrc lines consumed.

    Returns None (caller should fall back to the old outright-rejection
    behavior) if the fraction of OUR OWN lines that found a real match
    falls below `min_match_ratio` -- a genuinely different
    recording/arrangement (not just a differing repeat count) should
    still show up as a low match rate here, not get silently patched
    together from whatever happens to align. Otherwise returns a
    `LineReconciliation` whose `lrc_lines` is the candidate's own lines
    (or, for a merge, synthetic per-our-line entries derived from them --
    see above) filtered down to only the ones that matched something in
    our_lines -- everything downstream (time calibration, per-word line
    assignment) should use THIS list instead of the candidate's raw,
    un-reconciled lines.

    On a mismatch, the recovery search is a single JOINT, merge-aware
    search over (p, q) offsets from the current (i, j) -- not two
    independent single-axis searches (real bug found via real-audio
    validation, 2026-08-15, the SAME Video Games case above): a stretch
    where BOTH sides have real, genuinely DIFFERENT content before the
    next shared line (confirmed real case: our own file's own repeated
    ad-lib "But baby, now you do" / "Mm." has no exact counterpart in the
    candidate's own differently-worded "...ooh" / "...mmm" lines just
    before that same third chorus repeat) needs BOTH cursors to move
    together to find the next real correspondence -- a single-axis search
    (only ever drop from the LRC side, or only ever drop from OUR side,
    never both) can never reach it, and worse, blindly advancing one
    cursor while holding the other still can walk straight past a real,
    perfectly matchable line (the chorus itself) as collateral damage
    before ever finding a resync point. The joint search also checks for
    a MERGE (not just an exact single-line match) at every candidate
    (p, q) -- not only at the untouched, un-advanced (i, j) -- since the
    real resync point is frequently a merge, not a plain line.
    Increasing total offset (p+q) is tried in order, smallest first
    (ties prefer growing q over p, i.e. dropping from the LRC side
    first, the less-trusted external source, same tie-break as before);
    within the `max_skip` bound on EACH axis independently. If nothing
    aligns anywhere in that whole window, the current LRC line has no
    plausible match nearby at all -- drop it (advance j by 1) and keep
    walking, rather than aborting the whole reconciliation over one line.

    Both the plain-match check and the merge check ALSO try a FILLER-
    WORD-TOLERANT fallback (`FILLER_WORDS`/`_strip_filler_flat`, user's
    own explicit request, 2026-08-15) when the raw comparison fails: an
    LRC's own author and our existing file's own author often choose
    differently whether to write an ad-lib ("ooh"/"mmm"/"ooh"/"ohh") or a
    filler connector ("yeah"/"and"/"but") that the singer may or may not
    audibly include -- that's a transcription CHOICE, not the line
    actually being different content, and is specifically common in
    realignment (comparing our own file's lines against an independently
    -authored LRC candidate). Guarded so an all-filler line (e.g. "Ooh
    ooh ooh") can never fuzzy-match a different all-filler line via both
    sides stripping down to an empty string.
    """
    our_norm = [_normalize_line(t) for t in our_lines]
    lrc_norm = [_normalize_line(t) for _t, t in lrc_lines]
    # Equality/merge-consumption comparisons all use the FLATTENED (no
    # inter-word space) form, not our_norm/lrc_norm directly -- real bug
    # found via real-audio validation (2026-08-15, Trixie Mattel - Video
    # Games): our own file had a genuine typo ("livin'if" for "livin' if",
    # a missing space), which made an otherwise byte-identical line fail
    # to match at all (different token count/boundaries), desyncing the
    # whole walk from that point on with no recovery within max_skip.
    # Flattening removes stray/missing-space differences as a possible
    # cause of a false mismatch while still requiring an EXACT character
    # sequence match -- no fuzzy tolerance for a real word difference.
    our_flat = ["".join(t.split()) for t in our_norm]
    lrc_flat = ["".join(t.split()) for t in lrc_norm]
    # Filler-word-stripped parallel versions -- FALLBACK ONLY (see
    # _match_kind below), never used in place of the raw flat form. User's
    # own explicit request, 2026-08-15, motivated by realignment: an LRC's
    # own author and our existing file's own author may each choose
    # differently whether to write in an ad-lib ("ooh"/"mmm") or a filler
    # connector ("yeah"/"and"/"but") -- that's not the line actually being
    # different content, so a line that matches once those words are set
    # aside on both sides should still count as a real match.
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
        # Filler-word-tolerant fallback -- only once both sides still have
        # SOME real content left after stripping (an all-filler line, e.g.
        # "Ooh ooh ooh", must never fuzzy-match another all-filler line
        # with genuinely different filler words via an empty-string tie).
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
    """The forward-only CURSOR-based window search shared by every "find
    `target_tokens` somewhere after `cursor` in `haystack_norm`" mechanism
    in this codebase (`match_asr_to_lrc_lines` below, `realign.
    match_words_to_asr`) -- extracted 2026-08-16 specifically because the
    SAME three real bugs kept needing independent re-discovery and
    re-fixing in each caller's own copy of this logic (user's own
    observation, after the second time it happened): a fix landing in
    one caller did nothing for the other until someone noticed the same
    failure mode there too. This is now the ONE place this logic lives.

    Only finds and validates a window match -- returns the winning
    window's own `(opcodes, window)` (raw `difflib.SequenceMatcher`
    opcodes, `None` if neither window confirms), NOT what to DO with a
    match. Interpreting opcodes (marking matched positions, handling a
    `"replace"` block as a possible fuzzy ASR-mishearing match, computing
    how far to advance the caller's OWN cursor) stays caller-specific,
    since callers want genuinely different things from a match (a single
    per-line anchor vs. every individual word's own position) -- forcing
    that into a shared shape would be the over-abstraction this project's
    conventions warn against, for no real benefit (that part has never
    been where the repeated bugs lived).

    TIGHT-preferred-when-SUBSTANTIAL, WIDE otherwise (real bug found via
    real-audio validation on Our Lady Peace - "Somewhere Out There", a
    song with several lines repeating back-to-back 3-4 times in a row):
    even a completely FRESH search's base (wide) window is already wide
    enough to reach PAST the immediately-following correct occurrence
    into a LATER repeat of the same content, with no prior skip needed to
    get there at all -- a `difflib` search doesn't inherently prefer the
    NEAREST valid match over any other high-quality one it finds
    elsewhere in the window. Both a TIGHT window (barely more than
    `target_tokens`' own size) and the normal WIDE window are tried;
    TIGHT wins whenever it covers at least HALF `target_tokens`
    (`min_match_fraction`), regardless of what the wide search separately
    finds -- a near, mostly-complete match is trusted on its own local
    merits, not compared against whatever the wide window's own
    (possibly wrong, possibly further-away) optimization happens to
    prefer. Two thresholds were tried and rejected first, both found
    wanting against real data from TWO different real songs: requiring a
    fully COMPLETE tight match was too strict (a real correct nearby
    match can be missing 1-2 words to real ASR noise, and would still
    lose to a wrong, farther-away wide match); comparing raw match COUNTS
    (prefer whichever side found more) was too unprincipled -- a
    wrong-but-farther wide match can trivially have a higher raw count
    than a correct-but-partial near one simply because it has more room
    to search in. A REQUIRE-ANY-PASSING-TIGHT-MATCH threshold (tried even
    earlier) was too loose the other direction -- it regressed a
    different real song (Chicago), where a single coincidentally-found
    word in the tight window blocked the wide search from finding its
    own better, fuller answer.

    WIDE window is NOT fixed-size -- `pending_word_count` (the caller's
    own running total of skipped/unmatched target tokens since the
    cursor last actually advanced, capped by the caller) lets a real
    multi-target garbled/ad-lib stretch recover once real content
    resumes, without permanently stranding the cursor.

    Both windows share the SAME two quality guards (found via real-audio
    validation on Chappell Roan - "Pink Pony Club", a song whose chorus
    repeats 3 full times): a MINIMUM-MATCH-COUNT gate (`min_needed`,
    floored at `lenient_min_matches` even outside strict/fraction mode --
    see its own comment in `_try` below for why this floor is caller-
    tunable, not one universal constant: `match_asr_to_lrc_lines`'s own
    `mall_no_recovery` test needs 2, `realign.match_words_to_asr`'s own
    `sla_confident` test needs 1) rejects a coincidentally-shared common
    word being mistaken for a real match; a SPAN guard (bounded relative
    to the EVIDENCE actually found, not the window's own nominal size)
    rejects a technically-passing match whose few real tokens are
    scattered across almost the entire window -- letting such a match
    through once let the caller's own cursor jump all the way to the
    window's far edge, permanently stranding every later search for the
    rest of the song."""
    n = len(haystack_norm)

    def _try(window: List[str], require_strict_fraction: bool) -> Tuple[Optional[List], int]:
        if not window:
            return None, 0
        sm = difflib.SequenceMatcher(None, window, target_tokens, autojunk=False)
        opcodes = sm.get_opcodes()
        equal_wa = [(wa1, wa2) for tag, wa1, wa2, _ca1, _ca2 in opcodes if tag == "equal"]
        matched = sum(wa2 - wa1 for wa1, wa2 in equal_wa)
        # A genuine 1-word target has no way to require "half" of itself
        # (or "at least `lenient_min_matches`") -- accept its single token
        # either way. Anything longer needs at least `lenient_min_matches`
        # real matched tokens even in lenient (non-strict) mode -- real bug
        # found consolidating this logic (2026-08-16): an unconditional
        # 1-token floor let a single coincidentally-shared word validate a
        # multi-word target on its own, the exact risk the fraction gate
        # exists to prevent (caught by `match_asr_to_lrc_lines`'s own
        # `mall_no_recovery` regression test: a lone stray "the" must not
        # validate the 2-word target "the end"). `lenient_min_matches`
        # itself is caller-tunable, not one universal constant -- a real,
        # deliberately DIFFERENT threshold is needed by `realign.
        # match_words_to_asr`'s own pre-existing `sla_confident` regression
        # test: a single real match (e.g. "alpha") is the ONLY confident
        # evidence in the whole remaining ASR stream for a 4-word target,
        # and must still anchor -- real per-word transcripts routinely
        # confirm only a few of a chunk's words at a time, which isn't a
        # coincidence risk the way a whole extra unmatched LRC line is.
        # Strict mode additionally requires the full fraction on top, when
        # that's a higher bar than the floor.
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
    time.

    The window search itself (forward-only cursor, tight-vs-wide, quality
    gates) is `find_cursor_window_match` above -- see its own docstring
    for the full real-bug history. This function's only remaining job is
    interpreting a winning match for THIS shape: the EARLIEST matched ASR
    word's own start time becomes the line's anchor candidate, and the
    cursor advances to just past the LAST matched word (never a raw,
    unconfirmed opcode boundary)."""
    MAX_PENDING_WORDS = 60

    asr_norm = [_normalize(w.text) for w in asr_words]
    cursor = 0
    pending_word_count = 0
    candidates = []
    for li, (lrc_start, text) in enumerate(lrc_lines):
        line_tokens = [t for t in (_normalize(tok) for tok in text.split()) if t]
        if not line_tokens:
            continue
        pending_word_count = min(pending_word_count + len(line_tokens), MAX_PENDING_WORDS)
        found = find_cursor_window_match(cursor, asr_norm, line_tokens, pending_word_count)
        if found is None:
            # No match (or only a weak, likely-coincidental one) anywhere
            # in this (accumulated) window -- don't advance the cursor;
            # the NEXT line's own search starts from the same position,
            # its window growing to cover this line's skipped content too.
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
