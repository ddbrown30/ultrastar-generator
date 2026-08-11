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
from typing import Callable, List, Optional, Tuple

from . import config
from .lyrics_lookup import LrcLibCandidate, search_lrclib
from .lrc_timing import parse_lrc, two_tier_time_calibration, match_asr_to_lrc_lines
from .models import Syllable, Word
from .syllables import hyphenate, chunk_to_count


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
        if n.isChord:
            continue
        if not n.lyrics:
            # A note with no lyric at all -- if a word is currently being
            # built AND this note starts exactly where the last syllable's
            # note ends (no rest in between), it's a real continuation of
            # that word (a tied hold, or a slurred pitch change within the
            # same syllable, e.g. a slide), NOT silence to discard.
            # Confirmed real case: MXL notes encode a slide + fermata as an
            # untexted note tied/slurred immediately onto the previous
            # lyric-bearing one -- dropping it here used to lose both the
            # true held duration (tied, same pitch) and the real sung pitch
            # movement (slurred, different pitch) entirely.
            #
            # The contiguity check matters: a REST (rests aren't visited at
            # all here -- `.notes` excludes them) can separate the last
            # syllable from an unrelated later note that also happens to
            # carry no lyric (e.g. an instrumental/vocalise passage before
            # the next real word) -- confirmed real case in this same file,
            # ~8 quarter notes after "reciprocity"'s own fermata note, two
            # such notes would otherwise get wrongly glued onto that word.
            if cur_syllables:
                off, dur, prev_midi, text = cur_syllables[-1]
                if abs(float(n.offset) - (off + dur)) < 1e-6:
                    midi = int(n.pitch.midi)
                    if n.tie is not None and midi == prev_midi:
                        cur_syllables[-1] = (off, dur + float(n.quarterLength), prev_midi, text)
                    else:
                        cur_syllables.append((float(n.offset), float(n.quarterLength), midi, ""))
                # else: a real rest separates this note from the word in
                # progress -- not a continuation, leave it unattached.
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


def _merge_words_to_count(words: List[str], n_chunks: int) -> List[str]:
    """Word-level analogue of `syllables.chunk_to_count` -- merges a word
    list down to exactly n_chunks contiguous chunks, joined with a space
    (chunk_to_count itself joins with "", right for syllable fragments of
    one word, wrong for merging separate words back together)."""
    n_chunks = max(1, n_chunks)
    if n_chunks >= len(words):
        return list(words)
    chunks = []
    base = len(words) // n_chunks
    extra = len(words) % n_chunks
    idx = 0
    for c in range(n_chunks):
        take = base + (1 if c < extra else 0)
        take = max(1, take)
        chunks.append(" ".join(words[idx:idx + take]))
        idx += take
    return chunks


def _distribute_words_to_slots(lrc_words_raw: List[str], n_slots: int) -> List[str]:
    """Distributes a recovered stretch of raw LRC words across n_slots MXL
    word display slots -- OCR word-segmentation doesn't always match the
    real lyric's own word boundaries (see `assign_words_to_lines`'s own
    docstring for real confirmed cases). Always returns exactly n_slots
    items, in order:
      - Fewer real words than slots: first tries splitting any hyphenated
        word into its own pieces (real case: MXL notated "double"/"edged"
        as two separate words, but LRC's own line joins them into one
        "double-edged" token) -- if that still isn't enough, pads the
        remainder with `config.MELISMA_CONTINUATION_TEXT` (same
        convention `_text_for_mxl_syllables` already uses for a word with
        fewer syllables than notated).
      - More real words than slots: merges adjacent words evenly
        (`_merge_words_to_count`).
      - Equal counts: direct positional 1:1 assignment."""
    words = list(lrc_words_raw)
    if len(words) < n_slots:
        expanded = []
        for w in words:
            pieces = [p for p in w.split("-") if p] if "-" in w else None
            expanded.extend(pieces if pieces else [w])
        words = expanded
    if len(words) > n_slots:
        words = _merge_words_to_count(words, n_slots)
    elif len(words) < n_slots:
        words = words + [config.MELISMA_CONTINUATION_TEXT] * (n_slots - len(words))
    return words


def assign_words_to_lines(mxl_words: List[MxlWord],
                           lrc_lines: List[Tuple[float, str]]) -> Tuple[List[int], List[Optional[str]]]:
    """Assigns each MXL word to an LRC line index via word-level
    whole-sequence matching (order-preserving, resistant to picking a
    wrong repeated-phrase instance the same way this project's other
    whole-sequence alignments are). Words that don't directly match any
    LRC token (OCR-garbled MXL text, minor wording differences) inherit
    the nearest PRECEDING confirmed match's line -- falling back to the
    first confirmed line for any words before the first match.

    ALSO returns, per word, a clean-text replacement for the DISPLAYED
    lyric text (`None` if none was found) -- used by `build_syllables`.
    Real, confirmed bug this fixes: the MXL's own OCR'd syllable text
    ("MATIZON" for "Matron", "systern"/"eystern" for "system") was being
    used verbatim in the output; MXL should only ever supply pitch and
    relative rhythm, never displayed text, when a clean source is
    available.

    Three ways a word earns a clean-text replacement, all from the SAME
    single whole-sequence alignment (never an independent text search --
    a block's position is always fixed by its own real neighbors in this
    one ordered comparison, so a common short word like "oh"/"you" can't
    accidentally pick up a different, wrong occurrence elsewhere in the
    song):
      - An exact normalized match (difflib "equal" opcode) -- the common
        case.
      - A "replace" opcode pairing exactly ONE MXL word against exactly
        ONE LRC word (i.e. difflib's own whole-sequence alignment already
        decided these are the best positional fit, anchored by correctly-
        matched words on both sides) AND their character-level similarity
        clears `config.MXL_LRC_FUZZY_TEXT_MIN_RATIO` -- catches an MXL
        word that's the SAME word with an OCR/spelling difference (e.g.
        "systern" for "system") rather than genuinely missing.
      - A "replace" opcode covering a MULTI-word block (either side, or
        both) up to `config.MXL_LRC_BLOCK_MAX_WORDS` words -- bounded on
        both sides by real matches, same as above, just not a clean 1:1
        shape. Confirmed real cases (Ordinary Day, lrclib id 6210269):
        MXL "winnes" (1 word, OCR merge) for LRC "win now" (2 words);
        MXL "stomty"+"in" (2 words) for LRC "stop"+"trying," (2 words,
        each individually too garbled to clear the 1:1 fuzzy ratio on its
        own); MXL "double"+"edged"+"kide" (3 words) for LRC
        "double-edged"+"knife," (2 words, one hyphenated). The WHOLE
        block's concatenated characters (not each word pair individually)
        must clear `MXL_LRC_FUZZY_TEXT_MIN_RATIO`, then the recovered LRC
        words are distributed across the MXL block's own word slots via
        `_distribute_words_to_slots` -- splitting a hyphenated LRC word
        into pieces first if that's what's needed to make the counts
        line up (the "double-edged"/"kide" case), otherwise merging or
        melisma-padding the same way syllable counts are already
        reconciled elsewhere in this module."""
    lrc_flat: List[str] = []       # normalized, for matching
    lrc_flat_raw: List[str] = []   # raw, for display substitution
    lrc_line_idx: List[int] = []
    for li, (_, text) in enumerate(lrc_lines):
        for tok in text.split():
            n = _normalize(tok)
            if n:
                lrc_flat.append(n)
                lrc_flat_raw.append(tok)
                lrc_line_idx.append(li)

    mxl_norm = [w.norm for w in mxl_words]
    sm = difflib.SequenceMatcher(None, mxl_norm, lrc_flat, autojunk=False)
    word_line = {}
    word_clean_text: dict = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                word_line[i1 + k] = lrc_line_idx[j1 + k]
                word_clean_text[i1 + k] = lrc_flat_raw[j1 + k]
        elif tag == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
            ratio = difflib.SequenceMatcher(None, mxl_norm[i1], lrc_flat[j1]).ratio()
            if ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO:
                word_line[i1] = lrc_line_idx[j1]
                word_clean_text[i1] = lrc_flat_raw[j1]
            # else: genuinely too different -- leave unmatched, falls
            # through to MXL's own raw text same as before.
        elif (tag == "replace" and (i2 - i1) <= config.MXL_LRC_BLOCK_MAX_WORDS
              and (j2 - j1) <= config.MXL_LRC_BLOCK_MAX_WORDS):
            mxl_block_norm = "".join(mxl_norm[i1:i2])
            lrc_block_norm = "".join(lrc_flat[j1:j2])
            ratio = difflib.SequenceMatcher(None, mxl_block_norm, lrc_block_norm).ratio()
            if ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO:
                assigned = _distribute_words_to_slots(lrc_flat_raw[j1:j2], i2 - i1)
                for k in range(i2 - i1):
                    word_line[i1 + k] = lrc_line_idx[j1]
                    word_clean_text[i1 + k] = assigned[k]
            # else: genuinely too different -- leave unmatched.

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
    lines = [v if v is not None else first_known for v in filled]
    clean_text = [word_clean_text.get(i) for i in range(n)]
    return lines, clean_text


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
                         asr_words: List[Word],
                         word_clean_text: Optional[List[Optional[str]]] = None) -> Tuple[List[float], List[float], MxlLrcQuality]:
    """PASS 1: for each LRC line, matches that line's own MXL words
    against real ASR words whose own timestamp falls near the line's
    real-time window (order-preserving difflib, same technique used
    throughout this project for text alignment).

    Matches on the CLEAN text (`word_clean_text`, from
    `assign_words_to_lines`) when available, falling back to the MXL
    word's own raw OCR norm otherwise -- confirmed real bug fixed here:
    matching on the raw MXL norm alone missed a real ASR match entirely
    when the MXL's own OCR text and ASR's transcription independently
    garbled the same word differently (real case: MXL OCR'd "favors" as
    "favere"; ASR transcribed it correctly as "favors" -- neither string
    equals the other, so the word never matched even though a clean,
    already-verified "favors" was sitting right there in
    `word_clean_text`, unused).

    On top of that, a clean 1:1 "replace" pairing (difflib's own
    alignment already anchored it between correct matches on both sides)
    is ALSO accepted when the two words are merely CLOSE (character-level
    ratio >= `config.MXL_LRC_FUZZY_TEXT_MIN_RATIO`), same technique and
    threshold `assign_words_to_lines` already uses for display text.
    Needed for a real, distinct failure mode: ASR itself can mishear a
    word ("favors" transcribed as "favorites") independently of any MXL
    OCR issue -- an exact-only comparison, even against the already-clean
    text, still misses this, since the mismatch this time is ASR's own,
    not the MXL's.

    A matched word is only trusted if the ASR match ALSO clears
    `config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE` -- confirmed real case: a
    text match with confidence 0.003 had a genuinely wrong (0.77s off)
    timestamp, independent of anything else in this pipeline. A
    low-confidence "match" is treated as no match at all. A trusted
    match's START/END come directly from the ASR word's own values.

    PASS 2: every remaining (non-confident) word is placed by
    interpolating from its NEAREST CONFIDENT neighbors (by MXL offset
    order) -- NOT by stretching proportionally across the whole line's
    window, which was a real, confirmed bug: a line whose own (t0, t1)
    window includes trailing silence (e.g. an instrumental gap before
    the next line starts) stretches every non-confident word in it well
    past where it actually belongs, using an offset-to-time RATE that's
    diluted by real silence the words themselves never occupy. Real case
    that exposed this: the LRC line "Because the system works, the
    system called reciprocity" spans 16.5s, but the real singing ends
    ~10.6s in -- every fallback word after that packed together near the
    tail of the line's own window. Interpolating from the nearest real
    ASR anchors (before AND after, by MXL-offset order, not bounded to
    the same line) uses a locally-accurate rate instead. If only one
    side has an anchor, extrapolates using that anchor's own nearest
    neighbor pair; if no anchor exists anywhere (total ASR failure),
    falls back to the old whole-line-proportional formula so that
    degenerate case doesn't get worse. The result is always clamped into
    the word's own LRC line's (t0, t1) window as a sanity backstop.

    Non-decreasing order on START is then enforced (clamp) -- ASR/
    interpolation can occasionally produce a slightly out-of-order local
    result. ENDs are then clamped to never exceed the NEXT word's own
    start (no overlap) but are free to end EARLIER, leaving a real rest
    -- the fix for a word swallowing a real pause (e.g. "hen." held for
    3.1s, "The" held for 7.1s, both confirmed real bugs)."""
    line_word_idxs: dict = {}
    for i, li in enumerate(word_lines):
        line_word_idxs.setdefault(li, []).append(i)

    n = len(mxl_words)
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    confident: List[bool] = [False] * n
    quality = MxlLrcQuality(n_words=n)

    # --- Pass 1: confident ASR matches only. ---
    for li, idxs in line_word_idxs.items():
        idxs = sorted(idxs)
        t0, t1 = _line_window(lrc_lines, li)
        asr_in_window = [w for w in asr_words if t0 - 0.5 <= w.start <= t1 + 0.5]
        asr_norm = [_normalize(w.text) for w in asr_in_window]
        mxl_norm_line = [_normalize(word_clean_text[i]) if word_clean_text and word_clean_text[i]
                          else mxl_words[i].norm for i in idxs]
        sm = difflib.SequenceMatcher(None, mxl_norm_line, asr_norm, autojunk=False)
        matched_local = {}
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(a2 - a1):
                    asr_w = asr_in_window[b1 + k]
                    if asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                        matched_local[a1 + k] = asr_w
                    # else: leave unmatched -- falls through to pass 2.
            elif tag == "replace" and (a2 - a1) == 1:
                # A single unmatched MXL word against one or more ASR words
                # difflib's own alignment already anchored here, between
                # correct matches on both sides -- if one of them is merely
                # CLOSE (not identical) to the MXL word, trust it the same
                # way assign_words_to_lines already does for display text.
                # Confirmed real, necessary case: ASR itself mishears a word
                # ("favors" transcribed as "favorites") independently of any
                # MXL OCR issue -- an exact-only match (even against the
                # already-cleaned text) still misses this, since ASR's own
                # error is the mismatch here, not the MXL's.
                #
                # The ASR side of the block isn't always exactly one word:
                # `asr_in_window` is time-bounded, not line-bounded (a
                # deliberate +-0.5s slop so a confident match isn't missed
                # just for landing slightly outside the LRC line's own
                # window) -- so a single spilled-over word from the
                # NEXT line (e.g. "I'm") can tag along in the same replace
                # block as the real mismatch ("favorites"). Real confirmed
                # case: block was ['favors'] vs ['favorites', "i'm"], a 1:2
                # replace that the old code's exact `(b2-b1)==1` check
                # rejected outright even though the correct candidate
                # ("favorites") was sitting right there at the block's own
                # start. Only a single MXL word is unresolved here, so it
                # can only ever correspond to at most one real ASR word --
                # try each candidate in the block and keep the best-scoring
                # one that clears the threshold, rather than requiring the
                # block to already be exactly 1:1.
                best_ratio = 0.0
                best_asr_w = None
                for bk in range(b1, b2):
                    ratio = difflib.SequenceMatcher(None, mxl_norm_line[a1], asr_norm[bk]).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_asr_w = asr_in_window[bk]
                if best_ratio >= config.MXL_LRC_FUZZY_TEXT_MIN_RATIO and best_asr_w is not None:
                    if best_asr_w.confidence >= config.MXL_LRC_MIN_ASR_WORD_CONFIDENCE:
                        matched_local[a1] = best_asr_w
                # else: genuinely different words -- leave unmatched.

        for local_i, global_i in enumerate(idxs):
            if local_i not in matched_local:
                continue
            asr_w = matched_local[local_i]
            w = mxl_words[global_i]
            word_qtr_dur = sum(s[1] for s in w.syllables)
            starts[global_i] = asr_w.start
            asr_dur = asr_w.end - asr_w.start
            ends[global_i] = (asr_w.start + asr_dur if asr_dur > 0
                               else asr_w.start + word_qtr_dur * config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC)
            confident[global_i] = True
            quality.n_asr_placed += 1

    # --- Pass 2: nearest-confident-anchor interpolation for everything else. ---
    confident_idxs = [i for i in range(n) if confident[i]]

    def nearest_before(i: int) -> Optional[int]:
        best = None
        for ci in confident_idxs:
            if ci < i:
                best = ci
            else:
                break
        return best

    def nearest_after(i: int) -> Optional[int]:
        for ci in confident_idxs:
            if ci > i:
                return ci
        return None

    for i in range(n):
        if confident[i]:
            continue
        li = word_lines[i]
        t0, t1 = _line_window(lrc_lines, li)
        w = mxl_words[i]
        word_qtr_dur = sum(s[1] for s in w.syllables)

        pb = nearest_before(i)
        pa = nearest_after(i)
        rate = None
        base_idx = None
        if pb is not None and pa is not None:
            off_delta = mxl_words[pa].offset - mxl_words[pb].offset
            if off_delta > 0:
                rate = (starts[pa] - starts[pb]) / off_delta
            base_idx = pb
        elif pb is not None:
            pbb = nearest_before(pb)
            if pbb is not None:
                off_delta = mxl_words[pb].offset - mxl_words[pbb].offset
                if off_delta > 0:
                    rate = (starts[pb] - starts[pbb]) / off_delta
            base_idx = pb
        elif pa is not None:
            paa = nearest_after(pa)
            if paa is not None:
                off_delta = mxl_words[paa].offset - mxl_words[pa].offset
                if off_delta > 0:
                    rate = (starts[paa] - starts[pa]) / off_delta
            base_idx = pa

        if rate is not None and base_idx is not None:
            base = mxl_words[base_idx]
            est_start = starts[base_idx] + (w.offset - base.offset) * rate
        else:
            # No usable anchor anywhere in the whole song (total ASR
            # failure) -- fall back to the old whole-line-proportional
            # formula rather than leaving this word unplaced.
            idxs = sorted(line_word_idxs[li])
            offs = [mxl_words[j].offset for j in idxs]
            lo_off, hi_off = min(offs), max(offs)
            span = hi_off - lo_off
            frac = (w.offset - lo_off) / span if span > 0 else 0.0
            est_start = t0 + frac * (t1 - t0)
            rate = (t1 - t0) / span if span > 0 else config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC

        # Sanity backstop: never escape this word's own LRC line window,
        # even though the RATE may have been informed by an anchor in an
        # adjacent line.
        est_start = max(t0, min(est_start, t1))
        starts[i] = est_start
        ends[i] = est_start + word_qtr_dur * (rate if rate and rate > 0 else config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC)
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


def _text_for_mxl_syllables(clean_text: Optional[str], mxl_syllable_texts: List[str]) -> List[str]:
    """Returns the text to display on each of an MXL word's own syllable
    slots. If `clean_text` (the matched LRC token, see
    `assign_words_to_lines`) is available, it's hyphenated
    (`syllables.hyphenate`) and reconciled to the MXL's own syllable
    COUNT -- merged down (`syllables.chunk_to_count`) if the clean word
    hyphenates into MORE pieces than MXL notated, or padded with
    `config.MELISMA_CONTINUATION_TEXT` if FEWER -- mirroring
    `lyric_alignment._syllables_for_word`'s existing pattern exactly.
    Falls back to the MXL's own raw syllable text only when no clean
    match exists at all (better than nothing, but never preferred: the
    MXL's own OCR can be wrong, e.g. "systern"/"eystern" for "system")."""
    n = len(mxl_syllable_texts)
    if clean_text is None:
        return mxl_syllable_texts
    parts = hyphenate(clean_text)
    if len(parts) == n:
        return parts
    elif len(parts) > n:
        return chunk_to_count(parts, n)
    else:
        return list(parts) + [config.MELISMA_CONTINUATION_TEXT] * (n - len(parts))


def build_syllables(mxl_words: List[MxlWord], word_starts: List[float], word_ends: List[float],
                     word_lines: List[int], word_clean_text: List[Optional[str]]) -> List[Syllable]:
    """Splits each word's own syllables proportionally within
    [word_start, word_end) (see `place_words_via_asr` for how those are
    derived from ASR and/or MXL note values -- NOT simply "until the next
    word starts", which used to swallow real pauses between words) using
    the MXL's own relative sub-word offsets -- that part of the MXL data
    (syllable-to-syllable ratios within one word) is reliable, so there's
    no need to guess those from ASR too. The DISPLAYED text comes from
    `word_clean_text` (the matched LRC token) via `_text_for_mxl_syllables`
    whenever available -- MXL supplies pitch/timing only, never the
    displayed text, when a clean source exists. `line_id` is set from
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
        mxl_syllable_texts = [s[3] for s in w.syllables]
        display_texts = _text_for_mxl_syllables(word_clean_text[i], mxl_syllable_texts)
        for syl_i, ((off, dur, midi, _orig_text), text) in enumerate(zip(w.syllables, display_texts)):
            frac0 = (off - lo) / (hi - lo) if hi > lo else 0.0
            frac1 = (off + dur - lo) / (hi - lo) if hi > lo else 1.0
            syllables.append(Syllable(
                text=text, start=t0 + frac0 * (t1 - t0), end=t0 + frac1 * (t1 - t0),
                midi_note=midi - 60, is_word_start=(syl_i == 0), line_id=word_lines[i],
            ))
    return syllables


@dataclass
class TimeCalibration:
    offset_sec: Optional[float] = None
    slope: float = 0.0
    confidence: float = 0.0
    kind: Optional[str] = None   # "constant", "drift", "piecewise", "isotonic", or None if uncalibrated
    skipped_reason: Optional[str] = None
    # `correction_fn(lrc_start) -> corrected_real_time`, populated for
    # every successful `kind` -- see two_tier_time_calibration's own
    # docstring for why offset_sec/slope alone aren't enough for
    # "piecewise"/"isotonic".
    correction_fn: Optional[Callable[[float], float]] = None
    # Diagnostic-only odd/even-anchor holdout residual (seconds), tier 3
    # ("piecewise"/"isotonic") only -- see lrc_timing._holdout_residual_sec.
    holdout_residual_sec: Optional[float] = None


@dataclass
class MxlLrcResult:
    success: bool
    reason: str
    syllables: List[Syllable] = field(default_factory=list)
    quality: Optional[MxlLrcQuality] = None
    lrc_match: Optional[LrcMatch] = None
    mxl_path: Optional[str] = None
    part_names_used: List[str] = field(default_factory=list)
    time_calibration: Optional[TimeCalibration] = None


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

    # Calibrate away a systematic time offset (and, if needed, real drift)
    # between LRC's own line timestamps and OUR audio's real timing --
    # confirmed real case: "Ordinary Day" (lrclib id 6210269) has ~2.4s of
    # extra lead-in silence in our own audio vs. whichever recording
    # LRCLIB's synced lyrics were timed against. Left uncalibrated, every
    # line's ASR search window (`_line_window`, +-0.5s) and interpolation-
    # fallback window are off by the same amount, which blew past the
    # quality gate outright (22% non-monotonic placements) rather than
    # just being imprecise. A null/near-zero calibration is a no-op here
    # (offset+slope of ~0 shifts nothing), so this can't regress an
    # already-well-aligned candidate.
    #
    # No `structural_check` passed (2026-08-11 scope decision, see
    # CLAUDE.md's tier-3 real-validation writeup): would need "our own
    # lines" built from `mxl_words` (a flat per-syllable list with no
    # native line grouping the way realign.py's already-authored existing
    # file has), left as a follow-up rather than built speculatively here.
    # This means tier 3's "rescue" case (see two_tier_time_calibration's
    # own docstring) always declines for THIS path -- only "refine"
    # (tier 1/2 already found some real support) is available -- the
    # safe default, not a regression: this path's own downstream ASR-
    # placement-rate quality gate (MXL_LRC_MIN_ASR_PLACEMENT_RATE) is
    # also still there as a backstop either way.
    time_candidates = match_asr_to_lrc_lines(asr_words, lrc_match.lrc_lines)
    offset, slope, confidence, kind, skipped_reason, correction_fn, holdout = two_tier_time_calibration(
        time_candidates)
    time_cal = TimeCalibration(offset_sec=offset, slope=slope, confidence=confidence,
                                kind=kind, skipped_reason=skipped_reason, correction_fn=correction_fn,
                                holdout_residual_sec=holdout)
    if offset is not None:
        lrc_match.lrc_lines = [(correction_fn(t), text) for t, text in lrc_match.lrc_lines]

    word_lines, word_clean_text = assign_words_to_lines(mxl_words, lrc_match.lrc_lines)
    word_starts, word_ends, quality = place_words_via_asr(mxl_words, word_lines, lrc_match.lrc_lines, asr_words,
                                                            word_clean_text=word_clean_text)
    syllables = build_syllables(mxl_words, word_starts, word_ends, word_lines, word_clean_text)

    nonmonotonic_rate = quality.non_monotonic_fix_count / quality.n_words if quality.n_words else 1.0
    if quality.asr_placement_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:
        return MxlLrcResult(
            success=False,
            reason=(f"ASR/MXL word match rate too low ({quality.asr_placement_rate:.0%}, need "
                     f"{config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:.0%}) -- the matched lyrics likely don't "
                     f"correspond to this recording"),
            syllables=syllables, quality=quality, lrc_match=lrc_match,
            mxl_path=mxl_path, part_names_used=part_names, time_calibration=time_cal,
        )
    if nonmonotonic_rate > config.MXL_LRC_MAX_NONMONOTONIC_RATE:
        return MxlLrcResult(
            success=False,
            reason=f"too many out-of-order word placements ({nonmonotonic_rate:.0%})",
            syllables=syllables, quality=quality, lrc_match=lrc_match,
            mxl_path=mxl_path, part_names_used=part_names, time_calibration=time_cal,
        )

    return MxlLrcResult(success=True, reason="", syllables=syllables, quality=quality,
                         lrc_match=lrc_match, mxl_path=mxl_path, part_names_used=part_names,
                         time_calibration=time_cal)


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
