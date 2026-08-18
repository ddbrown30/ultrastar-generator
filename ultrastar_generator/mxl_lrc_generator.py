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
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple

from . import config
from .lyrics_lookup import LrcLibCandidate, search_lrclib, effective_lrc_duration
from .lrc_timing import (parse_lrc, two_tier_time_calibration, match_asr_to_lrc_lines, lrc_line_window,
                          match_block_to_candidates, words_in_time_window)
from .models import Syllable, Word
from .syllables import hyphenate, chunk_to_count
from .text_normalize import normalize_word as _normalize


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

    # Materialize each note's own (offset, duration, pitch, text,
    # syllabic, tied) up front -- decoupled from live music21 objects, to
    # allow the lookahead OCR-repair pass below (and to keep the main
    # grouping loop's own logic working over a plain list either way).
    raw_notes = []
    for n in chosen_part.flatten().notes:
        if n.isChord:
            continue
        text = syl = None
        for ly in n.lyrics:
            if ly.text:
                text, syl = ly.text, ly.syllabic
            break  # one lyric verse only
        raw_notes.append({
            "offset": float(n.offset), "dur": float(n.quarterLength),
            "midi": int(n.pitch.midi), "text": text, "syl": syl,
            "tied": n.tie is not None,
        })

    # Real OCR/engraving defect (2026-08-18, user-identified real case,
    # "Great Big Sea - Ordinary Day": a trailing ellipsis ("...") after
    # "know." got engraved as its OWN separate note/word ("..", a second
    # note immediately after "know."'s own note, contiguous, real
    # confirmed case) instead of staying part of "know."'s own trailing
    # punctuation. A note whose ENTIRE lyric text is punctuation (nothing
    # alphanumeric at all -- normalizes to "") is never a real word on
    # its own; if it's immediately contiguous with a preceding REAL word,
    # its text is absorbed onto that word's own end and the note itself
    # becomes a plain melisma continuation (no lyric of its own), same as
    # any other untexted tied/slurred note.
    for i, entry in enumerate(raw_notes):
        if not entry["text"] or _normalize(entry["text"]):
            continue  # no text, or it has real alphanumeric content -- not pure punctuation
        if i == 0:
            continue
        prev = raw_notes[i - 1]
        if not prev["text"]:
            continue
        if abs(entry["offset"] - (prev["offset"] + prev["dur"])) > 1e-6:
            continue  # not contiguous with the preceding word -- leave alone
        prev["text"] += entry["text"]
        entry["text"] = None
        entry["syl"] = None

    # Real OCR/engraving defect (2026-08-18, user-identified real case,
    # "Great Big Sea - Ordinary Day": "right,it's" -- two real words
    # merged onto ONE note's own lyric text, missing the space/note split
    # the engraver should have made, while the NEXT note was left with no
    # lyric of its own at all. Detected by an internal comma/period/space
    # in the text that still has real ALPHANUMERIC content AFTER it (not
    # just more punctuation -- a real confirmed near-miss: "know..." must
    # NOT be treated as "know."+".." merged wrong, the ".." isn't a real
    # word start). A genuine TRAILING comma/period (e.g. "battered,") has
    # nothing after it at all and is untouched either way. Only acted on
    # when the next note doesn't already carry its own real word -- if it
    # does, this isn't confidently an OCR mistake, so the merged text is
    # left alone rather than guessed at.
    merge_re = re.compile(r"^(.+?[,.\s])([A-Za-z0-9].*)$")
    for i, entry in enumerate(raw_notes):
        if not entry["text"]:
            continue
        m = merge_re.match(entry["text"])
        if not m:
            continue
        if i + 1 >= len(raw_notes) or raw_notes[i + 1]["text"]:
            continue  # no next note, or it already has its own real word
        entry["text"] = m.group(1)
        raw_notes[i + 1]["text"] = m.group(2)
        raw_notes[i + 1]["syl"] = "single"

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

    for entry in raw_notes:
        if not entry["text"]:
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
                if abs(entry["offset"] - (off + dur)) < 1e-6:
                    midi = entry["midi"]
                    if entry["tied"] and midi == prev_midi:
                        cur_syllables[-1] = (off, dur + entry["dur"], prev_midi, text)
                    else:
                        cur_syllables.append((entry["offset"], entry["dur"], midi, ""))
                # else: a real rest separates this note from the word in
                # progress -- not a continuation, leave it unattached.
            continue
        syl = entry["syl"]
        if syl in (None, "single", "begin"):
            flush()
            cur_text = entry["text"]
            cur_offset = entry["offset"]
        else:
            cur_text += entry["text"]
        cur_syllables.append((entry["offset"], entry["dur"], entry["midi"], entry["text"]))
    flush()

    return words, [chosen_part.partName]


@dataclass
class LrcMatch:
    candidate: LrcLibCandidate
    lrc_lines: List[Tuple[float, str]]
    content_match_ratio: float
    duration_delta: Optional[float]


def _artist_matches(our_artist: str, candidate_artist: str) -> bool:
    """Whether a candidate's own credited artist plausibly refers to the
    same performer/production as ours -- substring match after
    normalization (handles a YouTube "- Topic" channel suffix, "feat."
    credits, multi-name cast-recording listings, a show title embedded in
    a longer cast credit, etc. in EITHER direction) rather than exact
    string equality, which real LRCLIB data rarely gives you. Empty on
    either side never counts as a match."""
    a = _normalize(our_artist)
    b = _normalize(candidate_artist)
    if not a or not b:
        return False
    return a in b or b in a


def select_lrc_candidate(artist: str, title: str, mxl_words: List[MxlWord], audio_duration: float,
                          forced: Optional[LrcLibCandidate] = None) -> Optional[LrcMatch]:
    """Picks an LRC candidate to use for timing. If `forced` is given (a
    user-pinned or --lrclib-id-resolved candidate), it's used directly,
    no filtering -- the user already vetted it. Otherwise searches LRCLIB
    (both artist/title and free-text `q`, deduped -- the free-text search
    was found necessary this session: LRCLIB's artist/title search alone
    can miss a candidate its own free-text search finds), requires
    `synced_lyrics`, requires duration within
    `config.MXL_LRC_DURATION_TOLERANCE_SEC`, and requires a content-match
    (difflib ratio of MXL words vs the candidate's plain lyrics) clearing
    `config.MXL_LRC_MIN_CONTENT_MATCH_RATIO`. This bar is intentionally
    permissive -- see this module's docstring for why the real validity
    gate is downstream, not here. All duration comparisons here (filter,
    scoring, `duration_delta`) use `effective_lrc_duration`, not `c.duration`
    directly -- LRCLIB's own duration metadata isn't verified against the
    synced lyrics it ships with, and gets cross-checked/corrected against
    the candidate's own last real lyric timestamp (see that function's own
    docstring).

    Duration ranking above content ratio (see below) makes this metadata
    trust matter beyond just the upfront filter -- an untrustworthy
    `duration` could otherwise outrank the genuinely correct candidate
    on a fabricated tie or push a good candidate outside the tolerance
    filter entirely.

    Among candidates clearing those bars, ranks by (1) whether the
    candidate's own credited artist plausibly matches ours
    (`_artist_matches`) -- decisively, not just as a tiebreaker: a
    same-artist candidate always outranks a different-artist one,
    regardless of content ratio or duration -- (2) duration proximity to
    OUR real audio, (3) content-match ratio as the final tiebreaker. Real
    confirmed case (Trixie Mattel - "Video Games", 2026-08-15): the
    correct candidate (Trixie Mattel's own cover, duration within 0.7s of
    ours) was being passed over for Lana Del Rey's ORIGINAL (a different
    performer entirely, duration 14.5s off) purely because
    difflib.SequenceMatcher's ratio happened to score the wrong-performer
    candidate higher -- content-text similarity alone can't be trusted to
    prefer the right PERFORMER over a same-titled original/cover by
    someone else; a real, close duration + a real artist-credit match are
    much harder to fake by coincidence than a text ratio is. Duration
    ranks above ratio (not just artist above ratio) for the same reason:
    a genuine different edition/arrangement reliably shows up as a
    duration difference, while ratio among candidates that already
    cleared the content-match floor is noisier than it looks. `artist`
    empty/blank, or no candidate's credit resembling it at all (e.g. a
    cast-recording credited to individual performers, not the show name
    used as our own artist tag -- real case: Chicago), falls through to
    ranking purely by duration then ratio among whatever's left, same as
    before this ranking existed."""
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
        forced_duration = effective_lrc_duration(forced)
        delta = abs(forced_duration - audio_duration) if forced_duration is not None else None
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
        c_duration = effective_lrc_duration(c)
        delta = abs(c_duration - audio_duration)
        if delta > config.MXL_LRC_DURATION_TOLERANCE_SEC:
            continue
        lrc_norm = [_normalize(t) for t in (c.plain_lyrics or "").split()]
        lrc_norm = [w for w in lrc_norm if w]
        if not lrc_norm:
            continue
        ratio = difflib.SequenceMatcher(None, mxl_norm_words, lrc_norm, autojunk=False).ratio()
        if ratio < config.MXL_LRC_MIN_CONTENT_MATCH_RATIO:
            continue
        scored.append((_artist_matches(artist, c.artist_name), ratio, delta, c))

    if not scored:
        return None
    scored.sort(key=lambda t: (not t[0], t[2], -t[1]))
    _artist_match, ratio, delta, best = scored[0]
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


def _slice_by_weights(text: str, weights: List[int]) -> Optional[List[str]]:
    """Slices `text` into len(weights) contiguous pieces, sized
    proportionally to `weights` -- used to recover a real per-syllable
    split from the MXL's own notated NOTE structure (never a linguistic
    hyphenation guess: the weights are each MXL syllable/word's own
    character length, which reflects the real notated split even when
    its own letters can't be trusted, e.g. OCR garbling or a word
    mis-segmented into several single-syllable "words" -- see
    `_text_for_mxl_syllables`/`_distribute_words_to_slots`, the two
    callers). Returns None (defer to a hyphenation-based fallback
    instead) when `text` doesn't even have one character per slot -- that
    signals a genuine melisma (one syllable held across several notes),
    not a recoverable per-syllable split, and must never be force-split
    into fake pieces."""
    n = len(weights)
    if n <= 0 or len(text) < n:
        return None
    total = sum(weights) or n
    boundaries = [0]
    acc = 0
    for i, w in enumerate(weights[:-1]):
        acc += w
        remaining_slots = n - (i + 1)
        b = round(acc / total * len(text))
        b = max(b, boundaries[-1] + 1)
        b = min(b, len(text) - remaining_slots)
        boundaries.append(b)
    boundaries.append(len(text))
    return [text[boundaries[i]:boundaries[i + 1]] for i in range(n)]


def _distribute_words_to_slots(lrc_words_raw: List[str], n_slots: int,
                                mxl_slot_texts: Optional[List[str]] = None) -> List[str]:
    """Distributes a recovered stretch of raw LRC words across n_slots MXL
    word display slots -- OCR word-segmentation doesn't always match the
    real lyric's own word boundaries (see `assign_words_to_lines`'s own
    docstring for real confirmed cases). Always returns exactly n_slots
    items, in order:
      - Fewer real words than slots: first tries splitting any hyphenated
        word into its own pieces (real case: MXL notated "double"/"edged"
        as two separate words, but LRC's own line joins them into one
        "double-edged" token). If exactly ONE real (non-hyphenated) word
        remains short of the slot count, recovers the split from the
        MXL's OWN NOTES rather than guessing linguistically -- real
        confirmed case (Les Miserables - Stars, 2026-08-18): the MXL's
        own syllabic markers mis-notated "never" as two separate
        SINGLE-syllable words ("ne", "ver") instead of one word split
        "begin"/"end", so this function was handed exactly 1 LRC word
        ("never") for 2 MXL slots and produced "never"+"~" (a word-count
        mismatch, not a real melisma) instead of splitting it. When the
        caller passes `mxl_slot_texts` (each slot's own original MXL
        word/syllable text -- "ne"/"ver" here), `_slice_by_weights` uses
        THEIR character lengths to slice the recovered clean word at the
        matching position (never a hyphenation guess: "ne"(2)+"ver"(3)
        sums to 5, matching "never"'s own 5 characters exactly, so the
        slice lands precisely on "ne"/"ver" -- the real notated split).
        Only falls back to `syllables.hyphenate` when no usable
        `mxl_slot_texts` was given, or the word is too short for one
        character per slot (a genuine melisma, not a recoverable split --
        `_slice_by_weights` returns None for this itself). Gated on
        actually producing >1 piece so a genuinely monosyllabic word sung
        across multiple notes is never force-split into fake syllables --
        it still falls through to melisma-padding below unchanged. If
        that still isn't enough, pads the remainder with
        `config.MELISMA_CONTINUATION_TEXT` (same convention
        `_text_for_mxl_syllables` already uses for a word with fewer
        syllables than notated).
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
    if len(words) == 1 and n_slots > 1:
        sliced = None
        if mxl_slot_texts and len(mxl_slot_texts) == n_slots:
            sliced = _slice_by_weights(words[0], [max(1, len(t)) for t in mxl_slot_texts])
        if sliced is not None:
            words = sliced
        else:
            parts = hyphenate(words[0])
            if len(parts) > 1:
                words = chunk_to_count(parts, n_slots) if len(parts) > n_slots else parts
    if len(words) > n_slots:
        words = _merge_words_to_count(words, n_slots)
    elif len(words) < n_slots:
        words = words + [config.MELISMA_CONTINUATION_TEXT] * (n_slots - len(words))
    return words


def assign_words_to_lines(
        mxl_words: List[MxlWord],
        lrc_lines: List[Tuple[float, str]]) -> Tuple[List[int], List[Optional[str]], List[int], dict]:
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
        reconciled elsewhere in this module.

    ALSO returns `word_group` and `word_group_text` (2026-08-18, user's
    explicit correction: a word spanning several MXL NOTES is a normal,
    intentional sheet-music pattern, not a defect -- but this specific
    sub-case, several separate MXL WORDS recovering to exactly ONE real
    LRC word, needs to be treated as ONE semantic word downstream too,
    not just fixed for display text). `word_group[i]` is the group id
    (the group's own first MXL index) for word `i` -- identical to `i`
    itself for an ungrouped word. `word_group_text[gid]` is the group's
    own recovered whole-word text (e.g. "never"). Set ONLY for the
    "MULTIPLE MXL words -> exactly ONE real LRC word" shape above (real
    case: Stars' own "never" mis-notated as two separate single-syllable
    "words", "ne"+"ver", each syllabic="single" rather than one word
    split "begin"/"end") -- `place_words_via_asr` uses this to search the
    real ASR transcript for the whole word ONCE (e.g. "never"), instead
    of trying to match "ne" and "ver" separately against a transcript
    that only ever has the whole word, which can never succeed and
    silently drops what would otherwise be a confident, accurate ASR
    match down to a less-reliable interpolated guess. NOT set for the
    "double"+"edged"+"kide" shape -- there each MXL slot recovers its OWN
    distinct real word, and those genuinely are separate, independently
    matchable spoken words."""
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
    word_group = list(range(len(mxl_words)))  # identity by default -- each word its own group
    word_group_text: dict = {}
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
                assigned = _distribute_words_to_slots(
                    lrc_flat_raw[j1:j2], i2 - i1,
                    mxl_slot_texts=[w.text for w in mxl_words[i1:i2]])
                for k in range(i2 - i1):
                    word_line[i1 + k] = lrc_line_idx[j1]
                    word_clean_text[i1 + k] = assigned[k]
                if (i2 - i1) > 1 and (j2 - j1) == 1:
                    # Several separate MXL WORDS recovered to exactly ONE
                    # real LRC word -- group them (see this function's own
                    # docstring) so place_words_via_asr searches the real
                    # ASR transcript for the whole word ONCE.
                    for k in range(i1, i2):
                        word_group[k] = i1
                    word_group_text[i1] = lrc_flat_raw[j1]
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
    return lines, clean_text, word_group, word_group_text


@dataclass
class MxlLrcQuality:
    n_words: int = 0
    n_asr_placed: int = 0
    n_fallback: int = 0
    non_monotonic_fix_count: int = 0

    @property
    def asr_placement_rate(self) -> float:
        return self.n_asr_placed / self.n_words if self.n_words else 0.0


def place_words_via_asr(mxl_words: List[MxlWord], word_lines: List[int], lrc_lines: List[Tuple[float, str]],
                         asr_words: List[Word],
                         word_clean_text: Optional[List[Optional[str]]] = None,
                         word_group: Optional[List[int]] = None,
                         word_group_text: Optional[dict] = None) -> Tuple[List[float], List[float], MxlLrcQuality]:
    """PASS 0 (2026-08-18): for each GROUPED block (`word_group`/
    `word_group_text` from `assign_words_to_lines` -- several separate
    MXL words the MXL itself notated for what's really ONE spoken/sung
    word, e.g. Stars' own "never" mis-notated as "ne"+"ver"), searches
    the group's own line window ONCE for the real ASR token matching the
    group's WHOLE recovered word ("never"), via the same
    `match_block_to_candidates` Pass 1 already uses. This exists because
    the group's own INDIVIDUAL pieces ("ne", "ver") can never match a
    transcript that only ever has the whole word -- without this, they'd
    silently and permanently fall to the less-reliable Pass 2
    interpolation below, even when a perfectly good, confident ASR match
    for the whole word is sitting right there. On a confident match, the
    matched span is distributed across the group's own member notes
    proportionally by each note's own quarterLength -- the same
    proportional-split idea `build_syllables` already uses WITHIN one
    normal multi-syllable word, just applied across several MXL WORD
    entries instead of one word's own syllables. A group that doesn't
    match confidently as a whole is left alone; its members fall through
    to Pass 1/Pass 2 exactly as if ungrouped -- no regression either way.

    PASS 1: for each LRC line, matches that line's own (ungrouped, or
    unmatched-as-a-group) MXL words against real ASR words whose own
    timestamp falls near the line's real-time window (order-preserving
    difflib, same technique used throughout this project for text
    alignment).

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

    # --- Pass 0: grouped multi-MXL-word blocks, see this function's own docstring. ---
    grouped_handled: set = set()
    if word_group and word_group_text:
        groups: dict = {}
        for i, gid in enumerate(word_group):
            groups.setdefault(gid, []).append(i)
        for gid, members in groups.items():
            if len(members) <= 1 or gid not in word_group_text:
                continue
            li = word_lines[members[0]]
            t0, t1 = lrc_line_window(lrc_lines, li)
            asr_in_window = words_in_time_window(asr_words, t0, t1)
            target_norm = _normalize(word_group_text[gid])
            if not target_norm:
                continue
            matched = match_block_to_candidates([target_norm], asr_in_window)
            asr_w = matched.get(0)
            if asr_w is None:
                continue
            total_qtr = sum(sum(s[1] for s in mxl_words[m].syllables) for m in members) or len(members)
            asr_dur = asr_w.end - asr_w.start
            span = asr_dur if asr_dur > 0 else total_qtr * config.MXL_LRC_DEFAULT_QUARTER_NOTE_SEC
            cursor = asr_w.start
            for m in members:
                m_qtr = sum(s[1] for s in mxl_words[m].syllables) or 1.0
                frac = m_qtr / total_qtr
                starts[m] = cursor
                ends[m] = cursor + span * frac
                cursor = ends[m]
                confident[m] = True
                quality.n_asr_placed += 1
                grouped_handled.add(m)

    # --- Pass 1: confident ASR matches only. ---
    for li, idxs in line_word_idxs.items():
        idxs = sorted(i for i in idxs if i not in grouped_handled)
        t0, t1 = lrc_line_window(lrc_lines, li)
        asr_in_window = words_in_time_window(asr_words, t0, t1)
        mxl_norm_line = [_normalize(word_clean_text[i]) if word_clean_text and word_clean_text[i]
                          else mxl_words[i].norm for i in idxs]
        # matched_local: see lrc_timing.match_block_to_candidates -- ASR
        # mishearing a word (e.g. "favors" transcribed as "favorites") is
        # tolerated via a fuzzy-ratio fallback on a single-word replace
        # block; `asr_in_window`'s own +-0.5s time slop (not line-bounded)
        # means that block isn't always exactly 1:1 against the ASR side
        # (real confirmed case: a spilled-over next-line word "I'm" rode
        # along with the real mismatch "favorites" in the same block) --
        # match_block_to_candidates tries every candidate in the block and
        # keeps the best-scoring one, not just an already-1:1 slice.
        matched_local = match_block_to_candidates(mxl_norm_line, asr_in_window)

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
        t0, t1 = lrc_line_window(lrc_lines, li)
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
    slots.

    The MXL score's own per-note syllable split (`syllabic="begin"/
    "middle"/"end"`) is the musically-correct answer -- it's the
    composer/engraver's own notated syllabification, not a guess -- and
    is used DIRECTLY whenever its concatenation matches `clean_text` (the
    matched LRC token, see `assign_words_to_lines`) after normalization.
    This matters because `syllables.hyphenate` (a print-hyphenation
    dictionary, tuned for legal line-wrap points, not phonetic syllable
    boundaries) frequently disagrees with real singing syllabification --
    confirmed real cases: "never" -> hyphenate gives "nev"/"er" instead of
    the notated "ne"/"ver"; "Lucifer" -> hyphenate gives only "Lu"/"cifer"
    (2 pieces) instead of the notated "Lu"/"ci"/"fer" (3); "fugitive" ->
    "fugi"/"tive" (2) instead of "fu"/"gi"/"tive" (3) -- the last two
    aren't just wrong BREAK POINTS, they're the wrong SYLLABLE COUNT,
    which used to also trigger melisma-padding onto a musically-real
    syllable (see `Removed / rejected` note: none removed, but this was
    the "Stars" symptom the user reported 2026-08-18).

    When the MXL's own text DOESN'T match clean_text -- i.e. the MXL's
    own OCR is actually wrong (e.g. "systern"/"eystern" for "system") --
    its individual LETTERS can't be trusted, but its own per-syllable
    relative LENGTHS still reflect the real notated split, so
    `_slice_by_weights` uses them to slice the corrected clean word at
    the matching position -- still basing the split on the MXL's own
    notes, never a linguistic hyphenation guess. Only falls back to
    `hyphenate`+reconcile when that's not possible (the clean word is too
    short for one character per syllable -- a genuine case where the
    MXL's own note count can't be matched to real letters at all).
    Falls back to the MXL's own raw syllable text as a last resort when
    no clean match exists at all.

    Real bug (Les Miserables - Stars, 2026-08-18): a MELISMA (ONE real
    notated syllable held across several notes -- MXL's own untexted tied
    /slurred continuation notes, `mxl_syllable_texts` mostly "") is
    structurally NOT a multi-syllable word, even though it has multiple
    slots. When the MXL's own single real syllable is "flame," but the
    matched clean LRC token is "flames" (a genuine, expected spelling/
    inflection difference, not OCR garbage), the exact-match check above
    correctly declines (spelling differs) -- but `_slice_by_weights` then
    WRONGLY treated the melisma's 3 empty continuation slots as real
    syllable slots to redistribute "flames"'s own letters into, producing
    "fla"/"m"/"e"/"s" (a real word torn into meaningless fragments, not
    even valid text on any note). Fixed: when the MXL's own structure has
    AT MOST ONE real (non-empty) syllable slot -- the defining shape of a
    melisma, never a real multi-syllable word -- the whole clean word
    goes on that one real slot unsliced, and every other slot melisma-
    pads, regardless of spelling match. Slicing only ever applies when
    there are genuinely 2+ real notated syllables to redistribute.

    Two more real bugs (Great Big Sea - Ordinary Day, 2026-08-18), both
    with 2+ real syllable slots (the case above doesn't cover them):
      - A syllable-count-only MELISMA WITHIN a multi-syllable word (e.g.
        "al"+""+"right." -- "al" held across 2 notes, THEN a real second
        syllable "right.") still had its middle empty slot treated as a
        real slicing target. When the matched clean text was a genuinely
        DIFFERENT word ("all" for the MXL's own "alright.", not OCR noise
        on the SAME word), slicing "all" across all 3 raw slot weights
        produced "a"/"l"/"l" -- an invented extra syllable AND a doubled
        letter, losing the "~" melisma marker entirely.
      - A clean LRC token containing an internal SPACE ("all right," for
        the MXL's own single word "alright,") got its raw characters
        (space included) sliced across syllable boundaries, producing
        "al"/"l right," -- a literal space glued into the middle of a
        display syllable.
      Fixed with two changes: (1) slicing is now restricted to the REAL
      (non-empty) slots only, by their own weights -- every originally-
      empty slot always melisma-pads, never receives sliced letters, no
      matter how many real slots there are. (2) Before trusting
      clean_text for character-level slicing at all, a sanity gate now
      requires it to plausibly be THE SAME WORD as the MXL's own text:
      no internal whitespace (multiple real LRC words for one MXL
      word's own syllable slots is a genuine word-count mismatch, not a
      spelling variant to slice), and a `MXL_LRC_FUZZY_TEXT_MIN_RATIO`
      similarity to the MXL's own joined text (same bar
      `assign_words_to_lines` already uses for "is this really the same
      word", not a new threshold). Failing either falls back to the
      MXL's own raw syllable text -- still safer than guessing at a
      word LRC and MXL don't even agree is the same word."""
    def _as_display(texts: List[str]) -> List[str]:
        # A note with no lyric of its own (a tied hold / slurred pitch
        # move within a melisma) is stored as "" -- display it with this
        # project's own melisma-continuation marker, never a blank.
        return [t if t else config.MELISMA_CONTINUATION_TEXT for t in texts]

    def _place_at_real_slots(pieces: List[str], real_idxs: List[int]) -> List[str]:
        result = [config.MELISMA_CONTINUATION_TEXT] * n
        for idx, piece in zip(real_idxs, pieces):
            result[idx] = piece
        return result

    n = len(mxl_syllable_texts)
    if clean_text is None:
        return _as_display(mxl_syllable_texts)

    mxl_joined = _normalize("".join(mxl_syllable_texts))
    clean_norm = _normalize(clean_text)
    if mxl_joined and mxl_joined == clean_norm:
        return _as_display(mxl_syllable_texts)

    real_idxs = [i for i, t in enumerate(mxl_syllable_texts) if t]
    n_real = len(real_idxs)
    if n_real <= 1:
        real_idx = real_idxs[0] if real_idxs else 0
        result = [config.MELISMA_CONTINUATION_TEXT] * n
        result[real_idx] = clean_text
        return result

    is_multi_word = any(c.isspace() for c in clean_text)
    ratio = difflib.SequenceMatcher(None, mxl_joined, clean_norm).ratio() if mxl_joined else 0.0
    if is_multi_word or ratio < config.MXL_LRC_FUZZY_TEXT_MIN_RATIO:
        return _as_display(mxl_syllable_texts)

    real_weights = [max(1, len(mxl_syllable_texts[i])) for i in real_idxs]
    sliced = _slice_by_weights(clean_text, real_weights)
    if sliced is not None:
        return _place_at_real_slots(sliced, real_idxs)

    parts = hyphenate(clean_text)
    if len(parts) > n_real:
        parts = chunk_to_count(parts, n_real)
    return _place_at_real_slots(parts, real_idxs)


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


def calibrate_mxl_syllable_pitch(
    syllables: List[Syllable],
    vocals_path: Optional[str],
    min_calibration_samples: int = config.MUSICXML_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.MUSICXML_MIN_CALIBRATION_CONFIDENCE,
    force_calibration: bool = config.ENABLE_MUSICXML_FORCE_CALIBRATION,
    verbose: bool = True,
    debug_log=None,
) -> Tuple[List[Syllable], Optional[int], float]:
    """Corrects each syllable's PITCH CLASS (never octave or timing)
    against a per-song calibration offset -- the same correction
    `musicxml_reference.apply_musicxml_reference` (pass 4) and
    `pitch_refresh.apply_mxl_pitch_reference` already apply for the
    OTHER two paths that take pitch from a MusicXML file, using the
    exact same shared logic (`musicxml_reference._calibrate_pitch_class`/
    `nearest_pitch_for_class`). THIS path (MXL+LRC primary) skips pass 1
    entirely by design -- unlike those two, it has no independently-
    derived audio pitch to calibrate the MXL's own raw pitch against,
    so without this step a transposed MXL (confirmed real case: BATB's
    own score is +2 semitones off its actual SingStar-ground-truth
    vocal) is written completely uncorrected. Confirmed via real
    end-to-end comparison: 100% timing agreement, 0% pitch-class
    accuracy, every mismatch a uniform +2 semitones, before this fix.

    Runs a SINGLE whole-track pitch-class pass
    (`pitch_refresh.compute_pitch_class_predictions`, same RMVPE-by-
    default source pass 1 uses) over the already-ASR-placed syllables'
    own [start, end) spans -- never a per-word isolated clip (this
    project's own "don't run pitch inference on a tiny isolated clip"
    rule) -- purely to get an independent per-syllable pitch-class
    reading to calibrate the MXL's own pitch against; the syllables'
    own START/END TIMING is never touched here, only `midi_note`.

    A missing `vocals_path` or empty `syllables` is a no-op (returns
    `syllables` unchanged, offset None) -- keeps this function safe to
    call unconditionally without every caller needing its own guard.
    """
    if not vocals_path or not syllables:
        return syllables, None, 0.0

    import librosa

    from .musicxml_reference import _calibrate_pitch_class, nearest_pitch_for_class
    from .pitch_refresh import compute_pitch_class_predictions

    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)
    predicted_pc = compute_pitch_class_predictions(syllables, y, sr)

    calibration_samples: List[Tuple[int, int, float]] = []
    for syl, pred_pc in zip(syllables, predicted_pc):
        if pred_pc is None:
            continue
        mxl_absolute_midi = syl.midi_note + 60  # raw MXL pitch, pre-correction
        our_fake_pitch = pred_pc - 60  # so (our_fake_pitch + 60) % 12 == pred_pc
        calibration_samples.append((our_fake_pitch, mxl_absolute_midi, syl.confidence))

    calibration, confidence, _, skipped_reason = _calibrate_pitch_class(
        calibration_samples, min_calibration_samples, min_calibration_confidence,
        force_calibration, verbose, log_prefix="[mxl-lrc]",
    )
    if calibration is None:
        if verbose:
            print(f"[mxl-lrc] pitch-class calibration skipped: {skipped_reason}")
        return syllables, None, 0.0

    new_syllables = list(syllables)
    n_corrected = 0
    for i, syl in enumerate(new_syllables):
        mxl_p = syl.midi_note + 60
        target_pc = (mxl_p - calibration) % 12
        new_pitch = nearest_pitch_for_class(syl.midi_note, target_pc)
        if new_pitch == syl.midi_note:
            continue
        n_corrected += 1
        new_syllables[i] = replace(
            syl, midi_note=new_pitch,
            confidence=max(syl.confidence, config.MUSICXML_CORRECTED_CONFIDENCE),
        )
        if debug_log is not None:
            debug_log.line(f"[mxl-lrc] {syl.text!r} @ {syl.start:.2f}s: pitch class corrected "
                            f"{syl.midi_note:+d} -> {new_pitch:+d} (target class {target_pc}, "
                            f"calibration {calibration:+d})")

    if verbose:
        print(f"[mxl-lrc] pitch-class calibration: {calibration:+d} semitone(s), {confidence:.0%} "
              f"agreement over {len(calibration_samples)} sample(s), {n_corrected} note(s) corrected")

    return new_syllables, calibration, confidence


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
    pitch_calibration_offset: Optional[int] = None  # semitones; None if skipped, see calibrate_mxl_syllable_pitch
    pitch_calibration_confidence: float = 0.0


def generate_from_mxl_and_lrc(mxl_path: str, artist: str, title: str, audio_duration: float,
                               asr_words: List[Word], forced_candidate: Optional[LrcLibCandidate] = None,
                               preferred_part_name: Optional[str] = None,
                               vocals_path: Optional[str] = None, debug_log=None) -> MxlLrcResult:
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

    word_lines, word_clean_text, word_group, word_group_text = assign_words_to_lines(mxl_words, lrc_match.lrc_lines)
    word_starts, word_ends, quality = place_words_via_asr(mxl_words, word_lines, lrc_match.lrc_lines, asr_words,
                                                            word_clean_text=word_clean_text,
                                                            word_group=word_group,
                                                            word_group_text=word_group_text)
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

    # Only calibrated once both quality gates above have passed -- pitch
    # calibration loads the whole vocal track and runs a real pitch-source
    # inference pass over it, not worth paying for on a result that's
    # about to be discarded in favor of the standard fallback pipeline
    # anyway. See `calibrate_mxl_syllable_pitch`'s own docstring for why
    # this path needs its own calibration step at all (unlike pass 4 and
    # pitch_refresh.py, it has no independent audio-derived pitch of its
    # own to calibrate the MXL's raw pitch against otherwise).
    syllables, pitch_cal_offset, pitch_cal_confidence = calibrate_mxl_syllable_pitch(
        syllables, vocals_path, debug_log=debug_log,
    )

    return MxlLrcResult(success=True, reason="", syllables=syllables, quality=quality,
                         lrc_match=lrc_match, mxl_path=mxl_path, part_names_used=part_names,
                         time_calibration=time_cal, pitch_calibration_offset=pitch_cal_offset,
                         pitch_calibration_confidence=pitch_cal_confidence)


def try_mxl_lrc_primary(mxl_paths: List[str], artist: str, title: str, audio_duration: float,
                         asr_words: List[Word], forced_candidate: Optional[LrcLibCandidate] = None,
                         preferred_part_name: Optional[str] = None,
                         vocals_path: Optional[str] = None, debug_log=None) -> Optional[MxlLrcResult]:
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
            vocals_path=vocals_path, debug_log=debug_log,
        )
        if result.success:
            return result
        last_result = result
    return last_result
