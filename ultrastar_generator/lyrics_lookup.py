"""Online lyric lookup, used for two things per the current requirements:

  1. Correcting ASR mistranscriptions -- e.g. whisper hearing "is" where
     the singer actually sang "his". This is NOT gated on ASR confidence
     (an earlier version only tried to fix low-confidence words, which
     missed cases like "is"/"his" where the wrong word was transcribed
     with perfectly normal confidence, since "is" is a perfectly
     ordinary word on its own). Instead, the WHOLE ASR word sequence is
     aligned against the WHOLE reference word sequence with
     difflib.SequenceMatcher, the same technique used for word-error-rate
     scoring, and any ASR word matched to a different reference word gets
     its text swapped in (timing is never touched here).
  2. Determining phrase/line breaks -- every '\\n' in the reference lyrics
     is a real phrase boundary. Each reference word is tagged with which
     line it came from; that line id rides along onto whichever ASR word
     it gets matched to, and phrasing.py forces a line break wherever an
     aligned word's line id changes.

Two sources, tried in order:
  - LRCLIB (lrclib.net, free, no key required) -- tried FIRST. Has a real
    search API (artist_name/track_name query params, returns several
    candidates to choose from) rather than lyrics.ovh's rigid single-shot
    "/artist/title" path, and often has synced (per-line-timestamped)
    lyrics (`LyricsResult.synced_lyrics`, LRC format) -- not consumed by
    anything yet, but available for a future timing-anchored use.
  - lyrics.ovh -- fallback, tried only if LRCLIB has nothing usable.

This never touches note timing from the plain-lyrics path -- reference
lyrics text has no timing information at all, only text and line
structure. If both lookups fail (no network, song not found, etc.)
everything downstream just falls back to ASR text and gap-based phrasing,
same as if this were disabled.
"""

from __future__ import annotations

import difflib
import re
import urllib.parse
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from . import config
from .models import Word
from .syllables import hyphenate

# Lines that are pure section/annotation markers (e.g. "[Chorus]",
# "[Verse 2]") or a source's occasional trailing credit line, not actual
# sung content -- these get filtered out before line numbering.
_ANNOTATION_LINE_RE = re.compile(r"^\s*\[.*\]\s*$")
_CREDIT_LINE_RE = re.compile(r"^\s*(paroles|lyrics powered by|www\.)", re.IGNORECASE)


@dataclass
class LyricsResult:
    """A fetched reference-lyrics candidate, from whichever source
    answered first."""
    plain_lyrics: str
    synced_lyrics: Optional[str] = None  # LRC format ("[mm:ss.xx]text" per
                                          # line), LRCLIB only -- None from
                                          # lyrics.ovh, or when LRCLIB has
                                          # no synced version for this song.
    source: str = ""  # "lrclib" or "lyrics.ovh", for diagnostics/logging.


@dataclass
class LrcLibCandidate:
    """One raw LRCLIB search result -- richer than LyricsResult (keeps
    display metadata + `instrumental`/raw `duration`) so a human can
    browse/pick between candidates in the GUI's search popup, not just
    accept whichever one `_fetch_from_lrclib`'s own scoring would have
    auto-picked."""
    track_name: str
    artist_name: str
    album_name: str
    duration: Optional[float]
    plain_lyrics: str
    synced_lyrics: Optional[str]
    instrumental: bool
    id: Optional[int] = None  # LRCLIB's own numeric id -- lets a user who
                               # browsed lrclib.net directly and confirmed a
                               # perfect match paste the id back in, bypassing
                               # search/scoring entirely (see fetch_lrclib_by_id).

    def to_lyrics_result(self) -> "LyricsResult":
        return LyricsResult(plain_lyrics=self.plain_lyrics, synced_lyrics=self.synced_lyrics, source="lrclib")


def search_lrclib(artist: str = "", title: str = "", q: str = "") -> List[LrcLibCandidate]:
    """Raw LRCLIB search -- returns EVERY result LRCLIB gives back,
    unfiltered (including instrumental/lyric-less candidates, clearly
    tagged), for a human to browse in the GUI's manual search popup.
    Never picks a winner itself -- see `_fetch_from_lrclib` for the
    automatic-pick path built on top of this. Returns [] on any failure
    (no `requests`, no network, no results) -- best-effort only.

    `q`, if given, uses LRCLIB's own broader free-text search (matches
    across track/artist/album together) INSTEAD of the artist/title
    fields -- LRCLIB's API treats `q` as an alternative to `artist_name`/
    `track_name`, not something combined with them."""
    try:
        import requests
    except ImportError:
        return []
    params = {"q": q} if q else {"artist_name": artist, "track_name": title}
    try:
        resp = requests.get(
            "https://lrclib.net/api/search",
            params=params,
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        raw = resp.json()
    except Exception:
        return []
    if not raw:
        return []
    return [
        LrcLibCandidate(
            track_name=c.get("trackName") or "",
            artist_name=c.get("artistName") or "",
            album_name=c.get("albumName") or "",
            duration=c.get("duration"),
            plain_lyrics=c.get("plainLyrics") or "",
            synced_lyrics=c.get("syncedLyrics") or None,
            instrumental=bool(c.get("instrumental")),
            id=c.get("id"),
        )
        for c in raw
    ]


def fetch_lrclib_by_id(lrclib_id: int) -> Optional[LrcLibCandidate]:
    """Fetches ONE specific LRCLIB entry directly by its numeric id
    (`GET /api/get/<id>`), bypassing search/scoring entirely -- for a user
    who browsed lrclib.net themselves, confirmed a specific recording is a
    perfect match (e.g. by ear, against a linked video), and wants that
    exact entry used with no ambiguity. Same best-effort failure convention
    as `search_lrclib`: returns None on any failure (no `requests`, no
    network, non-200, bad id) rather than raising."""
    try:
        import requests
    except ImportError:
        return None
    try:
        resp = requests.get(f"https://lrclib.net/api/get/{lrclib_id}", timeout=8)
        if resp.status_code != 200:
            return None
        c = resp.json()
    except Exception:
        return None
    if not c or "id" not in c:
        return None
    return LrcLibCandidate(
        track_name=c.get("trackName") or "",
        artist_name=c.get("artistName") or "",
        album_name=c.get("albumName") or "",
        duration=c.get("duration"),
        plain_lyrics=c.get("plainLyrics") or "",
        synced_lyrics=c.get("syncedLyrics") or None,
        instrumental=bool(c.get("instrumental")),
        id=c.get("id"),
    )


def _real_lrclib_candidates(candidates: List[LrcLibCandidate],
                             duration_sec: Optional[float]) -> List[LrcLibCandidate]:
    """Filters to candidates worth treating as genuine options for
    ambiguity-prompt purposes: not instrumental, has real lyrics, and --
    if both durations are known -- not wildly off from our own audio's
    length (a generous 3x the normal scoring tolerance, just to exclude
    obviously-different recordings, not to pick a winner)."""
    real = []
    for c in candidates:
        if c.instrumental or not c.plain_lyrics:
            continue
        if duration_sec is not None and c.duration:
            if abs(c.duration - duration_sec) > 3 * config.LRCLIB_DURATION_TOLERANCE_SEC:
                continue
        real.append(c)
    return real


def _score_lrclib_candidate(c: LrcLibCandidate, duration_sec: Optional[float]) -> float:
    """Same scoring `_fetch_from_lrclib` always used, factored out so
    both the automatic-pick path and (if it ever needs one) a caller can
    share one definition of "best"."""
    if c.instrumental or not c.plain_lyrics:
        return -1.0
    s = 0.0
    if duration_sec is not None and c.duration:
        diff = abs(c.duration - duration_sec)
        s += max(0.0, 1.0 - diff / config.LRCLIB_DURATION_TOLERANCE_SEC)
    if c.synced_lyrics:
        s += 0.1
    return s


def _fetch_from_lrclib(
        artist: str, title: str, duration_sec: Optional[float] = None,
        on_ambiguous: Optional[Callable[[List[LrcLibCandidate]], Optional[LrcLibCandidate]]] = None,
) -> Optional[LyricsResult]:
    """Searches LRCLIB and picks the best candidate.

    LRCLIB can return multiple candidates for the same artist/title
    (different recordings, albums, or -- same failure mode as lyrics.ovh
    hit for Gaston this session -- an occasional wrong-language mistag).
    Duration closeness to OUR OWN audio is the main automatic
    disambiguator: an instrumental-only or lyric-less candidate is
    excluded outright, and a candidate whose duration is far from ours is
    heavily penalized (but not excluded -- still better than nothing if
    it's the only candidate). A small bonus favors a candidate that also
    has synced lyrics, since that's strictly more useful when a
    duration-tie needs breaking.

    `on_ambiguous`, if given, is called with the "real" (already
    instrumental/no-lyrics/wildly-off-duration filtered) candidates
    whenever there's more than one -- letting a human (the GUI's
    ambiguity-prompt checkbox) pick instead of trusting the automatic
    score. A returned candidate is used directly; returning None (user
    cancelled) falls through to the normal automatic pick below.
    """
    candidates = search_lrclib(artist, title)
    if not candidates:
        return None

    if on_ambiguous is not None:
        real = _real_lrclib_candidates(candidates, duration_sec)
        if len(real) > 1:
            chosen = on_ambiguous(real)
            if chosen is not None:
                return chosen.to_lyrics_result()

    best = max(candidates, key=lambda c: _score_lrclib_candidate(c, duration_sec))
    if _score_lrclib_candidate(best, duration_sec) < 0:
        return None
    return best.to_lyrics_result()


def _fetch_from_lyrics_ovh(artist: str, title: str) -> Optional[LyricsResult]:
    """Fetches raw lyric text from the free lyrics.ovh API. Returns None
    on any failure (network, not found, etc.) -- this is best-effort only.

    Artist/title are percent-encoded (lyrics.ovh's own docs show this,
    and it matters in practice: e.g. "Les Mis\u00e9rables" needs to become
    "Les%20Mis%C3%A9rables" or the request 404s).
    """
    try:
        import requests
    except ImportError:
        return None

    url = f"https://api.lyrics.ovh/v1/{urllib.parse.quote(artist)}/{urllib.parse.quote(title)}"
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        data = resp.json()
        lyrics = data.get("lyrics")
    except Exception:
        return None
    if not lyrics:
        return None
    return LyricsResult(plain_lyrics=lyrics, synced_lyrics=None, source="lyrics.ovh")


def fetch_reference_lyrics(
        artist: str, title: str, duration_sec: Optional[float] = None,
        on_ambiguous: Optional[Callable[[List[LrcLibCandidate]], Optional[LrcLibCandidate]]] = None,
) -> Optional[LyricsResult]:
    """Tries LRCLIB first, falls back to lyrics.ovh if LRCLIB has nothing
    usable. `duration_sec` (our own audio's length) helps LRCLIB's search
    disambiguate between same-title candidates; pass it when available.
    `on_ambiguous`, if given, lets a human resolve a genuinely ambiguous
    LRCLIB result (see `_fetch_from_lrclib`'s own docstring) -- never set
    outside the GUI, so the CLI's own behavior is completely unchanged.
    Returns None if neither source has anything -- best-effort only.
    """
    result = _fetch_from_lrclib(artist, title, duration_sec, on_ambiguous=on_ambiguous)
    if result is not None:
        return result
    return _fetch_from_lyrics_ovh(artist, title)


def reference_match_ratio(ref_lines: List[str], words: List[Word]) -> float:
    """The raw vocabulary-overlap ratio underlying `reference_matches_transcript`
    -- factored out so a caller that needs the actual NUMBER, not just a
    pass/fail against `REFERENCE_LYRICS_MIN_MATCH_RATIO`'s deliberately
    lenient "is this even the right song" bar, can reuse the exact same
    computation. Used by main.py's ASR-quality retry check (see
    `config.RETRY_ASR_MIN_REFERENCE_MATCH_RATIO`), which asks a stricter
    question -- "did ASR transcribe it well" -- against a reference that's
    already cleared the lower bar."""
    if not ref_lines or not words:
        return 0.0
    ref_norm, _, _ = _tokenize_lines(ref_lines)
    asr_norm = [_normalize(w.text) for w in words if _normalize(w.text)]
    if not ref_norm or not asr_norm:
        return 0.0
    return difflib.SequenceMatcher(None, asr_norm, ref_norm, autojunk=False).ratio()


def largest_unmatched_reference_run(ref_lines: List[str], words: List[Word]) -> int:
    """Size (in reference words) of the LARGEST contiguous run of reference
    words with NO corresponding ASR word at all -- a difflib 'insert'
    opcode in the SAME whole-sequence alignment `align_words_to_reference`
    itself uses. `align_words_to_reference` already recognizes this exact
    case (see its own 'tag == "insert"' comment: "reference has word(s)
    ASR completely missed... simply not represented") but has nothing to
    DO with an insert block -- there's no ASR word to attach a corrected
    text/timing to, so it's silently dropped from the final output with no
    signal anywhere.

    This is a materially different failure than a low match ratio spread
    thinly across many individually-mistranscribed words (what
    `reference_match_ratio` measures): it's ASR dropping a whole real
    passage outright (real case: "Trixie Mattel - Gold", the reference's
    "Doo doo doo doo doo. They start to play." is transcribed correctly at
    one repeat of this chorus later in the song but produces ZERO ASR
    words at an earlier, otherwise-identical repeat -- an aggregate ratio
    over the whole 326-word transcript stayed well above the retry bar
    even though this one passage was completely missing end to end). A
    long insert run is a precise, sharp fingerprint for exactly that,
    independent of how well the REST of the song transcribed."""
    if not ref_lines or not words:
        return 0
    ref_norm, _, _ = _tokenize_lines(ref_lines)
    asr_norm = [_normalize(w.text) for w in words]
    if not ref_norm or not asr_norm:
        return 0
    matcher = difflib.SequenceMatcher(None, asr_norm, ref_norm, autojunk=False)
    return max((b1 - b0 for tag, a0, a1, b0, b1 in matcher.get_opcodes() if tag == "insert"), default=0)


def reference_matches_transcript(ref_lines: List[str], words: List[Word],
                                  min_ratio: float = config.REFERENCE_LYRICS_MIN_MATCH_RATIO) -> bool:
    """Sanity-checks a fetched reference against the ASR transcript's OWN
    vocabulary before it's trusted at all -- catches a wrong-song or
    wrong-language reference (confirmed real case: Gaston's lyrics.ovh
    lookup silently returned Spanish lyrics for an English song) that
    would otherwise get treated as ground truth for TEXT and corrupt the
    whole song. A right reference and a same-audio ASR transcript should
    share the large majority of their words; a wrong one shares almost
    none. Source-independent by design -- applies whichever source
    answered.
    """
    return reference_match_ratio(ref_lines, words) >= min_ratio


def parse_lyrics_lines(raw_lyrics: str) -> List[str]:
    """Splits raw lyrics text into cleaned lines: strips blank lines,
    section-annotation-only lines ("[Chorus]"), and lyrics.ovh's
    occasional trailing credit line. Preserves original wording/casing
    of everything else, and preserves line order (which is what phrase
    breaks are keyed on)."""
    if not raw_lyrics:
        return []
    lines = []
    for raw_line in raw_lyrics.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _ANNOTATION_LINE_RE.match(line):
            continue
        if _CREDIT_LINE_RE.match(line):
            continue
        lines.append(line)
    return lines


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9']", "", s.lower())


def _tokenize_lines(lines: List[str]) -> Tuple[List[str], List[str], List[int]]:
    """Returns (normalized_tokens, original_tokens, line_ids) -- three
    parallel lists, one entry per word across all lines.

    A hyphenated token is split into its own separate tokens (2026-08-10)
    -- lyrics sites commonly write a repeated ad-lib/backing-vocal
    syllable as ONE hyphenated token ("Do-do-do-do-do" for 5 separately
    sung "do"s, real case: Trixie Mattel - Gold), which otherwise counts
    as a single reference word no matter how many real sung words it
    represents. This under-counted severity for
    `largest_unmatched_reference_run` (a whole dropped "Do-do-do-do-do"
    passage only ever scored 1, never enough to clear a sane threshold)
    and meant `align_words_to_reference`'s own alignment could only ever
    match/miss the whole 5-syllable blob as one unit rather than as 5
    separate words -- `align_words_to_reference`'s own uneven-"replace"-
    block handling already has a dedicated `is_repeat_clamp` path for
    when several ASR words map onto one repeated reference token; this
    split makes that the FALLBACK case (ASR's own word count still
    doesn't match) rather than the every-time case."""
    norm_tokens: List[str] = []
    orig_tokens: List[str] = []
    line_ids: List[int] = []
    for line_idx, line in enumerate(lines):
        for tok in line.split():
            parts = tok.split("-") if "-" in tok else [tok]
            for part in parts:
                n = _normalize(part)
                if not n:
                    continue
                norm_tokens.append(n)
                orig_tokens.append(part)
                line_ids.append(line_idx)
    return norm_tokens, orig_tokens, line_ids


def align_words_to_reference(words: List[Word], reference_lines: List[str]) -> List[Word]:
    """Aligns the full ASR word sequence against the full reference lyric
    sequence (whole-sequence alignment, not a per-word confidence-gated
    lookup), and returns a new word list where:
      - a word matched to a different-but-similar reference word gets its
        TEXT corrected (timing untouched)
      - every matched word gets `line_id` set to its reference line index
      - unmatched ASR words (e.g. ad-libs the reference doesn't have)
        keep their original text and inherit the nearest matched
        neighbor's line_id, so phrase breaks still work around them
    """
    if not reference_lines or not words:
        return words

    ref_norm, ref_orig, ref_line_ids = _tokenize_lines(reference_lines)
    if not ref_norm:
        return words

    asr_norm = [_normalize(w.text) for w in words]

    matcher = difflib.SequenceMatcher(None, asr_norm, ref_norm, autojunk=False)
    out: List[Optional[Word]] = [None] * len(words)

    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(a1 - a0):
                w = words[a0 + k]
                out[a0 + k] = Word(text=w.text, start=w.start, end=w.end,
                                    confidence=w.confidence, line_id=ref_line_ids[b0 + k],
                                    reference_text=ref_orig[b0 + k])
        elif tag == "replace":
            a_len = a1 - a0
            b_len = b1 - b0
            # 1:1 replace is the common case (a single mistranscribed
            # word) -- swap text in directly.
            if a_len == b_len:
                for k in range(a_len):
                    w = words[a0 + k]
                    ref_word = ref_orig[b0 + k]
                    sim = difflib.SequenceMatcher(None, asr_norm[a0 + k], ref_norm[b0 + k]).ratio()
                    new_text = ref_word if sim >= 0.3 else w.text
                    if new_text != w.text and w.text[:1].isupper() and new_text[:1].islower():
                        new_text = new_text[:1].upper() + new_text[1:]
                    out[a0 + k] = Word(text=new_text, start=w.start, end=w.end,
                                        confidence=w.confidence, line_id=ref_line_ids[b0 + k],
                                        reference_text=ref_word)
            else:
                # Uneven block (ASR split/merged words differently than
                # the reference does) -- keep ASR text as-is rather than
                # guess a risky word-for-word mapping, but still tag every
                # ASR word in the block with a reference line id (so
                # phrase breaks stay correct) AND a best-guess
                # reference_text (clamped to the block) -- not trusted
                # enough to substitute directly here, but a real signal
                # verification.py can cross-check a fresh, isolated
                # re-transcription against.
                #
                # When a_len > b_len, min(b0+k, b1-1) clamps several ASR
                # words onto the SAME single reference token -- fine for an
                # ordinary word (verification.py just won't confirm it),
                # but a real confirmed bug for a hyphenated repeated-unit
                # token like LRC's "Do-do-do-do-do" (one written token
                # standing in for 5 separately-sung "do"s, one per ASR
                # word here): every one of those ASR words was getting the
                # WHOLE 5-syllable blob as its reference_text, which
                # verification.py's fallback ("neither confirms, trust
                # reference") then stamped onto each word's final text
                # verbatim -- multiplying one 5-syllable phrase into 5
                # duplicate copies instead of splitting it across the 5
                # words it actually spans. Fixed by handing out ONE
                # hyphen-part of that token per repeated ASR word (reusing
                # hyphenate() -- same tool lyric_alignment.py already uses
                # to split a word across multiple notes, applied here at
                # the reference-token level instead). A token that isn't
                # itself hyphenated into >1 part (the common case) is
                # unaffected -- this only changes behavior for the
                # repeated-clamp case.
                b_idx_counts: dict = {}
                for k in range(a_len):
                    b_idx_counts[min(b0 + k, b1 - 1)] = b_idx_counts.get(min(b0 + k, b1 - 1), 0) + 1
                part_cache: dict = {}
                part_cursor: dict = {}
                for k in range(a_len):
                    w = words[a0 + k]
                    b_idx = min(b0 + k, b1 - 1)
                    is_repeat_clamp = b_idx_counts[b_idx] > 1
                    # Guard specifically for the repeat-clamp case: a real
                    # confirmed bug had an unrelated ASR word ~10.6s later,
                    # in silence, swept into this same block purely
                    # because it was the last thing before the whole-song
                    # sequence ran out on both sides -- never the same
                    # repeated-token run as its (much closer together)
                    # neighbors. See config.REFERENCE_CLAMP_MAX_GAP_SEC.
                    if is_repeat_clamp and k > 0 and (w.start - words[a0 + k - 1].end) > config.REFERENCE_CLAMP_MAX_GAP_SEC:
                        out[a0 + k] = Word(text=w.text, start=w.start, end=w.end,
                                            confidence=w.confidence, line_id=None)
                        continue
                    ref_text = ref_orig[b_idx]
                    if is_repeat_clamp:
                        if b_idx not in part_cache:
                            part_cache[b_idx] = hyphenate(ref_text)
                            part_cursor[b_idx] = 0
                        parts = part_cache[b_idx]
                        if len(parts) > 1:
                            p = min(part_cursor[b_idx], len(parts) - 1)
                            ref_text = parts[p]
                            part_cursor[b_idx] += 1
                    out[a0 + k] = Word(text=w.text, start=w.start, end=w.end,
                                        confidence=w.confidence, line_id=ref_line_ids[b_idx],
                                        reference_text=ref_text)
        elif tag == "delete":
            # ASR has word(s) the reference doesn't (ad-libs, hallucinated
            # filler, etc.) -- keep as-is; line_id filled in below from
            # neighboring context. No reference word exists to check
            # against at all.
            for k in range(a1 - a0):
                w = words[a0 + k]
                out[a0 + k] = Word(text=w.text, start=w.start, end=w.end, confidence=w.confidence, line_id=None)
        # tag == "insert": reference has word(s) ASR completely missed --
        # nothing to attach them to (no ASR timing exists for them), so
        # they're simply not represented. Their line boundary is still
        # captured by whatever comes before/after in the ASR sequence.

    # Fill any remaining unmatched (line_id is None) words by inheriting
    # from the nearest matched neighbor -- prefer the previous one so a
    # trailing ad-lib stays attached to the line it's part of.
    last_seen = None
    for i in range(len(out)):
        if out[i].line_id is not None:
            last_seen = out[i].line_id
        else:
            out[i] = Word(text=out[i].text, start=out[i].start, end=out[i].end,
                           confidence=out[i].confidence, line_id=last_seen)

    return out


def alignment_diff_summary(original: List[Word], corrected: List[Word]) -> List[str]:
    """Returns human-readable "word -> word" lines for every word whose
    text actually changed, for diagnostic logging."""
    diffs = []
    for o, c in zip(original, corrected):
        if o.text != c.text:
            diffs.append(f'"{o.text}" -> "{c.text}" (at {o.start:.2f}s)')
    return diffs
