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
from pathlib import Path
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


def load_lrc_file(path, artist: str = "", title: str = "") -> Optional[LrcLibCandidate]:
    """Loads a LOCAL .lrc file as a forced `LrcLibCandidate`, bypassing
    LRCLIB entirely -- same override tier as `--lrclib-id`/`fetch_lrclib_
    by_id` (both feed the same `forced_candidate` parameter downstream),
    for a user who already has a synced-lyrics file on disk (hand-typed,
    sourced elsewhere, or deliberately imprecise for testing -- see
    [[project-uskm-comparison-magic-dance]]-style investigations) and
    wants it used directly with no search/scoring involved. `duration` is
    left `None` (no local audio-duration number to compare against, and
    `select_lrc_candidate` never gates on it for a forced candidate
    anyway -- see its own docstring). Returns `None` on any read/parse
    failure (missing file, no parseable `[mm:ss.xx]` timestamp lines)
    rather than raising, same best-effort convention as the network
    fetchers above."""
    from .lrc_timing import parse_lrc
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    lrc_lines = parse_lrc(text)
    if not lrc_lines:
        return None
    return LrcLibCandidate(
        track_name=title, artist_name=artist, album_name="",
        duration=None, plain_lyrics="\n".join(t for _, t in lrc_lines),
        synced_lyrics=text, instrumental=False, id=None,
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


def recover_dropped_reference_words(ref_lines: List[str], words: List[Word], vocals_path: Path,
                                     *, debug_log=None) -> Tuple[List[Word], int]:
    """PROTOTYPE, see config.FORCE_ALIGN_GAPS's docstring. For each
    contiguous run of reference words with NO corresponding ASR word at
    all (a difflib 'insert' opcode -- the SAME one `largest_unmatched_
    reference_run` measures the size of), forces that KNOWN reference
    text onto the audio window between the nearest ASR-matched
    neighbors, via a real wav2vec2 CTC forced alignment
    (`transcription.force_align_words_in_window`) -- doesn't need ASR to
    have transcribed anything in the gap at all, since it isn't
    searching a transcript, it's measuring where the GIVEN text fits.

    A successful recovery is spliced into the returned word list as a
    new `Word` with real measured timing and text taken directly from
    the reference -- a later `align_words_to_reference` call will then
    match it as an ordinary "equal" opcode (the text matches exactly),
    assigning the correct `line_id` the normal way rather than needing
    special-casing here.

    Returns (new_words, n_recovered) -- `words` itself is never mutated;
    `new_words` is `words` unchanged if there was nothing to recover, or
    nothing could be. Lazily loads the audio/align model at most once,
    and only if there's an actual gap to try."""
    if not ref_lines or not words:
        return words, 0

    ref_norm, ref_orig, ref_line_ids = _tokenize_lines(ref_lines)
    asr_norm = [_normalize(w.text) for w in words]
    if not ref_norm or not asr_norm:
        return words, 0

    matcher = difflib.SequenceMatcher(None, asr_norm, ref_norm, autojunk=False)
    opcodes = matcher.get_opcodes()
    if not any(tag == "insert" for tag, *_ in opcodes):
        return words, 0

    from .transcription import force_align_words_in_window
    import whisperx
    from . import model_cache

    audio = None
    align_model = metadata = None
    audio_duration = 0.0

    out: List[Word] = []
    n_recovered = 0
    for tag, a0, a1, b0, b1 in opcodes:
        if tag != "insert":
            out.extend(words[a0:a1])
            continue

        if audio is None:
            audio = whisperx.load_audio(str(vocals_path))
            align_model, metadata = model_cache.get_whisperx_align_model()
            audio_duration = len(audio) / 16000.0

        win_start = words[a0 - 1].end if a0 > 0 else 0.0
        win_end = words[a1].start if a1 < len(words) else audio_duration
        win_start = max(0.0, min(win_start, audio_duration))
        win_end = max(0.0, min(win_end, audio_duration))

        gap_text = ref_orig[b0:b1]
        gap_line_ids = ref_line_ids[b0:b1]
        aligned = force_align_words_in_window(gap_text, win_start, win_end, align_model, metadata, audio)
        if aligned is None:
            if debug_log is not None:
                debug_log.line(f"  force-align (reference gap): [{win_start:8.3f}-{win_end:8.3f}] "
                                f"({len(gap_text)} word(s) {' '.join(gap_text)!r}) -- no usable result, "
                                f"stays dropped")
            continue

        for text, line_id, (ws, we, score) in zip(gap_text, gap_line_ids, aligned):
            out.append(Word(text=text, start=ws, end=we, confidence=score, line_id=line_id))
            n_recovered += 1
        if debug_log is not None:
            debug_log.line(f"  force-align (reference gap): [{win_start:8.3f}-{win_end:8.3f}] "
                            f"({len(gap_text)} word(s) {' '.join(gap_text)!r}) -- recovered via forced "
                            f"alignment")

    return out, n_recovered


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
    # Curly apostrophes (U+2019, e.g. a real hand-authored existing file's
    # own "Johnny's") must normalize the same as straight ones -- the old
    # regex below only ever kept straight apostrophes, silently deleting
    # every curly one, which desyncs otherwise-identical text (real bug
    # found via lrc_timing.reconcile_line_structure's own real-audio
    # validation, 2026-08-15 -- see that module's identical fix).
    # realign.py/mxl_lrc_generator.py's own _normalize already had this.
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", s)


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


def is_lrc_line_tracking_confident(words: List[Word], synced_lyrics_text: str) -> bool:
    """Whether `assign_lrc_line_ids_sequentially` should be trusted for
    this song at all -- reuses `lrc_timing.two_tier_time_calibration`
    purely as a PLAUSIBILITY gate (is this LRC candidate even the right
    recording?), same as `realign.py`'s `lrc_mode="windowed"` and
    `mxl_lrc_generator.place_words_via_asr` already do; an uncalibrated/
    wrong-recording LRC is known to be actively harmful, not just
    unhelpful, elsewhere in this codebase. The calibrated TIMESTAMPS
    themselves are NOT used for the actual line-splitting -- see
    `assign_lrc_line_ids_sequentially`'s own docstring for why."""
    from .lrc_timing import parse_lrc, match_asr_to_lrc_lines, two_tier_time_calibration

    lrc_lines = parse_lrc(synced_lyrics_text)
    if len(lrc_lines) < 2 or not words:
        return False
    candidates = match_asr_to_lrc_lines(words, lrc_lines)
    _offset, _slope, _confidence, _kind, _skipped, correction_fn, _holdout = two_tier_time_calibration(candidates)
    return correction_fn is not None


def assign_lrc_line_ids_sequentially(words: List[Word], synced_lyrics_text: str) -> Optional[List[Word]]:
    """Sequential, CURSOR-based LRC line assignment -- user's explicit
    design (2026-08-14): "going through each line of the LRC, one by
    one, finding the matching words, and then inserting line breaks
    accordingly." Returns None (caller falls back to the whole-song text
    match) when `is_lrc_line_tracking_confident` says this LRC candidate
    isn't trustworthy for this song at all.

    Walks the reference LINES in order. For each line, searches a
    generously-sized but bounded window of ASR words starting exactly
    where the PREVIOUS line's own match left off (a cursor that only
    ever advances forward -- reused via `align_words_to_reference`'s own
    single-line matching, so all of its existing word-level correction
    logic -- uneven blocks, repeat-clamp, hyphen-split, drop handling --
    applies unchanged), assigns every word up through the last real
    match to this line, and advances the cursor past it.

    This is deliberately NOT time/gap-based (two earlier, REJECTED
    attempts were): the cursor-based design makes both of those
    approaches' failure modes structurally impossible here --
      - A repeated phrase later in the song can never be confused with
        an earlier occurrence, because the cursor has already moved past
        the earlier occurrence's own words by the time a later line's
        search begins -- the later occurrence is simply unreachable.
      - A real line transition with NO audio pause is still split
        correctly, because the boundary comes from how many words THIS
        line's own text actually matched, not from a time boundary or a
        gap in the audio (the real failure mode of both earlier
        attempts: Video Games' "...with you" -> "Tell me..." transition
        has only a 0.14s gap, invisible to any gap-based grouping).
    """
    from .lrc_timing import parse_lrc

    lrc_lines = parse_lrc(synced_lyrics_text)
    if not is_lrc_line_tracking_confident(words, synced_lyrics_text):
        return None

    # Deliberately TIGHT -- a too-generous window (an earlier version used
    # 4x+15) can reach far enough into unrelated later content that a
    # coincidentally-shared common word (e.g. "on" appearing in both this
    # line and a much later one) gets matched too, making the cursor jump
    # far past this line's own real content. A modest multiplier plus a
    # small flat slack is enough to tolerate normal ASR noise (extra
    # hallucinated words, a few dropped/garbled words) without reaching
    # into a DIFFERENT line's own vocabulary.
    WINDOW_WORD_MULTIPLIER = 2
    WINDOW_WORD_SLACK = 8

    cursor = 0
    n = len(words)
    out: List[Word] = []
    for li, (_t, line_text) in enumerate(lrc_lines):
        line_word_count = max(1, len(line_text.split()))
        window_end = min(cursor + line_word_count * WINDOW_WORD_MULTIPLIER + WINDOW_WORD_SLACK, n)
        window_words = words[cursor:window_end]
        if not window_words:
            continue
        aligned = align_words_to_reference(window_words, [line_text])
        # Real match = has a reference_text, NOT just a non-None line_id.
        # The single-line recursive call above falls through to
        # align_words_to_reference's OWN "no synced_lyrics_text" fallback
        # path when it returns (no calibration to gate on for a single
        # line) -- its "fill remaining unmatched words from nearest
        # matched neighbor" step back-fills every line_id=None word in
        # the window (a real "delete"-tag rejection, e.g. the NEXT line's
        # own vocabulary bleeding into this window) with line_id=0
        # anyway, since there's nothing else in a single-line reference
        # for it to inherit. That made line_id alone useless as a stop
        # signal -- it was ALWAYS set by the time this loop saw it,
        # regardless of whether the word was ever a real match.
        # reference_text is NOT touched by that fallback (it only fills
        # line_id), so it still reliably tells real matches apart from
        # window overflow into the next line's own content.
        last_matched_offset = None
        for i, w in enumerate(aligned):
            if w.reference_text is not None:
                last_matched_offset = i
        if last_matched_offset is None:
            # No real match for this line anywhere in the window -- don't
            # advance the cursor; the NEXT line's search starts from the
            # same position (this line contributes 0 words rather than
            # guessing).
            continue
        for w in aligned[:last_matched_offset + 1]:
            out.append(w if w.line_id is None else Word(
                text=w.text, start=w.start, end=w.end, confidence=w.confidence,
                line_id=li, reference_text=w.reference_text, dropped=w.dropped,
            ))
        cursor += last_matched_offset + 1

    # Trailing content after the LAST line's own window (e.g. an outro
    # ad-lib) -- keep it, unmatched (line_id=None), preserving order and
    # total word count.
    out.extend(words[cursor:])
    return out


def align_words_to_reference(
    words: List[Word], reference_lines: List[str], synced_lyrics_text: Optional[str] = None,
) -> List[Word]:
    """Aligns the full ASR word sequence against the full reference lyric
    sequence (whole-sequence alignment, not a per-word confidence-gated
    lookup), and returns a new word list where:
      - a word matched to a different-but-similar reference word gets its
        TEXT corrected (timing untouched)
      - every matched word gets `line_id` set to its reference line index
      - an ASR word with NO reference counterpart at all (a difflib
        "delete" block) is DROPPED entirely, not kept, but ONLY when the
        unmatched run is short (<= config.REFERENCE_DELETE_MAX_RUN) --
        see the "delete" branch below for why a long run is kept instead.
      - a word inside an uneven "replace" block still keeps its own ASR
        text (a real, if imprecisely-mapped, reference correspondence
        exists there -- not the same as having none at all) and inherits
        the nearest matched neighbor's line_id, so phrase breaks still
        work around it.

    Only ever called once `reference_lines` has already cleared the
    caller's own "is this even the right song" gate (see
    `reference_matches_transcript`/`REFERENCE_LYRICS_MIN_MATCH_RATIO` in
    main.py) -- by the time this function runs, the reference is trusted
    as the definitive word LIST for the song, not just a text corrector.

    `synced_lyrics_text` (optional): when given and
    `is_lrc_line_tracking_confident` trusts it, this function's own
    whole-song text match below is BYPASSED ENTIRELY in favor of
    `assign_lrc_line_ids_sequentially` -- see that function's own
    docstring for the sequential, cursor-based design and why it
    replaced two earlier, rejected time/gap-based attempts. Falls
    through to the whole-song match below, completely unchanged,
    whenever no synced lyrics are given or the LRC candidate isn't
    confidently trusted for this song.
    """
    if synced_lyrics_text:
        sequential = assign_lrc_line_ids_sequentially(words, synced_lyrics_text)
        if sequential is not None:
            return sequential

    if not reference_lines or not words:
        return words

    ref_norm, ref_orig, ref_line_ids = _tokenize_lines(reference_lines)
    if not ref_norm:
        return words

    asr_norm = [_normalize(w.text) for w in words]

    matcher = difflib.SequenceMatcher(None, asr_norm, ref_norm, autojunk=False)
    out: List[Word] = []

    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(a1 - a0):
                w = words[a0 + k]
                out.append(Word(text=w.text, start=w.start, end=w.end,
                                 confidence=w.confidence, line_id=ref_line_ids[b0 + k],
                                 reference_text=ref_orig[b0 + k]))
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
                    # Below the similarity bar, this 1:1 positional pairing
                    # isn't trusted enough to even RECORD as a reference
                    # expectation, not just not trusted enough to substitute
                    # text with directly -- a real confirmed bug (Trixie
                    # Mattel - Video Games, 2026-08-13): a hallucinated ASR
                    # word ("You're") landed 1:1 against an unrelated
                    # reference word ("Swingin'") purely by sequence
                    # position, sim far below 0.3, text correctly left
                    # alone here -- but reference_text was still stamped
                    # with "Swingin'" unconditionally, and verification.py's
                    # own "neither confirms, default to reference" fallback
                    # later blindly trusted that bogus tag and overwrote the
                    # word's text with it anyway. Same "don't record a
                    # reference_text you don't trust" principle already
                    # applied to REFERENCE_CLAMP_MAX_REPEAT above.
                    out.append(Word(text=new_text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=ref_line_ids[b0 + k],
                                     reference_text=ref_word if sim >= 0.3 else None))
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
                    repeat_count = b_idx_counts[b_idx]
                    is_repeat_clamp = repeat_count > 1
                    # Guard specifically for the repeat-clamp case: a real
                    # confirmed bug had an unrelated ASR word ~10.6s later,
                    # in silence, swept into this same block purely
                    # because it was the last thing before the whole-song
                    # sequence ran out on both sides -- never the same
                    # repeated-token run as its (much closer together)
                    # neighbors. See config.REFERENCE_CLAMP_MAX_GAP_SEC.
                    #
                    # Checked against BOTH neighbors (missing on one side
                    # counts as "far" on that side, not a pass) -- a
                    # backward-only check missed the SAME failure at the
                    # very START of a clamp run, since the first word has no
                    # previous neighbor to compare against and always fell
                    # through unconditionally. Real confirmed case (Trixie
                    # Mattel - Video Games, 2026-08-13): a non-lyrical audio
                    # intro decoder-hallucinated as "You're welcome." (2
                    # words) landed as the whole song's first 2 ASR words,
                    # clamped onto reference word 0 ("Swingin'") along with
                    # the real "Swinging" ~16s later; "welcome." was already
                    # correctly rejected by the backward check, but "You're"
                    # (k=0, no backward neighbor) was not -- and kept a
                    # bogus reference_text of "Swingin'" that verification.py
                    # later trusted and used to overwrite "You're" outright.
                    # Neighbors are looked up in the GLOBAL `words` sequence
                    # (idx +-1), not clamped to this opcode block's own
                    # a0..a1 range -- a word can be the first/last element of
                    # ITS block while still having a real, close neighbor in
                    # the very next/previous block (e.g. "Swinging" is the
                    # last word of this block, but "in" -- its real, close
                    # neighbor -- is the FIRST word of the next block). Using
                    # a block-relative bound here would have deleted that
                    # real, correctly-transcribed word too.
                    idx = a0 + k
                    far_from_prev = idx == 0 or (w.start - words[idx - 1].end) > config.REFERENCE_CLAMP_MAX_GAP_SEC
                    far_from_next = (idx + 1 >= len(words)
                                      or (words[idx + 1].start - w.end) > config.REFERENCE_CLAMP_MAX_GAP_SEC)
                    if is_repeat_clamp and far_from_prev and far_from_next:
                        # DROPPED (word.dropped=True), not just unlabeled:
                        # this word has no real reference correspondence at
                        # all (same status as a short "delete"-opcode run
                        # above) and is isolated in time from its own clamp
                        # run -- keeping it unlabeled still put real
                        # hallucinated content in the final output. User's
                        # explicit policy (2026-08-14): a hallucinated word
                        # must never reach the final file, not just avoid
                        # being mislabeled as something it isn't. STILL
                        # appended to `out` (not `continue`d past) -- see
                        # Word.dropped's own docstring for why removing it
                        # from the sequence entirely is itself a regression.
                        out.append(Word(text=w.text, start=w.start, end=w.end,
                                         confidence=w.confidence, line_id=None, dropped=True))
                        continue
                    # A second, independent guard (real case: David Bowie -
                    # Magic Dance): this many ASR words clamping onto the
                    # SAME reference token is itself a sign of decoder
                    # hallucination/runaway repetition, not a legitimate
                    # repeated ad-lib -- see config.REFERENCE_CLAMP_MAX_REPEAT's
                    # own comment. Past that count, don't fabricate a
                    # reference_text at all; keep the ASR word's own raw
                    # text, same as the ordinary (non-repeat-clamp) uneven
                    # block default just below.
                    if is_repeat_clamp and repeat_count > config.REFERENCE_CLAMP_MAX_REPEAT:
                        out.append(Word(text=w.text, start=w.start, end=w.end,
                                         confidence=w.confidence, line_id=ref_line_ids[b_idx]))
                        continue
                    ref_text = ref_orig[b_idx]
                    if is_repeat_clamp:
                        if b_idx not in part_cache:
                            part_cache[b_idx] = hyphenate(ref_text)
                            part_cursor[b_idx] = 0
                        parts = part_cache[b_idx]
                        if len(parts) > 1:
                            # Wrap rather than freeze on the last syllable
                            # once the cursor runs past the real syllable
                            # count -- freezing is what turned "mag"/"ic"
                            # into ~89 straight repeats of "ic" in the Magic
                            # Dance case above the (now-capped) repeat count.
                            p = part_cursor[b_idx] % len(parts)
                            ref_text = parts[p]
                            part_cursor[b_idx] += 1
                    out.append(Word(text=w.text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=ref_line_ids[b_idx],
                                     reference_text=ref_text))
        elif tag == "delete":
            # ASR has word(s) with NO reference counterpart anywhere in the
            # whole song. A SHORT run is dropped entirely, not kept -- this
            # used to keep them as-is on the theory that an unmatched word
            # is probably a real ad-lib the reference source just didn't
            # transcribe, but this project's own WhisperX-hallucination
            # history (long-segment rewindow, ASR-quality retry,
            # --no-transcribe -- see CLAUDE.md) shows the far more damaging
            # real case: a non-lyrical stretch of audio (a spoken sample,
            # an intro, near-silence) gets decoded as confident-looking
            # FAKE lyrics with real timestamps, and those became permanent,
            # wrong "lyrics" in the final file. Real confirmed case: Trixie
            # Mattel - Video Games -- WhisperX decoded a non-lyrical audio
            # intro as "You're... welcome.", which used to become the
            # song's own first two lyric words even though nothing in the
            # reference remotely matches them. Once a reference lyrics
            # source is trusted enough to reach this function at all (see
            # this function's own docstring), it's trusted as the
            # definitive word LIST for a SHORT unmatched run -- a real
            # ad-lib the reference happens to omit is an accepted,
            # deliberate tradeoff of this policy (user's explicit
            # decision), not an oversight.
            #
            # A LONG run is the opposite case and is kept (old behavior),
            # per config.REFERENCE_DELETE_MAX_RUN's own docstring: a real
            # confirmed regression on this SAME song showed difflib's
            # global alignment can misclassify a large stretch of
            # genuinely-correct, real reference-matching content (a
            # repeated chorus) as one giant "delete" block -- mass-deleting
            # 219 of 355 words, nearly all of them real, correctly-heard
            # lyrics. A short delete-run is almost always genuine
            # hallucination; a long one is far more likely an alignment
            # failure than a genuinely long non-lyrical passage.
            run_len = a1 - a0
            if run_len > config.REFERENCE_DELETE_MAX_RUN:
                for k in range(run_len):
                    w = words[a0 + k]
                    out.append(Word(text=w.text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=None))
            else:
                # STILL appended to `out` with dropped=True, not omitted --
                # see Word.dropped's own docstring: omitting it from the
                # sequence entirely let a NEIGHBORING real word's pass-1
                # note zone silently swallow this word's own claimed notes
                # instead (real confirmed regression, Video Games,
                # 2026-08-14). lyric_alignment.py is what actually keeps
                # this word's text/notes out of the final output.
                for k in range(run_len):
                    w = words[a0 + k]
                    out.append(Word(text=w.text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=None, dropped=True))
            continue
        # tag == "insert": reference has word(s) ASR completely missed --
        # nothing to attach them to (no ASR timing exists for them), so
        # they're simply not represented. Their line boundary is still
        # captured by whatever comes before/after in the ASR sequence.
        # (`recover_dropped_reference_words`/force_align_gaps, run BEFORE
        # this function, is the real recovery mechanism for this case --
        # by the time we get here, most genuinely-missing content should
        # already be filled in.)

    # Fill any remaining unmatched (line_id is None) words by inheriting
    # from the nearest matched neighbor -- prefer the previous one so a
    # trailing ad-lib stays attached to the line it's part of. Only ever
    # reached here when `synced_lyrics_text` wasn't given/trusted (the
    # confident-LRC case returns early via
    # `assign_lrc_line_ids_sequentially`, above).
    last_seen = None
    for i in range(len(out)):
        if out[i].line_id is not None:
            last_seen = out[i].line_id
        else:
            out[i] = Word(text=out[i].text, start=out[i].start, end=out[i].end,
                           confidence=out[i].confidence, line_id=last_seen, dropped=out[i].dropped)

    return out


def alignment_diff_summary(original: List[Word], corrected: List[Word]) -> List[str]:
    """Returns human-readable "word -> word" (corrected) or "word ->
    [DROPPED]" (no reference counterpart at all -- see
    align_words_to_reference's "delete"/repeat-clamp-gap-guard handling)
    lines for every word that changed, for diagnostic logging. Matches
    original-to-corrected words by (start, end) rather than list
    position/`zip` -- more robust than a positional zip even though
    `corrected` is always the same length/order as `original` now (a
    dropped word is flagged via `Word.dropped`, not omitted -- see its
    own docstring for why omitting it broke downstream note-zone
    boundaries)."""
    corrected_by_span = {(c.start, c.end): c for c in corrected}
    diffs = []
    for o in original:
        c = corrected_by_span.get((o.start, o.end))
        if c is None or c.dropped:
            diffs.append(f'"{o.text}" -> [DROPPED, no reference match] (at {o.start:.2f}s)')
        elif o.text != c.text:
            diffs.append(f'"{o.text}" -> "{c.text}" (at {o.start:.2f}s)')
    return diffs
