"""Online reference-lyrics lookup: corrects ASR mistranscriptions via
whole-sequence diff against reference text, and tags words with
reference line ids to drive phrase/line breaks.

Sole source: LRCLIB. A candidate must have synced (per-line-timestamped)
lyrics to count; otherwise treated as no candidate at all.

Never touches timing. On lookup failure, downstream falls back to ASR
text and gap-based phrasing.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from . import config
from .models import Word
from .syllables import hyphenate
from .text_normalize import normalize_word as _normalize

# Section markers ("[Chorus]") and trailing credit lines -- filtered out, not sung content.
_ANNOTATION_LINE_RE = re.compile(r"^\s*\[.*\]\s*$")
_CREDIT_LINE_RE = re.compile(r"^\s*(paroles|lyrics powered by|www\.)", re.IGNORECASE)


@dataclass
class LyricsResult:
    """A fetched reference-lyrics candidate, always from LRCLIB."""
    plain_lyrics: str
    synced_lyrics: Optional[str] = None  # LRC format; optional since a pinned override may lack it.
    source: str = ""  # "lrclib", for diagnostics/logging.
    # Display metadata carried through from the source LrcLibCandidate, for logging.
    track_name: str = ""
    artist_name: str = ""
    lrclib_id: Optional[int] = None
    duration: Optional[float] = None


@dataclass
class LrcLibCandidate:
    """One raw LRCLIB search result -- lets the GUI's search popup let a user browse/pick manually."""
    track_name: str
    artist_name: str
    album_name: str
    duration: Optional[float]
    plain_lyrics: str
    synced_lyrics: Optional[str]
    instrumental: bool
    dupe_count: int = 0
    id: Optional[int] = None  # LRCLIB's own numeric id; lets a user paste a confirmed match directly.

    def to_lyrics_result(self) -> "LyricsResult":
        return LyricsResult(
            plain_lyrics=self.plain_lyrics, synced_lyrics=self.synced_lyrics, source="lrclib",
            track_name=self.track_name, artist_name=self.artist_name, lrclib_id=self.id,
            duration=effective_lrc_duration(self),
        )


def effective_lrc_duration(c: "LrcLibCandidate") -> Optional[float]:
    """LRCLIB's `duration` is unverified; if the last real lyric line is at/after it, use that timestamp instead."""
    if c.duration is None or not c.synced_lyrics:
        return c.duration
    from .lrc_timing import parse_lrc
    last_lyric_time = None
    for t, text in parse_lrc(c.synced_lyrics):
        if text.strip():
            last_lyric_time = t
    if last_lyric_time is not None and last_lyric_time >= c.duration:
        return last_lyric_time
    return c.duration


def search_lrclib(artist: str = "", title: str = "", q: str = "") -> List[LrcLibCandidate]:
    """Raw LRCLIB search -- returns every result unfiltered, no pick made. Returns [] on any failure.

    `q`, if given, is LRCLIB's free-text search, used instead of artist/title (not combined with them)."""
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
    """Fetches one LRCLIB entry by id, bypassing search/scoring. Returns None on failure."""
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
    """Loads a local .lrc file as a forced `LrcLibCandidate`, bypassing LRCLIB. Returns None on read/parse failure."""
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


def _score_lrclib_candidate(c: LrcLibCandidate, duration_sec: Optional[float]) -> float:
    """Scores a candidate for auto-pick; synced lyrics are required, instrumental/plain-only score the same as none."""
    if c.instrumental or not c.plain_lyrics or not c.synced_lyrics:
        return -1.0
    s = 0.0
    c_duration = effective_lrc_duration(c)
    if duration_sec is not None and c_duration:
        diff = abs(c_duration - duration_sec)
        s += max(0.0, 1.0 - diff / config.LRCLIB_DURATION_TOLERANCE_SEC)
    return s


def _fetch_from_lrclib(
        artist: str, title: str, duration_sec: Optional[float] = None,
) -> Optional[LyricsResult]:
    """Searches LRCLIB and picks the best candidate by duration closeness (see `_score_lrclib_candidate`)."""
    candidates = search_lrclib(artist, title)
    if not candidates:
        return None

    best = max(candidates, key=lambda c: _score_lrclib_candidate(c, duration_sec))
    if _score_lrclib_candidate(best, duration_sec) < 0:
        return None
    return best.to_lyrics_result()


def fetch_reference_lyrics(
        artist: str, title: str, duration_sec: Optional[float] = None,
) -> Optional[LyricsResult]:
    """Searches LRCLIB for a candidate with synced lyrics. Returns None if none found."""
    return _fetch_from_lrclib(artist, title, duration_sec)


def reference_match_ratio(ref_lines: List[str], words: List[Word]) -> float:
    """Raw vocabulary-overlap ratio between reference lines and ASR words; underlies `reference_matches_transcript`."""
    if not ref_lines or not words:
        return 0.0
    ref_norm, _, _ = _tokenize_lines(ref_lines)
    asr_norm = [_normalize(w.text) for w in words if _normalize(w.text)]
    if not ref_norm or not asr_norm:
        return 0.0
    return difflib.SequenceMatcher(None, asr_norm, ref_norm, autojunk=False).ratio()


def largest_unmatched_reference_run(ref_lines: List[str], words: List[Word]) -> int:
    """Size of the largest contiguous run of reference words ASR missed entirely (a difflib 'insert' block).

    Catches a locally-dropped passage that a whole-song aggregate ratio can hide."""
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
    """Force-aligns known reference text into the audio window for each ASR-dropped run (a difflib 'insert' block),
    via wav2vec2 CTC forced alignment. Returns (new_words, n_recovered); `words` itself is never mutated."""
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
    """Rejects a wrong-song/wrong-language reference before it's trusted, via vocabulary overlap with ASR."""
    return reference_match_ratio(ref_lines, words) >= min_ratio


def parse_lyrics_lines(raw_lyrics: str) -> List[str]:
    """Splits raw lyrics into cleaned lines, stripping blank/annotation/credit lines."""
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



def _tokenize_lines(lines: List[str]) -> Tuple[List[str], List[str], List[int]]:
    """Returns (normalized_tokens, original_tokens, line_ids), one entry per word.

    Hyphenated tokens (e.g. "Do-do-do-do-do") are split into separate words, not counted as one."""
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
    """Whether this LRC candidate is even the right recording, via `lrc_timing.two_tier_time_calibration` as a plausibility gate.
    Calibrated timestamps themselves aren't used for line-splitting."""
    from .lrc_timing import parse_lrc, match_asr_to_lrc_lines, two_tier_time_calibration

    lrc_lines = parse_lrc(synced_lyrics_text)
    if len(lrc_lines) < 2 or not words:
        return False
    candidates = match_asr_to_lrc_lines(words, lrc_lines)
    _offset, _slope, _confidence, _kind, _skipped, correction_fn, _holdout = two_tier_time_calibration(candidates)
    return correction_fn is not None


def assign_lrc_line_ids_sequentially(words: List[Word], synced_lyrics_text: str) -> Optional[List[Word]]:
    """Assigns LRC line ids via a forward-only cursor over ASR words, never a time/gap window --
    a repeated phrase later in the song can't be confused with an earlier occurrence.
    Returns None when `is_lrc_line_tracking_confident` distrusts this candidate.

    Window sizing/cursor-advance reuses `lrc_timing.find_cursor_window_match` (the same shared
    mechanism `match_asr_to_lrc_lines` and `mxl_lrc_generator.reconcile_mxl_to_lrc_lines` use for
    this identical problem): a miss grows the NEXT line's search window instead of leaving it
    fixed at that line's own small size. A real bug fixed by this (found via a real reported
    case, a heavily-repeated-chorus song): the previous fixed-size-per-line window never grew on
    a miss, so once one line's own text genuinely didn't match nearby ASR (a real LRC/audio
    structural difference -- an extra/missing repeat, a differently-worded ad-lib), every
    subsequent line could permanently lose sync too, since the window never grew wide enough to
    reach the next real occurrence. The actual text correction/reference-tagging within a chosen
    window still goes through `align_words_to_reference` unchanged -- only the window-sizing
    logic around it changed.

    Window-finding tolerates filler/ad-lib vocalise variation (`text_normalize.
    normalize_for_fuzzy_match` -- "ah-ah-ah" vs "na na na" transcribing the same real sound
    shouldn't count as a mismatch), except a line that's ENTIRELY filler is skipped outright
    (`is_all_filler`) -- no real content to safely anchor a match on. The actual text/reference
    tagging within a located window still goes through `align_words_to_reference`'s own exact
    matching, unaffected."""
    from .lrc_timing import parse_lrc, find_cursor_window_match
    from .text_normalize import normalize_for_fuzzy_match as _normalize_fuzzy, is_all_filler

    lrc_lines = parse_lrc(synced_lyrics_text)
    if not is_lrc_line_tracking_confident(words, synced_lyrics_text):
        return None

    MAX_PENDING_WORDS = 60  # mirrors lrc_timing.match_asr_to_lrc_lines's own cap
    asr_norm = [_normalize_fuzzy(w.text) for w in words]
    cursor = 0
    pending_word_count = 0
    n = len(words)
    out: List[Word] = []
    for li, (_t, line_text) in enumerate(lrc_lines):
        line_tokens = [t for t in (_normalize(tok) for tok in line_text.split()) if t]
        if not line_tokens or is_all_filler(line_tokens):
            continue
        line_tokens = [_normalize_fuzzy(tok) for tok in line_text.split() if _normalize(tok)]
        pending_word_count = min(pending_word_count + len(line_tokens), MAX_PENDING_WORDS)
        found = find_cursor_window_match(cursor, asr_norm, line_tokens, pending_word_count)
        if found is None:
            # No match anywhere in range -- don't advance the cursor, next line's window grows.
            continue
        _opcodes, window = found
        window_words = words[cursor:cursor + len(window)]
        if not window_words:
            continue
        aligned = align_words_to_reference(window_words, [line_text])
        # A real match has reference_text set; line_id alone isn't enough since the
        # neighbor-fill fallback back-fills line_id=0 on every unmatched word too.
        last_matched_offset = None
        for i, w in enumerate(aligned):
            if w.reference_text is not None:
                last_matched_offset = i
        if last_matched_offset is None:
            # No match for this line -- don't advance the cursor, let the next line try from here.
            continue
        for w in aligned[:last_matched_offset + 1]:
            out.append(w if w.line_id is None else Word(
                text=w.text, start=w.start, end=w.end, confidence=w.confidence,
                line_id=li, reference_text=w.reference_text, dropped=w.dropped,
            ))
        cursor += last_matched_offset + 1
        pending_word_count = 0

    # Trailing content after the last line's window (e.g. outro ad-lib) stays unmatched.
    out.extend(words[cursor:])
    return out


def align_words_to_reference(
    words: List[Word], reference_lines: List[str], synced_lyrics_text: Optional[str] = None,
) -> List[Word]:
    """Whole-sequence diff of ASR words against reference lyrics: corrects text on mismatched
    words, tags every word with its reference `line_id`, drops short unmatched runs
    (keeps long ones -- likely an alignment failure, not real hallucination).

    If `synced_lyrics_text` is given and trusted (`is_lrc_line_tracking_confident`), delegates
    entirely to `assign_lrc_line_ids_sequentially` instead of the whole-song diff below."""
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
            # 1:1 replace (single mistranscribed word) -- swap text in directly.
            if a_len == b_len:
                for k in range(a_len):
                    w = words[a0 + k]
                    ref_word = ref_orig[b0 + k]
                    sim = difflib.SequenceMatcher(None, asr_norm[a0 + k], ref_norm[b0 + k]).ratio()
                    new_text = ref_word if sim >= 0.3 else w.text
                    if new_text != w.text and w.text[:1].isupper() and new_text[:1].islower():
                        new_text = new_text[:1].upper() + new_text[1:]
                    # Below the similarity bar, don't even record reference_text -- a low-confidence
                    # value can get blindly trusted downstream (e.g. verification.py's fallback).
                    out.append(Word(text=new_text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=ref_line_ids[b0 + k],
                                     reference_text=ref_word if sim >= 0.3 else None))
            else:
                # Uneven block (ASR split/merged words differently than reference) -- keep ASR text,
                # but tag every word with a line id and best-guess reference_text (not substituted).
                #
                # When a_len > b_len, min(b0+k, b1-1) clamps several ASR words onto one reference
                # token; for a hyphenated repeat unit ("Do-do-do-do-do"), hand out one hyphen-part
                # per repeated word instead (via hyphenate()) rather than the whole blob to each.
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
                    # Guard against an unrelated word (isolated in silence) getting swept into a
                    # repeat-clamp block just by landing at its edge -- checked against both
                    # neighbors in the global `words` sequence (config.REFERENCE_CLAMP_MAX_GAP_SEC).
                    idx = a0 + k
                    far_from_prev = idx == 0 or (w.start - words[idx - 1].end) > config.REFERENCE_CLAMP_MAX_GAP_SEC
                    far_from_next = (idx + 1 >= len(words)
                                      or (words[idx + 1].start - w.end) > config.REFERENCE_CLAMP_MAX_GAP_SEC)
                    if is_repeat_clamp and far_from_prev and far_from_next:
                        # Isolated from its clamp run -- mark dropped, but still appended (see Word.dropped).
                        out.append(Word(text=w.text, start=w.start, end=w.end,
                                         confidence=w.confidence, line_id=None, dropped=True))
                        continue
                    # Too many words clamped onto one token signals decoder hallucination, not a real
                    # ad-lib (config.REFERENCE_CLAMP_MAX_REPEAT) -- past that, keep raw ASR text.
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
                            # Wrap, not freeze, once repeats exceed the real syllable count.
                            p = part_cursor[b_idx] % len(parts)
                            ref_text = parts[p]
                            part_cursor[b_idx] += 1
                    out.append(Word(text=w.text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=ref_line_ids[b_idx],
                                     reference_text=ref_text))
        elif tag == "delete":
            # ASR word(s) with no reference counterpart anywhere. A short run is dropped
            # entirely (likely hallucination); a long run is kept -- more likely a difflib
            # global-alignment misclassification than real non-lyrical content (config.REFERENCE_DELETE_MAX_RUN).
            run_len = a1 - a0
            if run_len > config.REFERENCE_DELETE_MAX_RUN:
                for k in range(run_len):
                    w = words[a0 + k]
                    out.append(Word(text=w.text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=None))
            else:
                # Appended with dropped=True, not omitted -- omitting it would let a neighboring
                # word's pass-1 note zone swallow its notes (see Word.dropped).
                for k in range(run_len):
                    w = words[a0 + k]
                    out.append(Word(text=w.text, start=w.start, end=w.end,
                                     confidence=w.confidence, line_id=None, dropped=True))
            continue
        # tag == "insert": reference word(s) ASR missed entirely, nothing to attach them to.
        # (`recover_dropped_reference_words`, run before this function, is the real recovery path.)

    # Fill remaining unmatched (line_id=None) words from the nearest preceding matched neighbor.
    last_seen = None
    for i in range(len(out)):
        if out[i].line_id is not None:
            last_seen = out[i].line_id
        else:
            out[i] = Word(text=out[i].text, start=out[i].start, end=out[i].end,
                           confidence=out[i].confidence, line_id=last_seen, dropped=out[i].dropped)

    return out


def alignment_diff_summary(original: List[Word], corrected: List[Word]) -> List[str]:
    """Returns diagnostic "word -> word"/"word -> [DROPPED]" lines for every changed word."""
    corrected_by_span = {(c.start, c.end): c for c in corrected}
    diffs = []
    for o in original:
        c = corrected_by_span.get((o.start, o.end))
        if c is None or c.dropped:
            diffs.append(f'"{o.text}" -> [DROPPED, no reference match] (at {o.start:.2f}s)')
        elif o.text != c.text:
            diffs.append(f'"{o.text}" -> "{c.text}" (at {o.start:.2f}s)')
    return diffs
