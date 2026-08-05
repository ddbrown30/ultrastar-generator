"""Online lyric lookup (lyrics.ovh), used for two things per the current
requirements:

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

This never touches note timing -- reference lyrics have no timing
information at all, only text and line structure. If the lookup fails
(no network, song not found, etc.) everything downstream just falls back
to ASR text and gap-based phrasing, same as if this were disabled.
"""

from __future__ import annotations

import difflib
import re
import urllib.parse
from typing import List, Optional, Tuple

from .models import Word

# Lines that are pure section/annotation markers (e.g. "[Chorus]",
# "[Verse 2]") or lyrics.ovh's occasional trailing credit line, not
# actual sung content -- these get filtered out before line numbering.
_ANNOTATION_LINE_RE = re.compile(r"^\s*\[.*\]\s*$")
_CREDIT_LINE_RE = re.compile(r"^\s*(paroles|lyrics powered by|www\.)", re.IGNORECASE)


def fetch_reference_lyrics(artist: str, title: str) -> Optional[str]:
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
        return data.get("lyrics")
    except Exception:
        return None


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
    parallel lists, one entry per word across all lines."""
    norm_tokens: List[str] = []
    orig_tokens: List[str] = []
    line_ids: List[int] = []
    for line_idx, line in enumerate(lines):
        for tok in line.split():
            n = _normalize(tok)
            if not n:
                continue
            norm_tokens.append(n)
            orig_tokens.append(tok)
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
                                    confidence=w.confidence, line_id=ref_line_ids[b0 + k])
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
                                        confidence=w.confidence, line_id=ref_line_ids[b0 + k])
            else:
                # Uneven block (ASR split/merged words differently than
                # the reference does) -- keep ASR text as-is rather than
                # guess a risky word-for-word mapping, but still tag every
                # ASR word in the block with a reference line id so
                # phrase breaks stay correct.
                for k in range(a_len):
                    w = words[a0 + k]
                    b_idx = min(b0 + k, b1 - 1)
                    out[a0 + k] = Word(text=w.text, start=w.start, end=w.end,
                                        confidence=w.confidence, line_id=ref_line_ids[b_idx])
        elif tag == "delete":
            # ASR has word(s) the reference doesn't (ad-libs, hallucinated
            # filler, etc.) -- keep as-is; line_id filled in below from
            # neighboring context.
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
