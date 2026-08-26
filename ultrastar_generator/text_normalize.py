"""Shared word-text normalization for text-matching (LRC/ASR/reference/ground-truth).

`verification.py` has its own separate `_normalize` for fuzzy-ratio matching -- not a duplicate."""
import re
from typing import List


def normalize_word(s: str) -> str:
    """Lowercase, fold curly apostrophes to straight, strip all but letters/digits/apostrophes."""
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", s)


# Ad-lib/connector words transcribers commonly add or drop without the line meaning being
# different; kept short and deliberately conservative (real lyric content, e.g. "Do-do-do-do-do",
# must never be caught here -- see is_filler_token's own docstring).
FILLER_WORDS = frozenset({
    "ooh", "ooo", "oh", "ohh", "mmm", "mm", "hmm", "hm",
    "yeah", "and", "but",
})

# Short vocalise syllables that only count as filler when the ENTIRE (already-normalized) token
# is a clean repetition of one of them (e.g. "ahahah" = "ah"*3, "nanana" = "na"*3, a single bare
# "na") -- never as a substring, so a real word that merely CONTAINS one of these (e.g. "banana",
# which is "ba"+"na"+"na", not a uniform repeat of either) is never misflagged.
_FILLER_SYLLABLES = ("na", "ah", "la")


def is_filler_token(normalized_token: str) -> bool:
    """Whether an already-normalized word is pure ad-lib "noise" (na/ah/la/mmm/etc, alone or
    repeated into one token, e.g. "ahahah") rather than real lyric content. Used to keep vocalise
    variation between two independent transcriptions of the same passage (one says "ah-ah-ah",
    the other hears "na na na") from being scored as a real text mismatch -- both in comparison
    tooling (verify_existing_song.py) and in the actual LRC/MXL reconciliation logic
    (lrc_timing.py), which already had a narrower version of this idea (FILLER_WORDS) for
    line-level comparison."""
    if not normalized_token:
        return False
    if normalized_token in FILLER_WORDS:
        return True
    for unit in _FILLER_SYLLABLES:
        unit_len = len(unit)
        if len(normalized_token) % unit_len == 0 and normalized_token == unit * (len(normalized_token) // unit_len):
            return True
    return False


# Never a real word (starts with a control character) -- used so difflib's own equality check
# treats every filler token as interchangeable with every other, regardless of which specific
# vocalise sound each side transcribed it as.
_FILLER_CANON = "\x00filler\x00"


def normalize_for_fuzzy_match(word: str) -> str:
    """`normalize_word`, except every filler/ad-lib token (`is_filler_token`) collapses to one
    shared canonical placeholder -- for any token-sequence MATCHING (difflib, cursor-window
    search) where "ah-ah-ah" and "na na na" transcribing the same real vocalise should be
    treated as equal. Never use this for DISPLAY text -- only for the matching/equality step."""
    n = normalize_word(word)
    return _FILLER_CANON if is_filler_token(n) else n


def is_all_filler(normalized_tokens: List[str]) -> bool:
    """Whether every token in an already-normalized (plain `normalize_word`, not
    `normalize_for_fuzzy_match`) token list is filler/ad-lib noise -- a line/chunk like this has
    no real distinguishing content to safely anchor a cursor-window match on (every filler token
    looks identical to every other once canonicalized, so two purely-filler passages could
    otherwise "match" each other anywhere in the song)."""
    return bool(normalized_tokens) and all(is_filler_token(t) for t in normalized_tokens)
