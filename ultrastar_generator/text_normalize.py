"""Shared word-text normalization for text-matching (LRC/ASR/reference/ground-truth).

`verification.py` has its own separate `_normalize` for fuzzy-ratio matching -- not a duplicate."""
import re


def normalize_word(s: str) -> str:
    """Lowercase, fold curly apostrophes to straight, strip all but letters/digits/apostrophes."""
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", s)
