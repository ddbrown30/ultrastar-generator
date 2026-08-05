"""Splits a word into syllables.

Uses pyphen (a hyphenation library based on the same dictionaries as
LibreOffice/Firefox spellcheck) when available, with a small vowel-group
regex fallback so the pipeline still works if pyphen isn't installed.
"""

from __future__ import annotations

import re
from typing import List

_dic = None


def _get_dic():
    global _dic
    if _dic is None:
        try:
            import pyphen
            _dic = pyphen.Pyphen(lang="en_US")
        except ImportError:
            _dic = False  # sentinel: unavailable
    return _dic


_VOWEL_GROUPS = re.compile(r"[^aeiouyAEIOUY]*[aeiouyAEIOUY]+(?:[^aeiouyAEIOUY]*$)?", re.UNICODE)


def _regex_fallback(word: str) -> List[str]:
    matches = _VOWEL_GROUPS.findall(word)
    matches = [m for m in matches if m]
    if not matches:
        return [word]
    # Reassemble consuming the whole word (regex above can leave gaps for
    # leading/trailing consonant clusters); simplest robust approach: just
    # split by vowel-group boundaries found via finditer with positions.
    parts = []
    pos = 0
    for m in _VOWEL_GROUPS.finditer(word):
        if m.start() > pos:
            continue
        parts.append(word[pos:m.end()])
        pos = m.end()
    if pos < len(word):
        if parts:
            parts[-1] += word[pos:]
        else:
            parts.append(word[pos:])
    return parts or [word]


def hyphenate(word: str) -> List[str]:
    """Returns a list of syllable strings that concatenate back to `word`
    exactly (punctuation included) -- important since UltraStar note text
    is displayed verbatim."""
    if not word:
        return [word]

    # Separate leading/trailing punctuation so the dictionary only sees the
    # alphabetic core, then reattach it to the first/last syllable.
    m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", word, re.UNICODE)
    lead, core, trail = m.groups() if m else ("", word, "")

    if not core:
        return [word]

    dic = _get_dic()
    if dic:
        hyphenated = dic.inserted(core, hyphen="\u00ad")
        parts = hyphenated.split("\u00ad")
    else:
        parts = _regex_fallback(core)

    parts = [p for p in parts if p] or [core]
    parts[0] = lead + parts[0]
    parts[-1] = parts[-1] + trail
    return parts
