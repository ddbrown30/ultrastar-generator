"""Replaces a word's ASR text with its reference_text wherever they disagree. Audio-free (no re-transcription recheck). Words with no reference_text are left untouched."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

from .models import Word


def _normalize(text: str) -> str:
    return text.strip().lower().strip(".,!?\"'")


def _fuzzy_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


@dataclass
class ReferenceOverrideResult:
    word_index: int
    original_text: str
    reference_text: str
    replaced: bool


def apply_reference_text(words: List[Word], indices: List[int]) -> tuple:
    """Replaces text for each of `indices` whose reference_text doesn't fuzzy-match its current text. Returns (words, results); only text is ever changed."""
    results: List[ReferenceOverrideResult] = []
    new_words = list(words)
    any_replaced = False
    for i in sorted(set(indices)):
        word = words[i]
        ref = word.reference_text
        if ref is None or _fuzzy_match(word.text, ref):
            continue
        new_words[i] = replace(word, text=ref)
        any_replaced = True
        results.append(ReferenceOverrideResult(i, word.text, ref, True))
    return (new_words if any_replaced else words), results
