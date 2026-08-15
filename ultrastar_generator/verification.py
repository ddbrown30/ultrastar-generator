"""Applies reference-lyrics text as the trusted TEXT source wherever it
disagrees with a word's current (ASR) text.

Used to be `verify_words` -- a real per-word isolated re-transcription
recheck (`_resolve`) that was supposed to GATE this replacement on
whether the recheck actively confirmed the reference (see the CLI help/
config.py comments this module used to have). Removed 2026-08-15 (user's
explicit request) after discovering `_resolve` never actually did that
gating: in every tested case -- recheck confirms, recheck disagrees, or
no recheck at all (`rechecked=None`) -- a reference-tagged word whose
text disagreed with its reference always ended up replaced with the
reference text regardless. The recheck's own result never influenced the
final text, only which log line got printed (locked in by this project's
own test: `"Stray"` vs. reference `"Stars"`, recheck says `"totally
different"`, asserted result is still `"Stars"` -- "forced to reference
despite no confirmation"). Every real Whisper inference call the old code
made (one per word, by default every word in the song) was therefore pure
overhead: this trivial, audio-free operation produces the exact same
output. See CLAUDE.md's "Removed / rejected approaches".

Deliberately NOT "fixed" to actually gate on a recheck instead: that
recheck ran on a tiny, isolated audio crop, which is exactly the class of
signal this project has repeatedly found unreliable (see CLAUDE.md's
"never run inference on a tiny isolated clip" lesson) -- gating a
reference-lyrics correction on that noisy signal risks withholding a
correct fix just because the crop mishears it, not obviously better than
always trusting the reference.

Words with no reference_text (ad-libs, or lyrics lookup unavailable/
didn't cover this word) are left untouched -- there's nothing to compare
against.
"""

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
    """For each of `indices` with a `reference_text` that doesn't already
    (fuzzy-)match its current text, replaces the text with the
    reference's own. Returns (words, results): `words` is the original
    list, or a copy with corrected text where a change was made;
    `results` is a `ReferenceOverrideResult` per word actually changed,
    for diagnostics/logging. Never touches start/end/confidence/line_id/
    dropped -- only text.
    """
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
