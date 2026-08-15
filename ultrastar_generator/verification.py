"""Chunk-based re-transcription for verifying words against reference
lyrics.

Every word gets a fresh, tightly-cropped, isolated re-transcription of
its own moment in the audio (default; see config.VERIFY_ALL_WORDS --
restrict to pass 3's flagged-suspicious words only via
--verify-suspicious-only if the extra passes aren't worth the time for a
given run). "Suspicious" words -- any word (singleton or part of a
matched reference line) whose own ASR span ended up with zero note pieces
after splitting (see lyric_alignment.py's _split_notes_by_word_boundaries)
-- are where this is most likely to catch something, but checking every
word costs little next to Demucs/WhisperX and catches cases pass 3's
heuristics don't (e.g. lyrics_lookup.py's own "uneven block" case, where a word gets
tagged with a reference line but its text is deliberately left
uncorrected because the block-level alignment couldn't confidently map
individual words 1:1).

The key difference from a naive "does the recheck agree with what we
already have" self-consistency check: whenever a word has a
reference_text (see lyrics_lookup.py/models.Word), the recheck is
compared against what the reference lyrics actually expected at that
position, not just against the word's own current text -- so a fresh
recheck can confirm and apply a correction even for a reference-tagged
word whose text was never actually verified against that reference.
Reference text remains the most trusted source when everything
disagrees (consistent with the rest of the pipeline treating reference
lyrics as ground truth for TEXT), but a recheck that actively confirms a
different, better answer is used instead of blindly keeping stale text.

This never touches note timing/pitch -- only word text.

Note: this module used to also offer a `verify_placement` check (cropped
a window at each word's FINAL note-assigned position and cross-checked
it, auto-correcting via forced alignment). Removed 2026-08-10 -- real
ground-truth comparison confirmed it a net regression on every pitch/
timing metric (see CLAUDE.md's "Removed / rejected approaches"), even
after further pipeline improvements; not worth keeping around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import config
from . import model_cache
from .models import Word
from .progress import ProgressReporter


@dataclass
class VerificationResult:
    word_index: int
    original_text: str
    reference_text: Optional[str]
    rechecked_text: Optional[str]
    final_text: str
    replaced: bool


def _normalize(text: str) -> str:
    return text.strip().lower().strip(".,!?\"'")


def _fuzzy_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _crop_audio(y: np.ndarray, sr: int, start: float, end: float) -> Optional[np.ndarray]:
    pad = config.RECHECK_PAD_SEC
    lo = max(0, int(round((start - pad) * sr)))
    hi = min(len(y), int(round((end + pad) * sr)))
    if hi <= lo:
        return None
    return y[lo:hi]


def _retranscribe_clip(model, clip: np.ndarray, sr: int) -> str:
    """Re-transcribes a clip with a (cached) whisperx ASR model -- text
    only, since verify_words doesn't need segment timing."""
    import librosa

    if sr != 16000:
        clip = librosa.resample(clip, orig_sr=sr, target_sr=16000)
    text, _seg_start, _seg_end = _transcribe_clip_whisperx(model, clip.astype(np.float32))
    return text


def _resolve(word: Word, rechecked: Optional[str], verbose: bool) -> tuple:
    """Decides the final text for one word given its current text, its
    reference-expected text (if any), and a fresh isolated recheck.
    Returns (final_text, replaced, log_line_or_None)."""
    cur, ref = word.text, word.reference_text

    if ref is not None:
        cur_matches_ref = _fuzzy_match(cur, ref)
        recheck_matches_ref = _fuzzy_match(rechecked, ref)

        if cur_matches_ref:
            return cur, False, None
        if recheck_matches_ref:
            log = (f'    recheck: "{cur}" @ {word.start:.2f}s -> heard "{rechecked}", '
                   f'confirming reference "{ref}" -- corrected')
            return ref, True, log
        # Neither the current text nor the recheck confirms the
        # reference -- a real three-way disagreement. Reference lyrics
        # are still the most trusted source of TEXT in this pipeline
        # (that's the whole point of fetching them), so default to it,
        # but log clearly since nothing actually confirmed it here.
        log = (f'    recheck: "{cur}" @ {word.start:.2f}s -> heard "{rechecked}" '
               f'(neither matches reference "{ref}") -- kept reference, unconfirmed')
        return ref, (ref != cur), log

    # No reference expectation at all (ad-lib, or lyrics lookup
    # unavailable/didn't cover this word). There's no independent
    # confirmation available here -- just one noisy signal (the isolated
    # recheck) against another (the original full-context ASR text) --
    # and the recheck is the LESS reliable of the two: it runs on a tiny,
    # few-hundred-ms crop with none of the surrounding-context whisper had
    # for its first pass, same failure class as this project's other
    # "never trust inference from a tiny isolated clip alone" lessons
    # (see CLAUDE.md). Confirmed in practice: real full-pipeline runs
    # where lyrics lookup failed showed this branch replacing already-
    # correct short words with hallucinated multi-word phrases (e.g.
    # "I" -> "Whoo-hoo!", "why" -> "the white little"). So: log a
    # disagreement for visibility, but never act on it unconfirmed --
    # keep the more reliable full-context text.
    if rechecked and not _fuzzy_match(cur, rechecked):
        log = (f'    recheck: "{cur}" @ {word.start:.2f}s -> heard "{rechecked}" '
               f'(no reference to confirm either way) -- kept original')
        return cur, False, log
    return cur, False, None


def verify_words(
    words: List[Word],
    indices: List[int],
    y: np.ndarray,
    sr: int,
    model_name: str,
    verbose: bool = True,
) -> tuple:
    """Re-transcribes a tight crop around each of `indices` and resolves
    the final text per-word (see _resolve). Returns (words, results):
    `words` is the original list, or a copy with corrected text where a
    change was made; `results` is a VerificationResult per word checked,
    for diagnostics/logging. Never modifies word start/end -- only text.
    """
    if not indices:
        return words, []

    model = model_cache.get_whisperx_asr_model(model_name)

    results: List[VerificationResult] = []
    new_words = list(words)
    any_replaced = False
    sorted_indices = sorted(set(indices))
    progress = ProgressReporter("verify_words", len(sorted_indices), verbose=verbose)
    for i in sorted_indices:
        word = words[i]
        clip = _crop_audio(y, sr, word.start, word.end)
        if clip is None or len(clip) == 0:
            progress.advance()
            continue
        rechecked = _retranscribe_clip(model, clip, sr) or None
        final_text, replaced, log_line = _resolve(word, rechecked, verbose)

        if replaced:
            new_words[i] = Word(
                text=final_text, start=word.start, end=word.end,
                confidence=word.confidence, line_id=word.line_id,
                reference_text=word.reference_text,
            )
            any_replaced = True
        results.append(VerificationResult(i, word.text, word.reference_text, rechecked, final_text, replaced))
        if verbose and log_line:
            print(log_line)
        n_replaced = sum(1 for r in results if r.replaced)
        progress.advance(extra=f"{n_replaced} corrected so far")

    return (new_words if any_replaced else words), results


def _transcribe_clip_whisperx(model, clip: np.ndarray) -> Tuple[str, Optional[float], Optional[float]]:
    """Returns (text, seg_start, seg_end) -- seg_start/seg_end are
    Whisper's OWN rough segment timing (relative to `clip`), not a
    fabricated guess. This matters: passing align() a made-up "start=0.0"
    hint for text that doesn't actually begin at the very start of the
    clip biases it to anchor everything near frame 0 regardless of where
    the audio really is (confirmed in practice: two adjacent words in the
    same real bug both came back at EXACTLY window_start, to the
    millisecond, when this was hardcoded to 0.0/clip_duration) -- using
    Whisper's own segment bounds as the hint instead fixes that."""
    result = model.transcribe(clip, language="en", batch_size=16)
    segments = result.get("segments", [])
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
    starts = [seg["start"] for seg in segments if seg.get("start") is not None]
    ends = [seg["end"] for seg in segments if seg.get("end") is not None]
    seg_start = min(starts) if starts else None
    seg_end = max(ends) if ends else None
    return text, seg_start, seg_end
