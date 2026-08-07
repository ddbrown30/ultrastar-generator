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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import config
from . import model_cache
from .models import Word, Syllable
from .progress import ProgressReporter


@dataclass
class VerificationResult:
    word_index: int
    original_text: str
    reference_text: Optional[str]
    rechecked_text: Optional[str]
    final_text: str
    replaced: bool


@dataclass
class PlacementWarning:
    """A word whose FINAL note-assigned position (not its original ASR
    timestamp) doesn't seem to match what's actually sung there -- see
    verify_placement."""
    word_index: int
    word_text: str
    reference_text: Optional[str]
    assigned_start: float
    assigned_end: float
    heard_text: str


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
    # unavailable/didn't cover this word) -- the recheck is the only
    # second opinion we have.
    if rechecked and not _fuzzy_match(cur, rechecked):
        log = f'    recheck: "{cur}" @ {word.start:.2f}s -> heard "{rechecked}" (no reference; replaced)'
        return rechecked, True, log
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


def _word_spans_from_syllables(words: List[Word], syllables: List[Syllable]) -> List[Tuple[float, float]]:
    """Reconstructs each word's FINAL note-assigned (start, end) span from
    the syllables lyric_alignment.align_words_to_notes produced for it.
    That function always emits exactly one contiguous run of syllables per
    word (the first marked is_word_start=True), in the same order as
    `words` -- see its own docstring. Falls back to each word's own ASR
    span if that invariant doesn't hold (shouldn't normally happen), since
    mis-mapping a span to the wrong word would be worse than not checking.
    """
    spans: List[Tuple[float, float]] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    for syl in syllables:
        if syl.is_word_start or cur_start is None:
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start, cur_end = syl.start, syl.end
        else:
            cur_end = max(cur_end, syl.end)
    if cur_start is not None:
        spans.append((cur_start, cur_end))

    if len(spans) != len(words):
        return [(w.start, w.end) for w in words]
    return spans


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


def verify_placement(
    words: List[Word],
    syllables: List[Syllable],
    indices: List[int],
    y: np.ndarray,
    sr: int,
    model_name: str,
    verbose: bool = True,
) -> List[PlacementWarning]:
    """Checks whether each word's FINAL note-assigned position is actually
    where it's sung -- catches pass 3 putting a word on the wrong notes
    even when the word's own TEXT is correct (verify_words can't see this:
    it re-transcribes around the word's ORIGINAL ASR timestamp, not where
    pass 3 actually placed it).

    For each word: crop a small window around its assigned start, run an
    open-vocabulary transcription (whisperx), and check whether the
    expected word is in the result. If not, the window expands (doubling
    each time, up to config.PLACEMENT_SEARCH_MAX_RADIUS_SEC) and retries --
    this distinguishes "the word IS in the audio, just not where pass 3
    put it" (a real bug) from "not findable nearby at all". Once found,
    the exact position is refined with a forced-alignment pass over that
    SAME confirmed window -- well-conditioned, since the text handed to
    the aligner is already known to genuinely be in that window (unlike
    an earlier, abandoned version of this check that force-aligned a wide
    window with no such confirmation first, and turned out unreliable:
    whisperx.align() anchors near wherever the window starts rather than
    truly searching it, so it produced confident-looking but wrong answers
    whenever the true position wasn't near the window's start).

    Confirmed on a real bug report: "Stars" ended up assigned to notes
    spanning beat 341 to beat ~421 (~11s) -- the audio at beat 341 actually
    says "the flame the sword", and "Stars" itself isn't sung until
    ~beat 381.

    Detection only -- never reassigns notes or changes timing. Two
    automatic-correction heuristics for this exact bug class (snapping the
    boundary to the nearest note gap; rebalancing notes by syllable-count
    deficit/surplus) were already tried and rejected as too risky (see
    CLAUDE.md's open threads) -- a flagged mismatch needs a human to look
    at it, not another heuristic guess.
    """
    if not indices:
        return []

    import whisperx

    asr_model = model_cache.get_whisperx_asr_model(model_name)
    align_model, align_metadata = model_cache.get_whisperx_align_model()

    if sr != 16000:
        import librosa
        audio16k = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
    else:
        audio16k = y.astype(np.float32)
    duration = len(audio16k) / 16000.0

    spans = _word_spans_from_syllables(words, syllables)

    warnings: List[PlacementWarning] = []
    sorted_indices = sorted(set(indices))
    progress = ProgressReporter("verify_placement", len(sorted_indices), verbose=verbose)
    for i in sorted_indices:
        progress.advance(extra=f"{len(warnings)} flagged so far")
        word = words[i]
        assigned_start, assigned_end = spans[i]
        expected = word.reference_text or word.text
        expected_norm = _normalize(expected)
        if not expected_norm:
            continue

        radius = config.PLACEMENT_SEARCH_INITIAL_RADIUS_SEC
        found_text: Optional[str] = None
        found_seg_start = found_seg_end = None
        window_start = window_end = 0.0
        while True:
            window_start = max(0.0, assigned_start - radius)
            window_end = min(duration, assigned_start + radius)
            if window_end <= window_start:
                break
            clip = audio16k[int(round(window_start * 16000)):int(round(window_end * 16000))]
            if len(clip) == 0:
                break
            text, seg_start, seg_end = _transcribe_clip_whisperx(asr_model, clip)
            tokens = {_normalize(t) for t in text.split()}
            if expected_norm in tokens:
                found_text = text
                found_seg_start = seg_start if seg_start is not None else 0.0
                found_seg_end = seg_end if seg_end is not None else (window_end - window_start)
                break
            if radius >= config.PLACEMENT_SEARCH_MAX_RADIUS_SEC:
                break
            radius = min(radius * config.PLACEMENT_SEARCH_GROWTH_FACTOR, config.PLACEMENT_SEARCH_MAX_RADIUS_SEC)

        if found_text is None:
            warnings.append(PlacementWarning(
                i, word.text, word.reference_text, assigned_start, assigned_end,
                f"not found within {radius:.1f}s of assigned position",
            ))
            if verbose:
                print(f'    [placement] "{word.text}" not found anywhere within {radius:.1f}s of its '
                      f'assigned position ({assigned_start:.2f}s) -- worth checking by hand')
            continue

        # Confirmed present somewhere in [window_start, window_end) --
        # refine the exact position with forced alignment over that SAME
        # window (safe now: the text is already known to genuinely be
        # there, unlike the abandoned blind-search approach).
        clip = audio16k[int(round(window_start * 16000)):int(round(window_end * 16000))]
        segment = {"text": found_text, "start": found_seg_start, "end": found_seg_end}
        true_start = None
        try:
            aligned = whisperx.align([segment], align_model, align_metadata, clip, device="cuda")
            # align() commonly splits one long segment's text into SEVERAL
            # returned segments (e.g. at sentence boundaries) -- checking
            # only the first was a real bug: any match past the first
            # segment was silently missed, always falling back to
            # window_start below, which looked exactly like (but wasn't)
            # whisperx anchoring near the window's start.
            for aligned_seg in (aligned.get("segments") or []):
                for w in aligned_seg.get("words", []):
                    if _normalize(w.get("word", "")) == expected_norm and w.get("start") is not None:
                        true_start = window_start + float(w["start"])
                        break
                if true_start is not None:
                    break
        except Exception:
            pass
        if true_start is None:
            true_start = window_start  # at least confirmed within this window

        if abs(true_start - assigned_start) <= config.PLACEMENT_MISMATCH_TOLERANCE_SEC:
            continue

        warnings.append(PlacementWarning(
            i, word.text, word.reference_text, assigned_start, assigned_end,
            f"found at ~{true_start:.2f}s (search radius {radius:.1f}s)",
        ))
        if verbose:
            print(f'    [placement] "{word.text}" assigned to notes starting at {assigned_start:.2f}s, '
                  f'but actually found at ~{true_start:.2f}s (search radius {radius:.1f}s) -- likely a '
                  f'note-boundary mismatch (not corrected automatically)')

    return warnings
