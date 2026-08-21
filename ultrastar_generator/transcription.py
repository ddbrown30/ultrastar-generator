"""Word-level lyric transcription. Prefers WhisperX (forced alignment, accurate word boundaries) over faster-whisper's decoder timestamps (less precise); falls back to the latter if WhisperX isn't available."""

from __future__ import annotations

import logging
import os
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from . import config
from . import model_cache
from .models import Word

# Harmless: pyannote/torchcodec warning about FFmpeg detection; WhisperX always feeds it a preloaded waveform.
warnings.filterwarnings(
    "ignore",
    message=r"\s*torchcodec is not installed correctly.*",
    category=UserWarning,
)

# Harmless: torch noting a Linux-only allocator optimization isn't available on Windows.
warnings.filterwarnings(
    "ignore",
    message=r".*expandable_segments not supported on this platform.*",
    category=UserWarning,
)

# Harmless: pyannote intentionally disables TF32 for reproducibility.
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
    category=UserWarning,
)

# Harmless: WhisperX's align-model checkpoint auto-upgrades in memory on load; nothing on disk changes.
logging.getLogger("lightning.pytorch.utilities.migration.utils").setLevel(logging.WARNING)

# whisperx shells out to ffmpeg via a bare subprocess.run we can't pass creationflags into.
# Patch Popen so every child gets CREATE_NO_WINDOW (avoids a console flash under pythonw.exe); no-op on non-Windows.
if os.name == "nt" and not getattr(subprocess.Popen, "_ultrastar_no_window_patched", False):
    _real_popen_init = subprocess.Popen.__init__

    def _no_window_popen_init(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        _real_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _no_window_popen_init
    subprocess.Popen._ultrastar_no_window_patched = True


def _mean_word_score(words: list) -> float:
    scores = [w.get("score") for w in words if w.get("score") is not None]
    return sum(scores) / len(scores) if scores else 0.0


def _find_best_window(segment: dict, align_model, metadata, audio, device: str,
                       debug_log=None) -> tuple:
    """Sweeps candidate windows across `segment`'s span, re-aligning the same text against each, and
    returns the best-scoring window if it beats baseline by `config.REWINDOW_MIN_SCORE_IMPROVEMENT`.
    Returns (start, end, improved)."""
    import whisperx

    t1, t2 = segment["start"], segment["end"]

    baseline = whisperx.align([segment], align_model, metadata, audio, device=device)
    baseline_score = _mean_word_score(baseline["word_segments"])

    width = config.REWINDOW_CANDIDATE_WIDTH_SEC
    step = config.REWINDOW_STEP_SEC
    best_score = baseline_score
    best_window = None

    offset = t1
    while offset < t2:
        w0, w1 = offset, min(offset + width, t2)
        cand_seg = dict(segment, start=w0, end=w1)
        cand = whisperx.align([cand_seg], align_model, metadata, audio, device=device)
        cand_score = _mean_word_score(cand["word_segments"])
        if debug_log is not None:
            debug_log.line(f"    candidate [{w0:8.3f},{w1:8.3f}]  mean_score={cand_score:.3f}")
        if cand_score > best_score:
            best_score = cand_score
            best_window = (w0, w1)
        offset += step

    if best_window is not None and best_score >= baseline_score + config.REWINDOW_MIN_SCORE_IMPROVEMENT:
        if debug_log is not None:
            debug_log.line(f"  RE-WINDOWED [{t1:.3f}-{t2:.3f}] {segment['text']!r}: baseline score "
                            f"{baseline_score:.3f} -> {best_window[0]:.3f}-{best_window[1]:.3f} "
                            f"score {best_score:.3f}")
        return best_window[0], best_window[1], True

    if debug_log is not None:
        debug_log.line(f"  kept baseline [{t1:.3f}-{t2:.3f}] {segment['text']!r}: baseline score "
                        f"{baseline_score:.3f}, best candidate {best_score:.3f} didn't clear the "
                        f"+{config.REWINDOW_MIN_SCORE_IMPROVEMENT} improvement bar")
    return t1, t2, False


def _rewindow_long_segments(segments: list, align_model, metadata, audio, device: str,
                             debug_log=None) -> list:
    """Returns a new segment list with corrected (start, end) for any segment at least
    `config.REWINDOW_MIN_SEGMENT_DURATION_SEC` long; text and everything else is untouched.
    Suppresses whisperx's WARNING logging during the sweep (many probed windows are expected to fail
    alignment) and restores it before returning."""
    align_logger = logging.getLogger("whisperx.alignment")
    prev_level = align_logger.level
    align_logger.setLevel(logging.ERROR)
    try:
        fixed = []
        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration < config.REWINDOW_MIN_SEGMENT_DURATION_SEC:
                fixed.append(seg)
                continue
            new_start, new_end, improved = _find_best_window(seg, align_model, metadata, audio, device,
                                                               debug_log=debug_log)
            fixed.append(dict(seg, start=new_start, end=new_end) if improved else seg)
        return fixed
    finally:
        align_logger.setLevel(prev_level)


def force_align_words_in_window(words_text: List[str], window_start: float, window_end: float,
                                 align_model, metadata, audio, device: str = "cuda"
                                 ) -> Optional[List[Tuple[float, float, float]]]:
    """Forces known `words_text` onto the audio inside [window_start, window_end] via wav2vec2 CTC
    forced alignment, without needing the decoder to have transcribed them. Returns None (caller
    keeps its fallback) on a too-short window, alignment failure, word-count mismatch, missing
    timestamp, or a word out of bounds/order."""
    import whisperx

    if not words_text:
        return None
    min_window = (config.FORCE_ALIGN_MIN_WINDOW_BASE_SEC
                  + config.FORCE_ALIGN_MIN_WINDOW_SEC_PER_WORD * len(words_text))
    if window_end - window_start < min_window:
        return None

    segment = {"start": window_start, "end": window_end, "text": " ".join(words_text)}
    try:
        result = whisperx.align([segment], align_model, metadata, audio, device=device,
                                 return_char_alignments=False)
    except Exception:
        return None

    raw_out = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    if len(raw_out) != len(words_text):
        return None

    out: List[Tuple[float, float, float]] = []
    last_start = window_start - 1e-3
    for w in raw_out:
        ws, we = w.get("start"), w.get("end")
        if ws is None or we is None:
            return None
        ws, we = float(ws), float(we)
        if (ws < window_start - config.FORCE_ALIGN_WINDOW_SLOP_SEC
                or we > window_end + config.FORCE_ALIGN_WINDOW_SLOP_SEC
                or ws < last_start):
            return None
        last_start = ws
        out.append((ws, max(we, ws + 0.02), float(w.get("score", 0.0))))
    return out


def force_align_reference_lyrics(vocals_path: Path, synced_lyrics: str, audio_duration: float,
                                  debug_log=None) -> List[Word]:
    """DIAGNOSTIC (--no-transcribe): builds the whole Word list via forced alignment of a pinned LRC
    candidate's known line text only, never running the WhisperX decoder -- isolates whether ASR
    hallucination is bleeding into output. Each LRC line gets its own window; a line that fails to
    force-align is skipped and logged, never guessed."""
    import whisperx
    from .lrc_timing import parse_lrc

    lines = parse_lrc(synced_lyrics)
    if not lines:
        return []

    audio = whisperx.load_audio(str(vocals_path))
    align_model, metadata = model_cache.get_whisperx_align_model()

    if debug_log is not None:
        debug_log.section("FORCED-ALIGNMENT-ONLY MODE (--no-transcribe) -- the WhisperX DECODER never ran; "
                           "every word below comes from forcing the pinned LRC candidate's own known line "
                           "text onto the audio via wav2vec2 CTC alone")

    words: List[Word] = []
    n_failed = 0
    for i, (line_start, line_text) in enumerate(lines):
        line_end = lines[i + 1][0] if i + 1 < len(lines) else audio_duration
        line_words_text = line_text.split()
        if not line_words_text or line_end <= line_start:
            continue
        result = force_align_words_in_window(line_words_text, line_start, line_end,
                                              align_model, metadata, audio, device="cuda")
        if result is None:
            n_failed += 1
            if debug_log is not None:
                debug_log.line(f"  [{line_start:8.3f} - {line_end:8.3f}]  FAILED to force-align: {line_text!r}")
            continue
        for text, (start, end, score) in zip(line_words_text, result):
            words.append(Word(text=text, start=start, end=end, confidence=score))
        if debug_log is not None:
            debug_log.line(f"  [{line_start:8.3f} - {line_end:8.3f}]  OK: {line_text!r}")

    if debug_log is not None:
        debug_log.line(f"  {len(lines) - n_failed}/{len(lines)} line(s) force-aligned successfully "
                        f"({n_failed} failed and were skipped -- no words for those lines).")

    return words


def _transcribe_with_whisperx(vocals_path: Path, model_name: str, debug_log=None,
                               vad_options: dict = None) -> List[Word]:
    import whisperx

    audio = whisperx.load_audio(str(vocals_path))

    model = model_cache.get_whisperx_asr_model(model_name, vad_options=vad_options)
    result = model.transcribe(audio, language="en", batch_size=16)

    if debug_log is not None:
        debug_log.section("RAW WHISPER DECODER SEGMENTS (before forced alignment -- shows what VAD chunked "
                           "and what the DECODER itself transcribed for each chunk, before wav2vec2 CTC "
                           "places individual word timestamps within it)")
        for seg in result["segments"]:
            debug_log.line(f"  {seg.get('start'):8.3f} - {seg.get('end'):8.3f}  {seg.get('text', '')!r}")

    align_model, metadata = model_cache.get_whisperx_align_model()

    if debug_log is not None:
        debug_log.section("LONG-SEGMENT RE-WINDOWING")
    segments = _rewindow_long_segments(result["segments"], align_model, metadata, audio, device="cuda",
                                        debug_log=debug_log)

    aligned = whisperx.align(segments, align_model, metadata, audio, device="cuda")

    if debug_log is not None:
        debug_log.section("RAW WHISPERX OUTPUT (direct from whisperx.align(), before any filtering)")
        debug_log.line("Columns: start, end, score, text -- a 'DROPPED' word has no usable start/end "
                        "and never becomes a Word at all (silently missing from everything downstream).")
        for seg in aligned["segments"]:
            for w in seg.get("words", []):
                raw_text = w.get("word")
                start = w.get("start")
                end = w.get("end")
                score = w.get("score")
                dropped = not (raw_text or "").strip() or start is None or end is None
                start_s = f"{start:8.3f}" if start is not None else "    None"
                end_s = f"{end:8.3f}" if end is not None else "    None"
                score_s = f"{score:.3f}" if score is not None else " None"
                if dropped:
                    flag = "  <-- DROPPED (missing text/timing)"
                elif score is not None and score < 0.3:
                    flag = "  <-- LOW SCORE"
                else:
                    flag = ""
                debug_log.line(f"  {start_s} - {end_s}  score={score_s}  {raw_text!r}{flag}")

    words: List[Word] = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if not text or start is None or end is None:
                continue  # unaligned word
            words.append(Word(
                text=text,
                start=float(start),
                end=float(end),
                confidence=float(w.get("score", 1.0)),
            ))
    return words


def _transcribe_with_faster_whisper(vocals_path: Path, model_name: str, debug_log=None,
                                     vad_filter: bool = True) -> List[Word]:
    model = model_cache.get_faster_whisper_model(model_name)

    segments, _info = model.transcribe(
        str(vocals_path),
        language="en",
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=300) if vad_filter else None,
    )

    words: List[Word] = []
    raw_log_lines = [] if debug_log is not None else None
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            text = w.word.strip()
            if raw_log_lines is not None:
                prob = getattr(w, "probability", None)
                prob_s = f"{prob:.3f}" if prob is not None else " None"
                flag = "  <-- DROPPED (empty text)" if not text else ""
                raw_log_lines.append(f"  {w.start:8.3f} - {w.end:8.3f}  prob={prob_s}  {w.word!r}{flag}")
            if not text:
                continue
            words.append(Word(
                text=text,
                start=float(w.start),
                end=float(w.end),
                confidence=float(getattr(w, "probability", 1.0)),
            ))

    if debug_log is not None:
        debug_log.section("RAW FASTER-WHISPER OUTPUT (direct from model.transcribe(), before any filtering)")
        debug_log.line("Columns: start, end, probability, text")
        for line in raw_log_lines:
            debug_log.line(line)

    return words


def transcribe_words(
    vocals_path: Path,
    model_name: str,
    prefer_whisperx: bool = config.PREFER_WHISPERX,
    debug_log=None,
    vad_filter: bool = True,
    whisperx_vad_options: dict = None,
) -> List[Word]:
    """Returns a flat, time-ordered list of Word objects for the whole track. `vad_filter` applies
    only to the faster-whisper path; `whisperx_vad_options` only to the whisperx path."""
    if prefer_whisperx:
        try:
            return _transcribe_with_whisperx(vocals_path, model_name, debug_log=debug_log,
                                              vad_options=whisperx_vad_options)
        except ImportError:
            print(
                "whisperx not installed -- falling back to faster-whisper's own "
                "word timestamps, which are noticeably less precise. "
                "For better timing accuracy: pip install whisperx"
            )
        except Exception as e:
            print(f"whisperx transcription failed ({e}); falling back to faster-whisper.")

    try:
        return _transcribe_with_faster_whisper(vocals_path, model_name, debug_log=debug_log, vad_filter=vad_filter)
    except ImportError as e:
        raise ImportError(
            "Neither whisperx nor faster-whisper is installed. "
            "Install at least one: pip install faster-whisper  (or)  pip install whisperx"
        ) from e
