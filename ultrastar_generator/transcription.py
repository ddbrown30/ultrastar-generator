"""Word-level lyric transcription.

Timing accuracy matters a lot here, since lyric_alignment.py fits words
onto an already-accurate note grid using these timestamps. Whisper's own
decoder-derived word timestamps (what faster-whisper reports by default)
are frequently off by a noticeable fraction of a second, because they're
a byproduct of cross-attention weights, not an actual alignment model.

WhisperX fixes this by running a second pass: a wav2vec2 CTC model does a
proper forced alignment of the transcript against the audio, which is
dramatically more accurate for word boundaries. We use it when available
and fall back to faster-whisper's own timestamps (with a warning) if not.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from . import config
from . import model_cache
from .models import Word


def _transcribe_with_whisperx(vocals_path: Path, model_name: str, debug_log=None,
                               vad_options: dict = None) -> List[Word]:
    import whisperx

    audio = whisperx.load_audio(str(vocals_path))

    model = model_cache.get_whisperx_asr_model(model_name, vad_options=vad_options)
    result = model.transcribe(audio, language="en", batch_size=16)

    align_model, metadata = model_cache.get_whisperx_align_model()
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device="cuda")

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
                continue  # whisperx leaves timing out for a few unaligned words
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
    """Returns a flat, time-ordered list of Word objects for the whole track.

    Note: these timestamps are a starting point for lyric_alignment.py, not
    the final note timing -- final timing comes from note_detection.py's
    audio-only analysis. Still, more accurate word boundaries here mean
    fewer/less-drastic corrections needed during alignment.

    `debug_log` records the RAW model output (every word/score, including
    ones later dropped for missing text/timing) before any of our own
    filtering -- see debug_log.DebugLog. `vad_filter` only applies to the
    faster-whisper path. `whisperx_vad_options` only applies to the
    whisperx path -- see config.WHISPERX_NO_VAD_OPTIONS's docstring for
    why this fixed a real, confirmed timing bug (word timestamps up to
    ~6s wrong around sustained/held notes).
    """
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
