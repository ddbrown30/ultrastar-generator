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
from .models import Word


def _transcribe_with_whisperx(vocals_path: Path, model_name: str, device: str) -> List[Word]:
    import whisperx

    compute_type = "float16" if device == "cuda" else "int8"
    audio = whisperx.load_audio(str(vocals_path))

    model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language="en")
    result = model.transcribe(audio, language="en", batch_size=16)

    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device=device)

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


def _transcribe_with_faster_whisper(vocals_path: Path, model_name: str, device: str) -> List[Word]:
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        str(vocals_path),
        language="en",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    words: List[Word] = []
    for segment in segments:
        if not segment.words:
            continue
        for w in segment.words:
            text = w.word.strip()
            if not text:
                continue
            words.append(Word(
                text=text,
                start=float(w.start),
                end=float(w.end),
                confidence=float(getattr(w, "probability", 1.0)),
            ))
    return words


def transcribe_words(
    vocals_path: Path,
    model_name: str,
    device: str = "cpu",
    prefer_whisperx: bool = config.PREFER_WHISPERX,
) -> List[Word]:
    """Returns a flat, time-ordered list of Word objects for the whole track.

    Note: these timestamps are a starting point for lyric_alignment.py, not
    the final note timing -- final timing comes from note_detection.py's
    audio-only analysis. Still, more accurate word boundaries here mean
    fewer/less-drastic corrections needed during alignment.
    """
    if prefer_whisperx:
        try:
            return _transcribe_with_whisperx(vocals_path, model_name, device)
        except ImportError:
            print(
                "whisperx not installed -- falling back to faster-whisper's own "
                "word timestamps, which are noticeably less precise. "
                "For better timing accuracy: pip install whisperx"
            )
        except Exception as e:
            print(f"whisperx transcription failed ({e}); falling back to faster-whisper.")

    try:
        return _transcribe_with_faster_whisper(vocals_path, model_name, device)
    except ImportError as e:
        raise ImportError(
            "Neither whisperx nor faster-whisper is installed. "
            "Install at least one: pip install faster-whisper  (or)  pip install whisperx"
        ) from e
