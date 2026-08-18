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

import logging
import os
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from . import config
from . import model_cache
from .models import Word

# pyannote.audio (pulled in by whisperx's VAD) tries to use torchcodec for
# audio decoding and warns loudly when torchcodec's compiled DLLs can't find
# a compatible FFmpeg build on this machine. Harmless: WhisperX always hands
# pyannote a preloaded in-memory waveform, which is the documented fallback
# path this same warning points to -- never triggers the broken decode path.
warnings.filterwarnings(
    "ignore",
    message=r"\s*torchcodec is not installed correctly.*",
    category=UserWarning,
)

# whisperx.audio.load_audio() shells out to ffmpeg via a bare
# `subprocess.run(cmd, ...)` call inside the third-party package -- a call
# site we don't own, unlike separation.py/media_extract.py's own ffmpeg/
# Demucs calls, which pass CREATE_NO_WINDOW directly. On Windows, a
# subprocess spawned by a parent with no console of its own (the GUI,
# launched via pythonw.exe per run_gui.bat) otherwise pops up a new console
# window for the moment it runs. Since we can't pass creationflags into
# whisperx's own call, patch subprocess.Popen.__init__ itself, once, so
# every child process this python process ever spawns picks up the flag by
# default -- harmless for calls that already set creationflags explicitly
# (ours), a no-op on non-Windows.
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
    """See `_rewindow_long_segments`'s own docstring for the real case this
    fixes and its validation history. Sweeps fixed-width candidate windows across `segment`'s own
    declared [start, end] span, re-running wav2vec2 CTC alignment of the
    SAME text against each, and keeps whichever gives the best mean word
    score -- provided it clears the baseline (whole-segment) score by
    `config.REWINDOW_MIN_SCORE_IMPROVEMENT`. Returns (start, end, improved)
    -- `improved=False` means the caller should keep the segment's own
    original bounds untouched."""
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
    """Returns a NEW segment list with corrected (start, end) for any
    segment at least `config.REWINDOW_MIN_SEGMENT_DURATION_SEC` long --
    everything else (text, downstream whisperx.align()/Word construction)
    is untouched; this only fixes segment BOUNDARIES before they're used,
    so it composes with the normal pipeline for free. Unconditional
    (rolled into core 2026-08-17, no CLI/GUI off-switch anymore) --
    real-validated across 12 total runs (8 realign-mode songs, 4 full
    generation-pipeline songs), zero regressions, 6 genuine verified
    fixes carried through to the actual written output (not just the
    intermediate ASR score). See CLAUDE.md's "Long-segment
    re-windowing" section for the full validation history.

    The sweep in `_find_best_window` deliberately probes many implausible
    candidate windows (including narrow tail-end ones as short as
    `REWINDOW_STEP_SEC`, clamped against the segment's own end) -- most of
    these predictably fail whisperx's own forced-alignment backtrack (too
    little audio for the given text), score 0.0 via `_mean_word_score`, and
    are correctly never selected. That's expected, not a real problem, but
    whisperx logs each one as a WARNING regardless -- real case (David Bowie
    - Absolute Beginners) produced ~20 of these in one run, all from sweep
    candidates, none from the segment bounds actually used downstream.
    Suppresses `whisperx.alignment`'s own WARNING logging for the duration of
    this sweep so expected candidate failures don't spam the console/GUI log;
    restored before returning, so a genuine failure in the FINAL alignment
    call (on the segment bounds this function actually returns) still
    surfaces normally."""
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
    """Forces KNOWN `words_text` onto the audio inside [window_start,
    window_end] via a real wav2vec2 CTC forced alignment (whisperx.
    align()) -- unlike a normal ASR pass, this doesn't need the decoder to
    have "heard" these words at all; it directly measures where the GIVEN
    text best fits within the window. This is what recovers content a
    free transcription pass can drop outright (a decoder segment that
    silently omits real sung words -- see `_rewindow_long_segments`'s
    docstring for the same underlying failure mode), rather than hoping a
    bigger model happens to transcribe it (config.RETRY_ASR_MODEL, which
    real-world testing found doesn't reliably recover a specific dropped
    passage even when it does help other parts of a song) or re-searching
    an ASR transcript that may never have had these words in it at all
    (a text-search rematch was tried and rejected as `realign.
    rematch_local_gaps` for exactly this reason -- see CLAUDE.md's
    "Removed / rejected approaches").

    Approach and validation directly adapted from UltraStarKaraokeMaker's
    own `realign_gap_windows` (github.com/walterfr/UltraStarKaraokeMaker,
    python-sidecar/pipeline/align.py, MIT-style OSS) -- real case that
    motivated bringing this over: their output on "Trixie Mattel - Gold"
    correctly recovered a "Do-do-do-do-do" backing-vocal passage our own
    pipeline dropped outright, via exactly this mechanism (their own
    song_data.json tags the recovered words `"source": "realign"`).

    Returns None (caller should keep whatever fallback -- interpolation
    -- it already had) if: the window is too short to plausibly hold
    this many words; whisperx.align() raises (a genuinely bad window
    shouldn't crash the pipeline); the returned word count doesn't match
    `words_text` (ambiguous mapping -- e.g. a word split into multiple
    tokens); any word is missing a timestamp; or any word falls outside
    the window (with `config.FORCE_ALIGN_WINDOW_SLOP_SEC` slop) or out of
    order relative to the previous word -- never applies a partial/
    ambiguous result."""
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
    """DIAGNOSTIC/EXPERIMENTAL (config.PipelineOptions.no_transcribe /
    --no-transcribe, 2026-08-10): builds the ENTIRE initial Word list
    without ever running the WhisperX DECODER (`model.transcribe()`, the
    component that guesses what was sung and can hallucinate/drop content
    -- see the David Bowie - Magic Dance 'ic ic ic...' case in CLAUDE.md,
    where a decoder hallucination on a real repeated passage ultimately
    corrupted output text even after reference-lyrics correction). Only
    the SEPARATE wav2vec2 CTC forced-alignment model runs here
    (`force_align_words_in_window`, same mechanism
    `recover_dropped_reference_words` already uses to recover
    known-but-undetected words), driven entirely
    by the KNOWN text from a pinned LRC candidate's own synced lyrics --
    there is no ASR output at all for anything downstream to inherit
    hallucinated text from.

    This is meant for isolating whether ASR-decoder hallucination is
    bleeding into output some OTHER way this project hasn't found yet --
    not a replacement for the normal transcription path in general (it
    requires a trustworthy pinned LRC candidate with per-line timestamps,
    and gives up whatever real information the decoder would have added
    for content the LRC doesn't cover, e.g. ad-libs).

    Each LRC line gets its own window: [this line's timestamp, next
    line's timestamp), last line to `audio_duration`. A line whose
    force-alignment fails (window implausibly short for its word count,
    whisperx.align() itself raises, or the result is out-of-order/
    ambiguous -- see force_align_words_in_window's own contract) is
    skipped entirely and logged; never guessed or interpolated here --
    downstream stages (force-align-gaps, pass 3's own fallback-to-
    nearest-note handling) already exist to cope with missing words."""
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
    ~6s wrong around sustained/held notes). The whisperx path always
    re-windows long decoder segments (see `_rewindow_long_segments`'s own
    docstring for the real case this fixes and its validation history).
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
