"""CLI entry point.

Usage:
    python -m ultrastar_generator.main "C:\\Songs\\Bon Jovi - Its My Life" --output-dir "C:\\Output"

The positional argument is a FOLDER (containing the audio -- or a video
that stands in for it, see song_input.py -- and optionally a companion
video/cover/background/MusicXML reference/existing .txt), not a single
file. See README.md for the full option list and setup instructions.
"""

from __future__ import annotations

import os
# Must be set before any CUDA context is created (i.e. before torch is
# imported anywhere, even transitively) for torch.use_deterministic_
# algorithms(True) to have full effect on cuBLAS ops -- confirmed
# non-deterministic run-to-run on identical audio otherwise for CUDA
# inference elsewhere in the pipeline (e.g. WhisperX).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import dataclasses
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import config
from .models import Song, Syllable, Word
from .file_discovery import (resolve_artist_title, headline_case, sanitize_filename,
                              AmbiguousInputError, NoAudioSourceFoundError)
from .song_input import resolve_song_folder
from .cover_fetch import fetch_cover_online
from .output_staging import stage_companions_to_output
from .youtube_source import download_youtube_source, YoutubeDownloadError
from .separation import isolate_vocals, SeparationError
from .transcription import transcribe_words, force_align_reference_lyrics
from .tempo import detect_bpm
from .note_detection import detect_notes, NoteEvent
from .alignment import align_words
from .phrasing import build_lines
from .lyrics_lookup import (fetch_reference_lyrics, parse_lyrics_lines, align_words_to_reference,
                             is_lrc_line_tracking_confident, alignment_diff_summary, reference_match_ratio,
                             largest_unmatched_reference_run, recover_dropped_reference_words,
                             fetch_lrclib_by_id, load_lrc_file, effective_lrc_duration)
from .musicxml_reference import apply_musicxml_references
from .mxl_lrc_generator import try_mxl_lrc_primary
from .lrc_timing import apply_lrc_timing_check
from .video_sync import estimate_videogap
from .usdx_writer import write_song
from .debug_output import write_pass1_debug_file
from .debug_log import DebugLog
from .worker_process import run_cancellable, WorkerError


@dataclass
class PipelineResult:
    """Outcome of one `run_pipeline` call. Never raises/exits on an
    "expected" failure (no usable audio, no notes detected, etc.) -- those
    all come back as success=False with a human-readable `error` instead,
    so callers (the CLI wrapper below, `run_batch` in batch.py, gui.py)
    can all handle failure uniformly without relying on process exit codes
    or exceptions for ordinary control flow."""
    success: bool
    output_txt_path: Optional[Path] = None
    error: Optional[str] = None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate an UltraStar Deluxe .txt song file from a song folder."
    )
    p.add_argument("input", help="Path to the song's folder (containing the audio, and optionally a "
                                  "video/cover/background/MusicXML reference/existing .txt). With "
                                  "--batch, this is a PARENT folder whose immediate subdirectories are "
                                  "each processed the same way. With --youtube-url, this folder is "
                                  "created if needed and the download lands directly in it.")
    p.add_argument("--audio-file", default=None,
                    help="Which file (bare filename, within the input folder) to use as the audio/video "
                         "source -- required only if the folder contains more than one real audio file "
                         "(mp3/ogg/oga/m4a), which this tool otherwise refuses to guess between.")
    p.add_argument("--youtube-url", default=None,
                    help="Download this video instead of using local files -- lands directly in the "
                         "input folder, then processed exactly like any other song folder. REQUIRES "
                         "--artist and --title (a YouTube video's own title isn't a reliable "
                         "'Artist - Title' source).")
    p.add_argument("--youtube-audio-only", dest="youtube_audio_only", action="store_true", default=True,
                    help="With --youtube-url: download audio only, as mp3 (default: on -- it's often "
                         "the case you don't want the video).")
    p.add_argument("--youtube-video", dest="youtube_audio_only", action="store_false",
                    help="With --youtube-url: download the full video (mp4) instead of audio-only.")
    p.add_argument("--batch", action="store_true",
                    help="Treat the positional argument as a PARENT folder: run the normal single-song "
                         "pipeline on each of its immediate subdirectories (not the parent itself), "
                         "mirroring the input structure into --output-dir (each song's own "
                         "'<Artist> - <Title>' folder is then created under its mirrored subfolder). "
                         "One song failing does not abort the rest. Not allowed together with "
                         "--artist/--title/--existing-txt (a single override doesn't make sense across "
                         "multiple songs), --youtube-url (one URL can't populate N subfolders), or "
                         "--work-dir (a shared override would collide every song's Demucs cache into "
                         "one directory).")
    p.add_argument("--artist", help="Override artist parsed from filename")
    p.add_argument("--title", help="Override title parsed from filename")
    p.add_argument("--output-dir",
                    help="PARENT folder under which a '<Artist> - <Title>' folder is created and "
                         "written to (e.g. --output-dir C:\\output produces "
                         "C:\\output\\<Artist> - <Title>\\). Default: <input-folder>\\Output.")
    p.add_argument("--work-dir", help="Scratch space for separated stems etc. (default: <input-folder>/.ultrastar_work "
                                        "-- deliberately tied to the input, not --output-dir, so separation is reused across "
                                        "multiple runs of the same song even when writing output elsewhere). Not allowed "
                                        "together with --batch, since a shared override would collide every song's cache "
                                        "into one directory.")
    p.add_argument("--delete-work-files", action="store_true",
                    help="Delete the large, fully-regeneratable work files under "
                         "<input-folder>/.ultrastar_work "
                         "Default: OFF (keeps them so re-runs reuse the cached separation).")
    p.add_argument("--whisper-model", default=config.DEFAULT_WHISPER_MODEL,
                    help=f"whisper model name for the main transcription pass (default: {config.DEFAULT_WHISPER_MODEL}). "
                         "This is what drives word timing accuracy feeding pass 3's note-zone assignment -- a bigger "
                         "model here is the lever worth pulling for accuracy.")
    p.add_argument("--demucs-model", default=config.DEFAULT_DEMUCS_MODEL,
                    help=f"Demucs model name (default: {config.DEFAULT_DEMUCS_MODEL})")
    p.add_argument("--bpm", type=float, default=None, help="Override detected BPM")
    p.add_argument("--skip-separation", action="store_true",
                    help="Skip Demucs; use --vocals-path instead (e.g. you already have an isolated vocal stem)")
    p.add_argument("--vocals-path", help="Path to a pre-isolated vocal stem (wav), used with --skip-separation")
    p.add_argument("--fetch-lyrics", dest="fetch_lyrics", action="store_true", default=True,
                    help="Fetch reference lyrics online (LRCLIB, synced lyrics only) to correct low-confidence "
                         "ASR words and mark line breaks (default: on)")
    p.add_argument("--no-fetch-lyrics", dest="fetch_lyrics", action="store_false",
                    help="Disable online lyric lookup/correction")
    p.add_argument("--fetch-cover", dest="fetch_cover", action="store_true", default=True,
                    help="Download a cover image online (MusicBrainz/Cover Art Archive, then iTunes, then "
                         "Deezer) when the input folder has no [CO]-tagged/embedded cover art of its own "
                         "(default: on)")
    p.add_argument("--no-fetch-cover", dest="fetch_cover", action="store_false",
                    help="Disable online cover-art download")
    p.add_argument("--no-video-sync", action="store_true",
                    help="Don't attempt to auto-compute #VIDEOGAP from the video's own audio track")
    p.add_argument("--no-whisperx", action="store_true",
                    help="Don't use whisperx forced alignment for word timing even if installed "
                         "(uses faster-whisper's own, less precise, timestamps instead)")
    p.add_argument("--no-transcribe", dest="no_transcribe", action="store_true", default=False,
                    help="DIAGNOSTIC (default: off). Skip the WhisperX DECODER entirely -- never guesses "
                         "what was sung, so it can't hallucinate/drop content. Instead force-aligns a "
                         "pinned LRC candidate's own known line text (--lrclib-id / pinned_lyrics, must "
                         "have synced lyrics) directly onto the audio via wav2vec2 CTC alone. Requires a "
                         "pinned LRC candidate with synced lyrics; falls back to normal transcription "
                         "with a warning if none is available.")
    p.add_argument("--whisperx-no-vad", dest="whisperx_no_vad", action="store_true",
                    default=config.ENABLE_WHISPERX_NO_VAD,
                    help="Force whisperx's own pyannote VAD to near-zero onset/offset thresholds "
                         "(no true off switch exists -- see config.WHISPERX_NO_VAD_OPTIONS) "
                         "(default: on). Confirmed fix for word timestamps being wrong by up to "
                         "~6 seconds around sustained/held sung notes -- VAD chunking appears to "
                         "corrupt the downstream wav2vec2 alignment's context on a long held note; "
                         "validated end-to-end (matches hand-verified reference timing exactly).")
    p.add_argument("--whisperx-vad", dest="whisperx_no_vad", action="store_false",
                    help="Use whisperx's own default pyannote VAD instead of the near-disabled "
                         "workaround -- re-enables the confirmed sustained-note timestamp bug; "
                         "kept only for comparison/debugging.")
    p.add_argument("--retry-low-quality-asr", dest="retry_low_quality_asr", action="store_true",
                    default=config.RETRY_LOW_QUALITY_ASR,
                    # NOTE: literal '%' in argparse help text must be escaped as '%%' -- argparse
                    # interpolates the stored help string with %-formatting when rendering --help.
                    help=(f"PROTOTYPE, default: ON. Retries the whole transcription once with "
                          f"'{config.RETRY_ASR_MODEL}' when ASR quality looks bad: the MXL+LRC primary path's "
                          f"own ASR-placement-rate gate (< {config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:.0%}), or, in "
                          f"the standard fallback path, either a right-song reference-lyrics match with weak "
                          f"vocabulary overlap (< {config.RETRY_ASR_MIN_REFERENCE_MATCH_RATIO:.0%}) or a run of "
                          f"{config.RETRY_ASR_MIN_UNMATCHED_REFERENCE_RUN}+ consecutive reference words ASR "
                          f"produced NO word for at all (one dropped passage, even if the rest of the song is "
                          f"fine). No-op if --whisper-model is already '{config.RETRY_ASR_MODEL}'; only kept if "
                          f"it actually scores better than the original. The actual re-transcription only runs "
                          f"in --batch mode (roughly doubles this run's ASR time) -- outside --batch, a "
                          f"triggered condition logs a WARNING suggesting --batch or a manual --whisper-model "
                          f"large-v3 instead. See CLAUDE.md.").replace("%", "%%"))
    p.add_argument("--no-retry-low-quality-asr", dest="retry_low_quality_asr", action="store_false")
    p.add_argument("--musicxml-reference", default=None,
                    help="Path to a MusicXML/.mxl file for this song (e.g. hand-downloaded sheet "
                         "music) -- pass 4, off unless given. Aligns by lyric text against pass 3's "
                         "output and corrects a syllable's PITCH CLASS (never octave, never timing) "
                         "where they disagree, once a per-song calibration offset can be trusted "
                         "(see config.MUSICXML_MIN_CALIBRATION_SAMPLES/_CONFIDENCE). No automatic "
                         "fetch exists for this file -- see CLAUDE.md.")
    p.add_argument("--musicxml-part", default=None,
                    help="Hint: which part name in the MusicXML file carries the lead vocal line, "
                         "for duet/ensemble arrangements where multiple parts have lyrics (e.g. a "
                         "character's own name, if the arrangement labels parts that way). Falls "
                         "back to the lyric-bearing part with the most notes if not given.")
    p.add_argument("--mxl-lrc-primary", dest="mxl_lrc_primary", action="store_true",
                    default=config.ENABLE_MXL_LRC_PRIMARY,
                    help="Default ON. When a MusicXML file (--musicxml-reference or auto-detected) AND "
                         "matching synced lyrics are both available, generate from those directly "
                         "(MusicXML for pitch, LRCLIB line starts as real-time anchors, real "
                         "transcription to place words within a line) instead of the standard "
                         "audio-only pass 1-4 pipeline -- validated real end-to-end: 100%% pitch-class "
                         "accuracy, 99%% timing within 500ms (see CLAUDE.md). Quality-gated: falls back "
                         "to the standard pipeline (with a warning) whenever no MusicXML/matching "
                         "lyrics are available or the result doesn't pass a consistency check.")
    p.add_argument("--no-mxl-lrc-primary", dest="mxl_lrc_primary", action="store_false",
                    help="Always use the standard audio-only pass 1-4 pipeline, even when a MusicXML "
                         "file and matching lyrics are available.")
    p.add_argument("--lrclib-id", dest="lrclib_id", type=int, default=None,
                    help="A specific LRCLIB entry id (browse lrclib.net yourself to find one, e.g. by "
                         "checking a linked video) -- always wins over search for both the MXL+LRC "
                         "primary path and the standard pipeline's own reference-lyrics fetch, no "
                         "ambiguity. Same idea as --existing-txt/--musicxml-reference: an explicit "
                         "override always wins over auto-detection.")
    p.add_argument("--lrc-file", dest="lrc_file", default=None,
                    help="Path to a LOCAL .lrc synced-lyrics file to use directly, bypassing LRCLIB "
                         "search/fetch entirely -- wins over both search and --lrclib-id if given. For "
                         "a user who already has a synced-lyrics file on disk, including a deliberately "
                         "imprecise one for testing how the LRC-seeding/windowing/MXL+LRC logic handles "
                         "an off-but-close candidate.")
    p.add_argument("--lrc-timing-check", dest="lrc_timing_check", action="store_true",
                    default=config.ENABLE_LRC_TIMING_CHECK,
                    help="EXPERIMENTAL, off by default. Cross-checks each line's assigned start time "
                         "against LRCLIB's synced lyrics (when available), once a per-song time "
                         "calibration offset can be trusted. DIAGNOSTIC ONLY -- flags disagreeing lines "
                         "in the console/debug log, never moves anything (see lrc_timing.py).")
    p.add_argument("--no-lrc-timing-check", dest="lrc_timing_check", action="store_false",
                    help="Disable the LRC timing check (no-op unless --lrc-timing-check or "
                         "config.ENABLE_LRC_TIMING_CHECK enabled it).")
    p.add_argument("--pitch-smooth-window", type=float, default=config.PITCH_SMOOTH_WINDOW_SEC,
                    help=f"Median-filter window (sec) for vibrato suppression before note "
                         f"segmentation (default: {config.PITCH_SMOOTH_WINDOW_SEC}). Raise this "
                         f"if notes are still fragmenting; lower it if fast melodic runs are "
                         f"getting smeared into one note.")
    p.add_argument("--note-split-semitones", type=float, default=config.NOTE_SPLIT_SEMITONES,
                    help=f"Pitch change (semitones) on the smoothed contour required to start a "
                         f"new note (default: {config.NOTE_SPLIT_SEMITONES})")
    p.add_argument("--min-note-beat-fraction", type=float, default=config.MIN_NOTE_BEATS_FRACTION,
                    help=f"Notes shorter than this fraction of one beat get merged into a "
                         f"neighbor (default: {config.MIN_NOTE_BEATS_FRACTION})")
    p.add_argument("--silence-threshold-db", type=float, default=config.SILENCE_THRESHOLD_DB_BELOW_PEAK,
                    help=f"A frame this many dB quieter than the track's own loud-reference level "
                         f"is treated as silence/noise regardless of pYIN's own voicing decision "
                         f"(default: {config.SILENCE_THRESHOLD_DB_BELOW_PEAK}). Lower this if real "
                         f"quiet singing is getting cut; raise it if silent sections are still "
                         f"generating hallucinated notes.")
    p.add_argument("--silence-floor-db", type=float, default=config.SILENCE_ABSOLUTE_FLOOR_DB,
                    help=f"Absolute dBFS floor below which a frame is always treated as silence, "
                         f"regardless of relative comparison to the rest of the track -- needed "
                         f"because a long/entirely silent stretch has no louder reference for "
                         f"--silence-threshold-db to compare against (default: "
                         f"{config.SILENCE_ABSOLUTE_FLOOR_DB})")
    p.add_argument("--spike-max-duration", type=float, default=config.SPIKE_MAX_DURATION_SEC,
                    help=f"A note this short (seconds) that jumps far in pitch from both neighbors "
                         f"(which are themselves close to each other) is treated as a tracking "
                         f"glitch and removed (default: {config.SPIKE_MAX_DURATION_SEC})")
    p.add_argument("--spike-jump-semitones", type=float, default=config.SPIKE_MIN_JUMP_SEMITONES,
                    help=f"Minimum pitch jump (semitones) from both neighbors for a short note to "
                         f"be treated as a spike/glitch (default: {config.SPIKE_MIN_JUMP_SEMITONES})")
    p.add_argument("--ambiguity-key-tiebreak", dest="ambiguity_key_tiebreak", action="store_true",
                    default=config.ENABLE_AMBIGUITY_KEY_TIEBREAK,
                    help="Default: ON. RMVPE-only (no-op for --pitch-source swiftf0). Recomputes each "
                         "note's PITCH CLASS (never octave) by summing RMVPE's own raw per-frame "
                         "salience distribution across the note's own span instead of a per-frame-"
                         "rounded confidence-weighted mode; when the top-2 candidate pitch classes are "
                         "genuinely close (--ambiguity-margin-threshold), breaks the tie using the "
                         "song's own detected key (published Krumhansl-Kessler profiles) -- a confident, "
                         "unambiguous note is never touched, in- or out-of-key. Real-audio validated: "
                         "weighted +2.4pp pitch-class accuracy across a 12-song roster, 9 songs improved, "
                         "2 modest regressions (not universal). See CLAUDE.md / pitch_ambiguity.py.")
    p.add_argument("--no-ambiguity-key-tiebreak", dest="ambiguity_key_tiebreak", action="store_false")
    p.add_argument("--ambiguity-margin-threshold", type=float, default=config.AMBIGUITY_MARGIN_THRESHOLD,
                    help="Relative salience-mass margin between the top-2 candidate pitch classes below "
                         "which a note is treated as genuinely ambiguous and eligible for the key-profile "
                         "tie-break (default: {0}, the single best value found sweeping 0.10-1.00 on real "
                         "audio -- see config.AMBIGUITY_MARGIN_THRESHOLD's own comment)".format(
                             config.AMBIGUITY_MARGIN_THRESHOLD))
    p.add_argument("--pitch-source", default=config.DEFAULT_PITCH_SOURCE, choices=["rmvpe", "swiftf0"],
                    help="Which pass-1 pitch source to use (default: "
                         f"{config.DEFAULT_PITCH_SOURCE!r}). The chosen source alone supplies both "
                         "the pitch value and the voicing decision -- no cross-check, no ensemble "
                         "with any other source. 'rmvpe': validated 2026-08-09 as a real, "
                         "reproducible +1.7pp average improvement over the old pyin-primary + "
                         "CREPE/RMVPE-cross-check ensemble (since removed). 'swiftf0': lightweight "
                         "CNN pitch detector with a real native voicing decision of its own.")
    p.add_argument("--no-pass1-debug", action="store_true",
                    help="Don't write the '[PASS1 DEBUG]' .txt (pass-1 notes only, no lyrics) "
                         "that's written by default into <input-folder>/.ultrastar_work -- load it "
                         "in the UltraStar editor to check pass 1's timing/pitch in isolation.")
    p.add_argument("--no-debug-log", action="store_true",
                    help="Don't write the '[DEBUG LOG]' .txt that's written by default into "
                         "<input-folder>/.ultrastar_work -- records raw ASR word timing/confidence, "
                         "reference-line grouping, note-zone boundary math, and syllable-proportional "
                         "split decisions, for tracing a wrong final word/note position back to which "
                         "pipeline stage actually caused it.")
    p.add_argument("--quiet", action="store_true",
                    help="Suppress the verbose [pass1] diagnostic logging (still prints "
                         "the main pipeline stage messages)")
    return p


def check_cuda_available() -> Optional[str]:
    """Returns an error message if CUDA isn't available, else None. Callers
    (the CLI wrapper below, gui.py) each call this once, up front -- it's
    not re-checked per song/per batch item."""
    import torch
    if not torch.cuda.is_available():
        return ("CUDA is not available, but this pipeline requires it (CPU support has "
                "been removed). Check your PyTorch install / GPU drivers.")
    return None


def delete_work_files(work_dir: Path) -> None:
    """Deletes the .ultrastar_work directory"""
    import shutil
    work_dir = Path(work_dir)
    if work_dir.is_dir():
        shutil.rmtree(work_dir, ignore_errors=True)


def _transcribe_words_cancellable(vocals_path: Path, model_name: str, opts: config.PipelineOptions,
                                   debug_log, log: Callable[[str], None]):
    """Same contract as transcription.transcribe_words, routed through a
    killable child process (worker_process.run_cancellable) whenever
    `opts.cancel_requested` is set -- see that module's own docstring for
    why (a GUI Stop press can then kill this mid-call instead of waiting
    for it to finish). Falls straight through to the in-process call
    otherwise (the CLI, which gets real instant cancellation via Ctrl+C
    for free and shouldn't pay subprocess overhead for nothing)."""
    whisperx_vad_options = config.WHISPERX_NO_VAD_OPTIONS if opts.whisperx_no_vad else None
    if opts.cancel_requested is None:
        return transcribe_words(
            vocals_path, model_name, prefer_whisperx=not opts.no_whisperx, debug_log=debug_log,
            whisperx_vad_options=whisperx_vad_options,
        )
    result = run_cancellable(
        "transcribe_words",
        {"vocals_path": str(vocals_path), "model_name": model_name, "prefer_whisperx": not opts.no_whisperx,
         "whisperx_vad_options": whisperx_vad_options},
        cancel_requested=opts.cancel_requested, debug_log=debug_log, log=log,
    )
    return [Word(**d) for d in result]


def _detect_notes_cancellable(vocals_path: Path, y, sr, bpm: float, opts: config.PipelineOptions,
                               debug_log, log: Callable[[str], None]):
    """Same contract as note_detection.detect_notes, routed through a
    killable child process when `opts.cancel_requested` is set -- see
    _transcribe_words_cancellable's own docstring for the full reasoning.
    The subprocess re-loads audio from `vocals_path` itself (cheap, and
    avoids pickling a potentially large `y` array through the request
    file) rather than being handed `y`/`sr` directly."""
    if opts.cancel_requested is None:
        return detect_notes(
            y, sr, bpm=bpm, pitch_source=opts.pitch_source, smooth_window_sec=opts.pitch_smooth_window,
            pitch_jump_semitones=opts.note_split_semitones, min_note_beats_fraction=opts.min_note_beat_fraction,
            silence_threshold_db=opts.silence_threshold_db, silence_absolute_floor_db=opts.silence_floor_db,
            spike_max_duration_sec=opts.spike_max_duration, spike_min_jump_semitones=opts.spike_jump_semitones,
            enable_ambiguity_key_tiebreak=opts.ambiguity_key_tiebreak,
            ambiguity_margin_threshold=opts.ambiguity_margin_threshold,
            verbose=not opts.quiet, debug_log=debug_log,
        )
    result = run_cancellable(
        "detect_notes",
        {"vocals_path": str(vocals_path), "bpm": bpm, "pitch_source": opts.pitch_source,
         "smooth_window_sec": opts.pitch_smooth_window, "pitch_jump_semitones": opts.note_split_semitones,
         "min_note_beats_fraction": opts.min_note_beat_fraction, "silence_threshold_db": opts.silence_threshold_db,
         "silence_floor_db": opts.silence_floor_db, "spike_max_duration_sec": opts.spike_max_duration,
         "spike_min_jump_semitones": opts.spike_jump_semitones,
         "ambiguity_key_tiebreak": opts.ambiguity_key_tiebreak,
         "ambiguity_margin_threshold": opts.ambiguity_margin_threshold,
         "verbose": not opts.quiet},
        cancel_requested=opts.cancel_requested, debug_log=debug_log, log=log,
    )
    return [NoteEvent(**d) for d in result]


def _recover_dropped_reference_words_cancellable(ref_lines, words, vocals_path: Path,
                                                   opts: config.PipelineOptions, debug_log,
                                                   log: Callable[[str], None]):
    """Same contract as lyrics_lookup.recover_dropped_reference_words,
    routed through a killable child process whenever `opts.cancel_
    requested` is set -- see _transcribe_words_cancellable's own
    docstring."""
    if opts.cancel_requested is None:
        return recover_dropped_reference_words(ref_lines, words, vocals_path, debug_log=debug_log)
    result = run_cancellable(
        "recover_dropped_reference_words",
        {"ref_lines": ref_lines, "words": [dataclasses.asdict(w) for w in words], "vocals_path": str(vocals_path)},
        cancel_requested=opts.cancel_requested, debug_log=debug_log, log=log,
    )
    return [Word(**d) for d in result["words"]], result["n_recovered"]


def _force_align_reference_lyrics_cancellable(vocals_path: Path, synced_lyrics: str, audio_duration: float,
                                                opts: config.PipelineOptions, debug_log,
                                                log: Callable[[str], None]):
    """Same contract as transcription.force_align_reference_lyrics,
    routed through a killable child process whenever `opts.cancel_
    requested` is set -- see _transcribe_words_cancellable's own
    docstring."""
    if opts.cancel_requested is None:
        return force_align_reference_lyrics(vocals_path, synced_lyrics, audio_duration, debug_log=debug_log)
    result = run_cancellable(
        "force_align_reference_lyrics",
        {"vocals_path": str(vocals_path), "synced_lyrics": synced_lyrics, "audio_duration": audio_duration},
        cancel_requested=opts.cancel_requested, debug_log=debug_log, log=log,
    )
    return [Word(**d) for d in result]


def _retry_transcribe(vocals_path: Path, opts: config.PipelineOptions, debug_log, log: Callable[[str], None]):
    """PROTOTYPE, see config.RETRY_LOW_QUALITY_ASR's docstring. Re-runs
    transcription with `config.RETRY_ASR_MODEL` -- shared by both retry
    sites in `_run_pipeline_body` (the MXL+LRC primary path's own
    `asr_placement_rate` gate, and the standard fallback path's
    reference-match-ratio check) so the re-transcribe call itself isn't
    duplicated. The caller writes its own (more specific, e.g. "MXL+LRC
    path" vs "standard fallback path") `debug_log.section(...)` marker
    before calling this -- this only logs the one line common to both."""
    config.check_cancelled(opts.cancel_requested)
    log(f"  Retrying transcription with '{config.RETRY_ASR_MODEL}'...")
    if debug_log is not None:
        debug_log.line(f"  Retrying transcription with '{config.RETRY_ASR_MODEL}'...")
    return _transcribe_words_cancellable(vocals_path, config.RETRY_ASR_MODEL, opts, debug_log, log)


def run_pipeline(input_dir: Path, output_dir: Optional[Path], opts: config.PipelineOptions,
                  *, log: Callable[[str], None] = print) -> PipelineResult:
    """Runs the full pipeline for one song folder. Never raises on an
    "expected" failure (no usable audio source, ambiguous folder contents,
    no notes/words detected, etc.) -- those come back as
    `PipelineResult(success=False, error=...)` instead, so this can be
    called directly from the CLI wrapper, `run_batch` (batch.py), or
    gui.py without any of them needing try/except or exit-code handling
    for ordinary control flow. `log` defaults to `print` but the GUI
    passes something that feeds a queue/log widget instead.

    Thin wrapper around `_run_pipeline_body` so that `opts.
    delete_work_files` can be honored via `finally`, regardless of
    which of `_run_pipeline_body`'s several early-return failure paths
    was taken -- work_dir may be partially populated (e.g. separation
    already ran) even on a failed run. work_dir's own location is
    recomputed here rather than threaded back out of the body: it's a
    pure function of (input_dir, opts.work_dir), so recomputing it can
    never diverge from what the body itself used.
    """
    try:
        return _run_pipeline_body(input_dir, output_dir, opts, log=log)
    except config.PipelineCancelled:
        log("Cancelled by user.")
        return PipelineResult(success=False, error="Cancelled by user.")
    except WorkerError as e:
        return PipelineResult(success=False, error=str(e))
    finally:
        if opts.delete_work_files:
            wd = Path(opts.work_dir).resolve() if opts.work_dir else (Path(input_dir) / ".ultrastar_work")
            delete_work_files(wd)


def _run_pipeline_body(input_dir: Path, output_dir: Optional[Path], opts: config.PipelineOptions,
                        *, log: Callable[[str], None] = print) -> PipelineResult:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else None
    # work_dir is tied to the INPUT folder, never output_dir -- separation
    # (and everything cached under it, including which vocals.wav BPM
    # detection sees) should be reused across multiple runs of the same
    # song even when --output-dir differs (e.g. comparing --whisper-model
    # choices into separate output folders). Demucs isn't bit-reproducible
    # run to run (see CLAUDE.md's "Lessons learned" -- confirmed in
    # practice: two separations of the same song produced different-
    # checksum vocals.wav, which was enough to flip detected BPM between
    # 105.47 and 109.96), so reusing the SAME cached separation avoids
    # that instability entirely rather than trying to fix Demucs itself.
    work_dir = Path(opts.work_dir).resolve() if opts.work_dir else (input_dir / ".ultrastar_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- YouTube input (feature 7): downloads directly INTO input_dir, then
    # falls through to the exact same folder-resolution logic as every other
    # song -- an otherwise-empty folder with one freshly-downloaded mp3/mp4
    # in it is auto-classified correctly (kind="audio" or "video_as_audio")
    # with no special-casing needed. A YouTube title isn't a reliable
    # "Artist - Title" source, so this requires --artist/--title explicitly
    # (checked here, not just left to fail later at filename-parsing).
    if opts.youtube_url:
        if not (opts.artist and opts.title):
            return PipelineResult(success=False, error=(
                "--youtube-url requires --artist and --title (a YouTube video's own "
                "title isn't a reliable 'Artist - Title' source)."))
        input_dir.mkdir(parents=True, exist_ok=True)
        log(f"Downloading from YouTube ({'audio only' if opts.youtube_audio_only else 'video'}): {opts.youtube_url}")
        try:
            downloaded = download_youtube_source(opts.youtube_url, input_dir, audio_only=opts.youtube_audio_only)
        except YoutubeDownloadError as e:
            return PipelineResult(success=False, error=str(e))
        log(f"Downloaded: {downloaded}")

    try:
        resolved = resolve_song_folder(input_dir, work_dir, audio_file_override=opts.audio_file)
    except (AmbiguousInputError, NoAudioSourceFoundError) as e:
        return PipelineResult(success=False, error=str(e))

    audio_path = resolved.analysis_audio  # real, decodable audio -- feeds Demucs/pass1/WhisperX

    if opts.artist and opts.title:
        artist, title = opts.artist, opts.title
    else:
        parsed_artist, parsed_title = resolve_artist_title(resolved.output_mp3_source, input_dir)
        if parsed_artist is None or parsed_title is None:
            return PipelineResult(success=False, error=(
                f'Could not parse "<Artist> - <Title>" from the input folder name '
                f"({input_dir.name!r}). Pass --artist and --title explicitly instead."))
        artist = opts.artist or parsed_artist
        title = opts.title or parsed_title

    # Headline-cased for every OUTPUT folder/file name and #ARTIST/#TITLE
    # tag from here on (e.g. "Beauty And The Beast" -> "Beauty and the
    # Beast") -- applied uniformly regardless of source (folder name or
    # explicit --artist/--title), since this is purely about the output
    # naming convention. Deliberately conservative (see headline_case's
    # own docstring): never touches an ALL CAPS or unusually-cased word
    # like "KPop", so intentional stylization survives untouched.
    artist = headline_case(artist)
    title = headline_case(title)

    # --output-dir is now the PARENT folder under which a "<Artist> -
    # <Title>" folder is created (e.g. --output-dir C:\output produces
    # C:\output\<Artist> - <Title>\) -- not the final folder itself.
    # Optional; defaults to <input folder>\Output as that parent.
    if output_dir is None:
        output_dir = input_dir / "Output"
    output_dir = output_dir / sanitize_filename(f"{artist} - {title}")

    # Checked here (not earlier) since it needs the FINAL folder, which
    # needs artist/title -- a given/defaulted PARENT equalling input_dir
    # is fine (e.g. --output-dir <same as input> puts output right inside
    # the song folder, organized by its own "<Artist> - <Title>"
    # subfolder); only an exact collision of the final folder itself with
    # input_dir is a real problem (would overwrite the song's own files).
    if input_dir.resolve() == output_dir.resolve():
        return PipelineResult(success=False, error=(
            f"The output folder ({output_dir}) would be the same as the input folder -- "
            f"pick a different --output-dir, or a different input folder name."))

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Forced/pinned LRCLIB candidate (feature: MXL+LRC primary path) --
    # An explicit --lrclib-id (CLI) or the GUI's own pre-search pin always
    # wins over automatic search, everywhere a candidate is needed -- both
    # the new MXL+LRC primary path below AND, on fallback, the old
    # reference-lyrics-fetch step. `pinned_lyrics` (a full object, already
    # resolved) takes priority if somehow both are set.
    forced_lrc_candidate = opts.pinned_lyrics
    if forced_lrc_candidate is None and opts.lrclib_id is not None:
        log(f"Fetching pinned LRCLIB entry (id={opts.lrclib_id})...")
        forced_lrc_candidate = fetch_lrclib_by_id(opts.lrclib_id)
        if forced_lrc_candidate is None:
            log(f"  Could not fetch LRCLIB id {opts.lrclib_id} (not found, or no network) -- ignoring.")
        else:
            log(f"  Using: {forced_lrc_candidate.track_name!r} / {forced_lrc_candidate.artist_name!r}")

    log(f"== {artist} - {title} ==")

    debug_log_path = (None if opts.no_debug_log else
                       (work_dir / sanitize_filename(f"{artist} - {title} [DEBUG LOG].txt")))
    debug_log = DebugLog(debug_log_path)
    if debug_log_path is not None:
        log(f"Writing debug log to: {debug_log_path}")

    # Mirrors a decision-narration message into the debug log FILE too, not just the console/GUI `log`
    # callback -- otherwise the file only shows a retry's raw ASR data (via `debug_log` threaded into
    # transcribe_words/try_mxl_lrc_primary) with no record of WHY it fired or what was decided, forcing
    # anyone auditing the file alone to go dig up separately-captured console output instead. Used by
    # both ASR-quality retry sites below (see config.RETRY_LOW_QUALITY_ASR's docstring).
    def dlog(msg: str) -> None:
        log(msg)
        debug_log.line(msg)

    # --- 1. Companion files (already resolved above) -------------------------
    for note in resolved.notes:
        log(note)
    if resolved.output_video_source:
        log(f"Found video: {resolved.output_video_source.name}")
    if resolved.cover:
        log(f"Found cover: {resolved.cover.name}")
    elif opts.fetch_cover:
        downloaded_cover = fetch_cover_online(artist, title, work_dir / "extracted", resolved.analysis_audio.stem)
        if downloaded_cover is not None:
            resolved.cover = downloaded_cover
            log(f"Downloaded cover art online -> {downloaded_cover.name}")
        else:
            log("  Could not find/download cover art online (not found on MusicBrainz/iTunes/Deezer, "
                "or no network).")
    if resolved.background:
        log(f"Found background: {resolved.background.name}")
    # An explicit --musicxml-reference always wins; otherwise falls back to
    # whatever file_discovery.find_companions auto-detected in the song's
    # own folder (may be zero, one, or several). Used by BOTH the MXL+LRC
    # primary path below and, on fallback, pass 4.
    mxl_paths = [opts.musicxml_reference] if opts.musicxml_reference else [str(p) for p in resolved.musicxml]
    if resolved.musicxml and not opts.musicxml_reference:
        names = ", ".join(p.name for p in resolved.musicxml)
        log(f"Found MusicXML reference file(s): {names}")

    # --- 2. Vocal isolation --------------------------------------------------
    config.check_cancelled(opts.cancel_requested)
    if opts.skip_separation:
        if not opts.vocals_path:
            return PipelineResult(success=False, error="--skip-separation requires --vocals-path")
        vocals_path = Path(opts.vocals_path).resolve()
    else:
        log("Isolating vocals with Demucs...")
        try:
            vocals_path = isolate_vocals(audio_path, work_dir, model=opts.demucs_model,
                                          cancel_requested=opts.cancel_requested)
        except SeparationError as e:
            return PipelineResult(success=False, error=f"Vocal isolation failed: {e}")
    log(f"Vocals: {vocals_path}")

    # --- 3. Load vocal audio for pitch analysis + tempo detection -----------
    import librosa
    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)
    audio_duration = len(y) / sr

    bpm = opts.bpm_override or detect_bpm(y, sr)
    # write_bpm (not bpm) is used for the actual .txt/#BPM/beat-quantization --
    # bpm itself stays the real detected tempo for pass 1's own audio analysis,
    # which is tuned against real beat duration, not display resolution.
    write_bpm = bpm * config.BPM_WRITE_MULTIPLIER
    log(f"BPM: {bpm} (detected/real tempo, used for pass-1 analysis); "
        f"{write_bpm} written to the .txt for finer beat-grid resolution "
        f"(UltraStar multiplies by 4 for the real note grid).")

    # --- 4. Transcription (lyrics text + rough timing) -- moved ahead of pass 1
    # so the MXL+LRC primary path below can use it without a second, redundant
    # transcription call; pass 1's own note detection never depended on it.
    # `no_transcribe` (DIAGNOSTIC, --no-transcribe, default off): bypasses this
    # entirely -- the WhisperX DECODER never runs, so it can never hallucinate/
    # drop content in the first place (see force_align_reference_lyrics's own
    # docstring, and the Magic Dance 'ic ic ic...' case in CLAUDE.md that
    # motivated this). Needs a pinned LRC candidate WITH synced lyrics already
    # resolved above (forced_lrc_candidate); falls back to normal transcription
    # with a warning otherwise, same as any other unmet-precondition case in
    # this pipeline.
    forced_align_only = False
    if opts.no_transcribe and forced_lrc_candidate is not None and forced_lrc_candidate.synced_lyrics:
        log("--no-transcribe: skipping the WhisperX decoder entirely -- force-aligning the pinned LRC "
            "candidate's own known line text onto the audio via wav2vec2 CTC alone...")
        words = _force_align_reference_lyrics_cancellable(vocals_path, forced_lrc_candidate.synced_lyrics,
                                                            audio_duration, opts, debug_log, log)
        if words:
            forced_align_only = True
            log(f"Force-aligned {len(words)} word(s) directly from the pinned LRC candidate's synced "
                f"lyrics -- no ASR decoder output exists anywhere downstream of this.")
        else:
            log("  Forced alignment against the pinned LRC candidate produced no words at all "
                "(every line failed -- check it actually matches this audio); falling back to normal "
                "transcription.")
    elif opts.no_transcribe:
        log("--no-transcribe requires a pinned LRC candidate WITH synced lyrics (--lrclib-id, or a "
            "pinned candidate with no per-line timestamps); none available here -- falling back to "
            "normal transcription.")

    if not forced_align_only:
        config.check_cancelled(opts.cancel_requested)
        log(f"Transcribing with {'whisperx' if not opts.no_whisperx else 'faster-whisper'} ({opts.whisper_model})"
            f"{' [VAD near-disabled]' if opts.whisperx_no_vad else ''}...")
        words = _transcribe_words_cancellable(vocals_path, opts.whisper_model, opts, debug_log, log)
    if not words:
        return PipelineResult(success=False,
                               error="No words were transcribed -- check the audio / vocal isolation quality.")
    log(f"Transcribed {len(words)} words.")

    # --- 5. MXL+LRC primary generation (default path) -- MusicXML for pitch,
    # LRCLIB synced-lyrics line starts as real-time anchors, real transcription
    # (above) to place words within a line. Quality-gated: falls back to the
    # standard pass 1-4 pipeline below whenever no MusicXML is available, no
    # LRC candidate can be found/forced, or the result fails the quality gate
    # -- see mxl_lrc_generator.py's module docstring for why the gate is
    # trusted over trying to perfect upfront candidate selection.
    syllables = None
    ref_lines = None
    synced_lyrics_text = None
    # Set True only by the standard fallback path below, when
    # lyrics_lookup.is_lrc_line_tracking_confident trusted real LRC line
    # timing against this song's own audio -- NOT simply whenever
    # synced_lyrics_text is set (that also happens on the unrelated
    # MXL+LRC primary path, which has its own separate word-placement
    # logic and was never validated against phrasing.py's strict mode).
    # Tells build_lines to trust every word's line_id completely (break
    # IFF the reference line changed, no other heuristic) instead of the
    # default gap/length-based safety net -- user's explicit directive:
    # when using LRC, match it 100%, no exceptions. See phrasing.py.
    strict_line_breaks_from_lrc = False
    # Tracks whichever model's transcription `words` currently holds --
    # NOT always opts.whisper_model, since a low-quality-ASR retry below
    # may have already swapped `words` for a config.RETRY_ASR_MODEL
    # transcription. Guards both retry sites so a song that retries once
    # in the MXL+LRC path never retries a second, redundant time in the
    # standard fallback path below.
    current_asr_model = opts.whisper_model
    if opts.mxl_lrc_primary and mxl_paths:
        config.check_cancelled(opts.cancel_requested)
        log("Attempting MXL+LRC primary generation (MusicXML pitch + synced-lyric line anchors + ASR word placement)...")
        mxl_lrc_result = try_mxl_lrc_primary(
            mxl_paths, artist, title, audio_duration, words,
            forced_candidate=forced_lrc_candidate, preferred_part_name=opts.musicxml_part,
            vocals_path=vocals_path, debug_log=debug_log,
        )
        # PROTOTYPE ASR-quality retry, see config.RETRY_LOW_QUALITY_ASR's
        # docstring. Reuses the SAME asr_placement_rate/MXL_LRC_MIN_ASR_
        # PLACEMENT_RATE gate `generate_from_mxl_and_lrc` already computes
        # for its own "does the matched candidate really correspond to
        # this recording" check -- if that's what's failing (not e.g. "no
        # candidate found at all", which a bigger ASR model can't fix),
        # retry the whole transcription once with config.RETRY_ASR_MODEL
        # and re-run try_mxl_lrc_primary against it, keeping whichever
        # attempt has the higher placement rate. A song where this retry
        # succeeds gets the retried transcription carried forward into
        # `words` even if used later (e.g. the standard fallback path).
        # PROTOTYPE ASR-quality retry gating (2026-08-10, user's explicit request): a whole-song
        # re-transcription roughly doubles this run's ASR time -- a reasonable cost to pay automatically
        # in an unattended --batch run, but not one to silently eat in a single-song run where the user is
        # presumably watching. Outside --batch, log a WARNING with the same explanation instead of retrying.
        mxl_retry_triggered = (opts.retry_low_quality_asr and mxl_lrc_result is not None
                                and mxl_lrc_result.quality is not None
                                and mxl_lrc_result.quality.asr_placement_rate < config.MXL_LRC_MIN_ASR_PLACEMENT_RATE
                                and current_asr_model != config.RETRY_ASR_MODEL)
        if mxl_retry_triggered and not opts.batch:
            dlog(f"  WARNING: only {mxl_lrc_result.quality.asr_placement_rate:.0%} of MXL words got a real ASR "
                 f"anchor (need {config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:.0%}) -- retrying with "
                 f"'{config.RETRY_ASR_MODEL}' would likely help, but automatic retries only run in --batch "
                 f"mode. Re-run with --batch, or pass --whisper-model {config.RETRY_ASR_MODEL} directly, to "
                 f"get this fixed.")
        elif mxl_retry_triggered:
            debug_log.section(f"ASR QUALITY RETRY (MXL+LRC path) -- re-transcribing with '{config.RETRY_ASR_MODEL}'")
            dlog(f"  Only {mxl_lrc_result.quality.asr_placement_rate:.0%} of MXL words got a real ASR anchor "
                 f"(need {config.MXL_LRC_MIN_ASR_PLACEMENT_RATE:.0%}) -- retrying before accepting this result.")
            retry_words = _retry_transcribe(vocals_path, opts, debug_log, log)
            if retry_words:
                retry_mxl_result = try_mxl_lrc_primary(
                    mxl_paths, artist, title, audio_duration, retry_words,
                    forced_candidate=forced_lrc_candidate, preferred_part_name=opts.musicxml_part,
                    vocals_path=vocals_path, debug_log=debug_log,
                )
                if (retry_mxl_result is not None and retry_mxl_result.quality is not None
                        and retry_mxl_result.quality.asr_placement_rate > mxl_lrc_result.quality.asr_placement_rate):
                    dlog(f"  Retry with '{config.RETRY_ASR_MODEL}' improved the ASR placement rate: "
                         f"{mxl_lrc_result.quality.asr_placement_rate:.0%} -> "
                         f"{retry_mxl_result.quality.asr_placement_rate:.0%} -- using the retry.")
                    mxl_lrc_result = retry_mxl_result
                    words = retry_words
                    current_asr_model = config.RETRY_ASR_MODEL
                else:
                    dlog(f"  Retry with '{config.RETRY_ASR_MODEL}' did not improve the ASR placement rate -- "
                         f"keeping the original transcription.")
            else:
                dlog(f"  Retry transcription with '{config.RETRY_ASR_MODEL}' produced no words -- keeping "
                     f"the original transcription.")
        if mxl_lrc_result is not None and mxl_lrc_result.time_calibration is not None:
            tc = mxl_lrc_result.time_calibration
            if tc.offset_sec is not None:
                drift_desc = f", drift {tc.slope:+.4f}s/LRC-s" if tc.kind == "drift" else ""
                rep_desc = " (representative only, see correction_fn)" if tc.kind in ("piecewise", "isotonic") else ""
                log(f"  LRC/audio time calibration ({tc.kind}): offset {tc.offset_sec:+.1f}s{drift_desc}{rep_desc} "
                    f"({tc.confidence:.0%} agreement) -- applied to LRC line timestamps before placement.")
            else:
                log(f"  LRC/audio time calibration: none found ({tc.skipped_reason}) -- using LRC "
                    f"timestamps as-is.")
        if mxl_lrc_result is not None and mxl_lrc_result.success:
            syllables = mxl_lrc_result.syllables
            synced_lyrics_text = mxl_lrc_result.lrc_match.candidate.synced_lyrics
            q = mxl_lrc_result.quality
            c = mxl_lrc_result.lrc_match.candidate
            log(f"  Success: {Path(mxl_lrc_result.mxl_path).name} (part(s): {mxl_lrc_result.part_names_used}) + "
                f"{c.track_name!r}/{c.artist_name!r} (lrclib id={c.id}) -- "
                f"{q.n_asr_placed}/{q.n_words} words placed via transcription, {q.n_fallback} via "
                f"proportional fallback, {q.non_monotonic_fix_count} monotonic fix(es).")
            log("  Skipping pass 1 (audio-only pitch detection) and pass 3/4 -- pitch comes directly "
                "from the MusicXML.")
            if mxl_lrc_result.pitch_calibration_offset is not None:
                log(f"  MXL pitch-class calibration: {mxl_lrc_result.pitch_calibration_offset:+d} "
                    f"semitone(s) ({mxl_lrc_result.pitch_calibration_confidence:.0%} agreement) -- "
                    f"corrected before writing.")
            else:
                log("  MXL pitch-class calibration: none applied (no confident offset found) -- "
                    "MXL pitch used as-is.")
            debug_log.log_lyrics_selection(
                source="lrclib (mxl+lrc primary path)", track_name=c.track_name, artist_name=c.artist_name,
                lrclib_id=c.id, duration=effective_lrc_duration(c), synced=bool(c.synced_lyrics),
                extra=f"{q.n_asr_placed}/{q.n_words} words placed via transcription, {q.n_fallback} via "
                      f"proportional fallback",
            )
        else:
            reason = mxl_lrc_result.reason if mxl_lrc_result is not None else "no MusicXML file available"
            log(f"  WARNING: MXL+LRC primary generation not usable ({reason}) -- falling back to "
                f"standard audio-based generation.")
            if opts.mxl_lrc_fallback_callback is not None:
                if not opts.mxl_lrc_fallback_callback(reason):
                    return PipelineResult(success=False, error=(
                        f"Cancelled: MXL+LRC primary generation unavailable ({reason}), user declined "
                        f"the standard-generation fallback."))
    elif opts.mxl_lrc_primary:
        log("No MusicXML file found for MXL+LRC primary generation; using standard audio-based generation.")

    if syllables is None:
        # --- FALLBACK: standard audio-based pass 1 -> lyrics fetch -> pass 3 -> pass 4, unchanged. ---

        # --- PASS 1: pitch/timing from audio alone, no lyrics involved -------
        config.check_cancelled(opts.cancel_requested)
        log("Pass 1: detecting notes from audio (pitch + timing only)...")
        notes = _detect_notes_cancellable(vocals_path, y, sr, bpm, opts, debug_log, log)
        if not notes:
            return PipelineResult(success=False,
                                   error="No notes were detected -- check the audio / vocal isolation quality.")

        if not opts.no_pass1_debug:
            debug_path = write_pass1_debug_file(notes, artist, title, resolved.output_mp3_source.name, write_bpm,
                                                 gap_ms=int(round(notes[0].start * 1000)),
                                                 output_dir=work_dir)
            log(f"Wrote pass-1 debug file (notes only, no lyrics): {debug_path}")
            log("  -> load this in the UltraStar editor to check timing/pitch BEFORE lyrics are involved.")

        # --- Reference lyrics: correct ASR text AND mark phrase/line breaks --
        if opts.fetch_lyrics:
            if forced_lrc_candidate is not None:
                log(f"Using pinned lyrics: {forced_lrc_candidate.track_name} - "
                    f"{forced_lrc_candidate.artist_name} (lrclib)")
                reference = forced_lrc_candidate.to_lyrics_result()
            else:
                log("Fetching reference lyrics (LRCLIB, synced lyrics only)...")
                reference = fetch_reference_lyrics(artist, title, duration_sec=audio_duration)
                if reference is None:
                    reason = (f"No valid synced-lyrics candidate was found on LRCLIB for "
                              f"{artist!r} - {title!r}.")
                    log(f"  {reason}")
                    if opts.no_lrc_fallback_callback is not None:
                        if not opts.no_lrc_fallback_callback(reason):
                            return PipelineResult(success=False, error=(
                                f"Cancelled: {reason} User declined to continue with pure "
                                f"transcription."))
                    log("  Continuing with pure transcription (no reference-lyrics correction or "
                        "forced line breaks).")
            if reference:
                candidate_lines = parse_lyrics_lines(reference.plain_lyrics)
                # Force-align known gaps (see recover_dropped_reference_words's own
                # docstring). Gated on the SAME wrong-song floor as everything else
                # in this block (reference_match_ratio computed on the ORIGINAL
                # ASR words, before any recovery) -- running this against a
                # wrong-song/wrong-language reference would force garbage text
                # onto our own audio, not recover anything real.
                if reference_match_ratio(candidate_lines, words) >= config.REFERENCE_LYRICS_MIN_MATCH_RATIO:
                    words, n_recovered = _recover_dropped_reference_words_cancellable(
                        candidate_lines, words, vocals_path, opts, debug_log, log)
                    if n_recovered:
                        dlog(f"  Force-alignment of known gaps: recovered {n_recovered} word(s) from the "
                             f"reference lyrics that ASR produced no word for at all, via a real wav2vec2 "
                             f"CTC forced alignment.")
                match_ratio = reference_match_ratio(candidate_lines, words)
                unmatched_run = largest_unmatched_reference_run(candidate_lines, words)
                # PROTOTYPE ASR-quality retry, see config.RETRY_LOW_QUALITY_ASR's
                # docstring. Only fires once this reference has already cleared
                # REFERENCE_LYRICS_MIN_MATCH_RATIO's much lower "is this even the
                # right song" bar -- below that bar means wrong song/language,
                # which a bigger model can't fix. Two INDEPENDENT triggers above
                # that bar: a low whole-song ratio (ASR mistranscribed a lot,
                # spread thinly), or a long unmatched-reference run (real case:
                # Trixie Mattel - Gold -- ASR completely dropped one repeat of
                # the "Doo doo doo doo doo. They start to play." chorus while the
                # rest of the 326-word transcript was fine, which kept the
                # aggregate ratio well above the bar and would have hidden this
                # from a whole-song-only check).
                low_ratio_trigger = (config.REFERENCE_LYRICS_MIN_MATCH_RATIO <= match_ratio
                                      < config.RETRY_ASR_MIN_REFERENCE_MATCH_RATIO)
                dropped_passage_trigger = (match_ratio >= config.REFERENCE_LYRICS_MIN_MATCH_RATIO
                                            and unmatched_run >= config.RETRY_ASR_MIN_UNMATCHED_REFERENCE_RUN)
                # PROTOTYPE ASR-quality retry gating (2026-08-10, user's explicit request): a whole-song
                # re-transcription roughly doubles this run's ASR time -- a reasonable cost to pay
                # automatically in an unattended --batch run, but not one to silently eat in a single-song
                # run where the user is presumably watching. Outside --batch, log a WARNING with the same
                # explanation instead of retrying.
                standard_retry_triggered = (opts.retry_low_quality_asr and (low_ratio_trigger or dropped_passage_trigger)
                                             and current_asr_model != config.RETRY_ASR_MODEL)
                if standard_retry_triggered and not opts.batch:
                    if dropped_passage_trigger and not low_ratio_trigger:
                        dlog(f"  WARNING: ASR transcript matches the reference lyrics overall ({match_ratio:.0%}), "
                             f"but a {unmatched_run}-word run of reference text has NO corresponding ASR word at "
                             f"all (a passage ASR completely dropped) -- retrying with '{config.RETRY_ASR_MODEL}' "
                             f"would likely help, but automatic retries only run in --batch mode. Re-run with "
                             f"--batch, or pass --whisper-model {config.RETRY_ASR_MODEL} directly, to get this "
                             f"fixed.")
                    else:
                        dlog(f"  WARNING: ASR transcript only matches the reference lyrics {match_ratio:.0%} "
                             f"(right song, but ASR quality looks weak) -- retrying with "
                             f"'{config.RETRY_ASR_MODEL}' would likely help, but automatic retries only run in "
                             f"--batch mode. Re-run with --batch, or pass --whisper-model "
                             f"{config.RETRY_ASR_MODEL} directly, to get this fixed.")
                elif standard_retry_triggered:
                    debug_log.section(f"ASR QUALITY RETRY (standard fallback path) -- re-transcribing with "
                                       f"'{config.RETRY_ASR_MODEL}'")
                    if dropped_passage_trigger and not low_ratio_trigger:
                        dlog(f"  ASR transcript matches the reference lyrics overall ({match_ratio:.0%}), but a "
                             f"{unmatched_run}-word run of reference text has NO corresponding ASR word at all "
                             f"(a passage ASR completely dropped) -- retrying before accepting this transcription.")
                    else:
                        dlog(f"  ASR transcript only matches the reference lyrics {match_ratio:.0%} (right song, "
                             f"but ASR quality looks weak) -- retrying before accepting this transcription.")
                    retry_words = _retry_transcribe(vocals_path, opts, debug_log, log)
                    if retry_words:
                        # Apply the SAME gap-recovery to the retry's own transcription before comparing --
                        # otherwise accepting the retry (its raw ratio can look better on the WHOLE song)
                        # silently throws away whatever the first pass's gap-recovery already found, and
                        # there's no guarantee the bigger model independently recovers the same passage.
                        if (reference_match_ratio(candidate_lines, retry_words)
                                >= config.REFERENCE_LYRICS_MIN_MATCH_RATIO):
                            retry_words, retry_n_recovered = _recover_dropped_reference_words_cancellable(
                                candidate_lines, retry_words, vocals_path, opts, debug_log, log)
                            if retry_n_recovered:
                                dlog(f"  Force-alignment of known gaps (retry transcription): recovered "
                                     f"{retry_n_recovered} more word(s).")
                        retry_ratio = reference_match_ratio(candidate_lines, retry_words)
                        retry_unmatched_run = largest_unmatched_reference_run(candidate_lines, retry_words)
                        # Accept if it's better on WHICHEVER signal actually triggered the retry -- comparing
                        # only the ratio could miss a retry that fixes one dropped passage (a handful of
                        # words out of hundreds barely moves the aggregate) without making it worse either.
                        if retry_ratio > match_ratio or retry_unmatched_run < unmatched_run:
                            dlog(f"  Retry with '{config.RETRY_ASR_MODEL}' improved: reference match "
                                 f"{match_ratio:.0%} -> {retry_ratio:.0%}, largest dropped-passage run "
                                 f"{unmatched_run} -> {retry_unmatched_run} word(s) -- using the retry.")
                            words = retry_words
                            match_ratio = retry_ratio
                            unmatched_run = retry_unmatched_run
                            current_asr_model = config.RETRY_ASR_MODEL
                        else:
                            dlog(f"  Retry with '{config.RETRY_ASR_MODEL}' did not improve either signal -- "
                                 f"keeping the original transcription.")
                    else:
                        dlog(f"  Retry transcription with '{config.RETRY_ASR_MODEL}' produced no words -- "
                             f"keeping the original transcription.")
                if match_ratio < config.REFERENCE_LYRICS_MIN_MATCH_RATIO:
                    log(f"  Got a reference from {reference.source}, but its words barely overlap "
                        f"the transcript (likely wrong song/language) -- discarding it, "
                        f"continuing with ASR text and gap-based phrasing only.")
                else:
                    ref_lines = candidate_lines
                    synced_lyrics_text = reference.synced_lyrics
                    log(f"  Got {len(ref_lines)} reference line(s) from {reference.source}"
                        f"{' (synced)' if reference.synced_lyrics else ''}.")
                    debug_log.log_lyrics_selection(
                        source=reference.source, track_name=reference.track_name,
                        artist_name=reference.artist_name, lrclib_id=reference.lrclib_id,
                        duration=reference.duration, synced=bool(reference.synced_lyrics),
                        extra=f"{len(ref_lines)} reference line(s), match ratio {match_ratio:.0%}",
                    )
                    corrected = align_words_to_reference(
                        words, ref_lines,
                        synced_lyrics_text=synced_lyrics_text,
                    )
                    if synced_lyrics_text:
                        strict_line_breaks_from_lrc = is_lrc_line_tracking_confident(words, synced_lyrics_text)
                        if strict_line_breaks_from_lrc:
                            log("  Confidently trusted LRC line tracking -- line breaks will match it "
                                "exactly (no gap/length heuristics).")
                    diffs = alignment_diff_summary(words, corrected)
                    # corrected is always the same length as words now --
                    # a dropped word is flagged via Word.dropped, not
                    # omitted (see its own docstring for why).
                    n_dropped = sum(1 for w in corrected if w.dropped)
                    if diffs:
                        log(f"  Corrected/dropped {len(diffs)} word(s) against the reference lyrics"
                            f"{f' ({n_dropped} dropped, no reference match at all)' if n_dropped else ''}:")
                        for d in diffs[:20]:
                            log(f"    {d}")
                        if len(diffs) > 20:
                            log(f"    ... and {len(diffs) - 20} more")
                    else:
                        log("  ASR text already matched the reference; no corrections needed.")
                    debug_log.log_reference_corrections(diffs)
                    words = corrected
            # else: reference is None -- already logged (and, in the GUI,
            # confirmed via no_lrc_fallback_callback) right after the fetch
            # above; nothing further to do here but continue with ASR text
            # and gap-based phrasing, same as if lyric lookup were disabled.
        else:
            log("Lyric lookup disabled (--no-fetch-lyrics); using ASR text and gap-based phrasing only.")

        # --- PASS 3: fit words onto the pass-1 note grid (timing untouched) --
        log("Pass 3: fitting words onto the pass-1 note grid...")
        syllables, stats = align_words(words, notes, y, sr, debug_log=debug_log)
        log(f"  {stats.words_with_notes}/{stats.total_words} words matched to pass-1 notes directly "
            f"({stats.total_notes_consumed} notes consumed); "
            f"{stats.words_with_fallback} word(s) needed a fallback note (no pass-1 note in their zone).")
        if stats.fallback_words:
            shown = stats.fallback_words[:15]
            log(f"    fallback words: {', '.join(shown)}" + (" ..." if len(stats.fallback_words) > 15 else ""))
            log(f"    (pitch source: {stats.fallback_used_neighbor} borrowed from the nearest pass-1 note, "
                f"{stats.fallback_used_fresh_analysis} needed a fresh isolated re-analysis because no "
                f"pass-1 notes existed at all -- the latter is the less reliable case)")
        if stats.words_with_melisma or stats.words_with_syllable_merge:
            log(f"    {stats.words_with_melisma} word(s) had melisma (fewer syllables than notes), "
                f"{stats.words_with_syllable_merge} word(s) had syllables merged (more syllables than notes)")
        if stats.lines_word_boundary_split:
            log(f"    {stats.lines_word_boundary_split} matched reference line(s) "
                f"({stats.words_in_word_boundary_split_lines} words) had their notes split by "
                f"each word's own ASR start/end time")
        if stats.verification_results:
            n_replaced = len(stats.verification_results)
            log(f"    reference-text override: forced {n_replaced} word(s) to match reference "
                f"lyrics exactly")

        # --- PASS 4 (optional): confirm/correct pitch class against MusicXML reference file(s).
        if mxl_paths:
            log(f"Pass 4: cross-checking pitch against {len(mxl_paths)} MusicXML reference file(s)...")
            syllables, mxl_stats_list = apply_musicxml_references(
                syllables, mxl_paths, preferred_part_name=opts.musicxml_part,
                verbose=not opts.quiet, debug_log=debug_log,
            )
            for path, mxl_stats in zip(mxl_paths, mxl_stats_list):
                label = Path(path).name
                if mxl_stats.skipped_reason:
                    log(f"  {label}: skipped -- {mxl_stats.skipped_reason}")
                else:
                    log(f"  {label}: parts used: {mxl_stats.part_names_used}, {mxl_stats.n_matched}/"
                        f"{mxl_stats.n_comparable_syllables} syllables matched by lyric text, "
                        f"calibration offset {mxl_stats.calibration_offset:+d} semitones "
                        f"({mxl_stats.calibration_confidence:.0%} agreement), "
                        f"{len(mxl_stats.corrections)} syllable(s) corrected")

    # --- 7c. LRC line-timing check (optional, off by default): flags lines whose
    # assigned start disagrees with LRCLIB's synced-lyrics timing. DIAGNOSTIC
    # ONLY -- never moves anything, see lrc_timing.py's module docstring for why.
    if opts.lrc_timing_check and synced_lyrics_text:
        log("Checking line timing against LRCLIB's synced lyrics (diagnostic only)...")
        lrc_stats = apply_lrc_timing_check(syllables, synced_lyrics_text,
                                            verbose=not opts.quiet, debug_log=debug_log)
        if lrc_stats.skipped_reason:
            log(f"  skipped -- {lrc_stats.skipped_reason}")
        else:
            log(f"  calibration offset {lrc_stats.calibration_offset_sec:+.1f}s "
                f"({lrc_stats.calibration_confidence:.0%} agreement), "
                f"{len(lrc_stats.flags)}/{lrc_stats.n_matched_lines} line(s) flagged")

    entries = build_lines(syllables, strict_reference_lines=strict_line_breaks_from_lrc)

    # --- 8. GAP = start of the first syllable --------------------------------
    first_syllable = next((e for e in entries if isinstance(e, Syllable)), None)
    gap_ms = int(round(first_syllable.start * 1000)) if first_syllable else 0

    # --- 9. VIDEOGAP ----------------------------------------------------
    # Skipped entirely (not just "no video") when the video and analysis
    # audio are the same file, or one was extracted directly from the
    # other (mp4-as-audio / avi-extract) -- correlating a signal against
    # itself or a trimmed copy of itself is meaningless, not just wasted
    # work.
    videogap = None
    if resolved.output_video_source and resolved.videogap_applicable and not opts.no_video_sync:
        log("Estimating VIDEOGAP from the video's audio track...")
        videogap = estimate_videogap(resolved.output_video_source, audio_path)
        if videogap is not None:
            log(f"Estimated VIDEOGAP: {videogap}s")
        else:
            log("Video has no usable audio track (or ffmpeg unavailable); leaving VIDEOGAP unset.")

    # --- 10. Preview start: default to first vocal, nudged back slightly ----
    preview_start = max(0.0, (first_syllable.start - 0.5)) if first_syllable else None

    # --- 11. Copy the companions this output actually references into
    # output_dir (input and output folders are required to differ -- see
    # the check at the top of this function), renamed to "<Artist> -
    # <Title>[.ext]" (images keep their own "[CO]"/"[BG]" tag) regardless
    # of what they were called in the input folder -----------------------
    staged = stage_companions_to_output(
        output_dir, artist, title,
        mp3_src=resolved.output_mp3_source,
        video_src=resolved.output_video_source,
        cover_src=resolved.cover,
        background_src=resolved.background,
    )

    # --- 12. Assemble + write Song. Companion staging above already ran --
    # an output_dir != input_dir still needs to be self-contained.
    out_name = sanitize_filename(f"{artist} - {title}.txt")
    out_path = output_dir / out_name

    song = Song(
        title=title,
        artist=artist,
        language=config.DEFAULT_LANGUAGE,
        mp3=staged.mp3,
        cover=staged.cover,
        background=staged.background,
        video=staged.video,
        videogap=videogap,
        bpm=write_bpm,
        gap_ms=gap_ms,
        preview_start=preview_start,
        entries=entries,
    )
    write_song(song, out_path, merge_connected_melisma=True)
    log(f"Wrote {out_path}")

    if not opts.skip_separation:
        if opts.delete_work_files:
            log(f"(Work files in {work_dir} will be deleted now that generation is complete.)")
        else:
            log(f"(Work files kept in {work_dir}; delete it to reclaim disk space.)")

    debug_log.close()
    return PipelineResult(success=True, output_txt_path=out_path)


def _opts_from_args(args: argparse.Namespace) -> config.PipelineOptions:
    """Builds a PipelineOptions from argparse's Namespace -- the CLI's own
    way of constructing the same options object gui.py builds directly
    from widget state."""
    pinned_lyrics = None
    if args.lrc_file:
        pinned_lyrics = load_lrc_file(args.lrc_file, artist=args.artist or "", title=args.title or "")
        if pinned_lyrics is None:
            print(f"Could not read/parse --lrc-file {args.lrc_file!r} -- ignoring.")
    return config.PipelineOptions(
        pinned_lyrics=pinned_lyrics,
        artist=args.artist, title=args.title, audio_file=args.audio_file, work_dir=args.work_dir,
        whisper_model=args.whisper_model,
        demucs_model=args.demucs_model, bpm_override=args.bpm,
        skip_separation=args.skip_separation, vocals_path=args.vocals_path,
        fetch_lyrics=args.fetch_lyrics, fetch_cover=args.fetch_cover, no_video_sync=args.no_video_sync,
        no_whisperx=args.no_whisperx, no_transcribe=args.no_transcribe, whisperx_no_vad=args.whisperx_no_vad,
        retry_low_quality_asr=args.retry_low_quality_asr,
        musicxml_reference=args.musicxml_reference, musicxml_part=args.musicxml_part,
        lrc_timing_check=args.lrc_timing_check,
        pitch_smooth_window=args.pitch_smooth_window, note_split_semitones=args.note_split_semitones,
        min_note_beat_fraction=args.min_note_beat_fraction, silence_threshold_db=args.silence_threshold_db,
        silence_floor_db=args.silence_floor_db, spike_max_duration=args.spike_max_duration,
        spike_jump_semitones=args.spike_jump_semitones,
        ambiguity_key_tiebreak=args.ambiguity_key_tiebreak,
        ambiguity_margin_threshold=args.ambiguity_margin_threshold,
        pitch_source=args.pitch_source,
        no_pass1_debug=args.no_pass1_debug,
        no_debug_log=args.no_debug_log, quiet=args.quiet,
        youtube_url=args.youtube_url, youtube_audio_only=args.youtube_audio_only,
        delete_work_files=args.delete_work_files,
        mxl_lrc_primary=args.mxl_lrc_primary, lrclib_id=args.lrclib_id,
        batch=args.batch,
    )


def run(argv=None) -> int:
    # Python fully block-buffers stdout by default whenever it isn't a live
    # terminal (redirected to a file, piped through `| grep`, etc.) -- on a
    # long run this can leave progress output invisible for the ENTIRE run,
    # only appearing all at once when the process exits or the buffer fills.
    # Force line buffering so every print (including the progress lines
    # below and in verification.py) shows up promptly regardless of how
    # stdout is connected.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass  # not a reconfigurable text stream -- nothing to do

    args = build_arg_parser().parse_args(argv)

    if args.batch:
        incompatible = []
        if args.artist or args.title:
            incompatible.append("--artist/--title")
        if args.youtube_url:
            incompatible.append("--youtube-url")
        if args.work_dir:
            incompatible.append("--work-dir")
        if args.lrclib_id:
            incompatible.append("--lrclib-id")
        if args.lrc_file:
            incompatible.append("--lrc-file")
        if args.audio_file:
            incompatible.append("--audio-file")
        if incompatible:
            print(f"--batch is not allowed together with {', '.join(incompatible)} "
                  f"(a single override doesn't make sense across multiple songs).", file=sys.stderr)
            return 1

    cuda_error = check_cuda_available()
    if cuda_error:
        print(cuda_error, file=sys.stderr)
        return 1

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    opts = _opts_from_args(args)

    if args.batch:
        from .batch import run_batch  # local import: batch.py imports FROM this module
        results = run_batch(input_dir, output_dir, opts)
        return 0 if all(r.success for _, r in results) else 2

    result = run_pipeline(input_dir, output_dir, opts)
    if not result.success:
        print(result.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
