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
# algorithms(True) to have full effect on cuBLAS ops -- see
# note_detection.py's _crepe_pitch, which scopes deterministic algorithm
# selection to CREPE's own inference (confirmed non-deterministic
# run-to-run on identical audio otherwise).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import config
from .models import Song, Syllable
from .file_discovery import resolve_artist_title, AmbiguousInputError, NoAudioSourceFoundError
from .song_input import resolve_song_folder
from .output_staging import stage_companions_to_output
from .usdx_parser import parse_usdx_file, UsdxParseError
from .verify_existing_song import verify_existing_song
from .youtube_source import download_youtube_source, YoutubeDownloadError
from .separation import isolate_vocals, SeparationError
from .transcription import transcribe_words
from .tempo import detect_bpm
from .note_detection import detect_notes
from .alignment import align_words
from .phrasing import build_lines
from .lyrics_lookup import (fetch_reference_lyrics, parse_lyrics_lines, align_words_to_reference,
                             alignment_diff_summary, reference_matches_transcript, fetch_lrclib_by_id)
from .musicxml_reference import apply_musicxml_references
from .mxl_lrc_generator import try_mxl_lrc_primary
from .lrc_timing import apply_lrc_timing_check
from .video_sync import estimate_videogap
from .usdx_writer import write_song
from .debug_output import write_pass1_debug_file
from .debug_log import DebugLog


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
    regenerated: bool = True  # False only when existing-txt verification passed (Phase B)


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
                         "(mp3/ogg/oga), which this tool otherwise refuses to guess between.")
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
    p.add_argument("--verify-whisper-model", default=config.DEFAULT_WHISPER_MODEL,
                    help=f"whisper model name for verify_words/verify_placement's re-transcription re-checks "
                         f"(default: {config.DEFAULT_WHISPER_MODEL}, independent of --whisper-model). These loops "
                         "call the model hundreds of times on tiny clips, where a big model's fixed per-call "
                         "overhead dominates -- --whisper-model large-v3 for both made the verify passes alone "
                         "take ~10x longer than a small model in one real run.")
    p.add_argument("--demucs-model", default=config.DEFAULT_DEMUCS_MODEL,
                    help=f"Demucs model name (default: {config.DEFAULT_DEMUCS_MODEL})")
    p.add_argument("--bpm", type=float, default=None, help="Override detected BPM")
    p.add_argument("--skip-separation", action="store_true",
                    help="Skip Demucs; use --vocals-path instead (e.g. you already have an isolated vocal stem)")
    p.add_argument("--vocals-path", help="Path to a pre-isolated vocal stem (wav), used with --skip-separation")
    p.add_argument("--fetch-lyrics", dest="fetch_lyrics", action="store_true", default=True,
                    help="Fetch reference lyrics online (lyrics.ovh) to correct low-confidence ASR words (default: on)")
    p.add_argument("--no-fetch-lyrics", dest="fetch_lyrics", action="store_false",
                    help="Disable online lyric lookup/correction")
    p.add_argument("--no-video-sync", action="store_true",
                    help="Don't attempt to auto-compute #VIDEOGAP from the video's own audio track")
    p.add_argument("--no-whisperx", action="store_true",
                    help="Don't use whisperx forced alignment for word timing even if installed "
                         "(uses faster-whisper's own, less precise, timestamps instead)")
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
    p.add_argument("--verify-words", dest="verify_words", action="store_true",
                    default=config.ENABLE_WORD_VERIFICATION,
                    help="Re-transcribe a tight, isolated audio crop around every word's own ASR "
                         "timestamp and cross-check it against the reference lyrics (default: on). "
                         "Never changes note timing/pitch, only swaps in text, and only when the "
                         "recheck actively confirms a different answer than what's already there.")
    p.add_argument("--no-verify-words", dest="verify_words", action="store_false",
                    help="Disable the text re-transcription check")
    p.add_argument("--verify-placement", dest="verify_placement", action="store_true",
                    default=config.ENABLE_PLACEMENT_VERIFICATION,
                    help="Crop a small window at every word's FINAL note-assigned position, "
                         "transcribe it, and expand the window until the expected word is found "
                         "(or give up) -- flags (never auto-corrects) cases where the word turns out "
                         "to be somewhere else, catching pass 3 putting a correctly-transcribed word "
                         "on the wrong notes (default: OFF -- detection-only and never wrong on what "
                         "it flagged, but the actual bugs it was catching trace back further upstream "
                         "to bad WhisperX word timestamps, and the check itself is an expensive "
                         "expand-search re-transcription loop over every word).")
    p.add_argument("--no-verify-placement", dest="verify_placement", action="store_false",
                    help="Explicitly disable the placement check (already off by default)")
    p.add_argument("--verify-suspicious-only", dest="verify_all_words", action="store_false",
                    default=config.VERIFY_ALL_WORDS,
                    help="Only run enabled verification checks on words pass 3 flagged suspicious "
                         "(fallback words that got zero note pieces) instead of every "
                         "word -- faster, at the cost of catching fewer mistakes.")
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
                         "audio-only pass 1-4 pipeline -- validated real end-to-end: 100% pitch-class "
                         "accuracy, 99% timing within 500ms (see CLAUDE.md). Quality-gated: falls back "
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
    p.add_argument("--no-musicxml-force-calibration", dest="musicxml_force_calibration",
                    action="store_false", default=config.ENABLE_MUSICXML_FORCE_CALIBRATION,
                    help="Without a confident calibration offset (config."
                         "MUSICXML_MIN_CALIBRATION_CONFIDENCE), pass 4 normally still applies the best "
                         "available offset anyway rather than skipping the file -- validated real "
                         "end-to-end on all 7 MXL-having songs in the test set: 0 regressions, up to "
                         "+21.6pp on songs where pass 1 was confirmed unreliable (see CLAUDE.md). Pass "
                         "this flag to go back to skipping uncalibratable files instead. Never touches "
                         "octave or timing, same as normal pass 4.")
    p.add_argument("--lrc-timing-check", dest="lrc_timing_check", action="store_true",
                    default=config.ENABLE_LRC_TIMING_CHECK,
                    help="EXPERIMENTAL, off by default. Cross-checks each line's assigned start time "
                         "against LRCLIB's synced lyrics (when available), once a per-song time "
                         "calibration offset can be trusted. DIAGNOSTIC ONLY -- flags disagreeing lines "
                         "in the console/debug log, never moves anything (see lrc_timing.py).")
    p.add_argument("--no-lrc-timing-check", dest="lrc_timing_check", action="store_false",
                    help="Disable the LRC timing check (no-op unless --lrc-timing-check or "
                         "config.ENABLE_LRC_TIMING_CHECK enabled it).")
    p.add_argument("--zone-boundary-snap", dest="zone_boundary_snap", action="store_true",
                    default=config.ENABLE_ZONE_BOUNDARY_SNAP,
                    help="EXPERIMENTAL, off by default. Refines pass 3's zone/word boundaries (which "
                         "are computed purely from ASR-timestamp midpoints) by snapping to a nearby "
                         "pass-1 note onset when exactly one exists within "
                         f"{config.ZONE_BOUNDARY_SNAP_RADIUS_SEC}s -- targets cases where an "
                         "imprecise ASR timestamp places the boundary near, but not exactly at, "
                         "where the audio actually starts a new note. Never adds/removes/moves a "
                         "note, only chooses a different cut point. Not yet validated end-to-end.")
    p.add_argument("--no-zone-boundary-snap", dest="zone_boundary_snap", action="store_false",
                    help="Explicitly disable zone-boundary snapping (already off by default).")
    p.add_argument("--existing-txt", dest="existing_txt_path", default=None,
                    help="Path to an existing UltraStar .txt for this song -- if given, this always wins "
                         "over auto-detection (no filename-matching required, same convention as "
                         "--musicxml-reference). Its own pitch/timing is compared against a fresh "
                         "pipeline run; a NEW file is only written if real problems are found.")
    p.add_argument("--existing-txt-check", dest="existing_txt_check", action="store_true",
                    default=config.ENABLE_EXISTING_TXT_CHECK,
                    help="Auto-detect an existing '<Artist> - <Title>.txt' already sitting in the input "
                         "folder and verify it the same way --existing-txt does (default: OFF -- unlike "
                         "this project's other on-by-default features, this one can result in NOT "
                         "writing output you expected on a plain re-run).")
    p.add_argument("--no-existing-txt-check", dest="existing_txt_check", action="store_false",
                    help="Explicitly disable existing-file auto-detection (already off by default).")
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
    p.add_argument("--pitch-source", default=config.DEFAULT_PITCH_SOURCE, choices=["rmvpe", "ensemble"],
                    help="Which pass-1 pitch source(s) to use (default: "
                         f"{config.DEFAULT_PITCH_SOURCE!r}). 'rmvpe': RMVPE alone, its own voicing "
                         "decision, no cross-check with any other source -- validated 2026-08-09 as "
                         "a real, reproducible +1.7pp average improvement over the old ensemble "
                         "default, and faster (no CREPE inference, no cross-check math). 'ensemble': "
                         "the original pyin-primary + CREPE/RMVPE-cross-check architecture -- "
                         "--no-crepe/--crepe-model only have any effect in this mode.")
    p.add_argument("--no-crepe", dest="use_crepe", action="store_false", default=config.ENABLE_CREPE,
                    help="(--pitch-source ensemble only) Don't cross-check pYIN against CREPE "
                         "(torchcrepe) per-frame (default: cross-check is on). Where they agree, "
                         "CREPE's pitch is used (generally more robust against accompaniment bleed); "
                         "where they disagree, pYIN's pitch is kept but downweighted rather than "
                         "discarded.")
    p.add_argument("--crepe-model", default=config.DEFAULT_CREPE_MODEL, choices=["full", "tiny"],
                    help=f"(--pitch-source ensemble only) torchcrepe model size -- 'full' is more "
                         f"accurate, 'tiny' is much faster (default: {config.DEFAULT_CREPE_MODEL})")
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
                f'Could not parse "<Artist> - <Title>" from the audio filename '
                f"({resolved.output_mp3_source.name!r}) or the input folder name "
                f"({input_dir.name!r}). Pass --artist and --title explicitly instead."))
        artist = opts.artist or parsed_artist
        title = opts.title or parsed_title

    # --output-dir is now the PARENT folder under which a "<Artist> -
    # <Title>" folder is created (e.g. --output-dir C:\output produces
    # C:\output\<Artist> - <Title>\) -- not the final folder itself.
    # Optional; defaults to <input folder>\Output as that parent.
    if output_dir is None:
        output_dir = input_dir / "Output"
    output_dir = output_dir / f"{artist} - {title}"

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

    # --- Existing-file verification (feature 6, OFF by default) -- an
    # explicit --existing-txt always wins; otherwise auto-detected only if
    # opts.existing_txt_check is on. Detected here (needs artist/title to
    # know the expected filename); actually PARSED and COMPARED much
    # later, once a fresh syllable sequence exists to compare against.
    existing_txt_path: Optional[Path] = None
    if opts.existing_txt_path:
        existing_txt_path = Path(opts.existing_txt_path)
    elif opts.existing_txt_check:
        candidate = input_dir / f"{artist} - {title}.txt"
        if candidate.is_file():
            existing_txt_path = candidate

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

    debug_log_path = None if opts.no_debug_log else (work_dir / f"{artist} - {title} [DEBUG LOG].txt")
    debug_log = DebugLog(debug_log_path)
    if debug_log_path is not None:
        log(f"Writing debug log to: {debug_log_path}")

    # --- 1. Companion files (already resolved above) -------------------------
    for note in resolved.notes:
        log(note)
    if resolved.output_video_source:
        log(f"Found video: {resolved.output_video_source.name}")
    if resolved.cover:
        log(f"Found cover: {resolved.cover.name}")
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
    if opts.skip_separation:
        if not opts.vocals_path:
            return PipelineResult(success=False, error="--skip-separation requires --vocals-path")
        vocals_path = Path(opts.vocals_path).resolve()
    else:
        log("Isolating vocals with Demucs...")
        try:
            vocals_path = isolate_vocals(audio_path, work_dir, model=opts.demucs_model)
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
    log(f"Transcribing with {'whisperx' if not opts.no_whisperx else 'faster-whisper'} ({opts.whisper_model})"
        f"{' [VAD near-disabled]' if opts.whisperx_no_vad else ''}...")
    words = transcribe_words(
        vocals_path, opts.whisper_model, prefer_whisperx=not opts.no_whisperx, debug_log=debug_log,
        whisperx_vad_options=config.WHISPERX_NO_VAD_OPTIONS if opts.whisperx_no_vad else None,
    )
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
    if opts.mxl_lrc_primary and mxl_paths:
        log("Attempting MXL+LRC primary generation (MusicXML pitch + synced-lyric line anchors + ASR word placement)...")
        mxl_lrc_result = try_mxl_lrc_primary(
            mxl_paths, artist, title, audio_duration, words,
            forced_candidate=forced_lrc_candidate, preferred_part_name=opts.musicxml_part,
        )
        if mxl_lrc_result is not None and mxl_lrc_result.time_calibration is not None:
            tc = mxl_lrc_result.time_calibration
            if tc.offset_sec is not None:
                drift_desc = f", drift {tc.slope:+.4f}s/LRC-s" if tc.kind == "drift" else ""
                log(f"  LRC/audio time calibration ({tc.kind}): offset {tc.offset_sec:+.1f}s{drift_desc} "
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
        log("Pass 1: detecting notes from audio (pitch + timing only)...")
        notes = detect_notes(
            y, sr, bpm=bpm,
            isolation_source="rmvpe" if opts.pitch_source == "rmvpe" else None,
            smooth_window_sec=opts.pitch_smooth_window,
            pitch_jump_semitones=opts.note_split_semitones,
            min_note_beats_fraction=opts.min_note_beat_fraction,
            silence_threshold_db=opts.silence_threshold_db,
            silence_absolute_floor_db=opts.silence_floor_db,
            spike_max_duration_sec=opts.spike_max_duration,
            spike_min_jump_semitones=opts.spike_jump_semitones,
            use_crepe=opts.use_crepe,
            crepe_model=opts.crepe_model,
            verbose=not opts.quiet,
            debug_log=debug_log,
        )
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
                log("Fetching reference lyrics (LRCLIB, falling back to lyrics.ovh)...")
                reference = fetch_reference_lyrics(
                    artist, title, duration_sec=audio_duration,
                    on_ambiguous=opts.lyrics_disambiguation_callback if opts.lyrics_ambiguity_prompt else None,
                )
            if reference:
                candidate_lines = parse_lyrics_lines(reference.plain_lyrics)
                if not reference_matches_transcript(candidate_lines, words):
                    log(f"  Got a reference from {reference.source}, but its words barely overlap "
                        f"the transcript (likely wrong song/language) -- discarding it, "
                        f"continuing with ASR text and gap-based phrasing only.")
                else:
                    ref_lines = candidate_lines
                    synced_lyrics_text = reference.synced_lyrics
                    log(f"  Got {len(ref_lines)} reference line(s) from {reference.source}"
                        f"{' (synced)' if reference.synced_lyrics else ''}.")
                    corrected = align_words_to_reference(words, ref_lines)
                    diffs = alignment_diff_summary(words, corrected)
                    if diffs:
                        log(f"  Corrected {len(diffs)} word(s) against the reference lyrics:")
                        for d in diffs[:20]:
                            log(f"    {d}")
                        if len(diffs) > 20:
                            log(f"    ... and {len(diffs) - 20} more")
                    else:
                        log("  ASR text already matched the reference; no corrections needed.")
                    debug_log.log_reference_corrections(diffs)
                    words = corrected
            else:
                log("  Could not fetch reference lyrics (not found on LRCLIB or lyrics.ovh, or no "
                    "network); continuing with ASR text and gap-based phrasing only.")
        else:
            log("Lyric lookup disabled (--no-fetch-lyrics); using ASR text and gap-based phrasing only.")

        # --- PASS 3: fit words onto the pass-1 note grid (timing untouched) --
        log("Pass 3: fitting words onto the pass-1 note grid...")
        syllables, stats = align_words(words, notes, y, sr,
                                        verify_words=opts.verify_words, verify_placement=opts.verify_placement,
                                        verify_all_words=opts.verify_all_words,
                                        verify_whisper_model=opts.verify_whisper_model,
                                        snap_boundaries=opts.zone_boundary_snap, debug_log=debug_log,
                                        verbose=not opts.quiet)
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
            n_checked = len(stats.verification_results)
            n_replaced = sum(1 for r in stats.verification_results if r.replaced)
            log(f"    verification: re-transcribed {n_checked} suspicious word(s) in isolation, "
                f"replaced {n_replaced}")
        if stats.placement_corrections:
            log(f"    placement check: corrected {len(stats.placement_corrections)} word(s) whose FINAL "
                f"note-assigned position didn't match what's actually sung there (see [placement] lines "
                f"above) -- pass 3 was re-run with the fix applied")
        if stats.placement_warnings:
            log(f"    placement check: {len(stats.placement_warnings)} word(s) flagged -- the audio at "
                f"their FINAL note-assigned position doesn't say the expected word (see [placement] lines "
                f"above); these were NOT corrected automatically and are worth checking by hand")

        # --- PASS 4 (optional): confirm/correct pitch class against MusicXML reference file(s).
        if mxl_paths:
            log(f"Pass 4: cross-checking pitch against {len(mxl_paths)} MusicXML reference file(s)...")
            syllables, mxl_stats_list = apply_musicxml_references(
                syllables, mxl_paths, preferred_part_name=opts.musicxml_part,
                force_calibration=opts.musicxml_force_calibration,
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

    # --- 7d. Existing-file verification (feature 6, OFF by default): compares
    # the existing file's own pitch/timing against THIS fresh syllable
    # sequence, before deciding whether to overwrite it. Runs here (after
    # pass 3/4, before phrasing) since it needs a fully fresh syllable
    # sequence to compare against -- line grouping isn't needed for the
    # comparison itself. Never saves pipeline compute vs. a full
    # regeneration; its value is avoiding unnecessary file churn on an
    # already-correct file, not avoiding processing time.
    existing_verification = None
    if existing_txt_path is not None:
        log(f"Checking existing file against this fresh run: {existing_txt_path}")
        try:
            existing_song = parse_usdx_file(existing_txt_path)
            existing_verification = verify_existing_song(existing_song, syllables, verbose=not opts.quiet,
                                                           debug_log=debug_log)
            log(f"  verdict: {existing_verification.verdict}"
                + (f" -- {existing_verification.reason}" if existing_verification.reason else ""))
        except UsdxParseError as e:
            log(f"  Could not parse existing file ({e}) -- generating fresh.")

    entries = build_lines(syllables)

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
    # the check at the top of this function) -----------------------------
    staged = stage_companions_to_output(
        output_dir,
        mp3_src=resolved.output_mp3_source,
        video_src=resolved.output_video_source,
        cover_src=resolved.cover,
        background_src=resolved.background,
    )

    # --- 12. Assemble + write Song, UNLESS existing-file verification passed
    # (in which case the existing file is kept, byte-for-byte, instead of
    # being overwritten by a fresh regeneration it didn't actually disagree
    # with). Companion staging above already ran either way -- an
    # output_dir != input_dir still needs to be self-contained even when
    # verification passes.
    out_name = f"{artist} - {title}.txt"
    out_path = output_dir / out_name
    regenerated = True

    if existing_verification is not None and existing_verification.verdict == "PASS":
        shutil.copy2(existing_txt_path, out_path)
        log(f"Existing file verified OK -- kept as-is (copied to {out_path}, not regenerated).")
        regenerated = False
    else:
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
        write_song(song, out_path)
        log(f"Wrote {out_path}")

    if not opts.skip_separation:
        if opts.delete_work_files:
            log(f"(Work files in {work_dir} will be deleted now that generation is complete.)")
        else:
            log(f"(Work files kept in {work_dir}; delete it to reclaim disk space.)")

    debug_log.close()
    return PipelineResult(success=True, output_txt_path=out_path, regenerated=regenerated)


def _opts_from_args(args: argparse.Namespace) -> config.PipelineOptions:
    """Builds a PipelineOptions from argparse's Namespace -- the CLI's own
    way of constructing the same options object gui.py builds directly
    from widget state."""
    return config.PipelineOptions(
        artist=args.artist, title=args.title, audio_file=args.audio_file, work_dir=args.work_dir,
        whisper_model=args.whisper_model, verify_whisper_model=args.verify_whisper_model,
        demucs_model=args.demucs_model, bpm_override=args.bpm,
        skip_separation=args.skip_separation, vocals_path=args.vocals_path,
        fetch_lyrics=args.fetch_lyrics, no_video_sync=args.no_video_sync,
        no_whisperx=args.no_whisperx, whisperx_no_vad=args.whisperx_no_vad,
        verify_words=args.verify_words,
        verify_placement=args.verify_placement, verify_all_words=args.verify_all_words,
        musicxml_reference=args.musicxml_reference, musicxml_part=args.musicxml_part,
        musicxml_force_calibration=args.musicxml_force_calibration,
        lrc_timing_check=args.lrc_timing_check, zone_boundary_snap=args.zone_boundary_snap,
        pitch_smooth_window=args.pitch_smooth_window, note_split_semitones=args.note_split_semitones,
        min_note_beat_fraction=args.min_note_beat_fraction, silence_threshold_db=args.silence_threshold_db,
        silence_floor_db=args.silence_floor_db, spike_max_duration=args.spike_max_duration,
        spike_jump_semitones=args.spike_jump_semitones, pitch_source=args.pitch_source,
        use_crepe=args.use_crepe, crepe_model=args.crepe_model,
        no_pass1_debug=args.no_pass1_debug,
        no_debug_log=args.no_debug_log, quiet=args.quiet,
        existing_txt_check=args.existing_txt_check, existing_txt_path=args.existing_txt_path,
        youtube_url=args.youtube_url, youtube_audio_only=args.youtube_audio_only,
        delete_work_files=args.delete_work_files,
        mxl_lrc_primary=args.mxl_lrc_primary, lrclib_id=args.lrclib_id,
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
        if args.existing_txt_path:
            incompatible.append("--existing-txt")
        if args.youtube_url:
            incompatible.append("--youtube-url")
        if args.work_dir:
            incompatible.append("--work-dir")
        if args.lrclib_id:
            incompatible.append("--lrclib-id")
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
