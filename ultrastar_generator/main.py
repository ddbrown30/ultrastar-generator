"""CLI entry point.

Usage:
    python -m ultrastar_generator.main "C:\\Songs\\Bon Jovi - Its My Life.mp3"

See README.md for the full option list and setup instructions.
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
import sys
from pathlib import Path

from . import config
from .models import Song, Syllable
from .file_discovery import find_companions, parse_artist_title
from .separation import isolate_vocals, SeparationError
from .transcription import transcribe_words
from .tempo import detect_bpm
from .note_detection import detect_notes
from .alignment import align_words
from .key_correction import snap_to_key
from .phrasing import build_lines
from .lyrics_lookup import fetch_reference_lyrics, parse_lyrics_lines, align_words_to_reference, alignment_diff_summary
from .video_sync import estimate_videogap
from .usdx_writer import write_song
from .debug_output import write_pass1_debug_file, write_notes_debug_file
from .debug_log import DebugLog


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate an UltraStar Deluxe .txt song file from an mp3/ogg/oga file."
    )
    p.add_argument("audio", help="Path to the audio file, named '<Artist> - <Title>.<mp3|ogg|oga>'")
    p.add_argument("--artist", help="Override artist parsed from filename")
    p.add_argument("--title", help="Override title parsed from filename")
    p.add_argument("--output-dir", help="Where to write the .txt (default: same folder as audio)")
    p.add_argument("--work-dir", help="Scratch space for separated stems etc. (default: <audio-file's-directory>/.ultrastar_work "
                                        "-- deliberately tied to the audio, not --output-dir, so separation is reused across "
                                        "multiple runs of the same song even when writing output elsewhere)")
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
    p.add_argument("--key-correction", dest="key_correction", action="store_true",
                    default=config.ENABLE_KEY_CORRECTION,
                    help="Enable pass 2's musical-key pitch-snapping (default: OFF -- "
                         "root-caused as actively harmful on real material: a single "
                         "global detected key applied blindly will snap legitimate "
                         "out-of-scale notes, e.g. deliberate modal-mixture/borrowed "
                         "tones, to the wrong pitch. See '[PASS2 DEBUG]' if you enable "
                         "it, to see exactly what it changed.)")
    p.add_argument("--no-key-correction", dest="key_correction", action="store_false",
                    help="Disable pass 2 (musical-key pitch-snapping) entirely")
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
    p.add_argument("--no-crepe", dest="use_crepe", action="store_false", default=config.ENABLE_CREPE,
                    help="Don't cross-check pYIN against CREPE (torchcrepe) per-frame (default: "
                         "cross-check is on). Where they agree, CREPE's pitch is used (generally more "
                         "robust against accompaniment bleed); where they disagree, pYIN's pitch is "
                         "kept but downweighted rather than discarded.")
    p.add_argument("--crepe-model", default=config.DEFAULT_CREPE_MODEL, choices=["full", "tiny"],
                    help=f"torchcrepe model size -- 'full' is more accurate, 'tiny' is much faster "
                         f"(default: {config.DEFAULT_CREPE_MODEL})")
    p.add_argument("--no-pass1-debug", action="store_true",
                    help="Don't write the '[PASS1 DEBUG]' .txt (pass-1 notes only, no lyrics) "
                         "that's written by default alongside the real output -- load it in the "
                         "UltraStar editor to check pass 1's timing/pitch in isolation.")
    p.add_argument("--no-pass2-debug", action="store_true",
                    help="Don't write the '[PASS2 DEBUG]' .txt (pass-2 key-corrected notes only, no "
                         "lyrics) that's written by default alongside the real output whenever key "
                         "correction is enabled -- diff it against '[PASS1 DEBUG]' to see exactly which "
                         "notes key correction changed, and by how much.")
    p.add_argument("--no-debug-log", action="store_true",
                    help="Don't write the '[DEBUG LOG]' .txt that's written by default alongside the "
                         "real output -- records raw ASR word timing/confidence, reference-line "
                         "grouping, note-zone boundary math, and syllable-proportional split decisions, "
                         "for tracing a wrong final word/note position back to which pipeline stage "
                         "actually caused it.")
    p.add_argument("--quiet", action="store_true",
                    help="Suppress the verbose [pass1]/[pass2] diagnostic logging (still prints "
                         "the main pipeline stage messages)")
    return p


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

    import torch
    if not torch.cuda.is_available():
        print(
            "CUDA is not available, but this pipeline requires it (CPU support has "
            "been removed). Check your PyTorch install / GPU drivers.",
            file=sys.stderr,
        )
        return 1

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 1
    if audio_path.suffix.lower() not in config.AUDIO_EXTS:
        print(f"Unsupported audio extension {audio_path.suffix!r}; expected one of {config.AUDIO_EXTS}", file=sys.stderr)
        return 1

    if args.artist and args.title:
        artist, title = args.artist, args.title
    else:
        try:
            parsed_artist, parsed_title = parse_artist_title(audio_path)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        artist = args.artist or parsed_artist
        title = args.title or parsed_title

    output_dir = Path(args.output_dir).resolve() if args.output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Defaults to the AUDIO's own directory, not output_dir -- separation
    # (and everything cached under it, including which vocals.wav BPM
    # detection sees) should be reused across multiple runs of the same
    # song even when --output-dir differs (e.g. comparing --whisper-model
    # choices into separate output folders). Demucs isn't bit-reproducible
    # run to run (see CLAUDE.md's "Lessons learned" -- confirmed in
    # practice: two separations of the same song produced different-
    # checksum vocals.wav, which was enough to flip detected BPM between
    # 105.47 and 109.96), so reusing the SAME cached separation avoids
    # that instability entirely rather than trying to fix Demucs itself.
    work_dir = Path(args.work_dir).resolve() if args.work_dir else (audio_path.parent / ".ultrastar_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {artist} - {title} ==")

    debug_log_path = None if args.no_debug_log else (output_dir / f"{artist} - {title} [DEBUG LOG].txt")
    debug_log = DebugLog(debug_log_path)
    if debug_log_path is not None:
        print(f"Writing debug log to: {debug_log_path}")

    # --- 1. Companion files -------------------------------------------------
    companions = find_companions(audio_path)
    if companions.video:
        print(f"Found video: {companions.video.name}")
    if companions.cover:
        print(f"Found cover: {companions.cover.name}")
    if companions.background:
        print(f"Found background: {companions.background.name}")

    # --- 2. Vocal isolation --------------------------------------------------
    if args.skip_separation:
        if not args.vocals_path:
            print("--skip-separation requires --vocals-path", file=sys.stderr)
            return 1
        vocals_path = Path(args.vocals_path).resolve()
    else:
        print("Isolating vocals with Demucs...")
        try:
            vocals_path = isolate_vocals(audio_path, work_dir, model=args.demucs_model)
        except SeparationError as e:
            print(f"Vocal isolation failed: {e}", file=sys.stderr)
            return 1
    print(f"Vocals: {vocals_path}")

    # --- 3. Load vocal audio for pitch analysis + tempo detection -----------
    import librosa
    y, sr = librosa.load(str(vocals_path), sr=None, mono=True)

    bpm = args.bpm or detect_bpm(y, sr)
    print(f"BPM (as written to txt; UltraStar multiplies by 4): {bpm}")

    # --- 4. PASS 1: pitch/timing from audio alone, no lyrics involved -------
    print("Pass 1: detecting notes from audio (pitch + timing only)...")
    notes = detect_notes(
        y, sr, bpm=bpm,
        smooth_window_sec=args.pitch_smooth_window,
        pitch_jump_semitones=args.note_split_semitones,
        min_note_beats_fraction=args.min_note_beat_fraction,
        silence_threshold_db=args.silence_threshold_db,
        silence_absolute_floor_db=args.silence_floor_db,
        spike_max_duration_sec=args.spike_max_duration,
        spike_min_jump_semitones=args.spike_jump_semitones,
        use_crepe=args.use_crepe,
        crepe_model=args.crepe_model,
        verbose=not args.quiet,
        debug_log=debug_log,
    )
    if not notes:
        print("No notes were detected -- check the audio / vocal isolation quality.", file=sys.stderr)
        return 1

    if not args.no_pass1_debug:
        debug_path = write_pass1_debug_file(notes, artist, title, audio_path.name, bpm,
                                             gap_ms=int(round(notes[0].start * 1000)),
                                             output_dir=output_dir)
        print(f"Wrote pass-1 debug file (notes only, no lyrics): {debug_path}")
        print("  -> load this in the UltraStar editor to check timing/pitch BEFORE lyrics are involved.")

    # --- 4b. PASS 2: key correction (optional, on by default) -- notes only,
    # no lyrics exist yet at this point ---------------------------------------
    if args.key_correction:
        print("Pass 2: snapping out-of-key notes...")
        notes = snap_to_key(notes, debug_log=debug_log)
        debug_log.log_notes(notes, "pass 2, key-corrected")
        if not args.no_pass2_debug:
            debug_path = write_notes_debug_file(notes, artist, title, audio_path.name, bpm,
                                                 gap_ms=int(round(notes[0].start * 1000)),
                                                 output_dir=output_dir, label="PASS2 DEBUG")
            print(f"Wrote pass-2 debug file (key-corrected notes only, no lyrics): {debug_path}")
            print("  -> diff this against '[PASS1 DEBUG]' to see exactly which notes key correction changed.")

    # --- 5. Transcription (lyrics text + rough timing) -----------------------
    print(f"Transcribing with {'whisperx' if not args.no_whisperx else 'faster-whisper'} ({args.whisper_model})"
          f"{' [VAD near-disabled]' if args.whisperx_no_vad else ''}...")
    words = transcribe_words(
        vocals_path, args.whisper_model, prefer_whisperx=not args.no_whisperx, debug_log=debug_log,
        whisperx_vad_options=config.WHISPERX_NO_VAD_OPTIONS if args.whisperx_no_vad else None,
    )
    if not words:
        print("No words were transcribed -- check the audio / vocal isolation quality.", file=sys.stderr)
        return 1
    print(f"Transcribed {len(words)} words.")

    # --- 6. Reference lyrics: correct ASR text AND mark phrase/line breaks --
    ref_lines = None
    if args.fetch_lyrics:
        print("Fetching reference lyrics from lyrics.ovh...")
        reference = fetch_reference_lyrics(artist, title)
        if reference:
            ref_lines = parse_lyrics_lines(reference)
            print(f"  Got {len(ref_lines)} reference line(s).")
            corrected = align_words_to_reference(words, ref_lines)
            diffs = alignment_diff_summary(words, corrected)
            if diffs:
                print(f"  Corrected {len(diffs)} word(s) against the reference lyrics:")
                for d in diffs[:20]:
                    print(f"    {d}")
                if len(diffs) > 20:
                    print(f"    ... and {len(diffs) - 20} more")
            else:
                print("  ASR text already matched the reference; no corrections needed.")
            debug_log.log_reference_corrections(diffs)
            words = corrected
        else:
            print("  Could not fetch reference lyrics (not found, or no network); "
                  "continuing with ASR text and gap-based phrasing only.")
    else:
        print("Lyric lookup disabled (--no-fetch-lyrics); using ASR text and gap-based phrasing only.")

    # --- 7. PASS 3: fit words onto the pass-2 note grid (timing untouched) --
    print("Pass 3: fitting words onto the pass-2 note grid...")
    syllables, stats = align_words(words, notes, y, sr,
                                    verify_words=args.verify_words, verify_placement=args.verify_placement,
                                    verify_all_words=args.verify_all_words,
                                    verify_whisper_model=args.verify_whisper_model, debug_log=debug_log,
                                    verbose=not args.quiet)
    print(f"  {stats.words_with_notes}/{stats.total_words} words matched to pass-2 notes directly "
          f"({stats.total_notes_consumed} notes consumed); "
          f"{stats.words_with_fallback} word(s) needed a fallback note (no pass-2 note in their zone).")
    if stats.fallback_words:
        shown = stats.fallback_words[:15]
        print(f"    fallback words: {', '.join(shown)}" + (" ..." if len(stats.fallback_words) > 15 else ""))
        print(f"    (pitch source: {stats.fallback_used_neighbor} borrowed from the nearest pass-2 note, "
              f"{stats.fallback_used_fresh_analysis} needed a fresh isolated re-analysis because no "
              f"pass-2 notes existed at all -- the latter is the less reliable case)")
    if stats.words_with_melisma or stats.words_with_syllable_merge:
        print(f"    {stats.words_with_melisma} word(s) had melisma (fewer syllables than notes), "
              f"{stats.words_with_syllable_merge} word(s) had syllables merged (more syllables than notes)")
    if stats.lines_word_boundary_split:
        print(f"    {stats.lines_word_boundary_split} matched reference line(s) "
              f"({stats.words_in_word_boundary_split_lines} words) had their notes split by "
              f"each word's own ASR start/end time")
    if stats.verification_results:
        n_checked = len(stats.verification_results)
        n_replaced = sum(1 for r in stats.verification_results if r.replaced)
        print(f"    verification: re-transcribed {n_checked} suspicious word(s) in isolation, "
              f"replaced {n_replaced}")
    if stats.placement_corrections:
        print(f"    placement check: corrected {len(stats.placement_corrections)} word(s) whose FINAL "
              f"note-assigned position didn't match what's actually sung there (see [placement] lines "
              f"above) -- pass 3 was re-run with the fix applied")
    if stats.placement_warnings:
        print(f"    placement check: {len(stats.placement_warnings)} word(s) flagged -- the audio at "
              f"their FINAL note-assigned position doesn't say the expected word (see [placement] lines "
              f"above); these were NOT corrected automatically and are worth checking by hand")

    entries = build_lines(syllables)

    # --- 8. GAP = start of the first syllable --------------------------------
    first_syllable = next((e for e in entries if isinstance(e, Syllable)), None)
    gap_ms = int(round(first_syllable.start * 1000)) if first_syllable else 0

    # --- 9. VIDEOGAP ----------------------------------------------------
    videogap = None
    if companions.video and not args.no_video_sync:
        print("Estimating VIDEOGAP from the video's audio track...")
        videogap = estimate_videogap(companions.video, audio_path)
        if videogap is not None:
            print(f"Estimated VIDEOGAP: {videogap}s")
        else:
            print("Video has no usable audio track (or ffmpeg unavailable); leaving VIDEOGAP unset.")

    # --- 10. Preview start: default to first vocal, nudged back slightly ----
    preview_start = max(0.0, (first_syllable.start - 0.5)) if first_syllable else None

    # --- 11. Assemble + write Song --------------------------------------------
    def rel(p):
        return p.name if p else None

    song = Song(
        title=title,
        artist=artist,
        language=config.DEFAULT_LANGUAGE,
        mp3=audio_path.name,
        cover=rel(companions.cover),
        background=rel(companions.background),
        video=rel(companions.video),
        videogap=videogap,
        bpm=bpm,
        gap_ms=gap_ms,
        preview_start=preview_start,
        entries=entries,
    )

    out_name = f"{artist} - {title}.txt"
    out_path = output_dir / out_name
    write_song(song, out_path)
    print(f"Wrote {out_path}")

    if not args.skip_separation:
        print(f"(Intermediate files kept in {work_dir}; delete it to reclaim disk space.)")

    debug_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
