"""CLI entry point.

Usage:
    python -m ultrastar_generator.main "C:\\Songs\\Bon Jovi - Its My Life.mp3"

See README.md for the full option list and setup instructions.
"""

from __future__ import annotations

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
from .alignment import build_entries
from .lyrics_lookup import fetch_reference_lyrics, parse_lyrics_lines, align_words_to_reference, alignment_diff_summary
from .video_sync import estimate_videogap
from .usdx_writer import write_song
from .debug_output import write_pass1_debug_file


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate an UltraStar Deluxe .txt song file from an mp3/ogg/oga file."
    )
    p.add_argument("audio", help="Path to the audio file, named '<Artist> - <Title>.<mp3|ogg|oga>'")
    p.add_argument("--artist", help="Override artist parsed from filename")
    p.add_argument("--title", help="Override title parsed from filename")
    p.add_argument("--output-dir", help="Where to write the .txt (default: same folder as audio)")
    p.add_argument("--work-dir", help="Scratch space for separated stems etc. (default: <output-dir>/.ultrastar_work)")
    p.add_argument("--whisper-model", default=config.DEFAULT_WHISPER_MODEL,
                    help=f"faster-whisper model name (default: {config.DEFAULT_WHISPER_MODEL})")
    p.add_argument("--demucs-model", default=config.DEFAULT_DEMUCS_MODEL,
                    help=f"Demucs model name (default: {config.DEFAULT_DEMUCS_MODEL})")
    p.add_argument("--bpm", type=float, default=None, help="Override detected BPM")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device for ML models")
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
    p.add_argument("--key-correction", dest="key_correction", action="store_true",
                    default=config.ENABLE_KEY_CORRECTION,
                    help="Enable the musical-key pitch-snapping polish pass (off by default -- "
                         "it can over-correct genuine chromatic/passing-tone notes)")
    p.add_argument("--no-key-correction", dest="key_correction", action="store_false",
                    help="Disable the musical-key pitch-snapping polish pass (default)")
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
    p.add_argument("--no-pass1-debug", action="store_true",
                    help="Don't write the '[PASS1 DEBUG]' .txt (pass-1 notes only, no lyrics) "
                         "that's written by default alongside the real output -- load it in the "
                         "UltraStar editor to check pass 1's timing/pitch in isolation.")
    p.add_argument("--quiet", action="store_true",
                    help="Suppress the verbose [pass1]/[pass2] diagnostic logging (still prints "
                         "the main pipeline stage messages)")
    return p


def run(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

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
    work_dir = Path(args.work_dir).resolve() if args.work_dir else (output_dir / ".ultrastar_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {artist} - {title} ==")

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
        print("Isolating vocals with Demucs (this can take a while on CPU)...")
        try:
            vocals_path = isolate_vocals(audio_path, work_dir, model=args.demucs_model, device=args.device)
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
        verbose=not args.quiet,
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

    # --- 5. Transcription (lyrics text + rough timing) -----------------------
    print(f"Transcribing with {'whisperx' if not args.no_whisperx else 'faster-whisper'} ({args.whisper_model})...")
    words = transcribe_words(vocals_path, args.whisper_model, device=args.device,
                              prefer_whisperx=not args.no_whisperx)
    if not words:
        print("No words were transcribed -- check the audio / vocal isolation quality.", file=sys.stderr)
        return 1
    print(f"Transcribed {len(words)} words.")

    # --- 6. Reference lyrics: correct ASR text AND mark phrase/line breaks --
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
            words = corrected
        else:
            print("  Could not fetch reference lyrics (not found, or no network); "
                  "continuing with ASR text and gap-based phrasing only.")
    else:
        print("Lyric lookup disabled (--no-fetch-lyrics); using ASR text and gap-based phrasing only.")

    # --- 7. PASS 2: fit words onto the pass-1 note grid (timing untouched) --
    print("Pass 2: fitting words onto the pass-1 note grid...")
    entries, stats = build_entries(words, notes, y, sr, key_correction=args.key_correction)
    print(f"  {stats.words_with_notes}/{stats.total_words} words matched to pass-1 notes directly "
          f"({stats.total_notes_consumed} notes consumed); "
          f"{stats.words_with_fallback} word(s) needed a fallback note (no pass-1 note in their zone).")
    if stats.fallback_words:
        shown = stats.fallback_words[:15]
        print(f"    fallback words: {', '.join(shown)}" + (" ..." if len(stats.fallback_words) > 15 else ""))
        print(f"    (pitch source: {stats.fallback_used_neighbor} borrowed from the nearest pass-1 note, "
              f"{stats.fallback_used_fresh_analysis} needed a fresh isolated re-analysis because no "
              f"pass-1 notes existed at all -- the latter is the less reliable case)")
    if stats.words_with_melisma or stats.words_with_syllable_merge:
        print(f"    {stats.words_with_melisma} word(s) had melisma (fewer syllables than notes), "
              f"{stats.words_with_syllable_merge} word(s) had syllables merged (more syllables than notes)")
    if stats.lines_syllable_distributed:
        print(f"    {stats.lines_syllable_distributed} matched reference line(s) "
              f"({stats.words_in_syllable_distributed_lines} words) had their notes distributed by "
              f"syllable count rather than individual ASR word timing")

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

    return 0


if __name__ == "__main__":
    sys.exit(run())
