#!/usr/bin/env python3
"""
process_media_library.py

Single entry point that runs this project's standalone media-library
maintenance scripts against a folder tree, in one pass:

  0. file-utilities.py --clean-all -- delete leftover .bak/.usdb files
     and .ultrastar_work directories before anything else runs, so
     later stages never trip over stale intermediate artifacts.
  1. strip_audio_from_video.py  -- remove a video's own embedded audio
     track when a real sibling audio file (mp3/ogg/m4a/etc.) already
     covers the same song, so the video's audio isn't redundant weight.
  2. strip_lyrics_from_video.py -- remove an embedded "lyrics" container
     metadata tag and/or any subtitle stream from a video, so it never
     visually clashes with UltraStar's own on-screen lyrics.
  3. mp3_loudnorm.py             -- EBU R128 loudness-normalize every
     real audio file (and any mp4 with no sibling audio file).
  4. find_static_videos.py --delete -- detect and delete videos that are
     essentially a single still image, removing their #VIDEO tag too.
  5. find_large_vids.py + reduce_large_vids.py -- detect videos above
     720p/30fps and re-encode them down (H.264 CRF, resized/capped fps).

Each of the six stages is a real, independent, already-existing script
-- this is an orchestrator, not a reimplementation. Every stage is
invoked as its own subprocess, exactly as if it had been run by hand,
so their tested behavior is unchanged.

Reordering for speed (deliberately NOT the order listed above): static-
video deletion (stage 4) runs BEFORE the large-video reduction (stage
5), even though the task described them in the opposite order. Static
detection is cheap (two downscaled frame grabs); reduction is a slow
x264 "preset=slow" re-encode of the whole video. Deleting static videos
first means reduce_large_vids.py never wastes a slow re-encode on a
video that's about to be deleted anyway. Stages 1-3 (strip audio, strip
lyrics, normalize) have no ordering dependency on each other or on 4/5,
so they keep their original relative order at the front -- strip-audio
and strip-lyrics are both cheap remux-only passes over the video
container, so they run back-to-back before the slower audio-decode
work in normalization.

Every real change is gated behind a single --apply flag. Without it,
every stage runs in preview/detection-only mode (no file is modified
or deleted) -- this mirrors the safest option each underlying script
already offers, made uniform across all four. The one exception is
stage 0 (file-utilities.py --clean-all): that script has no preview
mode of its own, so without --apply it's skipped entirely rather than
deleting anything unasked. --create-backup is
passed to mp3_loudnorm.py by default (its own docstring documents
"backing up originals first" as the intended default behavior, even
though its own --create-backup flag defaults off) since loudness
normalization is a lossy re-encode with no other undo path; pass
--no-audio-backup to skip that if disk space under the backup drive is
a concern.

Usage
-----
  python process_media_library.py "Z:\\Songs"                 # preview only
  python process_media_library.py "Z:\\Songs" --apply          # do it for real
  python process_media_library.py "Z:\\Songs" --apply --skip-normalize
  python process_media_library.py "Z:\\Songs" --apply --crf 26 --preset medium
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_LUFS = -14.0
DEFAULT_TP = -1.0
DEFAULT_CRF = 28
DEFAULT_PRESET = "slow"
DEFAULT_STATIC_WORKERS = 32
DEFAULT_PROBE_WORKERS = 16
DEFAULT_ENCODE_WORKERS = 1


def run_step(name, cmd):
    print()
    print("=" * 70)
    print(f"STEP: {name}")
    print("=" * 70)
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print()

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)

    if result.returncode not in (0, None):
        print(f"WARNING: {name} exited with code {result.returncode} -- continuing with remaining steps.")

    return result.returncode


def step_clean_all(root, apply_changes):
    if not apply_changes:
        print()
        print("=" * 70)
        print("STEP: Clean leftover .bak/.usdb/.ultrastar_work files (skipped in preview mode)")
        print("=" * 70)
        print("  file-utilities.py --clean-all has no preview mode -- pass --apply to run it.")
        return None

    cmd = [sys.executable, str(SCRIPT_DIR / "file-utilities.py"), "--clean-all", str(root)]
    return run_step("Clean leftover .bak/.usdb/.ultrastar_work files", cmd)


def step_strip_audio(root, apply_changes):
    cmd = [sys.executable, str(SCRIPT_DIR / "strip_audio_from_video.py"), str(root)]
    if apply_changes:
        cmd.append("--apply")
    return run_step("Strip redundant audio from videos", cmd)


def step_strip_lyrics(root, apply_changes):
    cmd = [sys.executable, str(SCRIPT_DIR / "strip_lyrics_from_video.py"), str(root)]
    if apply_changes:
        cmd.append("--apply")
    return run_step("Strip embedded lyrics from videos", cmd)


def step_normalize_audio(root, apply_changes, lufs, tp, create_backup):
    cmd = [sys.executable, str(SCRIPT_DIR / "mp3_loudnorm.py"), str(root),
           "--lufs", str(lufs), "--tp", str(tp)]
    if not apply_changes:
        cmd.append("--dry-run")
    elif create_backup:
        cmd.append("--create-backup")
    return run_step("Normalize audio loudness", cmd)


def step_remove_static_videos(root, apply_changes, workers):
    log_path = root / "still_videos.txt"
    crc_cache_path = root / "static_video_crc_cache.json"
    cmd = [sys.executable, str(SCRIPT_DIR / "find_static_videos.py"), str(root),
           "--workers", str(workers), "--log", str(log_path),
           "--crc-cache", str(crc_cache_path)]
    if apply_changes:
        cmd.append("--delete")
    return run_step("Find/remove static (still-image) videos", cmd)


def step_reduce_large_videos(root, apply_changes, crf, preset, probe_workers, encode_workers):
    with tempfile.TemporaryDirectory(prefix="video_report_") as tmp_dir:
        report_path = Path(tmp_dir) / "video_report.txt"

        find_cmd = [sys.executable, str(SCRIPT_DIR / "find_large_vids.py"), str(root),
                    "--workers", str(probe_workers), "--crf", str(crf),
                    "-o", str(report_path)]
        rc = run_step("Find oversized (>720p/30fps) videos", find_cmd)

        if not report_path.is_file():
            print("WARNING: find_large_vids.py did not produce a report; skipping reduction.")
            return rc

        report_text = report_path.read_text(encoding="utf-8")
        print(report_text)

        if not apply_changes:
            print("(preview only -- pass --apply to actually re-encode these)")
            return rc

        if report_text.strip().startswith("No videos"):
            return rc

        reduce_cmd = [sys.executable, str(SCRIPT_DIR / "reduce_large_vids.py"), str(report_path),
                      "--workers", str(encode_workers), "--crf", str(crf), "--preset", preset]
        rc = run_step("Re-encode oversized videos", reduce_cmd)

        return rc


def main():
    parser = argparse.ArgumentParser(
        description="Run this project's media-library maintenance scripts against a folder tree in one pass.",
    )

    parser.add_argument("directory", help="Root directory to process")

    parser.add_argument(
        "--apply", action="store_true",
        help="Actually make changes. Without this, every stage only previews/detects.",
    )

    parser.add_argument("--skip-clean", action="store_true", help="Skip stage 0 (clean leftover .bak/.usdb/.ultrastar_work)")
    parser.add_argument("--skip-strip-audio", action="store_true", help="Skip stage 1 (strip redundant video audio)")
    parser.add_argument("--skip-strip-lyrics", action="store_true", help="Skip stage 2 (strip embedded video lyrics)")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip stage 3 (loudness normalization)")
    parser.add_argument("--skip-static", action="store_true", help="Skip stage 4 (static video removal)")
    parser.add_argument("--skip-reduce", action="store_true", help="Skip stage 5 (oversized video reduction)")

    parser.add_argument("--lufs", type=float, default=DEFAULT_LUFS, help=f"Loudnorm target LUFS (default: {DEFAULT_LUFS})")
    parser.add_argument("--tp", type=float, default=DEFAULT_TP, help=f"Loudnorm true peak ceiling dBTP (default: {DEFAULT_TP})")
    parser.add_argument(
        "--no-audio-backup", action="store_true",
        help="Don't pass --create-backup to mp3_loudnorm.py (backups are on by default here since normalization is lossy and irreversible otherwise)",
    )

    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help=f"x264 CRF for oversized-video reduction (default: {DEFAULT_CRF})")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help=f"x264 preset for oversized-video reduction (default: {DEFAULT_PRESET})")

    parser.add_argument("--static-workers", type=int, default=DEFAULT_STATIC_WORKERS, help=f"Worker threads for static-video detection (default: {DEFAULT_STATIC_WORKERS})")
    parser.add_argument("--probe-workers", type=int, default=DEFAULT_PROBE_WORKERS, help=f"Worker threads for oversized-video probing (default: {DEFAULT_PROBE_WORKERS})")
    parser.add_argument("--encode-workers", type=int, default=DEFAULT_ENCODE_WORKERS, help=f"Worker threads for oversized-video re-encoding (default: {DEFAULT_ENCODE_WORKERS})")

    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Error: {root} is not a directory.")

    print(f"Root: {root}")
    print(f"Mode: {'APPLY' if args.apply else 'PREVIEW (pass --apply to make real changes)'}")

    if not args.skip_clean:
        step_clean_all(root, args.apply)

    if not args.skip_strip_audio:
        step_strip_audio(root, args.apply)

    if not args.skip_strip_lyrics:
        step_strip_lyrics(root, args.apply)

    if not args.skip_normalize:
        step_normalize_audio(root, args.apply, args.lufs, args.tp, create_backup=not args.no_audio_backup)

    if not args.skip_static:
        step_remove_static_videos(root, args.apply, args.static_workers)

    if not args.skip_reduce:
        step_reduce_large_videos(root, args.apply, args.crf, args.preset, args.probe_workers, args.encode_workers)

    print()
    print("=" * 70)
    print("All steps complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
