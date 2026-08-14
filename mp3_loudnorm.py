#!/usr/bin/env python3
"""
mp3_loudnorm.py

Recursively apply EBU R128 loudness normalization (default -14 LUFS integrated
/ -1 dBTP true peak -- the standard streaming-loudness target) to every MP3,
OGG (Vorbis), M4A, and MP4 file in a directory tree, using ffmpeg's loudnorm
filter in two-pass mode. For MP4 (and M4A) files, only the audio stream is
touched -- video (and any other streams: subtitles, chapters, attached
cover art) are copied through byte-for-byte, never re-encoded.

MP4 handling has two special rules, aimed at libraries (e.g. karaoke
collections) where a song folder can have both an audio track (mp3/ogg/m4a)
and an mp4 music video for the same song:
  - An .mp4 is only processed if its folder does NOT also contain an .mp3,
    .ogg, or .m4a file. If one does, the .mp4 is left alone entirely.
  - An .mp4 with no audio stream at all (e.g. a silent video) is skipped.
    This is not treated as an error.

Modes
-----
  (default)   Normalize all supported audio files under <directory>, backing
              up originals first.
  --verify    Compare normalized files against their backups (real decoded
              duration, sample rate, channel count, resulting loudness/true
              peak) and report any problems. Does not modify anything unless
              --restore-on-failure is also given.
  --restore   Copy every backup back over its current file, undoing
              normalization (after confirmation).
  --cleanup   Delete the backup files created by a previous normalize run
              (after confirmation).
  --dry-run   With no mode flags: list which files would be processed
              without touching anything.

Flags
-----
  --lufs FLOAT              Target integrated loudness (default: -14)
  --tp FLOAT                True peak ceiling in dBTP (default: -1.0)
  --force                   Reprocess files even if a backup already exists.
                             The existing backup is preserved, never
                             overwritten -- it always holds the true original.
  --stop-on-error           Stop the whole run as soon as one file errors
                             out (normalize mode only).
  --restore-on-failure      With --verify: automatically restore the backup
                             for any file that fails verification.

Backups are stored in a parallel directory tree, so they mirror your folder
structure and are easy to find, verify against, restore, or delete:

    <directory>/.loudnorm_backups/<same relative path as original file>

Requirements
------------
  - Python 3.8+
  - ffmpeg and ffprobe available on PATH, with libmp3lame, libvorbis, and aac
    encoders enabled (all three are included in standard ffmpeg builds).
    Note: MP4 audio is re-encoded with ffmpeg's built-in "aac" encoder, not
    the higher-quality libfdk_aac (which most ffmpeg builds omit for
    licensing reasons) -- quality is good but not best-in-class.
    (Windows: download from https://www.gyan.dev/ffmpeg/builds/, add the
    'bin' folder to PATH; macOS: brew install ffmpeg; Linux: apt/dnf install)

Usage
-----
  python mp3_loudnorm.py "C:\\Music"                       # normalize (-14 LUFS / -1 dBTP)
  python mp3_loudnorm.py "C:\\Music" --dry-run              # preview only
  python mp3_loudnorm.py "C:\\Music" --lufs -16 --tp -1.5   # different targets
  python mp3_loudnorm.py "C:\\Music" --force                # reprocess already-backed-up files
  python mp3_loudnorm.py "C:\\Music" --stop-on-error         # halt run on first error
  python mp3_loudnorm.py "C:\\Music" --verify                # check results
  python mp3_loudnorm.py "C:\\Music" --verify --restore-on-failure   # and auto-fix failures
  python mp3_loudnorm.py "C:\\Music" --restore               # undo normalization entirely
  python mp3_loudnorm.py "C:\\Music" --cleanup                # delete backups
"""

import argparse
import filecmp
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

BACKUP_DIRNAME = ".loudnorm_backups"
DURATION_ABORT_TOLERANCE = 0.5    # seconds; skip replacing file if exceeded
DURATION_VERIFY_TOLERANCE = 0.05  # seconds; flagged in --verify
LOUDNESS_VERIFY_TOLERANCE = 1.0   # LUFS
TRUE_PEAK_VERIFY_TOLERANCE = 0.5  # dBTP; flags files that overshoot the ceiling

# Sanity bounds for loudnorm's pass-1 measurement. Real audio is always well
# inside these. Values outside them mean something went wrong decoding the
# file (corruption, desync, etc.) -- feeding such values into pass 2 can
# make ffmpeg error out or crash, so we catch it here instead.
PLAUSIBLE_LUFS_RANGE = (-70.0, 0.0)
PLAUSIBLE_TP_RANGE = (-99.0, 9.0)

# Maps supported file extensions to the ffmpeg encoder used for their audio.
# For container formats that can carry video (mp4 always; m4a/mp3/ogg only
# incidentally, e.g. embedded cover art), any non-audio stream is always
# passed through untouched via "-c copy" -- see normalize_file() -- so only
# the mapping to an audio encoder is needed here.
SUPPORTED_EXTENSIONS = {
    ".mp3": "libmp3lame",
    ".ogg": "libvorbis",
    ".mp4": "aac",
    ".m4a": "aac",
}

# Extensions treated as "real" standalone audio tracks (as opposed to .mp4,
# which is treated as a video file that may incidentally also carry audio).
# A folder containing one of these causes any sibling .mp4 to be skipped --
# see find_files().
AUDIO_ONLY_EXTENSIONS = {".mp3", ".ogg", ".m4a"}


def check_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"Error: '{tool}' not found on PATH. Install ffmpeg and try again.")


def find_files(root: Path):
    candidates = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.endswith(".tmp" + p.suffix)
        and BACKUP_DIRNAME not in p.relative_to(root).parts
    ]

    # An .mp4 is only a candidate if its folder has no "real" audio file
    # (.mp3/.ogg/.m4a) alongside it -- e.g. a karaoke folder with both a
    # song's audio track and its music video should only touch the audio.
    dirs_with_audio = {
        p.parent for p in candidates if p.suffix.lower() in AUDIO_ONLY_EXTENSIONS
    }

    files = [
        p for p in candidates
        if p.suffix.lower() != ".mp4" or p.parent not in dirs_with_audio
    ]
    return sorted(files)


def ensure_writable(path: Path):
    """Clear the read-only bit if set, so the file can be backed up,
    overwritten, or restored. Files marked read-only (common with files
    synced from cloud storage, extracted from archives, etc.) would
    otherwise cause a permissions error partway through processing."""
    try:
        if not path.exists():
            return
        mode = path.stat().st_mode
        if not mode & stat.S_IWUSR:
            path.chmod(mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    except OSError:
        pass


def _remove_readonly_onerror(func, path, exc_info):
    """shutil.rmtree onerror handler: clear read-only and retry once."""
    try:
        Path(path).chmod(stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def verify_backup(original: Path, backup: Path) -> bool:
    """Confirm a freshly-created backup is a byte-for-byte faithful copy."""
    try:
        if not backup.exists():
            return False
        if backup.stat().st_size != original.stat().st_size:
            return False
        return filecmp.cmp(str(original), str(backup), shallow=False)
    except OSError:
        return False


def restore_file(backup_path: Path, current_path: Path):
    """Copy a backup back over the current file, clearing read-only first."""
    ensure_writable(current_path)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, current_path)


def ffprobe_json(path: Path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(
        cmd, capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    )
    return json.loads(out.stdout)


def get_audio_stream_info(path: Path):
    """Bitrate/sample rate/channel count, used to match encoding settings.
    Deliberately does NOT report duration here -- container-reported
    duration can be wrong (see get_decoded_duration).
    Returns None if the file has no audio stream at all (e.g. a silent
    mp4) -- callers should treat that as "skip, not an error"."""
    info = ffprobe_json(path)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if audio is None:
        return None
    bit_rate = audio.get("bit_rate") or info["format"].get("bit_rate")
    sample_rate = audio.get("sample_rate")
    channels = audio.get("channels")
    return {
        "bit_rate": int(bit_rate) if bit_rate else None,
        "sample_rate": int(sample_rate) if sample_rate else None,
        "channels": int(channels) if channels else None,
    }


def _parse_ffmpeg_timestamp(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _extract_out_time(stdout_text: str):
    """Pull the last out_time= value from ffmpeg -progress output."""
    last = None
    for line in stdout_text.splitlines():
        if line.startswith("out_time="):
            val = line.split("=", 1)[1].strip()
            if val and val != "N/A":
                last = val
    return _parse_ffmpeg_timestamp(last) if last else None


def get_decoded_duration(path: Path) -> float:
    """Decode the entire audio stream and report its real, played-back
    duration. This is more reliable than container/header-reported
    duration (ffprobe), which for some files -- e.g. VBR mp3s without an
    accurate Xing/LAME header -- can be an estimate that's slightly off."""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-i", str(path),
        "-map", "0:a:0",
        "-f", "null",
        "-progress", "pipe:1", "-nostats",
        "-",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    )
    duration = _extract_out_time(result.stdout)
    if duration is None:
        raise RuntimeError(f"Could not determine decoded duration for {path}")
    return duration


def measure_loudness(path: Path, target_lufs: float, target_tp: float):
    """Pass 1: measure the stats loudnorm needs for an accurate pass 2, and
    capture the real decoded duration of the source in the same pass (so we
    don't have to decode the file twice just to also check its duration)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
        "-af", f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=7:print_format=json",
        "-map", "0:a:0",
        "-f", "null",
        "-progress", "pipe:1",
        "-",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    stderr = result.stderr
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"Could not parse loudnorm measurement for {path}:\n{stderr[-2000:]}")
    return json.loads(stderr[start:end + 1])


def validate_measured_stats(stats: dict, path: Path):
    """Guard against feeding a bad pass-1 measurement into pass 2. A corrupt
    or desynced source file can make ffmpeg's decoder briefly read garbage
    as audio, producing physically impossible loudness/true-peak numbers
    that would otherwise crash or misbehave in the second pass."""
    try:
        i = float(stats["input_i"])
        tp = float(stats["input_tp"])
    except (KeyError, ValueError, TypeError) as e:
        raise RuntimeError(f"loudness measurement returned unparseable values ({e})")

    lo, hi = PLAUSIBLE_LUFS_RANGE
    if not (lo <= i <= hi):
        raise RuntimeError(
            f"loudness measurement looks invalid (integrated loudness {i:.2f} LUFS "
            f"is outside a plausible range) -- the file may be corrupt or have a "
            f"decoding issue"
        )

    lo, hi = PLAUSIBLE_TP_RANGE
    if not (lo <= tp <= hi):
        raise RuntimeError(
            f"loudness measurement looks invalid (true peak {tp:.2f} dBTP is "
            f"outside a plausible range) -- the file may be corrupt or have a "
            f"decoding issue"
        )


def normalize_file(path: Path, target_lufs: float, target_tp: float, info: dict, stats: dict):
    """Pass 2: apply the measured gain and encode. Returns (tmp_out_path,
    decoded_duration_of_output) -- the duration comes from the same encode
    pass so no extra decode is needed to check it afterward."""
    ext = path.suffix.lower()
    codec = SUPPORTED_EXTENSIONS[ext]

    # linear=true: applies normalization as a single static gain whenever
    # the measured dynamic range allows it, rather than a time-varying
    # (dynamics-altering) correction. This is what keeps the result as
    # close as possible to "same audio, different volume."
    filter_str = (
        f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=7:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:linear=true:print_format=summary"
    )

    # Keep the real extension on the temp file (e.g. "song.tmp.mp3") so
    # ffmpeg's output-format autodetection still works correctly.
    tmp_out = path.with_name(f"{path.stem}.tmp{ext}")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        # Map the primary audio stream explicitly (it's the only one we
        # touch), plus video/subtitle/data streams if present (optional "?"
        # so this doesn't error on mp3/ogg files that have none of those).
        # "-c copy" below then copies everything except audio byte-for-byte
        # -- this is what keeps an mp4's video untouched, and also protects
        # embedded cover art on mp3/ogg files instead of re-encoding it.
        "-map", "0:v?",
        "-map", "0:a:0",
        "-map", "0:s?",
        "-map", "0:d?",
        "-map_metadata", "0",
        "-map_chapters", "0",
        "-c", "copy",
        "-af", filter_str,
        "-c:a", codec,
    ]
    if ext == ".mp3":
        cmd += ["-id3v2_version", "3"]
    if info["bit_rate"]:
        cmd += ["-b:a", str(info["bit_rate"])]
    if info["sample_rate"]:
        cmd += ["-ar", str(info["sample_rate"])]
    if info["channels"]:
        cmd += ["-ac", str(info["channels"])]
    cmd += ["-progress", "pipe:1", "-nostats", str(tmp_out)]

    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    new_duration = _extract_out_time(result.stdout)
    if new_duration is None or abs(new_duration) < 1e-4:
        new_duration = get_decoded_duration(tmp_out)  # fallback, rarely needed
    return tmp_out, new_duration


def do_normalize(root: Path, target_lufs: float, target_tp: float, dry_run: bool,
                  force: bool, stop_on_error: bool):
    check_tools()
    backup_root = root / BACKUP_DIRNAME
    files = find_files(root)
    if not files:
        print("No mp3/ogg/mp4/m4a files found.")
        return

    print(f"Found {len(files)} audio file(s). Target: {target_lufs} LUFS integrated, "
          f"{target_tp} dBTP true peak ceiling.")
    if dry_run:
        print("(dry run -- no files will be changed)\n")

    processed = 0
    skipped = 0
    errors = 0

    for path in files:
        rel = path.relative_to(root)
        backup_path = backup_root / rel
        backup_exists = backup_path.exists()

        if backup_exists and not force:
            print(f"SKIP (backup already exists, looks already processed): {rel}")
            skipped += 1
            continue

        print(f"Processing: {rel}")
        if dry_run:
            continue

        tmp_out = None
        try:
            info = get_audio_stream_info(path)
            if info is None:
                print(f"  SKIP: no audio stream found.")
                skipped += 1
                continue

            stats = measure_loudness(path, target_lufs, target_tp)
            validate_measured_stats(stats, path)
            old_lufs = float(stats["input_i"])
            orig_duration = get_decoded_duration(path)

            tmp_out, new_duration = normalize_file(path, target_lufs, target_tp, info, stats)

            if abs(new_duration - orig_duration) > DURATION_ABORT_TOLERANCE:
                print(f"  ERROR: decoded duration changed from {orig_duration:.3f}s to "
                      f"{new_duration:.3f}s. Leaving original untouched.")
                tmp_out.unlink(missing_ok=True)
                tmp_out = None
                errors += 1
                if stop_on_error:
                    print("Stopping (--stop-on-error).")
                    break
                continue

            if not backup_exists:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_path)
                if not verify_backup(path, backup_path):
                    print(f"  ERROR: backup verification failed for {rel}. "
                          f"Skipping file, original left untouched.")
                    backup_path.unlink(missing_ok=True)
                    tmp_out.unlink(missing_ok=True)
                    tmp_out = None
                    errors += 1
                    if stop_on_error:
                        print("Stopping (--stop-on-error).")
                        break
                    continue

            ensure_writable(path)
            tmp_out.replace(path)
            tmp_out = None
            print(f"  Done. ({old_lufs:.1f} LUFS -> {target_lufs:.1f} LUFS target, "
                  f"{orig_duration:.3f}s)")
            processed += 1

        except subprocess.CalledProcessError as e:
            detail = (e.stderr or "").strip()
            if detail:
                print(f"  ERROR processing {rel}: {e}\n    ffmpeg said: {detail[-500:]}")
            else:
                print(f"  ERROR processing {rel}: {e}")
            errors += 1
            if stop_on_error:
                print("Stopping (--stop-on-error).")
                break
        except Exception as e:
            print(f"  ERROR processing {rel}: {e}")
            errors += 1
            if stop_on_error:
                print("Stopping (--stop-on-error).")
                break
        finally:
            if tmp_out and tmp_out.exists():
                tmp_out.unlink(missing_ok=True)

    if not dry_run:
        print(f"\n{processed} normalized, {skipped} skipped, {errors} error(s).")
        print(f"Backups stored under: {backup_root}")


def do_verify(root: Path, target_lufs: float, target_tp: float, restore_on_failure: bool):
    check_tools()
    backup_root = root / BACKUP_DIRNAME
    if not backup_root.exists():
        print("No backup directory found; nothing to verify.")
        return

    backups = sorted(
        p for p in backup_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not backups:
        print("No backup files found; nothing to verify.")
        return

    problems = 0
    checked = 0
    restored = 0

    for backup_path in backups:
        rel = backup_path.relative_to(backup_root)
        current_path = root / rel

        if not current_path.exists():
            print(f"MISSING: {rel} has a backup but no current file.")
            problems += 1
            continue

        checked += 1
        issues = []

        try:
            b_info = get_audio_stream_info(backup_path)
            c_info = get_audio_stream_info(current_path)

            if b_info["sample_rate"] and c_info["sample_rate"] and \
                    b_info["sample_rate"] != c_info["sample_rate"]:
                issues.append(f"sample rate changed ({b_info['sample_rate']} -> {c_info['sample_rate']})")

            if b_info["channels"] and c_info["channels"] and \
                    b_info["channels"] != c_info["channels"]:
                issues.append(f"channel count changed ({b_info['channels']} -> {c_info['channels']})")

            # Real decoded duration, not container metadata -- see
            # get_decoded_duration's docstring for why this matters.
            b_duration = get_decoded_duration(backup_path)
            c_duration = get_decoded_duration(current_path)
            dur_diff = abs(b_duration - c_duration)
            if dur_diff > DURATION_VERIFY_TOLERANCE:
                issues.append(
                    f"decoded duration differs by {dur_diff:.3f}s "
                    f"({b_duration:.3f}s -> {c_duration:.3f}s)"
                )

            stats = measure_loudness(current_path, target_lufs, target_tp)
            validate_measured_stats(stats, current_path)
            measured_i = float(stats["input_i"])
            lufs_diff = abs(measured_i - target_lufs)
            if lufs_diff > LOUDNESS_VERIFY_TOLERANCE:
                issues.append(
                    f"loudness off target: {measured_i:.1f} LUFS "
                    f"(target {target_lufs}, diff {lufs_diff:.1f})"
                )

            measured_tp = float(stats["input_tp"])
            if measured_tp > target_tp + TRUE_PEAK_VERIFY_TOLERANCE:
                issues.append(
                    f"true peak exceeds ceiling: {measured_tp:.1f} dBTP "
                    f"(ceiling {target_tp} dBTP)"
                )
        except subprocess.CalledProcessError as e:
            issues.append(f"could not probe/measure file: {e}")
        except Exception as e:
            issues.append(f"could not probe/measure file: {e}")

        if issues:
            problems += 1
            print(f"ISSUE: {rel}")
            for i in issues:
                print(f"    - {i}")
            if restore_on_failure:
                try:
                    restore_file(backup_path, current_path)
                    print(f"    -> restored from backup")
                    restored += 1
                except Exception as e:
                    print(f"    -> ERROR restoring from backup: {e}")
        else:
            print(f"OK: {rel}")

    print(f"\nVerified {checked} file(s). {problems} problem(s) found.")
    if restore_on_failure:
        print(f"{restored} file(s) restored from backup.")


def do_restore(root: Path):
    backup_root = root / BACKUP_DIRNAME
    if not backup_root.exists():
        print("No backup directory found; nothing to restore.")
        return

    backups = sorted(
        p for p in backup_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not backups:
        print("No backup files found; nothing to restore.")
        return

    print(f"This will restore {len(backups)} file(s) under:\n  {root}\n"
          f"from their backups, overwriting the current (normalized) versions.")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return

    restored = 0
    errors = 0
    for backup_path in backups:
        rel = backup_path.relative_to(backup_root)
        current_path = root / rel
        try:
            restore_file(backup_path, current_path)
            print(f"Restored: {rel}")
            restored += 1
        except Exception as e:
            print(f"ERROR restoring {rel}: {e}")
            errors += 1

    print(f"\n{restored} restored, {errors} error(s).")


def do_cleanup(root: Path):
    backup_root = root / BACKUP_DIRNAME
    if not backup_root.exists():
        print("No backup directory found; nothing to clean up.")
        return

    confirm = input(
        f"This will permanently delete all backups under:\n  {backup_root}\n"
        f"Type 'yes' to confirm: "
    )
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return

    shutil.rmtree(backup_root, onerror=_remove_readonly_onerror)
    print("Backups deleted.")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively apply loudness normalization to mp3/ogg/mp4/m4a files."
    )
    parser.add_argument("directory", type=str, help="Root directory to process")
    parser.add_argument(
        "--lufs", type=float, default=-14.0,
        help="Target integrated loudness in LUFS (default: -14, the typical "
             "streaming-music target; use -23 for EBU R128 broadcast)",
    )
    parser.add_argument(
        "--tp", type=float, default=-1.0,
        help="True peak ceiling in dBTP (default: -1.0)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify normalized files against backups instead of normalizing",
    )
    parser.add_argument(
        "--restore", action="store_true",
        help="Restore original files from their backups instead of normalizing",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete backup files instead of normalizing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be processed without changing anything (normalize mode only)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Process files even if a backup already exists. The existing "
             "backup is preserved, never overwritten (normalize mode only)",
    )
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="Stop processing as soon as an error is encountered (normalize mode only)",
    )
    parser.add_argument(
        "--restore-on-failure", action="store_true",
        help="With --verify: automatically restore the backup for any file "
             "that fails verification",
    )

    args = parser.parse_args()
    root = Path(args.directory).expanduser().resolve()

    if not root.is_dir():
        sys.exit(f"Error: {root} is not a directory.")

    mode_flags = [args.cleanup, args.verify, args.restore]
    if sum(bool(x) for x in mode_flags) > 1:
        sys.exit("Error: choose only one of --verify, --restore, or --cleanup.")

    if args.restore_on_failure and not args.verify:
        sys.exit("Error: --restore-on-failure requires --verify.")

    if args.cleanup:
        do_cleanup(root)
    elif args.restore:
        do_restore(root)
    elif args.verify:
        do_verify(root, args.lufs, args.tp, args.restore_on_failure)
    else:
        do_normalize(root, args.lufs, args.tp, args.dry_run, args.force, args.stop_on_error)


if __name__ == "__main__":
    main()