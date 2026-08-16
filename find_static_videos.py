import argparse
import concurrent.futures
import os
import re
import struct
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np


# Keep child processes (ffprobe/ffmpeg) out of the console's Ctrl+C
# signal group. Without this, on Windows a Ctrl+C is broadcast to
# every process attached to the console -- including in-flight
# ffprobe/ffmpeg calls -- which kills them and prints spurious error
# messages for jobs we intended to let finish. On POSIX,
# start_new_session has the equivalent effect by taking the child out
# of the terminal's foreground process group.
if os.name == "nt":
    _SUBPROCESS_KWARGS = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
else:
    _SUBPROCESS_KWARGS = {"start_new_session": True}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".flac",
    ".wav",
    ".ogg",
    ".oga",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".webm",
    ".flv",
    ".m4v",
    ".mpg",
    ".mpeg",
}

# Fast-pass settings
FAST_FRAME_SIZE = (64, 64)
FAST_THRESHOLD = 2.0

LOG_FILE = "still_videos.txt"
DEFAULT_WORKERS = 32


@contextmanager
def _timed(timing, key):
    """Accumulate wall-clock time spent in a block under `timing[key]`."""

    start = time.perf_counter()
    try:
        yield
    finally:
        timing[key] += time.perf_counter() - start


def get_video_duration(path, stop_event=None):
    """Get video duration in seconds using ffprobe."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            **_SUBPROCESS_KWARGS,
        )

        return float(result.stdout.strip())

    except (subprocess.SubprocessError, ValueError) as e:
        # If a stop was requested, a failure here is most likely a
        # side effect of shutting down rather than a real problem
        # with this file, so stay quiet about it.
        if stop_event is None or not stop_event.is_set():
            print(f"  ERROR getting duration: {e}")
        return None


def _extract_frame_seek(video_path, position_seconds, stop_event=None):
    """
    Extract one downscaled grayscale frame near the given time by
    asking ffmpeg to seek within the real file directly.

    Uses ffmpeg's own fast input-side seek (-ss before -i, plus
    -noaccurate_seek). This is the FALLBACK path -- see extract_frame
    for the fragment-bypass primary path and why it's needed: measured
    on real data, this direct-seek approach costs 2-7+ seconds per
    frame at scale on some real fragmented-mp4 files, confirmed to be
    CPU-bound (same cost on local disk as over a network share, and a
    raw byte-range read of the same file at the same offset is
    sub-millisecond) -- i.e. ffmpeg is doing real forward-decode work
    to satisfy the seek, not waiting on I/O, regardless of
    -noaccurate_seek.
    """

    width, height = FAST_FRAME_SIZE

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{position_seconds:.3f}",
        "-noaccurate_seek",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height}:flags=fast_bilinear,format=gray",
        "-f",
        "rawvideo",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            **_SUBPROCESS_KWARGS,
        )
    except subprocess.SubprocessError as e:
        if stop_event is None or not stop_event.is_set():
            print(f"  ERROR extracting frame at {position_seconds:.1f}s: {e}")
        return None

    expected_bytes = width * height

    if len(result.stdout) != expected_bytes:
        if stop_event is None or not stop_event.is_set():
            print(
                f"  ERROR: Unexpected frame size at {position_seconds:.1f}s "
                f"({len(result.stdout)} bytes, expected {expected_bytes})"
            )
        return None

    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width)


def _extract_frame_from_bytes(data, stop_event=None):
    """Decode the first available frame from an in-memory mp4 blob."""

    width, height = FAST_FRAME_SIZE

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "mp4",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height}:flags=fast_bilinear,format=gray",
        "-f",
        "rawvideo",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=data,
            capture_output=True,
            check=True,
            **_SUBPROCESS_KWARGS,
        )
    except subprocess.SubprocessError:
        if stop_event is None or not stop_event.is_set():
            print("  WARNING: Fragment-bypass decode failed, falling back")
        return None

    expected_bytes = width * height

    if len(result.stdout) != expected_bytes:
        return None

    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width)


def _read_box_header(f, offset, size_total):
    """Read one ISO-BMFF box header (type + total size, header included)."""

    if offset + 8 > size_total:
        return None

    f.seek(offset)
    header = f.read(8)

    if len(header) < 8:
        return None

    box_size, box_type_raw = struct.unpack(">I4s", header)
    box_type = box_type_raw.decode("ascii", "replace")

    if box_size == 1:
        ext = f.read(8)

        if len(ext) < 8:
            return None

        box_size = struct.unpack(">Q", ext)[0]
    elif box_size == 0:
        box_size = size_total - offset

    if box_size < 8:
        return None

    return box_type, box_size


def _read_fragment_span(f, offset, size_total):
    """Total byte length of one moof+mdat fragment starting at offset."""

    header = _read_box_header(f, offset, size_total)

    if header is None or header[0] != "moof":
        return None

    _, moof_size = header
    span = moof_size

    next_header = _read_box_header(f, offset + moof_size, size_total)

    if next_header is not None:
        span += next_header[1]

    return span


# Number of consecutive fragments to hand to ffmpeg as decode runway --
# more than 1 in case the very first fragment's own first frame isn't
# itself a keyframe for some encoder.
FRAGMENT_BUFFER_COUNT = 2


def locate_fragment_near_fraction(video_path, fraction):
    """
    For a fragmented ISO-BMFF (mp4/mov) file, find a small, self-
    contained chunk of bytes -- the "init segment" (everything before
    the first moof: ftyp/moov/sidx/etc, which carries codec init data)
    plus one or two fragments (moof+mdat) starting at/near byte
    position `fraction` of the file -- that ffmpeg can decode standalone
    via a pipe, without ever seeking within the real (possibly huge)
    file itself.

    Byte-fraction is used as a stand-in for time-fraction (bitrate is
    roughly constant within one encode) -- consistent with this
    script's existing several-seconds tolerance for sample position.

    Returns (init_bytes, fragment_bytes), or None if this doesn't look
    like a fragmented mp4 (no moof boxes found) -- callers should fall
    back to normal ffmpeg seeking in that case.
    """

    size_total = video_path.stat().st_size
    target_offset = int(size_total * fraction)

    with open(video_path, "rb") as f:
        offset = 0
        first_moof_offset = None
        prev_moof_offset = None
        chosen_offset = None

        while offset < size_total:
            header = _read_box_header(f, offset, size_total)

            if header is None:
                break

            box_type, box_size = header

            if box_type == "moof":
                if first_moof_offset is None:
                    first_moof_offset = offset

                if offset <= target_offset:
                    prev_moof_offset = offset
                else:
                    chosen_offset = prev_moof_offset
                    break

            offset += box_size
        else:
            chosen_offset = prev_moof_offset

        if first_moof_offset is None or chosen_offset is None:
            return None

        f.seek(0)
        init_bytes = f.read(first_moof_offset)

        total_span = 0
        frag_offset = chosen_offset

        for _ in range(FRAGMENT_BUFFER_COUNT):
            span = _read_fragment_span(f, frag_offset, size_total)

            if span is None:
                break

            total_span += span
            frag_offset += span

            if frag_offset >= size_total:
                break

        if total_span == 0:
            return None

        f.seek(chosen_offset)
        fragment_bytes = f.read(total_span)

    return init_bytes, fragment_bytes


def extract_frame(video_path, position_seconds, fraction, stop_event=None, timing=None):
    """
    Extract one downscaled grayscale frame near the given time.

    Primary path: locate a small, self-contained fragment at/near the
    target byte-fraction via a cheap box walk (see
    locate_fragment_near_fraction), and decode just that tiny blob via
    a piped ffmpeg call -- never asks ffmpeg to seek within the real
    file at all. Motivated by real measurement: asking ffmpeg to seek
    directly (_extract_frame_seek) cost 2-7+ seconds per frame at scale
    on a real fragmented-mp4 file, identically on local disk and over a
    network share, while a raw byte-range read of the same file at the
    same offset was sub-millisecond -- i.e. the cost is ffmpeg doing
    real forward-decode CPU work to satisfy a seek in this container
    layout, not I/O wait, and not something -noaccurate_seek prevents.

    Falls back to _extract_frame_seek when the file doesn't look like a
    fragmented mp4, or when the piped decode fails for any reason --
    correctness is never at risk, only speed.
    """

    if timing is None:
        timing = defaultdict(float)

    with _timed(timing, "box_walk"):
        try:
            located = locate_fragment_near_fraction(video_path, fraction)
        except OSError:
            located = None

    if located is not None:
        init_bytes, fragment_bytes = located

        with _timed(timing, "pipe_decode"):
            frame = _extract_frame_from_bytes(
                init_bytes + fragment_bytes, stop_event=stop_event
            )

        if frame is not None:
            timing["bypass_used"] += 1
            return frame

    timing["seek_fallback_used"] += 1

    with _timed(timing, "seek_fallback"):
        frame = _extract_frame_seek(video_path, position_seconds, stop_event=stop_event)

    return frame


FRACTIONS = (0.10, 0.90)


def fast_pass(video_path, stop_event=None, timing=None):
    """
    Quickly determine whether a video is potentially static.
    """

    if timing is None:
        timing = defaultdict(float)

    with _timed(timing, "ffprobe"):
        duration = get_video_duration(video_path, stop_event=stop_event)

    if duration is None or duration <= 0:
        return None

    frames = []

    for fraction in FRACTIONS:
        if stop_event is not None and stop_event.is_set():
            return None

        position = duration * fraction

        # Broken out per fraction (rather than one combined bucket) so
        # we can tell whether cost scales with how far into the file
        # the target position is. extract_frame records its own
        # internal box_walk/pipe_decode/seek_fallback timing on top of
        # this.
        with _timed(timing, f"total_{fraction:.2f}"):
            frame = extract_frame(
                video_path, position, fraction, stop_event=stop_event, timing=timing
            )

        if frame is None:
            return None

        frames.append(frame)

    with _timed(timing, "diff_compute"):
        reference = frames[0].astype(np.int16)

        differences = [
            float(np.abs(reference - frame.astype(np.int16)).mean())
            for frame in frames[1:]
        ]

        max_difference = max(differences)

    return max_difference <= FAST_THRESHOLD, max_difference


def analyze_video(video_path, stop_event=None):
    """
    Worker function for parallel video analysis.

    Returns:
        (video_path, result, difference, timing)
    """

    timing = defaultdict(float)

    try:
        result = fast_pass(video_path, stop_event=stop_event, timing=timing)

        if result is None:
            return video_path, None, None, timing

        is_static, difference = result

        return video_path, is_static, difference, timing

    except Exception as e:
        return video_path, None, str(e), timing


def find_song_file(video_path):
    """Find the associated UltraStar song file."""

    song_file = video_path.with_suffix(".txt")

    if song_file.is_file():
        return song_file

    return None


def remove_video_tag(song_file):
    """Remove the #VIDEO line from an UltraStar song file."""

    try:
        data = song_file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as e:
        print(f"  ERROR reading song file: {e}")
        return False

    new_data, count = re.subn(
        r"(?im)^#VIDEO:[^\r\n]*(?:\r\n|\n|\r|$)",
        "",
        data,
    )

    if count == 0:
        print("  WARNING: No #VIDEO tag found")
        return False

    try:
        song_file.write_text(
            new_data,
            encoding="utf-8",
            newline="",
        )
    except OSError as e:
        print(f"  ERROR writing song file: {e}")
        return False

    print(f"  Removed #VIDEO from: {song_file}")
    return True


def delete_static_video(video_path):
    """
    Remove the #VIDEO tag from the associated song file and
    delete the video.

    Returns True if the video was deleted.
    """

    if not video_path.is_file():
        print("  ERROR: Video file does not exist")
        return False

    song_file = find_song_file(video_path)

    if song_file is None:
        print("  ERROR: Could not find associated .txt song file")
        print(f"  Expected: {video_path.with_suffix('.txt')}")
        return False

    try:
        data = song_file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as e:
        print(f"  ERROR reading song file: {e}")
        return False

    if not re.search(r"(?im)^#VIDEO:", data):
        print("  ERROR: Associated song file has no #VIDEO tag")
        return False

    if not remove_video_tag(song_file):
        return False

    try:
        video_path.unlink()
    except OSError as e:
        print(f"  ERROR deleting video: {e}")
        print(
            "  WARNING: #VIDEO was removed but video "
            "could not be deleted."
        )
        return False

    print(f"  DELETED: {video_path}")

    return True


def find_video_files(root):
    """
    Find all video files recursively.

    Uses os.walk (backed by os.scandir) rather than Path.rglob("*") +
    path.is_file(): scandir already knows each entry's file/dir type
    from the directory read itself, so os.walk's filenames list needs
    no extra per-entry stat() call. Path.rglob("*") doesn't carry that
    info, so the separate .is_file() call re-stats every entry (files
    and directories alike) -- an extra round trip per entry on a
    network share.
    """

    videos = []        
    
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue

        files = {}

        for file in directory.iterdir():
            if not file.is_file():
                continue

            files.setdefault(file.stem.lower(), []).append(file)

        for stem, paths in files.items():
            audio_files = [
                p for p in paths
                if p.suffix.lower() in AUDIO_EXTENSIONS
            ]

            video_files = [
                p for p in paths
                if p.suffix.lower() in VIDEO_EXTENSIONS
            ]

            if audio_files and video_files:
                for video in video_files:
                    videos.append(video)

    return videos


_TIMING_PHASES = (
    ("ffprobe", "ffprobe duration"),
    ("box_walk", "Fragment box walk"),
    ("pipe_decode", "Piped ffmpeg decode (bypass)"),
    ("seek_fallback", "ffmpeg -ss seek (fallback)"),
    *(
        (f"total_{fraction:.2f}", f"Total time @ {fraction:.0%}")
        for fraction in FRACTIONS
    ),
    ("diff_compute", "Diff computation"),
)


def _print_timing_summary(walk_time, analysis_time, analyzed_count, totals, counts):
    print()
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Videos analyzed:    {analyzed_count}")
    print(f"Directory scan:     {walk_time:.2f}s")
    print(f"Analysis wall time: {analysis_time:.2f}s")
    print()
    print("Per-phase totals (summed across all worker threads --")
    print("will add up to more than the wall time above, since the")
    print("phases run concurrently across workers):")
    print()

    for key, label in _TIMING_PHASES:
        count = counts.get(key, 0)

        if count == 0:
            continue

        total = totals[key]
        avg_ms = (total / count) * 1000

        print(f"  {label:<36} {total:8.2f}s  (avg {avg_ms:7.1f}ms x{count})")

    bypass_count = int(totals.get("bypass_used", 0))
    fallback_count = int(totals.get("seek_fallback_used", 0))

    print()
    print(f"  Fragment-bypass path used: {bypass_count} time(s)")
    print(f"  Seek-fallback path used:   {fallback_count} time(s)")

    print("=" * 60)


def find_static_videos(root, workers, stop_event):
    """
    Find static videos using parallel fast-pass analysis.

    Ctrl+C causes queued work to be cancelled. Videos already being
    processed are allowed to finish their current operation.
    """

    walk_start = time.perf_counter()
    videos = find_video_files(root)
    walk_time = time.perf_counter() - walk_start

    print(f"Found {len(videos)} video files.")
    print(f"Directory scan took {walk_time:.2f}s.")
    print(f"Using {workers} workers.")
    print("Press Ctrl+C to stop after current operations finish.")
    print()

    matches = []
    timing_totals = defaultdict(float)
    timing_counts = defaultdict(int)
    analyzed_count = 0

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    )

    futures = {}

    analysis_start = time.perf_counter()

    try:
        for path in videos:
            if stop_event.is_set():
                break

            future = executor.submit(
                analyze_video,
                path,
                stop_event,
            )

            futures[future] = path

        try:
            for future in concurrent.futures.as_completed(futures):
                if stop_event.is_set():
                    break

                path = futures[future]

                try:
                    video_path, result, difference, timing = future.result()
                except Exception as e:
                    print()
                    print(f"ERROR: {path}")
                    print(f"  {e}")
                    continue

                analyzed_count += 1

                for key, value in timing.items():
                    timing_totals[key] += value
                    timing_counts[key] += 1

                print(f"Processing: {video_path}")

                if result is None:
                    # A None result while shutting down is expected
                    # (the job was cut short by the stop request),
                    # not a real failure worth reporting.
                    if stop_event.is_set():
                        print("  SKIPPED: Cancelled")
                    elif isinstance(difference, str):
                        print(f"  ERROR: {difference}")
                    else:
                        print("  SKIPPED: Could not analyze")
                    continue

                print(f"  Difference: {difference:.2f}")

                if not result:
                    continue

                print(f"  STATIC {video_path}")
                matches.append(video_path.resolve())

        except KeyboardInterrupt:
            print()
            print()
            print("Ctrl+C received. Stopping...")
            stop_event.set()

    finally:
        # Cancel anything that hasn't started.
        cancelled = 0

        for future in futures:
            if future.cancel():
                cancelled += 1

        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        if cancelled:
            print(f"Cancelled {cancelled} queued jobs.")

    analysis_time = time.perf_counter() - analysis_start

    _print_timing_summary(
        walk_time,
        analysis_time,
        analyzed_count,
        timing_totals,
        timing_counts,
    )

    return matches


def delete_from_log(log_file):
    """
    Delete videos listed in a previously generated log file.

    No video detection is performed.
    """

    if not log_file.is_file():
        print(f"ERROR: Log file not found: {log_file}")
        return 1

    try:
        paths = [
            Path(line.strip())
            for line in log_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as e:
        print(f"ERROR reading log file: {e}")
        return 1

    print(f"Found {len(paths)} video(s) in log.")
    print()

    successful = 0
    failed = 0

    try:
        for video_path in paths:
            print(f"Processing: {video_path}")

            if delete_static_video(video_path):
                successful += 1
            else:
                failed += 1

            print()

    except KeyboardInterrupt:
        print()
        print()
        print("Ctrl+C received. Stopping deletion.")
        print(
            f"Deleted {successful} of {len(paths)} "
            f"videos before interruption."
        )
        return 1

    print("=" * 60)
    print(f"Processed: {len(paths)}")
    print(f"Deleted:   {successful}")
    print(f"Failed:    {failed}")
    print("=" * 60)

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find videos containing essentially a single still image. "
            "Optionally delete them and remove their #VIDEO tags."
        )
    )

    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search recursively",
    )

    parser.add_argument(
        "--log",
        default=LOG_FILE,
        help=f"Output/input log file (default: {LOG_FILE})",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Number of videos to analyze simultaneously "
            f"(default: {DEFAULT_WORKERS})"
        ),
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Delete detected static videos and remove the associated "
            "#VIDEO tag from their UltraStar song file"
        ),
    )

    group.add_argument(
        "--delete-from-log",
        action="store_true",
        help=(
            "Delete videos listed in the log file instead of "
            "running video detection"
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    script_start = time.perf_counter()

    log_file = Path(args.log).expanduser().resolve()

    # ---------------------------------------------------------
    # DELETE FROM LOG
    # ---------------------------------------------------------

    if args.delete_from_log:
        exit_code = delete_from_log(log_file)
        print(f"Total time: {time.perf_counter() - script_start:.2f}s")
        return exit_code

    # ---------------------------------------------------------
    # NORMAL DETECTION
    # ---------------------------------------------------------

    root = Path(args.directory).expanduser().resolve()

    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 1

    if args.delete:
        print("WARNING: --delete mode enabled.")
        print("Static videos will be deleted and #VIDEO tags removed.")
        print()

    stop_event = threading.Event()

    try:
        matches = find_static_videos(
            root,
            workers=args.workers,
            stop_event=stop_event,
        )

    except KeyboardInterrupt:
        # Safety net in case Ctrl+C occurs during thread cleanup.
        print()
        print("Interrupted.")

        matches = []

    # Always write whatever results were collected.
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            for path in matches:
                f.write(f"{path}\n")
    except OSError as e:
        print(f"ERROR writing log: {e}")
        return 1

    if stop_event.is_set():
        print()
        print("=" * 60)
        print("SCAN INTERRUPTED")
        print(f"Static videos found so far: {len(matches)}")
        print(f"Partial results written to: {log_file}")
        print(f"Total time: {time.perf_counter() - script_start:.2f}s")
        print("=" * 60)
        return 130

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    if args.delete:
        print()
        print("Removing static videos and #VIDEO tags...")
        print()

        successful = 0
        failed = 0

        try:
            for video_path in matches:
                print(f"Processing: {video_path}")

                if delete_static_video(video_path):
                    successful += 1
                else:
                    failed += 1

                print()

        except KeyboardInterrupt:
            print()
            print()
            print("Ctrl+C received. Stopping deletion.")
            print(
                f"Deleted {successful} of {len(matches)} "
                f"detected videos before interruption."
            )
            return 130

        print("=" * 60)
        print(f"Found:   {len(matches)} static videos")
        print(f"Deleted: {successful}")
        print(f"Failed:  {failed}")
        print(f"Log:     {log_file}")
        print(f"Total time: {time.perf_counter() - script_start:.2f}s")
        print("=" * 60)

    else:
        print()
        print("=" * 60)
        print(f"Found {len(matches)} static videos.")
        print(f"Results written to: {log_file}")
        print("Mode: DRY RUN")
        print(f"Total time: {time.perf_counter() - script_start:.2f}s")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())