import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from crc_cache import CrcCache, compute_crc32


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
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".flv",
}

CRC_NAMESPACE = "strip_audio"


def get_streams(path):
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )

        return True, json.loads(result.stdout)["streams"]

    except Exception as e:
        print(f"ERROR reading {path}")
        print(f"  {e}")
        return False, []


def video_has_audio(path):
    """Returns (ok, has_audio) -- ok is False if ffprobe/parsing failed,
    in which case has_audio is meaningless and must not be cached."""

    ok, streams = get_streams(path)

    has_audio = any(
        stream.get("codec_type") == "audio"
        for stream in streams
    )

    return ok, has_audio


def remove_audio(video_path):
    temp_path = video_path.with_suffix(
        f".tmp{video_path.suffix}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "copy",
        "-an",
        str(temp_path),
    ]

    subprocess.run(cmd, check=True)

    backup_path = video_path.with_suffix(
        video_path.suffix + ".bak"
    )

    shutil.move(video_path, backup_path)
    shutil.move(temp_path, video_path)

    backup_path.unlink()


def find_matching_files(root):
    matches = []

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
                    matches.append(video)

    return matches
    
def find_song_file(video_path):
    """Find the associated UltraStar song file."""

    song_file = video_path.with_suffix(".txt")

    if song_file.is_file():
        return song_file

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to search",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove audio from matching videos",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-probe every video even if its CRC cache entry is still valid",
    )

    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()

    matches = find_matching_files(root)
    cache = CrcCache(CRC_NAMESPACE)

    print(f"CRC cache: {len(cache.entries)} entries loaded ({CRC_NAMESPACE})")
    print()

    cache_hits = 0

    for video in matches:
        try:
            check = cache.check(video, root, force=args.force)
        except OSError as e:
            print(video)
            print(f"  ERROR reading file: {e}")
            print()
            continue

        entry = None

        if check.fast_hit:
            entry = check.cached_entry
        elif check.moved_from is not None:
            entry = check.moved_entry

        if entry is not None:
            has_audio = entry.get("has_audio")
            from_cache = True
        else:
            ok, has_audio = video_has_audio(video)
            from_cache = False

            if not ok:
                # A failed probe is an unknown, not a verified result --
                # don't cache it either way.
                continue

        if from_cache:
            cache_hits += 1

        def _record(has_audio_value):
            cache.record(
                check.rel_key, check.crc, check.size, check.mtime_ns,
                moved_from=check.moved_from, has_audio=has_audio_value,
            )

        if not has_audio:
            if not check.fast_hit:
                _record(False)

            continue

        print(video)

        if check.moved_from is not None:
            print(f"  Audio stream found (cached, moved from {check.moved_from})")
        else:
            print("  Audio stream found" + (" (cached)" if from_cache else ""))

        song_file = find_song_file(video)

        if song_file is None:
            print("  ERROR: Could not find associated .txt song file")
            print(f"  Expected: {video.with_suffix('.txt')}")
            return False

        try:
            data = song_file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                data = song_file.read_text(encoding="cp1252")
            except (OSError, UnicodeError) as e:
                print(f"  ERROR reading song file: {e}")
                return False
        except OSError as e:
            print(f"  ERROR reading song file: {e}")
            return False
            
        if re.search(rf"(?m)^#(?:AUDIO|MP3):.*{re.escape(video.suffix)}$", data, re.IGNORECASE):
            print(video)
            print("  ERROR: Song uses the video as its audio. This means there's an unused audio file in this directory.")
            return False
       
        if not args.apply:
            print("  DRY RUN: would remove audio")

            if not check.fast_hit:
                _record(True)

            continue


        try:
            remove_audio(video)
            print("  Audio removed")

            # The file was just re-muxed, so its CRC has changed --
            # record the new one along with the now-known-clean state
            # rather than re-probing the freshly written file.
            new_st = video.stat()
            cache.record(
                check.rel_key, compute_crc32(video), new_st.st_size, new_st.st_mtime_ns,
                has_audio=False,
            )

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print()
    print(f"Matching videos found: {len(matches)}")
    print(f"Cache hits (skipped re-probe): {cache_hits}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")


if __name__ == "__main__":
    main()