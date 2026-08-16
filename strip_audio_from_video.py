import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


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
            text=True,
            check=True,
        )

        return json.loads(result.stdout)["streams"]

    except Exception as e:
        print(f"ERROR reading {path}")
        print(f"  {e}")
        return []


def video_has_audio(path):
    streams = get_streams(path)

    return any(
        stream.get("codec_type") == "audio"
        for stream in streams
    )


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

    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()

    matches = find_matching_files(root)

    print()

    for video in matches:
        if not video_has_audio(video):
            continue

        print(video)
        print("  Audio stream found")
        
        song_file = find_song_file(video)

        if song_file is None:
            print("  ERROR: Could not find associated .txt song file")
            print(f"  Expected: {video.with_suffix('.txt')}")
            return False

        try:
            data = song_file.read_text(encoding="cp1252")
        except (OSError, UnicodeError) as e:
            print(f"  ERROR reading song file: {e}")
            return False
            
        if re.search(rf"(?m)^#(?:AUDIO|MP3):.*{re.escape(video.suffix)}$", data, re.IGNORECASE):
            print(video)
            print("  ERROR: Song uses the video as its audio. This means there's an unused audio file in this directory.")
            return False
       
        if not args.apply:
            print("  DRY RUN: would remove audio")
            continue
            

        try:
            remove_audio(video)
            print("  Audio removed")

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print()
    print(f"Matching videos found: {len(matches)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")


if __name__ == "__main__":
    main()