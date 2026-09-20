"""
Strip embedded lyrics out of UltraStar library videos.

Real-world finding this is built around: some downloaded videos carry
their full lyrics as a CONTAINER METADATA TAG (ffprobe's
format.tags.lyrics -- an mp4 "©lyr" atom under the hood), not as a
subtitle stream. Confirmed on two real files via ffprobe (tag present,
never printed/inspected as text -- only its length was checked): one
video had a 46KB "lyrics" format tag with no separate audio stream at
all; a sibling song's .m4a (not its .mp4) carried the same tag instead.
So the tag's presence is per-file, not tied to one particular stream
layout -- this script checks the video's own container directly rather
than assuming where the tag will be.

A real embedded SUBTITLE stream (mp3_loudnorm.py already preserves
these via "-map 0:s?" when normalizing audio) is also dropped if
present, since UltraStar always renders its own lyrics and a muxed-in
subtitle track would be redundant either way -- but the metadata tag is
the confirmed, primary case.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


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

LYRICS_TAG_NAME = "lyrics"


def find_tag_key(tags):
    """Case-insensitive lookup of a 'lyrics' key among a tags dict."""

    for key in tags:
        if key.lower() == LYRICS_TAG_NAME:
            return key

    return None


def get_lyrics_info(path):
    """
    Inspect a video's own container for embedded lyrics.

    Returns (format_tag_key, stream_tag_hits, subtitle_indices):
      - format_tag_key: the container-level metadata key holding lyrics
        (e.g. "lyrics"), or None.
      - stream_tag_hits: [(stream_index, tag_key), ...] for any stream
        that itself carries a lyrics tag (not seen in the real files
        this was validated against, but checked for robustness -- an
        embedded lyrics tag could in principle live on a track's own
        metadata instead of the container's).
      - subtitle_indices: stream indices with codec_type == "subtitle".

    Only tag KEYS and stream metadata are ever read/printed by this
    module -- the tag's own text VALUE is never inspected or logged.
    """

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
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

        data = json.loads(result.stdout)

    except Exception as e:
        print(f"ERROR reading {path}")
        print(f"  {e}")
        return None, [], []

    format_tags = data.get("format", {}).get("tags", {}) or {}
    format_tag_key = find_tag_key(format_tags)

    stream_tag_hits = []
    subtitle_indices = []

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "subtitle":
            subtitle_indices.append(stream["index"])

        stream_tags = stream.get("tags", {}) or {}
        stream_tag_key = find_tag_key(stream_tags)

        if stream_tag_key is not None:
            stream_tag_hits.append((stream["index"], stream_tag_key))

    return format_tag_key, stream_tag_hits, subtitle_indices


def remove_lyrics(video_path, format_tag_key, stream_tag_hits, subtitle_indices):
    """
    Re-mux the video with the lyrics metadata cleared and any subtitle
    stream dropped. Every actual audio/video stream is copied
    byte-for-byte (no re-encode) -- only metadata and stream selection
    change.
    """

    temp_path = video_path.with_suffix(
        f".tmp{video_path.suffix}"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map_metadata",
        "0",
        "-c",
        "copy",
    ]

    if subtitle_indices:
        cmd += ["-sn"]

    if format_tag_key is not None:
        cmd += ["-metadata", f"{format_tag_key}="]

    for stream_index, tag_key in stream_tag_hits:
        cmd += [f"-metadata:s:{stream_index}", f"{tag_key}="]

    cmd += [str(temp_path)]

    subprocess.run(cmd, check=True)

    backup_path = video_path.with_suffix(
        video_path.suffix + ".bak"
    )

    shutil.move(video_path, backup_path)
    shutil.move(temp_path, video_path)

    backup_path.unlink()


def find_video_files(root):
    """
    Find video files that are part of an UltraStar song folder (have a
    companion .txt song file), recursively.
    """

    videos = []

    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue

        for file in directory.iterdir():
            if not file.is_file():
                continue

            if file.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            if file.with_suffix(".txt").is_file():
                videos.append(file)

    return videos


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
        help="Actually remove embedded lyrics from matching videos",
    )

    args = parser.parse_args()

    root = Path(args.directory).expanduser().resolve()

    matches = find_video_files(root)

    print()

    videos_with_lyrics = 0

    for video in matches:
        format_tag_key, stream_tag_hits, subtitle_indices = get_lyrics_info(video)

        if format_tag_key is None and not stream_tag_hits and not subtitle_indices:
            continue

        videos_with_lyrics += 1

        print(video)

        if format_tag_key is not None:
            print(f"  Container 'lyrics' metadata tag found ({format_tag_key})")

        for stream_index, tag_key in stream_tag_hits:
            print(f"  Stream {stream_index} 'lyrics' metadata tag found ({tag_key})")

        if subtitle_indices:
            print(f"  Subtitle stream(s) found: {len(subtitle_indices)}")

        if not args.apply:
            print("  DRY RUN: would remove embedded lyrics")
            continue

        try:
            remove_lyrics(video, format_tag_key, stream_tag_hits, subtitle_indices)
            print("  Lyrics removed")

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print()
    print(f"Matching videos found: {len(matches)}")
    print(f"Videos with embedded lyrics: {videos_with_lyrics}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")


if __name__ == "__main__":
    main()
