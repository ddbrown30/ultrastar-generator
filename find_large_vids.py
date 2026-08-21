import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import subprocess


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

DEFAULT_WORKERS = 16
OUTPUT_FILE = "video_report.txt"

# Rough reference point for the "+/-6 CRF roughly halves/doubles bitrate"
# rule of thumb, used only to estimate whether a source is already below
# our target bitrate when no embedded CRF value is available. This is a
# coarse heuristic, not a real prediction of encoder output - the
# post-encode size check is the actual safety net.
REFERENCE_CRF = 23.0
REFERENCE_BPP = 0.1

MAX_WIDTH = 1280
MAX_HEIGHT = 720
MAX_FPS = 30.0
DEFAULT_CRF = 28

FFPROBE_FORMAT_BITRATE_CMD = [
    "ffprobe",
    "-v",
    "error",
    "-show_entries",
    "format=bit_rate",
    "-of",
    "json",
]

def get_video_info(video_path):
    """
    Return (width, height, fps, codec_name) or None if the file can't be
    read.
    """

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,bit_rate",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)

        streams = data.get("streams", [])
        if not streams:
            return None

        stream = streams[0]

        width = stream.get("width")
        height = stream.get("height")
        codec_name = stream.get("codec_name")
        bit_rate = stream.get("bit_rate")
        bit_rate = int(bit_rate) if bit_rate is not None else None

        fps_string = stream.get("r_frame_rate", "0/1")

        try:
            fps = float(Fraction(fps_string))
        except Exception:
            fps = 0.0

        return width, height, fps, codec_name, bit_rate

    except Exception:
        return None


def get_container_bitrate(video_path):
    """
    Overall container bitrate (video + audio combined). Used only as a
    fallback when the video stream itself doesn't report a bit_rate.
    """
    cmd = FFPROBE_FORMAT_BITRATE_CMD + [str(video_path)]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)
    bit_rate = data.get("format", {}).get("bit_rate")

    return int(bit_rate) if bit_rate is not None else None


def get_source_crf(video_path):
    """
    Try to recover the CRF a video was originally encoded with, by
    reading the x264 encoder settings string libx264 embeds in the
    bitstream (as an SEI message) - ffprobe doesn't parse this, so this
    shells out to `mediainfo` instead. Returns None (meaning "unknown",
    not "no CRF was used") if mediainfo isn't installed, the source
    wasn't encoded with x264, or the settings string was stripped.
    """
    try:
        result = subprocess.run(
            ["mediainfo", "--Output=JSON", str(video_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None

    tracks = data.get("media", {}).get("track", [])

    for track in tracks:
        if track.get("@type") != "Video":
            continue

        # Field naming for the embedded encoder settings string has
        # varied across mediainfo versions, so scan every string value
        # on the video track rather than relying on one exact key.
        for value in track.values():
            if not isinstance(value, str):
                continue

            match = re.search(r"\bcrf=(\d+\.?\d*)", value)
            if match:
                return float(match.group(1))

    return None


def estimate_bitrate_for_crf(width, height, fps, crf):
    """
    Very rough bits-per-pixel estimate for x264 at a given CRF, anchored
    on the common "+/-6 CRF roughly halves/doubles bitrate" rule of
    thumb. Only used to decide whether it's worth attempting an encode
    when the source's real CRF can't be determined - not a substitute
    for the post-encode size check.
    """
    bpp = REFERENCE_BPP * (2 ** ((REFERENCE_CRF - crf) / 6.0))
    return bpp * width * height * fps


def should_report(width, height, fps):
    return (
        width > MAX_WIDTH
        or height > MAX_HEIGHT
        or fps > MAX_FPS
    )


def process_video(video_path, max_crf):
    info = get_video_info(video_path)

    if info is None:
        return f"ERROR: {video_path}"

    width, height, fps, codec_name, bit_rate = info

    # If it's already HEVC, reduce_large_vids.py will skip it anyway -
    # no point flagging it here.
    if codec_name == "hevc":
        return None

    if not should_report(width, height, fps):
        return None

    source_crf = get_source_crf(video_path)

    # Same reasoning as reduce_large_vids.py: if the source is already
    # encoded at a higher CRF (= more compressed) than what we'd target,
    # re-encoding it isn't going to help and reduce_large_vids.py will
    # skip it - don't bother listing it.
    if source_crf is not None:
        if source_crf > max_crf:
            return None
    else:
        # No embedded CRF info available (hardware encoder, metadata
        # stripped, non-x264 source, mediainfo not installed) - fall
        # back to a rough bitrate comparison instead.
        source_bit_rate = bit_rate
        if source_bit_rate is None:
            source_bit_rate = get_container_bitrate(video_path)

        estimated_bit_rate = estimate_bitrate_for_crf(width, height, fps, max_crf)

        if source_bit_rate is not None and source_bit_rate <= estimated_bit_rate:
            return None

    return (
        f"{video_path}\n"
        f"    Resolution: {width}x{height}\n"
        f"    FPS: {fps:.3f}"
    )


def find_videos(root):
    for path in root.rglob("*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder to scan.",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker threads (default: {DEFAULT_WORKERS}).",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_FILE,
        help=f"Output file (default: {OUTPUT_FILE}).",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        help=(
            "Sources already encoded at a higher CRF (more compressed) "
            f"than this are excluded, to match reduce_large_vids.py "
            f"(default: {DEFAULT_CRF})"
        ),
    )

    args = parser.parse_args()

    root = Path(args.folder).resolve()

    videos = list(find_videos(root))

    print(f"Found {len(videos)} video files.")

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_video, video, args.crf): video
            for video in videos
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if result:
                results.append(result)

            if i % 100 == 0 or i == len(videos):
                print(f"Processed {i}/{len(videos)}")

    results.sort()

    with open(args.output, "w", encoding="utf-8") as f:
        if results:
            f.write("\n\n".join(results))
        else:
            f.write("No videos exceeded 720p or 30 FPS.\n")

    print(f"Found {len(results)} matching files.")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()