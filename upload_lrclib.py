"""
Upload or update synchronized lyrics on LRCLIB from an LRC file.

Requirements:
    py -3.11 -m pip install requests lrclibapi

Usage:
    py upload_lrclib.py "C:\\path\\to\\song.lrc"

Behavior:
    - Reads artist/title/album from LRC metadata when available.
    - Prompts for missing metadata.
    - Prompts for duration when it isn't present in the LRC.
    - Converts the LRC into LRCLIB-compatible synced/plain lyrics.
    - Checks LRCLIB for an existing matching entry.
    - If an entry exists, displays it and asks whether to update it.
    - Updating an existing entry publishes a new LRCLIB revision.
    - Previous LRCLIB revisions are retained by the server.
    - Uses lrclibapi's built-in proof-of-work solver.
    - Sends the publish request directly so an empty 201 response
      does not cause a JSON parsing error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests
from lrclib.api import LrcLibAPI
from lrclib.exceptions import (
    APIError,
    IncorrectPublishTokenError,
    NotFoundError,
    RateLimitError,
    ServerError,
)


USER_AGENT = "LRCLIB-Lyrics-Uploader/1.0"
LRCLIB_URL = "https://lrclib.net"


# ---------------------------------------------------------------------------
# LRC parsing
# ---------------------------------------------------------------------------

def parse_lrc(path: Path) -> tuple[dict[str, str], str, str]:
    """
    Parse an LRC file.

    Returns:
        metadata
        synced lyrics
        plain lyrics
    """

    text = path.read_text(encoding="utf-8-sig")

    metadata: dict[str, str] = {}

    metadata_patterns = {
        "artist": re.compile(r"^\[ar:(.*?)\]\s*$", re.IGNORECASE),
        "title": re.compile(r"^\[ti:(.*?)\]\s*$", re.IGNORECASE),
        "album": re.compile(r"^\[al:(.*?)\]\s*$", re.IGNORECASE),
        "length": re.compile(r"^\[length:(.*?)\]\s*$", re.IGNORECASE),
    }

    for line in text.splitlines():
        for key, pattern in metadata_patterns.items():
            match = pattern.match(line)
            if match:
                metadata[key] = match.group(1).strip()
                break

    timestamp_pattern = re.compile(
        r"\[(\d+):(\d{2})(?:[.:](\d{1,3}))?\]"
    )

    timed_lines: list[tuple[int, str]] = []

    for line in text.splitlines():
        matches = list(timestamp_pattern.finditer(line))

        if not matches:
            continue

        lyric_text = line[matches[-1].end():].strip()

        for match in matches:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            fraction = match.group(3)

            if fraction is None:
                milliseconds = 0
            elif len(fraction) == 1:
                milliseconds = int(fraction) * 100
            elif len(fraction) == 2:
                milliseconds = int(fraction) * 10
            else:
                milliseconds = int(fraction[:3])

            timestamp_ms = (
                minutes * 60_000
                + seconds * 1_000
                + milliseconds
            )

            timed_lines.append((timestamp_ms, lyric_text))

    if not timed_lines:
        raise ValueError(
            f"No timed lyric lines were found in {path}"
        )

    timed_lines.sort(key=lambda item: item[0])

    synced_lines = []

    for timestamp_ms, lyric_text in timed_lines:
        minutes = timestamp_ms // 60_000
        seconds = (timestamp_ms % 60_000) // 1_000
        centiseconds = (timestamp_ms % 1_000) // 10

        synced_lines.append(
            f"[{minutes:02d}:{seconds:02d}.{centiseconds:02d}]"
            f"{lyric_text}"
        )

    synced_lyrics = "\n".join(synced_lines)

    plain_lyrics = "\n".join(
        lyric_text
        for _, lyric_text in timed_lines
    )

    return metadata, synced_lyrics, plain_lyrics


# ---------------------------------------------------------------------------
# Metadata / duration
# ---------------------------------------------------------------------------

def parse_duration(value: str | None) -> float | None:
    """
    Parse duration as:
        seconds
        MM:SS
        MM:SS.s
    """

    if not value:
        return None

    value = value.strip()

    try:
        return float(value)
    except ValueError:
        pass

    match = re.fullmatch(
        r"(\d+):(\d+(?:\.\d+)?)",
        value,
    )

    if match:
        return (
            int(match.group(1)) * 60
            + float(match.group(2))
        )

    raise ValueError(
        f"Invalid duration {value!r}. "
        "Use seconds or MM:SS."
    )


def get_metadata(metadata: dict[str, str]) -> tuple[
    str,
    str,
    str,
    int,
]:
    """Prompt for missing metadata and return LRCLIB values."""

    artist = metadata.get("artist", "").strip()
    title = metadata.get("title", "").strip()
    album = metadata.get("album", "").strip()

    if not artist:
        artist = input("Artist: ").strip()

    if not title:
        title = input("Title: ").strip()

    if not album:
        album = input("Album: ").strip()

    duration = parse_duration(metadata.get("length"))

    if duration is None:
        while True:
            value = input(
                "Duration (seconds or MM:SS): "
            ).strip()

            try:
                duration = parse_duration(value)
                if duration is not None:
                    break
            except ValueError as exc:
                print(f"  {exc}")

    return (
        artist,
        title,
        album,
        round(duration),
    )


# ---------------------------------------------------------------------------
# Existing LRCLIB entry
# ---------------------------------------------------------------------------

def find_existing(
    api: LrcLibAPI,
    artist: str,
    title: str,
    album: str,
    duration: int,
):
    """Find an existing LRCLIB entry for the track."""

    try:
        return api.get_lyrics(
            track_name=title,
            artist_name=artist,
            album_name=album,
            duration=duration,
        )

    except NotFoundError:
        return None


def display_existing(existing) -> None:
    """Display the existing LRCLIB entry."""

    print()
    print("=" * 70)
    print("EXISTING LRCLIB ENTRY")
    print("=" * 70)

    print(f"ID:       {existing.id}")
    print(f"Artist:   {existing.artist_name}")
    print(f"Title:    {existing.track_name}")
    print(f"Album:    {existing.album_name}")
    print(f"Duration: {existing.duration} seconds")

    synced = existing.synced_lyrics or ""
    plain = existing.plain_lyrics or ""

    print(f"Synced:   {'yes' if synced else 'no'}")
    print(f"Plain:    {'yes' if plain else 'no'}")

    if synced:
        print()
        print("Existing synchronized lyrics:")
        print("-" * 70)

        lines = synced.splitlines()
        preview = lines[:10]

        for line in preview:
            print(line)

        if len(lines) > len(preview):
            print(
                f"... ({len(lines) - len(preview)} more lines)"
            )

    print("=" * 70)


def confirm_update(existing) -> bool:
    """
    Ask whether the existing lyrics should be updated.

    LRCLIB does not overwrite the existing revision. A successful
    publish creates a new revision while retaining the old one.
    """

    print()
    print(
        "An LRCLIB entry already exists for this track."
    )
    print(
        "Updating it will publish your lyrics as a new revision."
    )
    print(
        "The existing revision will be retained by LRCLIB."
    )

    try:
        answer = input(
            "\nUpdate the existing lyrics? [y/N]: "
        ).strip().lower()

    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return False

    return answer in {"y", "yes"}


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

def publish(
    api: LrcLibAPI,
    artist: str,
    title: str,
    album: str,
    duration: int,
    plain_lyrics: str,
    synced_lyrics: str,
):
    """
    Publish lyrics directly to LRCLIB.

    We deliberately do not use api.publish_lyrics() because some
    lrclibapi versions attempt to parse an empty successful response
    as JSON, producing:

        Expecting value: line 1 column 1 (char 0)

    LRCLIB documents 201 Created as the successful response and does
    not require a JSON response body.
    """

    # Use lrclibapi's solver to obtain a valid one-use publish token.
    publish_token = api._obtain_publish_token()

    payload = {
        "trackName": title,
        "artistName": artist,
        "albumName": album,
        "duration": duration,
        "plainLyrics": plain_lyrics,
        "syncedLyrics": synced_lyrics,
    }

    response = api.session.post(
        f"{LRCLIB_URL}/api/publish",
        headers={
            "User-Agent": USER_AGENT,
            "X-Publish-Token": publish_token,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if not response.ok:
        print(
            f"LRCLIB returned HTTP {response.status_code}.",
            file=sys.stderr,
        )

        if response.text:
            print(response.text, file=sys.stderr)

        response.raise_for_status()

    # LRCLIB may return an empty 201 response.
    if not response.content:
        return None

    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    if "json" in content_type:
        return response.json()

    return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_upload(
    api: LrcLibAPI,
    artist: str,
    title: str,
    album: str,
    duration: int,
    expected_synced: str,
) -> bool:
    """
    Fetch the track again and verify that LRCLIB contains the
    synchronized lyrics we just published.
    """

    print()
    print("Verifying published lyrics...")

    try:
        result = api.get_lyrics(
            track_name=title,
            artist_name=artist,
            album_name=album,
            duration=duration,
        )
    except Exception as exc:
        print(
            f"WARNING: Could not verify the upload: {exc}"
        )
        return False

    actual_synced = result.synced_lyrics or ""

    if actual_synced.strip() == expected_synced.strip():
        print("Verification successful.")
        return True

    print(
        "WARNING: LRCLIB returned the track, but the synchronized "
        "lyrics do not exactly match the uploaded lyrics."
    )

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload or update an LRC file on LRCLIB."
    )

    parser.add_argument(
        "lrc_file",
        type=Path,
        help="Path to the .lrc file.",
    )

    args = parser.parse_args()
    path = args.lrc_file

    if not path.exists():
        print(
            f"ERROR: File does not exist:\n  {path}",
            file=sys.stderr,
        )
        return 1

    if not path.is_file():
        print(
            f"ERROR: Not a file:\n  {path}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Parse LRC.
    # ------------------------------------------------------------------

    try:
        metadata, synced_lyrics, plain_lyrics = parse_lrc(path)

    except UnicodeDecodeError:
        print(
            "ERROR: The LRC file could not be decoded as UTF-8.",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(
            f"ERROR: Failed to parse LRC:\n  {exc}",
            file=sys.stderr,
        )
        return 1

    print()
    print("=" * 70)
    print("LRCLIB LYRICS UPLOADER")
    print("=" * 70)
    print(f"File: {path}")

    # ------------------------------------------------------------------
    # Metadata.
    # ------------------------------------------------------------------

    try:
        (
            artist,
            title,
            album,
            duration,
        ) = get_metadata(metadata)

    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 0

    print()
    print("Upload information:")
    print(f"  Artist:   {artist}")
    print(f"  Title:    {title}")
    print(f"  Album:    {album}")
    print(f"  Duration: {duration} seconds")
    print(f"  Lines:    {len(synced_lyrics.splitlines())}")

    # ------------------------------------------------------------------
    # API client.
    # ------------------------------------------------------------------

    api = LrcLibAPI(
        user_agent=USER_AGENT,
    )

    # ------------------------------------------------------------------
    # Check for existing lyrics.
    # ------------------------------------------------------------------

    print()
    print("Checking LRCLIB for existing lyrics...")

    try:
        existing = find_existing(
            api,
            artist,
            title,
            album,
            duration,
        )

    except RateLimitError:
        print(
            "ERROR: LRCLIB rate-limited the lookup.",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(
            f"ERROR: LRCLIB lookup failed:\n  {exc}",
            file=sys.stderr,
        )
        return 1

    if existing is not None:
        display_existing(existing)

        if not confirm_update(existing):
            print("Update cancelled.")
            return 0

        operation = "Updating existing lyrics"

    else:
        operation = "Publishing new lyrics"

    # ------------------------------------------------------------------
    # Publish.
    # ------------------------------------------------------------------

    print()
    print(operation + "...")
    print()
    print("Requesting LRCLIB proof-of-work challenge...")
    print("Solving challenge using all available CPU cores...")
    print(
        "(Difficulty varies depending on LRCLIB's current "
        "challenge.)"
    )
    print()

    try:
        result = publish(
            api,
            artist,
            title,
            album,
            duration,
            plain_lyrics,
            synced_lyrics,
        )

    except IncorrectPublishTokenError:
        print(
            "ERROR: LRCLIB rejected the proof-of-work token.",
            file=sys.stderr,
        )
        print(
            "The challenge may have expired or the server's "
            "challenge format may have changed.",
            file=sys.stderr,
        )
        return 1

    except RateLimitError:
        print(
            "ERROR: LRCLIB rate-limited the publish request.",
            file=sys.stderr,
        )
        return 1

    except ServerError as exc:
        print(
            f"ERROR: LRCLIB server error:\n  {exc}",
            file=sys.stderr,
        )
        return 1

    except APIError as exc:
        print(
            f"ERROR: LRCLIB API error:\n  {exc}",
            file=sys.stderr,
        )
        return 1

    except requests.RequestException as exc:
        print(
            f"ERROR: Network request failed:\n  {exc}",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(
            f"ERROR: Upload failed:\n  {exc}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Verify.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("PUBLISH REQUEST ACCEPTED")
    print("=" * 70)

    if isinstance(result, dict):
        if "id" in result:
            print(f"LRCLIB ID: {result['id']}")

    # Give LRCLIB a moment before querying the record again.
    import time
    time.sleep(0.5)

    verified = verify_upload(
        api,
        artist,
        title,
        album,
        duration,
        synced_lyrics,
    )

    print()

    if verified:
        if existing is not None:
            print("Existing lyrics successfully updated.")
            print("A new LRCLIB revision was created.")
        else:
            print("Lyrics successfully published to LRCLIB.")
    else:
        print(
            "The publish request was accepted, but verification "
            "could not confirm the exact uploaded lyrics."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())