"""Extracts embedded cover art from an audio/video file's own metadata
tags, via `mutagen` (already an installed dependency -- requirements.txt
has listed it since early in this project, but nothing ever actually
called it until this module).

Only ever invoked when file_discovery.find_companions found no
.jpg/.jpeg companion next to the resolved audio source (see
song_input.resolve_song_folder) -- an embedded picture is a fallback, not
a replacement for a hand-placed cover file.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from . import config


def _sniff_image_ext(data: bytes) -> Optional[str]:
    """Real file type from magic bytes, not whatever MIME type the
    container's own tag claims (tags lie/are stale often enough that this
    project's own convention elsewhere is to verify content, not trust a
    claimed type -- see file_discovery._looks_like_musicxml)."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return None


def _from_id3(tags) -> Optional[bytes]:
    apics = tags.getall("APIC") if hasattr(tags, "getall") else []
    return apics[0].data if apics else None


def _from_mp4(tags) -> Optional[bytes]:
    covers = tags.get("covr") if tags else None
    return bytes(covers[0]) if covers else None


def _from_flac_native(mutagen_file) -> Optional[bytes]:
    pictures = getattr(mutagen_file, "pictures", None)
    return pictures[0].data if pictures else None


def _from_vorbis_comment_block(tags) -> Optional[bytes]:
    """OGG/Opus store embedded art as a base64-encoded FLAC-style Picture
    block under the 'metadata_block_picture' vorbis comment key -- not
    exposed as a `.pictures` list the way native FLAC files are."""
    if not tags or "metadata_block_picture" not in tags:
        return None
    try:
        from mutagen.flac import Picture
        raw = base64.b64decode(tags["metadata_block_picture"][0])
        return Picture(raw).data
    except Exception:
        return None


def _extract_raw_picture(audio_path: Path) -> Optional[bytes]:
    try:
        import mutagen
    except ImportError:
        return None

    try:
        f = mutagen.File(audio_path)
    except Exception:
        return None
    if f is None:
        return None

    for extractor in (
        lambda: _from_id3(f.tags),
        lambda: _from_mp4(f.tags),
        lambda: _from_flac_native(f),
        lambda: _from_vorbis_comment_block(f.tags),
    ):
        try:
            data = extractor()
        except Exception:
            data = None
        if data:
            return data
    return None


def extract_embedded_cover(audio_path: Path, dest_dir: Path) -> Optional[Path]:
    """Tries every embedded-picture convention this project's supported
    formats use (ID3 APIC for mp3, MP4 'covr' atom, FLAC's native picture
    list, and OGG/Opus's base64 vorbis-comment picture block), sniffs the
    real image type from magic bytes, and writes
    dest_dir/'<audio stem><EMBEDDED_COVER_TAG_SUFFIX>.<ext>' -- reusing
    file_discovery.find_companions' own "[CO]" tag convention so a later
    find_companions call over the same folder reads this identically to a
    hand-placed cover file. Returns None (never raises) on any failure:
    unsupported container, no embedded picture, corrupt tag data, unknown
    image format, mutagen not installed."""
    data = _extract_raw_picture(Path(audio_path))
    if not data:
        return None

    ext = _sniff_image_ext(data)
    if ext is None:
        return None

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{Path(audio_path).stem}{config.EMBEDDED_COVER_TAG_SUFFIX}{ext}"
    out_path.write_bytes(data)
    return out_path
