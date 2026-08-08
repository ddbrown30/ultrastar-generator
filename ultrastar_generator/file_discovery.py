"""Finds companion files (video, cover, background) next to an audio file.

Rules implemented (per the user's spec):
  * Audio files are named "<Artist> - <Title>.<ext>" (mp3/ogg/oga).
  * If an .avi/.mp4 exists with the SAME base name -> #VIDEO.
  * If exactly one .jpg/.jpeg exists with the same base name -> used for
    both #COVER and #BACKGROUND.
  * If multiple images exist, ones with "[CO]" in the name -> #COVER,
    ones with "[BG]" in the name -> #BACKGROUND.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import config


@dataclass
class Companions:
    video: Optional[Path] = None
    cover: Optional[Path] = None
    background: Optional[Path] = None
    musicxml: List[Path] = field(default_factory=list)  # .mxl/.musicxml/.xml
                                    # reference files for pass 4 -- unlike
                                    # video/cover, these are matched by
                                    # EXTENSION ALONE, not basename: a
                                    # downloaded MuseScore file keeps
                                    # whatever name the source gave it
                                    # (e.g. "beauty-and-the-beast.mxl"),
                                    # never "<Artist> - <Title>.mxl".


def _same_base(candidate: Path, base_stem: str) -> bool:
    """True if candidate's name starts with base_stem (allowing an
    optional " [TAG]" suffix before the extension), case-insensitively."""
    stem = candidate.stem
    if stem.lower() == base_stem.lower():
        return True
    # allow "<base> [CO]", "<base>[CO]", "<base> [BG]" etc.
    pattern = re.escape(base_stem) + r"\s*\[(CO|BG)\]$"
    return re.match(pattern, stem, flags=re.IGNORECASE) is not None


def _looks_like_musicxml(path: Path) -> bool:
    """Cheap content sniff for a bare ".xml" file: real MusicXML declares
    a "score-partwise"/"score-timewise" root somewhere near the top of
    the file. Reading raw text (not parsing) is enough and avoids paying
    a full XML-parse cost on every random .xml a song folder happens to
    contain."""
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    except OSError:
        return False
    return "score-partwise" in head or "score-timewise" in head


def find_companions(audio_path: Path) -> Companions:
    audio_path = Path(audio_path)
    base_stem = audio_path.stem
    directory = audio_path.parent
    result = Companions()

    candidates = [p for p in directory.iterdir() if p.is_file() and p != audio_path]

    # --- video ---------------------------------------------------------
    videos = [
        p for p in candidates
        if p.suffix.lower() in config.VIDEO_EXTS and _same_base(p, base_stem)
    ]
    if videos:
        # Prefer exact-stem match over a tagged one, if both somehow exist.
        videos.sort(key=lambda p: p.stem.lower() != base_stem.lower())
        result.video = videos[0]

    # --- images ----------------------------------------------------------
    images = [
        p for p in candidates
        if p.suffix.lower() in config.IMAGE_EXTS and _same_base(p, base_stem)
    ]

    tagged_co = [p for p in images if re.search(r"\[CO\]", p.stem, re.IGNORECASE)]
    tagged_bg = [p for p in images if re.search(r"\[BG\]", p.stem, re.IGNORECASE)]

    if tagged_co or tagged_bg:
        if tagged_co:
            result.cover = tagged_co[0]
        if tagged_bg:
            result.background = tagged_bg[0]
    elif len(images) == 1:
        result.cover = images[0]
        result.background = images[0]
    elif len(images) > 1:
        # Multiple untagged images: no reliable way to pick, so use the
        # first for both and let the user know at the call site.
        result.cover = images[0]
        result.background = images[0]

    # --- MusicXML reference (pass 4) --------------------------------------
    # ".mxl"/".musicxml" are unambiguous. Bare ".xml" is NOT -- e.g. these
    # SingStar rips ship their own "notes.xml" (a different, proprietary
    # format, root tag "{http://www.singstargame.com}MELODY") right next
    # to the real audio/lyrics, which crashed music21.converter.parse
    # when trusted by extension alone. Content-sniff any bare ".xml" so
    # only files that actually look like MusicXML get picked up.
    xml_candidates = [p for p in candidates if p.suffix.lower() in (".mxl", ".musicxml")]
    xml_candidates += [p for p in candidates if p.suffix.lower() == ".xml" and _looks_like_musicxml(p)]
    result.musicxml = sorted(xml_candidates, key=lambda p: p.name.lower())  # deterministic order

    return result


def parse_artist_title(audio_path: Path) -> tuple[str, str]:
    """Parses "<Artist> - <Title>.<ext>" into (artist, title).

    Splits on the FIRST " - " occurrence, since artist or title names could
    themselves contain hyphens without surrounding spaces (e.g. "Jean-Luc").
    """
    stem = Path(audio_path).stem
    if " - " not in stem:
        raise ValueError(
            f'Could not parse "<Artist> - <Title>" from filename: {stem!r}. '
            f"Pass --artist and --title explicitly instead."
        )
    artist, title = stem.split(" - ", 1)
    return artist.strip(), title.strip()
