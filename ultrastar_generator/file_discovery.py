"""Finds companion files (video, cover, background) next to an audio file."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from . import config


class AmbiguousInputError(ValueError):
    """More than one candidate could be the song's primary audio/video source."""


class NoAudioSourceFoundError(ValueError):
    """No usable audio source found in the input folder."""


@dataclass
class Companions:
    video: Optional[Path] = None
    cover: Optional[Path] = None
    background: Optional[Path] = None
    musicxml: List[Path] = field(default_factory=list)  # matched by extension, not basename


def _same_base(candidate: Path, base_stem: str) -> bool:
    """True if candidate's name matches base_stem, optionally with a " [CO]"/" [BG]" suffix."""
    stem = candidate.stem
    if stem.lower() == base_stem.lower():
        return True
    pattern = re.escape(base_stem) + r"\s*\[(CO|BG)\]$"
    return re.match(pattern, stem, flags=re.IGNORECASE) is not None


def _looks_like_musicxml(path: Path) -> bool:
    """Content-sniffs a bare ".xml" file for a MusicXML root tag."""
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
    if not videos:
        # Fall back to a single unrelated-named video if it's the only one.
        untagged_videos = [p for p in candidates if p.suffix.lower() in config.VIDEO_EXTS]
        if len(untagged_videos) == 1:
            videos = untagged_videos
    if videos:
        videos.sort(key=lambda p: p.stem.lower() != base_stem.lower())  # exact stem match first
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
        # No reliable way to pick among untagged images; use the first for both.
        result.cover = images[0]
        result.background = images[0]
    else:
        # Fall back to a single unrelated-named image if it's the only one.
        untagged_images = [p for p in candidates if p.suffix.lower() in config.IMAGE_EXTS]
        if len(untagged_images) == 1:
            result.cover = untagged_images[0]
            result.background = untagged_images[0]

    # --- MusicXML reference (pass 4) --------------------------------------
    # Bare ".xml" is content-sniffed since some games ship an unrelated "notes.xml".
    xml_candidates = [p for p in candidates if p.suffix.lower() in (".mxl", ".musicxml")]
    xml_candidates += [p for p in candidates if p.suffix.lower() == ".xml" and _looks_like_musicxml(p)]
    result.musicxml = sorted(xml_candidates, key=lambda p: p.name.lower())

    return result


def resolve_primary_source(input_dir: Path, audio_file_override: Optional[str] = None) -> Tuple[Path, str]:
    """Finds the primary audio/video source in a song folder. Returns (path, kind):
    "audio" (real audio file), "video_as_audio" (single .mp4/.mpg/.mpeg, serves as both
    #MP3 and #VIDEO), or "avi_extract" (single .avi, audio track needs extracting).
    `audio_file_override` picks a specific file, always resolving to "audio".
    Raises AmbiguousInputError on multiple candidates, NoAudioSourceFoundError on none."""
    input_dir = Path(input_dir)

    if audio_file_override:
        override_path = input_dir / audio_file_override
        if not override_path.is_file():
            raise NoAudioSourceFoundError(f"--audio-file {audio_file_override!r} not found in {input_dir}")
        return override_path, "audio"

    candidates = [p for p in input_dir.iterdir() if p.is_file()]

    audio_files = sorted(p for p in candidates if p.suffix.lower() in config.AUDIO_EXTS)
    if len(audio_files) == 1:
        return audio_files[0], "audio"
    if len(audio_files) > 1:
        names = ", ".join(p.name for p in audio_files)
        raise AmbiguousInputError(
            f"Found more than one audio file in {input_dir}: {names}. "
            f"Pass --audio-file <name> to pick which one to use."
        )

    direct_audio_video_files = sorted(
        p for p in candidates if p.suffix.lower() in config.VIDEO_DIRECT_AUDIO_EXTS)
    if len(direct_audio_video_files) == 1:
        return direct_audio_video_files[0], "video_as_audio"
    if len(direct_audio_video_files) > 1:
        names = ", ".join(p.name for p in direct_audio_video_files)
        raise AmbiguousInputError(
            f"No audio file, but found more than one usable video file in {input_dir}: {names}. "
            f"Pass --audio-file <name> to pick which one to use."
        )

    avi_files = sorted(p for p in candidates if p.suffix.lower() == ".avi")
    if len(avi_files) == 1:
        return avi_files[0], "avi_extract"
    if len(avi_files) > 1:
        names = ", ".join(p.name for p in avi_files)
        raise AmbiguousInputError(
            f"No audio file or usable video, but found more than one .avi in {input_dir}: {names}. "
            f"Pass --audio-file <name> to pick which one to use."
        )

    raise NoAudioSourceFoundError(
        f"No usable audio source found in {input_dir} -- expected one of "
        f"{config.AUDIO_EXTS}, or a single {config.VIDEO_DIRECT_AUDIO_EXTS} or .avi file."
    )


def _split_artist_title(name: str) -> tuple[str, str]:
    """Splits on the FIRST " - " occurrence."""
    if " - " not in name:
        raise ValueError(
            f'Could not parse "<Artist> - <Title>" from {name!r}. '
            f"Pass --artist and --title explicitly instead."
        )
    artist, title = name.split(" - ", 1)
    return artist.strip(), title.strip()


def parse_artist_title(audio_path: Path) -> tuple[str, str]:
    """Parses "<Artist> - <Title>.<ext>" into (artist, title) from the audio file's name."""
    return _split_artist_title(Path(audio_path).stem)


def resolve_artist_title(audio_path: Path, input_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Gets the artist and title from the input folder name, formatted as "<Artist> - <Title>"."""
    del audio_path  # unused; kept so call sites don't need special-casing
    try:
        return _split_artist_title(Path(input_dir).name)
    except ValueError:
        return None, None


_MINOR_WORDS = {
    "a", "an", "the",
    "and", "but", "or", "nor", "for", "so", "yet",
    "as", "at", "by", "in", "into", "of", "off", "on", "onto", "out", "over",
    "per", "to", "up", "via", "with", "from",
}


def _word_case_shape(word: str) -> str:
    """Classifies a word's casing: "simple" (lowercase or Capitalized, safe to re-case),
    "upper" (all-caps acronym), "mixed" (e.g. "KPop"), or "none" (no letters) -- only
    "simple" words are safe to touch."""
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "upper" if len(letters) > 1 else "simple"
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "simple"
    if all(c.islower() for c in letters):
        return "simple"
    return "mixed"


def _capitalize_first_letter(word: str) -> str:
    for i, c in enumerate(word):
        if c.isalpha():
            return word[:i] + c.upper() + word[i + 1:]
    return word


_WINDOWS_ILLEGAL_FILENAME_CHARS = '<>:"/\\|?*'


def sanitize_filename(text: str) -> str:
    """Strips Windows-illegal filename characters and trailing dots/spaces. Only for
    filesystem path components, never for a #ARTIST/#TITLE tag's own value."""
    out = "".join(c for c in text if c not in _WINDOWS_ILLEGAL_FILENAME_CHARS)
    return out.rstrip(" .")


def headline_case(text: str) -> str:
    """Title-cases text, lowercasing minor words (see `_MINOR_WORDS`) unless first/last.
    Only touches "simple"-shaped words (see `_word_case_shape`); leaves stylized
    capitalization (e.g. "KPop", "AND") alone."""
    words = text.split(" ")
    n = len(words)
    out = []
    for i, word in enumerate(words):
        if _word_case_shape(word) != "simple":
            out.append(word)
            continue
        lowered = word.lower()
        letters_only = "".join(c for c in lowered if c.isalpha())
        if i != 0 and i != n - 1 and letters_only in _MINOR_WORDS:
            out.append(lowered)
        else:
            out.append(_capitalize_first_letter(lowered))
    return " ".join(out)
