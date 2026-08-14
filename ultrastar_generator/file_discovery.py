"""Finds companion files (video, cover, background) next to an audio file.

Rules implemented (per the user's spec):
  * Audio files are named "<Artist> - <Title>.<ext>" (mp3/ogg/oga/m4a).
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
from typing import List, Optional, Tuple

from . import config


class AmbiguousInputError(ValueError):
    """More than one candidate file could plausibly be the song's primary
    audio/video source, and there's no principled way to auto-pick one."""


class NoAudioSourceFoundError(ValueError):
    """The input folder has no real audio file, no .mp4, and no .avi with
    an audio track -- nothing usable was found at all."""


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
    if not videos:
        # No basename-matched video -- if there's exactly one video file
        # in the folder at all, it's unambiguous even though its name
        # doesn't relate to the audio file's own name (real case: a
        # SingStar-style rip where the audio is "music.ogg" but there's a
        # single "video.mpg" sitting right next to it -- same reasoning
        # as file_discovery.resolve_artist_title's folder-name fallback:
        # a strict naming-convention match failing doesn't mean there's
        # nothing to find, just that this file predates/ignores the
        # convention).
        untagged_videos = [p for p in candidates if p.suffix.lower() in config.VIDEO_EXTS]
        if len(untagged_videos) == 1:
            videos = untagged_videos
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
    else:
        # No basename-matched/tagged image at all -- same single-
        # unambiguous-candidate fallback as video, above.
        untagged_images = [p for p in candidates if p.suffix.lower() in config.IMAGE_EXTS]
        if len(untagged_images) == 1:
            result.cover = untagged_images[0]
            result.background = untagged_images[0]

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


def resolve_primary_source(input_dir: Path, audio_file_override: Optional[str] = None) -> Tuple[Path, str]:
    """Figures out which file in a song folder is the primary audio/video
    source, and how it should be treated. Returns (path, kind), kind one of:
      "audio"          -- a real audio file (config.AUDIO_EXTS) was found.
      "video_as_audio" -- no real audio file, but exactly one video file
                           UltraStar Deluxe can use directly as #MP3 was
                           (config.VIDEO_DIRECT_AUDIO_EXTS: .mp4/.mpg/
                           .mpeg) -- it'll serve as BOTH the audio source
                           and #VIDEO.
      "avi_extract"    -- no real audio file and no direct-audio video,
                           but exactly one .avi was -- UltraStar Deluxe
                           can't use .avi as #MP3 directly, so its audio
                           track must be extracted into a real standalone
                           file (see media_extract.py).

    `audio_file_override` (a bare filename within input_dir) wins outright
    over all of the above, always resolving to kind "audio" -- this is the
    escape hatch for a folder with more than one real audio file, which
    this function otherwise refuses to guess between.

    Raises AmbiguousInputError if more than one candidate exists at
    whichever tier is reached (naming the candidates), or
    NoAudioSourceFoundError if nothing usable exists at any tier.
    """
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
    """Splits on the FIRST " - " occurrence, since artist or title names
    could themselves contain hyphens without surrounding spaces (e.g.
    "Jean-Luc")."""
    if " - " not in name:
        raise ValueError(
            f'Could not parse "<Artist> - <Title>" from {name!r}. '
            f"Pass --artist and --title explicitly instead."
        )
    artist, title = name.split(" - ", 1)
    return artist.strip(), title.strip()


def parse_artist_title(audio_path: Path) -> tuple[str, str]:
    """Parses "<Artist> - <Title>.<ext>" into (artist, title) from the
    AUDIO FILE's own name."""
    return _split_artist_title(Path(audio_path).stem)


def resolve_artist_title(audio_path: Path, input_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Input is folder-based (see module docstring), so the INPUT FOLDER's
    own name is the sole, authoritative "<Artist> - <Title>" source -- a
    file inside can be named anything at all (a ripped/downloaded file
    keeping a generic name like "music.ogg", multiple companion files with
    unrelated names, etc; real confirmed case: this project's own
    "Beauty And The Beast - Beauty And The Beast" SingStar-rip test song).
    `audio_path` is accepted but intentionally unused -- kept so call
    sites don't need special-casing depending on which source they have
    handy. Uses the folder's raw `.name` (not `.stem`, which would
    wrongly treat a dot inside a folder name as a file extension to
    strip, since folders don't have real extensions). Returns (None,
    None) if the folder name doesn't parse -- never raises, unlike
    `parse_artist_title` itself."""
    del audio_path
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
    """Classifies a word's letters-only casing so `headline_case` knows
    whether it's safe to touch: "simple" (all-lowercase, or exactly one
    leading capital + the rest lowercase -- including single-letter
    words, which are inherently ambiguous either way) can be safely
    re-cased; "upper" (2+ letters, ALL of them uppercase -- an acronym or
    stylized name, e.g. "AND") and "mixed" (any other pattern, e.g.
    "KPop", "McDonald") must be left completely alone since we can't
    safely guess whether that capitalization is intentional; "none" (no
    letters at all, e.g. "&") is never touched either."""
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
    """Strips characters Windows forbids in a file/folder name (e.g. the
    "?" in a real title like "Shall We Dance?") plus trailing dots/spaces
    (also illegal as a Windows filename's last character) -- for use ONLY
    when turning an artist/title into an actual filesystem path component.
    Never apply this to a #ARTIST/#TITLE tag's own value, which should
    keep the real, unmodified text."""
    out = "".join(c for c in text if c not in _WINDOWS_ILLEGAL_FILENAME_CHARS)
    return out.rstrip(" .")


def headline_case(text: str) -> str:
    """Title-cases `text` for display in output folder/file names (e.g.
    "Beauty And The Beast" -> "Beauty and the Beast") -- minor words
    (articles/conjunctions/short prepositions, see `_MINOR_WORDS`) are
    lowercased unless they're the first or last word. Deliberately NOT
    aggressive: only words in an unambiguous "simple" case shape (see
    `_word_case_shape`) are ever touched -- an ALL CAPS word (e.g. "AND")
    or one with unusual internal capitalization (e.g. "KPop") is left
    completely untouched, since forcing it into "and"/"Kpop" would
    destroy what's very likely intentional stylization. Splits on
    whitespace only; any punctuation attached to a word (e.g. "Beast,",
    "Don't") stays attached and untouched."""
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
