"""Copies the companion files a song's output .txt actually references
(audio, video, cover, background) into the output folder, per this
project's requirement that input and output folders differ -- an output
folder needs to be self-contained, not depend on files still sitting in
the input folder.

Each file is also renamed to "<Artist> - <Title>[.ext]" (images keep
their own "[CO]"/"[BG]" tag) regardless of what it was called in the
input folder -- folder-based input means individual files can be named
anything at all (a ripped/downloaded file's generic name, an image with
no relation to the song), so the output folder is the one place this
project guarantees the naming convention actually holds. Not used by
realign.py -- that mode only ever writes a single .txt back next to the
existing file, never a self-contained output folder.

A separately-staged video companion (i.e. video_src != mp3_src -- a real
standalone #MP3 already covers the audio, whether that's a genuinely
separate audio file or one extracted directly from this same video's own
audio track) has its own audio track stripped before being copied, via
ffmpeg stream-copy (no re-encode) -- it's always redundant with the real
#MP3 and only wastes output size. Falls back to a plain copy if ffmpeg
is unavailable/fails, or the video has no audio track to begin with.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .file_discovery import sanitize_filename
from .media_extract import has_audio_stream, strip_audio_track


@dataclass
class StagedCompanions:
    mp3: str
    video: Optional[str]
    cover: Optional[str]
    background: Optional[str]


def _copy_as(src: Path, output_dir: Path, target_name: str) -> str:
    """Copies src into output_dir under target_name unless a file is
    already there at that exact path (covers both "input_dir ==
    output_dir", not allowed elsewhere but harmless to handle here too,
    and "this file was already synthesized directly under output_dir").
    Returns the bare filename, matching models.Song's own string-filename
    field convention."""
    dst = output_dir / target_name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return target_name


def _copy_video_stripped(src: Path, output_dir: Path, target_name: str) -> str:
    """Same as _copy_as, but strips src's own audio track first (see this
    module's own docstring for why) whenever one exists. Falls back to a
    plain copy if ffmpeg is unavailable, the strip fails, or src has no
    audio track to begin with -- losing this size optimization is better
    than failing the whole staging step."""
    dst = output_dir / target_name
    if src.resolve() == dst.resolve():
        return target_name
    if has_audio_stream(src) and strip_audio_track(src, dst):
        return target_name
    shutil.copy2(src, dst)
    return target_name


def stage_companions_to_output(output_dir: Path, artist: str, title: str, *, mp3_src: Path,
                                video_src: Optional[Path] = None,
                                cover_src: Optional[Path] = None,
                                background_src: Optional[Path] = None) -> StagedCompanions:
    """Copies each given source into output_dir, renamed to
    "<artist> - <title>[.ext]". mp3_src/video_src may be the identical
    path (the mp4-as-audio case) -- copied once, both roles reference the
    same output filename. cover_src and background_src may also be
    identical to each other (find_companions' own "one untagged image
    serves both roles" convention) -- same single-copy handling, and no
    "[CO]"/"[BG]" tag is added since there's nothing to disambiguate; a
    genuinely separate cover/background pair keeps its own tag instead."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = sanitize_filename(f"{artist} - {title}")

    mp3_name = _copy_as(mp3_src, output_dir, f"{base}{mp3_src.suffix}")
    if video_src == mp3_src:
        video_name = mp3_name
    elif video_src:
        # A separately-staged video's own audio is always redundant here --
        # mp3_src already covers it (either a genuinely separate audio
        # file, or extracted directly from this same video) -- so strip it
        # to save output size (see module docstring).
        video_name = _copy_video_stripped(video_src, output_dir, f"{base}{video_src.suffix}")
    else:
        video_name = None

    if cover_src and cover_src == background_src:
        cover_name = _copy_as(cover_src, output_dir, f"{base}{cover_src.suffix}")
        background_name = cover_name
    else:
        cover_name = (_copy_as(cover_src, output_dir, f"{base} [CO]{cover_src.suffix}")
                      if cover_src else None)
        background_name = (_copy_as(background_src, output_dir, f"{base} [BG]{background_src.suffix}")
                            if background_src else None)

    return StagedCompanions(mp3=mp3_name, video=video_name, cover=cover_name, background=background_name)
