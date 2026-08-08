"""Copies the companion files a song's output .txt actually references
(audio, video, cover, background) into the output folder, per this
project's requirement that input and output folders differ -- an output
folder needs to be self-contained, not depend on files still sitting in
the input folder.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StagedCompanions:
    mp3: str
    video: Optional[str]
    cover: Optional[str]
    background: Optional[str]


def _copy_if_needed(src: Path, output_dir: Path) -> str:
    """Copies src into output_dir under its own basename unless a file is
    already there at that path (covers both "input_dir == output_dir",
    not allowed elsewhere but harmless to handle here too, and "this file
    was already synthesized directly under output_dir"). Returns the bare
    filename, matching models.Song's own string-filename field
    convention."""
    dst = output_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return src.name


def stage_companions_to_output(output_dir: Path, *, mp3_src: Path,
                                video_src: Optional[Path] = None,
                                cover_src: Optional[Path] = None,
                                background_src: Optional[Path] = None) -> StagedCompanions:
    """Copies each given source into output_dir. mp3_src/video_src may be
    the identical path (the mp4-as-audio case) -- copied once, both roles
    reference the same output filename. cover_src and background_src may
    also be identical to each other (find_companions' own "one untagged
    image serves both roles" convention) -- same single-copy handling."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mp3_name = _copy_if_needed(mp3_src, output_dir)
    video_name = (mp3_name if video_src == mp3_src
                  else _copy_if_needed(video_src, output_dir) if video_src else None)
    cover_name = _copy_if_needed(cover_src, output_dir) if cover_src else None
    background_name = (cover_name if background_src == cover_src
                        else _copy_if_needed(background_src, output_dir) if background_src else None)

    return StagedCompanions(mp3=mp3_name, video=video_name, cover=cover_name, background=background_name)
