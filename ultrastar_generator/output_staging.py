"""Copies a song's companion files (audio, video, cover, background) into the self-contained output folder, renamed to "<Artist> - <Title>[.ext]" (images get a "[CO]"/"[BG]" tag). Not used by realign.py. A separately-staged video has its own audio track stripped (ffmpeg stream-copy) since the real #MP3 already covers it."""

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
    """Copies src into output_dir as target_name (skipped if already there) and returns the bare filename."""
    dst = output_dir / target_name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return target_name


def _copy_video_stripped(src: Path, output_dir: Path, target_name: str) -> str:
    """Same as _copy_as, but strips src's audio track first if it has one; falls back to a plain copy on any failure."""
    dst = output_dir / target_name
    if src.resolve() == dst.resolve():
        return target_name
    if has_audio_stream(src) and strip_audio_track(src, dst):
        return target_name
    shutil.copy2(src, dst)
    return target_name


def _fix_unspaced_tag_file(output_dir: Path, base: str, tag: str) -> None:
    """Renames a stale unspaced "<base>[CO/BG].<ext>" in output_dir to the spaced "<base> [CO/BG].<ext>" form."""
    unspaced_stem = f"{base}[{tag}]"
    spaced_stem = f"{base} [{tag}]"
    for candidate in output_dir.iterdir():
        if candidate.is_file() and candidate.stem == unspaced_stem:
            target = output_dir / f"{spaced_stem}{candidate.suffix}"
            if candidate.resolve() != target.resolve():
                shutil.move(str(candidate), str(target))


def stage_companions_to_output(output_dir: Path, artist: str, title: str, *, mp3_src: Path,
                                video_src: Optional[Path] = None,
                                cover_src: Optional[Path] = None,
                                background_src: Optional[Path] = None) -> StagedCompanions:
    """Copies each given source into output_dir, renamed to "<artist> - <title>[.ext]". mp3_src/video_src (and cover_src/background_src) may be the identical path, in which case they're copied once and share the same output filename."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = sanitize_filename(f"{artist} - {title}")

    mp3_name = _copy_as(mp3_src, output_dir, f"{base}{mp3_src.suffix}")
    if video_src == mp3_src:
        video_name = mp3_name
    elif video_src:
        video_name = _copy_video_stripped(video_src, output_dir, f"{base}{video_src.suffix}")
    else:
        video_name = None

    _fix_unspaced_tag_file(output_dir, base, "CO")
    _fix_unspaced_tag_file(output_dir, base, "BG")

    if cover_src and cover_src == background_src:
        cover_name = _copy_as(cover_src, output_dir, f"{base}{cover_src.suffix}")
        background_name = cover_name
    else:
        cover_name = (_copy_as(cover_src, output_dir, f"{base} [CO]{cover_src.suffix}")
                      if cover_src else None)
        background_name = (_copy_as(background_src, output_dir, f"{base} [BG]{background_src.suffix}")
                            if background_src else None)

    return StagedCompanions(mp3=mp3_name, video=video_name, cover=cover_name, background=background_name)
