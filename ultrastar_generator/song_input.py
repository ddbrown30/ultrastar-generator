"""Orchestrates folder-based input resolution: figures out which file in a
song folder is the primary audio/video source (file_discovery.
resolve_primary_source), extracts real decodable audio from it when
needed (media_extract.py), finds companions (file_discovery.
find_companions), and falls back to embedded cover art (cover_extract.py)
when no cover image was found. See CLAUDE.md for the full feature writeup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .cover_extract import extract_embedded_cover
from .file_discovery import find_companions, resolve_primary_source, NoAudioSourceFoundError
from .media_extract import extract_audio_track, has_audio_stream


@dataclass
class ResolvedInput:
    primary_source_kind: str            # "audio" | "video_as_audio" | "avi_extract"
    output_mp3_source: Path             # copied to output_dir, referenced as #MP3
    output_video_source: Optional[Path] # copied to output_dir, referenced as #VIDEO
    analysis_audio: Path                # real decodable audio fed to Demucs/pass1/WhisperX
    cover: Optional[Path] = None
    background: Optional[Path] = None
    musicxml: List[Path] = field(default_factory=list)
    videogap_applicable: bool = True    # False when video and audio are the same
                                          # file or one was extracted directly from
                                          # the other -- correlating either against
                                          # itself would be a degenerate no-op, not
                                          # a real VIDEOGAP measurement.
    notes: List[str] = field(default_factory=list)  # human-readable decisions, for the log


def resolve_song_folder(input_dir: Path, work_dir: Path, *,
                         audio_file_override: Optional[str] = None) -> ResolvedInput:
    """Resolves everything about a song folder needed to run the pipeline.
    Never mutates input_dir. Any file this function synthesizes (an
    extracted analysis wav, an avi's extracted standalone mp3, an
    extracted embedded cover) is cached under work_dir/extracted/, keyed
    by name so re-runs skip re-extraction -- same check-then-extract
    idiom separation.isolate_vocals already uses for Demucs's own cache.
    """
    input_dir = Path(input_dir)
    work_dir = Path(work_dir)
    extracted_dir = work_dir / "extracted"

    primary_path, kind = resolve_primary_source(input_dir, audio_file_override=audio_file_override)
    notes: List[str] = []

    if kind == "audio":
        companions = find_companions(primary_path)
        resolved = ResolvedInput(
            primary_source_kind=kind,
            output_mp3_source=primary_path,
            output_video_source=companions.video,
            analysis_audio=primary_path,
            cover=companions.cover, background=companions.background, musicxml=companions.musicxml,
            videogap_applicable=True,
        )

    elif kind == "video_as_audio":
        companions = find_companions(primary_path)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        analysis_wav = extracted_dir / f"{primary_path.stem}.wav"
        if not analysis_wav.exists():
            if not extract_audio_track(primary_path, analysis_wav, as_mp3=False):
                raise NoAudioSourceFoundError(
                    f"Could not extract an audio track from {primary_path.name} "
                    f"(no audio stream, or ffmpeg unavailable)."
                )
            notes.append(f"Extracted analysis audio from {primary_path.name} -> {analysis_wav.name}")
        # Same file serves as both #MP3 and #VIDEO -- config.VIDEO_DIRECT_AUDIO_EXTS
        # (.mp4/.mpg/.mpeg) are all confirmed usable as #MP3 directly in
        # UltraStar Deluxe (.avi is NOT -- that's the separate "avi_extract"
        # branch below, which extracts a real standalone mp3 first).
        resolved = ResolvedInput(
            primary_source_kind=kind,
            output_mp3_source=primary_path,
            output_video_source=primary_path,
            analysis_audio=analysis_wav,
            cover=companions.cover, background=companions.background, musicxml=companions.musicxml,
            videogap_applicable=False,  # video IS the audio -- nothing to correlate against
            notes=notes,
        )

    elif kind == "avi_extract":
        if not has_audio_stream(primary_path):
            raise NoAudioSourceFoundError(
                f"{primary_path.name} has no audio track -- nothing to generate a song from."
            )
        companions = find_companions(primary_path)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        extracted_mp3 = extracted_dir / f"{primary_path.stem}.mp3"
        if not extracted_mp3.exists():
            if not extract_audio_track(primary_path, extracted_mp3, as_mp3=True):
                raise NoAudioSourceFoundError(
                    f"ffmpeg couldn't extract an mp3 from {primary_path.name}'s audio track "
                    f"(check your ffmpeg build includes libmp3lame)."
                )
            notes.append(f"Extracted {extracted_mp3.name} from {primary_path.name}'s audio track")
        resolved = ResolvedInput(
            primary_source_kind=kind,
            output_mp3_source=extracted_mp3,
            output_video_source=primary_path,      # the avi itself becomes #VIDEO
            analysis_audio=extracted_mp3,           # already real, decodable audio
            cover=companions.cover, background=companions.background, musicxml=companions.musicxml,
            videogap_applicable=False,  # the mp3 came directly from this avi's own audio --
            notes=notes,                # correlating them would be a trimmed copy of itself
        )

    else:
        raise AssertionError(f"unreachable: unknown resolve_primary_source kind {kind!r}")

    if resolved.cover is None:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        embedded = extract_embedded_cover(resolved.analysis_audio, extracted_dir)
        if embedded is not None:
            resolved.cover = embedded
            resolved.notes.append(f"Extracted embedded cover art -> {embedded.name}")

    return resolved
