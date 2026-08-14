"""Downloads a YouTube video as the primary source for a song folder
(feature 7), via `yt-dlp` -- an optional dependency, only ever imported
when `--youtube-url` is actually used, same "gracefully degrade if
missing" convention as `whisperx`/`rmvpe_onnx` elsewhere in this project.

Deliberately does NOT name the downloaded file from the video's own
title -- YouTube titles aren't a reliable "Artist - Title" source, which
is exactly why `--youtube-url` requires `--artist`/`--title` explicitly
(enforced in main.py, not here).

Also downloads the video's thumbnail and converts it to
'youtube_download [CO].jpg' -- the " [CO]" tag matches file_discovery.
find_companions' own existing cover-detection convention by construction
(it matches any "<audio file's own stem> [CO]" image, and the audio/video
file here is always named "youtube_download.<ext>"), so the thumbnail is
picked up as the cover with ZERO new code in file_discovery.py/
song_input.py. Best-effort: a missing/failed thumbnail never fails the
overall download, since the real audio/video content is what matters.
"""

from __future__ import annotations

from pathlib import Path


class YoutubeDownloadError(RuntimeError):
    """yt-dlp isn't installed, the download failed (network, private/
    removed/geo-blocked video, etc.), or it "succeeded" but the expected
    output file isn't where it should be."""


def download_youtube_source(url: str, dest_dir: Path, *, audio_only: bool) -> Path:
    """Downloads `url` into dest_dir as a single, deterministically-named
    file: 'youtube_download.mp3' (audio_only=True, yt-dlp's own audio
    extraction via ffmpeg) or 'youtube_download.mp4' (audio_only=False,
    best available muxed video+audio). Returns the downloaded file's path.
    Raises YoutubeDownloadError on any failure -- never returns a path
    that doesn't actually exist."""
    try:
        import yt_dlp
    except ImportError:
        raise YoutubeDownloadError(
            "yt-dlp is not installed. Install it with `pip install yt-dlp` "
            "(see requirements.txt)."
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / "youtube_download.%(ext)s")

    thumbnail_convertor = {"key": "FFmpegThumbnailsConvertor", "format": "jpg"}

    if audio_only:
        out_path = dest_dir / "youtube_download.mp3"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "writethumbnail": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}, thumbnail_convertor],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
    else:
        out_path = dest_dir / "youtube_download.mp4"
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": out_template,
            "merge_output_format": "mp4",
            "writethumbnail": True,
            "postprocessors": [thumbnail_convertor],
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise YoutubeDownloadError(f"Download failed: {e}") from e

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise YoutubeDownloadError(
            f"yt-dlp reported success but the expected output wasn't found at {out_path}."
        )

    _rename_thumbnail_to_cover(dest_dir)

    return out_path


def _rename_thumbnail_to_cover(dest_dir: Path) -> None:
    """Best-effort: renames yt-dlp's converted 'youtube_download.jpg'
    thumbnail to 'youtube_download [CO].jpg' so it's picked up by
    file_discovery.find_companions' existing [CO]-tag convention. Never
    raises -- a missing thumbnail (yt-dlp couldn't fetch/convert one for
    this video) is a silent no-op, not a failure."""
    thumb = dest_dir / "youtube_download.jpg"
    if not thumb.is_file():
        return
    try:
        thumb.rename(dest_dir / "youtube_download [CO].jpg")
    except OSError:
        pass
