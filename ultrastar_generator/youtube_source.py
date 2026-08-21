"""Downloads a YouTube video (via `yt-dlp`, optional dependency) as the primary source for a song folder.

`--artist`/`--title` are required separately (enforced in main.py) since YouTube titles aren't reliable.
Also downloads the thumbnail as 'youtube_download [CO].jpg' so it's picked up as the cover automatically."""

from __future__ import annotations

from pathlib import Path


class YoutubeDownloadError(RuntimeError):
    """yt-dlp isn't installed, the download failed, or the expected output file is missing."""


def download_youtube_source(url: str, dest_dir: Path, *, audio_only: bool) -> Path:
    """Downloads `url` into dest_dir as 'youtube_download.mp3' or '.mp4'. Returns the downloaded file's path."""
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
    """Best-effort rename of the downloaded thumbnail to the [CO]-tag cover convention. Never raises."""
    thumb = dest_dir / "youtube_download.jpg"
    if not thumb.is_file():
        return
    try:
        thumb.rename(dest_dir / "youtube_download [CO].jpg")
    except OSError:
        pass
