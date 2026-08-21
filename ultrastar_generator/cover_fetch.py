"""Downloads a cover image online (fallback when no companion file or embedded tag has one). Tries MusicBrainz+Cover Art Archive, then iTunes, then Deezer; best-effort, never raises."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from . import config
from .cover_extract import _sniff_image_ext

# MusicBrainz requires an identifiable User-Agent or it blocks the IP.
_USER_AGENT = "ultrastar-generator/1.0 (+https://github.com/ddbrown30/ultrastar-generator)"
_MB_BASE = "https://musicbrainz.org/ws/2"
_CAA_BASE = "https://coverartarchive.org"
_MB_MIN_INTERVAL_SEC = 1.1  # stays under MusicBrainz's ~1 req/sec limit
_HTTP_TIMEOUT_SEC = 6

_last_mb_call = 0.0


def _respect_mb_rate_limit() -> None:
    global _last_mb_call
    elapsed = time.monotonic() - _last_mb_call
    if elapsed < _MB_MIN_INTERVAL_SEC:
        time.sleep(_MB_MIN_INTERVAL_SEC - elapsed)
    _last_mb_call = time.monotonic()


def _get_bytes(url: str, *, headers: Optional[dict] = None, params: Optional[dict] = None) -> Optional[bytes]:
    import requests
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=_HTTP_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        return resp.content
    except Exception:
        return None


def _get_json(url: str, *, headers: Optional[dict] = None, params: Optional[dict] = None) -> Optional[dict]:
    import requests
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=_HTTP_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _mb_find_release_mbid(artist: str, title: str) -> Optional[str]:
    """Finds the most plausible MusicBrainz release MBID for (artist, title), preferring the earliest release date; None if nothing found."""
    _respect_mb_rate_limit()
    data = _get_json(
        f"{_MB_BASE}/recording",
        headers={"User-Agent": _USER_AGENT},
        params={"query": f'recording:"{title}" AND artist:"{artist}"', "fmt": "json", "limit": 10},
    )
    if not data:
        return None

    candidates = []
    fallback_mbid = None
    for rec in data.get("recordings", []):
        for release in rec.get("releases", []):
            mbid = release.get("id")
            if not mbid:
                continue
            if fallback_mbid is None:
                fallback_mbid = mbid
            date_str = release.get("date") or ""
            if date_str:
                candidates.append((mbid, date_str))

    if candidates:
        candidates.sort(key=lambda c: c[1])
        return candidates[0][0]
    return fallback_mbid


def _mb_caa_cover_bytes(artist: str, title: str) -> Optional[bytes]:
    """Looks up the release via MusicBrainz, fetches its front cover from Cover Art Archive."""
    release_mbid = _mb_find_release_mbid(artist, title)
    if not release_mbid:
        return None
    return _get_bytes(f"{_CAA_BASE}/release/{release_mbid}/front-500", headers={"User-Agent": _USER_AGENT})


def _itunes_cover_bytes(artist: str, title: str) -> Optional[bytes]:
    data = _get_json(
        "https://itunes.apple.com/search",
        headers={"User-Agent": _USER_AGENT},
        params={"term": f"{artist} {title}", "media": "music", "entity": "song", "limit": 5},
    )
    results = (data or {}).get("results") or []
    if not results:
        return None
    art_url = results[0].get("artworkUrl100")
    if not art_url:
        return None
    art_url = art_url.replace("100x100", "600x600")  # request higher res
    return _get_bytes(art_url, headers={"User-Agent": _USER_AGENT})


def _deezer_cover_bytes(artist: str, title: str) -> Optional[bytes]:
    data = _get_json(
        "https://api.deezer.com/search",
        headers={"User-Agent": _USER_AGENT},
        params={"q": f'artist:"{artist}" track:"{title}"', "limit": 5},
    )
    results = (data or {}).get("data") or []
    if not results:
        return None
    album = results[0].get("album") or {}
    cover_url = album.get("cover_xl") or album.get("cover_big")
    if not cover_url:
        return None
    return _get_bytes(cover_url, headers={"User-Agent": _USER_AGENT})


_SOURCES = (_mb_caa_cover_bytes, _itunes_cover_bytes, _deezer_cover_bytes)


def fetch_cover_online(artist: str, title: str, dest_dir: Path, base_name: str) -> Optional[Path]:
    """Tries each source in turn, writes the first hit to dest_dir/'<base_name><suffix>.<ext>', returns that path or None. Cached: a previously downloaded cover is reused with no network call."""
    dest_dir = Path(dest_dir)
    for ext in (".jpg", ".png"):
        cached = dest_dir / f"{base_name}{config.COVER_TAG_SUFFIX}{ext}"
        if cached.exists():
            return cached

    try:
        import requests  # noqa: F401
    except ImportError:
        return None

    for fetch in _SOURCES:
        try:
            data = fetch(artist, title)
        except Exception:
            data = None
        if not data:
            continue
        ext = _sniff_image_ext(data)
        if ext is None:
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f"{base_name}{config.COVER_TAG_SUFFIX}{ext}"
        out_path.write_bytes(data)
        return out_path
    return None
