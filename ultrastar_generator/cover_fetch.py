"""Downloads a cover image online when neither a companion file
(file_discovery.find_companions) nor an embedded audio tag
(cover_extract.py) provided one.

Adapted from UltraStarKaraokeMaker's own fetch_assets.py/metadata.py
cascade (see [[reference-uskmaker]]) -- same source order (MusicBrainz +
Cover Art Archive, then iTunes, then Deezer), scoped down from its full
design in two ways:
  - COVER IMAGE ONLY. USKM's own cascade also enriches #YEAR/#GENRE and
    fetches a separate 16:9 background from fanart.tv; none of that is
    built here.
  - Only the three sources that need no API key/account setup at all.
    USKM's own Last.fm/Discogs tiers are both gated behind an env var
    the user has to go create an account and set up first -- skipped
    here to keep this a true zero-configuration fallback, consistent
    with how this project's other network lookups (LRCLIB, lyrics.ovh's
    now-removed successor) work out of the box.

Best-effort, same convention as cover_extract.py and lyrics_lookup.py's
own network calls: any failure (no network, nothing found, corrupt image
data, `requests` not installed) falls through to the next source, and a
total failure returns None -- never raises, never blocks the pipeline on
a missing cover.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from . import config
from .cover_extract import _sniff_image_ext

# MusicBrainz requires an honest, identifiable User-Agent or it starts
# blocking the IP -- see https://musicbrainz.org/doc/MusicBrainz_API.
_USER_AGENT = "ultrastar-generator/1.0 (+https://github.com/ddbrown30/ultrastar-generator)"
_MB_BASE = "https://musicbrainz.org/ws/2"
_CAA_BASE = "https://coverartarchive.org"
# MusicBrainz's own rate limit is ~1 request/second; a small margin over
# that keeps repeated calls (e.g. --batch across many songs) from ever
# tripping it.
_MB_MIN_INTERVAL_SEC = 1.1
# Cover art is enrichment, not a requirement -- never worth blocking the
# pipeline long waiting on a server that may not have the answer.
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
    """Finds the most plausible MusicBrainz release MBID for
    (artist, title). Among every release attached to any matching
    recording, prefers the one with the EARLIEST known release date (the
    original release, not a later reissue/compilation) -- falls back to
    the first release seen at all if none have a date. Returns None if
    nothing plausible was found."""
    _respect_mb_rate_limit()
    data = _get_json(
        f"{_MB_BASE}/recording",
        headers={"User-Agent": _USER_AGENT},
        params={"query": f'recording:"{title}" AND artist:"{artist}"', "fmt": "json", "limit": 10},
    )
    if not data:
        return None

    candidates = []  # (mbid, date_str)
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
        candidates.sort(key=lambda c: c[1])  # ISO date strings sort chronologically as text
        return candidates[0][0]
    return fallback_mbid


def _mb_caa_cover_bytes(artist: str, title: str) -> Optional[bytes]:
    """MusicBrainz (to find the release) + Cover Art Archive (to fetch
    its front cover). The /front-500 endpoint 307-redirects to the real
    image on the Internet Archive -- `requests` follows redirects by
    default."""
    release_mbid = _mb_find_release_mbid(artist, title)
    if not release_mbid:
        return None
    return _get_bytes(f"{_CAA_BASE}/release/{release_mbid}/front-500", headers={"User-Agent": _USER_AGENT})


def _itunes_cover_bytes(artist: str, title: str) -> Optional[bytes]:
    """No key/account required; official rate limit (~20 req/min) is
    irrelevant for one call per song."""
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
    # Documented community trick: the thumbnail URL accepts other
    # resolutions by swapping the "100x100" suffix -- 600x600 exists for
    # nearly the whole catalog.
    art_url = art_url.replace("100x100", "600x600")
    return _get_bytes(art_url, headers={"User-Agent": _USER_AGENT})


def _deezer_cover_bytes(artist: str, title: str) -> Optional[bytes]:
    """No key/account required."""
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
    """Tries MusicBrainz+Cover Art Archive, then iTunes, then Deezer, in
    that order, stopping at the first real image any of them returns.
    Writes dest_dir/'<base_name><config.COVER_TAG_SUFFIX>.<ext>' (the
    real extension, sniffed from magic bytes) and returns that path, or
    None if every source failed or `requests` isn't installed -- never
    raises.

    Results are cached under dest_dir by name (same check-then-fetch
    idiom `separation.isolate_vocals` uses for Demucs): if a previously
    downloaded cover already exists there, it's returned directly with
    no network call at all, so a re-run against the same work_dir never
    re-downloads.
    """
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
