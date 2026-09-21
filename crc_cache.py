"""
Shared CRC-based change-detection cache for this project's standalone
media-library maintenance scripts (mp3_loudnorm.py, find_static_videos.py,
strip_lyrics_from_video.py, strip_audio_from_video.py).

One JSON file, one entry per file, keyed by the file's own path relative
to its drive (crc_key_for) rather than to whatever subfolder a given
invocation is pointed at -- so the same fixed cache file works correctly
regardless of which subfolder of the library a script is run against,
and a moved/renamed file is still recognized by its content instead of
being treated as new (CrcCache.check's moved_from/moved_entry).

Different scripts share the same file but never the same entries -- each
gets its own top-level NAMESPACE (e.g. "mp3_loudnorm", "static_video"),
so a video file checked by both strip_audio_from_video.py and
strip_lyrics_from_video.py gets two independent entries, one per
namespace, each free to carry whatever extra verdict fields that script
needs (is_static/difference, has_audio, format_tag_key/stream_tag_hits/
subtitle_indices, ...) alongside the common crc/size/mtime_ns fields
every namespace shares.

A file's CRC32 itself doesn't depend on which script is asking, though,
and three scripts (find_static_videos.py, strip_lyrics_from_video.py,
strip_audio_from_video.py) all look at the same video files -- so besides
each namespace's own verdict, there's one more reserved namespace,
IDENTITY_NAMESPACE, holding just {rel_key: {"crc", "size", "mtime_ns"}}
with no domain-specific fields, shared by every namespace. CrcCache.check
consults it whenever THIS namespace's own entry doesn't fast-hit: a
match there means some OTHER script already confirmed this exact
content at this exact path, so the crc is reused for free too -- only a
miss in BOTH places costs a real compute_crc32 read. Whichever namespace
calls record() first for a given file is the one that actually pays that
read; every later namespace (same run or a future one) gets it free
until the file's size/mtime change again.

Fast path: CrcCache.check() stats the file first (cheap) and, if a
cached entry's size AND mtime already match, returns it WITHOUT ever
reading the file's content -- no CRC32 computation at all. This is the
same quick-check rsync/make use, and it's what keeps a repeat run over
an already-processed library fast: previously every file's ENTIRE
content was re-read to compute a fresh CRC32 on every single run, even
when nothing had changed (confirmed the dominant cost for mp3_loudnorm.py
over a 1900+ song, network-share library -- several minutes just to
confirm nothing needed doing). Only when size/mtime don't match (a
genuinely new/changed file, one that's been moved/renamed, or an entry
from before this shortcut existed) does check() fall back to reading
the whole file via CRC32, which stays the real source of truth -- a
content change that somehow left size AND mtime both unchanged would be
missed by the fast path alone, but every caller backfills size/mtime
once it has confirmed a file via full CRC32, so this only ever applies
to a file the slow path has already vetted at least once before.
"""

import json
import os
import time
import uuid
import zlib
from pathlib import Path

CACHE_FILE = r"Z:\.media_crc_cache.json"  # fixed location, not relative to the processed root

# One legacy fixed-location file per script, from before they shared a
# single cache file. Merged (read-only, never modified or deleted) into
# the matching namespace on every load so existing progress isn't lost.
# Key: namespace this project uses for that script. Not consulted once
# every entry has migrated forward into CACHE_FILE's own namespace --
# checked every load rather than once, since that's a cheap handful of
# small-file exists() checks, not per-entry cost.
LEGACY_FILES = {
    "mp3_loudnorm": r"Z:\.loudnorm_crc.json",
    "static_video": r"Z:\.static_video_crc.json",
    "lyrics": r"Z:\.lyrics_crc.json",
    "strip_audio": r"Z:\.strip_audio_crc.json",
}

# Reserved namespace holding the cross-script {rel_key: {"crc", "size",
# "mtime_ns"}} identity record -- see this module's own docstring. Not a
# real script's namespace; never has a legacy file of its own.
IDENTITY_NAMESPACE = "_files"


def compute_crc32(path):
    """
    CRC32 checksum of a file's contents, as 8 hex digits.

    Only ever called when CrcCache.check()'s fast size/mtime path can't
    confirm a file is unchanged -- see this module's own docstring.
    """

    crc = 0

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            crc = zlib.crc32(chunk, crc)

    return format(crc & 0xFFFFFFFF, "08x")


def crc_key_for(path, root):
    """
    Cache key for a file: its path relative to its own drive (e.g.
    "Songs/Foo/track.mp3"), not relative to whatever subfolder was
    passed as root.

    The cache lives at one fixed location shared across every
    invocation regardless of which subfolder a script is pointed at
    for a given run, so keys have to mean the same thing every time --
    and keying by content (via CrcCache's own hash-based moved-file
    lookup) rather than by absolute path is what lets a moved/renamed
    file still be recognized instead of treated as new.
    """

    return path.relative_to(root.anchor).as_posix()


def _normalize_entry(value):
    """
    Coerce one registry value into {"crc", "size", "mtime_ns", ...extra}
    shape.

    A legacy per-script file predates size/mtime entirely and stores a
    bare crc32 string per key (mp3_loudnorm.py's original format); an
    already-migrated dict entry may also predate size/mtime (this
    project's first pass at the shared cache, before the fast path
    existed). Either way, wrapping/backfilling with size=None,
    mtime_ns=None keeps the entry loadable without a real migration
    step -- the missing size/mtime just means this one entry takes the
    slow (full CRC32 read) path exactly once more, then self-upgrades
    to the fast shape the next time its owning script records a result.
    """

    if isinstance(value, str):
        return {"crc": value, "size": None, "mtime_ns": None}

    if isinstance(value, dict) and "crc" in value:
        entry = dict(value)
        entry.setdefault("size", None)
        entry.setdefault("mtime_ns", None)
        return entry

    return {"crc": None, "size": None, "mtime_ns": None}


def load_registry():
    """
    Load the {namespace: {rel_key: entry}} registry from CACHE_FILE,
    merging in any not-yet-migrated legacy per-script files (see
    LEGACY_FILES) under their matching namespace. New-file entries win
    if a key exists in both. Missing/unreadable files are simply
    skipped -> that namespace starts empty.
    """

    registry = {}

    cache_path = Path(CACHE_FILE)

    if cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for namespace, entries in data.items():
                    if isinstance(entries, dict):
                        registry[namespace] = {
                            key: _normalize_entry(value)
                            for key, value in entries.items()
                        }
        except (OSError, json.JSONDecodeError):
            pass

    for namespace, legacy_file in LEGACY_FILES.items():
        legacy_path = Path(legacy_file)

        if not legacy_path.is_file():
            continue

        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(legacy_data, dict):
            continue

        namespace_entries = registry.setdefault(namespace, {})

        for key, value in legacy_data.items():
            namespace_entries.setdefault(key, _normalize_entry(value))

    return registry


def save_registry(registry):
    """
    Write the whole registry atomically (write to a temp file, then
    replace) at its fixed location (CACHE_FILE).

    Uses a unique temp filename per attempt and retries briefly on
    failure, since on a network share something external (antivirus,
    the SMB server itself, a sync/backup agent) can transiently hold a
    just-written temp file open for a moment. Called after every file
    whose entry actually changed, not batched, so a mid-run
    cancellation doesn't lose progress already made.
    """

    cache_path = Path(CACHE_FILE)
    data = json.dumps(registry, indent=2, sort_keys=True)

    last_error = None

    for attempt in range(5):
        tmp_path = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)

            tmp_path.replace(cache_path)
            return
        except OSError as e:
            last_error = e

            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

            time.sleep(0.25 * (attempt + 1))

    print(
        f"WARNING: could not save the CRC cache after several attempts "
        f"({last_error}). Continuing -- this update will be retried on "
        f"the next save."
    )


class CrcCheck:
    """One file's outcome from CrcCache.check() -- see its docstring."""

    def __init__(self, rel_key, crc, size, mtime_ns, fast_hit, cached_entry, moved_from, moved_entry):
        self.rel_key = rel_key
        self.crc = crc
        self.size = size
        self.mtime_ns = mtime_ns
        self.fast_hit = fast_hit
        self.cached_entry = cached_entry
        self.moved_from = moved_from
        self.moved_entry = moved_entry


class CrcCache:
    """
    Per-namespace view over the shared registry (see module docstring).

    Loads the WHOLE registry once at construction (every namespace, not
    just this one) so `save` below always writes back every other
    script's entries unchanged, then works against this namespace's own
    {rel_key: entry} slice for everything else.
    """

    def __init__(self, namespace):
        self.namespace = namespace
        self._registry = load_registry()
        self.entries = self._registry.setdefault(namespace, {})
        self._identity = self._registry.setdefault(IDENTITY_NAMESPACE, {})
        self._hash_to_key = {
            entry["crc"]: key
            for key, entry in self.entries.items()
            if entry.get("crc")
        }

    def check(self, path, root, force=False):
        """
        Stat `path` and compare against this namespace's own cached
        entry (if any) at its drive-relative key.

        Always returns a CrcCheck with a real `crc`. Three ways to get
        one, cheapest first: this namespace's own entry already matches
        size+mtime (fast_hit True, no read at all); some OTHER
        namespace already confirmed this exact path+size+mtime via the
        shared identity record (still no read -- see this module's own
        docstring -- but fast_hit is False, since THIS namespace still
        needs its own verdict); or a real compute_crc32 read, when
        neither cache has this file's current state. `moved_from`/
        `moved_entry` are only ever populated on a non-fast check, since
        a fast hit already IS the file's own correct entry -- there's
        nothing to have moved from.

        Raises OSError if `path` can't be stat()'d -- callers already
        need to handle that per their own error-reporting conventions,
        so it's not swallowed here.
        """

        st = path.stat()
        rel_key = crc_key_for(path, root)
        cached = self.entries.get(rel_key)

        if (
            not force
            and cached is not None
            and cached.get("size") == st.st_size
            and cached.get("mtime_ns") == st.st_mtime_ns
        ):
            return CrcCheck(
                rel_key, cached["crc"], st.st_size, st.st_mtime_ns,
                True, cached, None, None,
            )

        identity = self._identity.get(rel_key)

        if (
            not force
            and identity is not None
            and identity.get("size") == st.st_size
            and identity.get("mtime_ns") == st.st_mtime_ns
        ):
            crc = identity["crc"]
        else:
            crc = compute_crc32(path)

        moved_from = None
        moved_entry = None

        if not force:
            candidate = self._hash_to_key.get(crc)

            if candidate is not None and candidate != rel_key:
                moved_from = candidate
                moved_entry = self.entries.get(candidate)

        return CrcCheck(
            rel_key, crc, st.st_size, st.st_mtime_ns,
            False, cached, moved_from, moved_entry,
        )

    def record(self, rel_key, crc, size, mtime_ns, moved_from=None, **extra):
        """
        Persist (and immediately save) this namespace's entry for
        `rel_key`, and refresh the shared cross-namespace identity
        record for it too, so any OTHER namespace that next checks this
        same file gets the crc for free (see this module's own
        docstring). If `moved_from` names a different existing key (see
        CrcCheck), that stale entry is removed from THIS namespace as
        part of the same update -- the content has simply moved, not
        duplicated. The identity record at `moved_from` is left alone --
        it may still be correct for some other namespace that hasn't
        caught up to the move yet.
        """

        if moved_from is not None and moved_from in self.entries:
            del self.entries[moved_from]

        entry = {"crc": crc, "size": size, "mtime_ns": mtime_ns, **extra}
        self.entries[rel_key] = entry
        self._hash_to_key[crc] = rel_key
        self._identity[rel_key] = {"crc": crc, "size": size, "mtime_ns": mtime_ns}
        save_registry(self._registry)
        return entry

    def remove(self, rel_key):
        """Drop `rel_key`'s entry, if any, and save if it existed."""

        entry = self.entries.pop(rel_key, None)

        if entry is not None:
            save_registry(self._registry)

        return entry is not None
