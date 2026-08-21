"""Persistent per-run debug log, ON by default (--no-debug-log to skip). Writes pipeline decisions
not otherwise surfaced (ASR confidence, reference-line grouping, note-zone math, syllable splits) to
`<Artist> - <Title> [DEBUG LOG].txt`."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from .note_detection import NoteEvent


class DebugLog:
    """No-op if constructed with path=None (--no-debug-log)."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._f: Optional[TextIO] = None
        if path is not None:
            self._f = open(path, "w", encoding="utf-8")

    def section(self, title: str) -> None:
        if self._f is None:
            return
        self._f.write(f"\n{'=' * 12} {title} {'=' * 12}\n")

    def line(self, text: str = "") -> None:
        if self._f is None:
            return
        self._f.write(text + "\n")

    def log_frames(self, rows: List[str], header: str) -> None:
        """Dumps one pre-formatted line per pass-1 analysis frame (raw pitch-source output, before
        smoothing/merging/gating) under a section header."""
        if self._f is None:
            return
        self.section(header)
        for r in rows:
            self.line(r)

    def log_notes(self, notes: "List[NoteEvent]", label: str) -> None:
        """Dumps a NoteEvent list at full float-second precision, unlike the beat-quantized
        '[PASS1 DEBUG]' .txt file."""
        if self._f is None:
            return
        self.section(f"RAW NOTES ({label}, full float-second precision, no beat quantization)")
        self.line(f"  {len(notes)} note(s). Columns: start, end, duration, pitch, protected_start")
        for n in notes:
            self.line(f"    {n.start:9.4f} - {n.end:9.4f}  ({n.end - n.start:7.4f}s)  "
                      f"pitch={n.pitch:+3d}  protected_start={n.protected_start}")

    def log_lyrics_selection(self, *, source: str, track_name: str = "", artist_name: str = "",
                              lrclib_id: Optional[int] = None, duration: Optional[float] = None,
                              synced: bool = False, extra: str = "") -> None:
        """Records which lyrics candidate this run used (id/duration/track/artist)."""
        if self._f is None:
            return
        self.section("LYRICS SELECTION")
        self.line(f"  source: {source or '(none)'}")
        if track_name or artist_name:
            self.line(f"  track: {track_name!r} / artist: {artist_name!r}")
        self.line(f"  lrclib id: {lrclib_id if lrclib_id is not None else '(none)'}")
        self.line(f"  duration: {f'{duration:.1f}s' if duration is not None else '(unknown)'}")
        self.line(f"  synced lyrics: {synced}")
        if extra:
            self.line(f"  {extra}")

    def log_reference_corrections(self, diffs: List[str]) -> None:
        if self._f is None:
            return
        self.section("REFERENCE LYRICS TEXT CORRECTIONS")
        if not diffs:
            self.line("None -- ASR text already matched the reference.")
        else:
            for d in diffs:
                self.line(f"  {d}")

    def append_raw(self, text: str) -> None:
        """Appends pre-formatted text (its own section markers included) directly to this log's file.
        Used to merge a worker subprocess's own DebugLog output (worker_process.py) in after the fact."""
        if self._f is None:
            return
        self._f.write(text)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
