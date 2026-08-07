"""Persistent per-run debug log, ON by default (--no-debug-log to skip).

Captures pipeline decisions that are otherwise only visible via
--verbose console output (which isn't saved anywhere) or not surfaced at
all (raw ASR confidence, reference-line grouping, note-zone boundary
math, syllable-proportional splits) -- written to
`<Artist> - <Title> [DEBUG LOG].txt` next to the other output files.

Exists because of a real, confirmed failure mode (see CLAUDE.md's
"Lessons learned" on ASR timestamp trust): WhisperX's forced alignment
can produce a run of severely wrong, LOW-CONFIDENCE word timestamps --
confirmed correlated with sustained/held notes, where a long vowel with
no new phonetic content gives the wav2vec2 CTC aligner nothing to anchor
a boundary to. Pass 3 currently trusts every word timestamp equally
regardless of confidence, so this silently distorts reference-line zone
boundaries downstream. This log makes the whole chain (raw ASR timing +
confidence -> reference-line grouping -> zone boundaries -> notes
assigned per group -> syllable-proportional split) inspectable after the
fact, so a bad final position can be traced back to exactly where it
went wrong instead of guessed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from .note_detection import NoteEvent


class DebugLog:
    """No-op if constructed with path=None (--no-debug-log) -- callers
    don't need to branch on whether logging is enabled."""

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
        """Dumps one line per pass-1 analysis frame (one row per ~11.6ms of
        audio, so this is large -- a full song is several thousand lines).
        `rows` are pre-formatted strings (one per frame, caller controls
        exact columns); this just wraps them in a section so the raw
        pYIN/CREPE output (before any smoothing, merging, or energy-gating)
        is directly inspectable instead of only visible as post-processed
        notes -- exists specifically to distinguish "pass 1's underlying
        pitch tracker got this frame wrong" from "a later merge/cleanup
        pass distorted an originally-correct reading." """
        if self._f is None:
            return
        self.section(header)
        for r in rows:
            self.line(r)

    def log_notes(self, notes: "List[NoteEvent]", label: str) -> None:
        """Dumps a NoteEvent list with FULL FLOAT-SECOND precision -- unlike
        the '[PASS1 DEBUG]'/'[PASS2 DEBUG]' .txt files, which quantize every
        note's start/end to integer beats for the UltraStar format (lossy:
        a note's true continuous-time boundary gets rounded, and a very
        short note's true duration can be invisibly stretched to the
        minimum 1-beat display width). Reverse-converting those quantized
        beat numbers back to seconds to compare against pass 3's own
        (full-precision) note handling was confirmed to give misleading
        results -- this exists specifically so a real pass-1-vs-pass-3
        pitch/timing mismatch can be told apart from a quantization-display
        artifact."""
        if self._f is None:
            return
        self.section(f"RAW NOTES ({label}, full float-second precision, no beat quantization)")
        self.line(f"  {len(notes)} note(s). Columns: start, end, duration, pitch, protected_start")
        for n in notes:
            self.line(f"    {n.start:9.4f} - {n.end:9.4f}  ({n.end - n.start:7.4f}s)  "
                      f"pitch={n.pitch:+3d}  protected_start={n.protected_start}")

    def log_reference_corrections(self, diffs: List[str]) -> None:
        if self._f is None:
            return
        self.section("REFERENCE LYRICS TEXT CORRECTIONS")
        if not diffs:
            self.line("None -- ASR text already matched the reference.")
        else:
            for d in diffs:
                self.line(f"  {d}")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
