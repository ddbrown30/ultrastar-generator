"""Parses an UltraStar .txt file back into structured data (inverse of usdx_writer.render_song's
grammar). Tolerates a ',' BPM decimal separator and a leading P1/P2 duet marker (P1 parsed, P2
ignored) even though this project's own writer never produces either."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from .models import Syllable, LineBreak
from .tempo import beat_to_seconds

_NOTE_RE = re.compile(r"^([:*FRG])\s(-?\d+)\s(-?\d+)\s(-?\d+)\s(.*)$")
_BREAK_RE = re.compile(r"^-\s(-?\d+)(?:\s(-?\d+))?\s*$")
_TAG_RE = re.compile(r"^#([A-Za-z0-9_]+):(.*)$")


class UsdxParseError(ValueError):
    """The file isn't structurally valid UltraStar .txt."""


@dataclass
class ParsedSong:
    title: str
    artist: str
    bpm: float
    gap_ms: int
    entries: List[Union[Syllable, LineBreak]]  # start/end in seconds
    raw_tags: dict = field(default_factory=dict)


def _parse_bpm(raw: str) -> float:
    try:
        return float(raw.strip().replace(",", "."))
    except ValueError:
        raise UsdxParseError(f"Invalid #BPM value: {raw!r}")


def parse_usdx_file(path: Path) -> ParsedSong:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise UsdxParseError(f"Could not read {path}: {e}")

    tags: dict = {}
    note_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if line.startswith("#"):
            m = _TAG_RE.match(line)
            if not m:
                raise UsdxParseError(f"Malformed header line: {line!r}")
            tags[m.group(1).upper()] = m.group(2)
            continue
        if line in ("P1", "P2"):
            if line == "P2":
                break  # only P1 is parsed
            continue
        if line == "E":
            break
        note_lines.append(line)

    if "BPM" not in tags:
        raise UsdxParseError("Missing #BPM tag")
    bpm = _parse_bpm(tags["BPM"])
    if "GAP" in tags:
        try:
            gap_ms = int(round(float(tags["GAP"].strip().replace(",", "."))))
        except ValueError:
            raise UsdxParseError(f"Invalid #GAP value: {tags['GAP']!r}")
    else:
        gap_ms = 0  # #GAP is optional, unlike #BPM

    entries: List[Union[Syllable, LineBreak]] = []
    # First syllable, and the first syllable after any line break, is always a word start.
    force_word_start = True
    prev_raw_text = None
    for line in note_lines:
        m = _NOTE_RE.match(line)
        if m:
            note_type, start_beat, length_beats, pitch, text = m.groups()
            start_beat, length_beats, pitch = int(start_beat), int(length_beats), int(pitch)
            start = beat_to_seconds(start_beat, gap_ms, bpm)
            end = beat_to_seconds(start_beat + length_beats, gap_ms, bpm)
            # A word start can be marked by a leading space on this syllable (our own writer's
            # convention) or a trailing space on the previous syllable (some other files' convention).
            is_word_start = (
                force_word_start or text.startswith(" ")
                or (prev_raw_text is not None and prev_raw_text.endswith(" "))
            )
            force_word_start = False
            prev_raw_text = text
            entries.append(Syllable(
                text=text.strip(), start=start, end=end, midi_note=pitch,
                is_word_start=is_word_start, note_type=note_type,
            ))
            continue
        m = _BREAK_RE.match(line)
        if m:
            start_beat, end_beat = m.groups()
            start = beat_to_seconds(int(start_beat), gap_ms, bpm)
            end = beat_to_seconds(int(end_beat), gap_ms, bpm) if end_beat is not None else None
            entries.append(LineBreak(start=start, end=end))
            force_word_start = True
            continue
        raise UsdxParseError(f"Malformed note/break line: {line!r}")

    return ParsedSong(
        title=tags.get("TITLE", ""), artist=tags.get("ARTIST", ""),
        bpm=bpm, gap_ms=gap_ms, entries=entries, raw_tags=tags,
    )
