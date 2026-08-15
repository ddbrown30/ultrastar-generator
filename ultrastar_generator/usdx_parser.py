"""Parses an UltraStar .txt file back into structured data -- the exact
inverse of usdx_writer.render_song's grammar. Used by verify_existing_song
to compare an existing song file's own pitch/timing against a fresh
pipeline run, before deciding whether to overwrite it.

Grammar (mirrors usdx_writer.py exactly, in reverse):
  '#TAG:value'                              header lines
  '<Type> <StartBeat> <Length> <Pitch> <Text>'  note lines (Type in :*FRG)
  '- <StartBeat>[ <EndBeat>]'                line-break lines
  'E'                                        lone terminator

Tolerant of a few things a HAND-MADE or third-party-tool-authored file
might do that this project's own writer never does: ',' as the BPM
decimal separator (usdx_writer._fmt_bpm's own comment documents both '.'
and ',' exist in real files in the wild), and a leading 'P1'/'P2' duet
marker (this project never WRITES duets, but an existing file being
verified might be one -- P1 is parsed, P2 is ignored, not an error).
"""

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
    """The file isn't structurally valid UltraStar .txt -- fails closed
    (verify_existing_song must treat this as 'can't verify, regenerate',
    never partially trust a malformed parse)."""


@dataclass
class ParsedSong:
    title: str
    artist: str
    bpm: float
    gap_ms: int
    entries: List[Union[Syllable, LineBreak]]  # start/end already in SECONDS
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
                break  # only P1 (or a non-duet file's own single part) is parsed
            continue
        if line == "E":
            break
        note_lines.append(line)

    if "BPM" not in tags:
        raise UsdxParseError("Missing #BPM tag")
    if "GAP" not in tags:
        raise UsdxParseError("Missing #GAP tag")
    bpm = _parse_bpm(tags["BPM"])
    try:
        gap_ms = int(round(float(tags["GAP"].strip().replace(",", "."))))
    except ValueError:
        raise UsdxParseError(f"Invalid #GAP value: {tags['GAP']!r}")

    entries: List[Union[Syllable, LineBreak]] = []
    # A word's FIRST syllable is forced word-start=True whenever there's no
    # preceding word to separate it from -- the very first syllable in the
    # whole file, OR the first syllable right after a line break. A line
    # break already establishes the word boundary visually/semantically, so
    # real files routinely omit the leading space on a line's first word
    # (confirmed on a real file: "- 48" / ": 57 2 4 Keep" with no leading
    # space) -- relying on text.startswith(" ") alone silently merged that
    # word onto the END of the PREVIOUS line's last word instead ("idol" +
    # "Keep" + "ing" -> one bogus "idolkeeping" token), corrupting every
    # downstream word-level comparison (verify_existing_song, the
    # --existing-txt-check feature) for any file authored this way.
    force_word_start = True
    prev_raw_text = None
    for line in note_lines:
        m = _NOTE_RE.match(line)
        if m:
            note_type, start_beat, length_beats, pitch, text = m.groups()
            start_beat, length_beats, pitch = int(start_beat), int(length_beats), int(pitch)
            start = beat_to_seconds(start_beat, gap_ms, bpm)
            end = beat_to_seconds(start_beat + length_beats, gap_ms, bpm)
            # A new word's boundary is marked by whitespace -- but WHICH side
            # varies by source. This project's own writer (usdx_writer.py)
            # puts a LEADING space on a new word's text; real SingStar-shipped
            # ground-truth files instead put a TRAILING space at the end of a
            # word's LAST syllable and no leading space at all (confirmed:
            # "Bare" + "ly " forms "Barely", "e" + "ven " forms "even" --
            # trusting text.startswith(" ") alone here would treat every
            # syllable after the first in a line as a CONTINUATION, silently
            # merging a whole line into one bogus word). Checking both this
            # syllable's own leading space AND the previous syllable's
            # trailing space handles both conventions -- and is a strict
            # superset of the old leading-space-only check, so this project's
            # own output (leading-space only) parses identically to before.
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
