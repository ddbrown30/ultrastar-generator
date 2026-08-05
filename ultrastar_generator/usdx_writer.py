"""Serializes a Song object to a spec-compliant UltraStar .txt file.

Note-line format:   "<Type> <StartBeat> <Length> <Pitch> <Text>"
Line-break format:  "- <StartBeat>" or "- <StartBeat> <EndBeat>"
File ends with a lone "E".

Duet support (P1/P2) is deliberately NOT written in v1 even though the
Song model supports `parts`, per the current requirements. If/when duet
support is added, this function only needs a branch that writes "P1"/"P2"
markers before each part's entries -- everything else (beat math, syllable
spacing) is reusable as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .models import Song, Syllable, LineBreak
from .tempo import seconds_to_beat, seconds_to_beat_length


def _fmt_bpm(bpm: float) -> str:
    # UltraStar files in the wild use either '.' or ',' as decimal sep
    # depending on locale of the tool that made them; '.' is safest/most
    # broadly supported by UltraStar Deluxe itself.
    s = f"{bpm:.2f}".rstrip("0").rstrip(".")
    return s


def _quantize_entries(entries: List[object], bpm: float, gap_ms: int) -> List[tuple]:
    """Converts Syllable/LineBreak entries (float seconds) into integer
    beat values, AND enforces the "a note must never start before the
    previous note ends" rule at the integer-beat level.

    This is deliberately separate from postprocess.enforce_monotonic,
    which only guarantees non-overlap in continuous seconds. Two notes
    that are non-overlapping by a few milliseconds in seconds can still
    round to the *same* beat once quantized to a coarse beat grid (this
    is exactly what produced the duplicate-start-beat bug in practice) --
    so the final, authoritative non-overlap check has to happen here, in
    the same integer space the .txt file actually uses.
    """
    out: List[tuple] = []
    occupied_until: int = None

    for entry in entries:
        if isinstance(entry, Syllable):
            start_beat = seconds_to_beat(entry.start, gap_ms, bpm)
            length_beats = seconds_to_beat_length(entry.end - entry.start, bpm)
            if occupied_until is not None and start_beat < occupied_until:
                start_beat = occupied_until
            length_beats = max(1, length_beats)
            end_beat = start_beat + length_beats
            occupied_until = end_beat
            out.append(("syl", start_beat, length_beats, entry.midi_note, entry.text, entry.is_word_start, entry.note_type))
        elif isinstance(entry, LineBreak):
            start_beat = seconds_to_beat(entry.start, gap_ms, bpm)
            if occupied_until is not None and start_beat < occupied_until:
                start_beat = occupied_until
            end_beat = None
            if entry.end is not None:
                end_beat = seconds_to_beat(entry.end, gap_ms, bpm)
                if end_beat < start_beat:
                    end_beat = start_beat
            out.append(("break", start_beat, end_beat))
        else:
            raise TypeError(f"Unknown entry type in song.entries: {type(entry)!r}")

    return out


def render_song(song: Song) -> str:
    lines: List[str] = []

    def tag(name: str, value) -> None:
        if value is not None and value != "":
            lines.append(f"#{name}:{value}")

    tag("TITLE", song.title)
    tag("ARTIST", song.artist)
    tag("LANGUAGE", song.language)
    tag("MP3", song.mp3)
    if song.cover:
        tag("COVER", song.cover)
    if song.background:
        tag("BACKGROUND", song.background)
    if song.video:
        tag("VIDEO", song.video)
    if song.videogap is not None:
        tag("VIDEOGAP", song.videogap)
    if song.genre:
        tag("GENRE", song.genre)
    if song.year:
        tag("YEAR", song.year)
    if song.edition:
        tag("EDITION", song.edition)
    if song.creator:
        tag("CREATOR", song.creator)
    tag("BPM", _fmt_bpm(song.bpm))
    tag("GAP", int(round(song.gap_ms)))
    if song.preview_start is not None:
        tag("PREVIEWSTART", round(song.preview_start, 2))

    quantized = _quantize_entries(song.entries, song.bpm, song.gap_ms)

    first_syllable_seen = False
    for item in quantized:
        if item[0] == "syl":
            _, start_beat, length_beats, midi_note, text, is_word_start, note_type = item
            if is_word_start and first_syllable_seen:
                text = " " + text
            first_syllable_seen = True
            lines.append(f"{note_type} {start_beat} {length_beats} {midi_note} {text}")
        else:
            _, start_beat, end_beat = item
            if end_beat is not None:
                lines.append(f"- {start_beat} {end_beat}")
            else:
                lines.append(f"- {start_beat}")

    lines.append("E")
    return "\n".join(lines) + "\n"


def write_song(song: Song, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.write_text(render_song(song), encoding="utf-8")
