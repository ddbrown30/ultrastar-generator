"""Serializes a Song object to a spec-compliant UltraStar .txt file.

Note-line format:   "<Type> <StartBeat> <Length> <Pitch> <Text>"
Line-break format:  "- <StartBeat>" or "- <StartBeat> <EndBeat>"
File ends with a lone "E".

Duet support (P1/P2) is not written even though the Song model supports `parts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from . import config
from .models import Song, Syllable, LineBreak
from .tempo import seconds_to_beat, seconds_to_beat_length


def _fmt_bpm(bpm: float) -> str:
    s = f"{bpm:.2f}".rstrip("0").rstrip(".")  # '.' decimal sep, most broadly supported
    return s


def _quantize_entries(entries: List[object], bpm: float, gap_ms: int) -> List[tuple]:
    """Converts entries to integer beat values and enforces non-overlap at the integer-beat level
    (separate from postprocess.enforce_monotonic's seconds-level check, since quantization can round
    two non-overlapping seconds-level notes onto the same beat)."""
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


def _merge_connected_melisma_tails(quantized: List[tuple]) -> List[tuple]:
    """Folds a beat-adjacent, same-pitch melisma-continuation 'syl' entry into the preceding 'syl'
    entry (chains consecutive ones). Runs at the integer-beat level, post-quantization. Never merges
    across a 'break' entry."""
    out: List[tuple] = []
    for item in quantized:
        if (item[0] == "syl" and out and out[-1][0] == "syl"
                and item[4].strip() == config.MELISMA_CONTINUATION_TEXT
                and item[3] == out[-1][3]  # same pitch (midi_note)
                and item[1] == out[-1][1] + out[-1][2]):  # beat-adjacent: start == prev start+length
            prev = out[-1]
            out[-1] = (prev[0], prev[1], prev[2] + item[2], prev[3], prev[4], prev[5], prev[6])
        else:
            out.append(item)
    return out


def _remove_orphan_short_melisma_tails(quantized: List[tuple]) -> List[tuple]:
    """Deletes any melisma-continuation ('~') entry still only 1 beat long after
    _merge_connected_melisma_tails -- likely tracking noise, not a real continuation."""
    return [
        item for item in quantized
        if not (item[0] == "syl" and item[4].strip() == config.MELISMA_CONTINUATION_TEXT and item[2] == 1)
    ]


def render_song(song: Song, merge_connected_melisma: bool = False) -> str:
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
    if merge_connected_melisma:
        quantized = _merge_connected_melisma_tails(quantized)
        quantized = _remove_orphan_short_melisma_tails(quantized)

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


def write_song(song: Song, output_path: Path, merge_connected_melisma: bool = False) -> None:
    output_path = Path(output_path)
    output_path.write_text(render_song(song, merge_connected_melisma=merge_connected_melisma), encoding="utf-8")
