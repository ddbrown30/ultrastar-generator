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

from . import config
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


def _merge_connected_melisma_tails(quantized: List[tuple]) -> List[tuple]:
    """See config.py's "Final-step cleanup" comment (near MIN_NOTE_GAP_SEC)
    for the real motivating case. Folds a 'syl'
    entry into the immediately preceding 'syl' entry when they're beat-
    adjacent (no gap), same pitch, and THIS entry's own text is the
    melisma-continuation placeholder -- extending the previous entry's
    length and dropping this one. A single left-to-right pass, updating
    the last kept entry in place, correctly chains multiple consecutive
    same-pitch continuation notes into one (each new candidate is checked
    against whatever the previous one already collapsed into). Never
    merges across a LineBreak ('break') entry, and never touches the
    very first entry."""
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
    """Second, more aggressive cleanup step (user's explicit request,
    2026-08-10): after _merge_connected_melisma_tails, any melisma-
    continuation ('~') entry that's STILL only 1 beat long -- it didn't
    get absorbed because it wasn't beat-adjacent+same-pitch to its
    predecessor -- is deleted outright, leaving a gap on the beat grid
    rather than being merged into anything. A single isolated 1-beat '~'
    carries almost no real musical information (too short to actually
    sing) and is more often onset/release tracking noise than a genuine
    melisma continuation. Runs at the same INTEGER BEAT level as the
    merge pass, after it (so a '~' the merge pass already folded into a
    longer note is untouched -- only entries that SURVIVED as their own
    1-beat note are candidates here)."""
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
