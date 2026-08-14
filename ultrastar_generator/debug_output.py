"""Writes an intermediate, notes-only UltraStar .txt file: pass 1's raw
detected notes, correctly timed and pitched, with placeholder text
instead of real lyrics. This is a real, loadable UltraStar song file --
open it in the editor to check timing/pitch in isolation, with no chance
of a later pass (lyric fitting) having touched anything.

Each note's placeholder text is its note name (e.g. "G#3"), which is
usually more useful for debugging than a generic dot or number: you can
directly compare what was detected against what you expect to hear,
without needing to eyeball the editor's piano roll.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from . import config
from .file_discovery import sanitize_filename
from .models import Song, Syllable, LineBreak
from .note_detection import NoteEvent
from .pitch import ultrastar_pitch_to_note_name


def notes_to_debug_entries(notes: List[NoteEvent], line_gap_sec: float = None) -> List[object]:
    """Converts a raw NoteEvent list into Song.entries: one "word" (the
    note name) per note, with a line break inserted after any silence gap
    at least `line_gap_sec` long (defaults to config.MIN_LINE_GAP_SEC),
    purely so the debug file doesn't render as one giant unbroken line."""
    if line_gap_sec is None:
        line_gap_sec = config.MIN_LINE_GAP_SEC

    entries: List[object] = []
    prev_end = None
    for note in notes:
        if prev_end is not None and (note.start - prev_end) >= line_gap_sec:
            entries.append(LineBreak(start=prev_end, end=note.start))
        entries.append(Syllable(
            text=ultrastar_pitch_to_note_name(note.pitch),
            start=note.start,
            end=note.end,
            midi_note=note.pitch,
            is_word_start=True,
            note_type=(config.NOTE_GOLDEN if (note.end - note.start) >= config.GOLDEN_NOTE_MIN_DURATION_SEC
                       else config.NOTE_NORMAL),
            confidence=note.confidence,
        ))
        prev_end = note.end
    return entries


def build_notes_debug_song(
    notes: List[NoteEvent],
    artist: str,
    title: str,
    mp3: str,
    bpm: float,
    gap_ms: int,
    label: str,
) -> Song:
    return Song(
        title=f"{title} [{label}]",
        artist=artist,
        mp3=mp3,
        bpm=bpm,
        gap_ms=gap_ms,
        entries=notes_to_debug_entries(notes),
    )


def write_notes_debug_file(
    notes: List[NoteEvent],
    artist: str,
    title: str,
    mp3: str,
    bpm: float,
    gap_ms: int,
    output_dir: Path,
    label: str,
) -> Path:
    from .usdx_writer import write_song

    song = build_notes_debug_song(notes, artist, title, mp3, bpm, gap_ms, label)
    out_path = Path(output_dir) / sanitize_filename(f"{artist} - {title} [{label}].txt")
    write_song(song, out_path)
    return out_path


def build_pass1_debug_song(
    notes: List[NoteEvent],
    artist: str,
    title: str,
    mp3: str,
    bpm: float,
    gap_ms: int,
) -> Song:
    return build_notes_debug_song(notes, artist, title, mp3, bpm, gap_ms, "PASS1 DEBUG")


def write_pass1_debug_file(
    notes: List[NoteEvent],
    artist: str,
    title: str,
    mp3: str,
    bpm: float,
    gap_ms: int,
    output_dir: Path,
) -> Path:
    return write_notes_debug_file(notes, artist, title, mp3, bpm, gap_ms, output_dir, "PASS1 DEBUG")
