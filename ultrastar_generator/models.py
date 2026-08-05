"""Core data structures used across the pipeline.

These are intentionally independent of *how* the data was produced (ASR,
pitch tracker, etc.) so that each pipeline stage only has to deal with
plain, serializable objects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Word:
    """A single transcribed word with timing, as returned by ASR."""
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float = 1.0
    line_id: Optional[int] = None  # which reference-lyrics line this word
                                     # belongs to, if lyrics.ovh alignment
                                     # found a match; None if unmatched or
                                     # lyrics lookup wasn't used/available.


@dataclass
class Syllable:
    """A syllable (or whole word, if unsplit) with its own timing/pitch.

    `is_word_start` controls whether a leading space is emitted before this
    syllable's text when writing the txt file (UltraStar reconstructs
    words/sentences purely from spacing embedded in the note text).
    """
    text: str
    start: float          # seconds
    end: float             # seconds
    midi_note: int          # already shifted by -60 per the UltraStar spec
    is_word_start: bool = True
    note_type: str = ":"    # ':' normal, '*' golden, 'F' freestyle, etc.
    line_id: Optional[int] = None  # propagated from the owning Word; used
                                     # by phrasing.py to force a line break
                                     # exactly where the reference lyrics do.


@dataclass
class LineBreak:
    """Marks the end of a phrase/line. `start`/`end` are in seconds; they
    get converted to beats at write time."""
    start: float
    end: Optional[float] = None


@dataclass
class Song:
    title: str
    artist: str
    language: str = "English"
    mp3: str = ""
    cover: Optional[str] = None
    background: Optional[str] = None
    video: Optional[str] = None
    videogap: Optional[float] = None
    bpm: float = 200.0          # value as written to the txt (real tempo; UltraStar x4's it)
    gap_ms: int = 0
    preview_start: Optional[float] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    edition: Optional[str] = None
    creator: Optional[str] = None

    # Ordered list of Syllable / LineBreak objects, in the order they should
    # be written to the file, for the single (non-duet) singer.
    entries: List[object] = field(default_factory=list)

    # Reserved for future duet support: maps "P1"/"P2" -> list of entries,
    # exactly like `entries` above but per-singer. Not used/written in v1.
    parts: Optional[Dict[str, List[object]]] = None
