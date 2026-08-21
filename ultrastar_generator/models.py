"""Core data structures used across the pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Word:
    """A single transcribed word with timing, as returned by ASR."""
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float = 1.0
    line_id: Optional[int] = None  # matched reference-lyrics line, if any
    reference_text: Optional[str] = None  # aligned reference word; forces this word's text on mismatch
    dropped: bool = False  # no reference match found (likely hallucination); kept in sequence to bound neighbors' note zones, excluded only at syllable emission


@dataclass
class Syllable:
    """A syllable (or whole word, if unsplit) with its own timing/pitch."""
    text: str
    start: float          # seconds
    end: float             # seconds
    midi_note: int          # already shifted by -60 per the UltraStar spec
    is_word_start: bool = True  # whether a leading space is emitted before this syllable
    note_type: str = ":"    # ':' normal, '*' golden, 'F' freestyle, etc.
    line_id: Optional[int] = None  # propagated from the owning Word; forces a line break here
    confidence: float = 1.0  # propagated from owning NoteEvent; weights pitch-class calibration trust. Defaults 1.0 so "absent" reads as trusted, not bad


@dataclass
class LineBreak:
    """Marks the end of a phrase/line. start/end in seconds, converted to beats at write time."""
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
    bpm: float = 200.0          # written value; real tempo, UltraStar x4's it
    gap_ms: int = 0
    preview_start: Optional[float] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    edition: Optional[str] = None
    creator: Optional[str] = None

    entries: List[object] = field(default_factory=list)  # ordered Syllable/LineBreak objects, single-singer

    parts: Optional[Dict[str, List[object]]] = None  # reserved for future duet support, unused
