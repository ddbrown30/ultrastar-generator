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
                                     # belongs to, if reference-lyrics
                                     # alignment found a match; None if
                                     # unmatched or lyrics lookup wasn't
                                     # used/available.
    reference_text: Optional[str] = None  # the specific reference-lyrics
                                     # word this ASR word was aligned to,
                                     # if any (see lyrics_lookup.py) -- used
                                     # by verification.apply_reference_text
                                     # to force this word's own text to
                                     # match it whenever they disagree.
    dropped: bool = False  # lyrics_lookup.align_words_to_reference found NO
                                     # real reference correspondence for this
                                     # word at all (decoder hallucination --
                                     # a non-lyrical stretch of audio decoded
                                     # as confident-looking fake text). Kept
                                     # in the word SEQUENCE (not removed) so
                                     # its own ASR timing still correctly
                                     # bounds neighboring words' pass-1 note
                                     # zones in lyric_alignment.py -- real
                                     # confirmed regression (Video Games,
                                     # 2026-08-14): removing a dropped word
                                     # from the sequence entirely let the
                                     # NEXT real word's zone silently swallow
                                     # the dropped word's own ~16s of pass-1
                                     # notes instead. Only excluded at the
                                     # very end, when producing syllables for
                                     # writing -- never emitted as text/notes
                                     # in the final output.


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
    confidence: float = 1.0  # propagated from the owning pass-1 NoteEvent's
                               # own confidence where one exists (matched
                               # words); used by musicxml_reference.py to
                               # weight which syllables are trustworthy
                               # enough to anchor a per-song pitch-class
                               # calibration. 1.0 default (rather than 0.0)
                               # for any construction site that doesn't
                               # have a real value to report, so an absent
                               # value reads as "fully trusted", not
                               # "known bad" -- fine since only
                               # musicxml_reference.py's OPTIONAL pass 4
                               # currently reads this field at all.


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
