"""Pass 2 (optional, on by default): detect the song's most likely musical
key from pass 1's raw pitch-class distribution, and nudge notes that don't
fit that key to the nearest in-key neighbor -- BEFORE any lyrics exist.
Runs directly on pass 1's NoteEvent list, as its own separate step (see
main.py), never bundled into pass 3 (lyric_alignment.align_words_to_notes)
-- moved here specifically so key correction can never depend on or affect
which word gets which note, and so pass 3 always sees a note grid, not raw
vs. word-fitting concerns tangled together.

Key detection uses music21's implementation of the Krumhansl-Schmuckler
algorithm (correlating the song's pitch-class distribution against
empirically-derived major/minor key profiles) rather than a hand-rolled
"which diatonic scale covers the most notes" heuristic -- a real,
well-validated key-finding algorithm instead of an approximation of one.

Deliberately conservative: only snaps a note when doing so moves it by
exactly one semitone onto a scale tone, and only when the note isn't
already in the detected key. Larger disagreements are left alone, since
those are more likely to be genuine chromatic notes than tracking noise.

Known open issue (not yet redesigned): this is a single GLOBAL key
detected once for the whole song. Confirmed in practice on a real song
("Stars") that this can snap an already-correct note to a wrong one when
the detected global key excludes a pitch class that's actually very
common in the song (35 of 329 notes, 3rd-most-common pitch class overall)
-- likely because the song modulates, or because a whole-song pitch-class
histogram just isn't enough context for Krumhansl-Schmuckler on a single
monophonic melody line. The user's stated original intent for this pass
was narrower than a global key snap: catch notes that are implausible
against surrounding MUSICAL PATTERNS (e.g. a lone note jumping far from
an otherwise smooth melodic line, then jumping right back), not force
every note into one rigid scale. Investigating a windowed/per-section or
pattern-based redesign is planned but not started.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .note_detection import NoteEvent

# Semitone offsets (0-11, root = 0) for natural major and natural minor.
_MAJOR = {0, 2, 4, 5, 7, 9, 11}
_MINOR = {0, 2, 3, 5, 7, 8, 10}


def _scale_for(root: int, mode: str) -> set:
    intervals = _MAJOR if mode == "major" else _MINOR
    return {(root + i) % 12 for i in intervals}


def detect_key(pitch_classes: List[int]) -> tuple:
    """Returns (root 0-11, mode) that best fits the given pitch-class
    sequence (each value already reduced mod 12), via music21's
    Krumhansl-Schmuckler key analysis. Only major/minor are considered,
    matching the diatonic scales _scale_for() knows how to snap against."""
    from music21 import note, stream

    s = stream.Stream()
    for pc in pitch_classes:
        s.append(note.Note(60 + (pc % 12), quarterLength=1))
    analyzed = s.analyze("key")
    mode = analyzed.mode if analyzed.mode in ("major", "minor") else "major"
    return (analyzed.tonic.pitchClass, mode)


_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def snap_to_key(notes: List[NoteEvent], debug_log=None) -> List[NoteEvent]:
    """Pass 2 entry point. Takes pass 1's raw NoteEvent list (no lyrics
    involved yet) and returns a same-length list with out-of-key notes
    nudged onto the nearest in-key neighbor.

    `debug_log`, if given, records the detected key/scale, the pitch-class
    frequency distribution it was computed from, and every note this pass
    actually changed (time, old pitch -> new pitch, and why) -- see
    project memory / this module's docstring for the investigation this
    was built for.
    """
    if not notes:
        return notes

    pitch_classes = [n.pitch % 12 for n in notes]
    root, mode = detect_key(pitch_classes)
    scale = _scale_for(root, mode)

    # Pitch-class frequency across the whole song, used to break ties when
    # an out-of-scale note sits exactly one semitone from two in-scale
    # neighbors (the common case for a standard 7-note diatonic scale,
    # where every non-scale tone falls in a whole-tone gap): prefer
    # snapping toward whichever neighbor pitch class the song actually
    # uses more, rather than leaving it un-snapped.
    counts = np.zeros(12)
    for pc in pitch_classes:
        counts[pc] += 1

    if debug_log is not None:
        debug_log.section("KEY CORRECTION (pass 2, notes only -- no lyrics involved yet)")
        debug_log.line(f"  Detected key: {_NAMES[root]} {mode}")
        debug_log.line(f"  Scale: {[_NAMES[p] for p in sorted(scale)]}")
        debug_log.line("  Pitch-class frequency (pre-correction):")
        for pc in range(12):
            in_scale = "in-scale" if pc in scale else "OUT OF SCALE"
            debug_log.line(f"    {_NAMES[pc]:3s} pc={pc:2d}: count={int(counts[pc]):4d}  ({in_scale})")

    out = []
    n_snapped = 0
    for n in notes:
        pc = n.pitch % 12
        if pc in scale:
            out.append(n)
            continue
        up = (pc + 1) % 12
        down = (pc - 1) % 12
        up_in = up in scale
        down_in = down in scale

        if up_in and not down_in:
            new_pitch = n.pitch + 1
            reason = f"only {_NAMES[up]} (up) is in-scale"
        elif down_in and not up_in:
            new_pitch = n.pitch - 1
            reason = f"only {_NAMES[down]} (down) is in-scale"
        elif up_in and down_in:
            # Equidistant (the common case in a diatonic scale): snap
            # toward whichever neighbor the song uses more often.
            if counts[up] >= counts[down]:
                new_pitch = n.pitch + 1
                reason = f"both in-scale, tie-broken toward {_NAMES[up]} (count={int(counts[up])} >= {_NAMES[down]} count={int(counts[down])})"
            else:
                new_pitch = n.pitch - 1
                reason = f"both in-scale, tie-broken toward {_NAMES[down]} (count={int(counts[down])} > {_NAMES[up]} count={int(counts[up])})"
        else:
            # Neither neighbor is in-key either (double chromatic) --
            # leave it alone rather than compounding a guess.
            out.append(n)
            continue
        if debug_log is not None:
            n_snapped += 1
            debug_log.line(f"    SNAP @ {n.start:.3f}-{n.end:.3f}s: {_NAMES[pc]} -> "
                            f"{_NAMES[new_pitch % 12]}  ({reason})")
        out.append(NoteEvent(
            start=n.start, end=n.end, pitch=new_pitch,
            confidence=n.confidence, protected_start=n.protected_start,
        ))
    if debug_log is not None:
        debug_log.line(f"  Total notes snapped: {n_snapped}/{len(notes)}")
    return out
