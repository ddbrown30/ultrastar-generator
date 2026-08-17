"""Ambiguity-gated, Krumhansl-Kessler key-profile pitch-CLASS refinement
for pass-1's own RMVPE-based note detection (2026-08-16).

Real-audio validated across a 12-song roster, in the HARDER scenario --
pass-1's own self-detected note segmentation, not a given-correct-timing
shortcut -- via a resumable scratch harness
(scratchpad/pitch_comparison/tiebreak_selfdetected.py). Weighted pitch-
class accuracy: +2.4pp vs. the previously-shipped `_weighted_mode_pitch`
baseline (63.1% -> 65.5%, single fixed threshold applied uniformly, not
per-song-tuned), 9/12 songs improved (some substantially: Jungle Book
+7.7pp, BATB +6.4pp, Little Mermaid +4.6pp), 2 modest regressions
(Trixie Mattel - Gold -1.6pp, David Bowie - Magic Dance -1.4pp -- real,
known, accepted, not universal). See project_three_way_pitch_comparison
memory (and its own follow-up notes) for the full validation history,
including two OTHER related ideas from the same investigation that were
tried and explicitly DROPPED: tuning `ATTACK_TRIM_SEC`/
`CONFIDENCE_FLOOR_PERCENTILE` for pass-1's own segmentation (real churn,
net regression on the songs tested), and integrating usp's own separately
-trained ONNX pitch classifier directly (wins one song, loses another,
no consistent signal). This module is the one idea from that whole
investigation that held up.

Two-part design, both parts required together (validated as a unit, not
separately):

1. **Aggregation swap**: a note's pitch CLASS (0-11) is computed by
   SUMMING RMVPE's own raw per-frame activation (a 360-bin salience
   distribution over pitch, already computed on every RMVPE inference
   call via `rmvpe_onnx.RMVPE.predict`'s own 4th return value --
   previously discarded by `note_detection._rmvpe_pitch`) across the
   note's own [start, end) span, collapsing 360 bins down to 12 pitch
   classes, then taking the argmax -- instead of a per-frame-ROUNDED,
   confidence-weighted MODE over already-decoded semitone values. This
   is the same aggregation SHAPE `ultrastar_pitch` (usp)'s own trained
   classifier uses (sum per-block probabilities over a note, argmax),
   just applied to RMVPE's own raw output instead of a separately
   -trained model. The note's OCTAVE is never touched -- only pitch
   class -- same invariant `pitch_refresh.py`'s own MusicXML/audio pitch
   refinement already established.

2. **Ambiguity-gated tie-break**: when the resulting top-1 and runner-up
   (top-2) pitch classes are genuinely close (their aggregated salience
   masses differ by less than `AMBIGUITY_MARGIN_THRESHOLD`, a relative
   margin), the choice between them is broken by the song's own detected
   key, using the published Krumhansl & Kessler (1982/1990) key-profile
   ratings -- real empirical data from listener probe-tone studies,
   standard in MIR key-finding (e.g. music21's own
   analysis.discrete.KrumhanslSchmuckler). A note whose top-1 candidate
   is NOT genuinely contested (a single clear salience peak) is NEVER
   touched by the key profile, no matter how out-of-key it looks --
   deliberately narrower than this project's own twice-rejected blanket
   key-correction (see CLAUDE.md's "Removed / rejected approaches": a
   full global key-correction pass was built and fully removed for
   blindly snapping legitimate modal-mixture/borrowed notes; usp's own
   unconditional ±1-semitone `key_nudge`, vendored into `pitch_refresh.py`,
   is real but explicitly NOT universal -- a measured regression on
   Tarzan). Gating on genuine acoustic ambiguity (RMVPE's own salience
   was already torn between two candidates) rather than acting on every
   out-of-key note is what let this design survive real-audio validation
   where the broader approaches didn't.

RMVPE-only: SwiftF0 (this project's only other `PITCH_SOURCES` entry)
has no comparable raw multi-bin salience output, so this refinement is a
no-op whenever `pitch_source != "rmvpe"`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from .note_detection import NoteEvent

# Krumhansl & Kessler (1982) / Krumhansl (1990) key-profile ratings --
# published values from real listener probe-tone studies. Index 0 =
# tonic, 1 = minor 2nd above tonic, etc.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def bin_pitch_classes(cents_mapping: np.ndarray) -> np.ndarray:
    """Which pitch class (0-11) each RMVPE activation bin belongs to,
    from RMVPE's own real per-bin cents value (`rmvpe.cents_mapping`,
    the non-padded 360-length real range) -- reference frequency and
    cents-to-Hz conversion match `rmvpe_onnx.RMVPE.predict`'s own
    `frequency = 10 * (2 ** (cents / 1200))` exactly."""
    freqs = 10 * (2 ** (cents_mapping / 1200))
    midi = 69 + 12 * np.log2(np.clip(freqs, 1e-6, None) / 440.0)
    return np.round(midi).astype(int) % 12


def note_pc_mass(activation: np.ndarray, rtime: np.ndarray, bin_pc: np.ndarray,
                  start: float, end: float) -> np.ndarray:
    """12-bin pitch-class salience mass for one note's own time span --
    sums activation across the note's own frames first, then collapses
    360 bins down to 12 pitch classes."""
    i0 = np.searchsorted(rtime, start, side="left")
    i1 = np.searchsorted(rtime, end, side="right")
    if i1 <= i0:
        i1 = min(i0 + 1, len(rtime))
    seg = activation[i0:i1]
    if len(seg) == 0:
        return np.zeros(12)
    summed = seg.sum(axis=0)  # (360,)
    return np.bincount(bin_pc, weights=summed, minlength=12)


def detect_key(global_pc_mass: np.ndarray) -> Tuple[np.ndarray, int, str, float]:
    """Correlates the song's own observed pitch-class mass against all
    24 rotations of the published K-K major/minor profiles; returns
    (profile_rotated_to_tonic, tonic_pc, mode_name, correlation) for the
    best match."""
    best = None
    for mode_name, profile in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            rotated = np.roll(profile, tonic)
            corr = np.corrcoef(rotated, global_pc_mass)[0, 1]
            if best is None or corr > best[0]:
                best = (corr, rotated, tonic, mode_name)
    corr, rotated, tonic, mode_name = best
    return rotated, tonic, mode_name, float(corr)


def apply_ambiguity_tiebreak(
    notes: List[NoteEvent],
    activation: np.ndarray,
    rtime: np.ndarray,
    cents_mapping: np.ndarray,
    margin_threshold: float,
    debug_log=None,
    verbose: bool = True,
) -> List[NoteEvent]:
    """Returns a NEW note list with every note's PITCH CLASS (never
    octave) recomputed via the salience-mass-sum-argmax aggregation
    (see module docstring, part 1), with the top-1/top-2 tie broken by
    the song's own detected key ONLY when genuinely ambiguous (part 2).
    `notes` must already be pass-1's own FINAL, fully-segmented list
    (after all merge/cleanup passes) -- this never adds, removes, or
    reorders a note, and never changes timing."""
    if not notes:
        return notes

    bin_pc = bin_pitch_classes(cents_mapping)
    pc_masses = [note_pc_mass(activation, rtime, bin_pc, n.start, n.end) for n in notes]
    global_mass = np.sum(pc_masses, axis=0)
    profile, tonic, mode_name, corr = detect_key(global_mass)

    out: List[NoteEvent] = []
    n_ambiguous = 0
    n_changed = 0
    for note, mass in zip(notes, pc_masses):
        order = np.argsort(mass)[::-1]
        t1, t2 = int(order[0]), int(order[1])
        m1, m2 = float(mass[t1]), float(mass[t2])
        margin = (m1 - m2) / m1 if m1 > 0 else 1.0

        if margin < margin_threshold:
            n_ambiguous += 1
            new_pc = t1 if profile[t1] >= profile[t2] else t2
        else:
            new_pc = t1

        old_pc = note.pitch % 12
        if new_pc != old_pc:
            n_changed += 1
            new_pitch = note.pitch - old_pc + new_pc
            note = replace(note, pitch=new_pitch)
        out.append(note)

    msg = (f"Ambiguity key tie-break: detected pseudo-key {tonic} ({mode_name}, "
           f"profile-fit correlation={corr:.2f}), {n_ambiguous}/{len(notes)} note(s) "
           f"genuinely ambiguous, {n_changed} note(s) changed pitch class.")
    if verbose:
        print(f"[pass1] {msg}")
    if debug_log is not None:
        debug_log.line(f"[ambiguity-key-tiebreak] {msg}")

    return out
