"""Ambiguity-gated Krumhansl-Kessler key-profile pitch-CLASS refinement applied after pass-1's RMVPE note detection.

Two parts: (1) a note's pitch class is recomputed by summing RMVPE's raw per-frame 360-bin salience distribution across the note's span and taking the argmax, instead of a per-frame-rounded confidence-weighted mode -- octave is never touched. (2) when the top-1/top-2 pitch classes are genuinely close (`AMBIGUITY_MARGIN_THRESHOLD`), the tie is broken using the song's detected key via the published Krumhansl & Kessler key-profile ratings; an unambiguous note is never touched by the key profile, no matter how out-of-key it looks.

RMVPE-only: SwiftF0's ONNX graph exports only two scalars per frame (pitch_hz, confidence), no discretized salience distribution to disambiguate between, so this is a no-op when `pitch_source != "rmvpe"`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from .note_detection import NoteEvent

# Krumhansl & Kessler key-profile ratings. Index 0 = tonic, 1 = minor 2nd above tonic, etc.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def bin_pitch_classes(cents_mapping: np.ndarray) -> np.ndarray:
    """Which pitch class (0-11) each RMVPE activation bin belongs to, from its per-bin cents value."""
    freqs = 10 * (2 ** (cents_mapping / 1200))
    midi = 69 + 12 * np.log2(np.clip(freqs, 1e-6, None) / 440.0)
    return np.round(midi).astype(int) % 12


def note_pc_mass(activation: np.ndarray, rtime: np.ndarray, bin_pc: np.ndarray,
                  start: float, end: float) -> np.ndarray:
    """12-bin pitch-class salience mass for one note's time span: sums activation across the note's frames, then collapses 360 bins to 12 pitch classes."""
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
    """Correlates the song's observed pitch-class mass against all 24 rotations of the K-K major/minor profiles; returns (profile_rotated_to_tonic, tonic_pc, mode_name, correlation) for the best match."""
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
    """Returns a new note list with every note's pitch class (never octave, never timing/count/order) recomputed via salience-mass-sum-argmax, tie-broken by the song's detected key only when genuinely ambiguous."""
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
