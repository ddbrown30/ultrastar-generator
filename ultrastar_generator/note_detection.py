"""Pass 1 of the pipeline: detect a sequence of sung notes (start, end,
pitch) directly from the isolated vocal audio, with NO dependency on
transcription. Lyrics get fitted onto this grid afterwards
(see lyric_alignment.py).

v2 changes (fixing reported bugs: overlapping notes, vibrato causing a
single sustained syllable to fragment into many near-duplicate notes):

  1. The pitch contour is median-filtered in semitone space BEFORE
     segmentation. Vocal vibrato is typically a 4-8 Hz wobble (a
     125-250ms period); a ~110ms filter window removes it while still
     passing genuine note changes, which are usually sustained far
     longer than one vibrato cycle. This was the single biggest source
     of over-segmentation before (pYIN's raw frame-to-frame pitch was
     being treated as ground truth).
  2. An onset event alone no longer forces a split -- only an onset that
     ALSO coincides with a real pitch change does. Consonant/attack
     transients inside a single sustained note were previously causing
     spurious splits.
  3. Two explicit merge passes run after initial segmentation:
       - merge_similar_adjacent: collapses consecutive, near-contiguous
         notes whose pitch differs by only a semitone or two -- this is
         exactly the "There," example from feedback: several ~150ms
         fragments alternating by 1 semitone should become ONE note.
       - merge_short_notes: any note shorter than half a beat (given the
         song's tempo) gets folded into whichever neighbor has the
         closer pitch, since a note that short can't even be
         represented distinctly on the beat grid anyway.
  4. Each note's final pitch is a confidence-weighted mode over its
     frames (rounded to the nearest semitone) rather than a plain
     median, which is more robust when a note's frames straddle two
     adjacent pitches (e.g. vibrato that didn't fully get filtered out).

Because notes are only ever appended as the scan walks forward in time,
and the merge passes only ever combine adjacent notes (never reorder or
create overlaps), the output remains start/end monotonic and
non-overlapping by construction. usdx_writer.py additionally enforces
this at the integer-beat level as a last-resort safety net.

v3 addition (fixing reported hallucinated notes during actual silence):
pYIN's voicing decision is based on pitch/periodicity evidence alone, NOT
loudness -- a genuinely silent instrumental intro, or the quiet gap
between phrases, can still contain quantization noise, resampling
ringing, or a faint hum with enough incidental periodicity for pYIN to
report a confident, real-looking pitch. An explicit RMS-energy gate
(_energy_gate) now runs alongside pYIN's own voicing flag, and a frame is
only treated as voiced if BOTH agree. The energy floor is relative to the
track's own 90th-percentile RMS (not the absolute peak, so one loud
transient can't quietly raise the bar for everything else), tunable via
--silence-threshold-db.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import config
from .tempo import beat_duration_ms


@dataclass
class NoteEvent:
    start: float             # seconds
    end: float                # seconds
    pitch: int                 # UltraStar pitch (MIDI - 60)
    confidence: float = 1.0     # mean pYIN voiced-probability over the note
    protected_start: bool = False  # True if this note's start is a deliberate
                                     # re-articulation split (see
                                     # config.REARTICULATION_STRENGTH_PERCENTILE)
                                     # -- the merge passes must never fold it
                                     # back into its predecessor just because
                                     # the pitch happens to match.


def _hz_to_ultrastar_pitch(hz: float) -> int:
    midi = 69 + 12 * np.log2(hz / 440.0)
    return int(round(midi)) - 60


def _smooth_midi_contour(midi: np.ndarray, voiced: np.ndarray, window: int) -> np.ndarray:
    """Median-filters `midi` in-place per contiguous voiced run (never
    smoothing across a silence gap, which would blur real note
    boundaries at phrase edges)."""
    if window <= 1:
        return midi.copy()

    smoothed = midi.copy()
    n = len(midi)
    i = 0
    half = window // 2
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        # contiguous voiced run is [i, j)
        run = midi[i:j]
        if len(run) >= 3:
            padded = np.pad(run, (half, half), mode="edge")
            out = np.empty_like(run)
            for k in range(len(run)):
                window_vals = padded[k:k + window]
                out[k] = np.nanmedian(window_vals) if np.any(~np.isnan(window_vals)) else run[k]
            smoothed[i:j] = out
        i = j
    return smoothed


def _weighted_mode_pitch(pitches: List[float], confs: List[float]) -> int:
    """Confidence-weighted mode of rounded semitone values -- more robust
    than a plain median when a note's frames straddle two adjacent
    pitches (residual vibrato, or a genuinely ambiguous boundary)."""
    rounded = np.round(pitches).astype(int)
    weights = np.asarray(confs, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    votes = {}
    for r, w in zip(rounded, weights):
        votes[r] = votes.get(r, 0.0) + w
    return max(votes, key=votes.get)


def _trim_attack(
    pitches: List[float], confs: List[float], frame_dur: float, trim_sec: float,
) -> tuple:
    """Drops the first trim_sec worth of frames from a note's pitch/
    confidence lists before its final pitch is computed -- counters a
    systematic FLAT bias measured against real sheet music (OMR
    cross-validation, 2026-08-07): pYIN's per-word pitch reads flat of
    the correct semitone roughly 3x as often as sharp (18 of 39 misses
    exactly -1 semitone vs 6 at +1, out of 91 whole-song anchors),
    consistent with legato/portamento singing where a note is
    approached with a rising glide FROM the previous (lower) pitch --
    a confidence-weighted mode over the WHOLE note, attack included,
    gets pulled toward that glide's lower pitches rather than the note's
    true settled value.

    Falls back to the full, untrimmed lists if trimming would leave too
    few frames to be a meaningful vote (a very short note's entire
    duration effectively IS its attack -- there is no separate "settled"
    portion to trim toward, and discarding most/all of a short note's
    only evidence would make its pitch estimate LESS reliable, not more).
    """
    trim_frames = int(round(trim_sec / frame_dur)) if frame_dur > 0 else 0
    if trim_frames <= 0 or len(pitches) - trim_frames < max(3, trim_frames):
        return pitches, confs
    return pitches[trim_frames:], confs[trim_frames:]


def _merge_similar_adjacent(
    notes: List[NoteEvent], max_pitch_diff: int, max_gap: float
) -> List[NoteEvent]:
    """Merges consecutive, near-contiguous notes whose pitch is close
    enough to be the same note (residual vibrato/tracking noise), while
    refusing to let a chain of small steps drift into one giant note.

    Comparing each candidate only to the immediately preceding note's
    pitch is NOT enough: a real stepwise melodic run (e.g. -6, -8, -9,
    -11, each only 1-2 semitones from its neighbor) can pass a per-step
    threshold at every link and transitively collapse into a single
    note spanning the whole run -- this was a real bug (a 4-word phrase
    with real melodic movement was flattened into one ~6-second note,
    all reported at the same pitch). To prevent that, each merged
    group's pitch RANGE (min to max of everything folded into it, not
    just the latest step) must also stay within max_pitch_diff.
    """
    if not notes:
        return notes
    merged: List[NoteEvent] = [notes[0]]
    group_min = [notes[0].pitch]
    group_max = [notes[0].pitch]

    for note in notes[1:]:
        prev = merged[-1]
        new_min = min(group_min[-1], note.pitch)
        new_max = max(group_max[-1], note.pitch)
        close_enough = (
            (note.start - prev.end) <= max_gap
            and abs(note.pitch - prev.pitch) <= max_pitch_diff
            and not note.protected_start
        )
        within_total_range = (new_max - new_min) <= max_pitch_diff

        if close_enough and within_total_range:
            prev_dur = prev.end - prev.start
            note_dur = note.end - note.start
            new_pitch = prev.pitch if prev_dur >= note_dur else note.pitch
            merged[-1] = NoteEvent(
                start=prev.start,
                end=note.end,
                pitch=new_pitch,
                confidence=(prev.confidence * prev_dur + note.confidence * note_dur) / max(prev_dur + note_dur, 1e-9),
                protected_start=prev.protected_start,
            )
            group_min[-1] = new_min
            group_max[-1] = new_max
        else:
            merged.append(note)
            group_min.append(note.pitch)
            group_max.append(note.pitch)
    return merged


def _merge_short_notes(notes: List[NoteEvent], min_duration: float, max_gap: float) -> List[NoteEvent]:
    """Folds notes shorter than min_duration into a neighbor (closer pitch
    wins, among neighbors close enough in TIME to plausibly be the same
    note -- see max_gap below), to clean up vibrato/tracking-noise
    fragmentation.

    Must never fold away a protected_start=True note's own start, and must
    never fold a neighbor's protected_start=True start away either --
    both are a deliberate re-articulation boundary (see NoteEvent's
    protected_start docstring). This was a real bug: _merge_similar_adjacent
    already honored protected_start, but this pass ran right after it and
    didn't check the flag at all, so a genuinely re-attacked same-pitch
    note (protected specifically so it wouldn't get silently re-merged)
    could still be swallowed here whenever the real audio's re-attack
    happened to produce a short segment -- which is common, since a
    re-attack is often followed almost immediately by a brief unvoiced
    dip in the singer's own articulation (confirmed on a real bug report:
    "fall as Lucifer fell" collapsing back into one note on real audio
    despite the split firing correctly, because this pass then undid it).

    max_gap: a neighbor separated from this note by MORE than max_gap of
    real silence is never eligible as a merge target, regardless of how
    close its pitch is -- confirmed via a real bug (OMR cross-validation,
    2026-08-07): "fu" (correct pitch, -6) and "gi" (own pitch -5, one
    semitone of ordinary tracking noise) are two genuinely distinct fast
    syllables separated by a real ~120ms unvoiced gap, but this function
    used to consider ONLY pitch distance -- "fu" ended up merged into
    "gi" (losing "fu"'s correct pitch entirely) purely because gi's pitch
    happened to be 1 semitone closer than the contiguous previous note's,
    with no awareness that a real silence gap separated them. Vibrato/
    tracking-noise fragmentation (the case this pass exists to clean up)
    does NOT typically produce a genuine unvoiced dropout between the
    fragments -- the pitch wobbles while voicing stays continuous -- so
    gap distance is a reliable, cheap signal for telling "one note that
    briefly stuttered" apart from "two real short notes". Same threshold
    _merge_similar_adjacent already uses for its own gap check, reused
    here for consistency rather than a new tunable.
    """
    if not notes:
        return notes
    changed = True
    notes = list(notes)
    # Repeat until stable: merging can create new short-adjacent situations.
    guard = 0
    while changed and guard < 10:
        changed = False
        guard += 1
        out: List[NoteEvent] = []
        i = 0
        while i < len(notes):
            note = notes[i]
            dur = note.end - note.start
            if dur >= min_duration or len(notes) == 1 or note.protected_start:
                out.append(note)
                i += 1
                continue
            prev_note = out[-1] if out else None
            next_note = notes[i + 1] if i + 1 < len(notes) else None
            if next_note is not None and next_note.protected_start:
                next_note = None
            if prev_note is not None and (note.start - prev_note.end) > max_gap:
                prev_note = None
            if next_note is not None and (next_note.start - note.end) > max_gap:
                next_note = None
            prev_diff = abs(note.pitch - prev_note.pitch) if prev_note else None
            next_diff = abs(note.pitch - next_note.pitch) if next_note else None

            if prev_note is not None and (next_diff is None or (prev_diff is not None and prev_diff <= next_diff)):
                merged = NoteEvent(
                    start=prev_note.start, end=note.end,
                    pitch=prev_note.pitch,
                    confidence=max(prev_note.confidence, note.confidence),
                    protected_start=prev_note.protected_start,
                )
                out[-1] = merged
                changed = True
                i += 1
            elif next_note is not None:
                merged = NoteEvent(
                    start=note.start, end=next_note.end,
                    pitch=next_note.pitch,
                    confidence=max(next_note.confidence, note.confidence),
                    protected_start=note.protected_start,
                )
                out.append(merged)
                changed = True
                i += 2
            else:
                out.append(note)
                i += 1
        notes = out
    return notes


def _remove_pitch_spikes(
    notes: List[NoteEvent],
    max_duration: float,
    min_jump_semitones: float,
    neighbor_similarity_semitones: float,
    max_neighbor_gap: float,
) -> List[NoteEvent]:
    """Removes an isolated short note that jumps far in pitch from BOTH
    its neighbors and whose neighbors are close to each other in pitch --
    i.e. "a brief detour to a very different pitch that then returns to
    where it was", which is a strong signature of a tracking glitch
    rather than a real, intentional note. Confirmed useful in practice:
    caught exactly this pattern in a real fallback-note case (see
    lyric_alignment.py's neighbor-borrowing fix for the other half of
    that story).

    A removed spike gets folded into the PREVIOUS note (extending its end
    to cover the spike's duration) rather than dropped outright, so total
    time coverage is preserved and nothing downstream sees a gap.
    """
    if len(notes) < 3:
        return notes

    out = list(notes)
    i = 1
    while i < len(out) - 1:
        prev, cur, nxt = out[i - 1], out[i], out[i + 1]
        dur = cur.end - cur.start
        gap_before = cur.start - prev.end
        gap_after = nxt.start - cur.end

        is_spike = (
            dur <= max_duration
            and gap_before <= max_neighbor_gap
            and gap_after <= max_neighbor_gap
            and abs(prev.pitch - nxt.pitch) <= neighbor_similarity_semitones
            and abs(cur.pitch - prev.pitch) >= min_jump_semitones
            and abs(cur.pitch - nxt.pitch) >= min_jump_semitones
        )

        if is_spike:
            out[i - 1] = NoteEvent(
                start=prev.start, end=cur.end, pitch=prev.pitch,
                confidence=max(prev.confidence, cur.confidence),
            )
            del out[i]
            # Don't advance i: the new triplet at this position (in case
            # of back-to-back spikes) needs checking too.
            continue
        i += 1

    return out


def _absorb_trailing_artifacts(
    notes: List[NoteEvent],
    max_duration: float,
    confidence_ratio: float,
    max_gap: float,
    min_preceding_duration: float,
) -> List[NoteEvent]:
    """Absorbs a short, low-confidence note that trails a long sustained
    note into that note (extends its end), instead of keeping it as its
    own standalone note -- catches breath/release artifacts (a singer's
    exhale after a held note, or a trailing "-uh" as a held vowel
    releases) that neither _remove_pitch_spikes nor _merge_short_notes
    currently catch (see config.TRAILING_ARTIFACT_* for the real
    confirmed cases and why those two existing passes miss this specific
    signature: close pitch to the neighbor, not a big jump; genuinely
    low CONFIDENCE, which neither pass looks at at all).

    Only ever absorbs FORWARD (into the immediately preceding note) --
    never merges two candidate artifacts into each other directly (the
    while-loop re-scan below handles a chain of them one at a time,
    each absorption extending the SAME preceding note further), and
    never touches a protected_start note (a deliberate re-articulation
    boundary must never be silently swallowed).
    """
    if len(notes) < 2:
        return notes
    changed = True
    notes = list(notes)
    guard = 0
    while changed and guard < 10:
        changed = False
        guard += 1
        out: List[NoteEvent] = [notes[0]]
        for note in notes[1:]:
            prev = out[-1]
            dur = note.end - note.start
            prev_dur = prev.end - prev.start
            gap = note.start - prev.end
            is_artifact = (
                not note.protected_start
                and dur <= max_duration
                and 0 <= gap <= max_gap
                and prev_dur >= min_preceding_duration
                and note.confidence <= prev.confidence * confidence_ratio
            )
            if is_artifact:
                out[-1] = NoteEvent(
                    start=prev.start, end=note.end, pitch=prev.pitch,
                    confidence=prev.confidence, protected_start=prev.protected_start,
                )
                changed = True
            else:
                out.append(note)
        notes = out
    return notes


def _ensure_nonoverlapping(notes: List[NoteEvent], verbose: bool = True) -> List[NoteEvent]:
    """Hard, final guarantee for pass 1: no note may start before the
    previous one ends. The construction above (sequential frame walk +
    only-ever-combine-adjacent merges) should make this a no-op in
    practice -- but pass 1 is supposed to be the one place in the whole
    pipeline that's allowed to assume its own output is correct, so this
    checks that assumption explicitly instead of just trusting it. If
    this ever has to fix something, that's a real bug upstream worth
    looking into, not a normal occurrence -- hence the warning.

    A tiny epsilon guards against pure floating-point noise: two adjacent
    notes' boundary times are computed via different arithmetic paths
    (times[start_frame] for one note's start vs. times[end_frame-1] +
    frame_dur for the previous note's end), which are mathematically
    equal but can land a few ULPs apart -- confirmed on a real run where
    every "overlap" this caught was exactly 0.0000s to 4 decimal places.
    That's not a real bug to warn about; an overlap of a real frame's
    worth of time very much still is.
    """
    if not notes:
        return notes
    eps = 1e-6  # seconds -- far below any real frame duration (ms-scale)
    fixed: List[NoteEvent] = [notes[0]]
    n_fixed = 0
    for note in notes[1:]:
        prev = fixed[-1]
        if note.start < prev.end - eps:
            n_fixed += 1
            new_start = prev.end
            new_end = max(note.end, new_start + config.MIN_NOTE_DURATION_SEC)
            note = NoteEvent(start=new_start, end=new_end, pitch=note.pitch, confidence=note.confidence,
                              protected_start=note.protected_start)
        fixed.append(note)
    if n_fixed and verbose:
        print(f"[pass1] WARNING: had to fix {n_fixed} overlapping note(s) that shouldn't have "
              f"been possible by construction -- this points at a real bug, please report it.")
    return fixed


def _energy_gate(
    y: np.ndarray, sr: int, hop_length: int, frame_length: int, n_frames: int,
    reference_percentile: float, threshold_db_below_peak: float,
    absolute_floor_db: float = config.SILENCE_ABSOLUTE_FLOOR_DB,
) -> np.ndarray:
    """Returns a boolean array (length n_frames) marking which frames have
    enough RMS energy to plausibly contain a real sung note, independent
    of whatever pYIN's own pitch/voicing decision says. See the constant
    docstrings in config.py for why this needs BOTH a relative threshold
    (compared to the track's own loud sections) and an absolute one
    (catches the case where there's no louder reference to compare
    against at all, e.g. a long or fully silent clip)."""
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)), mode="edge")
    elif len(rms) > n_frames:
        rms = rms[:n_frames]

    rms_db = 20 * np.log10(rms + 1e-10)
    reference_db = float(np.percentile(rms_db, reference_percentile))
    relative_floor_db = reference_db - threshold_db_below_peak
    relative_voiced = rms_db >= relative_floor_db
    absolute_voiced = rms_db >= absolute_floor_db
    return relative_voiced & absolute_voiced, rms_db, reference_db, relative_floor_db


def _crepe_pitch(
    y: np.ndarray, sr: int, hop_length: int, fmin: float, fmax: float,
    n_frames: int, model: str, device: str,
) -> tuple:
    """Runs torchcrepe over the whole track once, returning (midi,
    periodicity) arrays fit to n_frames (torchcrepe's own frame count can
    be off by a handful of frames from librosa's due to differing
    padding/centering conventions -- padded/truncated the same way
    _energy_gate already handles this for librosa.feature.rms).

    Forces deterministic CUDA/cuDNN algorithm selection for the duration
    of this call only (restored after, via try/finally) -- confirmed in
    practice that CREPE's own inference is run-to-run non-deterministic
    on identical input (two back-to-back detect_notes() calls on the
    exact same cached audio array produced 295 vs 298 notes, 283 of 295
    compared notes differing in timing/pitch), almost certainly the same
    class of CUDA/cuDNN non-deterministic algorithm selection already
    documented for Demucs, but here perturbing pass 1's own note sequence
    directly rather than just the detected BPM. Scoped narrowly to this
    call (not set globally/left on) so it can't affect Demucs (runs as
    its own subprocess anyway, per separation.py -- unaffected either
    way) or WhisperX/pyannote (runs in-process, later in the pipeline,
    and hasn't been vetted for whether all its ops have deterministic
    kernels). `warn_only=True` so an op inside torchcrepe without a
    deterministic implementation degrades to a warning instead of
    crashing the whole pipeline."""
    import torch
    import torchcrepe

    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        audio = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
        pitch, periodicity = torchcrepe.predict(
            audio, sr, hop_length, fmin, fmax,
            model=model, batch_size=2048, device=device, return_periodicity=True,
        )
        pitch = pitch.squeeze(0).detach().cpu().numpy()
        periodicity = periodicity.squeeze(0).detach().cpu().numpy()
    finally:
        torch.use_deterministic_algorithms(prev_deterministic, warn_only=True)
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    def _fit(arr: np.ndarray, fill: float) -> np.ndarray:
        if len(arr) < n_frames:
            arr = np.pad(arr, (0, n_frames - len(arr)), mode="edge") if len(arr) else np.full(n_frames, fill)
        elif len(arr) > n_frames:
            arr = arr[:n_frames]
        return arr

    pitch = _fit(pitch, 0.0)
    periodicity = _fit(periodicity, 0.0)
    midi = np.where(pitch > 0, 69 + 12 * np.log2(np.clip(pitch, 1e-6, None) / 440.0), np.nan)
    return midi, periodicity


def _rmvpe_pitch(
    y: np.ndarray, sr: int, hop_length: int, fmin: float, fmax: float,
    n_frames: int, device: str,
) -> tuple:
    """Runs RMVPE (rmvpe_onnx) over the whole track once, returning (midi,
    confidence) arrays fit to n_frames.

    Unlike CREPE (which runs at our own sr/hop_length and only needs a
    small pad/truncate to fit -- see _crepe_pitch), RMVPE has its OWN
    fixed internal grid (audio resampled to 16kHz, 160-sample/10ms hop)
    that does not line up with ours, so fitting it to n_frames means real
    time-interpolation, not pad/truncate. Confidence and a voiced/unvoiced
    mask are interpolated separately from pitch so interpolation can't
    fabricate a pitch value by blending across a real unvoiced gap in
    RMVPE's own output into a fictitious in-between semitone.

    Like CREPE, RMVPE has no voicing decision of its own here -- it is
    only ever used as a pitch/confidence cross-check on top of whatever
    the primary source (pYIN or, if pitch_primary='rmvpe', RMVPE itself)
    already decided was voiced, never to mark a frame unvoiced (see
    _crepe_pitch's docstring for why)."""
    from rmvpe_onnx import RMVPE

    rmvpe = RMVPE(device=device)
    rtime, rfreq, rconf, _ = rmvpe.predict(y, sr)

    rmidi = np.where(
        (rfreq > 0) & (rfreq >= fmin) & (rfreq <= fmax),
        69 + 12 * np.log2(np.clip(rfreq, 1e-6, None) / 440.0),
        np.nan,
    )
    voiced_mask = (~np.isnan(rmidi)).astype(float)
    midi_filled = np.where(np.isnan(rmidi), 0.0, rmidi)

    target_times = np.arange(n_frames) * (hop_length / sr)
    conf_out = np.interp(target_times, rtime, rconf, left=0.0, right=0.0)
    midi_interp = np.interp(target_times, rtime, midi_filled, left=0.0, right=0.0)
    voiced_interp = np.interp(target_times, rtime, voiced_mask, left=0.0, right=0.0)
    midi_out = np.where(voiced_interp >= 0.5, midi_interp, np.nan)

    return midi_out, conf_out


def _cross_check(
    midi_raw: np.ndarray, voiced_prob: np.ndarray, voiced: np.ndarray,
    other_midi: np.ndarray, other_conf: np.ndarray,
    agreement_semitones: float, confidence_boost: float, disagreement_scale: float,
) -> tuple:
    """Shared 'does a second pitch source agree with the current primary
    reading' cross-check logic -- factored out of what used to be CREPE-
    specific inline code so RMVPE (or any future third/fourth source) can
    reuse it identically rather than duplicating the agree/disagree math.

    Where the other source agrees with the current midi_raw (within
    agreement_semitones), its pitch is trusted and used with a confidence
    boost. Where they disagree, midi_raw is left unchanged but that
    frame's confidence is downweighted -- deliberately never marked
    unvoiced, since that would fabricate a note boundary at exactly the
    frames we're least sure about (same reasoning as the original CREPE
    cross-check this was extracted from).

    Returns (new_midi_raw, new_voiced_prob, agree, disagree, comparable).
    """
    comparable = voiced & ~np.isnan(other_midi)
    agree = np.zeros(len(midi_raw), dtype=bool)
    agree[comparable] = np.abs(other_midi[comparable] - midi_raw[comparable]) <= agreement_semitones
    disagree = comparable & ~agree

    new_midi_raw = np.where(agree, other_midi, midi_raw)
    new_voiced_prob = voiced_prob.copy()
    new_voiced_prob[agree] = np.maximum(new_voiced_prob[agree], other_conf[agree] * confidence_boost)
    new_voiced_prob[disagree] *= disagreement_scale
    return new_midi_raw, new_voiced_prob, agree, disagree, comparable


def detect_notes(
    y: np.ndarray,
    sr: int,
    bpm: Optional[float] = None,
    fmin: float = 65.0,          # ~C2
    fmax: float = 1046.5,        # ~C6
    hop_length: int = 256,
    frame_length: int = 2048,
    min_note_dur: float = config.MIN_NOTE_DURATION_SEC,
    attack_trim_sec: float = config.ATTACK_TRIM_SEC,
    pitch_jump_semitones: float = config.NOTE_SPLIT_SEMITONES,
    smooth_window_sec: float = config.PITCH_SMOOTH_WINDOW_SEC,
    merge_semitones: int = config.NOTE_MERGE_SEMITONES,
    merge_max_gap_sec: float = config.NOTE_MERGE_MAX_GAP_SEC,
    min_note_beats_fraction: float = config.MIN_NOTE_BEATS_FRACTION,
    silence_reference_percentile: float = config.SILENCE_REFERENCE_PERCENTILE,
    silence_threshold_db: float = config.SILENCE_THRESHOLD_DB_BELOW_PEAK,
    silence_absolute_floor_db: float = config.SILENCE_ABSOLUTE_FLOOR_DB,
    spike_max_duration_sec: float = config.SPIKE_MAX_DURATION_SEC,
    spike_min_jump_semitones: float = config.SPIKE_MIN_JUMP_SEMITONES,
    trailing_artifact_max_duration_sec: float = config.TRAILING_ARTIFACT_MAX_DURATION_SEC,
    trailing_artifact_confidence_ratio: float = config.TRAILING_ARTIFACT_CONFIDENCE_RATIO,
    trailing_artifact_max_gap_sec: float = config.TRAILING_ARTIFACT_MAX_GAP_SEC,
    trailing_artifact_min_preceding_duration_sec: float = config.TRAILING_ARTIFACT_MIN_PRECEDING_DURATION_SEC,
    use_crepe: bool = config.ENABLE_CREPE,
    crepe_model: str = config.DEFAULT_CREPE_MODEL,
    use_rmvpe: bool = config.ENABLE_RMVPE,
    rmvpe_device: str = config.RMVPE_DEVICE,
    rmvpe_voicing: bool = False,  # let RMVPE's own confidence ALSO mark a
                                   # frame voiced when pYIN alone said
                                   # unvoiced (never removes a frame pYIN
                                   # found) -- see the voicing-decision
                                   # comment at its call site for why this
                                   # exists. PROTOTYPE, off by default.
    rmvpe_voicing_threshold: float = 0.5,
    cross_check_primary: bool = True,  # whether the OTHER base source
                                        # (pYIN when pitch_primary='rmvpe';
                                        # has no effect when pitch_primary=
                                        # 'pyin', since pYIN has no separate
                                        # cross-check slot of its own there)
                                        # cross-checks the primary at all --
                                        # set False to test a source
                                        # completely alone, no agreement/
                                        # confidence-boost logic applied.
    pitch_primary: str = "pyin",  # "pyin" (default) or "rmvpe" -- which
                                   # source's own reading starts as
                                   # midi_raw/voiced_prob before any
                                   # cross-check is applied. The OTHER
                                   # source(s) (pYIN, CREPE if use_crepe,
                                   # RMVPE if use_rmvpe) are then applied
                                   # as cross-checks via _cross_check, in
                                   # a fixed order (CREPE, then RMVPE) --
                                   # same agree/disagree logic regardless
                                   # of which source is primary.
    verbose: bool = True,
    debug_log=None,
) -> List[NoteEvent]:
    import librosa

    if pitch_primary not in ("pyin", "rmvpe"):
        raise ValueError(f"pitch_primary must be 'pyin' or 'rmvpe', got {pitch_primary!r}")

    f0, voiced_flag, voiced_prob_pyin = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    pyin_voiced = np.array([bool(v) for v in voiced_flag]) if voiced_flag is not None else ~np.isnan(f0)
    pyin_midi_raw = 69 + 12 * np.log2(np.where(f0 > 0, f0, np.nan) / 440.0)  # preserved
                                       # untouched for debug logging, regardless of
                                       # pitch_primary/cross-checks below.
    voiced_prob_pyin = (
        np.asarray(voiced_prob_pyin, dtype=float) if voiced_prob_pyin is not None else np.ones(len(times))
    )
    pyin_voiced_prob_raw = voiced_prob_pyin.copy()  # preserved untouched, same reason.

    energy_voiced, rms_db, reference_db, floor_db = _energy_gate(
        y, sr, hop_length, frame_length, len(times),
        silence_reference_percentile, silence_threshold_db, silence_absolute_floor_db,
    )

    rmvpe_midi = rmvpe_conf = None
    if pitch_primary == "rmvpe" or use_rmvpe or rmvpe_voicing:
        try:
            rmvpe_midi, rmvpe_conf = _rmvpe_pitch(y, sr, hop_length, fmin, fmax, len(times), rmvpe_device)
        except ImportError:
            if verbose:
                print("[pass1] rmvpe_onnx not installed -- skipping RMVPE, using pYIN alone. "
                      "For better pitch accuracy: pip install rmvpe-onnx")
        except Exception as e:
            if verbose:
                print(f"[pass1] RMVPE failed ({e}); continuing without it.")

    # Voicing decision: pYIN + energy gate by default. If rmvpe_voicing,
    # ALSO treat a frame as voiced when RMVPE's own confidence clears
    # rmvpe_voicing_threshold, even if pYIN said unvoiced -- this only ever
    # ADDS voiced frames pYIN would have missed, never removes one pYIN
    # found (mirrors the CREPE cross-check's own "never fabricate a
    # boundary" philosophy, just applied to the voicing gate instead of
    # the pitch value). Exists because "pure RMVPE, no cross-check" still
    # scored notably below RMVPE's own raw accuracy (2026-08-07 finding):
    # RMVPE's pitch was only ever being read on frames pYIN's OWN (known
    # imperfect -- see the pYIN-voicing-gate architectural gap elsewhere in
    # this file's history) voicing decision allowed through, discarding
    # RMVPE's real advantage on frames pYIN wrongly called unvoiced.
    rmvpe_adds_voicing = np.zeros(len(times), dtype=bool)
    if rmvpe_voicing and rmvpe_conf is not None:
        rmvpe_adds_voicing = rmvpe_conf >= rmvpe_voicing_threshold
        if verbose:
            n_added = int(np.sum(rmvpe_adds_voicing & ~pyin_voiced))
            print(f"[pass1] RMVPE voicing: {n_added} frame(s) newly marked voiced "
                  f"(RMVPE confidence >= {rmvpe_voicing_threshold}) that pYIN alone called unvoiced")
    voiced = (pyin_voiced | rmvpe_adds_voicing) & energy_voiced

    crepe_midi = crepe_periodicity = None
    crepe_agree = crepe_disagree = None
    pyin_agree = pyin_disagree = None
    rmvpe_agree = rmvpe_disagree = None

    if pitch_primary == "rmvpe" and rmvpe_midi is not None:
        midi_raw = np.where(voiced, rmvpe_midi, np.nan)
        voiced_prob = np.where(voiced, rmvpe_conf, 0.0)
        # pYIN becomes a cross-check source instead of primary -- gated by
        # cross_check_primary so a "pure RMVPE, no cross-checking at all"
        # config can be tested directly (see the 2026-08-07 finding that
        # chaining a strong source against weaker ones via the agree/
        # boost-confidence mechanism can make the FINAL result worse than
        # the strong source alone, even though it's individually the best
        # raw signal -- agreement boosts confidence regardless of whether
        # the agreed-upon pitch is actually correct).
        if cross_check_primary:
            midi_raw, voiced_prob, pyin_agree, pyin_disagree, comparable = _cross_check(
                midi_raw, voiced_prob, voiced, pyin_midi_raw, voiced_prob_pyin,
                config.CREPE_AGREEMENT_SEMITONES, config.CREPE_AGREEMENT_CONFIDENCE_BOOST,
                config.CREPE_DISAGREEMENT_CONFIDENCE_SCALE,
            )
        if verbose and cross_check_primary:
            print(f"[pass1] pitch_primary=rmvpe: pYIN cross-check, {int(np.sum(comparable))} frame(s) "
                  f"comparable, {int(np.sum(pyin_agree))} agreed (pYIN's pitch used), "
                  f"{int(np.sum(pyin_disagree))} disagreed (kept RMVPE's pitch, downweighted)")
    else:
        midi_raw = pyin_midi_raw.copy()
        voiced_prob = voiced_prob_pyin.copy()

    if use_crepe:
        try:
            crepe_midi, crepe_periodicity = _crepe_pitch(
                y, sr, hop_length, fmin, fmax, len(times), crepe_model, device="cuda",
            )
            midi_raw, voiced_prob, crepe_agree, crepe_disagree, comparable = _cross_check(
                midi_raw, voiced_prob, voiced, crepe_midi, crepe_periodicity,
                config.CREPE_AGREEMENT_SEMITONES, config.CREPE_AGREEMENT_CONFIDENCE_BOOST,
                config.CREPE_DISAGREEMENT_CONFIDENCE_SCALE,
            )
            if verbose:
                print(f"[pass1] CREPE ({crepe_model}): {int(np.sum(comparable))} frame(s) comparable to "
                      f"{pitch_primary}, {int(np.sum(crepe_agree))} agreed (within "
                      f"{config.CREPE_AGREEMENT_SEMITONES} semitone(s), CREPE's pitch used), "
                      f"{int(np.sum(crepe_disagree))} disagreed (kept {pitch_primary}'s pitch, downweighted)")
        except ImportError:
            if verbose:
                print("[pass1] torchcrepe not installed -- skipping CREPE cross-check, using "
                      f"{pitch_primary} alone. For better pitch accuracy: pip install torchcrepe")
        except Exception as e:
            if verbose:
                print(f"[pass1] CREPE cross-check failed ({e}); continuing with {pitch_primary} alone.")

    if use_rmvpe and pitch_primary != "rmvpe" and rmvpe_midi is not None:
        midi_raw, voiced_prob, rmvpe_agree, rmvpe_disagree, comparable = _cross_check(
            midi_raw, voiced_prob, voiced, rmvpe_midi, rmvpe_conf,
            config.RMVPE_AGREEMENT_SEMITONES, config.RMVPE_AGREEMENT_CONFIDENCE_BOOST,
            config.RMVPE_DISAGREEMENT_CONFIDENCE_SCALE,
        )
        if verbose:
            print(f"[pass1] RMVPE: {int(np.sum(comparable))} frame(s) comparable to current pitch, "
                  f"{int(np.sum(rmvpe_agree))} agreed (within {config.RMVPE_AGREEMENT_SEMITONES} semitone(s), "
                  f"RMVPE's pitch used), {int(np.sum(rmvpe_disagree))} disagreed (kept current pitch, downweighted)")

    if verbose:
        pyin_frac = float(np.mean(pyin_voiced)) if len(pyin_voiced) else 0.0
        energy_frac = float(np.mean(energy_voiced)) if len(energy_voiced) else 0.0
        combined_frac = float(np.mean(voiced)) if len(voiced) else 0.0
        rejected_by_energy = int(np.sum(pyin_voiced & ~energy_voiced))
        print(f"[pass1] {len(times)} frames ({times[-1] if len(times) else 0:.1f}s), "
              f"fmin={fmin:.0f}Hz fmax={fmax:.0f}Hz hop={hop_length} ({hop_length/sr*1000:.1f}ms/frame)")
        print(f"[pass1] voicing: pYIN alone {pyin_frac*100:.0f}%, energy gate {energy_frac*100:.0f}% "
              f"(reference={reference_db:.1f}dB @ p{silence_reference_percentile:.0f}, "
              f"relative floor={floor_db:.1f}dB [-{silence_threshold_db:.0f}dB below reference], "
              f"absolute floor={silence_absolute_floor_db:.0f}dB), combined {combined_frac*100:.0f}%")
        if rejected_by_energy:
            print(f"[pass1] energy gate rejected {rejected_by_energy} frame(s) pYIN thought were "
                  f"voiced but were too quiet to be a real note (likely noise/silence) -- "
                  f"if real quiet singing is getting cut, try raising --silence-threshold-db "
                  f"or lowering --silence-floor-db")

    frame_dur = hop_length / sr
    smooth_window_frames = max(1, int(round(smooth_window_sec / frame_dur)))
    if smooth_window_frames % 2 == 0:
        smooth_window_frames += 1  # median filter wants an odd window
    midi = _smooth_midi_contour(np.where(voiced, midi_raw, 0.0), voiced, smooth_window_frames)

    if debug_log is not None:
        def _pitch_str(m):
            return f"{int(round(m)) - 60:+3d}" if not np.isnan(m) else "  . "

        def _agree_str(agree_arr, disagree_arr, i):
            if agree_arr is not None and agree_arr[i]:
                return "agree"
            if disagree_arr is not None and disagree_arr[i]:
                return "disagree"
            return "n/a     "

        rows = []
        for i in range(len(times)):
            pyin_p = _pitch_str(pyin_midi_raw[i])
            pyin_conf = f"{pyin_voiced_prob_raw[i]:.3f}"
            crepe_p = _pitch_str(crepe_midi[i]) if crepe_midi is not None else "  .  "
            crepe_conf = f"{crepe_periodicity[i]:.3f}" if crepe_periodicity is not None else "  .  "
            rmvpe_p = _pitch_str(rmvpe_midi[i]) if rmvpe_midi is not None else "  .  "
            rmvpe_conf_str = f"{rmvpe_conf[i]:.3f}" if rmvpe_conf is not None else "  .  "
            crepe_agree_str = _agree_str(crepe_agree, crepe_disagree, i)
            rmvpe_agree_str = _agree_str(
                pyin_agree if pitch_primary == "rmvpe" else rmvpe_agree,
                pyin_disagree if pitch_primary == "rmvpe" else rmvpe_disagree,
                i,
            )
            combined_p = _pitch_str(midi_raw[i])
            smoothed_p = _pitch_str(midi[i]) if voiced[i] else "  . "
            rows.append(
                f"  {times[i]:8.4f}s  pyin={pyin_p} pyin_conf={pyin_conf} pyin_voiced={str(bool(pyin_voiced[i])):5}  "
                f"energy_voiced={str(bool(energy_voiced[i])):5}  crepe={crepe_p} crepe_conf={crepe_conf} "
                f"crepe_vs_primary={crepe_agree_str}  rmvpe={rmvpe_p} rmvpe_conf={rmvpe_conf_str} "
                f"rmvpe_vs_primary={rmvpe_agree_str}  "
                f"combined={combined_p}  final_voiced={str(bool(voiced[i])):5}  smoothed={smoothed_p}"
            )
        debug_log.log_frames(
            rows,
            f"RAW PASS-1 FRAMES (direct pYIN/CREPE/RMVPE output before any note segmentation/merging -- "
            f"pitch_primary={pitch_primary}; pitch columns are UltraStar pitch, i.e. semitones from C4; "
            "'.' = unvoiced/no reading/source not enabled; pyin_conf is pYIN's own voiced-probability "
            "(0-1, its OWN internal confidence, independent of the energy gate); crepe_conf/rmvpe_conf are "
            "each model's own pitch-confidence estimate (0-1) -- all RAW, direct model output, not affected "
            "by any later cross-check logic; '_vs_primary' compares that source against whichever pitch was "
            "primary BEFORE that source's own cross-check ran)",
        )

    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True, units="time"
    )
    onset_frames_arr = np.round(onset_times / frame_dur).astype(int)
    onset_set = set(onset_frames_arr.tolist())

    # Strong-onset detection for same-pitch re-articulation splits (see
    # config.REARTICULATION_STRENGTH_PERCENTILE): a bare onset is only
    # ever allowed to assist a real pitch change (below); to split two
    # notes at the SAME pitch we additionally require the onset to be
    # among the strongest this track actually has, since a weak one is
    # much more likely a consonant/attack transient inside one held note.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    strong_onset_set = set()
    n_frames_total = len(times)
    near_strong_onset = np.zeros(n_frames_total, dtype=bool)
    if len(onset_frames_arr) > 0:
        valid = onset_frames_arr[(onset_frames_arr >= 0) & (onset_frames_arr < len(onset_env))]
        if len(valid) > 0:
            strengths = onset_env[valid]
            strength_floor = float(np.percentile(strengths, config.REARTICULATION_STRENGTH_PERCENTILE))
            strong_onset_set = set(valid[strengths >= strength_floor].tolist())
            # Dilate strong onsets by a small window: a re-attack's onset
            # often lands right at (or just inside) a brief unvoiced dip,
            # not exactly on a voiced frame -- see
            # config.REARTICULATION_ONSET_WINDOW_SEC.
            window_frames = max(0, int(round(config.REARTICULATION_ONSET_WINDOW_SEC / frame_dur)))
            for f in strong_onset_set:
                lo = max(0, f - window_frames)
                hi = min(n_frames_total, f + window_frames + 1)
                near_strong_onset[lo:hi] = True

    if verbose:
        print(f"[pass1] smoothing window={smooth_window_frames} frames (~{smooth_window_frames*frame_dur*1000:.0f}ms), "
              f"{len(onset_times)} onsets detected ({len(strong_onset_set)} strong enough to split a "
              f"same-pitch re-articulation), split threshold={pitch_jump_semitones} semitones")

    raw_notes = []
    cur_start_frame: Optional[int] = None
    cur_pitches: List[float] = []
    cur_confs: List[float] = []
    cur_protected_start = False

    def _flush(end_frame: int):
        nonlocal cur_start_frame, cur_pitches, cur_confs, cur_protected_start
        if cur_start_frame is not None and cur_pitches:
            raw_notes.append((cur_start_frame, end_frame, cur_pitches, cur_confs, cur_protected_start))
        cur_start_frame = None
        cur_pitches = []
        cur_confs = []
        cur_protected_start = False

    n = len(times)
    # A weaker threshold is enough to notice a change once an onset has
    # already flagged "something happened here"; without an onset we
    # require the full jump threshold, since it's the only sustained-drift
    # (legato) evidence we have.
    onset_assist_semitones = pitch_jump_semitones * 0.5
    min_dur_before_rearticulation_frames = max(
        1, int(round(config.MIN_DURATION_BEFORE_REARTICULATION_SEC / frame_dur))
    )

    for i in range(n):
        v = bool(voiced[i])
        m = float(midi[i]) if v else np.nan

        if not v or np.isnan(m):
            _flush(i)
            continue

        if cur_start_frame is None:
            cur_start_frame = i
            cur_pitches = [m]
            cur_confs = [float(voiced_prob[i]) if voiced_prob is not None else 1.0]
            # This note is resuming right after a silence/unvoiced gap --
            # if that gap sits at a strong onset, the gap itself IS the
            # re-articulation boundary (a real, brief unvoiced dip at the
            # attack), so protect it the same way an in-voicing split
            # would be: otherwise the next merge pass sees "same pitch,
            # small gap" and silently stitches it back to its predecessor.
            cur_protected_start = bool(near_strong_onset[i])
            continue

        running_median = float(np.median(cur_pitches))
        deviation = abs(m - running_median)
        at_onset = i in onset_set
        at_strong_onset = bool(near_strong_onset[i])
        note_so_far_frames = i - cur_start_frame

        should_split = deviation >= pitch_jump_semitones or (at_onset and deviation >= onset_assist_semitones)
        is_rearticulation = (
            not should_split
            and at_strong_onset
            and note_so_far_frames >= min_dur_before_rearticulation_frames
        )

        if should_split or is_rearticulation:
            _flush(i)
            cur_start_frame = i
            cur_pitches = [m]
            cur_confs = [float(voiced_prob[i]) if voiced_prob is not None else 1.0]
            cur_protected_start = is_rearticulation
        else:
            cur_pitches.append(m)
            cur_confs.append(float(voiced_prob[i]) if voiced_prob is not None else 1.0)

    _flush(n)

    notes: List[NoteEvent] = []
    for idx, (start_frame, end_frame, pitches, confs, protected_start) in enumerate(raw_notes):
        start_t = float(times[start_frame])
        # end_frame is EXCLUSIVE (frames [start_frame, end_frame) belong to
        # this note -- see _flush): the note's end time is the end of the
        # LAST included frame (end_frame - 1), not the end of end_frame
        # itself, which isn't part of this note at all. Using end_frame
        # directly here was a real, pre-existing off-by-one: it made every
        # adjacent pair of raw segments overlap by exactly one frame_dur,
        # silently "fixed" (and warned about) by _ensure_nonoverlapping on
        # every run -- e.g. 49 fixes on a real ~200-note song, which is
        # anything but the "shouldn't normally happen" case that guard is
        # meant to catch.
        end_t = float(times[min(end_frame - 1, n - 1)]) + frame_dur
        if end_t - start_t < config.MIN_NOTE_DURATION_SEC:
            if protected_start:
                # A deliberate re-articulation split (see NoteEvent's
                # protected_start docstring) must not be silently erased
                # just because the raw segment it produced happens to be
                # very brief -- a real re-attack is often followed almost
                # immediately by a natural articulation dip (confirmed on
                # a real bug report: "fall as Lucifer fell" on real audio
                # -- the split fired correctly but got dropped right here,
                # before protected_start ever got a chance to protect it
                # in the merge passes below). Stretch it to the minimum
                # audible duration instead of dropping it -- but capped at
                # the next raw segment's own start, never past it: an
                # uncapped stretch here routinely created the exact
                # overlaps _ensure_nonoverlapping's docstring calls "real
                # bugs, not routine" when two protected splits landed close
                # together (confirmed: 12 overlaps on one real song before
                # this cap was added). If there's no room to stretch into,
                # drop it same as an unprotected short segment would be.
                next_start_t = (
                    float(times[raw_notes[idx + 1][0]]) if idx + 1 < len(raw_notes) else None
                )
                stretched_end = start_t + config.MIN_NOTE_DURATION_SEC
                if next_start_t is not None:
                    stretched_end = min(stretched_end, next_start_t)
                if stretched_end - start_t < config.MIN_PLAUSIBLE_REARTICULATION_DURATION_SEC:
                    continue
                end_t = stretched_end
            else:
                continue
        trimmed_pitches, trimmed_confs = _trim_attack(pitches, confs, frame_dur, attack_trim_sec)
        pitch = _weighted_mode_pitch(trimmed_pitches, trimmed_confs) - 60
        notes.append(NoteEvent(
            start=start_t, end=end_t, pitch=pitch,
            confidence=float(np.average(trimmed_confs)) if trimmed_confs else 0.0,
            protected_start=protected_start,
        ))

    n_raw = len(notes)
    if debug_log is not None:
        debug_log.log_notes(notes, "pass 1 note segmentation, stage 0: raw segments (before any merge/cleanup)")
    notes = _merge_similar_adjacent(
        notes,
        max_pitch_diff=merge_semitones,
        max_gap=merge_max_gap_sec,
    )
    n_after_similar = len(notes)
    if debug_log is not None:
        debug_log.log_notes(notes, f"pass 1 note segmentation, stage 1: after similar-pitch merge "
                                     f"(<= {merge_semitones} semitone(s), <= {merge_max_gap_sec*1000:.0f}ms gap)")

    if bpm:
        beat_sec = beat_duration_ms(bpm) / 1000.0
        min_dur = max(min_note_dur, beat_sec * min_note_beats_fraction)
    else:
        min_dur = min_note_dur
    notes = _merge_short_notes(notes, min_dur, merge_max_gap_sec)
    n_after_short = len(notes)
    if debug_log is not None:
        debug_log.log_notes(notes, f"pass 1 note segmentation, stage 2: after short-note merge (< {min_dur*1000:.0f}ms)")

    notes = _remove_pitch_spikes(
        notes,
        max_duration=spike_max_duration_sec,
        min_jump_semitones=spike_min_jump_semitones,
        neighbor_similarity_semitones=config.SPIKE_NEIGHBOR_SIMILARITY_SEMITONES,
        max_neighbor_gap=config.SPIKE_MAX_NEIGHBOR_GAP_SEC,
    )
    n_after_spike = len(notes)
    if debug_log is not None:
        debug_log.log_notes(notes, f"pass 1 note segmentation, stage 3: after spike-outlier removal "
                                     f"(<= {spike_max_duration_sec*1000:.0f}ms, "
                                     f">= {spike_min_jump_semitones} semitone jump from both neighbors)")

    # Removing a spike can leave two same-pitch notes newly adjacent
    # (the spike was the only thing between them) -- run the similar-
    # adjacent merge once more to clean those back up into one note.
    notes = _merge_similar_adjacent(notes, max_pitch_diff=merge_semitones, max_gap=merge_max_gap_sec)
    n_after_spike_merge = len(notes)
    if debug_log is not None:
        debug_log.log_notes(notes, "pass 1 note segmentation, stage 4: after re-merging notes the spike removal left adjacent")

    notes = _absorb_trailing_artifacts(
        notes,
        max_duration=trailing_artifact_max_duration_sec,
        confidence_ratio=trailing_artifact_confidence_ratio,
        max_gap=trailing_artifact_max_gap_sec,
        min_preceding_duration=trailing_artifact_min_preceding_duration_sec,
    )
    n_after_artifact_absorb = len(notes)
    if debug_log is not None:
        debug_log.log_notes(notes, f"pass 1 note segmentation, stage 5: after absorbing trailing "
                                     f"breath/release artifacts (<= {trailing_artifact_max_duration_sec*1000:.0f}ms, "
                                     f"confidence <= {trailing_artifact_confidence_ratio:.0%} of the preceding "
                                     f"note's, <= {trailing_artifact_max_gap_sec*1000:.0f}ms gap, preceding note "
                                     f">= {trailing_artifact_min_preceding_duration_sec*1000:.0f}ms)")

    notes = _ensure_nonoverlapping(notes, verbose=verbose)
    if debug_log is not None:
        debug_log.log_notes(notes, "pass 1 note segmentation, stage 6 (FINAL): after enforcing non-overlap")

    if verbose:
        print(f"[pass1] notes: {n_raw} raw segments -> {n_after_similar} after "
              f"similar-pitch merge (<= {merge_semitones} semitone(s), <= {merge_max_gap_sec*1000:.0f}ms gap) "
              f"-> {n_after_short} after short-note merge (< {min_dur*1000:.0f}ms, ~{min_note_beats_fraction} beat) "
              f"-> {n_after_spike} after spike-outlier removal "
              f"(<= {spike_max_duration_sec*1000:.0f}ms, >= {spike_min_jump_semitones} semitone jump from both neighbors) "
              f"-> {n_after_spike_merge} after re-merging notes the spike removal left adjacent "
              f"-> {n_after_artifact_absorb} after absorbing trailing breath/release artifacts")
        if n_after_short != n_after_spike:
            print(f"[pass1] removed {n_after_short - n_after_spike} pitch spike(s) -- isolated brief "
                  f"jumps to a very different pitch that returned to the surrounding pitch afterward")
        if n_after_spike_merge != n_after_artifact_absorb:
            print(f"[pass1] absorbed {n_after_spike_merge - n_after_artifact_absorb} trailing breath/"
                  f"release artifact(s) into the sustained note they followed")
        if notes:
            durations = [nn.end - nn.start for nn in notes]
            pitches_all = [nn.pitch for nn in notes]
            print(f"[pass1] final: {len(notes)} notes, "
                  f"duration min/median/max = {min(durations)*1000:.0f}/{sorted(durations)[len(durations)//2]*1000:.0f}/{max(durations)*1000:.0f}ms, "
                  f"pitch range = {min(pitches_all)}..{max(pitches_all)}, "
                  f"span {notes[0].start:.2f}s-{notes[-1].end:.2f}s")
            very_short = sum(1 for d in durations if d < 0.10)
            if very_short:
                print(f"[pass1] note: {very_short} note(s) are still under 100ms after merging -- "
                      f"if that seems like too many, try raising --min-note-beat-fraction")

    return notes
