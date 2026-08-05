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
        close_enough = (note.start - prev.end) <= max_gap and abs(note.pitch - prev.pitch) <= max_pitch_diff
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
            )
            group_min[-1] = new_min
            group_max[-1] = new_max
        else:
            merged.append(note)
            group_min.append(note.pitch)
            group_max.append(note.pitch)
    return merged


def _merge_short_notes(notes: List[NoteEvent], min_duration: float) -> List[NoteEvent]:
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
            if dur >= min_duration or len(notes) == 1:
                out.append(note)
                i += 1
                continue
            prev_note = out[-1] if out else None
            next_note = notes[i + 1] if i + 1 < len(notes) else None
            prev_diff = abs(note.pitch - prev_note.pitch) if prev_note else None
            next_diff = abs(note.pitch - next_note.pitch) if next_note else None

            if prev_note is not None and (next_diff is None or (prev_diff is not None and prev_diff <= next_diff)):
                merged = NoteEvent(
                    start=prev_note.start, end=note.end,
                    pitch=prev_note.pitch,
                    confidence=max(prev_note.confidence, note.confidence),
                )
                out[-1] = merged
                changed = True
                i += 1
            elif next_note is not None:
                merged = NoteEvent(
                    start=note.start, end=next_note.end,
                    pitch=next_note.pitch,
                    confidence=max(next_note.confidence, note.confidence),
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


def _ensure_nonoverlapping(notes: List[NoteEvent], verbose: bool = True) -> List[NoteEvent]:
    """Hard, final guarantee for pass 1: no note may start before the
    previous one ends. The construction above (sequential frame walk +
    only-ever-combine-adjacent merges) should make this a no-op in
    practice -- but pass 1 is supposed to be the one place in the whole
    pipeline that's allowed to assume its own output is correct, so this
    checks that assumption explicitly instead of just trusting it. If
    this ever has to fix something, that's a real bug upstream worth
    looking into, not a normal occurrence -- hence the warning.
    """
    if not notes:
        return notes
    fixed: List[NoteEvent] = [notes[0]]
    n_fixed = 0
    for note in notes[1:]:
        prev = fixed[-1]
        if note.start < prev.end:
            n_fixed += 1
            new_start = prev.end
            new_end = max(note.end, new_start + config.MIN_NOTE_DURATION_SEC)
            note = NoteEvent(start=new_start, end=new_end, pitch=note.pitch, confidence=note.confidence)
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


def detect_notes(
    y: np.ndarray,
    sr: int,
    bpm: Optional[float] = None,
    fmin: float = 65.0,          # ~C2
    fmax: float = 1046.5,        # ~C6
    hop_length: int = 256,
    frame_length: int = 2048,
    min_note_dur: float = config.MIN_NOTE_DURATION_SEC,
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
    verbose: bool = True,
) -> List[NoteEvent]:
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    pyin_voiced = np.array([bool(v) for v in voiced_flag]) if voiced_flag is not None else ~np.isnan(f0)
    midi_raw = 69 + 12 * np.log2(np.where(f0 > 0, f0, np.nan) / 440.0)

    energy_voiced, rms_db, reference_db, floor_db = _energy_gate(
        y, sr, hop_length, frame_length, len(times),
        silence_reference_percentile, silence_threshold_db, silence_absolute_floor_db,
    )
    voiced = pyin_voiced & energy_voiced

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

    onset_times = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True, units="time"
    )
    onset_set = set(np.round(onset_times / frame_dur).astype(int).tolist())
    if verbose:
        print(f"[pass1] smoothing window={smooth_window_frames} frames (~{smooth_window_frames*frame_dur*1000:.0f}ms), "
              f"{len(onset_times)} onsets detected, split threshold={pitch_jump_semitones} semitones")

    raw_notes = []
    cur_start_frame: Optional[int] = None
    cur_pitches: List[float] = []
    cur_confs: List[float] = []

    def _flush(end_frame: int):
        nonlocal cur_start_frame, cur_pitches, cur_confs
        if cur_start_frame is not None and cur_pitches:
            raw_notes.append((cur_start_frame, end_frame, cur_pitches, cur_confs))
        cur_start_frame = None
        cur_pitches = []
        cur_confs = []

    n = len(times)
    # A weaker threshold is enough to notice a change once an onset has
    # already flagged "something happened here"; without an onset we
    # require the full jump threshold, since it's the only sustained-drift
    # (legato) evidence we have.
    onset_assist_semitones = pitch_jump_semitones * 0.5

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
            continue

        running_median = float(np.median(cur_pitches))
        deviation = abs(m - running_median)
        at_onset = i in onset_set

        should_split = deviation >= pitch_jump_semitones or (at_onset and deviation >= onset_assist_semitones)

        if should_split:
            _flush(i)
            cur_start_frame = i
            cur_pitches = [m]
            cur_confs = [float(voiced_prob[i]) if voiced_prob is not None else 1.0]
        else:
            cur_pitches.append(m)
            cur_confs.append(float(voiced_prob[i]) if voiced_prob is not None else 1.0)

    _flush(n)

    notes: List[NoteEvent] = []
    for start_frame, end_frame, pitches, confs in raw_notes:
        start_t = float(times[start_frame])
        end_t = float(times[min(end_frame, n - 1)]) + frame_dur
        if end_t - start_t < config.MIN_NOTE_DURATION_SEC:
            continue
        pitch = _weighted_mode_pitch(pitches, confs) - 60
        notes.append(NoteEvent(
            start=start_t, end=end_t, pitch=pitch,
            confidence=float(np.average(confs)) if confs else 0.0,
        ))

    n_raw = len(notes)
    notes = _merge_similar_adjacent(
        notes,
        max_pitch_diff=merge_semitones,
        max_gap=merge_max_gap_sec,
    )
    n_after_similar = len(notes)

    if bpm:
        beat_sec = beat_duration_ms(bpm) / 1000.0
        min_dur = max(min_note_dur, beat_sec * min_note_beats_fraction)
    else:
        min_dur = min_note_dur
    notes = _merge_short_notes(notes, min_dur)
    n_after_short = len(notes)

    notes = _remove_pitch_spikes(
        notes,
        max_duration=spike_max_duration_sec,
        min_jump_semitones=spike_min_jump_semitones,
        neighbor_similarity_semitones=config.SPIKE_NEIGHBOR_SIMILARITY_SEMITONES,
        max_neighbor_gap=config.SPIKE_MAX_NEIGHBOR_GAP_SEC,
    )
    n_after_spike = len(notes)

    # Removing a spike can leave two same-pitch notes newly adjacent
    # (the spike was the only thing between them) -- run the similar-
    # adjacent merge once more to clean those back up into one note.
    notes = _merge_similar_adjacent(notes, max_pitch_diff=merge_semitones, max_gap=merge_max_gap_sec)
    n_after_spike_merge = len(notes)

    notes = _ensure_nonoverlapping(notes, verbose=verbose)

    if verbose:
        print(f"[pass1] notes: {n_raw} raw segments -> {n_after_similar} after "
              f"similar-pitch merge (<= {merge_semitones} semitone(s), <= {merge_max_gap_sec*1000:.0f}ms gap) "
              f"-> {n_after_short} after short-note merge (< {min_dur*1000:.0f}ms, ~{min_note_beats_fraction} beat) "
              f"-> {n_after_spike} after spike-outlier removal "
              f"(<= {spike_max_duration_sec*1000:.0f}ms, >= {spike_min_jump_semitones} semitone jump from both neighbors) "
              f"-> {n_after_spike_merge} after re-merging notes the spike removal left adjacent")
        if n_after_short != n_after_spike:
            print(f"[pass1] removed {n_after_short - n_after_spike} pitch spike(s) -- isolated brief "
                  f"jumps to a very different pitch that returned to the surrounding pitch afterward")
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
