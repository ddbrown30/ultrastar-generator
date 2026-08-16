"""Pitch-only refresh: given a FINISHED UltraStar .txt whose TIMING is
already trusted, re-detects each note's PITCH CLASS from the real audio
and rewrites it -- never touches timing, text, note count, or note order.
Same basic idea as the external reference tool `ultrastar_pitch` (usp,
see [[project-usp-comparison-gold]]): note segmentation is treated as a
solved, trusted INPUT, not something this module ever tries to detect.

Like `realign.py` (the timing-only counterpart), the existing file is
treated as READ-ONLY, unconditionally: `run_pitch_refresh_pipeline` never
writes to it, defaults to a separate "<name> [PITCH REFRESHED].txt"
output, and hard-refuses to run if an explicit output path resolves to
the same path as the existing file (reuses `realign.check_output_not_
existing_file`, the exact same guarantee, not a re-implementation).

Design choices that diverge from the rest of this project's pipelines,
deliberately, because this module's whole point is mirroring usp's own
scope and defaults rather than the main pipeline's:
  - Runs on Demucs-ISOLATED vocals by default (`isolate_vocals=True`,
    flipped 2026-08-15). The original mixed-audio default was based on a
    single-song finding (Trixie Mattel - Gold: both usp and our own
    RMVPE scored better on the mix than on isolated vocals, see
    [[project-usp-comparison-gold]]'s "mixed-vs-isolated" section) that
    was flagged there as "not yet validated on other songs." Broader
    9-song regression testing (this module's own `refresh_song_pitch`,
    both `rmvpe` and `swiftf0`, both this module's `default` and
    `gold_tuned` recipes) found that result does NOT generalize: for
    `rmvpe`, mixed vs. isolated is a coin flip (4/9 songs favor mixed,
    5/9 favor isolated, swings up to +-12pp with no consistent pattern);
    for `swiftf0`, mixed is a clear regression on 8/9 songs, sometimes by
    50+pp. Isolated is the safer default for either source -- roughly
    neutral for rmvpe, a real win for swiftf0. `--no-isolate-vocals`
    opts back into the old mixed-audio behavior (still matches usp's own
    default, which uses Spleeter only optionally).
  - CUDA is now required by DEFAULT (changed by the isolate_vocals flip
    above, 2026-08-15): `separation.isolate_vocals` invokes Demucs with a
    hardcoded `-d cuda`, and vocal isolation now runs by default. This
    module still never calls WhisperX, and the pitch source itself
    (RMVPE, `config.RMVPE_DEVICE = "cpu"`) still runs on CPU regardless --
    `--no-isolate-vocals` is the only way back to this module's original
    fully-CPU, no-CUDA-needed footprint (matching usp's own lightweight
    default).
  - `attack_trim_sec`/`confidence_floor_percentile` (existing, real,
    evidence-backed but still-unshipped-ELSEWHERE prototypes from
    2026-08-07, see `config.ATTACK_TRIM_SEC`/`CONFIDENCE_FLOOR_
    PERCENTILE`'s own docstrings -- those shared config constants
    themselves stay at their own off/0.0 default, since that's the MAIN
    pipeline's separate, still-unvalidated-there decision; this module's
    own defaults below are independent of them) plus `voicing_threshold`
    (raises `_rmvpe_source`'s own internal voicing gate above its 0.5
    default) and `key_nudge` (see `_KeyNudge` below) are
    all DEFAULT-ON here at the exact values real grid-search tuning
    found on Trixie Mattel - Gold (`attack_trim_sec=0.04`,
    `confidence_floor_percentile=50.0`, `voicing_threshold=0.58`,
    `key_nudge=True`) -- user's explicit decision, 2026-08-12, after
    regression-testing this recipe against 5 more ground-truth songs
    found it never regressed and was sometimes a real improvement (see
    project memory) -- with one important caveat: Chicago, one of those
    5 songs, was later found to have its OWN data-quality problems (its
    `.mxl` is missing roughly half the song and only partially agrees
    with the audio) that invalidate its result specifically, including a
    striking apparent `key_nudge`-caused regression there that may
    actually have been the nudge correctly compensating for bad data,
    not a real regression -- genuinely unknown until that song's data is
    cleaned up. Chicago is excluded from this module's validation pool
    for now. The other 4 songs' clean results (see project memory) still
    stand and were the real basis for this recipe becoming the default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from . import config
from .models import LineBreak, Song, Syllable
from .usdx_parser import ParsedSong, UsdxParseError, parse_usdx_file
from .usdx_writer import write_song
from .note_detection import PITCH_SOURCES, _weighted_mode_pitch, _trim_attack, _confidence_floor_filter
from .realign import (
    AmbiguousExistingTxtError,
    _song_from_existing,
    check_output_not_existing_file,
    find_existing_txt_in_folder,
)

# This module's own output naming convention, excluded (alongside
# realign.py's own "[REALIGNED]") when auto-detecting a folder's existing
# .txt -- otherwise re-running on a folder that already has a previous
# pitch-refresh output would either see two candidates (falsely
# "ambiguous") or, worse, pick this module's OWN prior output as the next
# run's INPUT, compounding drift across repeated runs.
OUTPUT_MARKER = "[PITCH REFRESHED]"
_EXCLUDE_MARKERS = ("[REALIGNED]", OUTPUT_MARKER)

# This module's own validated recipe (real grid-search tuning on Trixie Mattel - Gold,
# regression-tested against 4 more ground-truth songs with no real regression -- see
# [[project-usp-comparison-gold]] and the module docstring above for the full history,
# including the Chicago data-quality caveat) -- DEFAULT as of 2026-08-12, the user's
# explicit decision. Named constants (not bare literals scattered across every default-
# bearing signature below) so there is exactly one place to look up or change the recipe.
DEFAULT_ATTACK_TRIM_SEC = 0.04
DEFAULT_CONFIDENCE_FLOOR_PERCENTILE = 50.0
DEFAULT_VOICING_THRESHOLD = 0.58
DEFAULT_KEY_NUDGE = True


# --- key nudge (usp-inspired, DEFAULT ON -- see caveat below) -----------

class _KeyNudge:
    """Conservative +-1-semitone pitch-class nudge toward whichever
    neighbor better fits a detected 'pseudo key' -- a direct, vendored
    port of `ultrastar_pitch`'s (usp, MIT-licensed) own
    `StochasticPostprocessor`. Kept self-contained here rather than
    imported so this module has no runtime dependency on that separate
    reference-project folder being present in a given checkout.

    NOT an unconditional universal win -- see [[project-usp-comparison-
    gold]]: an earlier real 6-song test (isolated audio, nudge alone)
    found a real regression on one song (Tarzan) and substantial churn
    on two others alongside clean wins on three. A LATER regression run
    (this module's own roster test, 2026-08-12) found something starker
    on Chicago -- key_nudge alone appeared to cause a 23-point accuracy
    crash there -- but Chicago was SEPARATELY found to have real data-
    quality problems (its `.mxl` reference is missing roughly half the
    song and only partially matches the audio), so that specific result
    is INVALIDATED, not confirmed: it's genuinely unknown whether the
    nudge regressed Chicago or was correctly compensating for bad
    ground truth, pending the user cleaning up that song's data. Chicago
    is excluded from this module's validation pool until then. Default
    ON (`key_nudge=True`) as of 2026-08-12, the user's explicit decision
    based on the OTHER 4 songs' clean/positive results plus Gold's own
    validated win -- `key_nudge=False` opts back out (`--no-key-nudge`
    on the CLI)."""

    # circle-of-fifths pseudo-key probability table (usp's own, unchanged --
    # rows are pseudo keys, columns are the 12 pitch classes).
    KEY_TABLE = np.array([
        [0.19, 0.00, 0.21, 0.00, 0.21, 0.08, 0.00, 0.13, 0.00, 0.11, 0.00, 0.07],
        [0.09, 0.21, 0.00, 0.17, 0.00, 0.18, 0.08, 0.00, 0.13, 0.00, 0.14, 0.00],
        [0.00, 0.07, 0.19, 0.00, 0.18, 0.00, 0.17, 0.08, 0.00, 0.19, 0.00, 0.12],
        [0.11, 0.00, 0.10, 0.26, 0.00, 0.16, 0.00, 0.17, 0.07, 0.00, 0.13, 0.00],
        [0.00, 0.14, 0.00, 0.07, 0.28, 0.00, 0.17, 0.00, 0.13, 0.06, 0.00, 0.15],
        [0.15, 0.00, 0.16, 0.00, 0.13, 0.17, 0.00, 0.16, 0.00, 0.13, 0.10, 0.00],
        [0.00, 0.15, 0.00, 0.16, 0.00, 0.12, 0.17, 0.00, 0.14, 0.00, 0.14, 0.12],
        [0.09, 0.00, 0.16, 0.00, 0.16, 0.00, 0.11, 0.17, 0.00, 0.16, 0.00, 0.15],
        [0.18, 0.07, 0.00, 0.15, 0.00, 0.14, 0.00, 0.09, 0.19, 0.00, 0.18, 0.00],
        [0.00, 0.18, 0.10, 0.00, 0.14, 0.00, 0.15, 0.00, 0.10, 0.17, 0.00, 0.16],
        [0.13, 0.00, 0.15, 0.09, 0.00, 0.16, 0.00, 0.18, 0.00, 0.12, 0.17, 0.00],
        [0.00, 0.11, 0.00, 0.19, 0.13, 0.00, 0.16, 0.00, 0.12, 0.00, 0.08, 0.21],
    ])

    @classmethod
    def detect_key(cls, pitch_classes: List[int]) -> int:
        distribution = [pitch_classes.count(x) for x in range(12)]
        return int(np.argmax(cls.KEY_TABLE @ distribution))

    @classmethod
    def correct(cls, key: int, pitch_classes: List[int]) -> List[int]:
        out = list(pitch_classes)
        for i, pc in enumerate(out):
            if not cls.KEY_TABLE[key, pc]:
                prob_up = cls.KEY_TABLE[key, (pc + 1) % 12]
                prob_down = cls.KEY_TABLE[key, (pc - 1) % 12]
                out[i] = (pc + 1) % 12 if prob_up >= prob_down else (pc - 1) % 12
        return out


# --- core per-note pitch-class detection ---------------------------------

def _pred_for_note(
    start: float, end: float, frame_times: np.ndarray,
    midi: np.ndarray, conf: np.ndarray, voiced: np.ndarray, frame_dur: float,
    trim_sec: float, floor_pct: float,
) -> Optional[int]:
    """Pitch class (0-11) for one note's time window, or None if no
    voiced frame exists anywhere in it. Mirrors `note_detection.
    detect_notes`'s own per-note aggregation (trim -> floor -> confidence-
    weighted mode) but scoped to an ALREADY-KNOWN note window instead of
    a to-be-detected one."""
    i0 = np.searchsorted(frame_times, start, side="left")
    i1 = np.searchsorted(frame_times, end, side="right")
    if i1 <= i0:
        i1 = min(i0 + 1, len(frame_times))
    seg_midi = midi[i0:i1]
    seg_conf = conf[i0:i1]
    seg_voiced = voiced[i0:i1]
    mask = seg_voiced & ~np.isnan(seg_midi)
    if not mask.any():
        return None
    pitches = seg_midi[mask].tolist()
    confs = seg_conf[mask].tolist()
    if trim_sec > 0:
        pitches, confs = _trim_attack(pitches, confs, frame_dur, trim_sec)
    if floor_pct > 0:
        pitches, confs = _confidence_floor_filter(pitches, confs, floor_pct)
    if not pitches:
        return None
    return int(_weighted_mode_pitch(pitches, confs)) % 12


def _fill_gaps_nearest(preds: List[Optional[int]]) -> List[Optional[int]]:
    """Notes with no voiced signal at all borrow the temporally NEAREST
    scored neighbor's pitch class. Real-validated (this session, Gold) as
    the best available fallback for this specific failure mode -- three
    fancier alternatives (cascading to a second pitch source, same-lyric-
    word self-consistency voting, a relaxed-threshold retry) were all
    tried and scored WORSE; see [[project-usp-comparison-gold]]."""
    filled = list(preds)
    n = len(filled)
    for i in range(n):
        if filled[i] is not None:
            continue
        for d in range(1, n):
            if i - d >= 0 and filled[i - d] is not None:
                filled[i] = filled[i - d]
                break
            if i + d < n and filled[i + d] is not None:
                filled[i] = filled[i + d]
                break
    return filled


def compute_pitch_class_predictions(
    entries: List[Union[Syllable, LineBreak]],
    y: np.ndarray, sr: int, *,
    pitch_source: str = config.DEFAULT_PITCH_SOURCE,
    attack_trim_sec: float = DEFAULT_ATTACK_TRIM_SEC,
    confidence_floor_percentile: float = DEFAULT_CONFIDENCE_FLOOR_PERCENTILE,
    voicing_threshold: Optional[float] = DEFAULT_VOICING_THRESHOLD,
    fmin: float = 65.0, fmax: float = 1046.5, hop_length: int = 256, frame_length: int = 2048,
) -> List[Optional[int]]:
    """One pitch-class prediction (or None) per Syllable in `entries`, in
    order -- LineBreaks are skipped, not represented in the output list.
    `fmin`/`fmax`/`hop_length`/`frame_length` default to the exact same
    values `note_detection.detect_notes` uses, so results stay comparable
    with the rest of the pipeline's own pitch detection."""
    notes = [e for e in entries if isinstance(e, Syllable)]
    if not notes:
        return []
    if len(y) < frame_length:
        return [None] * len(notes)

    n_frames = 1 + (len(y) - frame_length) // hop_length
    frame_dur = hop_length / sr
    frame_times = np.arange(n_frames) * frame_dur

    source_kwargs = {}
    if pitch_source == "rmvpe":
        source_kwargs["device"] = config.RMVPE_DEVICE
    if pitch_source == "rmvpe" and voicing_threshold is not None:
        source_kwargs["voicing_threshold"] = voicing_threshold

    midi, conf, voiced = PITCH_SOURCES[pitch_source](
        y, sr, hop_length, frame_length, fmin, fmax, n_frames, **source_kwargs)

    return [
        _pred_for_note(note.start, note.end, frame_times, midi, conf, voiced, frame_dur,
                        attack_trim_sec, confidence_floor_percentile)
        for note in notes
    ]


def refresh_song_pitch(
    existing: ParsedSong, y: np.ndarray, sr: int, *,
    pitch_source: str = config.DEFAULT_PITCH_SOURCE,
    attack_trim_sec: float = DEFAULT_ATTACK_TRIM_SEC,
    confidence_floor_percentile: float = DEFAULT_CONFIDENCE_FLOOR_PERCENTILE,
    voicing_threshold: Optional[float] = DEFAULT_VOICING_THRESHOLD,
    key_nudge: bool = DEFAULT_KEY_NUDGE,
    log: Callable[[str], None] = print,
) -> Song:
    """Returns a new `Song` with every note's PITCH CLASS re-detected from
    `y`/`sr` -- timing, text, note count, and note order are all carried
    through from `existing` completely unchanged (only `Syllable.
    midi_note` is ever replaced, and only its pitch-class component: the
    original note's own OCTAVE is preserved, matching usp's own design --
    usp never predicts octave either, only chromatic class via `% 12` on
    both read and write)."""
    notes = [e for e in existing.entries if isinstance(e, Syllable)]
    if not notes:
        raise ValueError("Existing file has no notes to refresh.")

    preds = compute_pitch_class_predictions(
        existing.entries, y, sr, pitch_source=pitch_source, attack_trim_sec=attack_trim_sec,
        confidence_floor_percentile=confidence_floor_percentile, voicing_threshold=voicing_threshold,
    )

    n_unvoiced = sum(1 for p in preds if p is None)
    if n_unvoiced:
        log(f"{n_unvoiced}/{len(preds)} note(s) had no voiced signal at all in {pitch_source!r} -- "
            f"borrowing the nearest scored neighbor's pitch class for those.")
    filled = _fill_gaps_nearest(preds)

    if key_nudge:
        key = _KeyNudge.detect_key(filled)
        before = list(filled)
        filled = _KeyNudge.correct(key, filled)
        n_changed = sum(1 for a, b in zip(before, filled) if a != b)
        log(f"Key nudge: detected pseudo-key {key}, adjusted {n_changed} note(s).")

    out_entries: List[Union[Syllable, LineBreak]] = []
    note_i = 0
    for e in existing.entries:
        if isinstance(e, Syllable):
            new_pc = filled[note_i]
            note_i += 1
            new_midi = e.midi_note - (e.midi_note % 12) + new_pc
            out_entries.append(dataclasses.replace(e, midi_note=new_midi))
        else:
            out_entries.append(e)

    n_changed_total = sum(
        1 for e, out_e in zip(notes, (x for x in out_entries if isinstance(x, Syllable)))
        if e.midi_note != out_e.midi_note
    )
    log(f"Pitch refresh: {n_changed_total}/{len(notes)} note(s) changed pitch class.")

    return _song_from_existing(existing, out_entries, existing.gap_ms)


# --- CLI / pipeline glue (mirrors realign.py's own shape) ---------------

def resolve_pitch_refresh_output_path(existing_txt_path: Path, output_path_override: Optional[str]) -> Path:
    """Where a refreshed .txt gets written -- an explicit override if
    given, else a NEW file alongside the existing one, never the existing
    file itself. Same convention as `realign.resolve_realign_output_
    path`."""
    existing_txt_path = Path(existing_txt_path)
    return (Path(output_path_override).resolve() if output_path_override
            else existing_txt_path.with_name(f"{existing_txt_path.stem} {OUTPUT_MARKER}.txt"))


@dataclass
class PitchRefreshOptions:
    """Every knob `run_pitch_refresh_pipeline` needs, decoupled from
    argparse -- same shape/purpose as `RealignPipelineOptions`."""
    audio_file: Optional[str] = None
    work_dir: Optional[str] = None
    isolate_vocals: bool = True
    demucs_model: str = config.DEFAULT_DEMUCS_MODEL
    pitch_source: str = config.DEFAULT_PITCH_SOURCE
    attack_trim_sec: float = DEFAULT_ATTACK_TRIM_SEC
    confidence_floor_percentile: float = DEFAULT_CONFIDENCE_FLOOR_PERCENTILE
    voicing_threshold: Optional[float] = DEFAULT_VOICING_THRESHOLD
    key_nudge: bool = DEFAULT_KEY_NUDGE
    output_path: Optional[str] = None
    delete_work_files: bool = False
    batch: bool = False
    # GUI-only, never set by the CLI -- see config.PipelineOptions.
    # cancel_requested's own docstring.
    cancel_requested: Optional[Callable[[], bool]] = None


@dataclass
class PitchRefreshPipelineResult:
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None


def run_pitch_refresh_pipeline(input_dir: Path, existing_txt_path: Optional[Path], opts: PitchRefreshOptions,
                                *, log: Callable[[str], None] = print) -> PitchRefreshPipelineResult:
    """Thin wrapper so `opts.delete_work_files` is honored via `finally`
    regardless of which early-return failure path the body took -- mirrors
    `realign.run_realign_pipeline`'s own wrapper exactly."""
    try:
        return _run_pitch_refresh_pipeline_body(input_dir, existing_txt_path, opts, log=log)
    except config.PipelineCancelled:
        log("Cancelled by user.")
        return PitchRefreshPipelineResult(success=False, error="Cancelled by user.")
    finally:
        if opts.delete_work_files:
            from .main import delete_work_files as _delete_work_files
            wd = Path(opts.work_dir).resolve() if opts.work_dir else (Path(input_dir) / ".ultrastar_work")
            _delete_work_files(wd)


def _run_pitch_refresh_pipeline_body(input_dir: Path, existing_txt_path: Optional[Path],
                                      opts: PitchRefreshOptions,
                                      *, log: Callable[[str], None] = print) -> PitchRefreshPipelineResult:
    from .song_input import resolve_song_folder
    from .file_discovery import AmbiguousInputError, NoAudioSourceFoundError
    from .separation import isolate_vocals, SeparationError
    import librosa

    input_dir = Path(input_dir)
    if existing_txt_path is None:
        try:
            existing_txt_path = find_existing_txt_in_folder(input_dir, exclude_markers=_EXCLUDE_MARKERS)
        except AmbiguousExistingTxtError as e:
            return PitchRefreshPipelineResult(success=False, error=str(e))
        log(f"Auto-detected existing file: {existing_txt_path}")
    else:
        existing_txt_path = Path(existing_txt_path)

    work_dir = Path(opts.work_dir).resolve() if opts.work_dir else (input_dir / ".ultrastar_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        existing = parse_usdx_file(existing_txt_path)
    except UsdxParseError as e:
        return PitchRefreshPipelineResult(success=False, error=f"Could not parse {existing_txt_path}: {e}")

    try:
        resolved = resolve_song_folder(input_dir, work_dir, audio_file_override=opts.audio_file)
    except (AmbiguousInputError, NoAudioSourceFoundError) as e:
        return PitchRefreshPipelineResult(success=False, error=str(e))

    config.check_cancelled(opts.cancel_requested)
    if opts.isolate_vocals:
        log("Isolating vocals with Demucs...")
        try:
            audio_path = isolate_vocals(resolved.analysis_audio, work_dir, model=opts.demucs_model,
                                         cancel_requested=opts.cancel_requested)
        except SeparationError as e:
            return PitchRefreshPipelineResult(success=False, error=f"Vocal isolation failed: {e}")
    else:
        audio_path = resolved.analysis_audio

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    log(f"Refreshing pitch (source={opts.pitch_source!r}, "
        f"trim={opts.attack_trim_sec}s, floor={opts.confidence_floor_percentile}pct, "
        f"voicing_threshold={opts.voicing_threshold}, key_nudge={opts.key_nudge})...")
    song = refresh_song_pitch(
        existing, y, sr, pitch_source=opts.pitch_source, attack_trim_sec=opts.attack_trim_sec,
        confidence_floor_percentile=opts.confidence_floor_percentile,
        voicing_threshold=opts.voicing_threshold, key_nudge=opts.key_nudge, log=log,
    )

    output_path = resolve_pitch_refresh_output_path(existing_txt_path, opts.output_path)
    guard_error = check_output_not_existing_file(output_path, existing_txt_path)
    if guard_error is not None:
        return PitchRefreshPipelineResult(success=False, error=guard_error)
    write_song(song, output_path)
    log(f"Wrote {output_path}")
    return PitchRefreshPipelineResult(success=True, output_path=output_path)


def run_pitch_refresh_batch(parent_dir: Path, opts: PitchRefreshOptions,
                             *, log: Callable[[str], None] = print
                             ) -> List[tuple]:
    """Runs `run_pitch_refresh_pipeline` once per immediate subdirectory
    of `parent_dir`, auto-detecting each subfolder's own existing .txt.
    Mirrors `realign.run_realign_batch`'s shape exactly, including
    isolating one bad song's failure (even an unexpected exception) from
    aborting the rest."""
    parent_dir = Path(parent_dir)
    subdirs = sorted(p for p in parent_dir.iterdir() if p.is_dir())
    results: List[tuple] = []

    for i, sub in enumerate(subdirs, 1):
        if opts.cancel_requested is not None and opts.cancel_requested():
            log(f"Cancelled by user before {sub.name} ({i}/{len(subdirs)}).")
            break
        log(f"== Batch {i}/{len(subdirs)}: {sub.name} ==")
        try:
            result = run_pitch_refresh_pipeline(sub, None, opts, log=log)
        except Exception as e:
            log(f"  FAILED (unexpected error): {e}")
            result = PitchRefreshPipelineResult(success=False, error=str(e))
        if not result.success:
            log(f"  FAILED: {result.error}")
        results.append((sub.name, result))

    n_ok = sum(1 for _, r in results if r.success)
    log(f"\nBatch complete: {n_ok}/{len(results)} song(s) succeeded.")
    for name, result in results:
        status = "OK" if result.success else f"FAILED ({result.error})"
        log(f"  {name}: {status}")

    return results


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Pitch-only refresh: re-detect each note's PITCH CLASS in an EXISTING UltraStar "
                    ".txt from its audio (timing, text, and note count/order are never touched). Same "
                    "basic idea as the external ultrastar_pitch (usp) tool."
    )
    p.add_argument("input", help="Path to the song's folder. With --batch, this is a PARENT folder "
                                  "whose immediate subdirectories are each refreshed the same way.")
    p.add_argument("--batch", action="store_true",
                    help="Treat the positional argument as a PARENT folder: refresh each of its immediate "
                         "subdirectories (auto-detecting each one's own existing .txt). One song failing "
                         "does not abort the rest. Not allowed together with --existing-txt/--audio-file/"
                         "--work-dir.")
    p.add_argument("--existing-txt", dest="existing_txt_path", default=None,
                    help="Path to the existing UltraStar .txt to refresh. Optional in single-song mode -- "
                         "auto-detects the single .txt file in the input folder if omitted.")
    p.add_argument("--output", dest="output_path", default=None,
                    help="Where to write the refreshed .txt (default: alongside the existing file, named "
                         f"'<name> {OUTPUT_MARKER}.txt'). The existing file is always treated as "
                         "read-only -- refuses to run if --output resolves to that same path.")
    p.add_argument("--audio-file", default=None, help="Same as the main pipeline's --audio-file.")
    p.add_argument("--work-dir", default=None, help="Same as the main pipeline's --work-dir.")
    p.add_argument("--isolate-vocals", dest="isolate_vocals", action="store_true", default=True,
                    help="Run Demucs vocal isolation and detect pitch from the isolated vocals instead of "
                         "the original mixed audio. Default: ON (flipped 2026-08-15) -- the single-song "
                         "Gold finding that originally justified a mixed-audio default did not generalize "
                         "on a 9-song regression test: for rmvpe, mixed vs. isolated is a coin flip; for "
                         "swiftf0, mixed is a clear regression on 8/9 songs (up to -54pp). Requires CUDA "
                         "(Demucs runs with -d cuda) -- use --no-isolate-vocals for this module's original "
                         "fully-CPU, usp-matching behavior.")
    p.add_argument("--no-isolate-vocals", dest="isolate_vocals", action="store_false",
                    help="Detect pitch from the original mixed audio instead (the old default) -- matches "
                         "usp's own default and needs no CUDA.")
    p.add_argument("--demucs-model", default=config.DEFAULT_DEMUCS_MODEL)
    p.add_argument("--pitch-source", choices=sorted(PITCH_SOURCES.keys()), default=config.DEFAULT_PITCH_SOURCE)
    p.add_argument("--attack-trim-sec", type=float, default=DEFAULT_ATTACK_TRIM_SEC,
                    help="Drop this many seconds from the start of each note before aggregating its pitch "
                         "-- counters a real measured flat-attack bias. Default: this module's own tuned "
                         f"recipe ({DEFAULT_ATTACK_TRIM_SEC}s) -- real-validated on Gold plus 4 more "
                         "ground-truth songs with no regression, see CLAUDE.md / project memory.")
    p.add_argument("--confidence-floor-percentile", type=float, default=DEFAULT_CONFIDENCE_FLOOR_PERCENTILE,
                    help="Drop the bottom N%% of each note's frames by confidence before aggregating. "
                         f"Default: this module's own tuned recipe ({DEFAULT_CONFIDENCE_FLOOR_PERCENTILE}) "
                         "-- see --attack-trim-sec's own help for the same validation note.")
    p.add_argument("--voicing-threshold", type=float, default=DEFAULT_VOICING_THRESHOLD,
                    help="Raises the pitch source's own internal voicing-confidence gate above its default "
                         "(only affects 'rmvpe', which exposes this knob). Default: this module's "
                         f"own tuned recipe ({DEFAULT_VOICING_THRESHOLD}).")
    p.add_argument("--key-nudge", dest="key_nudge", action="store_true", default=DEFAULT_KEY_NUDGE,
                    help="Apply a conservative +-1-semitone pseudo-key nudge (usp-inspired) after "
                         "detection. Default: ON (2026-08-12, user's explicit decision) -- real multi-song "
                         "regression testing found this generalizes cleanly on 4 ground-truth songs (net "
                         "positive, never a real regression) plus a validated clean win on Gold. One "
                         "apparent severe regression on a 5th song (Chicago) is INVALIDATED, not confirmed "
                         "-- that song was separately found to have real data-quality problems (its MXL "
                         "reference is missing roughly half the song), so it's unknown whether the nudge "
                         "actually hurt there or was correctly compensating for bad data. See CLAUDE.md / "
                         "project memory. --no-key-nudge opts back out.")
    p.add_argument("--no-key-nudge", dest="key_nudge", action="store_false")
    p.add_argument("--delete-work-files", action="store_true",
                    help="Delete <input-folder>/.ultrastar_work after refreshing. Default: OFF (keeps "
                         "cached separation for re-runs).")
    return p


def _opts_from_args(args) -> PitchRefreshOptions:
    return PitchRefreshOptions(
        audio_file=args.audio_file, work_dir=args.work_dir, isolate_vocals=args.isolate_vocals,
        demucs_model=args.demucs_model, pitch_source=args.pitch_source,
        attack_trim_sec=args.attack_trim_sec, confidence_floor_percentile=args.confidence_floor_percentile,
        voicing_threshold=args.voicing_threshold, key_nudge=args.key_nudge, output_path=args.output_path,
        delete_work_files=args.delete_work_files, batch=args.batch,
    )


def run(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    args = build_arg_parser().parse_args(argv)

    if args.batch:
        incompatible = []
        if args.existing_txt_path:
            incompatible.append("--existing-txt")
        if args.audio_file:
            incompatible.append("--audio-file")
        if args.work_dir:
            incompatible.append("--work-dir")
        if incompatible:
            print(f"--batch is not allowed together with {', '.join(incompatible)} "
                  f"(a single override doesn't make sense across multiple songs).", file=sys.stderr)
            return 1

    input_dir = Path(args.input).resolve()
    opts = _opts_from_args(args)

    if args.batch:
        results = run_pitch_refresh_batch(input_dir, opts)
        return 0 if all(r.success for _, r in results) else 2

    existing_txt_path = Path(args.existing_txt_path).resolve() if args.existing_txt_path else None
    result = run_pitch_refresh_pipeline(input_dir, existing_txt_path, opts)
    if not result.success:
        print(result.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run())
