"""Pass 4 (optional): confirms or corrects pass-3 syllable pitches using a MusicXML file.

Only a local companion file is used (file_discovery.find_companions), no network
fetch. Multiple reference files are all applied sequentially, each one's corrections
feeding into the next (`apply_musicxml_references`). Corrects pitch CLASS only, never
octave -- matches how UltraStar Deluxe itself scores (octave-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import difflib
from collections import Counter, defaultdict

from . import config
from .models import Syllable
from .text_normalize import normalize_word as _normalize


@dataclass
class MusicXMLCorrection:
    syllable_index: int
    text: str
    old_pitch: int          # UltraStar convention (MIDI - 60)
    new_pitch: int
    mxl_pitch_class: int    # 0-11


@dataclass
class MusicXMLStats:
    mxl_path: str = ""
    part_names_used: List[str] = None
    n_vocal_notes: int = 0
    n_comparable_syllables: int = 0
    n_matched: int = 0
    match_ratio: float = 0.0
    calibration_offset: Optional[int] = None    # pitch-class semitones, 0-11
    calibration_confidence: float = 0.0          # fraction of matches agreeing with the chosen offset
    corrections: List[MusicXMLCorrection] = None
    skipped_reason: Optional[str] = None         # set if calibration wasn't trusted

    def __post_init__(self):
        if self.part_names_used is None:
            self.part_names_used = []
        if self.corrections is None:
            self.corrections = []



def load_vocal_notes(
    mxl_path: str, preferred_part_name: Optional[str] = None,
) -> Tuple[List[Tuple[float, int, str]], List[str]]:
    """Parses a MusicXML/.mxl file, returns (notes, part_names_used): notes is a
    time-ordered list of (score_offset, absolute_midi, normalized_lyric_text) for
    every lyric-bearing, non-chord note in the vocal part(s). If multiple parts have
    lyrics, merges them, preferring `preferred_part_name` where present and falling
    back to whichever other part has a note at that offset (or the part with the
    most notes, if no preference given)."""
    import music21

    score = music21.converter.parse(mxl_path)
    lyric_parts = []
    for part in score.parts:
        notes = list(part.flatten().notes)
        n_with_lyrics = sum(1 for n in notes if n.lyrics and not n.isChord)
        if n_with_lyrics > 0:
            lyric_parts.append((part, n_with_lyrics))

    if not lyric_parts:
        return [], []

    chosen = [p for p, _ in lyric_parts]  # all lyric-bearing parts, needed to fill gaps when merging

    by_offset: dict = defaultdict(dict)
    for part in chosen:
        for n in part.flatten().notes:
            if n.isChord or not n.lyrics:
                continue
            by_offset[float(n.offset)][part.partName] = (n.pitch.midi, _normalize(n.lyrics[0].text))

    valid_names = {p.partName for p in chosen}
    primary_name = preferred_part_name if preferred_part_name in valid_names else None
    if primary_name is None and len(chosen) > 1:
        # No matching hint: fall back to the part with the most notes.
        primary_name = max(((p.partName, cnt) for p, cnt in lyric_parts), key=lambda x: x[1])[0]

    notes_out = []
    for off in sorted(by_offset):
        parts_here = by_offset[off]
        if primary_name is not None and primary_name in parts_here:
            midi, text = parts_here[primary_name]
        else:
            midi, text = next(iter(parts_here.values()))
        if text:
            notes_out.append((off, midi, text))

    return notes_out, [p.partName for p in chosen]


def nearest_pitch_for_class(our_pitch: int, target_pc: int) -> int:
    """Nearest UltraStar-convention pitch to `our_pitch` with pitch CLASS `target_pc`
    (0-11) -- moves at most +-6 semitones, never jumping octave."""
    our_pc = (our_pitch + 60) % 12
    if our_pc == target_pc:
        return our_pitch
    diff = (target_pc - our_pc) % 12
    if diff > 6:
        diff -= 12
    return our_pitch + diff


def _calibrate_pitch_class(
    matches: List[Tuple[int, int, float]],  # (our_ultrastar_pitch, mxl_absolute_midi, our_confidence)
    min_calibration_samples: int,
    min_calibration_confidence: float,
    force_calibration: bool,
    verbose: bool,
    log_prefix: str = "[musicxml]",
) -> Tuple[Optional[int], float, List[Tuple[int, int, float]], Optional[str]]:
    """Shared per-song pitch-class calibration-offset logic (full population, then a
    high-confidence-subset retry, then optional force_calibration fallback), used by
    `apply_musicxml_reference` and `pitch_refresh.apply_mxl_pitch_reference`.
    Returns (offset, confidence, population_used, skipped_reason); offset is None
    (with skipped_reason set) when calibration isn't trusted and not forced."""
    if len(matches) < min_calibration_samples:
        reason = (f"only {len(matches)} matched note(s) (< {min_calibration_samples} required) -- "
                   f"not enough to trust a calibration offset")
        if verbose:
            print(f"{log_prefix} skipping correction: {reason}")
        return None, 0.0, [], reason

    def _pc_offset(m):
        our_p, mxl_p, _ = m
        return (mxl_p - (our_p + 60)) % 12

    def _best_offset(population):
        counts = Counter(_pc_offset(m) for m in population)
        offset, n_agree = counts.most_common(1)[0]
        return offset, n_agree / len(population)

    calibration, confidence = _best_offset(matches)
    calibration_population = matches

    if confidence < min_calibration_confidence:
        sorted_by_conf = sorted(matches, key=lambda m: -m[2])
        top_half = sorted_by_conf[:max(min_calibration_samples, len(matches) // 2)]
        alt_calibration, alt_confidence = _best_offset(top_half)
        cleared_alt_bar = alt_confidence >= config.MUSICXML_MIN_CALIBRATION_CONFIDENCE_HIGH_CONF_SUBSET
        use_alt = cleared_alt_bar or (force_calibration and alt_confidence > confidence)
        if use_alt:
            if verbose:
                print(f"{log_prefix} full-population calibration too weak ({confidence:.0%}) -- "
                      f"retrying with top {len(top_half)}/{len(matches)} matches by our own "
                      f"confidence: {alt_calibration:+d} semitones, {alt_confidence:.0%} agreement")
            calibration, confidence = alt_calibration, alt_confidence
            calibration_population = top_half
        elif not force_calibration:
            reason = (f"no clear per-song calibration offset (best candidate {calibration} semitones "
                       f"covers {confidence:.0%} of all {len(matches)} matches, "
                       f"{alt_confidence:.0%} of the top {len(top_half)} by our own confidence -- "
                       f"neither clears the required bar)")
            if verbose:
                print(f"{log_prefix} skipping correction: {reason}")
            return None, 0.0, [], reason
        elif verbose:
            print(f"{log_prefix} force_calibration: neither candidate cleared its bar "
                  f"({confidence:.0%} full population, {alt_confidence:.0%} high-confidence subset) "
                  f"-- proceeding anyway with the full-population offset {calibration:+d}")

    if verbose:
        print(f"{log_prefix} calibration offset: {calibration:+d} semitones (pitch-class), "
              f"{confidence:.0%} agreement over {len(calibration_population)} match(es)"
              f"{' [FORCED]' if force_calibration and confidence < min_calibration_confidence else ''}")

    return calibration, confidence, calibration_population, None


def apply_musicxml_reference(
    syllables: List[Syllable],
    mxl_path: str,
    preferred_part_name: Optional[str] = None,
    min_calibration_samples: int = config.MUSICXML_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.MUSICXML_MIN_CALIBRATION_CONFIDENCE,
    force_calibration: bool = config.ENABLE_MUSICXML_FORCE_CALIBRATION,
    verbose: bool = True,
    debug_log=None,
) -> Tuple[List[Syllable], MusicXMLStats]:
    """Aligns `syllables` against the vocal-melody notes in `mxl_path` by lyric text
    (difflib), and corrects a syllable's pitch class (never octave or timing) where
    they disagree -- only once a per-song calibration offset clears the confidence
    bar (>= min_calibration_samples matches, modal offset covering
    >= min_calibration_confidence of them); otherwise a no-op, `stats.skipped_reason`
    explains why. `force_calibration=True` (default, unconditional) skips the
    confidence bar and applies the best available offset regardless -- useful when
    pass-1 pitch itself is unreliable enough that the confidence bar can never clear."""
    stats = MusicXMLStats(mxl_path=mxl_path)

    vocal_notes, part_names = load_vocal_notes(mxl_path, preferred_part_name)
    stats.part_names_used = part_names
    stats.n_vocal_notes = len(vocal_notes)
    if not vocal_notes:
        stats.skipped_reason = "no lyric-bearing notes found in the MusicXML file"
        return syllables, stats

    mxl_words = [text for _, _, text in vocal_notes]
    mxl_midi = [midi for _, midi, _ in vocal_notes]

    comparable_indices = [i for i, s in enumerate(syllables) if _normalize(s.text)]
    our_words = [_normalize(syllables[i].text) for i in comparable_indices]
    stats.n_comparable_syllables = len(comparable_indices)

    sm = difflib.SequenceMatcher(a=our_words, b=mxl_words, autojunk=False)
    stats.match_ratio = sm.ratio()

    matches = []  # (syllable_index, our_ultrastar_pitch, mxl_absolute_pitch, our_confidence)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            syl_idx = comparable_indices[i1 + k]
            matches.append((syl_idx, syllables[syl_idx].midi_note, mxl_midi[j1 + k], syllables[syl_idx].confidence))
    stats.n_matched = len(matches)

    if verbose:
        print(f"[musicxml] {mxl_path}: parts used={part_names}, {len(vocal_notes)} vocal notes, "
              f"{len(matches)}/{len(comparable_indices)} syllables matched by lyric text "
              f"(ratio={sm.ratio():.3f})")

    # Calibrate at the pitch-class level (mod 12) to absorb transposition and
    # octave-only notation inconsistency.
    calibration, confidence, _, skipped_reason = _calibrate_pitch_class(
        [(our_p, mxl_p, conf) for (_, our_p, mxl_p, conf) in matches],
        min_calibration_samples, min_calibration_confidence, force_calibration, verbose,
    )
    stats.calibration_offset = calibration
    stats.calibration_confidence = confidence
    if calibration is None:
        stats.skipped_reason = skipped_reason
        return syllables, stats

    # Applies to every matched syllable once calibration is trusted, regardless
    # of that syllable's own confidence.
    new_syllables = list(syllables)
    for syl_idx, our_p, mxl_p, _ in matches:
        target_pc = (mxl_p - calibration) % 12
        new_pitch = nearest_pitch_for_class(our_p, target_pc)
        if new_pitch == our_p:
            continue
        stats.corrections.append(MusicXMLCorrection(syl_idx, syllables[syl_idx].text, our_p, new_pitch, target_pc))
        old = new_syllables[syl_idx]
        new_syllables[syl_idx] = Syllable(
            text=old.text, start=old.start, end=old.end, midi_note=new_pitch,
            is_word_start=old.is_word_start, note_type=old.note_type, line_id=old.line_id,
            # Boosted (not to 1.0) since it's now confirmed by a second, inferred source.
            confidence=max(old.confidence, config.MUSICXML_CORRECTED_CONFIDENCE),
        )
        if debug_log is not None:
            debug_log.line(f"[musicxml] {old.text!r} @ {old.start:.2f}s: pitch {our_p:+d} -> {new_pitch:+d} "
                            f"(MXL pitch class {target_pc}, calibration {calibration:+d})")

    if verbose:
        print(f"[musicxml] corrected {len(stats.corrections)}/{len(matches)} matched syllable(s)")

    return new_syllables, stats


def apply_musicxml_references(
    syllables: List[Syllable],
    mxl_paths: List[str],
    preferred_part_name: Optional[str] = None,
    min_calibration_samples: int = config.MUSICXML_MIN_CALIBRATION_SAMPLES,
    min_calibration_confidence: float = config.MUSICXML_MIN_CALIBRATION_CONFIDENCE,
    force_calibration: bool = config.ENABLE_MUSICXML_FORCE_CALIBRATION,
    verbose: bool = True,
    debug_log=None,
) -> Tuple[List[Syllable], List[MusicXMLStats]]:
    """Applies `apply_musicxml_reference` for every path in `mxl_paths` in order, each
    one's output feeding into the next; each file gets its own independent
    calibration, and a file that can't calibrate is skipped without blocking the
    others. Returns (final_syllables, stats_per_file), same order as `mxl_paths`."""
    all_stats: List[MusicXMLStats] = []
    for path in mxl_paths:
        if verbose and len(mxl_paths) > 1:
            print(f"[musicxml] -- file {len(all_stats) + 1}/{len(mxl_paths)}: {path} --")
        try:
            syllables, stats = apply_musicxml_reference(
                syllables, path, preferred_part_name=preferred_part_name,
                min_calibration_samples=min_calibration_samples,
                min_calibration_confidence=min_calibration_confidence,
                force_calibration=force_calibration,
                verbose=verbose, debug_log=debug_log,
            )
        except Exception as e:
            # A file that fails to parse shouldn't take down the whole run.
            stats = MusicXMLStats(mxl_path=path, skipped_reason=f"failed to parse: {e}")
            if verbose:
                print(f"[musicxml] {path}: skipped -- failed to parse: {e}")
        all_stats.append(stats)
    return syllables, all_stats
