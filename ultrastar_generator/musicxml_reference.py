"""Pass 4 (optional): confirms or corrects pass-3 syllable pitches using
a MusicXML file (a sheet-music transcription of the same song, e.g.
hand-downloaded from MuseScore -- see CLAUDE.md's "MuseScore reference
data" section for why this isn't fetched automatically).

No automatic INTERNET lookup/fetch exists, unlike lyrics_lookup.py's
LRCLIB call -- MuseScore access was found to be actively blocked
platform-side (see CLAUDE.md). But a file already sitting in the song's
own folder IS auto-detected (file_discovery.find_companions, matched by
extension: .mxl/.musicxml/.xml -- see that module for why basename
matching, used for video/cover, doesn't work for these), same as this
project already does for video/cover companion files. `main.py` only
falls back to auto-detection when `--musicxml-reference` isn't given
explicitly; an explicit path always wins.

If more than one reference file is found (or given), ALL of them are
tried -- see `apply_musicxml_references` (plural). Different arrangements
of the same song often lyric-tag different, only partially-overlapping
portions (confirmed on Once Upon A Dream: one file covered 52.6% of the
song, a second, different arrangement covered a different 25.1% -- using
only one would leave real coverage on the table the other file has).
Applied sequentially, each file's corrections feeding into the next, so
coverage accumulates across files rather than only the first (or a
single "best") file's own reach.

Real validation this pass is based on (2026-08-08, manually downloaded
MXL files for several already-validated songs, compared against their
existing trusted ground truth): once a per-song PITCH-CLASS calibration
offset is removed (arrangements are routinely transposed, or use
inconsistent absolute octave notation -- neither is a real error), sheet
music vocal-melody data agrees with ground truth on pitch class 93-98%
of the time, far above any of this project's own audio-only pitch
sources. Coverage (how much of the song a given arrangement actually
lyric-tags) varies a lot more, 23-91% depending on the arrangement.

Deliberately calibrates and corrects at the PITCH-CLASS level only (never
absolute octave) for two reasons: (1) sheet-music octave notation across
different parts/arrangements of the same song was found to be internally
inconsistent even when transposition-corrected (e.g. Gaston: merging all
3 vocal parts gave a bimodal +0/-12 semitone split against the SAME
ground truth), so trusting it for octave decisions would be unfounded;
(2) our own audio-derived pitch already comes from a real physical
register the singer used, which pitch-class-only correction never
overrides -- only the semitone-within-register gets nudged. This also
matches how UltraStar Deluxe itself scores: pitch CLASS, octave-agnostic
(confirmed by the user) -- so this is the level of accuracy that actually
matters for the shipped output.
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
    skipped_reason: Optional[str] = None         # set if calibration wasn't trusted -- no corrections applied

    def __post_init__(self):
        if self.part_names_used is None:
            self.part_names_used = []
        if self.corrections is None:
            self.corrections = []



def load_vocal_notes(
    mxl_path: str, preferred_part_name: Optional[str] = None,
) -> Tuple[List[Tuple[float, int, str]], List[str]]:
    """Parses a MusicXML/.mxl file and returns (notes, part_names_used):
    notes is a time-ordered list of (score_offset, absolute_midi,
    normalized_lyric_text) for every lyric-bearing, non-chord note in
    whichever part(s) carry the vocal melody.

    Part selection:
      - Exactly one part has lyrics on any of its notes -> that part alone
        (the common case: a "Piano" or "Voice"-labeled part carrying the
        lead melody+lyrics, everything else is pure accompaniment).
      - Multiple parts have lyrics (a duet/ensemble arrangement) ->
        merged: if `preferred_part_name` matches a part by name, that
        part's own notes are used wherever present, falling back to
        whichever other lyric-bearing part has a note at a given score
        offset where the preferred part has none (real case: Gaston,
        where the "Gaston" part alone only covered 23% of his sung lines
        despite being the character's own line -- merging with the other
        2 vocal parts raised that to 88%, since they're mostly unison).
        With no hint given, falls back to the single lyric-bearing part
        with the MOST total notes as an imperfect generic default -- see
        this function's caller-facing docs for why passing the hint when
        known (e.g. the character's own name, if the arrangement labels
        parts that way) gives better results than this fallback.
    """
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

    # `chosen` is always EVERY lyric-bearing part -- merging needs all of
    # them available to fill gaps. `preferred_part_name` only decides
    # which one wins at an offset where more than one has a note (see
    # `primary_name` below) -- it must NOT narrow `chosen` down to just
    # that one part, or merging never happens at all (a real bug found
    # during validation: with a part-name hint given, this used to
    # silently drop back to single-part coverage instead of merging).
    chosen = [p for p, _ in lyric_parts]

    by_offset: dict = defaultdict(dict)
    for part in chosen:
        for n in part.flatten().notes:
            if n.isChord or not n.lyrics:
                continue
            by_offset[float(n.offset)][part.partName] = (n.pitch.midi, _normalize(n.lyrics[0].text))

    valid_names = {p.partName for p in chosen}
    primary_name = preferred_part_name if preferred_part_name in valid_names else None
    if primary_name is None and len(chosen) > 1:
        # No hint (or the hint didn't match any real part name): fall
        # back to the part with the most total notes as primary.
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
    """Nearest UltraStar-convention pitch (MIDI - 60) to `our_pitch` with
    pitch CLASS `target_pc` (0-11) -- moves by at most +-6 semitones,
    staying in the same octave region `our_pitch` is already in rather
    than jumping to a distant octave that happens to share the target
    class. Shared by `apply_musicxml_reference` and
    `pitch_refresh.apply_mxl_pitch_reference` (same "correct pitch class
    only, never octave" rule both places)."""
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
    """Shared by `apply_musicxml_reference` (matches syllable-for-syllable
    via `load_vocal_notes`) and `pitch_refresh.apply_mxl_pitch_reference`
    (matches word-for-word via `mxl_lrc_generator.load_mxl_vocal_words`,
    then position-aligns each matched word's own syllables) -- both need
    the exact same per-song PITCH-CLASS calibration-offset logic (full
    population, then a high-confidence-subset retry, then optional
    force_calibration fallback), just fed from a different alignment
    mechanism. See `apply_musicxml_reference`'s own docstring for the
    full rationale (real Gaston-based validation of the two-tier retry).

    Returns (offset, confidence, population_used, skipped_reason) --
    offset is None (with skipped_reason set) whenever calibration isn't
    trusted and `force_calibration` is False."""
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
    """Aligns `syllables` (pass 3's final output) against the vocal-melody
    notes in `mxl_path` by lyric text (same whole-sequence difflib
    technique lyrics_lookup.py uses for reference-lyric alignment), and
    corrects a syllable's PITCH CLASS (never its octave or timing) where
    the two disagree -- but only once a per-song calibration offset
    (transposition and/or octave-notation quirks) is established with
    real confidence (>= min_calibration_samples matched notes, and the
    modal offset accounting for >= min_calibration_confidence of them);
    otherwise this is a no-op and `stats.skipped_reason` explains why.

    `force_calibration=True` (ON by default, `config.
    ENABLE_MUSICXML_FORCE_CALIBRATION` -- unconditional in both callers as
    of 2026-08-17, no CLI/GUI off-switch anymore) skips the confidence bar
    entirely and always applies the
    best available calibration offset (full population, or the high-
    confidence-subset fallback if that was tried), however weak.
    Validated real end-to-end on all 7 MXL-having songs in the test set
    before being made the default: 0 regressions (4 songs unaffected --
    their normal calibration already clears the bar on its own, so this
    is provably a no-op for them), 1 small real gain (+1.9pp), 2 large
    real gains (+21.6pp, +19.0pp). Built for a specific real case: songs
    where OUR OWN pass-1 pitch detection is confirmed unreliable for
    acoustic reasons unrelated to any pitch-source choice
    (real case, 2026-08-08: little_mermaid/jungle_book_bare_necessities
    -- FOUR independently-trained/architected pitch estimators (pyin,
    CREPE-class, RMVPE, SwiftF0, PENN) all converged on the SAME wrong
    answer, pointing at genuine acoustic ambiguity in rough/character
    vocal production, not a detector-choice problem) -- for those songs
    the normal confidence bar can never be met, because it's measuring
    agreement against a baseline that's the actual problem. When our own
    pitch is this unreliable, an MXL reference's pitch is a better bet
    even calibrated with low confidence than trusting pass 1 at all.

    Never touches timing. Never guesses an absolute octave -- a
    correction only ever moves a syllable's pitch to the nearest MIDI
    value with the target pitch class, staying within a few semitones of
    where our own (audio-derived, real-register) detection already had
    it -- this stays true even under force_calibration, since octave
    doesn't affect real UltraStar scoring anyway (pitch-class only).
    """
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

    # Calibrate at the PITCH-CLASS level (mod 12) -- absorbs both genuine
    # transpositions (e.g. +2, +5 semitones, confirmed on real files) and
    # octave-only notation inconsistency (e.g. -12), which a plain
    # semitone-offset calibration would treat as conflicting evidence.
    # See `_calibrate_pitch_class`'s own docstring for the two-tier
    # retry/force_calibration logic (real Gaston-based validation: full-
    # population 39.5% agreement vs. 46.1% over the top-confidence half,
    # same winning offset both times, just less noise around it).
    calibration, confidence, _, skipped_reason = _calibrate_pitch_class(
        [(our_p, mxl_p, conf) for (_, our_p, mxl_p, conf) in matches],
        min_calibration_samples, min_calibration_confidence, force_calibration, verbose,
    )
    stats.calibration_offset = calibration
    stats.calibration_confidence = confidence
    if calibration is None:
        stats.skipped_reason = skipped_reason
        return syllables, stats

    # Correction applies to EVERY matched syllable once calibration is
    # trusted, regardless of that syllable's own confidence -- a
    # disagreement at a low-confidence syllable is exactly the case this
    # is FOR (our own detection is least trustworthy there), and a
    # disagreement at a high-confidence syllable, once a real per-song
    # calibration is established, is still evidence of an actual error
    # rather than something to leave alone just because we were sure.
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
            # Boosted, not left as-is -- this pitch is now independently
            # confirmed by a second source, so a syllable that came in
            # low-confidence (the case this whole mechanism targets)
            # shouldn't still read as uncertain after being corrected.
            # Not set to 1.0: still a DIFFERENT source's data, calibrated
            # by inference, not a direct pass-1 acoustic measurement.
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
    """Applies `apply_musicxml_reference` for EVERY path in `mxl_paths`,
    in order, each one's output syllables feeding into the next --
    coverage accumulates across files rather than stopping at whichever
    file happens to be tried first. Each file gets its OWN independent
    calibration (different arrangements can be transposed differently,
    or have different octave-notation quirks -- see the module
    docstring), so a file that can't establish confident calibration is
    skipped on its own (same graceful no-op as the single-file function)
    without blocking the others. Returns (final_syllables, stats_per_file)
    -- same order as `mxl_paths`.
    """
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
            # A companion file that LOOKED like a usable reference (right
            # extension, or passed find_companions' MusicXML content
            # sniff) but still fails to parse shouldn't take down the
            # whole run -- skip just this file, same as a low-confidence
            # calibration would.
            stats = MusicXMLStats(mxl_path=path, skipped_reason=f"failed to parse: {e}")
            if verbose:
                print(f"[musicxml] {path}: skipped -- failed to parse: {e}")
        all_stats.append(stats)
    return syllables, all_stats
