"""Rebases an existing UltraStar .txt's #GAP so its first note lands on
beat 0 -- the format's own hard invariant (see usdx_writer.py: every
generated song's first note is always beat 0 by construction, since GAP
is defined as that note's own start time). Pure integer-beat arithmetic:
every note/line-break's own beat NUMBER just shifts by a constant, and
#GAP is adjusted by the equivalent number of milliseconds to compensate
-- BPM and every note's real-time position, pitch, text, and the note
sequence itself are completely unchanged. No audio, ASR, or CUDA needed
at all -- the cheapest mode in the whole project.

Deliberately operates on the file's RAW INTEGER beat values via regex
(the same patterns usdx_parser.py uses), not via usdx_parser's own
seconds-converted ParsedSong + usdx_writer's render/re-quantize pipeline
-- a first attempt built on that round-trip (parse to seconds, rebuild a
Song with a new gap_ms, re-render) looked correct in principle (GAP is
defined by the exact same `first_syllable.start * 1000` rule main.py/
realign.py already use) but FAILED real validation: re-quantizing a
value that came from an EXACT integer beat, through
seconds_to_beat_length's `math.floor()`, can round a beat length DOWN by
one whenever float imprecision leaves the recovered seconds value a hair
under the true integer boundary (e.g. 11.999999999997 beats instead of
12.0) -- a classic floor-of-a-should-be-integer trap. Confirmed on the
real "Stars" ground-truth file: several note lengths silently lost a
beat, and `_quantize_entries`'s own non-overlap push-forward
("occupied_until") then cascaded that single-beat loss into much larger
drift on subsequent notes/line-breaks (one line-break landed 1.045s /
15 beats off). Pure integer addition on the raw beat values -- no float
round-trip anywhere except the single unavoidable GAP-in-milliseconds
rounding -- has no such failure mode.

Motivating real case: a hand-corrected "Stars" ground-truth file needed
its first note moved to beat -14 to match the real audio (the file's
existing GAP was for a slightly later moment than the true vocal onset).
This mode is the general-purpose way to take a file that was authored
(or manually corrected) with a nonzero first-note beat and convert it
back into this project's own beat-0 convention, without touching any
note's real audio timing, pitch, text, or the note sequence itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .tempo import beat_duration_ms
from .usdx_parser import UsdxParseError, _NOTE_RE, _BREAK_RE, _TAG_RE, _parse_bpm
from .realign import find_existing_txt_in_folder, AmbiguousExistingTxtError


@dataclass
class FixStartNoteBeatResult:
    success: bool
    error: str = ""
    output_path: Optional[Path] = None
    old_gap_ms: int = 0
    new_gap_ms: int = 0
    beat_shift: int = 0  # the first note's ORIGINAL beat; 0 means no change was needed


def fix_start_note_beat_text(text: str) -> Tuple[str, FixStartNoteBeatResult]:
    """Pure function over the file's own text -- returns (new_text,
    result-without-output_path). Raises UsdxParseError if #BPM or any
    note line can't be found (same failure-closed convention as
    usdx_parser.parse_usdx_file)."""
    lines = text.splitlines()

    bpm = None
    gap_ms = 0
    gap_line_idx = None
    bpm_line_idx = None
    for i, raw in enumerate(lines):
        line = raw.rstrip("\r")
        if not line:
            continue
        if not line.startswith("#"):
            break  # header ends at the first non-tag, non-blank line
        m = _TAG_RE.match(line)
        if not m:
            continue
        name = m.group(1).upper()
        if name == "BPM":
            bpm = _parse_bpm(m.group(2))
            bpm_line_idx = i
        elif name == "GAP":
            try:
                gap_ms = int(round(float(m.group(2).strip().replace(",", "."))))
            except ValueError:
                raise UsdxParseError(f"Invalid #GAP value: {m.group(2)!r}")
            gap_line_idx = i
    if bpm is None:
        raise UsdxParseError("Missing #BPM tag")

    # The very first note line in the file is the anchor, regardless of
    # a possible P1/P2 duet marker -- both parts share one GAP, so only
    # ONE global shift is ever needed.
    first_beat = None
    for raw in lines:
        m = _NOTE_RE.match(raw.rstrip("\r"))
        if m:
            first_beat = int(m.group(2))
            break
    if first_beat is None:
        raise UsdxParseError("No note lines found in file")

    if first_beat == 0:
        return text, FixStartNoteBeatResult(success=True, old_gap_ms=gap_ms, new_gap_ms=gap_ms, beat_shift=0)

    shift = -first_beat
    new_gap_ms = int(round(gap_ms + first_beat * beat_duration_ms(bpm)))

    out_lines: List[str] = []
    for i, raw in enumerate(lines):
        line = raw.rstrip("\r")
        if i == gap_line_idx:
            out_lines.append(f"#GAP:{new_gap_ms}")
            continue
        m = _NOTE_RE.match(line)
        if m:
            note_type, start_beat, length_beats, pitch, note_text = m.groups()
            out_lines.append(f"{note_type} {int(start_beat) + shift} {length_beats} {pitch} {note_text}")
            continue
        m = _BREAK_RE.match(line)
        if m:
            start_beat, end_beat = m.groups()
            if end_beat is not None:
                out_lines.append(f"- {int(start_beat) + shift} {int(end_beat) + shift}")
            else:
                out_lines.append(f"- {int(start_beat) + shift}")
            continue
        out_lines.append(line)
        if i == bpm_line_idx and gap_line_idx is None:
            # File had no #GAP tag at all (legitimate -- usdx itself
            # treats a missing #GAP as 0, see usdx_parser.py's own
            # comment) -- insert a real one now that one is needed.
            out_lines.append(f"#GAP:{new_gap_ms}")

    return "\n".join(out_lines) + "\n", FixStartNoteBeatResult(
        success=True, old_gap_ms=gap_ms, new_gap_ms=new_gap_ms, beat_shift=first_beat)


def resolve_fix_start_note_beat_output_path(existing_txt_path: Path, output_path_override: Optional[str]) -> Path:
    """Where a fixed .txt gets written -- an explicit override if given,
    else a NEW file alongside the existing one, never the existing file
    itself (input files are always read-only, see check_output_not_existing_file)."""
    existing_txt_path = Path(existing_txt_path)
    return (Path(output_path_override).resolve() if output_path_override
            else existing_txt_path.with_name(existing_txt_path.stem + " [START BEAT FIXED].txt"))


def check_output_not_existing_file(output_path: Path, existing_txt_path: Path) -> Optional[str]:
    """HARD guarantee, not just a default: the existing file being fixed
    is treated as read-only, unconditionally -- never overwritten, even
    if an explicit output path happens to resolve to the same path. No
    override exists for this on purpose -- don't add one."""
    if Path(output_path).resolve() == Path(existing_txt_path).resolve():
        return (f"Refusing to overwrite the existing file ({existing_txt_path}) -- "
                f"the input file is always treated as read-only. Pass a different --output path.")
    return None


def run_fix_start_note_beat(existing_txt_path: Path, output_path_override: Optional[str] = None,
                             log=print) -> FixStartNoteBeatResult:
    existing_txt_path = Path(existing_txt_path)
    output_path = resolve_fix_start_note_beat_output_path(existing_txt_path, output_path_override)
    guard_error = check_output_not_existing_file(output_path, existing_txt_path)
    if guard_error:
        return FixStartNoteBeatResult(success=False, error=guard_error)

    try:
        text = existing_txt_path.read_text(encoding="utf-8")
    except OSError as e:
        return FixStartNoteBeatResult(success=False, error=f"Could not read {existing_txt_path}: {e}")

    try:
        new_text, result = fix_start_note_beat_text(text)
    except UsdxParseError as e:
        return FixStartNoteBeatResult(success=False, error=f"Could not parse {existing_txt_path}: {e}")

    if result.beat_shift == 0:
        log(f"First note is already on beat 0 (GAP {result.old_gap_ms}) -- nothing to fix.")
    else:
        log(f"First note was on beat {result.beat_shift} -- shifting every beat by {-result.beat_shift} and "
            f"adjusting GAP {result.old_gap_ms} -> {result.new_gap_ms} to compensate. No note's real audio "
            f"timing changes.")

    output_path.write_text(new_text, encoding="utf-8")
    log(f"Wrote {output_path}")
    result.success = True
    result.output_path = output_path
    return result


def run_fix_start_note_beat_pipeline(input_dir: Path, existing_txt_path: Optional[Path] = None,
                                      output_path_override: Optional[str] = None,
                                      *, log=print) -> FixStartNoteBeatResult:
    """Same `(input_dir, existing_txt_path, ...)` calling shape as
    `realign.run_realign_pipeline`/`pitch_refresh.run_pitch_refresh_pipeline`
    -- auto-detects the folder's single .txt when `existing_txt_path` is
    None, so the GUI/CLI can dispatch to all three modes uniformly."""
    if existing_txt_path is None:
        try:
            existing_txt_path = find_existing_txt_in_folder(
                Path(input_dir), exclude_markers=("[REALIGNED]", "[START BEAT FIXED]"))
        except AmbiguousExistingTxtError as e:
            return FixStartNoteBeatResult(success=False, error=str(e))
    return run_fix_start_note_beat(existing_txt_path, output_path_override, log=log)


def run_fix_start_note_beat_batch(parent_dir: Path, log=print) -> List[Tuple[Path, FixStartNoteBeatResult]]:
    parent_dir = Path(parent_dir)
    results: List[Tuple[Path, FixStartNoteBeatResult]] = []
    for sub in sorted(p for p in parent_dir.iterdir() if p.is_dir()):
        try:
            existing_txt_path = find_existing_txt_in_folder(
                sub, exclude_markers=("[REALIGNED]", "[START BEAT FIXED]"))
        except AmbiguousExistingTxtError as e:
            log(f"[{sub.name}] skipped: {e}")
            results.append((sub, FixStartNoteBeatResult(success=False, error=str(e))))
            continue
        log(f"=== {sub.name} ===")
        result = run_fix_start_note_beat(existing_txt_path, log=log)
        results.append((sub, result))
    return results


# --- CLI -----------------------------------------------------------------

def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Rebase an existing UltraStar .txt's #GAP so its first note lands on beat 0 -- "
                     "pure GAP/beat-grid arithmetic, no audio needed. Never changes any note's real "
                     "timing, pitch, text, or the note sequence itself.")
    p.add_argument("input", help="Folder containing the .txt to fix (auto-detected), or a direct path "
                                  "to the .txt itself.")
    p.add_argument("--existing-txt", help="Explicit path to the .txt file (overrides auto-detection).")
    p.add_argument("--output", help="Output path (default: '<name> [START BEAT FIXED].txt' next to the input).")
    p.add_argument("--batch", action="store_true", help="Process every immediate subfolder of the input folder.")
    return p


def run(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_path = Path(args.input)

    if args.batch:
        if args.existing_txt or args.output:
            print("--batch can't be combined with --existing-txt/--output.")
            return 1
        results = run_fix_start_note_beat_batch(input_path)
        ok = sum(1 for _, r in results if r.success)
        print(f"\n=== Batch finished: {ok}/{len(results)} succeeded ===")
        return 0 if ok == len(results) else 2

    if args.existing_txt:
        existing_txt_path = Path(args.existing_txt)
    elif input_path.is_file():
        existing_txt_path = input_path
    else:
        existing_txt_path = None  # run_fix_start_note_beat_pipeline auto-detects

    result = run_fix_start_note_beat_pipeline(input_path, existing_txt_path, args.output)
    if not result.success:
        print(f"FAILED: {result.error}")
        return 1
    print(f"Done: {result.output_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run())
