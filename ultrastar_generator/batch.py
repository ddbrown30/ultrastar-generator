"""Multi-song batch processing (feature 10): given a PARENT directory,
runs the normal single-song pipeline on each of its immediate
subdirectories (never the parent itself), mirroring the input structure
into the output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import config
from .main import run_pipeline, PipelineResult


def run_batch(
    parent_dir: Path,
    output_parent_dir: Optional[Path],
    opts: config.PipelineOptions,
    *,
    log: Callable[[str], None] = print,
) -> List[Tuple[str, PipelineResult]]:
    """Runs run_pipeline() once per immediate subdirectory of parent_dir
    (e.g. parent_dir="Songs" containing "Songs/Song A", "Songs/Song B",
    "Songs/Song C" processes each of A/B/C the same way single-song mode
    would process it directly). Output mirrors the input structure 1:1 at
    the top level: output_parent_dir/<song folder name>/ is passed to
    run_pipeline as ITS OWN output_dir (parent-of-final-folder, per
    run_pipeline's own contract), which then creates
    output_parent_dir/<song folder name>/<Artist> - <Title>/ underneath
    it -- or, if output_parent_dir is None, each song falls through to
    run_pipeline's own per-song default (<song folder>/Output/<Artist> -
    <Title>) independently.

    Any exception from a single song -- even one run_pipeline itself
    didn't already catch into a PipelineResult -- is caught HERE, logged,
    and recorded as a failed result. One bad song must never abort the
    rest of the batch; that's the entire point of this function existing
    instead of just shelling a loop around the CLI.
    """
    parent_dir = Path(parent_dir)
    output_parent_dir = Path(output_parent_dir) if output_parent_dir is not None else None

    subdirs = sorted(p for p in parent_dir.iterdir() if p.is_dir())
    results: List[Tuple[str, PipelineResult]] = []

    for i, sub in enumerate(subdirs, 1):
        if opts.cancel_requested is not None and opts.cancel_requested():
            log(f"Cancelled by user before {sub.name} ({i}/{len(subdirs)}).")
            break
        log(f"== Batch {i}/{len(subdirs)}: {sub.name} ==")
        sub_output_dir = (output_parent_dir / sub.name) if output_parent_dir is not None else None
        try:
            result = run_pipeline(sub, sub_output_dir, opts, log=log)
        except Exception as e:
            log(f"  FAILED (unexpected error): {e}")
            result = PipelineResult(success=False, error=str(e))
        if not result.success:
            log(f"  FAILED: {result.error}")
        results.append((sub.name, result))

    n_ok = sum(1 for _, r in results if r.success)
    log(f"\nBatch complete: {n_ok}/{len(results)} song(s) succeeded.")
    for name, result in results:
        status = "OK" if result.success else f"FAILED ({result.error})"
        log(f"  {name}: {status}")

    return results
