"""Multi-song batch processing: runs the single-song pipeline on each immediate subdirectory of parent_dir."""

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
    """Runs run_pipeline() once per immediate subdirectory of parent_dir, mirroring output structure 1:1.

    Catches any exception per-song so one bad song doesn't abort the rest of the batch."""
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
