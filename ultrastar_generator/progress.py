"""Lightweight time-based progress reporting for long-running loops.

Reports at most once every `min_interval_sec` (default 5s), not on every
item -- a fast run (small model, few words) stays quiet, but a slow run
(e.g. --whisper-model large-v3 re-transcribing hundreds of words) doesn't
go silent for minutes with no sign anything is happening. Always flushes,
since stdout piped through anything other than a live terminal (a log
file, `| grep`, etc.) defaults to full block buffering in Python and can
otherwise sit invisible until the process exits.
"""

from __future__ import annotations

import time


class ProgressReporter:
    def __init__(self, label: str, total: int, min_interval_sec: float = 5.0, verbose: bool = True):
        self.label = label
        self.total = total
        self.min_interval_sec = min_interval_sec
        self.verbose = verbose
        self.done = 0
        self._last_report = time.monotonic()
        self._start = self._last_report

    def advance(self, n: int = 1, extra: str = "") -> None:
        self.done += n
        if not self.verbose:
            return
        now = time.monotonic()
        finished = self.done >= self.total
        if now - self._last_report >= self.min_interval_sec or finished:
            self._last_report = now
            elapsed = now - self._start
            suffix = f" ({extra})" if extra else ""
            print(f"  {self.label}: {self.done}/{self.total} ({elapsed:.0f}s elapsed){suffix}", flush=True)
