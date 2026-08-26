"""Vocal isolation using Demucs, invoked as a subprocess (not imported) so it can be killed outright on cancellation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from . import config

# Suppresses console-window flashing when launched from the GUI (pythonw.exe); no-op on non-Windows.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_POLL_INTERVAL_SEC = 0.15
_TERMINATE_GRACE_SEC = 5.0


class SeparationError(RuntimeError):
    pass


def _drain_stdout(stream, chunks: List[str]) -> None:
    for line in iter(stream.readline, ""):
        chunks.append(line)
    stream.close()


def isolate_vocals(
    audio_path: Path,
    work_dir: Path,
    model: str = config.DEFAULT_DEMUCS_MODEL,
    *,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> Path:
    """Runs Demucs two-stem separation and returns the path to vocals.wav. Cached under work_dir/separated/<model>/<track>/vocals.wav. `cancel_requested`, if given, is polled while Demucs runs; if it returns True, the subprocess is killed and config.PipelineCancelled is raised."""
    audio_path = Path(audio_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    track_name = audio_path.stem
    out_dir = work_dir / "separated"
    vocals_path = out_dir / model / track_name / "vocals.wav"

    if vocals_path.exists():
        return vocals_path

    if shutil.which("demucs") is None and shutil.which("python") is None:
        raise SeparationError(
            "Could not find the 'demucs' command. Install it with "
            "`pip install demucs` (requires PyTorch)."
        )

    demucs_args = ["--two-stems", "vocals", "-n", model, "-d", "cuda", "-o", str(out_dir), str(audio_path)]

    if os.name == "nt":
        # Demucs shells out to ffmpeg/ffprobe itself; patch its own process for no-window before it starts.
        _bootstrap = (
            "import sys, subprocess, runpy\n"
            "sys.argv = ['demucs'] + sys.argv[1:]\n"
            "_orig_popen_init = subprocess.Popen.__init__\n"
            "def _no_window_popen_init(self, *a, **kw):\n"
            "    kw['creationflags'] = kw.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW\n"
            "    _orig_popen_init(self, *a, **kw)\n"
            "subprocess.Popen.__init__ = _no_window_popen_init\n"
            "runpy.run_module('demucs', run_name='__main__')\n"
        )
        cmd = [sys.executable, "-c", _bootstrap] + demucs_args
    else:
        cmd = [sys.executable, "-m", "demucs"] + demucs_args

    # Demucs/ffmpeg's own console output isn't guaranteed to be encodable in the system's default
    # codepage (real case: a byte not valid in cp1252 crashed the drain thread below, which then
    # left the pipe undrained -- Demucs blocks once its own stdout buffer fills, hanging the whole
    # run). encoding="utf-8" is deliberately paired with errors="replace", not "strict": a
    # replacement character in this diagnostic-only captured output is harmless, an uncaught
    # decode exception here is not.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    output_chunks: List[str] = []
    reader = threading.Thread(target=_drain_stdout, args=(proc.stdout, output_chunks), daemon=True)
    reader.start()

    cancelled = False
    while True:
        ret = proc.poll()
        if ret is not None:
            break
        if cancel_requested is not None and cancel_requested():
            cancelled = True
            proc.terminate()
            try:
                proc.wait(timeout=_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break
        time.sleep(_POLL_INTERVAL_SEC)
    reader.join(timeout=_TERMINATE_GRACE_SEC)

    if cancelled:
        raise config.PipelineCancelled()

    if proc.returncode != 0:
        raise SeparationError(f"Demucs failed (exit {proc.returncode}).\noutput:\n{''.join(output_chunks)}")

    if not vocals_path.exists():
        raise SeparationError(
            f"Demucs finished but expected output not found at {vocals_path}. "
            f"Check the model name / demucs version."
        )

    return vocals_path
