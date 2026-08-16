"""Vocal isolation using Demucs.

Demucs is invoked as a subprocess (its own CLI) rather than imported,
which sidesteps a lot of version/import fragility and matches how most
people already have it installed (`pip install demucs`), and -- as of
the `cancel_requested` param below -- also means a caller with a real
cancellation signal (the GUI's Stop button) can kill it OUTRIGHT the
moment cancellation is requested, not just check for it before/after.
"""

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

# See media_extract.py's own comment on this exact constant -- suppresses
# the console window Windows otherwise pops up for a subprocess when the
# parent has none of its own (the GUI, launched via pythonw.exe).
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
    """Runs Demucs two-stem separation and returns the path to vocals.wav.

    Results are cached under work_dir/separated/<model>/<track>/vocals.wav;
    if that file already exists, separation is skipped.

    `cancel_requested`, if given, is polled every _POLL_INTERVAL_SEC while
    Demucs runs -- the moment it returns True, the Demucs subprocess is
    killed OUTRIGHT (terminate, then kill if it doesn't die promptly) and
    config.PipelineCancelled is raised immediately, without waiting for
    Demucs to finish. None (the CLI's own default) means no cancellation
    is possible here at all, same as before this param existed.
    """
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
        # demucs's own AudioFile (demucs/audio.py) unconditionally shells out
        # to ffmpeg/ffprobe to read ANY input format -- a call site we don't
        # own, same problem class as whisperx.audio.load_audio (see
        # transcription.py's own identical fix and its comment for the full
        # explanation). But demucs runs as its own separate Python process
        # here (see module docstring for why it's a subprocess, not an
        # import), so our process's own subprocess.Popen patch never reaches
        # it -- bootstrap the SAME patch into demucs's own process before
        # demucs itself starts, via -c instead of -m.
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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             creationflags=_NO_WINDOW)
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
