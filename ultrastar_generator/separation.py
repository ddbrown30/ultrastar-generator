"""Vocal isolation using Demucs.

Demucs is invoked as a subprocess (its own CLI) rather than imported,
which sidesteps a lot of version/import fragility and matches how most
people already have it installed (`pip install demucs`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

# See media_extract.py's own comment on this exact constant -- suppresses
# the console window Windows otherwise pops up for a subprocess when the
# parent has none of its own (the GUI, launched via pythonw.exe).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class SeparationError(RuntimeError):
    pass


def isolate_vocals(
    audio_path: Path,
    work_dir: Path,
    model: str = config.DEFAULT_DEMUCS_MODEL,
) -> Path:
    """Runs Demucs two-stem separation and returns the path to vocals.wav.

    Results are cached under work_dir/separated/<model>/<track>/vocals.wav;
    if that file already exists, separation is skipped.
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

    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        raise SeparationError(
            f"Demucs failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    if not vocals_path.exists():
        raise SeparationError(
            f"Demucs finished but expected output not found at {vocals_path}. "
            f"Check the model name / demucs version."
        )

    return vocals_path
