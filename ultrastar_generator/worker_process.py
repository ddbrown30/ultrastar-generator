"""Runs long, GPU-bound pipeline calls (WhisperX transcription, pass-1 pitch detection,
forced-alignment gap recovery) in a fresh child process so a caller with a cancel_requested callback
(the GUI's Stop button) can kill it outright instead of waiting for a checkpoint. Only used by GUI
runs; the CLI calls these functions in-process, since Ctrl+C is already instant.

`run_cancellable` (parent side) spawns the child and manages polling/cancellation/log-forwarding;
`_main` (child entry point) reads the request, dispatches via `_DISPATCH`, writes the result back.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import config

# Suppresses the console window flash under pythonw.exe.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_POLL_INTERVAL_SEC = 0.15
_TERMINATE_GRACE_SEC = 5.0


class WorkerError(RuntimeError):
    """The worker subprocess crashed, raised, or exited non-zero (not a cancellation)."""


def _serialize_words(words) -> list:
    return [dataclasses.asdict(w) for w in words]


def _deserialize_words(data: list):
    from .models import Word
    return [Word(**d) for d in data]


# --- parent-side: spawn + poll + cancel ------------------------------------

def _drain_stdout(stream, log: Callable[[str], None]) -> None:
    for line in iter(stream.readline, ""):
        if line:
            log(line.rstrip("\n"))
    stream.close()


def run_cancellable(func_name: str, kwargs: dict, *, cancel_requested: Optional[Callable[[], bool]],
                     debug_log=None, log: Callable[[str], None] = print) -> Any:
    """Runs `func_name` (a key of `_DISPATCH`) in a fresh child process, forwarding stdout to `log`
    and DebugLog output into `debug_log`. Kills the child outright (terminate, then kill) as soon as
    cancel_requested() returns True, raising config.PipelineCancelled. Raises WorkerError otherwise."""
    with tempfile.TemporaryDirectory(prefix="usg_worker_") as tmp:
        tmp_path = Path(tmp)
        request_path = tmp_path / "request.json"
        response_path = tmp_path / "response.json"
        worker_debug_log_path = (tmp_path / "debug_log.txt") if debug_log is not None else None

        request = {
            "func": func_name,
            "kwargs": kwargs,
            "debug_log_path": str(worker_debug_log_path) if worker_debug_log_path else None,
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")

        cmd = [sys.executable, "-m", "ultrastar_generator.worker_process",
               str(request_path), str(response_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, creationflags=_NO_WINDOW)
        reader = threading.Thread(target=_drain_stdout, args=(proc.stdout, log), daemon=True)
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

        if worker_debug_log_path is not None and worker_debug_log_path.exists():
            debug_log.append_raw(worker_debug_log_path.read_text(encoding="utf-8"))

        if not response_path.exists():
            # Crashed before writing a response (e.g. during argv parsing).
            raise WorkerError(f"'{func_name}' worker process failed (exit {proc.returncode}) -- see log above.")

        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not response.get("ok"):
            raise WorkerError(response.get("error") or f"'{func_name}' worker failed with no error message.")
        return response["result"]


# --- child-side: dispatch table + entry point -------------------------------

def _do_transcribe_words(kwargs: dict, debug_log):
    from .transcription import transcribe_words
    words = transcribe_words(
        Path(kwargs["vocals_path"]), kwargs["model_name"],
        prefer_whisperx=kwargs.get("prefer_whisperx", True),
        debug_log=debug_log,
        vad_filter=kwargs.get("vad_filter", True),
        whisperx_vad_options=kwargs.get("whisperx_vad_options"),
    )
    return _serialize_words(words)


def _do_detect_notes(kwargs: dict, debug_log):
    import librosa
    from .note_detection import detect_notes
    y, sr = librosa.load(kwargs["vocals_path"], sr=None, mono=True)
    notes = detect_notes(
        y, sr,
        bpm=kwargs.get("bpm"),
        pitch_source=kwargs.get("pitch_source"),
        smooth_window_sec=kwargs.get("smooth_window_sec"),
        pitch_jump_semitones=kwargs.get("pitch_jump_semitones"),
        min_note_beats_fraction=kwargs.get("min_note_beats_fraction"),
        silence_threshold_db=kwargs.get("silence_threshold_db"),
        silence_absolute_floor_db=kwargs.get("silence_floor_db"),
        spike_max_duration_sec=kwargs.get("spike_max_duration_sec"),
        spike_min_jump_semitones=kwargs.get("spike_min_jump_semitones"),
        enable_ambiguity_key_tiebreak=kwargs.get("ambiguity_key_tiebreak"),
        ambiguity_margin_threshold=kwargs.get("ambiguity_margin_threshold"),
        verbose=kwargs.get("verbose", True),
        debug_log=debug_log,
    )
    return [dataclasses.asdict(n) for n in notes]


def _do_recover_dropped_reference_words(kwargs: dict, debug_log):
    from .lyrics_lookup import recover_dropped_reference_words
    words = _deserialize_words(kwargs["words"])
    new_words, n_recovered = recover_dropped_reference_words(
        kwargs["ref_lines"], words, Path(kwargs["vocals_path"]), debug_log=debug_log)
    return {"words": _serialize_words(new_words), "n_recovered": n_recovered}


def _do_force_align_reference_lyrics(kwargs: dict, debug_log):
    from .transcription import force_align_reference_lyrics
    words = force_align_reference_lyrics(
        Path(kwargs["vocals_path"]), kwargs["synced_lyrics"], kwargs["audio_duration"],
        debug_log=debug_log)
    return _serialize_words(words)


def _do_force_align_unconfident_runs(kwargs: dict, debug_log):
    from .realign import _force_align_unconfident_runs
    starts = list(kwargs["starts"])
    ends = list(kwargs["ends"])
    confident = list(kwargs["confident"])
    promoted = _force_align_unconfident_runs(
        kwargs["words_text"], starts, ends, confident, Path(kwargs["vocals_path"]), debug_log=debug_log)
    return {"starts": starts, "ends": ends, "confident": confident, "promoted": promoted}


_DISPATCH = {
    "transcribe_words": _do_transcribe_words,
    "detect_notes": _do_detect_notes,
    "recover_dropped_reference_words": _do_recover_dropped_reference_words,
    "force_align_reference_lyrics": _do_force_align_reference_lyrics,
    "force_align_unconfident_runs": _do_force_align_unconfident_runs,
}


def _main() -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before any CUDA/torch import

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    func_name = request["func"]
    kwargs = request["kwargs"]
    debug_log_path = request.get("debug_log_path")

    from .debug_log import DebugLog
    debug_log = DebugLog(Path(debug_log_path)) if debug_log_path else None
    try:
        handler = _DISPATCH.get(func_name)
        if handler is None:
            raise ValueError(f"Unknown worker func {func_name!r}")
        result = handler(kwargs, debug_log)
        response = {"ok": True, "result": result}
    except Exception as e:
        import traceback
        response = {"ok": False, "error": f"{e}\n{traceback.format_exc()}"}
    finally:
        if debug_log is not None:
            debug_log.close()

    response_path.write_text(json.dumps(response), encoding="utf-8")
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    sys.exit(_main())
