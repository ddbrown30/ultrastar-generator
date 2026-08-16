"""Runs specific long, GPU-bound pipeline calls (WhisperX transcription,
pass-1 pitch detection, forced-alignment gap recovery) in a fresh CHILD
PROCESS, so a caller with a real cancel_requested callback (the GUI's
Stop button) can kill the call OUTRIGHT the moment cancellation is
requested, instead of waiting for it to finish -- see
config.PipelineCancelled's own docstring for why in-process cancellation
of a CUDA call can't be instant otherwise. Demucs already runs this way
(see separation.py, which now supports the same instant-kill via a
cancel_requested param of its own); this generalizes the same "external
subprocess, killed on demand" idea to the whisperx/note-detection/
forced-alignment calls that previously ran in-process.

Each call through here pays real subprocess startup + model (re)load
cost -- a fresh interpreter starts from scratch every time, so there's
no cross-call model_cache.py reuse the way an in-process run gets (e.g.
across songs in a --batch run). Callers only route through here when
they actually have a cancel_requested callback (GUI runs); the CLI
keeps calling the underlying functions directly, in-process, since
Ctrl+C already gives it real instant cancellation for free and paying
subprocess overhead there would be pure loss.

Two halves live in this one module: `run_cancellable` (called from the
PARENT process -- main.py/realign.py) spawns the child and manages
polling/cancellation/log-forwarding/debug-log-merging; `_main` (the
child's own entry point, `python -m ultrastar_generator.worker_process
<request.json> <response.json>`) reads the request, dispatches to the
right underlying function via `_DISPATCH`, and writes the result back.
Kept in one file since both halves need to agree on the exact same
`_DISPATCH` func-name/kwargs contract -- splitting them risks the two
sides silently drifting apart.
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

# See media_extract.py's own identical comment -- suppresses the console
# window Windows otherwise pops up for a subprocess when the parent has
# none of its own (the GUI, launched via pythonw.exe).
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_POLL_INTERVAL_SEC = 0.15
_TERMINATE_GRACE_SEC = 5.0


class WorkerError(RuntimeError):
    """The worker subprocess itself failed (crashed, raised, or exited
    non-zero) -- distinct from config.PipelineCancelled, which means it
    was killed on purpose and isn't a real failure."""


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
    """Runs `func_name` (a key of `_DISPATCH`, below) in a fresh child
    process with `kwargs` (must be JSON-serializable), forwarding its
    stdout to `log` line by line and its own DebugLog output (if any)
    into `debug_log`. Polls cancel_requested() every _POLL_INTERVAL_SEC
    and kills the child OUTRIGHT (terminate, then kill if it doesn't die
    promptly) the moment it returns True -- raises
    config.PipelineCancelled immediately, without waiting for the child
    to reach any checkpoint of its own. Raises WorkerError on any other
    (non-cancellation) failure."""
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
            # A real crash before the child's own try/finally could write
            # anything at all (e.g. killed by something other than our own
            # terminate/kill above, or a crash during argv/request-file
            # parsing itself, before _main's own try block starts) -- no
            # structured error to read, only the exit code and whatever
            # was already forwarded to `log` via stdout.
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
        rewindow_long_segments=kwargs.get("rewindow_long_segments", True),
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
    # Must be set before any CUDA context is created (i.e. before torch is
    # imported anywhere, even transitively) -- see main.py's own identical
    # top-of-file guard, which this child process doesn't otherwise inherit
    # (it's a fresh interpreter, not a fork).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
