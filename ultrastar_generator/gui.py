"""Tkinter GUI -- wraps the same `run_pipeline`/`run_batch` functions the CLI uses.

Runs the pipeline on a background thread and captures all print output via `contextlib.redirect_stdout`.
Launch with `python -m ultrastar_generator.gui`, or via `run_gui.bat` (no console window)."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional

from . import config
from .batch import run_batch
from .file_discovery import (AmbiguousInputError, NoAudioSourceFoundError, headline_case,
                              resolve_artist_title, resolve_primary_source)
from .lyrics_lookup import LrcLibCandidate, search_lrclib, load_lrc_file, effective_lrc_duration
from .main import check_cuda_available, delete_work_files, run_pipeline
from .media_extract import probe_duration_sec
from .realign import RealignPipelineOptions, run_realign_batch, run_realign_pipeline
from .pitch_refresh import (DEFAULT_KEY_NUDGE, PITCH_SOURCES, PitchRefreshOptions,
                             run_pitch_refresh_batch, run_pitch_refresh_pipeline)
from .fix_start_note_beat import run_fix_start_note_beat_pipeline, run_fix_start_note_beat_batch

_DONE = object()  # sentinel, distinct from any real log line
_OUTPUT_PARENT = "__OUTPUT_PARENT__"  # queue-item tag: payload is the Path to open on "Open Output Folder"

_SETTINGS_PATH = Path.home() / ".ultrastar_generator" / "gui_settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(settings: dict) -> None:
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort, no persistence on failure


class Tooltip:
    """Small hover tooltip: borderless Toplevel shown on <Enter>, destroyed on <Leave>."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None):
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self._tip, text=self.text, justify="left", background="#ffffe0",
                          relief="solid", borderwidth=1, wraplength=360, font=("Segoe UI", 8))
        label.pack(ipadx=4, ipady=2)

    def _hide(self, _event=None):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class LrcLibSearchDialog(tk.Toplevel):
    """Modal LRCLIB candidate picker: editable Artist/Title search, results list, lyrics preview.

    `.result` is the chosen LrcLibCandidate, or None if cancelled. `audio_duration`, if known, picks
    the closest-duration winner among candidates with identical synced lyrics."""

    def __init__(self, parent: tk.Misc, *, initial_artist: str = "", initial_title: str = "",
                 title: str = "Select lyrics", audio_duration: Optional[float] = None):
        super().__init__(parent)
        self.title(title)
        self.iconbitmap('assets/lrcicon.ico')
        self.geometry("900x500")
        self.minsize(500, 340)
        self.candidates: List[LrcLibCandidate] = []
        self.result: Optional[LrcLibCandidate] = None
        self.audio_duration = audio_duration

        search_frame = ttk.Frame(self)
        search_frame.pack(fill="x", padx=8, pady=(8, 4))
        row1 = ttk.Frame(search_frame)
        row1.pack(fill="x")
        ttk.Label(row1, text="Artist:").pack(side="left")
        self.search_artist = tk.StringVar(value=initial_artist)
        artist_entry = ttk.Entry(row1, textvariable=self.search_artist, width=22)
        artist_entry.pack(side="left", padx=(4, 12))
        ttk.Label(row1, text="Title:").pack(side="left")
        self.search_title = tk.StringVar(value=initial_title)
        title_entry = ttk.Entry(row1, textvariable=self.search_title, width=22)
        title_entry.pack(side="left", padx=(4, 12))

        row2 = ttk.Frame(search_frame)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="Search (any):").pack(side="left")
        self.search_query = tk.StringVar(value="")
        query_entry = ttk.Entry(row2, textvariable=self.search_query, width=22)
        query_entry.pack(side="left", padx=(4, 12))
        Tooltip(query_entry, "Broader free-text search across track/artist/album together (LRCLIB's "
                              "own general search) -- used INSTEAD of Artist/Title above when filled in.")
        search_button = ttk.Button(row2, text="Search", command=self._do_search)
        search_button.pack(side="left", padx=(0, 4))
        clear_button = ttk.Button(row2, text="Clear", command=self._on_clear_search)
        clear_button.pack(side="left")
        Tooltip(clear_button, "Clears the Artist, Title, and Search (any) fields.")

        for entry in (artist_entry, title_entry, query_entry):
            entry.bind("<Return>", lambda _e: self._do_search())

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        list_frame = ttk.Frame(paned)
        self.listbox = tk.Text(
            list_frame,
            width=1,
            wrap="none",
            cursor="arrow",
            state="disabled",
            highlightthickness=0,
            padx=4,
            pady=4,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        list_scroll = ttk.Scrollbar(list_frame, command=self.listbox.yview)
        list_scroll.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=list_scroll.set)
        self.listbox.bind("<Button-1>", self._on_select)
        self.listbox.bind("<B1-Motion>", lambda _e: "break")
        self.listbox.bind("<ButtonRelease-1>", lambda _e: "break")
        self.listbox.tag_configure("synced", foreground="green")
        self.listbox.tag_configure("even", background="whitesmoke")
        self.listbox.tag_configure("selected", background="lightblue")
        paned.add(list_frame)

        preview_frame = ttk.Frame(paned)
        self.preview = tk.Text(preview_frame, wrap="word", state="disabled")
        preview_scroll = ttk.Scrollbar(preview_frame, command=self.preview.yview)
        self.preview.configure(yscrollcommand=preview_scroll.set)
        self.preview.pack(side="left", fill="both", expand=True)
        preview_scroll.pack(side="right", fill="y")
        paned.add(preview_frame)

        # Deferred: PanedWindow only knows real widget sizes after a layout pass.
        self.after_idle(lambda: paned.sash_place(0, 350, 0))

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.use_button = ttk.Button(button_frame, text="Use Selected", command=self._on_use, state=tk.DISABLED)
        self.use_button.pack(side="right")
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side="right", padx=4)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        if initial_artist or initial_title:
            self._do_search()

        self.grab_set()

    def _on_clear_search(self):
        self.search_artist.set("")
        self.search_title.set("")
        self.search_query.set("")

    def _do_search(self):
        query = self.search_query.get().strip()
        artist = self.search_artist.get().strip()
        title = self.search_title.get().strip()
        if not query and not title:
            messagebox.showerror("Missing search terms",
                                  "Enter a search query, or a title and optional artist, to search for.", parent=self)
            return
        try:
            results = search_lrclib(q=query) if query else search_lrclib(artist, title)
        except Exception as e:
            messagebox.showerror("Search failed", str(e), parent=self)
            return
        # Synced-lyrics only; unsynced entries are useless to this pipeline.
        results = [c for c in results if not c.instrumental and c.synced_lyrics]
        self._set_candidates(results)
        if not results:
            what = repr(query) if query else f"{artist!r} / {title!r}"
            messagebox.showinfo("No results", f"No LRCLIB results with synced lyrics found for {what}.",
                                 parent=self)

    def _dedupe_identical_synced(self, candidates: List[LrcLibCandidate]) -> List[LrcLibCandidate]:
        """Collapses candidates with byte-identical synced lyrics to one, keeping the closest-duration match
        (or the first, if audio duration is unknown). Unsynced candidates are never merged."""
        groups: dict = {}
        for i, c in enumerate(candidates):
            key = (c.synced_lyrics or "").strip()
            groups.setdefault(key if key else ("__no_sync__", i), []).append(i)

        audio_duration = self.audio_duration
        keep_candidates: List[LrcLibCandidate] = []
        for idxs in groups.values():
            if len(idxs) == 1 or audio_duration is None:
                keep_candidate = candidates[idxs[0]]
                keep_candidate.dupe_count = len(idxs) - 1
                keep_candidates.append(keep_candidate)
                continue

            def _distance(i, _target=audio_duration):
                d = effective_lrc_duration(candidates[i])
                return abs(d - _target) if d is not None else float("inf")

            keep_candidate = candidates[min(idxs, key=_distance)]
            keep_candidate.dupe_count = len(idxs) - 1
            keep_candidates.append(keep_candidate)

        return keep_candidates

    def _set_candidates(self, candidates: List[LrcLibCandidate]):
        candidates = self._dedupe_identical_synced(candidates)
        self.candidates = candidates
        self.listbox.configure(state="normal")
        self.listbox.delete("1.0", tk.END)
        even = ""
        for c in candidates:
            self.listbox.insert(tk.END, f"[{c.id}][{int(c.duration // 60)}:{int(c.duration % 60):02d}]", even)
            if c.synced_lyrics:
                self.listbox.insert(tk.END, "[Synced]", ("synced", even))
            if c.dupe_count > 0:
                self.listbox.insert(tk.END, f"[{c.dupe_count} dupes]")
            self.listbox.insert(tk.END,
                f"\n{c.track_name}\n"
                f"{c.artist_name}{f' ({c.album_name})' if c.album_name else ''}\n",
                even
            )
            even = "even" if not even else ""
        self.listbox.configure(state="disabled")
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        self.preview.configure(state="disabled")
        self.use_button.config(state=tk.DISABLED)
        if candidates:
            self.selected_index = 0
            self._show_preview(0)
            self.use_button.config(state=tk.NORMAL)

    def _on_select(self, event):
        line = int(self.listbox.index(f"@{event.x},{event.y}").split(".")[0])
        index = (line - 1) // 3

        if not 0 <= index < len(self.candidates):
            return

        self.selected_index = index

        self.listbox.configure(state="normal")
        self.listbox.tag_remove("selected", "1.0", "end")

        start_line = index * 3 + 1
        end_line = start_line + 3

        self.listbox.tag_add("selected", f"{start_line}.0", f"{end_line}.0")
        self.listbox.configure(state="disabled")

        self._show_preview(index)
        self.use_button.config(state=tk.NORMAL)

    def _show_preview(self, idx: int):
        c = self.candidates[idx]
        text = c.synced_lyrics or c.plain_lyrics or "(no lyrics text for this candidate)"
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def _on_use(self):
        if self.selected_index is not None:
            self.result = self.candidates[self.selected_index]
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


class PlaceholderEntry(ttk.Entry):
    """An Entry that shows grey preview text when empty and unfocused; `is_placeholder` flags display-only text."""

    def __init__(self, master, textvariable: tk.StringVar, get_placeholder: Callable[[], str], **kwargs):
        super().__init__(master, textvariable=textvariable, **kwargs)
        self._var = textvariable
        self._get_placeholder = get_placeholder
        self.is_placeholder = False
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.refresh_placeholder()

    def _on_focus_in(self, _event=None):
        if self.is_placeholder:
            self._var.set("")
            self.configure(foreground="black")
            self.is_placeholder = False

    def _on_focus_out(self, _event=None):
        self.refresh_placeholder()

    def refresh_placeholder(self):
        """Re-shows the (possibly changed) preview if this field is empty and not focused."""
        if self.focus_get() is self:
            return  # actively being edited
        if self.is_placeholder or not self._var.get().strip():
            self._var.set(self._get_placeholder())
            self.configure(foreground="grey50")
            self.is_placeholder = True

    def effective_value(self) -> Optional[str]:
        """The real user-entered value, or None if only the preview is showing."""
        return None if self.is_placeholder else (self._var.get().strip() or None)

    def set_real_value(self, value: str) -> None:
        """Sets a real value programmatically (e.g. from a Browse dialog). Use instead of `.set()` on the var
        directly -- a Browse callback never focuses the entry, so only this clears `is_placeholder`."""
        self.is_placeholder = False
        self.configure(foreground="black")
        self._var.set(value)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UltraStar Generator")
        self.geometry("820x780")
        self.iconbitmap('assets/guiicon.ico')
        self.minsize(700, 500)

        self._settings = _load_settings()
        self._launch_dir = str(Path.cwd())

        self.mode = tk.StringVar(value="generate")  # "generate" | "youtube" | "realign" | "pitch_refresh" | "fix_start_beat"
        # Batch is a modifier orthogonal to mode; disabled (not just ignored) for "youtube".
        self.batch_mode = tk.BooleanVar(value=False)
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.audio_file = tk.StringVar()
        self.youtube_url = tk.StringVar()
        self.youtube_audio_only = tk.BooleanVar(value=True)
        self.artist = tk.StringVar()
        self.title_var = tk.StringVar()

        # Realign (alignment-only) mode -- see realign.py.
        self.existing_txt_path = tk.StringVar()
        self.realign_use_lrc = tk.BooleanVar(value=True)
        self.lrc_mode = tk.StringVar(value="windowed")
        self.realign_strategy = tk.StringVar(value="validate")
        self.realign_delete_work_files = tk.BooleanVar(value=False)

        # Pitch-refresh mode -- see pitch_refresh.py. Re-detects only pitch from existing timing, no lyrics/ASR.
        self.pitch_refresh_source = tk.StringVar(value=config.DEFAULT_PITCH_SOURCE)
        self.pitch_refresh_isolate_vocals = tk.BooleanVar(value=True)
        self.pitch_refresh_key_nudge = tk.BooleanVar(value=DEFAULT_KEY_NUDGE)
        self.pitch_refresh_musicxml = tk.BooleanVar(value=True)
        self.pitch_refresh_delete_work_files = tk.BooleanVar(value=False)

        # Curated main-surface options; diagnostic/expert-only flags stay CLI-only.
        self.fetch_lyrics = tk.BooleanVar(value=True)
        self.fetch_cover = tk.BooleanVar(value=True)
        self.whisper_model = tk.StringVar(value=config.DEFAULT_WHISPER_MODEL)
        self.pitch_source = tk.StringVar(value=config.DEFAULT_PITCH_SOURCE)
        self.demucs_model = tk.StringVar(value=config.DEFAULT_DEMUCS_MODEL)
        self.mxl_lrc_primary = tk.BooleanVar(value=config.ENABLE_MXL_LRC_PRIMARY)

        # Advanced/experimental -- collapsed by default.
        self.lrc_timing_check = tk.BooleanVar(value=config.ENABLE_LRC_TIMING_CHECK)
        self.ambiguity_key_tiebreak = tk.BooleanVar(value=config.ENABLE_AMBIGUITY_KEY_TIEBREAK)
        self.no_video_sync = tk.BooleanVar(value=False)
        self.quiet = tk.BooleanVar(value=False)
        self._advanced_visible = False

        self.delete_work_files = tk.BooleanVar(value=False)
        self._last_output_parent: Optional[Path] = None  # for "Open Output Folder"

        # Manual pre-run pick via "Search Lyrics..."; always wins. Single-song mode only.
        self.pinned_lyrics: Optional[LrcLibCandidate] = None
        # Explicit LRCLIB id override, bypasses search/scoring.
        self.lrclib_id = tk.StringVar(value="")
        # Local .lrc file override, wins over search and --lrclib-id.
        self.lrc_file = tk.StringVar(value="")
        # MusicXML reference override; auto-detected otherwise. musicxml_part is for multi-part/duet arrangements.
        self.musicxml_reference = tk.StringVar(value="")
        self.musicxml_part = tk.StringVar(value="")

        self._running = False
        # Set by _on_stop, polled at stage boundaries -- only takes effect between stages, not instantly.
        self._cancel_event = threading.Event()
        self._build_widgets()
        self._on_mode_change()

        # Wire refresh on the SOURCE vars only, never a var a PlaceholderEntry writes to itself
        # (avoids a self-referential trace loop).
        self.input_dir.trace_add("write", lambda *_: self._refresh_artist_title_placeholders())
        self.audio_file.trace_add("write", lambda *_: self._refresh_artist_title_placeholders())
        self.artist.trace_add("write", lambda *_: self.output_dir_entry.refresh_placeholder())
        self.title_var.trace_add("write", lambda *_: self.output_dir_entry.refresh_placeholder())

        cuda_error = check_cuda_available()
        if cuda_error:
            self.run_button.config(state=tk.DISABLED)
            self.status_label.config(text=cuda_error, foreground="red")
            messagebox.showerror("CUDA not available", cuda_error)

    # --- placeholder preview computation --------------------------------

    def _resolved_artist_title(self) -> tuple:
        """Real artist/title if typed, else auto-detected via the same function run_pipeline uses.
        Returns (artist_or_None, title_or_None), headline-cased."""
        artist = self.artist_entry.effective_value() if hasattr(self, "artist_entry") else None
        title = self.title_entry.effective_value() if hasattr(self, "title_entry") else None
        if not (artist and title):
            input_dir = self.input_dir.get().strip()
            if input_dir and Path(input_dir).is_dir():
                try:
                    audio_path, _kind = resolve_primary_source(
                        Path(input_dir), audio_file_override=(self.audio_file.get().strip() or None))
                except (AmbiguousInputError, NoAudioSourceFoundError, OSError):
                    audio_path = None
                if audio_path is not None:
                    parsed_artist, parsed_title = resolve_artist_title(audio_path, Path(input_dir))
                    artist = artist or parsed_artist
                    title = title or parsed_title
        return (headline_case(artist) if artist else artist,
                headline_case(title) if title else title)

    def _resolved_audio_duration(self) -> Optional[float]:
        """Best-effort audio duration for the selected input folder, used by LrcLibSearchDialog's dedupe.
        Returns None if not resolvable."""
        input_dir = self.input_dir.get().strip()
        if not input_dir or not Path(input_dir).is_dir() or self._is_batch():
            return None
        try:
            audio_path, _kind = resolve_primary_source(
                Path(input_dir), audio_file_override=(self.audio_file.get().strip() or None))
        except (AmbiguousInputError, NoAudioSourceFoundError, OSError):
            return None
        return probe_duration_sec(audio_path)

    def _artist_placeholder_text(self) -> str:
        # YouTube mode requires artist/title explicitly; no auto-detect placeholder.
        if self.mode.get() == "youtube":
            return ""
        artist, _ = self._resolved_artist_title()
        return artist or "(auto-detected from input folder)"

    def _title_placeholder_text(self) -> str:
        if self.mode.get() == "youtube":
            return ""
        _, title = self._resolved_artist_title()
        return title or "(auto-detected from input folder)"

    def _output_dir_placeholder_text(self) -> str:
        # Parent folder under which "<Artist> - <Title>" gets created; shown relative to the input folder.
        return ".\\Output\\"

    def _existing_txt_placeholder_text(self) -> str:
        # Static hint; the real file is only resolved at run time.
        return "(auto-detected from input folder)"

    def _refresh_artist_title_placeholders(self):
        self.artist_entry.refresh_placeholder()
        self.title_entry.refresh_placeholder()
        self.output_dir_entry.refresh_placeholder()

    # --- widget construction -------------------------------------------------

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        mode_frame = ttk.LabelFrame(self, text="Mode")
        mode_frame.pack(fill="x", **pad)
        for text, value in [("Generate song file", "generate"), ("YouTube URL", "youtube"),
                             ("Realign existing file", "realign"), ("Refresh pitch", "pitch_refresh"),
                             ("Fix Start Note Beat", "fix_start_beat")]:
            rb = ttk.Radiobutton(mode_frame, text=text, value=value, variable=self.mode,
                                  command=self._on_mode_change)
            rb.pack(side="left", padx=8, pady=4)
        self.batch_check = ttk.Checkbutton(mode_frame, text="Batch (parent folder)", variable=self.batch_mode,
                                            command=self._on_mode_change)
        self.batch_check.pack(side="left", padx=(24, 8), pady=4)
        Tooltip(mode_frame, "Generate song file: process one song folder from scratch.\n"
                             "YouTube URL: download a video/audio first, then process it like a normal folder.\n"
                             "Realign existing file: alignment-only mode -- re-time an EXISTING .txt's own "
                             "notes against its audio (GAP, note start/length only) without touching pitch "
                             "or the note sequence itself.\n"
                             "Refresh pitch: pitch-only mode -- re-detect an EXISTING .txt's own notes' PITCH "
                             "from its audio, without touching timing, text, or the note sequence itself.\n"
                             "Fix Start Note Beat: rebases an EXISTING .txt's #GAP so its first note lands "
                             "on beat 0 (this project's own hard invariant) -- pure GAP/beat-grid arithmetic, "
                             "no audio needed at all. Every note shifts by the same constant beat count; no "
                             "note's real audio timing, pitch, text, or the note sequence itself changes.\n"
                             "Batch: process every immediate subfolder of the input folder the same way, "
                             "instead of the input folder itself. Usable with Generate, Realign, Refresh "
                             "pitch, or Fix Start Note Beat -- not with YouTube (a single URL can't populate "
                             "multiple subfolders).")

        io_frame = ttk.LabelFrame(self, text="Folders")
        io_frame.pack(fill="x", **pad)
        io_frame.columnconfigure(1, weight=1)
        self.io_frame = io_frame

        self.input_label = ttk.Label(io_frame, text="Input folder:")
        self.input_label.grid(row=0, column=0, sticky="w", **pad)
        input_entry = ttk.Entry(io_frame, textvariable=self.input_dir)
        input_entry.grid(row=0, column=1, sticky="ew", **pad)
        input_browse = ttk.Button(io_frame, text="Browse...", command=self._browse_input)
        input_browse.grid(row=0, column=2, **pad)
        Tooltip(input_entry, "Folder containing the song's audio (and optionally video/cover/background). "
                              "In batch mode, this is the PARENT folder -- each of its immediate subfolders "
                              "is processed as its own song.")

        self.output_dir_label = ttk.Label(io_frame, text="Output folder:")
        self.output_dir_label.grid(row=1, column=0, sticky="w", **pad)
        self.output_dir_entry = PlaceholderEntry(io_frame, self.output_dir, self._output_dir_placeholder_text)
        self.output_dir_entry.grid(row=1, column=1, sticky="ew", **pad)
        self.output_dir_browse = ttk.Button(io_frame, text="Browse...", command=self._browse_output)
        self.output_dir_browse.grid(row=1, column=2, **pad)
        Tooltip(self.output_dir_entry, "Where the .txt and copied companion files are written. Must differ "
                                        "from the input folder. Optional -- leave blank to use the default "
                                        "shown in grey.")

        self.audio_file_label = ttk.Label(io_frame, text="Audio file (if ambiguous):")
        self.audio_file_label.grid(row=2, column=0, sticky="w", **pad)
        self.audio_file_entry = ttk.Entry(io_frame, textvariable=self.audio_file)
        self.audio_file_entry.grid(row=2, column=1, sticky="ew", **pad)
        self.audio_file_browse = ttk.Button(io_frame, text="Browse...", command=self._browse_audio_file)
        self.audio_file_browse.grid(row=2, column=2, **pad)
        Tooltip(self.audio_file_entry, "Only needed if the input folder has more than one audio/video file "
                                        "and the pipeline can't tell which one is the song -- pick it here.")

        self.youtube_label = ttk.Label(io_frame, text="YouTube URL:")
        self.youtube_entry = ttk.Entry(io_frame, textvariable=self.youtube_url)
        Tooltip(self.youtube_entry, "The video will be downloaded into the input folder, then processed "
                                     "like a normal song folder.")
        self.youtube_audio_only_check = ttk.Checkbutton(
            io_frame, text="Audio only (uncheck to also download video)", variable=self.youtube_audio_only)
        Tooltip(self.youtube_audio_only_check, "On: download only audio (mp3) -- the common case when you "
                                                "don't need the music video itself. Off: download the full "
                                                "video (mp4) too, used as #VIDEO.")

        self.existing_txt_label = ttk.Label(io_frame, text="Existing .txt file:")
        self.existing_txt_entry = PlaceholderEntry(io_frame, self.existing_txt_path,
                                                     self._existing_txt_placeholder_text)
        self.existing_txt_browse = ttk.Button(io_frame, text="Browse...", command=self._browse_existing_txt)
        Tooltip(self.existing_txt_entry, "The UltraStar .txt file to use as the existing base for this mode. "
                                          "Realign: its notes/pitches/note sequence are kept exactly as-is; "
                                          "only GAP and note start/length are adjusted to better match the "
                                          "audio. Refresh pitch: its timing/text/note sequence are kept "
                                          "exactly as-is; only each note's PITCH is re-detected from the "
                                          "audio. Fix Start Note Beat: its notes/pitches/text are kept exactly "
                                          "as-is; only GAP and every note's BEAT NUMBER shift by the same "
                                          "constant so the first note lands on beat 0 -- no audio needed. "
                                          "Optional -- leave blank to auto-detect the single .txt file "
                                          "in the input folder (if more than one exists, tries '<folder "
                                          "name>.txt' before giving up).")

        self.artist_frame = ttk.LabelFrame(self, text="Artist / Title")
        artist_frame = self.artist_frame
        artist_frame.pack(fill="x", **pad)
        ttk.Label(artist_frame, text="Artist:").pack(side="left", padx=8)
        self.artist_entry = PlaceholderEntry(artist_frame, self.artist, self._artist_placeholder_text, width=25)
        self.artist_entry.pack(side="left", padx=4)
        ttk.Label(artist_frame, text="Title:").pack(side="left", padx=8)
        self.title_entry = PlaceholderEntry(artist_frame, self.title_var, self._title_placeholder_text, width=25)
        self.title_entry.pack(side="left", padx=4)
        Tooltip(self.artist_entry, "Overrides the artist parsed from the folder. Required for YouTube mode.")
        Tooltip(self.title_entry, "Overrides the title parsed from the folder. Required for YouTube mode.")

        self.lyrics_frame = ttk.LabelFrame(self, text="Lyrics (single-song mode only)")
        lyrics_frame = self.lyrics_frame
        lyrics_frame.pack(fill="x", **pad)
        self.search_lyrics_button = ttk.Button(lyrics_frame, text="Search Lyrics...", command=self._on_search_lyrics)
        self.search_lyrics_button.pack(side="left", padx=8, pady=4)
        Tooltip(self.search_lyrics_button, "Search LRCLIB directly and pick which result to use, BEFORE running "
                                            "-- overrides the automatic lookup/pick for this run.")
        self.pinned_lyrics_label = ttk.Label(lyrics_frame, text="")
        self.pinned_lyrics_label.pack(side="left", padx=8)
        self.clear_pinned_button = ttk.Button(lyrics_frame, text="Clear", command=self._on_clear_pinned_lyrics)
        Tooltip(self.clear_pinned_button, "Revert to automatic lyrics lookup for this run.")
        ttk.Label(lyrics_frame, text="LRCLIB ID:").pack(side="left", padx=(16, 2))
        self.lrclib_id_entry = ttk.Entry(lyrics_frame, textvariable=self.lrclib_id, width=10)
        self.lrclib_id_entry.pack(side="left")
        Tooltip(self.lrclib_id_entry, "A specific LRCLIB entry id (browse lrclib.net yourself, e.g. by checking "
                                       "a linked video, and paste the id here) -- always wins over search and "
                                       "over the pick above, no ambiguity. Used for both MXL+LRC primary "
                                       "generation and the standard pipeline's own lyrics fetch.")
        ttk.Label(lyrics_frame, text="LRC file:").pack(side="left", padx=(16, 2))
        self.lrc_file_entry = ttk.Entry(lyrics_frame, textvariable=self.lrc_file, width=18)
        self.lrc_file_entry.pack(side="left")
        Tooltip(self.lrc_file_entry, "Path to a LOCAL .lrc synced-lyrics file to use directly, bypassing LRCLIB "
                                      "entirely -- wins over both search and LRCLIB ID above if given.")
        self.lrc_file_browse_button = ttk.Button(lyrics_frame, text="Browse...", command=self._browse_lrc_file)
        self.lrc_file_browse_button.pack(side="left", padx=(2, 0))
        self._update_pinned_lyrics_label()

        # MusicXML reference override, separate from Lyrics -- same Entry+Browse shape.
        self.musicxml_frame = ttk.LabelFrame(self, text="MusicXML (single-song mode only)")
        musicxml_frame = self.musicxml_frame
        ttk.Label(musicxml_frame, text="Reference file:").pack(side="left", padx=(8, 2), pady=4)
        self.musicxml_reference_entry = ttk.Entry(musicxml_frame, textvariable=self.musicxml_reference, width=24)
        self.musicxml_reference_entry.pack(side="left", pady=4)
        Tooltip(self.musicxml_reference_entry, "Path to a MusicXML/.mxl file for this song (e.g. hand-downloaded "
                                                 "sheet music) -- normally auto-detected from companion files in "
                                                 "the input folder; only needed if that picks the wrong file or "
                                                 "finds none. Feeds both pass 4's pitch-class correction and the "
                                                 "MXL+LRC primary path.")
        self.musicxml_reference_browse_button = ttk.Button(
            musicxml_frame, text="Browse...", command=self._browse_musicxml_reference)
        self.musicxml_reference_browse_button.pack(side="left", padx=(2, 0), pady=4)
        ttk.Label(musicxml_frame, text="Part name:").pack(side="left", padx=(16, 2), pady=4)
        self.musicxml_part_entry = ttk.Entry(musicxml_frame, textvariable=self.musicxml_part, width=14)
        self.musicxml_part_entry.pack(side="left", pady=4)
        Tooltip(self.musicxml_part_entry, "Which part name in the MusicXML file carries the lead vocal line, "
                                            "for duet/ensemble arrangements where multiple parts have lyrics -- "
                                            "rarely needed. Falls back to the lyric-bearing part with the most "
                                            "notes if left blank.")

        # Mirrors the normal pipeline's Lyrics/Options split.
        self.realign_lyrics_frame = ttk.LabelFrame(self, text="Lyrics (single-song mode only)")
        realign_lyrics_frame = self.realign_lyrics_frame
        self.realign_search_lyrics_button = ttk.Button(realign_lyrics_frame, text="Search Lyrics...", command=self._on_search_lyrics)
        self.realign_search_lyrics_button.pack(side="left", padx=8, pady=4)
        Tooltip(self.realign_search_lyrics_button, "Search LRCLIB directly and pick which result to use, BEFORE running "
                                            "-- overrides the automatic lookup/pick for this run.")
        ttk.Label(realign_lyrics_frame, text="LRCLIB ID:").pack(side="left", padx=8, pady=4)
        self.realign_lrclib_id_entry = ttk.Entry(realign_lyrics_frame, textvariable=self.lrclib_id, width=10)
        self.realign_lrclib_id_entry.pack(side="left", padx=4, pady=4)
        Tooltip(self.realign_lrclib_id_entry, "A specific LRCLIB entry id (browse lrclib.net yourself and "
                                               "paste the id here) -- always wins over automatic search.")
        ttk.Label(realign_lyrics_frame, text="LRC file:").pack(side="left", padx=(16, 2), pady=4)
        self.realign_lrc_file_entry = ttk.Entry(realign_lyrics_frame, textvariable=self.lrc_file, width=18)
        self.realign_lrc_file_entry.pack(side="left", padx=4, pady=4)
        Tooltip(self.realign_lrc_file_entry, "Path to a LOCAL .lrc synced-lyrics file to use directly, bypassing "
                                              "LRCLIB entirely -- wins over both search and LRCLIB ID above.")
        self.realign_lrc_file_browse_button = ttk.Button(
            realign_lyrics_frame, text="Browse...", command=self._browse_lrc_file)
        self.realign_lrc_file_browse_button.pack(side="left", padx=(2, 8), pady=4)

        self.realign_options_frame = ttk.LabelFrame(self, text="Options")
        realign_options_frame = self.realign_options_frame
        ttk.Label(realign_options_frame, text="Whisper model:").grid(row=0, column=0, sticky="w", padx=8, pady=2)
        realign_whisper_entry = ttk.Entry(realign_options_frame, textvariable=self.whisper_model, width=15)
        realign_whisper_entry.grid(row=0, column=1, sticky="w", padx=8, pady=2)
        Tooltip(realign_whisper_entry, "ASR model size, e.g. small.en, medium.en (default), large-v3. "
                                        "Bigger is more accurate but slower.")
        realign_use_lrc_check = ttk.Checkbutton(realign_options_frame, text="Use LRC synced lyrics",
                                                 variable=self.realign_use_lrc)
        realign_use_lrc_check.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        Tooltip(realign_use_lrc_check, "When available, LRCLIB synced-lyrics line timestamps help place "
                                        "words the audio transcription alone can't confidently reach. Off: "
                                        "audio transcription only.")
        ttk.Label(realign_options_frame, text="LRC mode:").grid(row=1, column=1, sticky="e", padx=(8, 2), pady=2)
        lrc_mode_combo = ttk.Combobox(realign_options_frame, textvariable=self.lrc_mode,
                                       values=["windowed", "seed"], state="readonly", width=10)
        lrc_mode_combo.grid(row=1, column=2, sticky="w", padx=(0, 8), pady=2)
        Tooltip(lrc_mode_combo, "windowed (default): LRC line starts window the audio search, but ONLY when "
                                 "LRCLIB's timing is confidently calibrated against this audio -- otherwise "
                                 "transparently falls back to whole-song matching, same as 'seed'. seed: "
                                 "always whole-song-transcription-primary, LRC only fills residual gaps. "
                                 "Real comparison found 'windowed' never worse and sometimes much better "
                                 "-- see CLAUDE.md.")
        ttk.Label(realign_options_frame, text="Strategy:").grid(row=2, column=0, sticky="w", padx=8, pady=2)
        realign_strategy_combo = ttk.Combobox(realign_options_frame, textvariable=self.realign_strategy,
                                               values=["replace", "validate"], state="readonly", width=10)
        realign_strategy_combo.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=2)
        Tooltip(realign_strategy_combo, "validate (default): a word whose existing position roughly agrees "
                                          "with the audio (after one global GAP correction) is left completely "
                                          "untouched instead of being overwritten -- best when the file is "
                                          "already mostly accurate. replace: a word confidently matched to the "
                                          "audio has its timing REPLACED with the transcription's own value "
                                          "instead -- better when the file's own timing can't be trusted at "
                                          "all. See CLAUDE.md.")
        realign_delete_work_files_check = ttk.Checkbutton(
            realign_options_frame, text="Delete work files after realigning",
            variable=self.realign_delete_work_files)
        realign_delete_work_files_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        Tooltip(realign_delete_work_files_check, "Deletes the entire .ultrastar_work directory (cached "
                                                   "Demucs separation, debug files) once realigning completes. "
                                                   "Leave off if you'll re-run this song again soon -- it "
                                                   "avoids re-paying separation cost.")

        # Pitch-refresh mode has no lyrics functionality, so no matching "Lyrics" frame.
        self.pitch_refresh_options_frame = ttk.LabelFrame(self, text="Options")
        pr_options_frame = self.pitch_refresh_options_frame
        ttk.Label(pr_options_frame, text="Pitch source:").grid(row=0, column=0, sticky="w", padx=8, pady=2)
        pr_source_combo = ttk.Combobox(pr_options_frame, textvariable=self.pitch_refresh_source,
                                        values=sorted(PITCH_SOURCES.keys()), state="readonly", width=12)
        pr_source_combo.grid(row=0, column=1, sticky="w", padx=8, pady=2)
        Tooltip(pr_source_combo, "Which pitch detector to run, given the existing file's own note timing. "
                                  "rmvpe (default) is this project's own shipped default elsewhere too.")
        pr_isolate_check = ttk.Checkbutton(pr_options_frame, text="Isolate vocals with Demucs first",
                                            variable=self.pitch_refresh_isolate_vocals)
        pr_isolate_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        Tooltip(pr_isolate_check, "On by default -- a 9-song regression test found the original mixed-audio "
                                   "default didn't generalize (coin flip for rmvpe, a clear regression for "
                                   "swiftf0). Requires CUDA. Uncheck to detect pitch from the original mixed "
                                   "audio instead (no CUDA needed) -- see CLAUDE.md / project memory.")
        pr_key_nudge_check = ttk.Checkbutton(pr_options_frame,
                                              text="Key nudge (conservative +-1-semitone correction)",
                                              variable=self.pitch_refresh_key_nudge)
        pr_key_nudge_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        Tooltip(pr_key_nudge_check, "On by default -- nudges an out-of-key note by up to 1 semitone toward the "
                                     "detected song key. Real multi-song regression testing found this "
                                     "generalizes cleanly (no confirmed regression) and is a real improvement "
                                     "on several songs -- see CLAUDE.md / project memory.")
        pr_musicxml_check = ttk.Checkbutton(pr_options_frame, text="Use MusicXML when available",
                                             variable=self.pitch_refresh_musicxml)
        pr_musicxml_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        Tooltip(pr_musicxml_check, "On by default. When a MusicXML file is found in the input folder AND a "
                                     "per-song pitch-class calibration can be established, refresh pitch "
                                     "from it directly instead of audio-based detection -- never touches "
                                     "timing/text/note count, except splitting an existing note the "
                                     "MusicXML reveals to be a multi-note melisma (only the first of the "
                                     "new notes keeps the original text). Falls back to audio-based "
                                     "detection when no usable MusicXML is found.")
        pr_delete_work_files_check = ttk.Checkbutton(
            pr_options_frame, text="Delete work files after refreshing",
            variable=self.pitch_refresh_delete_work_files)
        pr_delete_work_files_check.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        Tooltip(pr_delete_work_files_check, "Deletes the entire .ultrastar_work directory (cached Demucs "
                                             "separation, if vocal isolation was used) once pitch refreshing "
                                             "completes. Leave off if you'll re-run this song again soon.")
        ttk.Label(pr_options_frame, text="Demucs model:").grid(row=5, column=0, sticky="w", padx=8, pady=2)
        pr_demucs_entry = ttk.Entry(pr_options_frame, textvariable=self.demucs_model, width=15)
        pr_demucs_entry.grid(row=5, column=1, sticky="w", padx=8, pady=2)
        Tooltip(pr_demucs_entry, f"Vocal-separation model name (default: {config.DEFAULT_DEMUCS_MODEL}), used "
                                   "only when \"Isolate vocals with Demucs first\" is checked above. Same "
                                   "field/value as the main pipeline's own Demucs model setting.")

        self.opts_frame = ttk.LabelFrame(self, text="Options")
        opts_frame = self.opts_frame
        opts_frame.pack(fill="x", **pad)
        c1 = ttk.Checkbutton(opts_frame, text="Fetch reference lyrics", variable=self.fetch_lyrics)
        c1.grid(row=0, column=0, sticky="w", padx=8, pady=2)
        Tooltip(c1, "Look up reference lyrics (LRCLIB, synced lyrics only) to correct "
                    "mistranscribed words and force phrase breaks at real line breaks. If no valid "
                    "synced candidate is found, you'll be asked whether to continue with pure "
                    "transcription instead.")
        c2 = ttk.Checkbutton(opts_frame, text="Fetch cover art", variable=self.fetch_cover)
        c2.grid(row=0, column=1, sticky="w", padx=8, pady=2)
        Tooltip(c2, "Download a cover image online (MusicBrainz/Cover Art Archive, then iTunes, then "
                    "Deezer) when the input folder has no [CO]-tagged or embedded cover art of its own.")
        c3 = ttk.Checkbutton(opts_frame, text="MXL+LRC primary generation", variable=self.mxl_lrc_primary)
        c3.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        Tooltip(c3, "Default ON. When a MusicXML file (Reference file above, or auto-detected) AND matching "
                    "synced lyrics are both available, generate from those directly (MusicXML for pitch, "
                    "LRCLIB line starts as real-time anchors, real transcription to place words within a "
                    "line) instead of the standard audio-only pass 1-4 pipeline. Quality-gated -- falls back "
                    "to the standard pipeline automatically if the match doesn't check out.")
        demucs_frame = ttk.Frame(opts_frame)
        demucs_frame.grid(row=1, column=1, sticky="w", padx=8, pady=2)
        ttk.Label(demucs_frame, text="Demucs model:").pack(side="left")
        demucs_entry = ttk.Entry(demucs_frame, textvariable=self.demucs_model, width=15)
        demucs_entry.pack(side="left", padx=(4, 0))
        Tooltip(demucs_entry, f"Vocal-separation model name (default: {config.DEFAULT_DEMUCS_MODEL}), e.g. "
                               "htdemucs_ft (higher quality, slower). Passed straight through to Demucs.")
        # Sub-frame per label+input so the wider checkbox rows don't push them apart.
        whisper_frame = ttk.Frame(opts_frame)
        whisper_frame.grid(row=2, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(whisper_frame, text="Whisper model:").pack(side="left")
        whisper_entry = ttk.Entry(whisper_frame, textvariable=self.whisper_model, width=15)
        whisper_entry.pack(side="left", padx=(4, 0))
        Tooltip(whisper_entry, "ASR model size, e.g. small.en (default), medium.en, large-v3. "
                                "Bigger is more accurate but slower.")
        pitch_frame = ttk.Frame(opts_frame)
        pitch_frame.grid(row=2, column=1, sticky="w", padx=8, pady=2)
        ttk.Label(pitch_frame, text="Pitch source:").pack(side="left")
        pitch_combo = ttk.Combobox(pitch_frame, textvariable=self.pitch_source, values=sorted(PITCH_SOURCES.keys()),
                                    state="readonly", width=12)
        pitch_combo.pack(side="left", padx=(4, 0))
        Tooltip(pitch_combo, "rmvpe (default): RMVPE's own pitch/voicing decision, fastest and most "
                              "accurate on average. swiftf0: lightweight CNN pitch detector with a "
                              "real native voicing decision of its own. Whichever is chosen supplies "
                              "both pitch and voicing exclusively -- no cross-check with any other source.")

        c6 = ttk.Checkbutton(opts_frame, text="Delete work files after generating",
                              variable=self.delete_work_files)
        c6.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        Tooltip(c6, "Deletes the entire .ultrastar_work directory (Demucs separation output, extracted "
                    "audio/covers, and debug files) once generation completes. Leave off if you'll re-run "
                    "this song again soon -- it avoids re-paying separation cost.")

        self.advanced_toggle = ttk.Button(self, text="▶ Advanced (experimental flags)", command=self._toggle_advanced)
        self.advanced_toggle.pack(fill="x", **pad)
        self.advanced_frame = ttk.LabelFrame(self, text="Advanced")
        a1 = ttk.Checkbutton(self.advanced_frame, text="LRC timing check (diagnostic only, EXPERIMENTAL)",
                              variable=self.lrc_timing_check)
        a1.grid(row=0, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a1, "Cross-checks line timing against LRCLIB synced lyrics and FLAGS (never corrects) "
                    "lines that disagree by more than a couple seconds. Diagnostic only.")
        a3 = ttk.Checkbutton(self.advanced_frame, text="Skip VIDEOGAP estimation", variable=self.no_video_sync)
        a3.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a3, "Skip auto-detecting #VIDEOGAP (the audio/video sync offset) even if a video is present.")
        a4 = ttk.Checkbutton(self.advanced_frame, text="Quiet (suppress verbose per-frame logging)", variable=self.quiet)
        a4.grid(row=2, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a4, "Suppress the verbose [pass1] diagnostic logging (still prints the main pipeline "
                    "stage messages).")
        a6 = ttk.Checkbutton(self.advanced_frame, text="Ambiguity key tie-break (pass 1)",
                              variable=self.ambiguity_key_tiebreak)
        a6.grid(row=4, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a6, "RMVPE-only. Recomputes each note's pitch CLASS by summing RMVPE's own raw pitch "
                    "salience across the note's own span; when the top-2 candidates are genuinely close, "
                    "breaks the tie using the song's own detected key (published Krumhansl-Kessler "
                    "profiles). A confident, unambiguous note is never touched. Real-audio validated: "
                    "+2.4pp average pitch-class accuracy across a 12-song test, 9 songs improved, 2 "
                    "modest regressions (not universal).")

        self.run_frame = ttk.Frame(self)
        self.run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(self.run_frame, text="Run", command=self._on_run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(self.run_frame, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=4)
        Tooltip(self.stop_button, "Cancels the current run. Takes effect at the next stage boundary (e.g. "
                    "after the current vocal-separation/transcription/pass-1 step finishes), not instantly "
                    "-- there's no safe way to interrupt GPU inference mid-call. In batch mode, the current "
                    "song still finishes; no further songs are started.")
        self.status_label = ttk.Label(self.run_frame, text="")
        self.status_label.pack(side="left", padx=12)
        open_output_btn = ttk.Button(self.run_frame, text="Open Output Folder", command=self._open_output_folder)
        open_output_btn.pack(side="right", padx=4)
        Tooltip(open_output_btn, "Opens the output folder in File Explorer -- the folder that CONTAINS the "
                                  "per-song output (e.g. .../Output/, not .../Output/<Artist> - <Title>/).")
        delete_now_btn = ttk.Button(self.run_frame, text="Delete Work Files Now",
                                     command=self._delete_work_files_now)
        delete_now_btn.pack(side="right", padx=4)
        Tooltip(delete_now_btn, "Deletes the work folder for "
                                 "the current input folder immediately, without generating anything. Asks "
                                 "for confirmation first.")

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(log_frame, wrap="word", state="normal")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="darkorange")
        scroll.pack(side="right", fill="y")

    def _toggle_advanced(self):
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self.advanced_frame.pack(fill="x", padx=8, pady=4, before=self.run_frame)
            self.advanced_toggle.config(text="▼ Advanced (experimental flags)")
        else:
            self.advanced_frame.pack_forget()
            self.advanced_toggle.config(text="▶ Advanced (experimental flags)")

    def _on_mode_change(self):
        mode = self.mode.get()
        is_youtube = mode == "youtube"
        is_realign = mode == "realign"
        is_pitch_refresh = mode == "pitch_refresh"
        is_fix_start_beat = mode == "fix_start_beat"
        # Realign, pitch_refresh, and fix_start_beat all work from an existing .txt and write next to it.
        uses_existing_txt = is_realign or is_pitch_refresh or is_fix_start_beat
        # Batch is meaningless for YouTube; disabled and ignored there.
        self.batch_check.config(state=tk.DISABLED if is_youtube else tk.NORMAL)
        is_batch = self.batch_mode.get() and not is_youtube

        self.input_label.config(text="Parent folder:" if is_batch else "Input folder:")
        if is_youtube:
            self.youtube_label.grid(row=2, column=0, sticky="w", padx=8, pady=4)
            self.youtube_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=4)
            self.youtube_audio_only_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=4)
            self.audio_file_label.grid_remove()
            self.audio_file_entry.grid_remove()
            self.audio_file_browse.grid_remove()
            self.existing_txt_label.grid_remove()
            self.existing_txt_entry.grid_remove()
            self.existing_txt_browse.grid_remove()
        else:
            self.youtube_label.grid_remove()
            self.youtube_entry.grid_remove()
            self.youtube_audio_only_check.grid_remove()
            # Audio file: disabled (not hidden) in batch mode; hidden entirely for fix_start_beat (needs no audio).
            if is_fix_start_beat:
                self.audio_file_label.grid_remove()
                self.audio_file_entry.grid_remove()
                self.audio_file_browse.grid_remove()
            else:
                self.audio_file_label.grid(row=2, column=0, sticky="w", padx=8, pady=4)
                self.audio_file_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=4)
                self.audio_file_browse.grid(row=2, column=2, padx=8, pady=4)
                audio_file_state = tk.DISABLED if is_batch else tk.NORMAL
                self.audio_file_entry.config(state=audio_file_state)
                self.audio_file_browse.config(state=audio_file_state)
            if uses_existing_txt:
                self.existing_txt_label.grid(row=3, column=0, sticky="w", padx=8, pady=4)
                self.existing_txt_entry.grid(row=3, column=1, sticky="ew", padx=8, pady=4)
                self.existing_txt_browse.grid(row=3, column=2, padx=8, pady=4)
                # Same reasoning as audio file: disabled, not hidden, in batch mode.
                existing_txt_state = tk.DISABLED if is_batch else tk.NORMAL
                self.existing_txt_entry.config(state=existing_txt_state)
                self.existing_txt_browse.config(state=existing_txt_state)
            else:
                self.existing_txt_label.grid_remove()
                self.existing_txt_entry.grid_remove()
                self.existing_txt_browse.grid_remove()

        # Realign and pitch_refresh always write next to the existing file; Output folder doesn't apply.
        if uses_existing_txt:
            self.output_dir_label.grid_remove()
            self.output_dir_entry.grid_remove()
            self.output_dir_browse.grid_remove()
        else:
            self.output_dir_label.grid(row=1, column=0, sticky="w", padx=8, pady=4)
            self.output_dir_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
            self.output_dir_browse.grid(row=1, column=2, padx=8, pady=4)

        if is_realign:
            self.artist_frame.config(text="Artist / Title (overrides the existing file's own tags for LRCLIB lookup)")
        elif is_youtube:
            self.artist_frame.config(text="Artist / Title (required for YouTube)")
        else:
            self.artist_frame.config(text="Artist / Title")
        # Artist/title: disabled in batch mode, same as above; always disabled for pitch_refresh/fix_start_beat
        # (no artist/title concept there).
        artist_title_state = tk.DISABLED if (is_batch or is_pitch_refresh or is_fix_start_beat) else tk.NORMAL
        self.artist_entry.config(state=artist_title_state)
        self.title_entry.config(state=artist_title_state)
        # Refresh now rather than waiting for FocusOut, so switching to/from YouTube updates ghost text immediately.
        self.artist_entry.refresh_placeholder()
        self.title_entry.refresh_placeholder()

        # Each mode toggles a whole options frame. `io_frame` is always packed, so it's a safe `after=`
        # anchor; `artist_frame` isn't (hidden entirely in pitch_refresh mode).
        if is_realign:
            self.artist_frame.pack(fill="x", padx=8, pady=4, after=self.io_frame)
            self.lyrics_frame.pack_forget()
            self.musicxml_frame.pack_forget()
            self.opts_frame.pack_forget()
            self.advanced_toggle.pack_forget()
            self.advanced_frame.pack_forget()
            self.pitch_refresh_options_frame.pack_forget()
            self.realign_lyrics_frame.pack(fill="x", padx=8, pady=4, after=self.artist_frame)
            self.realign_options_frame.pack(fill="x", padx=8, pady=4, after=self.realign_lyrics_frame)
        elif is_pitch_refresh:
            self.artist_frame.pack_forget()
            self.lyrics_frame.pack_forget()
            self.musicxml_frame.pack_forget()
            self.opts_frame.pack_forget()
            self.advanced_toggle.pack_forget()
            self.advanced_frame.pack_forget()
            self.realign_lyrics_frame.pack_forget()
            self.realign_options_frame.pack_forget()
            self.pitch_refresh_options_frame.pack(fill="x", padx=8, pady=4, after=self.io_frame)
        elif is_fix_start_beat:
            # No option surface -- pure GAP/beat-grid arithmetic, nothing to configure.
            self.artist_frame.pack_forget()
            self.lyrics_frame.pack_forget()
            self.musicxml_frame.pack_forget()
            self.opts_frame.pack_forget()
            self.advanced_toggle.pack_forget()
            self.advanced_frame.pack_forget()
            self.realign_lyrics_frame.pack_forget()
            self.realign_options_frame.pack_forget()
            self.pitch_refresh_options_frame.pack_forget()
        else:
            self.artist_frame.pack(fill="x", padx=8, pady=4, after=self.io_frame)
            self.realign_lyrics_frame.pack_forget()
            self.realign_options_frame.pack_forget()
            self.pitch_refresh_options_frame.pack_forget()
            self.lyrics_frame.pack(fill="x", padx=8, pady=4, after=self.artist_frame)
            self.musicxml_frame.pack(fill="x", padx=8, pady=4, after=self.lyrics_frame)
            self.opts_frame.pack(fill="x", padx=8, pady=4, after=self.musicxml_frame)
            self.advanced_toggle.pack(fill="x", padx=8, pady=4, after=self.opts_frame)
            if self._advanced_visible:
                self.advanced_frame.pack(fill="x", padx=8, pady=4, after=self.advanced_toggle)

        # Interactive lyrics disambiguation is single-song-mode only.
        lyrics_controls_state = tk.DISABLED if is_batch else tk.NORMAL
        self.search_lyrics_button.config(state=lyrics_controls_state)
        self.lrclib_id_entry.config(state=lyrics_controls_state)
        self.lrc_file_entry.config(state=lyrics_controls_state)
        self.lrc_file_browse_button.config(state=lyrics_controls_state)
        # Same reasoning: MusicXML file/part override is single-song-mode only.
        self.musicxml_reference_entry.config(state=lyrics_controls_state)
        self.musicxml_reference_browse_button.config(state=lyrics_controls_state)
        self.musicxml_part_entry.config(state=lyrics_controls_state)
        self.realign_lrclib_id_entry.config(state=artist_title_state)
        self.realign_lrc_file_entry.config(state=artist_title_state)
        self.realign_lrc_file_browse_button.config(state=artist_title_state)

    # --- LRCLIB lyrics search / disambiguation -----------------------------

    def _on_search_lyrics(self):
        # Pre-fills the search fields with resolved artist/title; freely editable.
        artist, title = self._resolved_artist_title()
        dlg = LrcLibSearchDialog(self, initial_artist=artist or "", initial_title=title or "",
                                  title="Search lyrics", audio_duration=self._resolved_audio_duration())
        self.wait_window(dlg)
        if dlg.result is not None:
            self.pinned_lyrics = dlg.result
            self._update_pinned_lyrics_label()

    def _on_clear_pinned_lyrics(self):
        self.pinned_lyrics = None
        self._update_pinned_lyrics_label()

    def _update_pinned_lyrics_label(self):
        if self.pinned_lyrics is not None:
            self.lrclib_id.set(self.pinned_lyrics.id)
            self.pinned_lyrics_label.config(
                text=f"Using: {self.pinned_lyrics.track_name} - {self.pinned_lyrics.artist_name}")
            self.clear_pinned_button.pack(side="left", padx=4)
        else:
            self.pinned_lyrics_label.config(text="")
            self.clear_pinned_button.pack_forget()

    def _make_mxl_lrc_fallback_callback(self) -> Callable[[str], bool]:
        """Callback for PipelineOptions.mxl_lrc_fallback_callback: schedules a blocking dialog on the main
        thread via `self.after(0, ...)` + `threading.Event`, shown whenever the MXL+LRC quality gate fails."""
        def callback(reason: str) -> bool:
            result_holder = {}
            done_event = threading.Event()

            def show_dialog():
                result_holder["continue"] = messagebox.askyesno(
                    "MXL+LRC generation unavailable",
                    f"{reason}\n\nContinue with standard audio-based generation?",
                    parent=self,
                )
                done_event.set()

            self.after(0, show_dialog)
            done_event.wait()
            return result_holder.get("continue", False)
        return callback

    def _make_no_lrc_fallback_callback(self) -> Callable[[str], bool]:
        """Callback for PipelineOptions.no_lrc_fallback_callback, shown whenever no valid LRCLIB candidate
        was found. Same thread-hop shape as `_make_mxl_lrc_fallback_callback`."""
        def callback(reason: str) -> bool:
            result_holder = {}
            done_event = threading.Event()

            def show_dialog():
                result_holder["continue"] = messagebox.askyesno(
                    "No valid lyrics found",
                    f"{reason}\n\nContinue with pure transcription (no reference-lyrics correction "
                    f"or forced line breaks)?",
                    parent=self,
                )
                done_event.set()

            self.after(0, show_dialog)
            done_event.wait()
            return result_holder.get("continue", False)
        return callback

    # --- folder/file pickers (remember last-used folder; default to the
    # folder the program was launched from otherwise) ---------------------

    def _last_dir(self, key: str) -> str:
        return self._settings.get(key) or self._launch_dir

    def _remember_dir(self, key: str, d: str):
        self._settings[key] = d
        _save_settings(self._settings)

    def _browse_input(self):
        d = filedialog.askdirectory(title="Select input folder", initialdir=self.input_dir.get().strip() or self._last_dir("input_dir"))
        if d:
            # askdirectory returns forward-slash paths; normalize to native backslash form.
            d = str(Path(d))
            self.input_dir.set(d)
            self._remember_dir("input_dir", d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output folder", initialdir=self.input_dir.get().strip() or self._last_dir("output_dir"))
        if d:
            d = str(Path(d))  # see _browse_input's comment
            self.output_dir_entry.set_real_value(d)
            self._remember_dir("output_dir", d)

    def _browse_audio_file(self):
        initial = self.input_dir.get().strip() or self._last_dir("input_dir")
        exts = config.AUDIO_EXTS + config.VIDEO_EXTS
        filetypes = [("Audio/video files", " ".join(f"*{e}" for e in exts)), ("All files", "*.*")]
        f = filedialog.askopenfilename(title="Select audio/video file", initialdir=initial, filetypes=filetypes)
        if f:
            self.audio_file.set(Path(f).name)

    def _browse_existing_txt(self):
        initial = self.input_dir.get().strip() or self._last_dir("input_dir")
        filetypes = [("UltraStar txt files", "*.txt"), ("All files", "*.*")]
        f = filedialog.askopenfilename(title="Select existing .txt file", initialdir=initial, filetypes=filetypes)
        if f:
            self.existing_txt_entry.set_real_value(str(Path(f)))

    def _browse_lrc_file(self):
        initial = self.input_dir.get().strip() or self._last_dir("input_dir")
        filetypes = [("LRC synced lyrics files", "*.lrc"), ("All files", "*.*")]
        f = filedialog.askopenfilename(title="Select .lrc file", initialdir=initial, filetypes=filetypes)
        if f:
            self.lrc_file.set(str(Path(f)))

    def _browse_musicxml_reference(self):
        initial = self.input_dir.get().strip() or self._last_dir("input_dir")
        filetypes = [("MusicXML files", "*.mxl *.musicxml *.xml"), ("All files", "*.*")]
        f = filedialog.askopenfilename(title="Select MusicXML file", initialdir=initial, filetypes=filetypes)
        if f:
            self.musicxml_reference.set(str(Path(f)))

    # --- work-file cleanup / open-output-folder -------------------

    def _delete_work_files_now(self):
        input_dir = self.input_dir.get().strip()
        if not input_dir:
            messagebox.showerror("Missing folder", "Set the input folder first.")
            return
        work_dir = Path(input_dir) / ".ultrastar_work"
        if not work_dir.is_dir():
            messagebox.showinfo("Nothing to delete", f"No work folder found at:\n{work_dir}")
            return
        if not messagebox.askyesno(
                "Delete work files?",
                f"This will permanently delete the work directory:\n\n{work_dir}\n\n"
                f"Continue?"):
            return
        delete_work_files(work_dir)
        messagebox.showinfo("Done", f"{work_dir} deleted")

    def _open_output_folder(self):
        # Prefer the last completed run's own path; else resolve the output-folder field against the input folder.
        target = self._last_output_parent
        if target is None:
            effective = self.output_dir_entry.effective_value()
            if effective:
                target = Path(effective)
            else:
                input_dir = self.input_dir.get().strip()
                if input_dir:
                    target = Path(input_dir) / "Output"
        if target is None or not target.is_dir():
            messagebox.showinfo("Output folder", "No output folder to open yet -- run the pipeline first, "
                                                    "or set the input folder so a default can be computed.")
            return
        os.startfile(str(target))

    # --- actions ---------------------------------------------------------

    def _is_batch(self) -> bool:
        # Batch is meaningless for YouTube; ignored there.
        return self.batch_mode.get() and self.mode.get() != "youtube"

    def _build_opts(self) -> config.PipelineOptions:
        mode = self.mode.get()
        is_batch = self._is_batch()
        return config.PipelineOptions(
            # Per-song overrides forced None in batch mode, matching the disabled fields in _on_mode_change.
            artist=None if is_batch else self.artist_entry.effective_value(),
            title=None if is_batch else self.title_entry.effective_value(),
            audio_file=None if is_batch else (self.audio_file.get().strip() or None),
            fetch_lyrics=self.fetch_lyrics.get(),
            fetch_cover=self.fetch_cover.get(),
            whisper_model=self.whisper_model.get().strip() or config.DEFAULT_WHISPER_MODEL,
            pitch_source=self.pitch_source.get(),
            demucs_model=self.demucs_model.get().strip() or config.DEFAULT_DEMUCS_MODEL,
            mxl_lrc_primary=self.mxl_lrc_primary.get(),
            # Single-song-mode only.
            musicxml_reference=None if is_batch else (self.musicxml_reference.get().strip() or None),
            musicxml_part=None if is_batch else (self.musicxml_part.get().strip() or None),
            lrc_timing_check=self.lrc_timing_check.get(),
            ambiguity_key_tiebreak=self.ambiguity_key_tiebreak.get(),
            no_video_sync=self.no_video_sync.get(),
            quiet=self.quiet.get(),
            youtube_url=(self.youtube_url.get().strip() or None) if mode == "youtube" else None,
            youtube_audio_only=self.youtube_audio_only.get(),
            delete_work_files=self.delete_work_files.get(),
            batch=is_batch,
            # Interactive LRCLIB disambiguation -- single-song-mode only; a manual pre-run pick always wins.
            pinned_lyrics=self._effective_pinned_lyrics() if not is_batch else None,
            # LRCLIB id override -- single-song-mode only.
            lrclib_id=self._effective_lrclib_id() if not is_batch else None,
            # MXL+LRC fallback confirmation -- single-song-mode only; batch always auto-falls-back silently.
            mxl_lrc_fallback_callback=self._make_mxl_lrc_fallback_callback() if not is_batch else None,
            # No-valid-LRC fallback confirmation -- same convention as above.
            no_lrc_fallback_callback=self._make_no_lrc_fallback_callback() if not is_batch else None,
            cancel_requested=self._cancel_event.is_set,
        )

    def _effective_lrclib_id(self) -> Optional[int]:
        raw = self.lrclib_id.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _effective_pinned_lyrics(self) -> Optional[LrcLibCandidate]:
        """A typed LRC file path wins over a manual Search Lyrics... pick."""
        raw = self.lrc_file.get().strip()
        if raw:
            candidate = load_lrc_file(raw, artist=self.artist_entry.effective_value() or "",
                                       title=self.title_entry.effective_value() or "")
            if candidate is not None:
                return candidate
        return self.pinned_lyrics

    def _build_realign_opts(self) -> RealignPipelineOptions:
        is_batch = self._is_batch()
        return RealignPipelineOptions(
            # Same batch-mode reasoning as _build_opts above.
            audio_file=None if is_batch else (self.audio_file.get().strip() or None),
            whisper_model=self.whisper_model.get().strip() or config.DEFAULT_WHISPER_MODEL,
            artist=None if is_batch else self.artist_entry.effective_value(),
            title=None if is_batch else self.title_entry.effective_value(),
            lrclib_id=None if is_batch else self._effective_lrclib_id(),
            lrc_file=None if is_batch else (self.lrc_file.get().strip() or None),
            use_lrc=self.realign_use_lrc.get(),
            lrc_mode=self.lrc_mode.get(),
            strategy=self.realign_strategy.get(),
            delete_work_files=self.realign_delete_work_files.get(),
            batch=is_batch,
            cancel_requested=self._cancel_event.is_set,
        )

    def _build_pitch_refresh_opts(self) -> PitchRefreshOptions:
        is_batch = self._is_batch()
        return PitchRefreshOptions(
            # Same batch-mode reasoning as _build_opts/_build_realign_opts. No artist/title/lyrics fields here.
            audio_file=None if is_batch else (self.audio_file.get().strip() or None),
            isolate_vocals=self.pitch_refresh_isolate_vocals.get(),
            demucs_model=self.demucs_model.get().strip() or config.DEFAULT_DEMUCS_MODEL,
            pitch_source=self.pitch_refresh_source.get(),
            key_nudge=self.pitch_refresh_key_nudge.get(),
            musicxml_pitch=self.pitch_refresh_musicxml.get(),
            delete_work_files=self.pitch_refresh_delete_work_files.get(),
            batch=is_batch,
            cancel_requested=self._cancel_event.is_set,
        )

    def _on_run(self):
        if self._running:
            return
        mode = self.mode.get()
        is_batch = self._is_batch()
        input_dir = self.input_dir.get().strip()
        if not input_dir:
            messagebox.showerror("Missing folder", "An input folder is required.")
            return
        # None -> run_pipeline's own default; unused by realign/pitch_refresh/fix_start_beat.
        output_dir_value = (self.output_dir_entry.effective_value()
                             if mode not in ("realign", "pitch_refresh", "fix_start_beat") else None)
        if mode == "youtube" and not (self.artist_entry.effective_value() and self.title_entry.effective_value()):
            messagebox.showerror("Missing artist/title", "YouTube mode requires both Artist and Title to be typed in "
                                                            "(not left as the grey preview).")
            return
        if mode == "youtube" and not self.youtube_url.get().strip():
            messagebox.showerror("Missing URL", "YouTube mode requires a URL.")
            return
        # In batch mode each subfolder auto-detects its own existing .txt; a real typed value is used as-is,
        # otherwise it's auto-detected too (find_existing_txt_in_folder).
        existing_txt_path = None
        if mode in ("realign", "pitch_refresh", "fix_start_beat") and not is_batch:
            existing_txt_value = self.existing_txt_entry.effective_value()
            if existing_txt_value:
                existing_txt_path = Path(existing_txt_value)
                if not existing_txt_path.is_file():
                    messagebox.showerror("File not found", f"Existing .txt file not found:\n{existing_txt_path}")
                    return

        if mode == "realign":
            opts = self._build_realign_opts()
        elif mode == "pitch_refresh":
            opts = self._build_pitch_refresh_opts()
        elif mode == "fix_start_beat":
            opts = None  # no option surface at all -- pure GAP/beat-grid arithmetic
        else:
            opts = self._build_opts()
        self.log_text.delete("1.0", tk.END)
        self._cancel_event.clear()
        self._set_running(True)

        q: "queue.Queue" = queue.Queue()
        output_dir_path = Path(output_dir_value) if output_dir_value else None
        thread = threading.Thread(
            target=self._run_worker,
            args=(mode, is_batch, Path(input_dir), output_dir_path, opts, q, existing_txt_path),
            daemon=True)
        thread.start()
        self.after(100, self._drain_queue, q)

    def _run_worker(self, mode: str, is_batch: bool, input_dir: Path, output_dir: Optional[Path],
                     opts, q: "queue.Queue", existing_txt_path: Optional[Path] = None):
        class _QueueWriter:
            def write(self, s):
                if s.strip():
                    q.put(s)

            def flush(self):
                pass

        try:
            with contextlib.redirect_stdout(_QueueWriter()):
                if mode == "realign":
                    if is_batch:
                        results = run_realign_batch(input_dir, opts, log=q.put)
                        ok = sum(1 for _, r in results if r.success)
                        q.put(f"\n=== Batch finished: {ok}/{len(results)} succeeded ===")
                        # Each result is written next to its own subfolder's file; no output-folder mirroring.
                        q.put((_OUTPUT_PARENT, input_dir))
                    else:
                        result = run_realign_pipeline(input_dir, existing_txt_path, opts, log=q.put)
                        if result.success:
                            q.put(f"\n=== Done: {result.output_path} ===")
                            if result.output_path is not None:
                                q.put((_OUTPUT_PARENT, result.output_path.parent))
                        else:
                            q.put(f"\n=== FAILED: {result.error} ===")
                elif mode == "pitch_refresh":
                    if is_batch:
                        results = run_pitch_refresh_batch(input_dir, opts, log=q.put)
                        ok = sum(1 for _, r in results if r.success)
                        q.put(f"\n=== Batch finished: {ok}/{len(results)} succeeded ===")
                        # Same reasoning as realign's batch branch above.
                        q.put((_OUTPUT_PARENT, input_dir))
                    else:
                        result = run_pitch_refresh_pipeline(input_dir, existing_txt_path, opts, log=q.put)
                        if result.success:
                            q.put(f"\n=== Done: {result.output_path} ===")
                            if result.output_path is not None:
                                q.put((_OUTPUT_PARENT, result.output_path.parent))
                        else:
                            q.put(f"\n=== FAILED: {result.error} ===")
                elif mode == "fix_start_beat":
                    if is_batch:
                        results = run_fix_start_note_beat_batch(input_dir, log=q.put)
                        ok = sum(1 for _, r in results if r.success)
                        q.put(f"\n=== Batch finished: {ok}/{len(results)} succeeded ===")
                        # Same reasoning as realign/pitch_refresh's batch branches above.
                        q.put((_OUTPUT_PARENT, input_dir))
                    else:
                        result = run_fix_start_note_beat_pipeline(input_dir, existing_txt_path, log=q.put)
                        if result.success:
                            q.put(f"\n=== Done: {result.output_path} ===")
                            if result.output_path is not None:
                                q.put((_OUTPUT_PARENT, result.output_path.parent))
                        else:
                            q.put(f"\n=== FAILED: {result.error} ===")
                elif is_batch:
                    results = run_batch(input_dir, output_dir, opts, log=q.put)
                    ok = sum(1 for _, r in results if r.success)
                    q.put(f"\n=== Batch finished: {ok}/{len(results)} succeeded ===")
                    # Only a meaningful target if the user gave an explicit output_dir.
                    if output_dir is not None:
                        q.put((_OUTPUT_PARENT, output_dir))
                else:
                    result = run_pipeline(input_dir, output_dir, opts, log=q.put)
                    if result.success:
                        q.put(f"\n=== Done: {result.output_txt_path} ===")
                        if result.output_txt_path is not None:
                            q.put((_OUTPUT_PARENT, result.output_txt_path.parent.parent))
                    else:
                        q.put(f"\n=== FAILED: {result.error} ===")
        except Exception as e:
            q.put(f"\n=== Unexpected error: {e} ===")
        q.put(_DONE)

    def _drain_queue(self, q: "queue.Queue"):
        try:
            while True:
                item = q.get_nowait()
                if item is _DONE:
                    self._set_running(False)
                    return
                if isinstance(item, tuple) and len(item) == 2 and item[0] == _OUTPUT_PARENT:
                    self._last_output_parent = item[1]
                    continue
                line = str(item)
                at_bottom = self.log_text.yview()[1] >= 0.99
                text_tag = ""
                if "warning" in line.lower():
                    text_tag = "warning"
                if "error" in line.lower() or "failed" in line.lower():
                    text_tag = "error"
                self.log_text.insert(tk.END, line if line.endswith("\n") else line + "\n", text_tag)
                if at_bottom:
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        if self._running:
            self.after(100, self._drain_queue, q)

    def _on_stop(self):
        if not self._running or self._cancel_event.is_set():
            return
        if not messagebox.askyesno(
                "Stop run?",
                "Stop the current run?\n\nThis takes effect at the next stage boundary (e.g. after the "
                "current vocal-separation/transcription/pass-1 step finishes), not instantly -- there's no "
                "safe way to interrupt GPU inference mid-call. In batch mode, the song currently in "
                "progress still finishes; no further songs are started."):
            return
        self._cancel_event.set()
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Cancelling... (finishing the current stage)")

    def _set_running(self, running: bool):
        self._running = running
        self.run_button.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)
        self.status_label.config(text="Running... (this can take several minutes per song)" if running else "")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
