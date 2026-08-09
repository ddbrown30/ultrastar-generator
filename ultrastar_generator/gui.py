"""Tkinter GUI (feature 8) -- wraps the exact same `run_pipeline`/
`run_batch` functions the CLI uses, so there is only ever one real
pipeline implementation. Runs the pipeline on a background thread (Tk
itself is not thread-safe -- only the main thread may touch widgets) and
captures ALL of its output, including `print()` calls from deep inside
pitch/timing submodules that were deliberately NOT rewired to accept a
`log` callback (see main.py's own docstring for why), via
`contextlib.redirect_stdout` at the call boundary instead.

Launch with `python -m ultrastar_generator.gui`, or via the `run_gui.bat`
launcher at the repo root (uses `pythonw`, so no console window appears).
"""

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
from .file_discovery import AmbiguousInputError, NoAudioSourceFoundError, resolve_artist_title, resolve_primary_source
from .lyrics_lookup import LrcLibCandidate, search_lrclib
from .main import PipelineResult, check_cuda_available, delete_intermediates, run_pipeline

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
        pass  # best-effort -- a failed save just means no persistence this run


class Tooltip:
    """Small hover tooltip -- a borderless Toplevel near the cursor,
    shown on <Enter> and destroyed on <Leave>. Text is fixed at
    construction (this codebase's tooltips are all static, condensed
    versions of the CLI's own --help strings)."""

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
    """Modal candidate picker: an editable Artist/Title search bar at the
    top (pre-filled from the current run's own artist/title when known,
    but freely editable -- lets the user search for anything, not just
    what was auto-detected), LRCLIB results listed on the left, the
    selected one's lyrics preview on the right. Used for BOTH the manual
    pre-run "Search Lyrics..." button and the automatic mid-run ambiguity
    prompt -- the caller decides how/when to show it and what to do with
    `.result` afterward; this class only handles the search/picking UI
    itself. `.result` is the chosen LrcLibCandidate, or None if the user
    cancelled (closed the window or clicked Cancel)."""

    def __init__(self, parent: tk.Misc, *, initial_artist: str = "", initial_title: str = "",
                 initial_candidates: Optional[List[LrcLibCandidate]] = None, title: str = "Select lyrics"):
        super().__init__(parent)
        self.title(title)
        self.iconbitmap('assets/lrcicon.ico')
        self.geometry("780x500")
        self.minsize(500, 340)
        self.transient(parent)
        self.candidates: List[LrcLibCandidate] = []
        self.result: Optional[LrcLibCandidate] = None

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

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        list_frame = ttk.Frame(paned)
        self.listbox = tk.Listbox(list_frame, exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        list_scroll = ttk.Scrollbar(list_frame, command=self.listbox.yview)
        list_scroll.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=list_scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        paned.add(list_frame, weight=1)

        preview_frame = ttk.Frame(paned)
        self.preview = tk.Text(preview_frame, wrap="word", state="disabled")
        preview_scroll = ttk.Scrollbar(preview_frame, command=self.preview.yview)
        self.preview.configure(yscrollcommand=preview_scroll.set)
        self.preview.pack(side="left", fill="both", expand=True)
        preview_scroll.pack(side="right", fill="y")
        paned.add(preview_frame, weight=2)

        # Draggable sash defaults to roughly a 1:2 split (list:preview) --
        # PanedWindow only knows real widget sizes after a layout pass, so
        # this is deferred; the user can freely drag it afterward regardless.
        self.after(10, lambda: paned.sashpos(0, 240))

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.use_button = ttk.Button(button_frame, text="Use Selected", command=self._on_use, state=tk.DISABLED)
        self.use_button.pack(side="right")
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side="right", padx=4)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        if initial_candidates is not None:
            self._set_candidates(initial_candidates)
        elif initial_artist or initial_title:
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
        if not query and not artist and not title:
            messagebox.showerror("Missing search terms",
                                  "Enter a search query, or an artist/title, to search for.", parent=self)
            return
        try:
            results = search_lrclib(q=query) if query else search_lrclib(artist, title)
        except Exception as e:
            messagebox.showerror("Search failed", str(e), parent=self)
            return
        self._set_candidates(results)
        if not results:
            what = repr(query) if query else f"{artist!r} / {title!r}"
            messagebox.showinfo("No results", f"No LRCLIB results found for {what}.", parent=self)

    def _set_candidates(self, candidates: List[LrcLibCandidate]):
        self.candidates = candidates
        self.listbox.delete(0, tk.END)
        for c in candidates:
            self.listbox.insert(tk.END, self._label_for(c))
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        self.preview.configure(state="disabled")
        self.use_button.config(state=tk.DISABLED)
        if candidates:
            self.listbox.selection_set(0)
            self._show_preview(0)
            self.use_button.config(state=tk.NORMAL)

    @staticmethod
    def _label_for(c: LrcLibCandidate) -> str:
        dur = f"{int(c.duration // 60)}:{int(c.duration % 60):02d}" if c.duration else "?:??"
        label = f"[{c.id}] " if c.id is not None else ""
        label += f"{c.track_name} - {c.artist_name}"
        if c.album_name:
            label += f" ({c.album_name})"
        label += f" [{dur}]"
        if c.instrumental:
            label += " [Instrumental]"
        return label

    def _on_select(self, _event=None):
        sel = self.listbox.curselection()
        if sel:
            self._show_preview(sel[0])
            self.use_button.config(state=tk.NORMAL)

    def _show_preview(self, idx: int):
        c = self.candidates[idx]
        text = c.plain_lyrics or "(no lyrics text for this candidate)"
        if c.synced_lyrics:
            text = "[synced lyrics also available for this candidate]\n\n" + text
        self.preview.configure(state="normal")
        self.preview.delete("1.0", tk.END)
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def _on_use(self):
        sel = self.listbox.curselection()
        if sel:
            self.result = self.candidates[sel[0]]
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


class PlaceholderEntry(ttk.Entry):
    """An Entry that shows grey preview text when empty and unfocused.
    The preview is DISPLAY-ONLY -- `is_placeholder` tells callers whether
    the currently-shown text is a real user value or just a preview, so
    opts-building can treat a placeholder-showing field as None and let
    run_pipeline's own real logic (resolve_artist_title, the output-dir
    default) compute the authoritative value once, not twice.

    `get_placeholder` is called lazily, only when a refresh is needed
    (construction, FocusOut, or an explicit external refresh_placeholder()
    call) -- never on every keystroke."""

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
        """Re-shows the (possibly changed) preview if this field is empty
        and not currently being edited. Safe to call any time something
        the preview depends on (input folder, artist, title, ...) changes
        -- a no-op if the user has real text here or is actively typing."""
        if self.focus_get() is self:
            return  # actively being edited -- don't stomp on it
        if self.is_placeholder or not self._var.get().strip():
            self._var.set(self._get_placeholder())
            self.configure(foreground="grey50")
            self.is_placeholder = True

    def effective_value(self) -> Optional[str]:
        """The real user-entered value, or None if only the preview is showing."""
        return None if self.is_placeholder else (self._var.get().strip() or None)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UltraStar Generator")
        self.geometry("820x780")
        self.iconbitmap('assets/guiicon.ico')
        self.minsize(700, 500)

        self._settings = _load_settings()
        self._launch_dir = str(Path.cwd())

        self.mode = tk.StringVar(value="single")  # "single" | "batch" | "youtube"
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.audio_file = tk.StringVar()
        self.youtube_url = tk.StringVar()
        self.youtube_audio_only = tk.BooleanVar(value=True)
        self.artist = tk.StringVar()
        self.title_var = tk.StringVar()

        # Curated main-surface options (see gui.py's own module docstring
        # for why only a subset of the ~30 CLI flags are exposed here).
        self.fetch_lyrics = tk.BooleanVar(value=True)
        self.verify_words = tk.BooleanVar(value=config.ENABLE_WORD_VERIFICATION)
        self.verify_placement = tk.BooleanVar(value=config.ENABLE_PLACEMENT_VERIFICATION)
        self.existing_txt_check = tk.BooleanVar(value=config.ENABLE_EXISTING_TXT_CHECK)
        self.musicxml_force_calibration = tk.BooleanVar(value=config.ENABLE_MUSICXML_FORCE_CALIBRATION)
        self.whisper_model = tk.StringVar(value=config.DEFAULT_WHISPER_MODEL)
        self.pitch_source = tk.StringVar(value=config.DEFAULT_PITCH_SOURCE)

        # Advanced/experimental -- collapsed by default.
        self.lrc_timing_check = tk.BooleanVar(value=config.ENABLE_LRC_TIMING_CHECK)
        self.zone_boundary_snap = tk.BooleanVar(value=config.ENABLE_ZONE_BOUNDARY_SNAP)
        self.no_video_sync = tk.BooleanVar(value=False)
        self.quiet = tk.BooleanVar(value=False)
        self._advanced_visible = False

        self.delete_intermediates = tk.BooleanVar(value=False)
        self._last_output_parent: Optional[Path] = None  # set after a successful run, for "Open Output Folder"

        # Interactive LRCLIB disambiguation. pinned_lyrics is a manual
        # pre-run pick (via the "Search Lyrics..." button) that always
        # wins outright; lyrics_ambiguity_prompt is the checkbox-gated
        # mid-run fallback for when nothing was pre-picked. Both are
        # single-song-mode only -- batch mode always auto-picks silently.
        self.pinned_lyrics: Optional[LrcLibCandidate] = None
        self.lyrics_ambiguity_prompt = tk.BooleanVar(value=False)
        # LRCLIB numeric id override -- lets a user who browsed lrclib.net
        # directly and confirmed a perfect match paste the id back in,
        # bypassing search/scoring entirely for both the MXL+LRC primary
        # path and the standard pipeline's own reference-lyrics fetch.
        self.lrclib_id = tk.StringVar(value="")

        self._running = False
        self._build_widgets()
        self._on_mode_change()

        # Placeholders depend on each other (output folder's preview needs
        # artist/title, which need input_dir) -- wire refresh via trace_add
        # on the SOURCE vars only (input_dir, audio_file, artist, title),
        # never on a var a PlaceholderEntry itself writes to as part of
        # showing its OWN preview, to avoid a self-referential trace loop.
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
        """Real artist/title if the user typed them, else auto-detected
        via the SAME real function run_pipeline itself uses
        (file_discovery.resolve_artist_title -- tries the audio file's
        own name first, falls back to the INPUT FOLDER's own name if
        that doesn't parse, e.g. a ripped/downloaded file keeping a
        generic name like "music.ogg" while its folder is still named
        "<Artist> - <Title>") -- never a separate guess. Returns
        (artist_or_None, title_or_None)."""
        artist = self.artist_entry.effective_value() if hasattr(self, "artist_entry") else None
        title = self.title_entry.effective_value() if hasattr(self, "title_entry") else None
        if artist and title:
            return artist, title
        input_dir = self.input_dir.get().strip()
        if not input_dir or not Path(input_dir).is_dir():
            return artist, title
        try:
            audio_path, _kind = resolve_primary_source(
                Path(input_dir), audio_file_override=(self.audio_file.get().strip() or None))
        except (AmbiguousInputError, NoAudioSourceFoundError, OSError):
            return artist, title
        parsed_artist, parsed_title = resolve_artist_title(audio_path, Path(input_dir))
        return artist or parsed_artist, title or parsed_title

    def _artist_placeholder_text(self) -> str:
        artist, _ = self._resolved_artist_title()
        return artist or "(auto-detected from filename)"

    def _title_placeholder_text(self) -> str:
        _, title = self._resolved_artist_title()
        return title or "(auto-detected from filename)"

    def _output_dir_placeholder_text(self) -> str:
        # The output folder is now the PARENT under which "<Artist> -
        # <Title>" gets created (see run_pipeline's own docstring) --
        # shown as a relative path (always ".\Output\", relative to
        # whatever the input folder ends up being), not an absolute one,
        # since it's the same relationship regardless of where the input
        # folder actually lives.
        return ".\\Output\\"

    def _refresh_artist_title_placeholders(self):
        self.artist_entry.refresh_placeholder()
        self.title_entry.refresh_placeholder()
        self.output_dir_entry.refresh_placeholder()

    # --- widget construction -------------------------------------------------

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        mode_frame = ttk.LabelFrame(self, text="Mode")
        mode_frame.pack(fill="x", **pad)
        for text, value in [("Single song folder", "single"), ("Batch (parent folder)", "batch"),
                             ("YouTube URL", "youtube")]:
            rb = ttk.Radiobutton(mode_frame, text=text, value=value, variable=self.mode,
                                  command=self._on_mode_change)
            rb.pack(side="left", padx=8, pady=4)
        Tooltip(mode_frame, "Single song folder: process one song folder.\n"
                             "Batch: process every immediate subfolder of a parent folder the same way.\n"
                             "YouTube URL: download a video/audio first, then process it like a normal folder.")

        io_frame = ttk.LabelFrame(self, text="Folders")
        io_frame.pack(fill="x", **pad)
        io_frame.columnconfigure(1, weight=1)

        self.input_label = ttk.Label(io_frame, text="Input folder:")
        self.input_label.grid(row=0, column=0, sticky="w", **pad)
        input_entry = ttk.Entry(io_frame, textvariable=self.input_dir)
        input_entry.grid(row=0, column=1, sticky="ew", **pad)
        input_browse = ttk.Button(io_frame, text="Browse...", command=self._browse_input)
        input_browse.grid(row=0, column=2, **pad)
        Tooltip(input_entry, "Folder containing the song's audio (and optionally video/cover/background). "
                              "In batch mode, this is the PARENT folder -- each of its immediate subfolders "
                              "is processed as its own song.")

        ttk.Label(io_frame, text="Output folder:").grid(row=1, column=0, sticky="w", **pad)
        self.output_dir_entry = PlaceholderEntry(io_frame, self.output_dir, self._output_dir_placeholder_text)
        self.output_dir_entry.grid(row=1, column=1, sticky="ew", **pad)
        output_browse = ttk.Button(io_frame, text="Browse...", command=self._browse_output)
        output_browse.grid(row=1, column=2, **pad)
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

        artist_frame = ttk.LabelFrame(self, text="Artist / Title (required for YouTube; overrides filename parsing otherwise)")
        artist_frame.pack(fill="x", **pad)
        ttk.Label(artist_frame, text="Artist:").pack(side="left", padx=8)
        self.artist_entry = PlaceholderEntry(artist_frame, self.artist, self._artist_placeholder_text, width=25)
        self.artist_entry.pack(side="left", padx=4)
        ttk.Label(artist_frame, text="Title:").pack(side="left", padx=8)
        self.title_entry = PlaceholderEntry(artist_frame, self.title_var, self._title_placeholder_text, width=25)
        self.title_entry.pack(side="left", padx=4)
        Tooltip(self.artist_entry, "Overrides the artist parsed from the filename. Required (must be typed, "
                                    "not left as the preview) for YouTube mode.")
        Tooltip(self.title_entry, "Overrides the title parsed from the filename. Required (must be typed, "
                                   "not left as the preview) for YouTube mode.")

        lyrics_frame = ttk.LabelFrame(self, text="Lyrics (single-song mode only)")
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
        self.lyrics_ambiguity_check = ttk.Checkbutton(
            lyrics_frame, text="Ask when lyrics are ambiguous", variable=self.lyrics_ambiguity_prompt)
        self.lyrics_ambiguity_check.pack(side="left", padx=16)
        Tooltip(self.lyrics_ambiguity_check, "Off by default. When on (and nothing was manually searched above), "
                                              "pauses mid-run to let you pick if the automatic LRCLIB search finds "
                                              "more than one real candidate. Single-song mode only -- batch mode "
                                              "always auto-picks silently.")
        self._update_pinned_lyrics_label()

        opts_frame = ttk.LabelFrame(self, text="Options")
        opts_frame.pack(fill="x", **pad)
        c1 = ttk.Checkbutton(opts_frame, text="Fetch reference lyrics", variable=self.fetch_lyrics)
        c1.grid(row=0, column=0, sticky="w", padx=8, pady=2)
        Tooltip(c1, "Look up reference lyrics (LRCLIB, falling back to lyrics.ovh) to correct "
                    "mistranscribed words and force phrase breaks at real line breaks.")
        c2 = ttk.Checkbutton(opts_frame, text="Verify words", variable=self.verify_words)
        c2.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        Tooltip(c2, "Re-transcribe each word in isolation and cross-check against reference lyrics. "
                    "Never changes timing/pitch, only swaps in text when the recheck actively confirms "
                    "a different answer.")
        c3 = ttk.Checkbutton(opts_frame, text="Verify placement", variable=self.verify_placement)
        c3.grid(row=1, column=1, sticky="w", padx=8, pady=2)
        Tooltip(c3, "Expensive expand-search re-transcription pass that can correct a word's timing "
                    "position, not just its text. Off by default.")
        c4 = ttk.Checkbutton(opts_frame, text="Check existing .txt before overwriting", variable=self.existing_txt_check)
        c4.grid(row=2, column=0, sticky="w", padx=8, pady=2)
        Tooltip(c4, "If an existing '<Artist> - <Title>.txt' is found in the input folder, verify its "
                    "pitch/timing first and only regenerate if problems are found.")
        c5 = ttk.Checkbutton(opts_frame, text="MusicXML force calibration", variable=self.musicxml_force_calibration)
        c5.grid(row=2, column=1, sticky="w", padx=8, pady=2)
        Tooltip(c5, "When a MusicXML reference file is found but our own pitch can't confidently "
                    "calibrate against it, use the best available offset anyway rather than skipping.")

        ttk.Label(opts_frame, text="Whisper model:").grid(row=3, column=0, sticky="w", padx=8, pady=2)
        whisper_entry = ttk.Entry(opts_frame, textvariable=self.whisper_model, width=15)
        whisper_entry.grid(row=3, column=0, sticky="e", padx=8)
        Tooltip(whisper_entry, "ASR model size, e.g. small.en (default), medium.en, large-v3. "
                                "Bigger is more accurate but slower.")
        ttk.Label(opts_frame, text="Pitch source:").grid(row=3, column=1, sticky="w", padx=8, pady=2)
        pitch_combo = ttk.Combobox(opts_frame, textvariable=self.pitch_source, values=["rmvpe", "ensemble"],
                                    state="readonly", width=12)
        pitch_combo.grid(row=3, column=1, sticky="e", padx=8)
        Tooltip(pitch_combo, "rmvpe (default): RMVPE's own isolation-mode pitch/voicing, fastest and "
                              "most accurate on average. ensemble: pYIN + CREPE cross-check instead.")

        c6 = ttk.Checkbutton(opts_frame, text="Delete intermediate files after generating",
                              variable=self.delete_intermediates)
        c6.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=2)
        Tooltip(c6, "Deletes the Demucs separation output and any extracted audio/covers under "
                    ".ultrastar_work once generation completes. Debug files, if enabled, are left alone. "
                    "Leave off if you'll re-run this song again soon -- it avoids re-paying separation cost.")

        self.advanced_toggle = ttk.Button(self, text="▶ Advanced (experimental flags)", command=self._toggle_advanced)
        self.advanced_toggle.pack(fill="x", **pad)
        self.advanced_frame = ttk.LabelFrame(self, text="Advanced")
        a1 = ttk.Checkbutton(self.advanced_frame, text="LRC timing check (diagnostic only, EXPERIMENTAL)",
                              variable=self.lrc_timing_check)
        a1.grid(row=0, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a1, "Cross-checks line timing against LRCLIB synced lyrics and FLAGS (never corrects) "
                    "lines that disagree by more than a couple seconds. Diagnostic only.")
        a2 = ttk.Checkbutton(self.advanced_frame, text="Zone-boundary snap (EXPERIMENTAL, not recommended -- see CLAUDE.md)",
                              variable=self.zone_boundary_snap)
        a2.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a2, "Snaps word-zone boundaries to a nearby detected note onset. Real end-to-end testing "
                    "found no net improvement -- kept as an opt-in experiment only.")
        a3 = ttk.Checkbutton(self.advanced_frame, text="Skip VIDEOGAP estimation", variable=self.no_video_sync)
        a3.grid(row=2, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a3, "Skip auto-detecting #VIDEOGAP (the audio/video sync offset) even if a video is present.")
        a4 = ttk.Checkbutton(self.advanced_frame, text="Quiet (suppress verbose per-frame logging)", variable=self.quiet)
        a4.grid(row=3, column=0, sticky="w", padx=8, pady=2)
        Tooltip(a4, "Suppress the verbose [pass1] diagnostic logging (still prints the main pipeline "
                    "stage messages).")

        self.run_frame = ttk.Frame(self)
        self.run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(self.run_frame, text="Run", command=self._on_run)
        self.run_button.pack(side="left")
        self.status_label = ttk.Label(self.run_frame, text="")
        self.status_label.pack(side="left", padx=12)
        open_output_btn = ttk.Button(self.run_frame, text="Open Output Folder", command=self._open_output_folder)
        open_output_btn.pack(side="right", padx=4)
        Tooltip(open_output_btn, "Opens the output folder in File Explorer -- the folder that CONTAINS the "
                                  "per-song output (e.g. .../Output/, not .../Output/<Artist> - <Title>/).")
        delete_now_btn = ttk.Button(self.run_frame, text="Delete Intermediate Files Now",
                                     command=self._delete_intermediates_now)
        delete_now_btn.pack(side="right", padx=4)
        Tooltip(delete_now_btn, "Deletes the Demucs separation output and any extracted audio/covers for "
                                 "the current input folder immediately, without generating anything. Asks "
                                 "for confirmation first.")

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(log_frame, wrap="word", state="normal")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
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
        is_batch = mode == "batch"

        self.input_label.config(text="Parent folder:" if is_batch else "Input folder:")
        if is_youtube:
            self.youtube_label.grid(row=2, column=0, sticky="w", padx=8, pady=4)
            self.youtube_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=4)
            self.youtube_audio_only_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=4)
            self.audio_file_label.grid_remove()
            self.audio_file_entry.grid_remove()
            self.audio_file_browse.grid_remove()
        else:
            self.youtube_label.grid_remove()
            self.youtube_entry.grid_remove()
            self.youtube_audio_only_check.grid_remove()
            if is_batch:
                self.audio_file_label.grid_remove()
                self.audio_file_entry.grid_remove()
                self.audio_file_browse.grid_remove()
            else:
                self.audio_file_label.grid(row=2, column=0, sticky="w", padx=8, pady=4)
                self.audio_file_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=4)
                self.audio_file_browse.grid(row=2, column=2, padx=8, pady=4)

        # Interactive LRCLIB disambiguation is single-song-mode only --
        # batch mode always auto-picks silently, never pausing for input.
        lyrics_controls_state = tk.DISABLED if is_batch else tk.NORMAL
        self.search_lyrics_button.config(state=lyrics_controls_state)
        self.lyrics_ambiguity_check.config(state=lyrics_controls_state)

    # --- LRCLIB lyrics search / disambiguation -----------------------------

    def _on_search_lyrics(self):
        # Pre-fills the dialog's own search fields with the resolved
        # artist/title (real or auto-detected), but the dialog lets the
        # user freely edit and re-search for anything -- these are just a
        # starting point, not a requirement.
        artist, title = self._resolved_artist_title()
        dlg = LrcLibSearchDialog(self, initial_artist=artist or "", initial_title=title or "",
                                  title="Search lyrics")
        self.wait_window(dlg)
        if dlg.result is not None:
            self.pinned_lyrics = dlg.result
            self._update_pinned_lyrics_label()

    def _on_clear_pinned_lyrics(self):
        self.pinned_lyrics = None
        self._update_pinned_lyrics_label()

    def _update_pinned_lyrics_label(self):
        if self.pinned_lyrics is not None:
            self.pinned_lyrics_label.config(
                text=f"Using: {self.pinned_lyrics.track_name} - {self.pinned_lyrics.artist_name}")
            self.clear_pinned_button.pack(side="left", padx=4)
        else:
            self.pinned_lyrics_label.config(text="")
            self.clear_pinned_button.pack_forget()

    def _make_ambiguity_callback(self) -> Callable[[List[LrcLibCandidate]], Optional[LrcLibCandidate]]:
        """Returns a callback for PipelineOptions.lyrics_disambiguation_callback.
        Called from the BACKGROUND pipeline thread (inside fetch_reference_lyrics);
        schedules the real dialog on the MAIN thread via self.after(0, ...) (the
        only safe way to touch Tk widgets from another thread), then blocks the
        background thread on a threading.Event until the dialog closes --
        wait_window() runs its own nested event loop while waiting, so the main
        thread (including this same polling/log-draining) stays fully responsive
        during the pause; only the background pipeline thread is parked.

        Artist/title are captured HERE (before the run starts) to pre-fill the
        dialog's own search fields -- the callback itself only receives
        `candidates`, not the artist/title actually used inside run_pipeline,
        but they should match in the vast majority of cases (same input
        folder/fields)."""
        artist, title = self._resolved_artist_title()

        def callback(candidates: List[LrcLibCandidate]) -> Optional[LrcLibCandidate]:
            result_holder = {}
            done_event = threading.Event()

            def show_dialog():
                dlg = LrcLibSearchDialog(self, initial_artist=artist or "", initial_title=title or "",
                                          initial_candidates=candidates,
                                          title="Multiple lyrics results found -- pick one")
                self.wait_window(dlg)
                result_holder["choice"] = dlg.result
                done_event.set()

            self.after(0, show_dialog)
            done_event.wait()
            return result_holder.get("choice")
        return callback

    def _make_mxl_lrc_fallback_callback(self) -> Callable[[str], bool]:
        """Returns a callback for PipelineOptions.mxl_lrc_fallback_callback.
        Same thread-hop shape as `_make_ambiguity_callback` (the only safe
        pattern for a blocking dialog from the background pipeline thread):
        `self.after(0, ...)` schedules a `messagebox.askyesno` on the main
        thread, the background thread blocks on a `threading.Event` until
        it's answered. Shown by default whenever the MXL+LRC quality gate
        fails or nothing usable was found -- no separate opt-in checkbox,
        per the user's explicit "ask what they want to do" requirement."""
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

    # --- folder/file pickers (remember last-used folder; default to the
    # folder the program was launched from otherwise) ---------------------

    def _last_dir(self, key: str) -> str:
        return self._settings.get(key) or self._launch_dir

    def _remember_dir(self, key: str, d: str):
        self._settings[key] = d
        _save_settings(self._settings)

    def _browse_input(self):
        d = filedialog.askdirectory(title="Select input folder", initialdir=self._last_dir("input_dir"))
        if d:
            # tkinter's askdirectory always returns forward-slash paths on
            # Windows, even though Explorer/native dialogs display and
            # expect backslashes -- if the user copies this value back
            # into a fresh Browse dialog's own address/filename box (a
            # real thing people do), that classic Windows folder-picker
            # rejects a forward-slash path as "not valid" even though
            # it's a perfectly real folder. Normalizing to the native
            # backslash form here (both what's shown AND what's
            # persisted to gui_settings.json) avoids handing the user a
            # value that breaks when pasted back into a picker.
            d = str(Path(d))
            self.input_dir.set(d)
            self._remember_dir("input_dir", d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output folder", initialdir=self._last_dir("output_dir"))
        if d:
            d = str(Path(d))  # see _browse_input's comment on why this normalization matters
            self.output_dir.set(d)
            self._remember_dir("output_dir", d)

    def _browse_audio_file(self):
        initial = self.input_dir.get().strip() or self._last_dir("input_dir")
        exts = config.AUDIO_EXTS + config.VIDEO_EXTS
        filetypes = [("Audio/video files", " ".join(f"*{e}" for e in exts)), ("All files", "*.*")]
        f = filedialog.askopenfilename(title="Select audio/video file", initialdir=initial, filetypes=filetypes)
        if f:
            self.audio_file.set(Path(f).name)

    # --- intermediate-file cleanup / open-output-folder -------------------

    def _delete_intermediates_now(self):
        input_dir = self.input_dir.get().strip()
        if not input_dir:
            messagebox.showerror("Missing folder", "Set the input folder first.")
            return
        work_dir = Path(input_dir) / ".ultrastar_work"
        if not work_dir.is_dir():
            messagebox.showinfo("Nothing to delete", f"No intermediate files found at:\n{work_dir}")
            return
        if not messagebox.askyesno(
                "Delete intermediate files?",
                f"This will permanently delete the Demucs separation output and any extracted "
                f"audio/covers under:\n\n{work_dir}\n\n"
                f"(Debug files, if any, are left alone.) Continue?"):
            return
        delete_intermediates(work_dir)
        messagebox.showinfo("Done", f"Intermediate files deleted from:\n{work_dir}")

    def _open_output_folder(self):
        # Prefer the last real completed run's own path (exact, unambiguous).
        # Otherwise, the output-folder FIELD now already holds the parent
        # directly (see run_pipeline's own contract) -- a real typed value
        # is used as-is; an untouched placeholder (always the relative
        # ".\Output\" preview) is resolved against the current input folder.
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

    def _build_opts(self) -> config.PipelineOptions:
        mode = self.mode.get()
        return config.PipelineOptions(
            artist=self.artist_entry.effective_value(),
            title=self.title_entry.effective_value(),
            audio_file=self.audio_file.get().strip() or None,
            fetch_lyrics=self.fetch_lyrics.get(),
            verify_words=self.verify_words.get(),
            verify_placement=self.verify_placement.get(),
            existing_txt_check=self.existing_txt_check.get(),
            musicxml_force_calibration=self.musicxml_force_calibration.get(),
            whisper_model=self.whisper_model.get().strip() or config.DEFAULT_WHISPER_MODEL,
            pitch_source=self.pitch_source.get(),
            lrc_timing_check=self.lrc_timing_check.get(),
            zone_boundary_snap=self.zone_boundary_snap.get(),
            no_video_sync=self.no_video_sync.get(),
            quiet=self.quiet.get(),
            youtube_url=(self.youtube_url.get().strip() or None) if mode == "youtube" else None,
            youtube_audio_only=self.youtube_audio_only.get(),
            delete_intermediates=self.delete_intermediates.get(),
            batch=(mode == "batch"),
            # Interactive LRCLIB disambiguation -- single-song-mode only
            # (see _on_mode_change, which also disables the controls
            # themselves in batch mode). A manual pre-run pick always wins
            # outright; the ambiguity-prompt callback is only wired up as
            # a fallback when nothing was pre-picked.
            pinned_lyrics=self.pinned_lyrics if mode != "batch" else None,
            lyrics_ambiguity_prompt=self.lyrics_ambiguity_prompt.get() if mode != "batch" else False,
            lyrics_disambiguation_callback=(
                self._make_ambiguity_callback()
                if (mode != "batch" and self.lyrics_ambiguity_prompt.get() and self.pinned_lyrics is None)
                else None
            ),
            # LRCLIB id override -- single-song-mode only, same scoping as
            # pinned_lyrics above (a per-song override doesn't make sense
            # across a batch run of different songs).
            lrclib_id=self._effective_lrclib_id() if mode != "batch" else None,
            # MXL+LRC fallback confirmation -- single-song-mode only, shown
            # by default (no opt-in checkbox) whenever the quality gate
            # fails, per the user's explicit "ask what they want to do".
            # Batch mode always auto-falls-back silently with just the log
            # warning, same convention as the lyrics ambiguity prompt above.
            mxl_lrc_fallback_callback=self._make_mxl_lrc_fallback_callback() if mode != "batch" else None,
        )

    def _effective_lrclib_id(self) -> Optional[int]:
        raw = self.lrclib_id.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _on_run(self):
        if self._running:
            return
        mode = self.mode.get()
        input_dir = self.input_dir.get().strip()
        if not input_dir:
            messagebox.showerror("Missing folder", "An input folder is required.")
            return
        output_dir_value = self.output_dir_entry.effective_value()  # None -> run_pipeline computes its own default
        if mode == "youtube" and not (self.artist_entry.effective_value() and self.title_entry.effective_value()):
            messagebox.showerror("Missing artist/title", "YouTube mode requires both Artist and Title to be typed in "
                                                            "(not left as the grey preview).")
            return
        if mode == "youtube" and not self.youtube_url.get().strip():
            messagebox.showerror("Missing URL", "YouTube mode requires a URL.")
            return

        opts = self._build_opts()
        self.log_text.delete("1.0", tk.END)
        self._set_running(True)

        q: "queue.Queue" = queue.Queue()
        output_dir_path = Path(output_dir_value) if output_dir_value else None
        thread = threading.Thread(
            target=self._run_worker, args=(mode, Path(input_dir), output_dir_path, opts, q), daemon=True)
        thread.start()
        self.after(100, self._drain_queue, q)

    def _run_worker(self, mode: str, input_dir: Path, output_dir: Optional[Path],
                     opts: config.PipelineOptions, q: "queue.Queue"):
        class _QueueWriter:
            def write(self_, s):
                if s.strip():
                    q.put(s)

            def flush(self_):
                pass

        try:
            with contextlib.redirect_stdout(_QueueWriter()):
                if mode == "batch":
                    results = run_batch(input_dir, output_dir, opts, log=q.put)
                    ok = sum(1 for _, r in results if r.success)
                    q.put(f"\n=== Batch finished: {ok}/{len(results)} succeeded ===")
                    # "Open Output Folder" only has one obvious target in batch mode
                    # when the user gave an explicit output_dir -- per-song defaults
                    # scatter each result under its own subfolder.
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
                self.log_text.insert(tk.END, line if line.endswith("\n") else line + "\n")
                if at_bottom:
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        if self._running:
            self.after(100, self._drain_queue, q)

    def _set_running(self, running: bool):
        self._running = running
        self.run_button.config(state=tk.DISABLED if running else tk.NORMAL)
        self.status_label.config(text="Running... (this can take several minutes per song)" if running else "")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
