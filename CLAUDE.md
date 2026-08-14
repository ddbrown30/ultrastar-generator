# ultrastar_generator

Generates UltraStar Deluxe `.txt` karaoke files from raw audio: isolates
vocals, detects notes, transcribes lyrics, fits lyrics to notes, writes
a spec-compliant `.txt`. Run `python test_dry_run.py` before/after any
change (synthetic regression suite, no audio/models needed, ~114+ checks,
must stay green).

**Project conventions**: new features default ON unless there's a
specific risk (state it). Every new/changed feature needs a **real-audio
validation run**, not just `test_dry_run.py`. Don't commit to git unless
asked. When a "clean-looking" validation result seems too good, print/
inspect individual values before trusting the aggregate stat — this
project has been burned twice by text-matching silently pairing the
wrong repeated instance.

Any time a new option/argument is added/removed, it should be added/removed
to the gui at the same time.

## Architecture: pitch/timing-first, lyrics-second

1. **Pass 1** (`note_detection.py`): detects notes from vocal audio
   alone, no lyrics knowledge. `_ensure_nonoverlapping` is a hard
   guarantee — if it ever fires, that's a real bug, not routine.
2. **Transcription** (`transcription.py`): WhisperX (forced alignment)
   or faster-whisper for lyric text + rough word timing.
3. **Lyrics correction + phrasing** (`lyrics_lookup.py`/`phrasing.py`):
   fetches reference lyrics (LRCLIB first, lyrics.ovh fallback), aligns
   to ASR via difflib, tags words with reference line ids — each
   reference line break forces a `-` break in output, overriding
   gap-based heuristics.
4. **Pass 2 fitting** (`lyric_alignment.py`): fits words onto pass-1's
   notes; never changes timing/pitch for words with real notes; fallback
   words borrow nearest pass-1 note's pitch rather than re-analyzing
   audio.
5. Matched reference lines distribute notes proportionally by syllable
   count (interior ASR word timestamps are unreliable enough to swallow
   a passage into one word's melisma).
6. Non-overlap enforced twice more: `postprocess.enforce_monotonic`
   (seconds-level, preserves given order, never sorts by timestamp) and
   `usdx_writer.py` (integer-beat level).

## Diagnostics (check before speculating about a bug)

- `<Artist> - <Title> [PASS1 DEBUG].txt` (default on, `--no-pass1-debug`
  to skip): pass-1 notes with note-name text instead of lyrics — diff
  against real output to isolate pass-1 vs. lyric-fitting bugs.
- Console logs: pass-1 frame/voicing/merge stats; pass-2 match-vs-
  fallback counts (long fallback list ⇒ pass 1 under-detecting); lyrics
  corrections and reference-line counts (zero ⇒ lookup failed, fell back
  to gap-based phrasing). `--quiet` suppresses.
- `<Artist> - <Title> [DEBUG LOG].txt` (default on, `--no-debug-log` to
  skip, used by both `main.py` and `realign.py`): has a PER-WORD TRACE
  (each word + which mechanism placed it), a RAW ASR TRANSCRIPT (what
  WhisperX actually produced), and RAW WHISPER DECODER SEGMENTS
  (pre-forced-alignment). **Don't conflate the per-word trace with raw
  ASR data** — they answer different questions and diagnosing from the
  wrong one produces confidently-wrong conclusions (real incident: a
  user-pasted per-word trace was mistaken for raw ASR, leading nowhere
  until the raw-ASR section was actually read).

## Lessons learned (do not reintroduce)

- Never run pYIN on a tiny isolated clip — needs context. Fixed by
  analyzing the whole track once; pass-2 fallback borrows pitch instead
  of re-running pYIN on ~0.1s clips. Same failure class shows up
  elsewhere too: forced-alignment (wav2vec2 CTC) also loses accuracy
  when forced to align a short, isolated chunk instead of a
  longer-context window (confirmed for `REWINDOW_SPLIT_ENABLED`, see
  below) — treat "don't run inference on a tiny isolated clip" as a
  general rule, not a pYIN-specific one.
- Don't trust individual ASR word timestamps for fine-grained
  boundaries — coarse anchoring only.
- Don't sort notes by timestamp as "harmless" cleanup — trust given
  order, only push overlaps forward.
- pYIN voicing reflects periodicity, not loudness — gate on RMS
  separately, using both a relative threshold and an absolute dBFS
  floor.
- A short note that jumps far from both neighbors (which match each
  other) is likely a glitch (`_remove_pitch_spikes`); tune
  `--spike-max-duration`/`--spike-jump-semitones` before assuming a new
  bug class.
- Merge/threshold constants need a cap on total drift across a whole
  chain, not just per-step (`_merge_similar_adjacent`'s
  `group_min`/`group_max`).
- An onset with no pitch change still needs a split path, or two
  re-attacked same-pitch notes merge. Fixed: only a STRONG onset (top
  percentile) can split same-pitch audio, only after the in-progress
  note ran long enough (`REARTICULATION_STRENGTH_PERCENTILE`,
  `MIN_DURATION_BEFORE_REARTICULATION_SEC`); result tagged
  `protected_start=True` so the next merge pass leaves it alone.
- A note's end time must come from its last INCLUDED frame
  (`times[end_frame - 1]`), not the frame after it — an off-by-one here
  silently caused constant `_ensure_nonoverlapping` warnings. A
  hard-guarantee check firing "routinely" means go find the bug, don't
  normalize it.
- **Demucs separation is not bit-reproducible** — can flip detected BPM
  for tempo-ambiguous songs. Mitigation: `work_dir` defaults to
  `<audio dir>/.ultrastar_work` so separation is cached/reused
  regardless of `--output-dir`. Check `#BPM` before assuming a
  regression when comparing runs.
- **Pass-1 CREPE inference is also not bit-reproducible** (bigger blast
  radius — can shift note pitch/timing, not just one BPM value).
  Partial mitigation in `_crepe_pitch`: deterministic algorithms/cudnn
  flags scoped to that call; `CUBLAS_WORKSPACE_CONFIG` set at import.
  Free (+1.2% runtime), no accuracy downside, but incomplete (still
  ~78% of notes can differ between runs). **Pass-1 output is still not
  guaranteed identical between runs** — check before treating a small
  diff as a regression.
- Real WhisperX ASR is *also* not deterministic in the literal words it
  transcribes (not just timestamps) — confirmed real case: "favors" vs.
  "favorites" for the same audio across different runs (see the MXL+LRC
  bug list below).
- **A whole-song/whole-file AGGREGATE quality metric can hide a real,
  badly-broken local passage.** Confirmed independently in two different
  mechanisms: a decoder-hallucination loop kept a song-wide ASR anchor
  rate healthy while one real passage was misplaced by 12+ seconds
  (Magic Dance); a single fully-dropped repeated line kept a reference-
  match ratio at 89-90% (Gold). Any future "does this look good enough"
  gate should consider a local/run-based signal (longest unmatched run,
  etc.), not just an aggregate ratio.
- **Repeated-phrase/occurrence disambiguation is an unsolved, recurring
  failure class**, not a one-off bug — hit independently in at least 5
  different mechanisms across this project's history (`--verify-
  placement`'s expand-search, `realign.py`'s `"windowed"` and `"seed"`
  matching, both rejected rewindow follow-up prototypes). A search/match
  window containing more than one real occurrence of a repeated word or
  phrase has no way to pick the right one on its own, and confidently
  returns whichever it finds — sometimes far away. No aggregate
  confidence signal reliably catches this; only a real disambiguation
  mechanism (not yet built) would. Don't re-propose "just search a
  window and check if the text is there" as a fix for a placement bug
  without addressing this directly.

## Removed / rejected approaches (don't re-attempt without new evidence)

- **Key correction**: deleted entirely. A single global detected key
  blindly snaps legitimate out-of-scale/modal-mixture notes to the
  wrong pitch. Confirmed harmful on real audio (a real, deliberate
  modal-mixture/borrowed note got blanket-snapped wrong every
  occurrence). A narrower, external ±1-semitone-only variant
  (`ultrastar_pitch`'s key nudge, vendored into `pitch_refresh.py`, see
  its own memory) reduces the blast radius of this same failure class
  but doesn't eliminate it — still not safe as a blanket rule.
- **RMVPE as cross-checked primary pitch source**: reverted — net
  regression end-to-end despite winning in isolation-mode comparisons.
  Isolation-mode accuracy does NOT reliably predict end-to-end impact.
  (What *did* ship: `isolation_source="rmvpe"`, RMVPE's own voicing, no
  cross-check at all — see Shipped Defaults below.)
- **`--verify-placement`** (expand-search re-transcription to fix note
  boundaries): fixed individual real cases but was a net regression on
  every pitch/timing metric end-to-end on every tested song, confirmed
  on two separate occasions months apart (most recently: BATB words
  matched 105→101, timing agreement 97.1%→93.1%). Root cause: the "is
  the expected word's text present anywhere in this search window"
  check can't disambiguate WHICH occurrence of a repeated word is
  correct when the window contains more than one — the general
  repeated-phrase disambiguation problem noted above. **Fully removed
  from the codebase** (user's explicit request — different treatment
  from `--zone-boundary-snap` below, which stays as dead-but-present
  code): `verification.verify_placement`, `PlacementCorrection`/
  `PlacementWarning`, the CLI flags, the GUI checkbox, and its tests are
  all gone.
- **`--zone-boundary-snap`** (snap zone/word boundaries to nearby pass-1
  note onsets): synthetically verified, but flat-to-negative on all 5
  real songs tested. Kept in codebase, off by default, not worth
  further tuning.
- **`lrc_timing.py`'s flagging as an auto-correction signal**: built as
  diagnostic-only (flags, never auto-corrects). Ground-truth
  cross-validation showed flagged lines are NOT reliably less accurate
  than unflagged — the drift it detects reflects LRCLIB being timed to
  a *different recording*, not a defect in our output. **Don't build
  auto-correction directly from this signal.** Stays diagnostic-only
  (a separate calibration/correction use of the same module's
  `two_tier_time_calibration` IS shipped and trusted elsewhere — see
  the MXL+LRC and `realign.py` sections below; the rejection here is
  specifically about the standalone flagging check, not the whole
  module).
- **Lead-vocal/harmony separation** (to fix `little_mermaid`'s poor
  pitch accuracy): tested 3 ways (two separator models, pre- and
  post-Demucs), all null results. Root cause instead: genuine acoustic
  ambiguity in rough/character vocal styles (confirmed across 4
  independent pitch estimators converging on the same wrong answer) —
  fixed via `musicxml_reference.py`'s `force_calibration` instead (see
  below).
- **Reference-line vs. musical-phrase tension** (e.g. "under the sea
  under the sea" merged into one un-splittable line): a refined
  detection rule was validated against real data but the user decided
  to leave `phrasing.py` as-is and accept this as a known rare issue
  rather than add another mechanism.
- **Local-rematch of unmatched word runs** (`realign.py`'s
  `rematch_local_gaps`, retries a run of unmatched words against only
  nearby-in-time ASR instead of the whole song): net regression across
  a controlled 4-song A/B (BATB −24pp within 100ms). The reference
  songs' existing local timing is already trustworthy, so
  `interpolate_fallback`'s proportional guess usually beats a local
  rematch — and a local rematch can lock onto the wrong nearby repeat
  just as easily as a whole-song search can. Shipped OFF, code kept
  (not CLI/GUI-wired); would need a way to distinguish "original timing
  is wrong here" from "ASR is just sparse here" before it's safe.
- **Split-into-sub-phrases rewindow** (`REWINDOW_SPLIT_ENABLED`,
  `transcription._rewindow_split_segment`): breaking a long decoder
  segment into pieces and windowing each separately scored WORSE than
  whole-block alignment on every attempted case — the "don't run
  inference on a tiny isolated clip" failure class again. Kept off by
  default, dead code.
- **LRC-line-anchored recovery** (used LRC line timing as independent
  ground truth to re-align a whole flagged decoder-hallucinated line):
  real-tested on Chicago — one confirmed regression (moved an
  already-correct passage 0.65-1.09s away from ground truth), one
  wasted no-op (pass 3's own line-level proportional distribution
  already absorbs a raw-word-level "fix"), one safe decline, zero wins.
  **Fully removed from the codebase** (user's explicit "reject, don't
  keep the functionality around").

## Shipped defaults / current config (as of 2026-08-10)

- `isolation_source="rmvpe"` is the real pitch-source default (RMVPE's
  own voicing, no cross-check, no CREPE) — reproducible, faster, and a
  real average +1.7pp accuracy win across the 4-song core set.
- `ENABLE_MUSICXML_FORCE_CALIBRATION = True`: when a user supplies (or
  auto-detects) a MusicXML reference and normal pass-4 calibration can't
  clear its confidence bar, apply the best available pitch-class offset
  anyway rather than skipping. Zero regressions across all 7 tested
  MXL-having songs; big wins (+19–22pp) on rough/character-vocal songs.
  Never touches octave or timing.
- `file_discovery.find_companions` auto-detects `.mxl`/`.musicxml`/
  `.xml` reference files by extension; bare `.xml` is content-sniffed
  (`_looks_like_musicxml`) before trusting it, since some games ship an
  unrelated `notes.xml` with the same extension. Multiple reference
  files are all applied sequentially (different arrangements often
  cover different, only-partly-overlapping portions of a song).
- `lyrics_lookup.py`: tries LRCLIB first, falls back to lyrics.ovh.
  `reference_matches_transcript` (difflib vocabulary-ratio check,
  `REFERENCE_LYRICS_MIN_MATCH_RATIO = 0.25`) rejects a wrong-song/
  wrong-language reference before it's trusted.
  `REFERENCE_CLAMP_MAX_REPEAT = 8`: when a decoder hallucination
  produces a wildly uneven `align_words_to_reference` replace block
  (many ASR words vs. one reference token — a real case had ~90 ASR
  words clamp onto one reference token), clamping that many words onto
  a single reference token IS ITSELF the hallucination signal — past
  the cap, no `reference_text` is fabricated at all (falls back to the
  ASR word's own raw, still-garbled text, same as an ordinary uneven
  block) instead of confidently asserting a wrong syllable. The
  syllable-cursor for repeat-clamped blocks under the cap also wraps
  (not freezes) across repeats.
- `verification.py`'s no-reference fallback no longer blindly replaces
  correct ASR text with an isolated recheck's guess — keeps
  full-context text when nothing can confirm the recheck (same
  "don't trust a tiny isolated clip" lesson as pitch).
- `phrasing.py`: when current line and next word share a KNOWN,
  matching reference `line_id`, nothing breaks the line early except
  the 1.5x-`MAX_SYLLABLES_PER_LINE` implausible-length safety net — a
  reference line wins outright even over a long gap.
- CUDA is the only supported device; `--device`/CPU fallback removed
  entirely, pipeline aborts at startup if CUDA unavailable.
- CREPE runs alongside pYIN (`--no-crepe` to disable) as a cross-check
  where relevant, but note: RMVPE isolation mode above bypasses this.
- `BPM_WRITE_MULTIPLIER = 2`: written `#BPM` is 2x the detected tempo
  for finer beat-grid resolution (display/write-time only, not fed into
  `detect_notes()`'s own segmentation).
- `config.REWINDOW_ENABLED = True` (default everywhere): see
  "Long-segment rewindowing" under the `realign.py` section below —
  applies to both `realign.py` and `main.py`.
- `config.FORCE_ALIGN_GAPS = True`, `config.MERGE_CONNECTED_MELISMA_TAILS
  = True` (the latter `main.py`-only): see "UltraStarKaraokeMaker-
  inspired improvements" below.
- `NOTE_MERGE_SEMITONES = 1` (default, investigated and confirmed to
  stay) — a wider `=2` was tried to fix note over-segmentation, found a
  clean win on BATB but ~15-20% genuinely-ambiguous real melodic steps
  in a structural analysis, and this constant already has a specific
  confirmed prior-regression history at `=2` (see its own docstring in
  `config.py`). Not worth the risk without better ground truth.

### Folder-based input / batch / GUI / launcher (all shipped)

- CLI input is a folder, not a file. `file_discovery.
  resolve_primary_source` auto-classifies: single audio file → normal;
  single `.mp4` → serves as both `#MP3` and `#VIDEO`; single `.avi` →
  audio extracted via `media_extract.py`, avi becomes `#VIDEO`
  (aborts cleanly if the avi has no audio track); ambiguous folder →
  `AmbiguousInputError`, requires `--audio-file` override (never
  silently guesses).
- Embedded cover art extraction via `mutagen` (ID3 APIC, MP4 `covr`,
  FLAC picture list, OGG/Opus vorbis-comment picture), sniffed by magic
  bytes not claimed MIME type — used as fallback when no `.jpg`
  companion exists.
- **Existing-file verification** (`--existing-txt`/`--existing-txt-check`,
  **off by default** — the one deliberate exception to "default on",
  since it can result in NOT regenerating output the user expected):
  parses an existing `.txt`, compares pitch-class/timing against a
  fresh pass, keeps the existing file byte-for-byte on `PASS`, else
  regenerates normally. Gate includes `EXISTING_TXT_MIN_COVERAGE = 0.85`
  (fraction of words that matched at all) alongside pitch/timing
  accuracy — a clean-looking matched subset can no longer report PASS
  while a real chunk of the song went unscored.
- **YouTube input** (`--youtube-url`, requires `--artist`/`--title`):
  `yt-dlp` downloads to a deterministic filename, falls through the
  same folder-resolution logic as local files (no special-casing
  needed). Thumbnail auto-downloaded and renamed as the `[CO]` cover
  companion.
- **Batch mode** (`--batch`): runs per-subfolder, isolates failures
  (one bad song doesn't abort the batch), mirrors output structure by
  subfolder name. Rejects combination with `--work-dir`/`--audio-file`/
  `--artist`/`--title`/`--existing-txt`/`--youtube-url` up front (none
  make sense as a single override across multiple songs). Exit code 2 =
  partial failure, distinct from 1 = total failure. `realign.py` has
  its own `--batch` with the same incompatibility-checking convention
  and its own `run_realign_batch` — no `output_parent_dir` concept
  there, since every result always writes next to its own subfolder's
  existing file.
- **Tkinter GUI** (`gui.py`, stdlib only): wraps `run_pipeline`/
  `run_batch`/`run_realign_pipeline`/`run_realign_batch`. Three modes
  ("Generate song file" / "Realign existing file" / "YouTube URL") plus
  an orthogonal Batch checkbox (disabled for YouTube — a single URL
  can't populate multiple subfolders). Toggling Batch DISABLES (not
  hides) the audio-file/Artist/Title/LRCLIB-ID fields rather than
  removing them — a single override silently applied to every subfolder
  was a real pre-existing bug (the GUI calls `run_batch` directly,
  bypassing the CLI's own incompatibility checks). Folder-picker memory,
  live placeholders, tooltips, non-yanking log auto-scroll, "Open Output
  Folder"/"Delete Intermediate Files Now" buttons. Runs pipeline on a
  background thread; captures ALL print output (including deep
  submodules never rewired to a `log` callback) via
  `contextlib.redirect_stdout` at the call boundary into a polled
  queue — deliberately not threading `log` further down.
  **`PlaceholderEntry` gotcha** (real bug, fixed): setting a
  `PlaceholderEntry`'s value via a Browse-dialog callback must use
  `set_real_value()`, not `.set()` on the underlying `StringVar`
  directly — only `<FocusIn>` clears `is_placeholder`, and a Browse
  button click never focuses the entry itself, so a direct `.set()`
  silently leaves `effective_value()` returning `None` even though the
  field visibly shows the picked path. Apply this to any future
  `PlaceholderEntry` write from code, not just user typing.
- **`run_gui.bat`** launches via `pythonw.exe` (no console window),
  checks the venv exists first with a clear error.
- **`--output-dir`** is the PARENT folder a `<Artist> - <Title>` folder
  gets created under (not the final folder itself); omitted defaults to
  `<input_dir>/Output`. Debug files (`[DEBUG LOG]`, `[PASS1 DEBUG]`)
  write to `<input>/.ultrastar_work`, not the output folder.
  `main.delete_work_files` deletes the ENTIRE work_dir, debug files
  included (previously scoped to just `separated/`/`extracted/`).
  `realign.py`/`pitch_refresh.py` have their own `--delete-work-files`
  following the same wrapper/`finally` shape.

### Interactive LRCLIB lyrics selection (GUI)

- Manual "Search Lyrics..." button + dialog to pick/pin a candidate
  (`PipelineOptions.pinned_lyrics` always wins over automatic fetch).
- Automatic mid-run ambiguity prompt (opt-in checkbox, single-song mode
  only): pauses the background pipeline thread, opens the same dialog
  on the main thread via `self.after(0, ...)` + `threading.Event`,
  resumes on selection/cancel. Never triggered in batch mode. **Gotcha**:
  cross-thread `self.after(...)` only works when the main thread is
  genuinely inside a real `Tk.mainloop()` — a manual `update()`-polling
  test loop raises `RuntimeError: main thread is not in main loop`, so
  any test exercising this path needs a real `mainloop()` (a watchdog
  thread calling `app.after(0, app.quit)` to stop it).
- `--lrclib-id <id>` / GUI "LRCLIB ID" field fetches one specific entry
  directly (`/api/get/<id>`), always wins over search/pinning.
  `--lrc-file <path>` (CLI + GUI Browse) builds the same pinned
  candidate directly from a local `.lrc` file's text, no network fetch
  — wins over `--lrclib-id` if both given.

## MXL + synced-lyrics as PRIMARY generation path (shipped default,
## `ENABLE_MXL_LRC_PRIMARY = True`)

For songs with a MusicXML score AND matching LRCLIB synced lyrics, skip
audio-only pass 1–4 entirely: MXL supplies pitch/rhythm shape directly,
LRC line timestamps anchor real time, real ASR of our own audio places
words precisely within each line.

**Design**: trust LRC line starts as hard anchors; place words within a
line via real ASR match (order-preserving, timestamp inside the line's
window); non-confident/unmatched words fall back to interpolation from
nearest CONFIDENT neighbors (by MXL offset, not bounded to one line)
using a locally-calibrated real-seconds-per-quarter-note rate; clamped
into the word's own LRC line window as a backstop. Real result on a
validated song: 100% pitch-class, 99% timing within 500ms, 105ms mean
error, using **zero audio pitch detection**.

**Quality gate is downstream, not upfront**: upfront duration+content
filtering on LRCLIB candidates is NOT sufficient — a candidate can pass
generously and still be timed to a *different recording* entirely
(confirmed on multiple real songs). Real gate:
`MxlLrcQuality.asr_placement_rate >= MXL_LRC_MIN_ASR_PLACEMENT_RATE`
(0.5) — fraction of MXL words confidently matched against our own
audio's real ASR transcript. A wrong-recording candidate collapses this
on its own. On gate failure: CLI logs a warning and falls through to
standard pass 1–4 pipeline; GUI (single-song only) prompts
Continue/Cancel via the same thread-hop pattern as the lyrics
ambiguity prompt. An `ASR quality retry` (re-transcribe with large-v3,
see the `realign.py` section below) fires before giving up, when
`opts.batch` is set.

**Real production bugs found and fixed** (validated against the actual
written file, not in-memory floats — a critical distinction: the
original prototype's "99% accurate" numbers were computed by comparing
in-memory `Syllable` floats directly, and wiring the same logic through
the real CLI/writer dropped measured accuracy from 99% to 24.5% — three
of the bugs below were hiding entirely behind that gap):

1. Word end time comes from ASR's own end (trusted match) or
   MXL-note-value × local tempo rate, clamped to never exceed the next
   word's start — not stretched to fill the whole gap to the next word
   (which swallowed real pauses).
2. ASR matches are gated on `MXL_LRC_MIN_ASR_WORD_CONFIDENCE = 0.3` — a
   near-zero-confidence match used to be trusted anyway.
3. `BPM_WRITE_MULTIPLIER = 2` (general fix, not MXL-specific) — MXL-
   derived syllable density needed finer write-time beat-grid
   resolution than the detected tempo alone gave.
4. Lyric text prefers a clean-text replacement from the matched LRC
   token over the MXL's raw OCR'd syllable text (exact match, or fuzzy
   ≥ `MXL_LRC_FUZZY_TEXT_MIN_RATIO = 0.6` for a 1:1 anchored slot); a
   word with no anchor on either side keeps its raw OCR text rather
   than guessing.
5. Fallback duration/position interpolates from nearest CONFIDENT
   neighbors (by MXL offset) instead of the whole LRC line span, which
   broke badly on lines with trailing silence.
6. `verify_existing_song.py`'s coverage gate (`coverage_fresh`/
   `coverage_existing` + `EXISTING_TXT_MIN_COVERAGE = 0.85`) exists
   because early comparisons silently dropped non-matching words
   instead of counting them as failures, inflating reported accuracy.
   **Use `verify_existing_song.verify_existing_song` directly for any
   future real-output-vs-ground-truth comparison** — don't write
   another ad hoc script.
7. A lyric-less MXL continuation note (tied hold / slurred pitch slide)
   was silently dropped — now extends the in-progress syllable
   (tied+same-pitch) or becomes a new empty-text syllable
   (slurred/different-pitch), via the existing melisma-padding
   mechanism. Gated on exact contiguity so an unrelated post-rest note
   can't glue onto the wrong word.
8. ASR word-matching now prefers the resolved clean LRC text over the
   MXL's raw OCR text, with a fuzzy fallback — WhisperX isn't
   deterministic in which literal word it transcribes run to run
   (confirmed: "favors" vs. "favorites" for the same audio, different
   runs).
9. The fuzzy fallback above only checked a clean 1:1 replace block
   (`(b2-b1)==1`) — but the per-line ASR candidate window
   (`asr_in_window`) is TIME-bounded, not line-bounded, so the very
   next line's first word can ride into the same window (real Chicago
   case: "I'm," the next line's first word, rode into "favorites"'s
   window and turned a clean 1:1 match into a false 1:2 block that got
   rejected outright even though "favorites" — the correct answer — was
   sitting right there). Fixed by relaxing to `(a2-a1)==1` only (still
   only ever a single unresolved MXL word) and trying every ASR word in
   the block, keeping the best fuzzy-ratio match.
10. **No calibration for a systematic LRC-vs-audio time offset** (e.g.
    extra lead-in silence in our recording vs. whichever recording LRCLIB
    was timed against) — real case: "Ordinary Day" had a consistent
    ~+2.4s offset that blew the quality gate outright (22% non-monotonic
    placements) since every line's search/interpolation window was off
    by the same amount. Fixed: a global offset(+drift) is calibrated
    BEFORE any word is placed, via `_match_asr_to_lrc_lines` feeding
    `lrc_timing.two_tier_time_calibration` (shared with `realign.py`,
    not reimplemented). A null/near-zero calibration is a no-op —
    already-well-aligned candidates are unaffected.
11. **Multi-word MXL OCR garbling was left completely unmatched** when
    it spanned more than one word (e.g. `"winnes"` for real "win now";
    `"stomty"`+`"in"` for real "stop"+"trying,"; a 3-word OCR block for
    a real 2-word hyphenated "double-edged"). Fixed via a whole-block
    fuzzy match (up to `MXL_LRC_BLOCK_MAX_WORDS = 6` words) +
    `_distribute_words_to_slots` (hyphen-splits or melisma-pads to
    reconcile word-count mismatches) — still never an unconstrained text
    search, always bounded by the block's own real neighboring matches.

**Candidate override**: `LrcLibCandidate.id` + `fetch_lrclib_by_id` +
`--lrclib-id`/GUI field let a user paste a manually-confirmed LRCLIB id,
always wins over search/auto-pick.

## Alignment-only mode (`realign.py`, shipped 2026-08-09)

A separate CLI (`python -m ultrastar_generator.realign <folder>
[--existing-txt <path>]`, own `run()`/`build_arg_parser()`, GUI's
"Realign existing file" mode calling the same `run_realign_pipeline`/
`run_realign_batch`) that takes an ALREADY-WRITTEN UltraStar `.txt` plus
its audio and re-times it: `#GAP`, note start, and note length only.
Never touches pitch, never adds/removes/reorders a note. Assumes the
input's notes are in the right order and its lyric TEXT is correct;
makes NO other assumption about its timing quality (survives the
degenerate case of a flat list of equal-length placeholder notes that
don't match the audio at all). `--existing-txt` is optional — both
single-song and batch mode auto-detect the folder's single `.txt` via
`find_existing_txt_in_folder` (fails closed with
`AmbiguousExistingTxtError` on zero/multiple candidates; excludes this
module's own `"[REALIGNED]"` naming so re-running on an already-realigned
folder can't pick its own output as the next input; a `"<folder
name>.txt"` basename match is tried as one further disambiguation before
giving up).

**Design** (deliberately different from `mxl_lrc_generator.py`'s
per-LINE-windowed ASR search, though the anchor/interpolate shape is
borrowed from it): a WHOLE-SONG, order-preserving text match of the
existing file's own words against real ASR words (`match_words_to_asr`)
— never time-windowed by default, since this mode can't trust the
input's own timing enough to window a search with it (see `lrc_mode`
below for the exception). Everything unanchored is placed by
`interpolate_fallback`: two confident neighbors → rate-interpolate using
the word's OWN original start as a proportional offset; one neighbor →
constant shift, not a rate extrapolated from a single point; zero
anchors anywhere in the whole song → keep the word's ORIGINAL timing
completely unchanged. Sub-word syllables redistribute within a word's
new span using its OWN original relative syllable timing.

**Prerequisite bug fix, `usdx_parser.py`**: word-boundary detection
(`is_word_start`) only ever checked for a LEADING space (this project's
own writer convention). A real SingStar-shipped ground-truth file (BATB)
uses a TRAILING space on a word's LAST syllable instead and no leading
spaces at all, which silently merged lines into one bogus word. Fixed by
also checking the PREVIOUS syllable's own trailing whitespace — a strict
superset of the old check.

**Real validation (BATB, hand-verified SingStar ground truth used as an
"already correct" input)**: 140/140 syllables preserved, 0 pitch
mismatches, 108/113 words matched directly to ASR. One real outlier
("Tune," landed 2.6s early — WhisperX's own forced-alignment crossing a
real musical rest) not chased further, per this project's own precedent
of individually-well-motivated fixes for this exact case turning out to
be net regressions (`--verify-placement`/`--zone-boundary-snap`).

### Shipped behavior & defaults

- `lrc_mode="windowed"` (LRC lines primary, ASR only resolves position
  within a line — mirrors `mxl_lrc_generator.place_words_via_asr`'s
  per-line window, adapted to match against the existing file's own
  trusted text) is the default, but ONLY when `LrcPrep.
  calibration_offset is not None` (i.e. `two_tier_time_calibration`
  found a confident fit) — otherwise it transparently falls back to
  whole-song ASR matching, identical to `lrc_mode="seed"`'s own
  behavior. **This gate is required, not optional**: windowing an
  UNCALIBRATED candidate is actively harmful, since every word's match
  routes through the same untrusted, possibly-drifting signal (a real
  Chicago regression: deltas grew smoothly to −2.6s across the back half
  of the song). `"seed"` (whole-song ASR primary, LRC only seeds
  residual gaps) stays available as an explicit opt-out. Real 5-song
  comparison: `"windowed"` wins clearly on 2, ties on 2, loses on 1 (a
  highly-repetitive-chorus song, where per-line windowing can lock onto
  the wrong occurrence of a repeated phrase — the general
  repeated-phrase problem noted in Lessons learned) — stays default.
- `interpolate_fallback`'s trailing monotonic clamp is CONFIDENCE-AWARE:
  a confident word's value is a fixed point, never rewritten by either
  direction of the clamp; only fallback words get pushed/pulled to
  preserve ordering. (Before this fix, one bad anchor — an uncalibrated
  LRC seed, or a per-line window that locked onto the wrong occurrence
  of a repeated phrase — could flatten or overshoot many already-correct
  neighboring matches; see "Real bugs" below.)
- `lrc_timing.check_repeat_structure` (moved here from `realign.py`,
  re-exported from `realign.py` for backward compat) rejects an LRC
  candidate whose repeat structure doesn't match ours — compares WORD
  occurrence counts of the most-repeated line's own distinctive content
  words (not exact-line-repeat counts, since a chorus is often split
  across several near-duplicate line variants that would each pass a
  naive per-line check individually), tolerance ±15%/min ±1. Catches a
  wrong-edition/wrong-arrangement candidate that duration/content
  matching alone misses (real case: a candidate with 9 extra chorus
  repeats, invisible in duration since it coincidentally matched within
  the existing tolerance).
- `force_align_gaps` and `retry_low_quality_asr` (see below) default ON.
  `rewindow_long_segments` (see below) defaults ON, independently of the
  shared `config.REWINDOW_ENABLED` used elsewhere.
- Strategy: `"replace"` (default, shipped) vs. `"validate"`
  (`--strategy validate`, GUI-selectable, trusts the original position
  when ASR confirms it's close) — validate only helps when the input
  file is ALREADY accurate, which defeats the point for the real
  problem (files that don't match the audio well); kept as an explicit,
  non-default option, not pursued as the general solution.

### Real bugs found & fixed — root causes worth remembering

- **A single bad LRC anchor's blast radius is unbounded**, regardless of
  how few words it directly seeds, once a monotonic clamp propagates it
  forward/backward to the next real anchor. Confirmed twice: "Heroes"
  (the auto-picked candidate was a choral COVER, not Bowie's own
  recording — `calibration_confidence=0.0` — and the old `"seed"` mode
  had NO calibration gate on its own LRC-seeding step, on the
  since-disproven theory that seeding only a few words keeps the blast
  radius small); "I'm Afraid of Americans" (a per-line window locked
  onto the wrong occurrence of a repeated phrase, producing a negative
  interpolation rate for the word just before it, which triggered a
  degenerate fallback that overshot 14 later genuinely-confident
  matches, then flattened all of them to one wrong point via the old
  clamp). Both fixed by the calibration gate + confidence-aware clamp
  described above.
- **Decoder hallucination on a long, repeat-heavy or ad-lib-containing
  passage can DROP real content and misplace what remains inside one
  oversized decoder segment** — a literal, exact-text decoder-level
  failure (confirmed via raw decoder-segment dumps), not a matching or
  occurrence-ambiguity bug. Real cases: Americans' "Johnny wants a
  brain..." (a scat/ad-lib line dropped entirely; the few remaining
  words spread across a ~19s window landed ~6-7s too early); Magic
  Dance's "Dance Magic Dance..." chorus (decoded as ~90 garbled repeats
  instead of 5 clean ones — see `REFERENCE_CLAMP_MAX_REPEAT` above).
  **Tested and rejected**: reverting to WhisperX's default VAD makes
  segments BIGGER, not smaller (one 26.6s segment swallowed a whole
  verse, with visible text reordering) — don't revert the existing
  near-zero-VAD default (`WHISPERX_NO_VAD_OPTIONS`, chosen for a
  different real fix on "Stars"). `condition_on_previous_text` is
  already `False` in whisperx's own defaults;
  `hallucination_silence_threshold` doesn't apply (real singing, not
  silence). This motivated three real, shipped mitigations: long-segment
  rewindowing, ASR-quality retry, and `--no-transcribe` (all below).

### Long-segment rewindowing (`config.REWINDOW_ENABLED`, default ON everywhere)

For any whisper decoder segment ≥ `REWINDOW_MIN_SEGMENT_DURATION_SEC`
(10s), sweeps fixed-10s-width candidate windows at 1s steps across the
segment's own span, re-running real `whisperx.align()` per candidate,
and keeps whichever wins by mean word score — only if it beats the
baseline (whole-segment) score by ≥ `REWINDOW_MIN_SCORE_IMPROVEMENT`
(0.10). Only ever corrects segment BOUNDARIES before the normal
alignment call; everything downstream is untouched, so it composes for
free with everything else. **Design lesson**: width matters far less
than exact offset — a coarse or under-covered offset sweep can converge
on a confident-looking but WRONG local optimum (this happened twice
during development; the second false optimum was only caught because
the user listened and confirmed the true position by ear). A fine
(1s-step), sufficiently wide sweep is what actually finds the right
answer, and does so with a sharp, unambiguous score peak, not a fuzzy
one. 12 real validation runs across 8 songs found 6 genuine fixes (all
verified against trusted/written output, not just the raw score) and
zero regressions before shipping default-on everywhere; a low-confidence
segment that's already correctly placed (e.g. a sustained note dragging
CTC score down without being misaligned) correctly does not trigger a
change.

### ASR quality retry (`retry_low_quality_asr`, default ON, gated behind `--batch`)

Re-transcribes the whole song once with `RETRY_ASR_MODEL` ("large-v3")
when a quality signal is low, keeping the retry only if it's actually
better on whichever signal triggered it (never retries a second time).
Two independent triggers per call site — a whole-song AGGREGATE metric
alone is not enough (see "Lessons learned" above): a whole-song rate
(`anchor_rate` / `asr_placement_rate` / `reference_match_ratio`,
thresholds `MXL_LRC_MIN_ASR_PLACEMENT_RATE` / `RETRY_ASR_MIN_REFERENCE_
MATCH_RATIO=0.6`) AND a longest-unmatched-RUN trigger
(`RETRY_ASR_MIN_UNCONFIDENT_RUN` / `RETRY_ASR_MIN_UNMATCHED_REFERENCE_
RUN`, both first estimates at 5). The run-trigger needed
`lyrics_lookup._tokenize_lines` to split every token on `-` as well as
whitespace — a hyphenated multi-syllable token (e.g. "Do-do-do-do-do")
was counting as ONE word, splitting what's really one long dropped
passage into several under-threshold pieces that individually never
tripped the trigger. **Gated to `--batch` mode only**: outside batch, a
low-quality trigger logs a warning (naming the metric, suggesting
`--batch` or `--whisper-model large-v3`) instead of silently doubling
that run's ASR time — an interactive single run shouldn't pay that cost
without being asked. Wired at 3 call sites: `realign.py` (`strategy=
"replace"` only — `"validate"`'s `ValidateQuality` has no `anchor_rate`),
`main.py`'s MXL+LRC path, and `main.py`'s standard fallback path (uses
`lyrics_lookup.reference_match_ratio` — has no retry available at all
when `--no-fetch-lyrics`/no reference is found, nothing to measure
against).

### `--no-transcribe` (`main.py` only, diagnostic, off by default)

Skips WhisperX's decoder ENTIRELY for the whole initial word list —
`transcription.force_align_reference_lyrics` builds every word via real
wav2vec2 CTC forced alignment of a pinned LRC candidate's own line text
only (no free transcription anywhere), each LRC line getting its own
alignment window. Requires a pinned LRC candidate with synced lyrics
(`--lrclib-id`/`pinned_lyrics`); without one, warns and falls back to
normal transcription. Applies uniformly to BOTH the MXL+LRC and standard
generation paths (deliberately not special-cased out of either — the
point is checking for decoder-hallucination bleed everywhere). Real-
validated on Magic Dance: 254/254 words force-aligned, zero decoder
calls, the "Dance Magic Dance"/"Jump Magic Jump" passages that were
hallucinated garbage under normal transcription came back verbatim
correct. Off by default — it's a diagnostic/isolation tool (needs a
trustworthy pinned candidate, and gives up whatever real info the
decoder would add for content the LRC doesn't cover, e.g. ad-libs), not
a general default replacement.

## UltraStarKaraokeMaker-inspired improvements (2026-08-10)

Compared our own output against a real run of UltraStarKaraokeMaker
(github.com/walterfr/UltraStarKaraokeMaker, MIT-style OSS — see its own
memory entry for the broader comparison) on "Trixie Mattel - Gold."
Three concrete outcomes:

1. **Force-align known gaps — shipped, default ON.**
   `transcription.force_align_words_in_window` (real wav2vec2 CTC
   forced alignment of KNOWN text into a bounded audio window), adapted
   from USKMaker's own `realign_gap_windows`. Wired into both
   `realign.py` (`_force_align_unconfident_runs`) and `main.py`'s
   standard fallback path (`lyrics_lookup.
   recover_dropped_reference_words`). `config.FORCE_ALIGN_GAPS`,
   `--no-force-align-gaps` opts out.
2. **Note over-segmentation — investigated, NOT shipped.** Confirmed
   real: pass-1's raw note count on Stars was ~2.8x the true (OMR)
   count, and untexted `~` continuation-note rate varies wildly by song
   (0-33%) under our pitch-first architecture vs. USKMaker's much lower
   rate (forced-alignment-first, pitch extracted per-syllable
   afterward). See `NOTE_MERGE_SEMITONES` in Shipped Defaults above for
   why the obvious fix (widen the merge threshold) was rejected.
3. **Melisma-tail merge — shipped, default ON in `main.py` only.**
   `usdx_writer._merge_connected_melisma_tails`: a beat-adjacent,
   exact-same-pitch `~` continuation note immediately following a real
   syllable is redundant and gets folded into the preceding note
   (chains across several connected same-pitch `~` notes; never crosses
   a LineBreak). Runs at the INTEGER BEAT level (post-quantization), not
   float-seconds, since quantization itself can change what counts as
   adjacent. Deliberately NOT applied in `realign.py` — its own contract
   (never add/remove/reorder a note) would be violated by deleting a
   note. `config.MERGE_CONNECTED_MELISMA_TAILS`,
   `--no-merge-connected-melisma` opts out.

All three, plus `retry_low_quality_asr`, validated together (not just in
isolation) across 6 real gen+realign runs — zero regressions.

## Environment

- Windows, venv at `E:\Projects\ultrastar_generator\venv`.
- CUDA required — pipeline aborts at startup if unavailable. WhisperX
  pulls in pyannote/torch; expect noisy-but-harmless startup warnings.
- Demucs stems cached in `sandbox/.ultrastar_work/` — safe to delete
  between runs; separation is skipped if `vocals.wav` already exists.
- **No console-window flashing**: every `subprocess.run` call that
  spawns a real console-subsystem executable (ffprobe/ffmpeg in
  `media_extract.py`, Demucs in `separation.py` — the only 3 call sites
  in the repo) passes `creationflags=subprocess.CREATE_NO_WINDOW` (a
  module-level `_NO_WINDOW` constant per file, `0` on non-Windows) —
  otherwise Windows pops a new console window per call whenever the
  parent process has none of its own (the GUI, launched via
  `pythonw.exe`). Harmless when a console DOES already exist (CLI run
  from a terminal). yt-dlp (YouTube mode) is a Python library call, not
  a subprocess, so it isn't affected.

## Open / deferred (not yet built, needs direction before starting)

- Consuming `LyricsResult.synced_lyrics` for line-timing correction —
  explicitly rejected as designed (see `lrc_timing.py` above); would
  need a fundamentally different signal, not a refinement.
- MXL+LRC-only generation for multi-vocal-harmony songs (prototype only,
  KPop Demon Hunters case) — real structural-cut handling and residual
  timing/pitch precision questions still open, deliberately deferred in
  favor of validating general viability first (done: works very well on
  simple single-part songs like BATB — 100% pitch-class, but timing
  capped ~37% within 500ms there for unresolved reasons, possibly LRC
  per-line timestamp imprecision or real rubato).
- Adjacent-and-full-line-coverage repeated-phrase detection for
  `phrasing.py` (validated against real data, not implemented — see
  "Removed / rejected" above, user chose not to build it).
- Repeated-phrase/occurrence disambiguation, generally (see "Lessons
  learned" above) — the single biggest recurring unsolved failure class
  in this codebase; no design proposed yet.
