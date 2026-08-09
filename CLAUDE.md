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

## Lessons learned (do not reintroduce)

- Never run pYIN on a tiny isolated clip — needs context. Fixed by
  analyzing the whole track once; pass-2 fallback borrows pitch instead
  of re-running pYIN on ~0.1s clips.
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
  transcribes (not just timestamps) — confirmed real case (see
  MXL+LRC section).

## Removed / rejected approaches (don't re-attempt without new evidence)

- **Key correction**: deleted entirely. A single global detected key
  blindly snaps legitimate out-of-scale/modal-mixture notes to the
  wrong pitch. Confirmed harmful on real audio.
- **RMVPE as cross-checked primary pitch source**: reverted — net
  regression end-to-end despite winning in isolation-mode comparisons.
  Isolation-mode accuracy does NOT reliably predict end-to-end impact.
  (What *did* ship: `isolation_source="rmvpe"`, RMVPE's own voicing, no
  cross-check at all — see Shipped Defaults below.)
- **`--verify-placement`** (expand-search re-transcription to fix note
  boundaries): fixes individual real cases but is a net regression on
  every pitch/timing metric end-to-end on tested songs. Off by default;
  don't flip on without addressing this.
- **`--zone-boundary-snap`** (snap zone/word boundaries to nearby pass-1
  note onsets): synthetically verified, but flat-to-negative on all 5
  real songs tested. Second independent instance of "well-motivated
  mechanism fails to generalize" (same shape as verify-placement). Kept
  in codebase, off by default, not worth further tuning.
- **`lrc_timing.py`** (LRCLIB synced-lyrics line-timing cross-check):
  built as diagnostic-only (flags, never auto-corrects). Ground-truth
  cross-validation showed flagged lines are NOT reliably less accurate
  than unflagged — the drift it detects reflects LRCLIB being timed to
  a *different recording*, not a defect in our output. **Don't build
  auto-correction from this signal as designed.** Stays off by default,
  diagnostic value only.
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

## Shipped defaults / current config (as of 2026-08-09)

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
  subfolder name. Rejects combination with `--work-dir`/`--artist`/
  `--title`/`--existing-txt`/`--youtube-url` up front. Exit code 2 =
  partial failure, distinct from 1 = total failure.
- **Tkinter GUI** (`gui.py`, stdlib only): wraps the same
  `run_pipeline`/`run_batch` as the CLI. Folder-picker memory, live
  placeholders, tooltips, non-yanking log auto-scroll, "Open Output
  Folder"/"Delete Intermediate Files Now" buttons. Runs pipeline on a
  background thread; captures ALL print output (including deep
  submodules never rewired to a `log` callback) via
  `contextlib.redirect_stdout` at the call boundary into a polled
  queue — deliberately not threading `log` further down.
- **`run_gui.bat`** launches via `pythonw.exe` (no console window),
  checks the venv exists first with a clear error.
- **`--output-dir`** is the PARENT folder a `<Artist> - <Title>` folder
  gets created under (not the final folder itself); omitted defaults to
  `<input_dir>/Output`. Debug files (`[DEBUG LOG]`, `[PASS1 DEBUG]`)
  write to `<input>/.ultrastar_work`, not the output folder.
  `main.delete_work_files` deletes the ENTIRE work_dir, debug files
  included (changed 2026-08-09, user's explicit decision — previously
  scoped to just `separated/`/`extracted/`).

### Interactive LRCLIB lyrics selection (GUI)

- Manual "Search Lyrics..." button + dialog to pick/pin a candidate
  (`PipelineOptions.pinned_lyrics` always wins over automatic fetch).
- Automatic mid-run ambiguity prompt (opt-in checkbox, single-song mode
  only): pauses the background pipeline thread, opens the same dialog
  on the main thread via `self.after(0, ...)` + `threading.Event`,
  resumes on selection/cancel. Never triggered in batch mode.
- `--lrclib-id <id>` / GUI "LRCLIB ID" field fetches one specific entry
  directly (`/api/get/<id>`), always wins over search/pinning.

## MXL + synced-lyrics as PRIMARY generation path (shipped default,
## `ENABLE_MXL_LRC_PRIMARY = True`)

For songs with a MusicXML score AND matching LRCLIB synced lyrics, skip
audio-only pass 1–4 entirely: MXL supplies pitch/rhythm shape directly,
LRC line timestamps anchor real time, real ASR of our own audio places
words precisely within each line.

**Design** (3rd iteration, what shipped): trust LRC line starts as hard
anchors; place words within a line via real ASR match (order-preserving,
timestamp inside the line's window); non-confident/unmatched words fall
back to interpolation from nearest CONFIDENT neighbors (by MXL offset,
not bounded to one line) using a locally-calibrated real-seconds-per-
quarter-note rate; clamped into the word's own LRC line window as a
backstop. Real result on a validated song: 100% pitch-class, 99% timing
within 500ms, 105ms mean error, using **zero audio pitch detection**.

**Quality gate is downstream, not upfront**: upfront duration+content
filtering on LRCLIB candidates is NOT sufficient — a candidate can pass
generously and still be timed to a *different recording* entirely
(confirmed twice on real songs). Real gate:
`MxlLrcQuality.asr_placement_rate >= MXL_LRC_MIN_ASR_PLACEMENT_RATE`
(0.5) — fraction of MXL words confidently matched against our own
audio's real ASR transcript. A wrong-recording candidate collapses this
on its own. On gate failure: CLI logs a warning and falls through to
standard pass 1–4 pipeline; GUI (single-song only) prompts
Continue/Cancel via the same thread-hop pattern as the lyrics
ambiguity prompt.

**Real production bugs found and fixed** (validate against the actual
written file / real production use, not just in-memory floats):
1. Word duration was always stretched to fill the entire gap to the
   next word (no representation of real pauses) — fixed by deriving end
   time from ASR's own end (trusted match) or MXL-note-value × local
   tempo rate, clamped to never exceed the next word's start but free
   to end earlier.
2. ASR confidence was never checked — a text match with near-zero
   alignment confidence was trusted anyway. Fixed: gate matches on
   `MXL_LRC_MIN_ASR_WORD_CONFIDENCE = 0.3`.
3. Written BPM was too coarse for MXL-derived syllable density — fixed
   via `BPM_WRITE_MULTIPLIER = 2` (general, not scoped to this path).
4. Lyric text always came from the MXL's own OCR'd syllable text, never
   corrected — fixed: `assign_words_to_lines` also returns a clean-text
   replacement from the matched LRC token (exact match, or fuzzy match
   for a 1:1 anchored "replace" slot clearing
   `MXL_LRC_FUZZY_TEXT_MIN_RATIO = 0.6`). A word with no anchor on
   either side is left on its raw OCR text rather than guessing.
5. Fallback duration/position estimate used the WHOLE LRC line window,
   breaking badly when the line had trailing silence — fixed by
   interpolating from nearest CONFIDENT neighbors instead of the whole
   line span.
6. Verification methodology (including the shipped
   `verify_existing_song.py`) silently dropped non-matching words
   instead of counting them as failures, inflating reported accuracy.
   Fixed: added `coverage_fresh`/`coverage_existing` +
   `EXISTING_TXT_MIN_COVERAGE = 0.85` to the PASS gate. **Use
   `verify_existing_song.verify_existing_song` directly for any future
   real-output-vs-ground-truth comparison** — don't write another ad
   hoc script.
7. Untexted MXL continuation notes (tied holds, slurred pitch slides)
   were silently dropped — fixed: a lyric-less note contiguous with the
   in-progress syllable's note extends it (tied+same-pitch) or becomes
   a new empty-text syllable (slurred/different-pitch), handled by
   existing melisma-padding. Gated on exact contiguity to avoid gluing
   unrelated post-rest notes onto the wrong word.
8. ASR word-matching compared against the MXL's raw OCR text instead of
   the already-resolved clean LRC text — fixed by preferring clean text
   in the match, with a fuzzy (not just exact) fallback since WhisperX
   itself is not deterministic in which literal word it transcribes run
   to run (confirmed: "favors" vs. "favorites" for the same audio across
   different runs).
9. The fuzzy-replace fix in (8) only checked a CLEAN 1:1 replace block
   (`(b2-b1)==1`) — but `asr_in_window` is time-bounded, not
   line-bounded, so a word belonging to the NEXT line can spill into the
   same window and turn a real 1:1 mismatch into a 1:N replace block,
   which the exact-size check rejected outright even though the correct
   candidate was sitting right there at the block's own start. Fixed by
   trying every ASR word in the block against the single unmatched MXL
   word and keeping the best fuzzy-ratio match that clears the
   threshold, instead of requiring the block to already be 1:1.
10. **No calibration at all for a systematic time offset between LRC
    line timestamps and our own audio** (e.g. extra lead-in silence in
    our recording vs. whichever recording LRCLIB's synced lyrics were
    timed against) — real confirmed case: "Ordinary Day" (lrclib id
    6210269) has a consistent ~+2.4s offset (recording-to-recording, not
    a bug in our own timing), which blew the quality gate outright (22%
    non-monotonic placements) rather than just being imprecise, since
    every line's ASR search window and interpolation-fallback window
    were off by the same amount. Fixed: `generate_from_mxl_and_lrc` now
    calibrates a global offset(+drift) BEFORE any MXL word is ever
    placed, using `_match_asr_to_lrc_lines` (matches ASR's own flat word
    stream against LRC line text, word-level, order-preserving — same
    "never a text search, always positionally anchored" technique used
    throughout this project) feeding `lrc_timing.two_tier_time_calibration`
    (factored out of `apply_lrc_timing_check` for reuse, not
    reimplemented a third time — see that module for the constant/drift
    two-tier design). A null/near-zero calibration is a no-op, so an
    already-well-aligned candidate (Chicago) is unaffected —
    confirmed: offset calibrates to +0.0s/95% agreement there, same
    114/116 placement as before this change.
11. **MXL OCR errors spanning more than one word were left completely
    unmatched** — `assign_words_to_lines`'s fuzzy-replace handling was
    restricted to a clean 1:1 MXL-word-vs-LRC-word shape, same
    limitation class as (9) but in the OTHER MXL+LRC matching function.
    Real confirmed cases (also "Ordinary Day", inspecting its actual MXL
    OCR quality): `"winnes"` (1 MXL word, OCR-merged) for LRC `"win
    now"` (2 real words); `"stomty"`+`"in"` (2 MXL words) for LRC
    `"stop"`+`"trying,"` (2 words, each individually too garbled to
    clear the 1:1 ratio alone); `"double"`+`"edged"`+`"kide"` (3 MXL
    words) for LRC `"double-edged"`+`"knife,"` (2 words, one
    hyphenated). Fixed: `assign_words_to_lines` now also attempts a
    WHOLE-BLOCK fuzzy match (up to `MXL_LRC_BLOCK_MAX_WORDS = 6` words
    either side) when a replace opcode isn't 1:1 — concatenates both
    sides' characters, checks the SAME `MXL_LRC_FUZZY_TEXT_MIN_RATIO`
    ratio, then distributes the recovered LRC words across the MXL
    block's own word slots via new `_distribute_words_to_slots`
    (splits a hyphenated LRC word into pieces first if that's what's
    needed to match the slot count, otherwise merges/melisma-pads —
    same reconciliation pattern `_text_for_mxl_syllables` already uses
    for syllable counts, just at word level). Still never an
    unconstrained text search — a block's LRC counterpart is always
    fixed by its own real neighboring matches in the ONE whole-sequence
    alignment, so a common word like "oh"/"you" can't accidentally
    match a different occurrence elsewhere in the song.

Both (10) and (11) real-validated together on "Ordinary Day" through the
actual written output file: MXL+LRC primary generation, which previously
failed its own quality gate outright (22% non-monotonic) and fell back
to the standard pipeline, now succeeds — 233/280 words placed via real
ASR transcription (up from a pre-fix 55% ASR match rate that still
wasn't enough to pass), 0 monotonic fixes, and the three inspected OCR
passages all display the correct recovered lyrics in the final file
("win now", "stop"/"trying,", "double"/"edged"/"knife,") instead of the
raw OCR garbage.

**Candidate override**: `LrcLibCandidate.id` + `fetch_lrclib_by_id` +
`--lrclib-id`/GUI field let a user paste a manually-confirmed LRCLIB id,
always wins over search/auto-pick.

## Alignment-only mode (`realign.py`, shipped 2026-08-09)

A new, separate CLI (`python -m ultrastar_generator.realign <folder>
--existing-txt <path>`, own `run()`/`build_arg_parser()`) that takes an
ALREADY-WRITTEN UltraStar `.txt` plus its audio and re-times it: `#GAP`,
note start, and note length only. Never touches pitch, never adds/
removes/reorders a note. Assumes the input's notes are in the right
order and its lyric TEXT is correct; makes NO other assumption about its
timing quality (explicitly designed to survive the degenerate case of a
flat list of equal-length placeholder notes that don't match the audio
at all).

**Design** (deliberately different from `mxl_lrc_generator.py`'s
per-LINE-windowed ASR search, even though the anchor/interpolate shape
is borrowed from it): a WHOLE-SONG, order-preserving text match of the
existing file's own words against real ASR words (`match_words_to_asr`)
— never time-windowed, since this mode can't trust the input's own
timing enough to window a search with it. Confident matches use ASR's
own start/end directly. LRCLIB synced lyrics, when available, seed one
extra anchor at the first not-yet-confident word of each matched line
(`seed_lrc_anchors`, reusing `mxl_lrc_generator.select_lrc_candidate`/
`assign_words_to_lines` as-is via throwaway `MxlWord` wrappers around
the existing words — those two functions only ever read `.norm`).
Everything still unanchored is placed by `interpolate_fallback`: two
confident neighbors → rate-interpolate using the word's OWN original
start purely as a proportional offset (recovers real local tempo
variation if the original timing roughly tracks the audio; degrades to
even spacing if it doesn't); one neighbor → constant shift from it,
not a rate extrapolated from a single point; zero anchors anywhere in
the whole song → keep that word's ORIGINAL timing completely unchanged
— this mode always has a safe fallback, unlike `mxl_lrc_generator`'s
equivalent pass, which has none. Sub-word syllables are redistributed
within a word's new span using the word's OWN original relative
syllable timing (pitch/note count are never touched anywhere in this
module). `match_asr_to_lrc_lines` (the ASR-vs-LRC-line calibration
step) was promoted from `mxl_lrc_generator.py` (private) to
`lrc_timing.py` (public) alongside `two_tier_time_calibration` — a
second module needed the exact same, already-generic function, so it
moved to the shared home rather than being reimplemented.

**Prerequisite bug fix, `usdx_parser.py`**: word-boundary detection
(`is_word_start`) only ever checked for a LEADING space (this project's
own writer convention) — a real SingStar-shipped ground-truth file
(Beauty and the Beast's `notes.txt`) uses a TRAILING space on a word's
LAST syllable instead ("Bare"+"ly " → "Barely") and has NO leading
spaces at all, which the old check would have silently merged into one
bogus word per line (already flagged as a known gap in project memory,
never fixed since nothing exercised it before this feature). Fixed by
also checking the PREVIOUS syllable's own trailing whitespace — a
strict superset of the old check, so this project's own leading-space-
only output parses identically to before.

**Real validation (Beauty and the Beast)**: no existing-file ground
truth was available in the usual sense, so the SingStar `notes.txt`
(hand-verified-accurate, see `sandbox/Beauty And The Beast - Beauty And
The Beast/notes.txt`) was converted to a standard-convention `.txt`
(pitch −60, `sandbox/realign_test/BATB/`) and used AS the "already
correct" input, per the user's explicit ask to validate that realigning
an already-good file doesn't meaningfully perturb it. Real result: 140/
140 syllables preserved (no notes added/removed), 0 pitch mismatches,
BPM unchanged, 108/113 words matched directly to real ASR, 1 seeded
from an LRC line anchor, 4 interpolated, 0 kept-original. Mean start
delta from the original +43ms, median +74ms. One real outlier: "Tune"
(bridge section) landed 2.6s earlier than the original — WhisperX's own
forced-alignment gave that word an implausibly early start crossing
what should be a real musical rest, the same class of ASR word-boundary
imprecision already documented under "Lessons learned" above, not a
bug in this module's own logic; not chased further per this project's
own precedent (`--verify-placement`/`--zone-boundary-snap`: individually
well-motivated fixes for exactly this kind of case were both tried and
found to be NET REGRESSIONS end-to-end). Confirmed BATB has no valid
LRCLIB candidate via the dedicated MXL+LRC quality gate (see
`project_validation_song_roster` memory), yet this mode's own (more
permissive, since LRC is only ever a small supplementary anchor here,
not the primary placement signal the way it is in `mxl_lrc_generator`)
candidate search found a same-musical, different-cast "(Finale)"
recording with a weak (54%) time calibration — it seeded exactly one
word, which landed within 600ms of the correct position. Left as-is
for v1 (the risk surface is inherently small: only words with no
direct ASR match ever reach this path at all) rather than adding an
unvalidated extra confidence gate on top.

**GUI integration (same day)**: `run_realign_pipeline`/`RealignPipelineOptions`
factored out of `realign.py`'s CLI `run()` (same shape as `main.
run_pipeline`/`config.PipelineOptions`) so gui.py calls the exact same
pipeline code the CLI does. New "Realign existing file" radio mode in
gui.py, alongside single/batch/YouTube: its own "Existing .txt file"
field (with Browse), a "Realign options" frame (Whisper model, "Use LRC
synced lyrics" checkbox, LRC mode dropdown, LRCLIB ID -- reusing the
Artist/Title fields and the existing threaded-run/log-draining/
Open-Output-Folder machinery unchanged). The normal pipeline's own
Lyrics/Options/Advanced frames (fetch-lyrics, verify-words, MusicXML,
pitch source, etc. -- none of which apply once pass 1-4 is skipped
entirely) are hidden as whole frames rather than picked apart
widget-by-widget; Output folder is hidden too (realign writes directly
next to the existing file by default, not into a fresh `<Artist> -
<Title>` folder the way the normal pipeline does). Verified two ways:
(1) programmatic mode-switch round-trip through all 4 modes with no Tk
packing errors, (2) a full live run (real Demucs+WhisperX, GUI's own
background thread + queue-drain loop, no mainloop() needed --
`app.update()` polled in a loop instead) against the BATB test folder,
producing the same result as the equivalent CLI invocation.

**Not yet done**: no `test_dry_run.py`-adjacent real-audio smoke test
beyond the manual BATB/Stars/Chicago runs in this file -- rerun manually
if `realign.py`'s core matching/interpolation logic changes.

### `lrc_mode="windowed"` prototype + real 3-song comparison (2026-08-09)

User's hypothesis (from prior sessions' MXL+LRC work): LRC-first,
ASR-second placement beats ASR-first, "so long as we can trust the LRC
timing matches the audio." Built a second strategy,
`match_words_to_asr_windowed` (mirrors `mxl_lrc_generator.
place_words_via_asr`'s per-LINE real-time window exactly, adapted to
match against the existing file's own already-trusted text instead of
MXL's OCR'd text), selectable via `realign_song(lrc_mode=...)`: `"seed"`
(shipped default, unchanged -- whole-song ASR primary, LRC only seeds
residual gaps) vs `"windowed"` (LRC lines primary, ASR only resolves
position within a line). `prepare_lrc` factored out of `seed_lrc_anchors`
so both strategies share one candidate-selection/calibration call
instead of two.

**Real comparison, one ASR transcription per song reused for both modes**
(avoids WhisperX's own non-determinism confounding the A/B — see
"Lessons learned"), timing accuracy measured against each song's own
trusted/reference existing file:

| song | LRC calibration | seed (within 100ms) | windowed (within 100ms) |
|---|---|---|---|
| BATB (auto-picked, wrong-cast "(Finale)" candidate) | constant, 54% agreement | 63% | **84%** |
| Chicago (auto-picked candidate, id 34321033) | FAILED (no offset found) | 65% | **41%** |
| Chicago (pinned id 37066985, the one validated in [[project-mxl-lrc-ordinary-day-fixes]]'s session) | constant, 95% agreement | 67% | 66% |
| Stars (auto-picked candidate, id 29680748) | drift, 100% agreement | 55% | 53% |

Zero pitch/text/note-count differences in any run -- only timing varied.

**Root cause of the Chicago regression, and the fix**: windowing an
UNCALIBRATED LRC candidate is actively harmful -- EVERY word's match
routes through the same untrusted signal, so an uncorrected drift (see
`lrc_timing.py`'s own module docstring: real per-song drift, not just
noise around a constant, is a confirmed real phenomenon) corrupts
placement across the whole back half of the song (deltas grew smoothly
from near-zero to -2.6s as the song progressed -- a drift signature, not
random noise). `"seed"` mode never had this failure mode because LRC
there only ever touches a handful of residual words ASR alone couldn't
place, so a bad candidate's blast radius was already small by
construction -- `"windowed"`'s blast radius is not, since it decides
EVERY word's search window. **Fixed**: `"windowed"` now additionally
requires `LrcPrep.calibration_offset is not None` (i.e. `two_tier_time_
calibration` actually found a confident constant-offset OR drift fit)
before it's used at all; otherwise it transparently falls back to
whole-song ASR matching (identical to `"seed"`'s own fallback). Re-ran
Chicago's auto-picked candidate after the fix: `"windowed"` now produces
BYTE-IDENTICAL output to `"seed"` (65% within 100ms both), confirming the
gate closes the regression without needing a pinned/forced candidate.
BATB's win (63% -> 84%) is unaffected by the gate (its calibration
already clears the bar at 54%).

**Interpretation**: with the gate in place, `"windowed"` was never worse
than `"seed"` in any of the 4 real test cases, and was a large win in
exactly the case the user's hypothesis predicts (BATB: LRC timing IS
trustworthy -- confidently calibrated, even at just 54% agreement --
despite being from a different-cast recording of the same musical).
Where ASR was already excellent on its own (Chicago pinned, Stars: both
>=90% direct ASR match rate even in `"seed"` mode), windowing made
little difference either way -- its real value shows up specifically
when ASR's OWN forced-alignment can be confidently wrong on an
individual word (BATB's "Tune" case from the earlier validation run)
and a trustworthy LRC window prevents committing to that wrong answer.

**Status (updated same day, user's explicit decision)**: `lrc_mode`
default flipped from `"seed"` to `"windowed"` -- since the gate makes
`"windowed"` an auto-select on its own (windowing when calibration is
confident, transparently identical to `"seed"`'s own behavior
otherwise), this needed no new mode value, just changing which string
is the default in `realign_song`'s signature and `--lrc-mode`'s CLI
default. `"seed"` (always whole-song-ASR-primary, even with a
confidently-calibrated candidate available) stays available as an
explicit opt-out / for future A-B comparisons, but isn't needed for
normal use anymore.

## Environment

- Windows, venv at `E:\Projects\ultrastar_generator\venv`.
- CUDA required — pipeline aborts at startup if unavailable. WhisperX
  pulls in pyannote/torch; expect noisy-but-harmless startup warnings.
- Demucs stems cached in `sandbox/.ultrastar_work/` — safe to delete
  between runs; separation is skipped if `vocals.wav` already exists.

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

### Real bug: the "favorites" fuzzy fix didn't fire when a next-line word
### spilled into the same ASR window (2026-08-09)

User reported the "There's a lot of favors" region was *still* wrong
after the fuzzy-match fix above, from a real GUI run (folder input +
`--lrclib-id 37066985`, no other options changed). Located the user's own
run on disk (`C:\Users\Dan\Desktop\Chicago - When You're Good to Mama`,
NOT the sandbox copy — its cached Demucs separation is a different, also
non-deterministic run, `#BPM:369.14` vs. the sandbox's `356.42`) and read
its real `[DEBUG LOG].txt` directly rather than guessing: WhisperX
transcribed the word as `'favorites'` again this run, with GOOD
confidence and timing (`71.143-71.804 score=0.895`) — so the fuzzy fix
from the previous entry *should* have caught it, and initially appeared
not to (output showed a ~1.85s fallback-interpolated span starting
~1.7s later than the real ASR timestamp).

Root-caused by replaying the exact real data through
`difflib.SequenceMatcher`: the per-line ASR candidate window
(`asr_in_window = [w for w in asr_words if t0-0.5 <= w.start <= t1+0.5]`)
is time-bounded, not restricted to this line's own words — "I'm" (the
very next MXL word, start of the NEXT line) starts only ~1.1s after
"favorites" ends, well inside the `+0.5s` slop, so it rode along into the
same window. That turned the alignment into a 1-MXL-word-vs-2-ASR-words
replace block (`['favors'] vs ['favorites', 'im']`) instead of the clean
1:1 block the previous fix's `(b2 - b1) == 1` condition required —
rejected outright even though `asr_in_window[b1]` (the block's own first
word) was "favorites", the correct answer, sitting right there.

Fixed: relaxed the condition to `(a2 - a1) == 1` only (still restricted
to a single unresolved MXL word — never guesses across multiple ambiguous
MXL words at once, same conservatism as before), and instead of assuming
the block's one ASR word is the candidate, tries every ASR word in the
block `[b1:b2)` and keeps whichever clears
`MXL_LRC_FUZZY_TEXT_MIN_RATIO` with the highest ratio. Verified directly
against the real ASR data reconstructed from the user's own failing run's
debug log (not synthetic-only): `place_words_via_asr` now places the word
at `71.143-71.804`, matching real ASR exactly. New regression test added
(`test_dry_run.py`) reproducing the exact spillover shape (a next-line
word riding into the same time window). `test_dry_run.py` full suite
green throughout.

(A `test_dry_run.py` failure hit while validating this turned out to be
a stale test, not a bug: the user had intentionally changed
`main.delete_work_files` the same day to delete the ENTIRE work_dir,
debug files included, rather than just `separated/`+`extracted/` — see
Shipped Defaults above. Test updated to match.)