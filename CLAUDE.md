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
  don't flip on without addressing this. **Re-validated 2026-08-10**
  (user's explicit ask: "is it still worth keeping around?") after this
  session's other changes (rewindow, ASR-quality retry, force-align-gaps,
  melisma-tail merge) — regression confirmed to still hold. Controlled
  BATB run (same song, same ground truth, only this flag differs): words
  matched 105→101, timing agreement 97.1%→93.1%, coverage 78%→75%,
  timing mismatches 3→7. Root cause identified this time (not just
  "regresses," but why): the "is the expected word's text present
  anywhere in this search window" check has no way to disambiguate WHICH
  occurrence of a REPEATED word is correct — for words that recur often
  (Gold's "do" in its 4x-repeated chorus, BATB's "as"/"the"/"the"), the
  window can contain more than one real instance, and forced-alignment
  then confidently returns whichever one it finds, sometimes 6-10s away
  (near `PLACEMENT_SEARCH_MAX_RADIUS_SEC`) from the correct one — the
  same repeated-phrase disambiguation failure class already documented
  for `realign.py`'s `"windowed"` mode (Heroes/Americans/Ordinary Day,
  see below). Confirms this isn't fixable by more tuning without adding
  occurrence-disambiguation, same conclusion as `--zone-boundary-snap`.
  **Fully removed from the codebase 2026-08-10** (user's explicit
  request: "Completely remove verify-placement" — different treatment
  from `--zone-boundary-snap` below, which stays as dead-but-present
  code) — `verification.verify_placement`, `PlacementCorrection`/
  `PlacementWarning`, the CLI flags, the GUI checkbox, and its tests are
  all gone; `alignment.py`/`lyric_alignment.py`'s `placement_corrections`/
  `placement_warnings` plumbing removed too.
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

### Real bug: `lrc_mode="seed"` also needed the calibration gate, not just
### `"windowed"` (David Bowie - "Heroes", 2026-08-09)

User reported a real, reproducible failure: in the shipped file, "Just
for one day" (beats 4185-4279) is correct; in the realigned output, that
passage landed in the middle of a big silent stretch. Root-caused by
replaying the real run's own data (not guessed): the auto-picked LRCLIB
candidate for this song was **"Heroes" by "Kolacny Brothers"** -- a
CHORAL COVER, not Bowie's own recording (same wrong-recording failure
class as BATB's "(Finale)" cast recording and Chicago's auto-picked
candidate, see the `lrc_mode="windowed"` section above) -- confirmed via
`calibration_confidence=0.0`, a real "zero agreement" case, not merely
"not enough samples." Because this candidate's calibration failed,
`"windowed"` mode correctly declined it and fell back to `"seed"` -- but
`"seed"`'s own LRC-anchor-seeding step (`seed_from_prep`) had NO
calibration gate at all, on the (now-disproven) theory that seeding only
a handful of words keeps its "blast radius" small by construction. It
seeded word "Then" at 176.98s from the uncalibrated candidate's raw line
timestamp -- a time LATER than several of ITS OWN neighbors' already-
independently-correct ASR matches (170-172s) AND later than "Just for
one day"'s own correct ASR matches (175.0-176.2s). `interpolate_fallback`'s
monotonic clamp (forward-push only, no notion that an earlier anchor
might itself be wrong) then dragged ALL of those already-correct
neighbors forward to 176.98s to preserve ordering -- corrupting real,
independently-confirmed-correct matches, not just filling a genuine gap.
This is why the blast-radius theory was wrong: a single bad anchor's
effective blast radius is unbounded once the monotonic clamp propagates
it forward to the next real anchor, regardless of how few words were
directly seeded.

**Fixed**: both `"windowed"` and `"seed"` now use the exact same gate
(`lrc_prep.calibration_offset is not None`) before trusting an LRC
candidate for ANYTHING, seeding included -- an uncalibrated candidate now
falls all the way back to pure ASR + `interpolate_fallback`, same as "no
candidate found" would. Real-validated: re-ran David Bowie - Heroes,
`n_lrc_seeded` dropped from 5 to 0, "Just for one day" now lands within
0.2-0.5s of the original (previously off by 1-6.7s across that whole
passage, with word "Then" off by +6.7s). Regression-checked BATB
afterward (unaffected -- its own candidate's calibration still clears
the bar at 54% confidence, same as before this fix). New synthetic
regression test added (`test_dry_run.py`) reproducing the exact failure
shape (a seeded anchor landing later than already-confident neighbors)
without needing real audio.

### Real bug #2: `interpolate_fallback`'s monotonic clamp could overwrite
### an already-CONFIDENT match too, not just fallback ones (David Bowie -
### "I'm Afraid of Americans", 2026-08-09)

Second real user report, same day: a passage repeating "I'm afraid of
Americans" / "I'm afraid of the world" / "I'm afraid I can't help it" /
"I'm afraid I can't" (this song's chorus repeats this whole 4-line block
roughly a dozen times, "afraid" alone appears 31 times) ended up in the
output as 16 consecutive words each exactly 1 beat long, essentially
frozen at one instant -- but with CORRECT pitches, confirming the note
data itself was untouched and this was a pure timing bug.

Root-caused via the real run's own data (`match_words_to_asr_windowed`'s
raw per-word matches, captured BEFORE `interpolate_fallback` ran): this
song's picked LRCLIB candidate is genuinely David Bowie's own real
recording (unlike Heroes' wrong-artist case) and its calibration cleared
the confidence gate (48%, `kind="constant"`) -- so `"windowed"` mode was
used, correctly per the fix above. But the per-LINE window computation
for FOUR consecutive LRC lines in this repeated block searched the WRONG
occurrence of a repeated phrase: their real matches landed ~12s EARLIER
than they should have (confirmed: raw matched ASR timestamps for these
lines were nearly identical to a DIFFERENT, earlier real occurrence of
the same repeated text -- a real repeated-phrase disambiguation failure
the GLOBAL calibration confidence check can't see, since it only
measures overall agreement, not per-line reliability in one specific
repetitive stretch). This alone would have been a real but bounded
~12s-early error for that block -- what made it catastrophic was a
SEPARATE bug: the fallback word immediately before this mis-matched
block computed a NEGATIVE interpolation rate (because its two "confident"
anchors were themselves chronologically inverted -- the later-in-
sequence anchor's real match, from the mis-aimed window, was chrono-
logically EARLIER than the earlier anchor's), triggering
`interpolate_fallback`'s degenerate fallback formula (a constant shift
from the earlier anchor ALONE, ignoring the later one entirely), which
overshot PAST all 14 of the following genuinely-confident matches. The
OLD monotonic clamp (forward-push, no notion of `confident` at all) then
flattened every one of those 14 correct matches up to that one wrong,
overshot value -- destroying real, independently-correct information,
not just filling a gap.

**Fixed**: `interpolate_fallback`'s trailing monotonic-fixup is now
confident-aware -- a confident word's own value is a FIXED POINT, never
rewritten by either direction of the clamp; only fallback words are
adjusted (pass 1, forward: push a fallback word up past a larger
preceding value; pass 2, backward: pull a fallback word back down if it
exceeds the next confident word's own value). Real-validated: re-running
with the fix, the 14-word flatten is gone -- each word now gets its own
distinct (still ~12s early, since the underlying mis-aimed window isn't
fixed) value instead of one shared wrong point. New synthetic regression
test added reproducing the exact negative-rate-inversion shape without
needing real audio.

**Open question the underlying mis-aimed-window problem exposes**: ran
`"seed"` mode (whole-song ASR matching, immune to per-line windowing
since it doesn't depend on LRC line correspondence for most of the song)
on this SAME song for comparison -- it placed this entire passage within
tens of milliseconds of the original throughout, dramatically better
than `"windowed"` even after the clamp fix (which still leaves the whole
block ~12s early, just no longer flattened to one point). This is a
SECOND real case (after Heroes) where `"windowed"` has a real, distinct
failure mode `"seed"` doesn't share -- here, a highly-repetitive song
defeats per-line LRC correspondence even with a correctly-calibrated,
correctly-attributed candidate, which the global calibration-confidence
gate cannot detect (it only measures aggregate agreement, not per-line
reliability within one repetitive stretch). Combined with Heroes
(wrong-recording candidate), `"windowed"` now has two independent
confirmed real failure classes beyond what the original 3-song
comparison (BATB/Stars/Chicago) exercised, while `"seed"` has been
robust or better in every real case tried so far. **Not yet acted on** --
surfaced to the user as an open question (whether `"windowed"` should
remain the default) rather than unilaterally flipping it again without
their input, since this is a real tradeoff (windowed's upside on BATB
was real and validated) with limited data on each side.

**Resolution (same day)**: extended the comparison to Stars/Chicago/
Ordinary Day with both fixes active. Ordinary Day -- ALSO a repeat-heavy
song (a 4x-repeated chorus) -- turned out to be the mirror image of
Americans: there `"seed"` mode is the one that catastrophically fails (a
44-SECOND misalignment on the repeated chorus, 46% within 100ms, mean
error 5.4s) while `"windowed"` handles it cleanly (71%, mean error
0.15s) -- `"seed"`'s own whole-song matching has the exact same class of
repeated-phrase vulnerability as `"windowed"`'s per-line matching, just
triggered differently (no real-time window to bound the search, so a
long enough repeated stretch can make the whole-sequence diff jump back
to an earlier occurrence). Net across 5 real songs: `"windowed"` wins
clearly on 2 (BATB, Ordinary Day), ties on 2 (Chicago, Stars), loses on
1 (Americans) -- **`"windowed"` stays the default**, this was not a close
call once Ordinary Day was added to the comparison.

Investigated Americans further per the user's own suspicion ("the LRC
data is for the album version but I have the audio from the music
video") -- confirmed directly: our file's own `afraid` count is 31; the
auto-picked LRCLIB candidate's (id 34342583, from the *Brilliant
Adventure* box set) is 40 -- a real, structural arrangement difference
(9 extra chorus repeats), not just timing noise. Candidate DURATION
alone doesn't catch this (267.0s our audio vs 270.0s declared -- well
within the existing 15s selection tolerance, coincidentally close
despite the real structural difference). Led directly to the repeat-
structure check below.

### Repeat-structure consistency check for LRC candidates (2026-08-09)

New `check_repeat_structure` (in `realign.py`) rejects an LRC candidate
whose repeat structure doesn't match ours, wired into `prepare_lrc` via
an optional `our_lines` (the existing file's own lines, reconstructed by
new `_reconstruct_our_lines` from its `-` LineBreak markers) + `log`
param -- both optional so existing callers/tests that don't have the
original entries handy (e.g. `seed_lrc_anchors`'s standalone/test path)
are unaffected.

**Design note, found the hard way**: the first version compared only the
SINGLE most-repeated exact-duplicate LINE (10x ours vs 12x the
candidate's, within tolerance -- would NOT have caught the real bug).
A real chorus is often split across several near-duplicate variants
("I'm afraid of Americans"/"...of the world"/"...I can't help it"/"...I
can't"), each individually landing within tolerance on its own even
though the true repeat count is split across all four. Fixed by
comparing WORD occurrences instead: find the most-repeated line's own
content words (filtering out short/filler words under 4 chars, since a
common word repeating a lot doesn't indicate a repeated CHORUS the way a
shared distinctive word across every variant does), pick whichever
qualifying word has the highest count in OUR OWN file (confirmed this
selects "afraid" -- present in all four variants -- over a less-complete
signal like "Americans", present in only one), and compare that word's
total count on both sides. Tolerance +-15%/minimum +-1 (absorbs ordinary
per-song noise like an intro/outro repeat some editions add or drop).

Real-validated against all 5 songs already in the comparison set:
correctly rejects Americans (31 vs 40, un-blocked "windowed" falls back
to whole-song ASR matching, which independently is known to place this
whole passage well) while leaving BATB/Chicago/Stars/Ordinary Day's own
already-validated candidates completely unaffected (all pass the check).
Confirmed end-to-end through the real written output, not just the
isolated check function.

### GUI form split into Lyrics/Options + `--delete-work-files` (2026-08-09)

Realign's GUI form restructured to match the normal pipeline's own
Lyrics/Options split (user's explicit request), rather than one combined
"Realign options" frame: `self.realign_lyrics_frame` (LRCLIB ID only, for
now -- no Search/pin/ambiguity-prompt support in realign.py yet) +
`self.realign_options_frame` (Whisper model, Use LRC synced lyrics, LRC
mode, and a new "Delete work files after realigning" checkbox). Both
toggle together as a pair in `_on_mode_change` exactly where the single
`realign_frame` used to.

`--delete-work-files` support added to `realign.py` itself for this (it
didn't exist there at all before): `RealignPipelineOptions.
delete_work_files`, and `run_realign_pipeline` split into a thin wrapper
+ `_run_realign_pipeline_body`, mirroring `main.run_pipeline`'s own
wrapper/`finally` shape exactly (deletion happens regardless of which
early-return failure path the body took). `run_realign_batch` gets this
for free -- it already calls the wrapper per subfolder. CLI gained a
matching `--delete-work-files` flag.

Real-validated end-to-end: ran with `--skip-separation --vocals-path
<a DIFFERENT folder's cached vocals> --delete-work-files` against a
throwaway folder -- output file written correctly, that folder's OWN
`.ultrastar_work` was deleted afterward, and the separately-referenced
cache folder (BATB) was confirmed untouched.

### Batch support + Mode/Batch UI restructure (2026-08-09)

Added batch mode to `realign.py`: `find_existing_txt_in_folder` auto-
detects the single .txt to realign within a folder (fails closed --
`AmbiguousExistingTxtError` -- on zero or multiple candidates; excludes
this module's own `"[REALIGNED]"` naming convention so re-running batch
on an already-realigned folder doesn't see two candidates, or worse pick
the REALIGNED file as the next run's INPUT and compound drift).
`run_realign_pipeline`'s `existing_txt_path` is now `Optional` (auto-
detects when `None`); new `run_realign_batch` mirrors `batch.run_batch`'s
shape but needs no `output_parent_dir` at all, since every result is
always written next to ITS OWN subfolder's existing file (no mirroring
concept the way the normal pipeline's batch has). CLI gained `--batch`
with the same incompatibility-checking convention as `main.py`'s
(`--existing-txt`/`--audio-file`/`--work-dir`/`--artist`/`--title`/
`--lrclib-id` all rejected together with `--batch` -- none make sense as
a single override across multiple songs). Also fixed the same latent gap
in `main.py`'s own `--batch` validation, which was missing `--audio-file`
from that list. Real end-to-end validation: batch-ran all 3 of
`sandbox/realign_test/{BATB,Chicago,Stars}` in one CLI invocation --
correctly auto-detected each subfolder's own file (BATB's folder already
had a leftover `[REALIGNED].txt` from earlier testing, correctly
excluded), 3/3 succeeded, zero writes to any original input file
(confirmed via mtime).

**GUI restructure (user's explicit request)**: "Single song folder" mode
renamed to "Generate song file"; Batch changed from a 4th mutually-
exclusive radio option to a `Checkbutton` orthogonal to mode (usable with
both "Generate song file" and "Realign existing file", disabled --
`state=DISABLED`, and its own checked state ignored regardless -- for
"YouTube URL", since a single URL can't populate multiple subfolders).
`self.mode` narrowed to `"generate"|"youtube"|"realign"`; a new
`_is_batch()` helper (`self.batch_mode.get() and mode != "youtube"`) is
the single source of truth both `_on_mode_change` (UI) and `_build_opts`/
`_build_realign_opts` (options) consult, so they can never disagree about
whether batch is actually active.

**Explicit new requirement**: toggling Batch must DISABLE the audio-file
field and its related elements (label/entry/browse button), not hide
them via `grid_remove()` the old batch-as-a-mode code did -- same
treatment now extended, for consistency and to close a real pre-existing
gap, to Artist/Title (a single override across multiple batch songs was
ALREADY silently applied identically to every subfolder before this
session, an existing bug -- the GUI calls `run_batch` directly, bypassing
`main.run()`'s own CLI-level incompatibility checks entirely) and the
LRCLIB ID field (both the normal-pipeline and realign-mode copies).
Existing .txt (realign mode) gets the same disable-not-hide treatment.
Fields keep whatever text the user already typed while disabled --
confirmed the underlying `StringVar` value survives a
disable-then-re-enable cycle intact, only OPTS BUILDING ignores it
(forces `None`) while batch is checked.

Verified via a live (real Tk root, `update_idletasks()`, no mainloop)
mode x batch matrix covering all 3 modes -- correct enable/disable state
and correct forced-`None` opts in every cell, plus the YouTube-disables-
the-checkbox-and-ignores-its-state case specifically.

**Auto-detect extended to single-song mode too (same day, user request)**:
`find_existing_txt_in_folder` (originally batch-only) now also backs
single-song mode -- CLI `--existing-txt` is optional (already was);
gui.py's "Existing .txt file" field became a `PlaceholderEntry`
("(auto-detected from input folder)"), and `_on_run` no longer requires
it. Also extended the auto-detect logic itself: when a folder has
MULTIPLE real `.txt` candidates (common in single-song mode, where a
folder often has other stray `.txt` files a pure batch subfolder
wouldn't), try exactly one further disambiguation -- a file named
`"<folder name>.txt"` (case-insensitive, same basename-matching
convention `file_discovery.py` already uses for cover/background) -- and
only trust it if it narrows the field to EXACTLY one match; still fails
closed (`AmbiguousExistingTxtError`) otherwise, never guessing further.

**Found and fixed a real, pre-existing, unrelated GUI bug while wiring
the new field**: `PlaceholderEntry.effective_value()` incorrectly kept
returning `None` after a Browse-dialog selection, because `_browse_output`
set the underlying `StringVar` directly, which never clears
`is_placeholder` (only `<FocusIn>` does, and a Browse button click never
focuses the entry itself) -- confirmed via a direct repro BEFORE writing
any fix, not assumed. This silently discarded a user's picked output
folder in real usage any time they used Browse without first clicking
into the field. Fixed at the root: new `PlaceholderEntry.set_real_value()`
clears `is_placeholder` correctly; `_browse_output` and the new
`_browse_existing_txt` both updated to use it instead of `.set()` on the
var directly.

Real validation: single-song CLI mode with no `--existing-txt` at all
against `sandbox/realign_test/BATB` (which has both the original file AND
a leftover `[REALIGNED].txt` from earlier testing) correctly auto-detected
and used the ORIGINAL file only.

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

### `"validate"` strategy prototype + local-rematch second pass -- real
### comparison, local-rematch REJECTED as a net regression (2026-08-09)

**"validate" strategy** (`realign_song_validate`, `--strategy validate`):
trusts a word's own (GAP-corrected) original position over ASR's own
value whenever ASR confirms it's close, rather than always replacing with
ASR's own less-precise timing. Built and real-validated (100% exact
recovery on "I'm Afraid Of Americans", which happened to have a near-
perfect original file needing only a GAP fix). **User's explicit
correction after seeing this result**: `"validate"` is not a generally
useful strategy -- it only shines when the input file is ALREADY
accurate, which defeats the purpose for the actual problem (files whose
lyrics/timing genuinely don't match the audio well). Kept in the codebase
as a prototype (`--strategy validate`, not GUI-exposed) but not pursued
further as the general solution.

**Real bug motivating the next investigation**: user reported "Johnny
wants a brain, Johnny wants to suck on a coke" (Americans) landing
compressed/wrong in a real failing run, even under the shipped `"replace"`
strategy -- the section immediately before and after recovers fine, only
this one block (with several nearby "Johnny wants X" near-repeats, plus
an ASR-unfriendly "Ah-ah-ah" ad-lib line right before it) goes wrong.
Root cause (confirmed via real, same-transcription diagnostics): a
repeated phrase ELSEWHERE in the song can steal `match_words_to_asr`'s
whole-song text match, leaving the real local occurrence completely
unmatched -- it then falls straight to `interpolate_fallback`'s
proportional guess, which has no way to know a real silence/pause exists
inside that gap, and compresses the words together.

**Fix attempt: `rematch_local_gaps`** (new function, `config.
REALIGN_LOCAL_REMATCH_SLACK_SEC`) -- a second pass that retries any
still-unmatched run of words against ONLY the ASR words whose own
timestamp falls between that run's nearest confident neighbors (+/- 1s
slack), instead of the whole song -- much less ambiguous, so a repeat
elsewhere can no longer steal the match. Synthetic regression test added
(`test_dry_run.py`) confirming it recovers a same-text-decoy scenario
correctly.

**Real, CONTROLLED (single transcription, A/B in one process -- avoids
WhisperX's own non-determinism confounding the comparison) validation
against all 4 reference songs found this is a NET REGRESSION**:

| song | within 100ms, WITHOUT local-rematch | WITH local-rematch | delta |
|---|---|---|---|
| BATB | 80.5% | 56.6% | **-23.9pp** |
| Chicago | 66.7% | 66.7% | 0 (0 words touched) |
| Stars | 63.1% | 64.1% | +1.0pp |
| OrdinaryDay | 70.5% | 68.7% | -1.8pp |

**Why it fails to generalize** (same shape as `--verify-placement`/
`--zone-boundary-snap` in the main pipeline's own "Removed / rejected"
history -- a well-motivated mechanism that doesn't survive real data):
the 4 reference songs are all cases where the EXISTING file's own local
timing is already trustworthy (that's why they're used as references) --
so when ASR is sparse/ambiguous in some region,
`interpolate_fallback`'s original-timing-proportional guess is regularly
BETTER than forcing a local ASR rematch, which can just as easily lock
onto the wrong nearby repeat within its own (still non-zero-width)
window as the whole-song pass did. It only actually helps in the
opposite case -- the original file's local timing is ALSO wrong there
(true for the Americans case that motivated it) -- and there's no cheap
signal available to tell those two situations apart before deciding
whether to trust a local rematch over interpolation.

**Decision: shipped OFF by default** (`realign_song(use_local_rematch=
False)`, not wired into CLI/GUI). Code and its synthetic test kept in
place since the underlying mechanism is real value for the case it was
built for -- just needs a way to gate it (e.g. distinguishing "original
file wrong here" from "ASR just sparse here") before it's safe to enable
generally. Don't flip this on without addressing that gating question.

**Separately found, NOT fixed (needs its own investigation)**: Americans'
own outro ("God is an American" x8, tight ~6s spacing) shows the whole-
song matcher's repeat-occurrence-count mismatch drifting the matched
position by increasing multiples of the repeat period (+6s, then +18s,
then stabilizing at +30s) as the song progresses -- a different failure
shape than the "Johnny" case (there, ALL words in the region go
unmatched; here, they get CONFIDENTLY matched, just to the wrong
occurrence, so neither `interpolate_fallback` nor `rematch_local_gaps`
can help -- the whole-song text-only match has no time information to
break the tie with). Pre-existing, not introduced by local-rematch (
confirmed via the same controlled single-transcription A/B). Left as a
known open issue.

### realign debug log (2026-08-09)

User noticed realign never wrote a debug file (unlike the main
pipeline's `[DEBUG LOG].txt`), which would have made this session's
whole-song-vs-local-rematch investigation trivial instead of needing
several throwaway scratchpad scripts. Fixed: `realign_song` now accepts
an optional `debug_log: DebugLog` (same class the main pipeline uses),
writing a per-word trace (text, orig start/end, WHICH mechanism placed
it -- asr/lrc/local_rematch/interpolated/kept_original -- final
start/end, delta) plus the existing summary stats.
`_run_realign_pipeline_body` wires this up the same way `main.py` does:
`<Artist> - <Title> [DEBUG LOG].txt` under the work dir, ON by default,
`--no-debug-log`/`RealignPipelineOptions.no_debug_log` to skip -- no GUI
checkbox, matching the main pipeline's own convention.

**Follow-up same day**: also added a "RAW ASR TRANSCRIPT" section (every
`Word` actually fed into matching, with confidence) and wired
`transcribe_words(..., debug_log=debug_log)` through from realign (it
existed in `transcription.py` already but was never actually connected
from realign's call site -- a real, silent gap). This is what made the
hallucination root-cause below findable at all: a user-pasted "per-word
trace" section was initially mistaken for raw ASR data, leading nowhere,
until the actual raw-ASR section was added and read -- see
[[feedback-verify-data-labels-before-concluding]] (auto-memory). Also
added a "RAW WHISPER DECODER SEGMENTS" section (the decoder's OWN
segment-level output, BEFORE wav2vec2 forced alignment) -- this is what
let the hallucination be pinned to the DECODER stage specifically, not
the alignment stage (see below). `realign_song_validate` now also
accepts `debug_log` (parity with `realign_song`), same 3 sections.

### `"validate"` strategy properly wired up (GUI + parity), still OFF by
### default (2026-08-09)

User: "properly implement the validate flow we did earlier. Leave it off
by default for now, just so we can work on improving the other methods."
`realign_song_validate` was previously reachable only via `--strategy
validate` on the CLI, explicitly "not GUI-exposed yet". Added a
"Strategy" combobox (`replace`/`validate`) to the GUI's realign Options
frame (same frame as Whisper model/Use LRC/LRC mode/delete-work-files),
wired through `_build_realign_opts` -> `RealignPipelineOptions.strategy`
-> `_run_realign_pipeline_body`'s existing branch. Default stays
`"replace"` in both the GUI (`self.realign_strategy = tk.StringVar(value=
"replace")`) and the CLI (`--strategy` default unchanged) -- this is
purely making an already-real-validated option properly selectable, not
changing what runs by default. Verified via a GUI smoke test (mode-switch
matrix + Strategy round-trip through `_build_realign_opts`) and a real
end-to-end run confirming all 3 debug-log sections (summary/per-word
trace/raw ASR) appear correctly under `strategy=validate` too.

### Real bug root-caused to WhisperX DECODER hallucination, not our own
### matching logic (David Bowie - "I'm Afraid Of Americans", 2026-08-09)

User reported (again) that "Johnny wants a brain, Johnny wants to suck on
a coke" lands badly wrong via the GUI's realign mode, real audio position
confirmed by ear as ~81-84s (matching the EXISTING FILE's own original
timing almost exactly). Initial hypothesis (repeated-phrase text stealing
the whole-song match, same class as the "I'm afraid" collapse) did NOT
reproduce across 2 of my own fresh runs -- both landed this passage
correctly. **User caught a real mistake**: the debug data I was
diagnosing from (their earlier pasted lines) was the PER-WORD TRACE
section, not raw ASR -- I hadn't built the raw-ASR dump yet at that
point, so I was drawing conclusions from the wrong data entirely (see
[[feedback-verify-data-labels-before-concluding]]).

With the new RAW ASR TRANSCRIPT + RAW WHISPER DECODER SEGMENTS sections
in place, user re-ran via the GUI (fresh work dir) and the real failing
run's data showed the true mechanism: the DECODER's own segment
`66.856 - 85.975` (a single ~19s chunk) transcribed as just `"Johnny
wants a brain, Johnny wants to suck on the conch."` -- it completely
DROPPED the real intervening content ("Johnny's in America... Ah-ah-ah,
ah-ah, ah-ah-ah", a scat/ad-lib line) and also misheard "coke" as
"conch". Since the decoder declared this whole 19s span as ONE segment
containing only those few words, wav2vec2's forced alignment had no
choice but to spread them across the full window, landing "Johnny wants
a brain" ~6-7s too early. **Confirmed via `match_words_to_asr`'s own
raw output that this is a real, literal exact-text match** (not a fuzzy
mismatch or occurrence-ambiguity bug) -- the words genuinely are in the
transcript, just with catastrophically wrong decoder-level segment
timing. This is a THIRD distinct failure class this session (see
[[project-realign-alignment-mode]]): outright decoder hallucination/
content-drop within an oversized segment, not a matching-algorithm issue
at all.

**Tested the obvious lever: reverting to WhisperX's own default VAD**
(`whisperx_no_vad=False`, i.e. `vad_onset=0.500/vad_offset=0.363` instead
of the existing `WHISPERX_NO_VAD_OPTIONS` near-zero override) -- **this
is NOT a fix, it's a regression**. Decoder segments got even BIGGER (one
26.6s segment swallowed an entire verse, `Chicago`-style, plus visible
text reordering -- "Johnny's in America" ended up transcribed AFTER
"Johnny wants to think of a joke" despite being sung first), and the
whole back half of the song's word-matching collapsed (`n_asr_matched`
cratered, deltas of +62s appearing from that point on). This directly
confirms the existing near-zero-VAD default (chosen for a DIFFERENT real
case, see `WHISPERX_NO_VAD_OPTIONS`'s own docstring: fixed a 5.88s->0.15s
error on "Stars") should stay -- don't revert it. Also confirmed
`condition_on_previous_text` is already `False` in whisperx's own
defaults (checked `whisperx/asr.py` directly), so that's not an available
lever either. `hallucination_silence_threshold` (whisperx default `None`)
is specifically for SILENCE-triggered hallucination, not applicable here
(this segment has real singing throughout, just misrecognized).

**Status: unresolved, no fix attempted yet.** No cheap, low-risk
WhisperX-level knob found. A downstream defense (e.g. distrusting ASR
words whose parent decoder segment is anomalously long/word-sparse -- a
direct fingerprint of this exact failure, unlike per-word confidence
alone which wasn't reliably low enough to catch it) would need new
plumbing (`Word` doesn't currently carry which decoder segment it came
from) and, per this session's own repeated lesson (`rematch_local_gaps`
above), needs real controlled validation before shipping -- not attempted
without the user's go-ahead given two "well-motivated mechanism didn't
generalize" results already this session.

### Long-segment re-windowing for whisperx forced alignment (PROTOTYPE,
### OFF by default, 2026-08-09/10)

Follow-up to the decoder-hallucination finding above -- user did their own
manual experiments (short clips align correctly where the full segment
doesn't) and proposed testing candidate windows for a long decoder
segment and picking whichever forced-alignment gives the best score.

**Real case (David Bowie - Magic Dance, via GUI, `large-v3`)**: decoder
segment `104.352-124.501` (20.1s) transcribed as just `"In 9 hours and 23
minutes, you'll be mine."` (8 words) -- the real "Jump Magic Jump!..."
content that's actually sung in there got dropped, same failure class as
the Americans case above. Forcing the 8 words against the whole 20.1s
window crammed them into 105.192-108.274 (score 0.564) -- nowhere near
the user's by-ear-confirmed real position, ~114-118s.

**Investigation (iterative, see full detail in session transcript)**:
1. First tried the user's exact proposal -- 4 fixed 8s windows tiling the
   segment (104-112/108-116/112-120/116-124). No clear winner (0.56-0.59
   band); every window crammed the phrase into its own first ~1-3s
   regardless of position -- a real, load-bearing finding: WIDTH doesn't
   matter, and a window even CENTERED on the true position (113-119)
   scored WORSE (0.513) than the eventual answer, because it wasn't
   exactly aligned to where the correct placement wants to start.
2. Widened to 12-20s at various offsets -- offset=106 emerged as best
   (mean 0.636), but this was later found to be a FALSE local optimum:
   the 2s-step sweep (96-112) happened to stop 2 seconds short of the
   true peak.
3. User confirmed by listening: true position is ~114s, not 106s.
   Re-tested at MATCHED widths, offset=114 head-to-head vs offset=106:
   114 won at every width (10s: 0.751 vs 0.649). A fine 1s-step sweep
   (110-118) confirmed a SHARP, isolated peak exactly at 114 (0.751,
   dropping to 0.55-0.58 just 1-2s either side) -- not noise, and not
   beaten by anything else in the whole 96-124 range tested. **This
   validated the core approach**: fine-grained, wide-enough offset search
   + pick-best-score DOES find the true answer; the earlier "106" false
   positive was a search-coverage gap, not a fundamental flaw in using
   CTC score to compare candidates.

**Shipped as a prototype** (`transcription.py`): `_rewindow_long_segments`
(+ `_find_best_window`, `_mean_word_score`) -- for any whisper decoder
segment >= `config.REWINDOW_MIN_SEGMENT_DURATION_SEC` (10.0s), sweeps
fixed-10s-width candidate windows at 1s steps (`REWINDOW_CANDIDATE_
WIDTH_SEC`/`REWINDOW_STEP_SEC`) across the segment's own declared span,
re-running `whisperx.align()` per candidate (real, unmodified whisperx
code, just called repeatedly with different `(start, end)` on the SAME
segment text) and keeping whichever wins by mean word score -- but ONLY
if it beats the baseline (whole-segment) score by at least
`REWINDOW_MIN_SCORE_IMPROVEMENT` (0.10, a first estimate from this one
real case, not yet broadly validated). Only ever corrects segment
BOUNDARIES before the normal `whisperx.align(result["segments"], ...)`
call -- everything downstream (Word construction, realign's own matching)
is untouched, so this composes for free. Wired through `transcribe_words
(..., rewindow_long_segments=...)` -> `RealignPipelineOptions.
rewindow_long_segments` / `--rewindow-long-segments` -- **OFF by default,
user's explicit request to keep it opt-in** while validated on more real
cases. Not wired into the main (non-realign) pipeline yet.

**Real validation, 2026-08-10**:
- Magic Dance (`large-v3`, real GUI-produced audio): fired exactly on the
  target segment, `104.352-124.501 -> 113.352-123.352`, score 0.564 ->
  0.710. Final written output places the line at 114.213-117.580s,
  matching the user's confirmed position. Clean, validated fix.
- Americans (fresh transcription, small.en default): the "Johnny wants a
  brain...coke" segment WAS flagged (18.6s) and a marginal improvement was
  found (0.584 -> 0.680, +0.096) but fell just short of the +0.10 bar, so
  baseline was correctly kept -- and baseline was ALREADY fine in this
  particular run (74.28-82.88s, the known-good pattern), so this was the
  right call, not a missed fix. Didn't get an unlucky-enough transcription
  in this attempt to re-exercise the ORIGINAL dropped-content failure.
- `test_dry_run.py` stays green throughout (no synthetic tests added --
  this calls real whisperx CTC internals, not mockable the way the rest
  of realign.py's matching logic is; validation is real-audio-only, per
  the CLAUDE.md convention that real validation is required regardless).

**Broader real validation (2026-08-10, same day, user's request: "try to
reproduce Americans a couple times [fresh Demucs each time], try BATB
(maybe helps Tune), pick 1-2 other songs")** -- 6 total real runs:

| Song | Fired? | Result |
|---|---|---|
| Magic Dance | Yes | 0.564->0.710, fixed to the confirmed 114-118s position |
| Americans (fresh Demucs #1) | No | baseline already fine (0.758) this run |
| Americans (fresh Demucs #2) | **Yes** | 0.593->0.723 -- reproduced the ORIGINAL bug fresh and fixed it, landing within 0.06s of the trusted existing file's own timing |
| BATB | No | baseline already fine (0.801); "Tune" lands correctly (86.44s vs original 86.38s) -- confirms the earlier catastrophic "Tune at 32s" case was the unrelated `validate`-strategy LRC-seed bug (see above), not a decoder-segment issue this mechanism would ever touch |
| Chicago | **Yes** | 0.564->0.674, landed within 0.15s of trusted timing |
| Stars | No | every segment already scored 0.72-0.87, no improvement found anywhere |

**Zero regressions across all 6** -- every fire was independently verified
as a genuine improvement (checked against each song's own trusted
original timing, not just the raw score delta), and every non-fire
correctly left an already-good baseline alone. ASR non-determinism is
still real and visible (Americans needed 2 fresh Demucs+transcribe
attempts to reproduce the original failure at all) but doesn't undermine
the mechanism's safety -- it only ever acts when it finds a clear,
verified-in-practice improvement.

**2 more songs tested same day (Ordinary Day, Heroes)** -- also zero
regressions: Ordinary Day fired once (0.652->0.801, fixed an ASR-garbled
opening line -- "four balls of rugby" mis-hearing -- to within 0.2s of
trusted timing); Heroes never fired, including on a segment with the
SAME oversized shape as the Magic Dance bug ("Just for one day I will be
king", 29.5s/8 words, baseline score only 0.598) where the sweep
correctly found NOTHING better -- checked the actual placement, it was
already correct (a sustained/held note dragging confidence down without
actually being misaligned). Good evidence the mechanism doesn't
over-fire on merely-low-confidence-but-correct segments.

**Total: 8 real songs, 4 genuine verified fixes (Magic Dance, Americans,
Chicago, Ordinary Day), 4 correct no-ops (Americans attempt 1, BATB,
Stars, Heroes), zero regressions.**

**Decision (user, 2026-08-10): flipped to DEFAULT-ON for realign mode.**
`RealignPipelineOptions.rewindow_long_segments` now defaults `True`
(hardcoded, decoupled from `config.REWINDOW_ENABLED` which stays the
main-pipeline's own separate switch); CLI flag inverted to `--no-
rewindow-long-segments` opt-out (mirrors `--whisperx-vad`'s pattern).
GUI needed no change -- it doesn't set this field explicitly, so it
picks up the new default automatically.

**Wired into the main (non-realign) generation pipeline the same day**,
STILL OFF by default there (`config.REWINDOW_ENABLED = False`,
`PipelineOptions.rewindow_long_segments`, `--rewindow-long-segments`/
`--no-rewindow-long-segments` in `main.py`) -- user's explicit ask to
start a validation series there before considering flipping it too,
same reasoning as everywhere else in this project (a different code
path needs its own real validation, isolation-mode/one-pipeline results
don't reliably predict another).

**Generation-pipeline validation, 4 real full pass 1-4 runs so far
(2026-08-10)**, all completed cleanly through note-fitting/verification/
output with no crashes or anomalies:
- Magic Dance (`large-v3`): decoder happened to keep the "Jump Magic
  Jump!...you'll be mine!" text together this run (not truncated the
  way the original bug run was) -- correctly no-op.
- BATB (`small.en`): matched its own realign-mode result exactly (same
  ASR under the hood) -- no fires needed, all baselines already good.
- Americans (`small.en`): fired, IDENTICAL numbers to the earlier
  realign-mode fix (0.593->0.723, same segment bounds) -- the WRITTEN
  output's fallback-word list confirms "coke." now lands at 83.86s (the
  corrected position), not the original ~76-79s bug. Full pass 1-4 +
  verification completed normally on top of the corrected timing.
- Ordinary Day (`small.en`): fired, identical to realign mode again
  (0.652->0.801 for the "four balls of rugby"-garbled opening line) --
  the WRITTEN output's `#GAP:16951` (16.95s) directly confirms the
  corrected position made it all the way through to the final note
  timing, not just the intermediate ASR word list.

Notable: Americans and Ordinary Day's generation-pipeline fires matched
their earlier realign-mode fires' scores EXACTLY (down to 3 decimals) on
reused cached vocals -- WhisperX's documented non-determinism doesn't
always manifest run-to-run on IDENTICAL audio input; it's real (confirmed
elsewhere this session with fresh Demucs runs) but not guaranteed to
differ every time.

**Decision (user, 2026-08-10): flipped to DEFAULT-ON EVERYWHERE.**
`config.REWINDOW_ENABLED = True` (the single shared master switch --
`transcription.transcribe_words`'s own default AND `config.
PipelineOptions.rewindow_long_segments` both read it, so this one flip
covers the main generation pipeline; realign mode already had its own
hardcoded `True` default set earlier the same day). `--no-rewindow-
long-segments` (both `main.py` and `realign.py` CLIs) opts back out if
ever needed. Final tally before this decision: 12 real runs across 8
songs (Magic Dance, Americans, BATB, Chicago, Stars, Ordinary Day,
Heroes across realign mode; Magic Dance/BATB/Americans/Ordinary Day
again through the full generation pipeline), 6 genuine verified fixes
(carried through to the actual written output, not just an intermediate
score), 0 regressions anywhere.

## ASR quality retry: re-transcribe with large-v3 on a low match rate
## (PROTOTYPE, real-validated 2026-08-10, CLI-only)

Motivated by a real case ("Trixie Mattel - Gold") where WhisperX
silently dropped whole lines rather than just mistranscribing individual
words. Deliberately NOT scoped to "count missing lines" specifically
(per the user's own ask for "something smarter than that") -- instead
reuses whichever real per-word match-rate metric a code path already
computes for its own "does this really match the audio" gate, since
that also catches partial degradation (garbled decoding, wrong-language
ASR, hallucinated long segments -- see the rewindow section above) the
same way it catches fully-dropped lines. Wired into all 3 places in the
codebase that already have such a metric:

- `realign.py`: `_retry_asr_if_low_quality` reuses `RealignQuality.
  anchor_rate` against the SAME `config.MXL_LRC_MIN_ASR_PLACEMENT_RATE`
  bar `realign_song` already warns on. Only wired for `strategy=
  "replace"` -- `realign_song_validate`'s `ValidateQuality` has no
  `anchor_rate`, not addressed here.
- `main.py`'s MXL+LRC primary path: retries via `mxl_lrc_generator.
  try_mxl_lrc_primary`'s own `MxlLrcQuality.asr_placement_rate` against
  the same bar (the exact gate that already decides success/fallback
  there).
- `main.py`'s standard (non-MXL) fallback path: no existing-file/MXL
  candidate to measure against there, so uses `lyrics_lookup.
  reference_match_ratio` (factored out of `reference_matches_transcript`,
  which only returned a bool) against a fetched reference lyrics
  candidate that has ALREADY cleared `REFERENCE_LYRICS_MIN_MATCH_RATIO`'s
  (0.25) much lower "is this even the right song" bar.
  `RETRY_ASR_MIN_REFERENCE_MATCH_RATIO = 0.6` is a first estimate,
  validated only on Gold so far. **Known gap**: if `--no-fetch-lyrics`
  or no reference is found at all, there's no ground truth to measure
  against in this path, so the retry never fires there -- not addressed.

Every retry site follows the same shape: fires only when the current
model isn't already `config.RETRY_ASR_MODEL` ("large-v3") and the
existing `retry_low_quality_asr` option is on (default ON in both
`PipelineOptions`/`RealignPipelineOptions` -- `--no-retry-low-quality-
asr` opts out on both CLIs); re-transcribes the WHOLE song once; keeps
the retry's own result only if it's better on WHICHEVER signal actually
triggered it, otherwise keeps the original untouched. Never retries a
second time. `current_asr_model` tracking in `main.py` prevents a song
that already retried in the MXL+LRC path from retrying AGAIN in the
standard fallback path if it falls through to it.

### Real validation found the initial whole-song-only design missed real cases

First real runs (Gold via `main.py`, David Bowie - Magic Dance via
`realign.py` with `--whisper-model small.en` to force a weak starting
transcription) exposed TWO independent real cases where a whole-song
aggregate metric hid a genuinely bad passage:

- **Magic Dance**: `small.en` produced a real decoder hallucination (a
  repeated "Dance Magic Dance Dance Magic Dance..." loop). Song-wide
  anchor rate landed at 58% -- comfortably above the 50% retry bar -- so
  the retry never fired, yet the actual written output had a real
  passage ("Slap that baby... I saw my baby trying hard...") placed
  ~12-14s late, because the rest of the song transcribed well enough to
  keep the aggregate healthy.
- **Gold**: the reference's own `"Do-do-do-do-do"` backing-vocal line
  (LRCLIB writes it repeated at 3 separate chorus repeats) produced ZERO
  ASR words at the FIRST repeat while the SAME phrase was transcribed
  correctly at a later repeat -- confirmed directly from the raw decoder
  segment text. `reference_match_ratio` stayed at 89-90% (well above the
  0.6 retry bar) across multiple fresh runs that all reproduced the
  identical drop, because 320+ other words in the song were fine.

**Fix: added a second, independent per-PASSAGE trigger to each path**
(`config.RETRY_ASR_MIN_UNCONFIDENT_RUN` / `RETRY_ASR_MIN_UNMATCHED_
REFERENCE_RUN`, both first estimates at 5):
- `realign.py`: `RealignQuality.longest_unconfident_run` -- longest
  CONSECUTIVE run of words with no real anchor at all (computed for
  free from the `confident` array `match_words_to_asr`/`_windowed`
  already produce).
- `main.py` standard fallback path: `lyrics_lookup.
  largest_unmatched_reference_run` -- longest contiguous run of
  reference words with NO corresponding ASR word (a difflib `insert`
  opcode `align_words_to_reference` already recognized but had nothing
  to DO with -- "reference has words ASR completely missed... simply
  not represented").

**First version of the per-passage trigger STILL missed Gold**: the
reference's `"Do-do-do-do-do"` is written as ONE hyphenated token, and a
correctly-matched `"They start to play"` line sitting between two
occurrences split what's really one ~15s dropped passage into two
separate 1-token `insert` blocks -- both under the threshold of 5. Root
cause was really `_tokenize_lines` counting a hyphenated token as a
single word no matter how many real sung syllables it represents.
**Fixed at the tokenizer, not the threshold**: `_tokenize_lines` now
splits every token on `-` as well as whitespace (2026-08-10) -- a
general improvement, not just for this signal, since
`align_words_to_reference`'s own alignment can now match/correct
"Do-do-do-do-do" as 5 separate words instead of one opaque blob (its
existing `is_repeat_clamp` uneven-block handling becomes the FALLBACK
for when ASR's own word count still doesn't line up, rather than the
every-time case). Confirmed against the real fetched reference + real
parsed ASR debug-log data for this exact song:
`largest_unmatched_reference_run` went from 1 (pre-fix) to 7 (post-fix,
whole song) / 5 (an isolated single dropped repeat, synthetic test).

**Real end-to-end re-validation after the fix (Gold, 4 total real
runs)**: the per-passage trigger fired (`"a 5-word run of reference
text has NO corresponding ASR word at all"`), retried with `large-v3`,
and the retry was accepted (reference match 90% -> 95%). Confirmed
DIRECTLY in the written output file (not just the intermediate metric):
both previously-missing "Do do... do do." passages now have real text
and timing, with "They start to play." correctly matched in between --
`492` synced lines, same overall structure as before, no regression
elsewhere. Note: the retry's own `largest_unmatched_reference_run`
stayed at 5 (didn't drop to 0) even though it was accepted -- it was
kept because the WHOLE-SONG ratio improved (90%->95%), not because the
per-passage signal itself cleared; the specific passage that DID get
fixed was a genuinely separate occurrence recovered by the bigger model,
not proof that large-v3 fixes every dropped repeat every time. Also
confirmed the OLD design's own two whole-song-only checks work as
intended when they're what's actually needed: an early Gold run's
`asr_placement_rate`-style checks correctly stayed quiet on a run where
ASR quality was genuinely fine end to end (no false-fire).

**Debug log gained the retry's decision narration, not just its raw
data**: the retry's raw ASR/realign trace was already being captured
(since `debug_log` was already threaded into the retry's own
`transcribe_words`/`realign_song`/`try_mxl_lrc_primary` calls), but WHY
it fired and what was decided (before/after numbers, accept/reject) only
ever went to the console `log` callback, not the file -- a real gap
found when asked directly "does the debug log contain the re-run data".
Fixed: both `main.py` (both retry sites) and `realign.py` now mirror
every top-level decision message into the debug log file via a small
local `dlog()` helper, plus a labeled `debug_log.section(...)` marker
(e.g. `"ASR QUALITY RETRY (standard fallback path)..."`) before each
retry. Verified two ways: `realign.py`'s function exercised directly
against a real file-backed `DebugLog`; `main.py` re-validated via a real
Gold run showing both the trigger line and the outcome line landing in
the actual file.

Synthetic regression tests: `realign.py`'s `_retry_asr_if_low_quality`
(whole-song trigger: fires/adopts an improvement, no-ops at the retry
model, no-ops above the bar; per-passage trigger: fires on a long
unconfident run even with a fine anchor rate, no-ops just under the
per-passage bar) and `lyrics_lookup.largest_unmatched_reference_run` /
`_tokenize_lines` (hyphenated-token splitting, matching the real Gold
shape). `main.py`'s two retry sites are still NOT unit tested (same
reasoning as the rewindow section above -- this is orchestration around
real transcription calls) but ARE now real-audio validated end to end.

**Resolved same day**: Magic Dance's per-passage fix has now been
re-validated end-to-end with default options (`medium.en`, not the
small.en used to originally expose the gap) -- 150/180 words matched
directly, 26 more recovered via `force_align_gaps`, only 4 interpolated,
0 kept-original. CLI-only for now (no GUI checkbox) -- same "ship
CLI-only, add GUI once validated" pattern used for `--strategy
validate` initially. The standard fallback path's no-reference-available
gap (noted above) is unchanged.

### Retry gated to `--batch` mode only (2026-08-10, user's explicit ask)

User's request: "Make the large-v3 re-run only happen during batch mode.
In single mode, add a warning to the log that suggests doing that" --
motivated by cost: a whole-song large-v3 re-transcription roughly
doubles that run's ASR time, which `--batch` (unattended, cost-tolerant
by its own existing convention) can absorb but an interactive single run
shouldn't spend without asking first. All 3 retry sites (`realign.py`'s
`_retry_asr_if_low_quality`, `main.py`'s MXL+LRC-path retry, `main.py`'s
standard-fallback-path retry) now check `opts.batch` before firing: in
`--batch` mode, unchanged (retries as before); outside `--batch`, logs a
WARNING naming the specific metric that triggered, that a retry would
likely help, and that `--batch` (or `--whisper-model large-v3` directly)
would fix it -- then returns the original untouched result, same as the
existing "no reference to check against" no-op path. Required adding
`batch: bool = False` to `RealignPipelineOptions` (didn't exist there
before) and wiring `batch=opts.batch`/`batch=is_batch` through both
`gui.py`'s `_build_opts`/`_build_realign_opts` (the realign one was
previously missing entirely) and `main.py`'s `_opts_from_args` (a
pre-existing, unrelated gap -- `PipelineOptions.batch` existed but was
never actually populated from the CLI args before this session).

Real-validated live via the BATB and Gold `--verify-placement` test runs
below (run in single-song mode, not `--batch`): both correctly logged
the WARNING instead of retrying -- BATB's MXL+LRC path (8%/9% ASR
placement rate, known wrong-cast "(Finale)" candidate) and Gold's
standard fallback path (a 5-word unmatched-reference run) both declined
to retry and fell through to their normal non-retried behavior, exactly
as designed.

### `--verify-placement` re-tested, regression reconfirmed -- see
### "Removed / rejected approaches" above for the full write-up

User's second ask this session: "Run some tests on the 'Verify
placement' option to see if it's still worth keeping around." Full
methodology, numbers, and root-cause explanation are recorded in the
"Removed / rejected approaches" section near the top of this file
(kept there since that's the canonical home for this constant's
history) -- summary: still a net regression on real ground truth
(BATB: 105->101 words matched, 97.1%->93.1% timing agreement) even
after this session's other improvements, root-caused this time to the
in-window text-presence check having no way to disambiguate WHICH
occurrence of a repeated word is correct. **Fully removed from the
codebase same day** (user's explicit follow-up: "Completely remove
verify-placement") rather than kept as dead-but-present code.

## Rewindow follow-ups: split-into-sub-phrases and LRC-line-anchored
## recovery, both real-validated and REJECTED (2026-08-10)

User's motivating question: could long-segment rewindowing (see above)
be made more effective at actually improving WORD ACCURACY, either by
searching more aggressively or by breaking a long segment into smaller
pieces first? Two different designs were built and real-tested; both
were rejected.

### 1. Split-into-sub-phrases rewindow (`REWINDOW_SPLIT_ENABLED`) --
### tried, real-validated, REJECTED, code KEPT (off by default)

Instead of `_find_best_window`'s one sliding window over a long
segment's ENTIRE text as a single block, `_rewindow_split_segment`
splits the segment's own text into sub-phrases (`_split_segment_text` --
sentence-ending punctuation primarily, fixed word-count chunks as a
fallback for punctuation-less repeated-chorus text) and searches each
sub-phrase's own best window SEQUENTIALLY (each subsequent search
starts no earlier than where the previous one landed, so total
`whisperx.align()` calls stay roughly the same order as the existing
whole-block search, not multiplied by sub-phrase count).

**Real-tested on Chicago** (properly LRC-matched -- see
[[feedback-lrc-required-for-tests]], which this investigation's first
attempt, on "I'm Afraid of Americans", violated and had to be redone):
every segment where a split was attempted scored WORSE than the
whole-block baseline, not marginally -- e.g. 0.716 vs 0.826, 0.671 vs
0.793. Root cause: forcing wav2vec2 CTC to align a short (6-8 word)
sub-phrase in isolation loses the longer-context signal that helps it
lock onto the right audio -- the SAME "don't trust inference from a
tiny isolated clip" failure class this project has already hit with
pYIN pitch analysis and `verify_words`'s own isolated recheck (see
Lessons learned above). Kept in the codebase as a documented dead end
(`config.REWINDOW_SPLIT_ENABLED = False`), same treatment as
`--zone-boundary-snap` -- not pursued further without new evidence.

### 2. LRC-line-anchored recovery -- tried, real-validated, REJECTED,
### code FULLY REMOVED (user's explicit request, different treatment
### from the split prototype above)

A structurally different mechanism, proposed as a way to avoid split's
context-loss problem: instead of a blind search (no ground truth, just
comparing CTC scores against itself), use LRC line timing as
INDEPENDENT ground truth. Calibrates a per-song time offset between LRC
line timestamps and our own audio (reusing `lrc_timing.py`'s own
`two_tier_time_calibration` -- an uncalibrated candidate never touches
word timing, same gate `realign.py`'s `lrc_mode` uses). For any LRC
line whose matched words are AT LEAST half outside the calibrated
expected window (the "Johnny wants a brain" failure signature -- a
whole decoder segment placed against the wrong stretch of audio, not
just a couple of individually noisy words), re-aligns that line's OWN
reference text (not whatever ASR decoded -- the decoder's own text can
be wrong in exactly this failure case) via a SINGLE
`force_align_words_in_window` call scoped to a real line's worth of
text, avoiding the split prototype's context-loss failure.

**Real-tested on Chicago, standard fallback path forced via
`--no-mxl-lrc-primary`** (Chicago's own MXL+LRC primary path would have
skipped this code entirely). Fired on 3 of 4 flagged lines (1 correctly
declined -- force-align failed, left unchanged, safe fallback working
as designed). Compared against Chicago's own real ground-truth `.txt`
(`sandbox/Chicago - When You're Good to Mama/Chicago - When You're Good
to Mama.txt` -- note this file's PITCH convention isn't calibrated to
ours, so pitch-class comparisons against it are meaningless; only
TIMING is usable) via `verify_existing_song`: baseline (no recovery)
scored 99% timing agreement (188 matched words, only 2 mismatches
unrelated to this line); with LRC-anchor-recovery active, timing
agreement DROPPED to 97% -- the mechanism took an ALREADY-CORRECT
passage ("and I'll boost you up yours, let's...", none of these 5 words
were mismatches in the baseline) and moved every word 0.65-1.09s away
from its correct ground-truth position. A second flagged line
("chickies") fired but had ZERO net effect on the final written output
-- pass 3's own line-level proportional distribution for a matched
reference line already absorbs this kind of raw-ASR-word-level error,
so "fixing" the raw word achieved nothing either way. Net real result:
one confirmed regression, one wasted no-op, one safe decline, zero
genuine wins -- same shape as `--verify-placement`/`--zone-boundary-
snap`'s own history of a well-motivated mechanism that doesn't survive
contact with real data.

**User's explicit instruction, different from every other rejected
prototype in this file**: "Reject. Don't keep the functionality
around." -- `lrc_timing.recover_misaligned_lrc_lines`/
`_match_words_to_lrc_lines_full`, the `--lrc-anchor-recovery` CLI flag,
`config.LRC_ANCHOR_RECOVERY_*`, `PipelineOptions.lrc_anchor_recovery`,
and its tests are all fully removed, not kept as dead-but-present code
the way `--zone-boundary-snap`/`REWINDOW_SPLIT_ENABLED` are. Don't
re-add without new evidence.

## UltraStarKaraokeMaker-inspired improvements: force-align gaps, note
## over-segmentation investigation, and melisma-tail merging (2026-08-10)

Compared our own output on "Trixie Mattel - Gold" against a real run of
UltraStarKaraokeMaker (github.com/walterfr/UltraStarKaraokeMaker, MIT-
style OSS) on the same song. Three concrete things came out of this:

### 1. Force-align known gaps -- shipped, see the section above
`transcription.force_align_words_in_window` (real wav2vec2 CTC forced
alignment of KNOWN text into a bounded audio window) directly adapted
from USKMaker's own `realign_gap_windows`. Wired into both `realign.py`
(`_force_align_unconfident_runs`) and `main.py`'s standard fallback path
(`lyrics_lookup.recover_dropped_reference_words`). Default ON in both
(`config.FORCE_ALIGN_GAPS`), `--no-force-align-gaps` opts out.

### 2. Note over-segmentation ("extra notes") investigation -- explored,
### NOT shipped, reverted to default
User's own real-ear comparison flagged that our output has far more
notes than USKMaker's for the same song, hypothesized as coming from
our pitch-first architecture (pass-1 detects pitch from audio ALONE, no
lyrics knowledge, vs. USKMaker's forced-alignment-first + per-syllable
pitch extraction afterward). Confirmed with real numbers: untexted `~`
continuation-note rate varies wildly by song under our OWN pipeline
(Gold 33%, Stars 32%, Magic Dance 0%, Ordinary Day 8%) -- not a uniform
defect. Cross-checked against Stars' real sheet-music OMR ground truth:
our rate (32%) is ~7x the REAL notated melisma rate (4.4%), and pass-1's
own RAW note count (258) is already ~2.8x the real note count (91)
*before* any lyrics are even involved -- confirms genuine pass-1
over-segmentation, not "this song is just melismatic."

Tried widening `NOTE_MERGE_SEMITONES` (1 -> 2) as the fix. Structural
analysis of what actually gets merged (28 groups, same-audio controlled
sweep) found ~64% are clearly transient/vibrato artifacts (a dominant
sustained note bracketed by much-shorter same-or-near-pitch fragments,
several with the exact same pitch on BOTH sides of a brief dip --
onset/release tracking noise, not real melody), but ~15-20% are
genuinely ambiguous 2-note comparable-duration steps that could be real
stepwise melodic movement -- and `NOTE_MERGE_SEMITONES` was *already*
lowered from 2 to 1 once before, specifically because a looser threshold
was confirmed to chain-merge real melodic movement into flattened notes
(see the constant's own docstring in config.py). A real controlled
same-audio-no-separation comparison avoided the confound from an EARLIER
attempt at this same comparison, which accidentally compared two
different GENERATION PATHS (MXL+LRC vs pass-1) rather than the merge
threshold, because Stars has its own `.mxl` file and MXL+LRC-primary
(the default when an MXL is present) skips pass-1 entirely --
`[PASS1 DEBUG].txt`'s own timestamp not matching the run in question is
the tell for this; don't assume a full-pipeline run exercised pass-1
just because pass-1-specific flags were on.

Real controlled validation (both non-MXL forced via `--no-mxl-lrc-
primary`, run against real ground truth): **BATB vs its own hand-
verified SingStar ground truth showed 100% pitch-class accuracy at
`merge=2`** (103 matched words, 97% timing agreement) -- no pitch cost
found on the one comparison with genuinely trustworthy ground truth.
Stars vs its own MXL score showed only 60% agreement, but that
comparison is inherently weaker (an MXL score and a real performance can
legitimately diverge in key/arrangement independent of detector
accuracy) and there was no `merge=1` baseline via the same method to
compare against.

**Decision: NOT shipped.** `NOTE_MERGE_SEMITONES` reverted to `1`
(default) after the investigation. Despite BATB's clean 100% result, the
~15-20% ambiguous-case rate combined with this constant's own prior,
specific, confirmed real-bug history (see its own docstring) was judged
not worth the risk without either better ground truth or actual
listening validation. Both `--no-mxl-lrc-primary` real outputs from
this investigation are left on disk for the user to listen to directly:
`sandbox/Les Misérables - Stars/Output_nomxl_merge2/` and `sandbox/
Beauty And The Beast - Beauty And The Beast/Output_nomxl_merge2/`.

### 3. Melisma-tail merge -- shipped, default ON
Separate, much narrower, real-user-reported fix: a melisma-continuation
note (`~`) that's beat-adjacent (no gap) to the PRECEDING note at the
*exact* same pitch is redundant on the beat grid and just makes the
chart busier to read/sing -- e.g. "ly" (a real word syllable) followed
immediately by a same-pitch `~` should be ONE longer note, not two.
Real example that motivated this ("Barely even friends"):
```
: 263 1 3 ly          : 263 4 3 ly
: 264 3 3 ~      ->   (merged away)
```
`usdx_writer._merge_connected_melisma_tails` runs at the INTEGER BEAT
level (on exactly what will be written, post-quantization -- not on the
pre-quantization float-second timing, since quantization itself can
change what counts as "adjacent"), folding a `~` into whichever note
precedes it (a real word syllable OR another `~`) and extending that
note's own length -- chains, so several connected same-pitch `~` notes
in a row all collapse into one. Never merges across a LineBreak.
Deliberately NOT applied to `realign.py` (default `False` at the
`render_song`/`write_song` function level itself, only `main.py`
explicitly opts in via `opts.merge_connected_melisma`) -- realign has
its own stricter "never add/remove/reorder a note" contract that
deleting a note would violate. Default ON in `main.py`
(`config.MERGE_CONNECTED_MELISMA_TAILS`), `--no-merge-connected-melisma`
opts out. Verified against real output: zero remaining same-pitch
beat-adjacent `~` violations across all 3 gen-test outputs below.

### Final 6-song real validation (2026-08-10), all default options
Gen tests (BATB, Chicago, Gold) and realign tests (Chicago, Americans,
Magic Dance) run with every new default (`force_align_gaps`,
`retry_low_quality_asr`, `merge_connected_melisma` all ON,
`NOTE_MERGE_SEMITONES` reverted to `1`) together for the first time:

- **BATB gen**: MXL+LRC primary correctly rejected again (8%->9% ASR
  placement rate even after the large-v3 retry -- the same known
  wrong-cast "(Finale)" candidate issue, not a new bug), fell back to
  standard generation. `force_align_gaps` recovered 20 words there.
  100% pitch-class accuracy against real ground truth (105 matched
  words), 97.1% timing agreement.
- **Chicago gen**: MXL+LRC primary succeeded cleanly on the first try --
  113/116 words placed via transcription, 0 monotonic fixes.
- **Gold gen**: exactly reproduced the already-validated force-align-
  gaps + large-v3-retry behavior (10 words recovered directly, retry
  improves reference match 91%->95% on the residual gap).
- **Chicago realign**: 208/210 words matched directly to ASR (99%), 2
  more recovered via `force_align_gaps`.
- **Americans realign**: correctly rejected the wrong-edition LRC
  candidate again (31 vs 40 "afraid" occurrences, same repeat-structure
  check as before) -- the per-passage trigger (39-word unconfident run)
  fired the large-v3 retry, which brought anchor rate from 62% to 100%
  (`force_align_gaps` alone had already recovered 29 words; the retry's
  own force-align pass recovered 129 more).
- **Magic Dance realign**: 150/180 matched directly, 26 recovered via
  `force_align_gaps`, only 4 interpolated, 0 kept-original -- see the
  "resolved same day" note above.

Zero regressions, `test_dry_run.py` green throughout. This is the first
time all of this session's mechanisms have been real-validated running
together rather than in isolation.

## Environment

- Windows, venv at `E:\Projects\ultrastar_generator\venv`.
- CUDA required — pipeline aborts at startup if unavailable. WhisperX
  pulls in pyannote/torch; expect noisy-but-harmless startup warnings.
- Demucs stems cached in `sandbox/.ultrastar_work/` — safe to delete
  between runs; separation is skipped if `vocals.wav` already exists.
- **No console-window flashing** (2026-08-09, user-reported): every
  `subprocess.run` call that spawns a real console-subsystem executable
  (ffprobe/ffmpeg in `media_extract.py`, Demucs in `separation.py` --
  the only 3 call sites in the repo) passes
  `creationflags=subprocess.CREATE_NO_WINDOW` (a module-level `_NO_WINDOW`
  constant per file, `0` on non-Windows where the flag doesn't exist) --
  otherwise Windows pops a new console window per call whenever the
  parent process has none of its own (the GUI, launched via `pythonw.exe`
  per `run_gui.bat`). Harmless when a console DOES already exist (CLI run
  from a terminal) since output is captured via `capture_output=True`
  regardless of window visibility either way. Real-validated: `ffprobe`
  (`has_audio_stream`) still correctly detects a real audio file's stream
  with the flag applied. yt-dlp (YouTube mode) is invoked as a Python
  library call, not a subprocess, so it isn't affected by or in need of
  this fix; any console flash it produces internally (unlikely, but not
  independently checked) would be third-party behavior outside this
  project's control.

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