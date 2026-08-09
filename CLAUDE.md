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

### Real bug: ASR word-matching compared against the MXL's raw OCR
### text instead of the already-resolved clean text (2026-08-09)

User pushed back on the "not fully resolved" note above with real
WhisperX output for "There's a lot of favors":
```
69.865 -  70.325  score=0.753  "There's"
70.385 -  70.425  score=0.896  'a'
70.585 -  70.885  score=0.996  'lot'
70.945 -  71.005  score=0.532  'of'
71.145 -  71.826  score=0.835  'favorites'
```
and asked directly why "favors"/"favorites"' own precise ASR timing
wasn't used for "fa"'s start and "vors"' end.

Reconstructed the REAL `Word` list from this run's own debug log (same
data `transcribe_words` itself produces -- confirmed by reading
`transcription.py`'s debug-dump code path directly) and replayed
`place_words_via_asr`'s exact per-line matching for this line:
```
mxl_norm_line: ["there's", 'a', 'lot', 'of', 'favere']
asr_norm:      ["there's", 'a', 'lot', 'of', 'favors', "i'm"]
equal   ["there's", 'a', 'lot', 'of'] <-> ["there's", 'a', 'lot', 'of']
replace ['favere']                    <-> ['favors', "i'm"]
```
Root cause: `place_words_via_asr` matched against `mxl_words[i].norm` --
the MXL's own raw OCR text ("favere") -- never the already-resolved
CLEAN text (`word_clean_text[i]` == "favors", computed by
`assign_words_to_lines` via the LRC line, already used for DISPLAY text
since the earlier "MATIZON"/"systern" fix) -- so a word whose MXL OCR and
ASR transcription each garbled it differently never matched even though
a clean, mutually-agreeing "favors" was available and simply unused for
this purpose. Confirmed directly, and separately confirmed real
(non-bug): "There's"/"a"/"lot"/"of" were ALREADY landing within ~30ms of
their own real ASR timestamps in the shipped output before this fix --
verified by direct comparison against the reconstructed real `Word`
list -- so if those specific words still sound off, that's ASR's own
word-boundary precision for this passage (a different, harder problem),
not a matching bug.

Fixed: `place_words_via_asr` gained an optional `word_clean_text` param;
`mxl_norm_line` now prefers `_normalize(word_clean_text[i])` over the raw
MXL norm whenever a clean match exists, falling back to the MXL norm
otherwise (unaffected when no clean text was resolved). Reuses
`assign_words_to_lines`' own resolution rather than adding a second,
separate fuzzy-match step -- a word is only ever resolved against the
LRC once. New regression test (`test_dry_run.py`): an MXL word OCR'd one
way, transcribed by ASR a different way, with a clean LRC-resolved text
that matches the ASR exactly -- confirms it's now confidently matched
(and confirms the OLD behavior, run without `word_clean_text`, correctly
still misses it -- proving the fix is really doing the work).

**Real re-validation, same Chicago song, same `--lrclib-id 37066985`**:
ASR placement rate rose from 108/116 to 114/116 words (6 more words
recovered a real confident ASR match, not just this one). The "favors"
region directly:
```
: 1135 5 18  fa
: 1141 11 18 vors
- 1152 1178
: 1178 5 9  I'm
```
fa+vors now span 71.120s-71.843s (723ms) -- matching real ASR's own
"favors" span (71.129s-71.809s, 680ms) almost exactly, and close to
ground truth's fa/vors timing (71.05-71.85s, 800ms: fa start Δ70ms, vors
start Δ18ms, vors end Δ7ms). Previously this same region spanned
71.204s-72.972s (1.77s), swallowing more than half of "I'm" -- "I'm"
itself is now untouched, starting cleanly at beat 1178 in both runs.
`test_dry_run.py` full suite green throughout.

**Same-day follow-up: the exact-match fix above was NOT sufficient on its
own -- WhisperX itself is not deterministic in the literal WORDS it
transcribes, not just their timestamps.** The user re-ran the exact same
command and got the OLD broken numbers back (`fa` len=14, `vors` len=29,
matching the state from BEFORE this fix). Root cause: in their run, ASR
transcribed the word as "favorites" (confirmed independently -- this is
literally what the user's own first WhisperX excerpt in this thread
showed, before any of these fixes existed), not "favors". `word_clean_text`
still resolves to the correct, exact "favors" (that resolution comes from
LRC text matching, not ASR, so it's unaffected) -- but comparing clean
text ("favors") against ASR's own wording ("favorites") is still an exact
non-match, so the fix from the previous entry didn't fire for this run.

Fixed by extending `place_words_via_asr`'s per-line matching to also
accept a close-but-not-identical 1:1 "replace" pairing (character-level
ratio >= `config.MXL_LRC_FUZZY_TEXT_MIN_RATIO`, same threshold and
technique `assign_words_to_lines` already uses for display text) --
confirmed real ratios: "favors"~"favorites" 0.80 (clears easily),
"favere"~"favorites" 0.53 (correctly stays below threshold, confirming
clean-text reuse is still doing independent, necessary work -- fuzzy
matching against the raw uncorrected OCR text alone would NOT have caught
this specific case). New regression test covers both: the mishearing
case now matches, and a genuinely unrelated word in the same slot is
still correctly rejected.

Re-ran the real pipeline 3 times total across this investigation;
WhisperX transcribed this specific word as "favors" (not "favorites")
in all 3 of my own runs, so I could not directly reproduce the user's
exact mishearing locally -- the fix is verified via a dedicated synthetic
test built from the user's own literal reported ASR output ("favorites",
score=0.835) rather than a fresh real-audio repro. This is a new,
previously-undocumented instance of ASR non-determinism in this
pipeline: prior "Lessons learned" entries in this file already document
Demucs and pass-1 CREPE/RMVPE non-determinism at the AUDIO/PITCH level;
this is the first confirmed case of WhisperX itself producing a
different literal WORD (not just a different timestamp/score) for the
same input across runs.

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