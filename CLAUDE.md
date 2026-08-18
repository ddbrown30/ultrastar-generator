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
  until the raw-ASR section was actually read). Also has, for pass 1:
  a RAW PASS-1 FRAMES section (per-frame direct output of whichever
  single `pitch_source` ran — raw pitch/confidence/voicing BEFORE any
  smoothing, segmentation, or merge pass, vs. the same frame's value
  AFTER contour smoothing — isolates "the pitch tracker got this frame
  wrong" from "a later pass distorted an originally-correct reading"),
  a per-stage RAW NOTES dump (note list snapshotted after each
  segmentation/merge/cleanup stage), and inline per-decision lines for
  the note-shaping passes that make individual judgment calls (`[spike-
  removed]`, `[trailing-artifact-absorbed]`, `[rearticulation-
  reconcile]`) — each names the specific note(s)/timestamps involved and
  why, not just a before/after count.

## Lessons learned (do not reintroduce)

- Never run pYIN on a tiny isolated clip — needs context. Fixed by
  analyzing the whole track once; pass-2 fallback borrows pitch instead
  of re-running pYIN on ~0.1s clips. Same failure class shows up
  elsewhere too: forced-alignment (wav2vec2 CTC) also loses accuracy
  when forced to align a short, isolated chunk instead of a
  longer-context window (confirmed for the split-into-sub-phrases
  rewindow prototype, see "Removed / rejected approaches" below) —
  treat "don't run inference on a tiny isolated clip" as a general
  rule, not a pYIN-specific one.
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
- **Repeated-phrase/occurrence disambiguation is a recurring failure
  class**, not a one-off bug — hit independently in at least 6 different
  mechanisms across this project's history (`--verify-placement`'s
  expand-search, `realign.py`'s `"windowed"` and `"seed"` matching, both
  rejected rewindow follow-up prototypes, and `lrc_timing.
  match_asr_to_lrc_lines`, see below). A TIME- or plain SEARCH-window
  containing more than one real occurrence of a repeated word or phrase
  has no way to pick the right one on its own, and confidently returns
  whichever it finds — sometimes far away. No aggregate confidence
  signal reliably catches this. Don't re-propose "just search a window
  and check if the text is there" as a fix for a placement bug without
  addressing this directly. **A real fix pattern now exists and is
  shipped in multiple places** (`reconcile_line_structure`, `lyrics_
  lookup.assign_lrc_line_ids_sequentially`, `match_asr_to_lrc_lines`):
  a forward-only SEQUENCE-POSITION cursor, never a time/search window —
  walk both sequences forward together, search only the NOT-YET-
  CONSUMED remainder on each step. This is a structurally different
  category from the rejected time-windowed attempts: a repeated phrase
  later in a sequence is provably unreachable once the cursor has
  already advanced past its earlier occurrence, rather than merely
  unlikely to be picked. Still worth defaulting to for any NEW placement
  mechanism hitting this class, rather than re-inventing a window-search.

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
  (What *did* ship, and what the whole pass-1 architecture below was
  simplified down to entirely: a single pitch source with its own
  voicing, no cross-check at all.)
- **pYIN, CREPE, PENN, and the whole pitch cross-check/ensemble/
  consensus-override architecture** (2026-08-14, user's explicit
  request, fully removed): `note_detection.py` used to run a primary
  source (pYIN by default) plus up to several other sources (CREPE,
  RMVPE, PENN, SwiftF0) as agree/disagree cross-checks
  (`_cross_check`), with an optional final unanimous-consensus pitch
  override (`_consensus_pitch_override`) on top. All of it is gone —
  `detect_notes()` now always runs through exactly ONE `PITCH_SOURCES`
  entry (`pitch_source` param, `"rmvpe"` or `"swiftf0"`), which supplies
  both the pitch value and the voicing decision, exclusively. This was
  already the better-performing real shipped path (see above); the
  ensemble code was dead weight duplicating the isolation branch, not a
  live alternative. `pitch.median_pitch_in_span` (lyric_alignment.py's
  last-resort fallback for a word with no note anywhere nearby) also
  switched from raw `librosa.pyin` to the same `PITCH_SOURCES` registry.
- **`--verify-placement`** (expand-search re-transcription to fix note
  boundaries): fixed individual real cases but was a net regression on
  every pitch/timing metric end-to-end on every tested song, confirmed
  on two separate occasions months apart (most recently: BATB words
  matched 105→101, timing agreement 97.1%→93.1%). Root cause: the "is
  the expected word's text present anywhere in this search window"
  check can't disambiguate WHICH occurrence of a repeated word is
  correct when the window contains more than one — the general
  repeated-phrase disambiguation problem noted above. **Fully removed
  from the codebase**: `verification.verify_placement`, `PlacementCorrection`/
  `PlacementWarning`, the CLI flags, the GUI checkbox, and its tests are
  all gone.
- **`main.py`'s existing-file verification** (was `--existing-txt`/
  `--existing-txt-check`, GUI "Check existing .txt before overwriting"
  checkbox — parsed an existing `.txt`, compared pitch-class/timing
  against a fresh pass via `verify_existing_song.verify_existing_song`,
  kept the existing file byte-for-byte on `PASS` instead of
  regenerating): **fully removed from the codebase 2026-08-16** (user's
  explicit request): the CLI flags, `config.ENABLE_EXISTING_TXT_CHECK`,
  `PipelineOptions.existing_txt_check`/`existing_txt_path`, the
  `_run_pipeline_body` detection/comparison/keep-existing-file logic,
  `PipelineResult.regenerated`, and the GUI checkbox are all gone.
  `verify_existing_song.py` itself (and its own threshold constants,
  e.g. `EXISTING_TXT_MIN_COVERAGE`) is **not** removed — still a live,
  git-tracked utility for real-output-vs-ground-truth comparisons (see
  its own entry earlier in this doc), just no longer wired into
  `main.py`'s own default pipeline flow.
- **`--zone-boundary-snap`** (was: snap zone/word boundaries to nearby
  pass-1 note onsets): synthetically verified, but flat-to-negative on
  all 5 real songs tested. Kept off-by-default/dead-but-present for a
  while (still had a real CLI flag and GUI checkbox, unlike the other
  entries in this section); **fully removed from the codebase
  2026-08-15** (user's explicit request): `lyric_alignment.
  _snap_boundary_to_note_onset`, the `snap_boundaries`/`snap_radius_sec`
  params threaded through `_assign_notes_to_groups`/
  `_split_notes_by_word_boundaries`/`align_words_to_notes`/
  `alignment.align_words`, `config.ENABLE_ZONE_BOUNDARY_SNAP`/
  `ZONE_BOUNDARY_SNAP_RADIUS_SEC`, the CLI flags, the GUI checkbox, and
  its tests are all gone.
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
- **Local-rematch of unmatched word runs** (was `realign.py`'s
  `rematch_local_gaps`/`use_local_rematch`, retried a run of unmatched
  words against only nearby-in-time ASR instead of the whole song): net
  regression across a controlled 4-song A/B (BATB −24pp within 100ms).
  The reference songs' existing local timing is already trustworthy, so
  `interpolate_fallback`'s proportional guess usually beats a local
  rematch — and a local rematch can lock onto the wrong nearby repeat
  just as easily as a whole-song search can. Shipped OFF, code kept (not
  CLI/GUI-wired) for a while; **fully removed from the codebase
  2026-08-15** (user's explicit request, same treatment as
  `--verify-placement`/split-rewindow above — don't re-attempt without a
  way to distinguish "original timing is wrong here" from "ASR is just
  sparse here").
- **Split-into-sub-phrases rewindow** (was `REWINDOW_SPLIT_ENABLED`,
  `transcription._rewindow_split_segment`/`_split_segment_text`):
  breaking a long decoder segment into pieces and windowing each
  separately scored WORSE than whole-block alignment on every attempted
  case — the "don't run inference on a tiny isolated clip" failure class
  again. Kept off-by-default/dead for a while; **fully removed from the
  codebase 2026-08-15** (user's explicit request, same treatment as
  `--verify-placement`/LRC-line-anchored recovery below — don't
  re-attempt without new evidence).
- **LRC-line-anchored recovery** (used LRC line timing as independent
  ground truth to re-align a whole flagged decoder-hallucinated line):
  real-tested on Chicago — one confirmed regression (moved an
  already-correct passage 0.65-1.09s away from ground truth), one
  wasted no-op (pass 3's own line-level proportional distribution
  already absorbs a raw-word-level "fix"), one safe decline, zero wins.
  **Fully removed from the codebase** (user's explicit "reject, don't
  keep the functionality around").
- **Mixed (non-isolated) audio for pass 1's own segmentation, and a
  hybrid isolated-segmentation/mixed-pitch-class variant** (investigated
  2026-08-15, nothing shipped to `main.py`/`note_detection.py`): motivated
  by a real single-song finding (Trixie Mattel - Gold: RMVPE pitch-class
  86.3% mixed vs. 83.2% isolated, GIVEN GROUND-TRUTH TIMING — see
  `pitch_refresh.py`'s own history below) that was explicitly flagged as
  untested beyond Gold. Real-tested across the full 9-song roster (BATB,
  Tarzan, Little Mermaid, Ordinary Day, Gold, plus Pink Pony Club/Heroes/
  OLP/Video Games — the last 4 with less-certain ground truth), both
  RMVPE and SwiftF0, via `note_detection.detect_notes()` run directly on
  each audio source (no given timing, matching pass 1's own from-scratch
  task) and scored against ground truth by best-time-overlap matching
  (no text/lyrics involved). **Running pass 1 itself on mixed audio is a
  clear net loss for both pitch sources**: coverage (did ANY detected
  note overlap a given GT note at all — a voicing/segmentation signal the
  Gold-only given-timing test couldn't see) drops 14-24pp on most songs
  for RMVPE, and even more severely for SwiftF0 (up to -55pp); pitch-
  class accuracy given ISOLATED's own segmentation is a small, inconsistent
  edge for RMVPE (roughly a wash) and a clear loss for SwiftF0. A hybrid
  that keeps isolated's segmentation/voicing but re-samples each note's
  pitch CLASS from a mixed-audio contour computed once over the whole
  track (reusing `pitch_refresh.compute_pitch_class_predictions`
  unmodified, never a tiny isolated clip) was also tried: coverage is
  preserved exactly by construction, but pitch-class accuracy nets out to
  a wash-to-slightly-negative for RMVPE (3 wins/4 losses/1 tie across the
  9 songs, one bad outlier at -5.1pp on Heroes) and a clear regression for
  SwiftF0 (losses up to -30pp). **Conclusion: pass 1 stays on isolated
  vocals for both pitch sources** — same "well-motivated but doesn't
  survive contact with real data" pattern as `--verify-placement`/
  `--zone-boundary-snap` above. Also surfaced, independent of this
  question: **Video Games' ground-truth `.txt` has a systematic pitch
  offset** (near-0%, below 1/12 random chance, pitch-class accuracy
  across every combination of audio source/pitch source/recipe tested) —
  a `truth.txt` the user initially pointed at as a possibly-cleaner
  ground truth turned out to be byte-identical to the same file, then was
  removed; the systematic offset itself is still unresolved and Video
  Games should stay excluded from any future pitch-accuracy comparison
  until it is. All investigation scripts kept in `scratchpad/` (not
  committed): `mixed_vs_isolated_pass1.py`, `hybrid_pitch_prototype.py`
  (both runnable with an optional `rmvpe`/`swiftf0` CLI arg).
- **GUI automatic mid-run lyrics-ambiguity prompt**: fully removed
  2026-08-17 (user's explicit request, not a data-quality finding like
  the other entries here). Was an opt-in checkbox ("Ask when lyrics are
  ambiguous") that paused the background pipeline thread mid-run and
  opened `LrcLibSearchDialog` on the main thread via the `self.after(0,
  ...)` + `threading.Event` pattern when LRCLIB search found more than
  one real candidate and nothing was pre-picked. Removed entirely:
  `config.PipelineOptions.lyrics_ambiguity_prompt`/
  `lyrics_disambiguation_callback`; `lyrics_lookup._fetch_from_lrclib`/
  `fetch_reference_lyrics`'s `on_ambiguous` parameter and the now-dead
  `_real_lrclib_candidates` helper that fed it; `gui.py`'s checkbox
  widget, `App._make_ambiguity_callback`, and `LrcLibSearchDialog`'s
  `initial_candidates` param (its only caller). The manual pre-run
  "Search Lyrics..." button/dialog and `pinned_lyrics`/`--lrclib-id`/
  `--lrc-file` are all unrelated and unaffected — this only removed the
  automatic mid-run popup.

## Shipped defaults / current config (as of 2026-08-15)

- `pitch_source="rmvpe"` (`config.DEFAULT_PITCH_SOURCE`, `--pitch-source
  {rmvpe,swiftf0}` on the CLI, same dropdown in the GUI) is the real
  pitch-source default — RMVPE's own pitch AND voicing decision,
  exclusively, no cross-check with any other source — reproducible,
  faster, and a real average +1.7pp accuracy win across the 4-song core
  set over the old pyin-primary + CREPE/RMVPE-cross-check ensemble
  (since fully removed — see "Removed / rejected approaches" above).
  `"swiftf0"` (lightweight CNN pitch detector, own native voicing
  decision) is the only other supported source.
- **`config.ENABLE_AMBIGUITY_KEY_TIEBREAK = True`** (`--ambiguity-key-
  tiebreak`/`--no-ambiguity-key-tiebreak`, `--ambiguity-margin-threshold`
  default `0.35`, GUI "Ambiguity key tie-break (pass 1)" checkbox under
  Advanced — the margin threshold is CLI-only, no existing GUI float-
  entry pattern to mirror), 2026-08-16. `pitch_ambiguity.py` (new
  module): RMVPE-only (no-op for `--pitch-source swiftf0`) refinement of
  pass 1's own note pitch-CLASS computation, applied as a real
  post-processing step after `detect_notes()`'s own final segmentation
  (never changes timing/note count/order). Two parts, validated and
  shipped together as a unit (not separately): (1) a note's pitch class
  is recomputed by SUMMING RMVPE's own raw per-frame 360-bin salience
  distribution (`rmvpe_onnx.RMVPE.predict`'s own 4th return value,
  previously computed and silently discarded by `_rmvpe_pitch` on every
  single inference call) across the note's own span and taking the
  argmax, instead of a per-frame-rounded confidence-weighted mode — same
  aggregation shape usp's own trained classifier uses, applied to
  RMVPE's own output instead of a separate model; (2) when the resulting
  top-1/top-2 candidates are genuinely close (`AMBIGUITY_MARGIN_
  THRESHOLD`, relative salience-mass margin), the tie is broken using
  the song's own detected key via the real published Krumhansl & Kessler
  (1982/1990) key-profile ratings (listener probe-tone study data,
  standard in MIR key-finding) — deliberately narrower than this
  project's own twice-rejected blanket key-correction (see "Removed /
  rejected approaches" above: a full global key-correction pass was
  built and fully removed for blindly snapping legitimate modal-mixture/
  borrowed notes; usp's own unconditional ±1-semitone `key_nudge`,
  vendored into `pitch_refresh.py`, is real but explicitly NOT universal
  — a measured regression on Tarzan). A confident, unambiguous note
  (RMVPE's own salience shows one clear peak) is NEVER touched by the
  key profile, no matter how out-of-key it looks — gating on genuine
  acoustic ambiguity rather than acting on every out-of-key note is what
  let this survive real-audio validation where the broader approaches
  didn't. Threaded through `note_detection._rmvpe_pitch`/`_rmvpe_source`
  via an optional `activation_out` side-channel dict (costs nothing
  extra — same single `rmvpe.predict()` call either way, never a second
  inference) and both `detect_notes()` call sites (`main.py`'s in-process
  path and `worker_process.py`'s cancellable-subprocess path).
  **Real-audio validated at full-pipeline scale** (pass 1's own SELF-
  DETECTED segmentation — the harder, riskier scenario, not a given-
  correct-timing shortcut) across a 12-song roster via a resumable
  scratch harness (`scratchpad/pitch_comparison/tiebreak_selfdetected.py`
  + `ambiguity_tiebreak_prototype.py`): a single FIXED threshold applied
  uniformly (not per-song-tuned) gives weighted pitch-class accuracy
  63.1% → 65.5% (+2.4pp) vs. the previously-shipped baseline, 9/12 songs
  improved (Jungle Book +7.7pp, BATB +6.4pp, Little Mermaid +4.6pp),
  2 modest regressions (Trixie Mattel - Gold −1.6pp, David Bowie - Magic
  Dance −1.4pp — real, known, accepted, not universal, same risk
  profile already accepted once for usp's own `key_nudge`). Threshold
  0.35 was the single best value found sweeping 0.10–1.00 on real
  audio — weighted accuracy peaks in the 0.30–0.50 range and clearly
  REVERSES past ~0.50 as the threshold widens toward "tie-break nearly
  everything," converging on the already-rejected blanket key-correction
  failure mode — confirms the ambiguity gate is doing real work, not
  coincidentally correlating with something else. Two closely related
  ideas from the SAME investigation were tried on the same real-audio
  harness and explicitly REJECTED — do not re-attempt without new
  evidence: tuning `ATTACK_TRIM_SEC`/`CONFIDENCE_FLOOR_PERCENTILE` for
  pass 1's own segmentation (real word-for-word churn analysis
  — `scratchpad/pitch_comparison/churn_check.py` — showed a "flat"
  aggregate was hiding real bidirectional regressions, net negative on
  Trixie Mattel - Gold specifically); integrating usp's own separately
  -trained ONNX pitch classifier directly (`usp_classifier_prototype.py`
  — wins Trixie Mattel - Gold, loses Great Big Sea - Ordinary Day, no
  consistent cross-song signal even in the easier given-correct-timing
  scenario). See `project_three_way_pitch_comparison` memory for the
  full investigation history.
- **`pitch_refresh.py`'s `isolate_vocals` default flipped to `True`
  (2026-08-15)**, reversing its original mixed-audio-by-default design —
  see the "Removed / rejected approaches" entry above for the full
  9-song real-data finding that motivated it (the single-song Gold result
  that justified mixed audio doesn't generalize: a coin flip for RMVPE,
  a clear regression for SwiftF0 on 8/9 songs). `--no-isolate-vocals`/
  unchecking the GUI box opts back into the old mixed-audio, no-CUDA-
  needed behavior. **This makes CUDA required by pitch_refresh.py's
  default invocation** (`separation.isolate_vocals` hardcodes Demucs to
  `-d cuda`) — previously the one fully-CPU-only tool in this project;
  `--no-isolate-vocals` is now the only way back to that footprint. Full
  design rationale lives in `pitch_refresh.py`'s own module docstring,
  not duplicated here.
- **`pitch_refresh.py` MusicXML support** (2026-08-16, `musicxml_pitch`
  default ON, `apply_mxl_pitch_reference(s)`): tried BEFORE audio-based
  pitch detection, same "MXL if possible, else fall back" priority as
  main.py's own MXL+LRC primary path — skips Demucs/audio pitch
  detection entirely when it succeeds. Built on `mxl_lrc_generator.
  load_mxl_vocal_words` (word-level, captures tied/slurred lyric-less
  continuation notes) rather than `musicxml_reference.load_vocal_notes`
  (flat single-note list, pass 4's own parser) — needed to see a word's
  REAL note count. Existing words (`realign.extract_words`) are aligned
  to MXL words by a whole-sequence lyric-text diff, then a per-song
  pitch-class calibration offset is established via the same shared
  logic pass 4 uses (extracted into `musicxml_reference.
  _calibrate_pitch_class`/`nearest_pitch_for_class`, now used by both).
  Once calibrated, a matched word whose existing single note the MXL
  reveals to be a multi-note melisma is SPLIT into that many notes
  (proportional to the MXL's own relative note durations, pitched to the
  calibrated class nearest the ORIGINAL note's own octave) — the one
  deliberate exception to this module's "never touches note count"
  invariant, explicitly authorized by the user for the MXL path only.
  Only the first of the new notes keeps the original word's own text;
  the rest are `config.MELISMA_CONTINUATION_TEXT` ("~") continuation
  notes (this project's existing melisma-tail convention, e.g.
  `lyric_alignment.py`'s own identical fallback — not left blank), so
  the word's own real text is completely unchanged either way. A word
  whose existing note count
  can't be safely split further (already > 1 syllable) or where MXL has
  FEWER notes than the existing word instead picks the position-
  proportional-nearest MXL syllable per existing note, same as an exact
  1:1 match, never changing note count — audio-based pitch detection's
  own `_pred_for_note`/`_weighted_mode_pitch` (already existed, per-note
  confidence-weighted-mode aggregation) is the one deciding "which pitch
  is correct" whenever a note can't be split, unchanged by this feature.
  Real-validated: Beauty and the Beast (100% calibration agreement over
  138 matches, 0 corrections needed — the existing SingStar ground truth
  was already correct once the MXL's own +2 semitone transposition is
  accounted for); Great Big Sea - Ordinary Day (94% agreement, 22 notes
  corrected, 9 real melisma words split into 9 extra notes) — confirmed
  via the real written output, not in-memory: GAP/BPM identical, full
  concatenated lyric text byte-identical before/after, note count
  increased by exactly 9, every split note's own text/timing/pitch
  inspected directly and correct. `--musicxml-reference`/`--musicxml-
  part` mirror the main pipeline's own flags; `--no-musicxml-pitch` opts
  out entirely.
- `ENABLE_MUSICXML_FORCE_CALIBRATION = True` (unconditional, no CLI/GUI
  off-switch as of 2026-08-17): when a user supplies (or auto-detects) a
  MusicXML reference and normal pass-4 calibration can't clear its
  confidence bar, apply the best available pitch-class offset anyway
  rather than skipping. Zero regressions across all 7 tested MXL-having
  songs; big wins (+19–22pp) on rough/character-vocal songs. Never
  touches octave or timing.
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
- `lyrics_lookup.effective_lrc_duration` (2026-08-15, user's explicit
  request, used everywhere an `LrcLibCandidate.duration` is compared
  against real audio length — `mxl_lrc_generator.select_lrc_candidate`'s
  filter/scoring/`duration_delta`, `lyrics_lookup._real_lrclib_candidates`
  /`_score_lrclib_candidate`): don't trust LRCLIB's own `duration` field
  blindly — cross-check it against the candidate's OWN synced lyrics. If
  the last line with real text (not just a timestamp) already occurs AT
  OR AFTER the claimed duration, the claimed value is internally
  inconsistent (lyrics can't end after the song does) and untrustworthy —
  use that last real lyric's own timestamp as the effective duration
  instead. Real-network validation across the whole `sandbox/` roster
  found several genuinely wrong real durations this now catches/corrects
  (Chicago's Queen Latifah candidate: reported 3.0s vs. a real last lyric
  at 181.9s; several Magic Dance and Les Misérables "Stars" candidates
  reporting durations 40-241s off their own real last-lyric timestamp).
  Confirmed no regression on both known-good validation candidates
  (Great Big Sea - Ordinary Day id 6210269: already-accurate duration,
  unaffected; David Bowie - Magic Dance: no real LRCLIB candidate passes
  the duration filter either before or after this fix, consistent with
  it being a known no-valid-candidate case).
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
- `BPM_WRITE_MULTIPLIER = 2`: written `#BPM` is 2x the detected tempo
  for finer beat-grid resolution (display/write-time only, not fed into
  `detect_notes()`'s own segmentation).
- Long-segment re-windowing (unconditional, no CLI/GUI off-switch as of
  2026-08-17): see "Long-segment rewindowing" under the `realign.py`
  section below — applies to both `realign.py` and `main.py`.
- Force-align known-text gaps and melisma-tail merging (the latter
  `main.py`-only) are both unconditional, no CLI/GUI off-switch as of
  2026-08-17: see "UltraStarKaraokeMaker-inspired improvements" below.
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
- **Online cover-art download** (`cover_fetch.py`, `--fetch-cover`/
  `--no-fetch-cover`, default ON, GUI checkbox), 2026-08-16: last-resort
  fallback when NEITHER a companion file NOR an embedded audio tag
  provided a cover. Adapted from UltraStarKaraokeMaker's own asset-fetch
  cascade (see `reference-uskmaker` memory) — same source order
  (MusicBrainz + Cover Art Archive, then iTunes, then Deezer), scoped
  down to cover image only (USKM's own cascade also enriches #YEAR/
  #GENRE and fetches a separate fanart.tv background; neither built
  here) and to the three sources needing no API key/account setup (its
  Last.fm/Discogs tiers are both gated behind a user-supplied env-var
  key — skipped to keep this zero-configuration, consistent with this
  project's other network lookups). Real-network-validated directly
  (multiple real songs; MusicBrainz+CAA and iTunes both confirmed
  answering correctly; a nonsense artist/title correctly returns `None`,
  never raises). Results are cached under `work_dir/extracted/` by the
  same check-then-fetch idiom `separation.isolate_vocals` uses, keyed by
  the resolved audio's own filename stem plus `config.COVER_TAG_SUFFIX`
  (renamed from `EMBEDDED_COVER_TAG_SUFFIX`, now shared by both this and
  `cover_extract.py`) — a re-run against the same `work_dir` never
  re-downloads. Runs in `main.py` only, after artist/title are resolved
  (needed for the query) — not wired into `realign.py`.
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
Continue/Cancel via the same `self.after(0, ...)` + `threading.Event`
thread-hop pattern `no_lrc_fallback_callback` also uses. An `ASR quality
retry` (re-transcribe with large-v3, see the `realign.py` section below)
fires before giving up, when `opts.batch` is set.

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
  re-exported from `realign.py` for backward compat) detects an LRC
  candidate whose repeat structure doesn't match ours — compares WORD
  occurrence counts of the most-repeated line's own distinctive content
  words (not exact-line-repeat counts, since a chorus is often split
  across several near-duplicate line variants that would each pass a
  naive per-line check individually), tolerance ±15%/min ±1. Still used
  as-is for `two_tier_time_calibration`'s tier-3 rescue gate (is this
  fundamentally the WRONG recording, e.g. Heroes' choral cover — see
  that constant's own memory) — real case: a candidate with 9 extra
  chorus repeats, invisible in duration since it coincidentally matched
  within the existing tolerance.
- `lrc_timing.reconcile_line_structure` (2026-08-14, user's own design,
  replaces `check_repeat_structure`'s use as `prepare_lrc`'s upfront
  accept/reject gate — that function itself is unchanged, still used for
  the tier-3 rescue gate above): rather than rejecting a whole candidate
  outright over a differing repeat count (the "I'm Afraid of Americans"
  case above), walks our own lines and the candidate's own lines forward
  together — two cursors, each only ever advancing — matching each pair
  by exact normalized text; on a mismatch, looks up to 8 lines ahead on
  BOTH sides for the next real match (whichever side needs the smaller
  skip wins, ties drop the LRC side) and drops the skipped lines on
  whichever side had the extras. `prepare_lrc` then uses the reconciled,
  filtered candidate lines (not the raw ones) for both time calibration
  and per-word line assignment. Still declines the candidate outright
  (same fallback as the old rejection) if under half of our own lines
  find no match at all — a genuinely different recording, not just a
  differing repeat count. Real validation: reproduces the "Americans"
  case's 3-extra-repeat structure exactly (all of our own lines matched,
  the 3 extra candidate lines dropped, landing back in sync at the next
  shared line) — see `test_dry_run.py`'s own `reconcile_line_structure`
  tests.

  **REAL validation, 2026-08-15** (scanned every `sandbox/` song with an
  existing `.txt` against its real, live-network-searched LRCLIB
  candidate): 4 real candidates would have tripped the old
  `check_repeat_structure` reject (Absolute Beginners, "I'm Afraid of
  Americans", Magic Dance, Video Games). Found and fixed a real
  pre-existing bug this surfaced: `lrc_timing.py`'s and
  `lyrics_lookup.py`'s own `_normalize()` only kept STRAIGHT apostrophes
  (regex `[^a-z0-9']`), silently deleting a real existing file's CURLY
  ones (e.g. "Johnny’s") — desyncing an otherwise byte-identical line
  ("johnnys" vs "johnny's") and blocking EVERY apostrophe-containing line
  from matching at all. `realign.py`/`mxl_lrc_generator.py`'s own
  `_normalize` already had the curly→straight fix; the other two module's
  copies had drifted out of sync with it — now fixed identically in all
  four. After the fix: Absolute Beginners (34/47 matched — the candidate
  simply omits the "Bababauuu" ad-lib bridge, correctly left unmatched
  rather than force-fit) and Americans (47/64 matched — our own
  recording's outro genuinely differs from the candidate's own extended
  outro, correctly left unmatched) now reconcile and get real LRC timing
  instead of zero. Magic Dance still correctly declines — inspected
  directly: NOT a repeat-count difference at all, the candidate is a
  longer arrangement with a whole extra bridge section and a different
  backing-vocal-echo line-splitting convention; declining is correct, not
  a gap, and no `max_skip` widening (tried up to 40) changes that.

  **Video Games initially also looked like a decline, but wasn't a real
  gap** (user caught this, 2026-08-15) — two separate causes, neither in
  `reconcile_line_structure` itself: (1) the validation script's own
  audio-duration estimate (last syllable's end time) missed ~22s of real
  trailing outro, pushing the correct candidate outside
  `MXL_LRC_DURATION_TOLERANCE_SEC` in the SCRIPT only, not the real
  pipeline (fixed: use the real audio file's own duration); (2) a real,
  separate bug in `mxl_lrc_generator.select_lrc_candidate`'s own ranking
  — see its own docstring/memory below. Once both were fixed,
  `check_repeat_structure` never even rejected the correct candidate
  (id 2958984) — it was only ever hitting the WRONG one. Against the
  correct candidate, reconciliation matched 33/59 (56%) initially. Four
  more real bugs were found and fixed chasing this ONE song end-to-end
  (2026-08-15, all user-directed — "run a real end-to-end generation on
  Video Games to confirm" surfaced every one of these; none would have
  been found by unit-testing `reconcile_line_structure` in isolation):

  1. **Whitespace-flattened comparison.** Our own existing file had a
     real typo ("livin'if" for "livin' if", a missing space) that made
     an otherwise byte-identical line fail to match at all (different
     word-token boundaries), desyncing the whole walk from that point on
     with no recovery within `max_skip`. Fixed by comparing lines with
     ALL inter-word whitespace stripped (`our_flat`/`lrc_flat`, not
     `our_norm`/`lrc_norm`'s single-space-joined form) — still an EXACT
     character-sequence match, just immune to a stray/missing space at a
     word boundary.
  2. **Joint, merge-aware recovery search.** The original recovery
     logic searched each axis independently (lrc_skip OR our_skip,
     never both) and only checked for a merge at the CURRENT, untouched
     position — real case: a stretch where BOTH sides have genuinely
     different content right before the third chorus repeat (our own
     file's own repeated ad-lib "But baby, now you do" / "Mm." vs the
     candidate's differently-worded "...ooh" / "...mmm") needs BOTH
     cursors to move together, and the real resync point was a MERGE
     reachable only a few lines further in — a single-axis search could
     never reach it, and worse, blindly dropping one side while holding
     the other still walked straight past the (perfectly matchable)
     chorus as collateral damage. Fixed: one unified search over (p, q)
     offsets, increasing total distance first, checking BOTH a plain
     match and a merge (either direction) at every candidate position,
     within `max_skip` on each axis independently.
  3. **Fuzzy character-level tolerance for elided-letter contractions**
     (user's own explicit request — "Ev'rything" for "Everything",
     "livin'" for "living"): `_flat_fuzzy_equal` (SequenceMatcher ratio
     ≥ 0.85, exact match always checked first) as a fallback everywhere
     lines are compared, including inside `_consume_as_merge`'s own
     per-piece slice check (length-preserving, since both real examples
     are same-length substitutions — an apostrophe standing in for the
     elided letter, not an insertion/deletion). Deliberately NOT the
     same risk class as this project's other rejected fuzzy-matching
     attempts (`--verify-placement`, etc.): those widened a SEARCH across
     many competing positions; this only ever relaxes the equality test
     between two positions the exact-structure walk already identified
     as the ones to compare.
  4. **`assign_words_to_lines` bypass** (the deepest one — found only by
     running the REAL pipeline end-to-end, not by testing reconciliation
     in isolation): even with the three fixes above making
     `reconcile_line_structure` itself correct (56/59, 95%), the actual
     `[REALIGNED].txt` output STILL misplaced the third chorus repeat —
     confirmed against the raw ASR transcript (real occurrence at
     172.5s; written output at 140.3s). Root cause: `prepare_lrc` handed
     the reconciled lines to `assign_words_to_lines`, which re-derives
     word-to-line correspondence from scratch via its OWN independent
     whole-song WORD-level diff — no line-boundary information at all,
     and it disagreed with what `reconcile_line_structure` had already
     correctly resolved at the (more reliable) LINE level. Exactly the
     "repeated-phrase disambiguation" failure class this project has
     been burned by repeatedly (see Lessons learned), just one level
     downstream of where it was expected. Fixed architecturally, not by
     patching the diff: `reconcile_line_structure` now also returns
     `our_line_index` (which `our_lines[i]` each reconciled entry came
     from — it already knows this, it just wasn't being surfaced), and
     `prepare_lrc` (given a new `our_line_of_word` param, built by
     `_word_line_indices`) uses that DIRECTLY to build `word_lines`,
     completely bypassing `assign_words_to_lines` for the reconciled
     case rather than re-deriving a potentially-different answer.
     A word whose own line has no LRC counterpart now correctly gets
     `None` (skipped by both "seed" and "windowed" downstream) instead
     of silently inheriting the nearest PRECEDING matched line's index
     (assign_words_to_lines's own old fallback) — which was the direct
     mechanism putting words in the wrong time window. Confirmed fixed:
     re-ran end-to-end after this change — all three chorus repeats and
     all three "It's better than..." lines now land within ~0.1s of the
     raw ASR transcript.

  End state, real end-to-end run: 304/354 words matched directly to ASR
  (up from 244/354 with reconciliation alone, before the
  `assign_words_to_lines` bypass), longest unanchored run down to 9
  words (from 82). Re-scanned all of `sandbox/` after shipping all four
  fixes: zero regressions, several songs now at 100% (Chicago, Tarzan,
  Gold) or very high match (Video Games 95%, Under The Sea 90%, Gaston
  91%) that previously declined or matched partially.
- `reconcile_line_structure`'s FILLER-WORD tolerance (2026-08-15, user's
  own explicit request): `FILLER_WORDS`/`_strip_filler_flat` in
  `lrc_timing.py` — a short, deliberately conservative list (`ooh`,
  `ooo`, `oh`, `ohh`, `mmm`, `mm`, `yeah`, `and`, `but`) tried as a
  FALLBACK (never in place of the raw comparison) in both the plain-line
  match and the merge check (`_match_kind`): an LRC's own author and our
  existing file's own author often choose differently whether to write
  an ad-lib or a filler connector, which is a transcription CHOICE, not
  the line actually being different content — specifically motivated by
  realignment, where this disagreement is common between independently-
  authored sources. Guarded so two lines that are BOTH entirely filler
  words (but different ones) can never fuzzy-match via an empty-string
  tie once stripped. Real-network validation across the whole `sandbox/`
  roster (comparing reconciliation `match_ratio` with vs. without this
  tolerance, real LRC candidates, real audio durations): zero
  regressions anywhere, with genuine real wins — Chappell Roan - Pink
  Pony Club 96.7%→98.9%, The Little Mermaid - Under The Sea 90.1%→91.4%,
  Trixie Mattel - Video Games 94.9%→96.6%.
- `mxl_lrc_generator.select_lrc_candidate` ranking fixed (2026-08-15,
  real bug found via the Video Games investigation above): used to rank
  candidates by content-match ratio first, duration only as a tiebreaker
  that floating-point ratios essentially never reach — real case: the
  CORRECT candidate (Trixie Mattel's own "Video Games" cover, id
  2958984, real duration within 0.7s of ours) was passed over for Lana
  Del Rey's ORIGINAL (a different performer entirely, 14.5s off) purely
  because `difflib.SequenceMatcher`'s ratio happened to score the
  wrong-performer candidate higher — text-similarity alone can't be
  trusted to prefer the right PERFORMER over a same-titled original/cover
  by someone else. Now ranks by (1) `_artist_matches` (substring match
  after normalization, catches a YouTube "- Topic" suffix, "feat."
  credits, cast-recording listings, etc. in either direction) —
  DECISIVELY, any same-artist candidate beats any different-artist one
  regardless of ratio/duration — (2) duration proximity, (3) content
  ratio as the final tiebreaker. `artist` blank or no candidate
  resembling it at all (real case: Chicago, credited to individual cast
  members like "Marcia Lewis - Topic" rather than the show name we use as
  our own artist tag) falls through to ranking by duration then ratio
  among whatever's left, same as before this existed — this is
  deliberate, not a gap: the user's own words were "basically never
  allow the wrong artist unless there is literally nothing available for
  the correct artist."
- `lrc_timing.match_asr_to_lrc_lines` rewritten to a forward-only CURSOR
  (2026-08-15, user's explicit request after reporting a real failure on
  Chappell Roan - "Pink Pony Club", a song whose chorus repeats 3 full
  times): used by BOTH `realign.py` and `mxl_lrc_generator.py`'s primary
  MXL+LRC generation path to find, per LRC line, a real ASR anchor
  BEFORE trusting LRC line timestamps for calibration (`two_tier_time_
  calibration`). The OLD implementation ran one global, non-chronological
  `difflib.SequenceMatcher` over the WHOLE ASR word stream vs. the WHOLE
  LRC word stream (every line's words concatenated, all repeats
  included) — the same "repeated-phrase disambiguation" failure class
  documented above, just never fixed in this mechanism. Real confirmed
  failure: a genuine ~130s garbled/ad-lib stretch in the middle of the
  song got ZERO anchors under the global diff, and every line after that
  gap was then anchored ~134-135s TOO EARLY — matched against an earlier
  occurrence of the same repeated chorus instead of its own real, later
  one; `two_tier_time_calibration` then correctly refused to calibrate
  at all (confidence 0%), leaving the whole LRC candidate unused and
  producing the "only 16%/35% of words validated or got an anchor"
  warning even though the user's own LRC file was, by their own report,
  an exact match never more than ~1s off. Fixed with the same forward-
  only cursor principle as `reconcile_line_structure`/`assign_lrc_line_
  ids_sequentially`: walk LRC lines in order, search only a window of
  ASR words starting where the PREVIOUS line's own match left off — a
  repeated phrase later in the song can never be confused with an
  earlier occurrence, because the cursor has already advanced past it.
  The window is NOT fixed-size per line: it accumulates the word count
  of every skipped (no-match) line since the cursor last actually
  advanced (`pending_word_count`, capped at `MAX_PENDING_WORDS = 60`),
  so a real multi-line garbled stretch doesn't permanently strand the
  cursor once real content resumes. A real second bug was found DURING
  this same real-audio validation and fixed before shipping: an
  uncapped/too-permissive window let a single coincidentally-shared
  common word (e.g. "the") register as a false anchor once the window
  grew large — fixed with a minimum-match-quality gate
  (`MIN_MATCH_TOKEN_FRACTION = 0.5`: at least half of a line's own
  tokens, minimum 2 for anything longer than 1 word, must actually
  match before a candidate is trusted enough to advance the cursor).
  Real validation: the exact reported song now calibrates successfully
  (offset +1.0s, 50% agreement, 80/87 reconciled lines anchored — up
  from 0 confident calibration, 34/87 anchored) confirmed via a real
  end-to-end `realign.py` run; a broader real-data comparison (OLD vs
  NEW, using cached real ASR transcripts, both reconciled and raw-
  candidate-line inputs) found zero regressions elsewhere and one more
  real win (Trixie Mattel - "Video Games": calibration confidence
  52%→82-84%, matched-line count 21-23/51-57→51-57/51-57, i.e. every
  line). **This fix alone was NOT sufficient** — the user reported the
  SAME song still misplacing words after this shipped; see `realign.
  match_words_to_asr` below for the deeper, separate bug this surfaced.
- `realign.match_words_to_asr` rewritten to a forward-only CURSOR over
  REAL LINES (2026-08-15, found immediately after the `match_asr_to_lrc_
  lines` fix above, when the user reported the SAME Pink Pony Club file
  still misplacing words after that fix shipped): this is the function
  that places the BULK of individual word timings (not just LRC line
  anchors) for realign.py's whole "replace"/"seed"/GAP-check pipeline —
  it had the exact same "repeated-phrase disambiguation" vulnerability,
  just never fixed. Confirmed via a real per-word trace: the "I'm"
  starting each of two repeated "...Pink Pony Club" verses landed within
  ~0.5s of its true position (correct), but every word BETWEEN those two
  correct anchors — itself repeated content — was confidently matched
  104-136 SECONDS too early, to an earlier occurrence of the same
  repeated phrase; being marked "confident", those words were never
  handed to `interpolate_fallback`, which would otherwise have smoothly
  placed them.

  **Three real, escalating bugs were found and fixed while validating
  this rewrite against the SAME real file** (the project's own
  "iterative bugfix protocol" 3-strikes convention was hit here — after
  the third attempt still failed on the real data, the fix was
  redesigned around real line boundaries rather than tuned further):
  1. First attempt: fixed-size 6-word chunks with a forward cursor
     (mirroring `match_asr_to_lrc_lines`'s own shape) + a match-fraction
     gate. Broke a real, legitimate case immediately: the gate applied
     even to a FRESH (non-inflated) chunk rejected normal SPARSE ASR
     coverage (most real chunks only confidently transcribe a few of
     their own words — not a coincidence risk, just incomplete
     transcription). Fixed: the fraction gate only applies once the
     search window has actually grown from accumulated skipped chunks,
     not on a chunk's own first attempt.
  2. Second: a borderline chunk (exactly meeting the fraction gate) had
     its few real matched tokens SCATTERED across nearly the entire
     width of a large accumulated window — advancing the cursor to that
     match's own raw opcode boundary jumped it all the way to the very
     END of the ASR stream, permanently stranding every later chunk for
     the REST OF THE SONG (an empty search window forever after). Fixed
     with two changes: (a) a span guard rejecting a match whose tokens
     are scattered further than the actual EVIDENCE found justifies
     (`matched_token_count`-relative, not chunk-size-relative); (b) the
     cursor only ever advances to just past the last ACTUALLY CONFIRMED
     match position, never a raw (possibly much further) opcode
     boundary — a `"replace"` block's own span is just wherever
     `difflib` decided the mismatched region ends, not a real match.
  3. Third: even with both guards above, the fixed-6-word-chunk version
     STILL occasionally mismatched the same real repeated-chorus
     stretch (wrong-but-plausible-looking deltas of +8 to +36s) —
     traced to the root cause: an ARBITRARY chunk boundary can slice a
     real line in half, leaving neither half enough distinctive content
     to reliably self-disambiguate against a nearby repeat. Fixed by
     abandoning fixed-size chunks entirely: `match_words_to_asr` gained
     an optional `line_of_word` param (from `_word_line_indices`, real
     line boundaries the caller already has) and now chunks by REAL
     LYRIC LINES when given — the same unit `reconcile_line_structure`/
     `match_asr_to_lrc_lines` already use successfully for this exact
     failure class. Falls back to the old fixed-size chunking only when
     `line_of_word` isn't given (kept for callers/tests with no entries
     to derive real lines from). This was the change that actually
     fixed the real reported case cleanly, confirming the two-tier
     "match_asr_to_lrc_lines then match_words_to_asr, both by real
     line" pattern is the right general answer to this failure class,
     not just a per-mechanism workaround.

  Real end-to-end validation on the exact reported song after all three
  fixes: GAP-check agreement 49%→96%, and the previously-broken
  repeated-chorus stretch (deltas of -104s to -136s) now lands within
  0.1-0.8s, with zero warnings on the final run. A broader real-data
  check against two other real songs' cached ASR transcripts (Chicago,
  Trixie Mattel - "Video Games") found high confident-match rates
  (93.8%, 88.7%) with median deltas near zero and no catastrophic
  outliers, confirming this generalizes beyond the one reported song.

  **A FOURTH bug in this same function was found immediately after
  shipping the above**, from a real user report on a NEW song (Our Lady
  Peace - "Somewhere Out There", 2026-08-15) with several lines
  repeating BACK-TO-BACK 3-4 times in a row (denser repetition than Pink
  Pony Club's, no bridge/skip needed to trigger it): even a completely
  FRESH chunk's own base search window (multiplier 3 + slack 10 -- e.g.
  ~25 ASR words for a 5-word line) is ALREADY wide enough to reach past
  the immediately-following correct occurrence into a LATER repeat of
  the same line, with no prior skip needed at all -- the match-quality/
  span guards don't catch this, since the wrong-but-later match can
  itself be perfectly clean and tightly clustered; `difflib` doesn't
  inherently prefer the nearest valid match over any other high-quality
  one it finds elsewhere in the window. Fixed with a genuinely TIGHT
  window (barely more than the chunk's own size) tried ALONGSIDE the
  normal wide one; tight wins whenever it covers at least HALF the
  chunk's own tokens (`MIN_MATCH_TOKEN_FRACTION`, the same bar already
  used elsewhere), regardless of what wide separately finds. Two
  stricter/looser thresholds were tried first and rejected, both against
  real data from two different real songs: requiring a FULLY complete
  tight match was too strict (Our Lady Peace's own correct nearby match
  was missing 1-2 of 5 words to real ASR noise, and still lost to a
  wrong, farther-away wide match); letting ANY passing tight match win
  outright (even a single coincidentally-found word) was too loose --
  regressed Chicago, where the wide window's own fuller context found a
  better answer than the tight window's sparse one could. Real
  validation with the final (half-or-more) threshold: Our Lady Peace's
  GAP-check agreement 84%→93%, `validate` strategy succeeds outright
  (zero warnings, no fallback to `"seed"` needed) where it previously
  warned and fell back; Chicago and Video Games both confirmed back to
  their own original (pre-regression) confident-match rates.
- Force-align known-text gaps and long-segment re-windowing (see below)
  are both unconditional. `retry_low_quality_asr` (see below) defaults
  ON, with a real CLI/GUI off-switch (it's a real cost tradeoff, unlike
  the two above).
- Strategy: `"validate"` (DEFAULT as of 2026-08-15, user's explicit
  request — previously an explicit non-default `--strategy validate`
  option; `"replace"` was the default before this) vs. `"replace"`
  (`--strategy replace`, GUI-selectable). `"validate"` only ever trusts
  ASR's own start when it's already CONFIRMED close to the (GAP-
  corrected) original — a real safety net `"replace"` doesn't have, and
  the reason `"validate"` exists at all — but, once confirmed, uses
  ASR's own start value (not the original's), keeping only the word's
  own original LENGTH. Automatically falls back to whole-song ASR-
  primary matching (the same mechanism `"replace"` uses) when too few
  words validate — see `realign_song_validate`'s own `MXL_LRC_MIN_ASR_
  PLACEMENT_RATE` fallback. This fallback (already shipped, not new) is
  what makes `"validate"` safe as the default even on a file whose own
  timing turns out not to be trustworthy at all.

  **`validate_words_against_asr` real bug fixed via real hand-timed
  ground truth** (2026-08-15, user's own `truth.txt` for Our Lady Peace -
  "Somewhere Out There"): the ORIGINAL version kept the GAP-corrected
  ORIGINAL start EXACTLY once confirmed, discarding ASR's own
  already-known-close reading entirely — reasonable-sounding ("don't fix
  what isn't broken"), but wrong in practice: when the original file's
  own timing carried a small systematic bias (~0.12s here, comfortably
  inside the default 0.3s tolerance), every validated word inherited
  that bias, and so did every INTERPOLATED word between validated
  anchors (interpolation is relative to them). Real comparison against
  `truth.txt`: only 27.6%/51.2% of words landed within 100ms/150ms
  under the old logic — WORSE than `--strategy replace` on the identical
  audio (51.9%/74.2%), the opposite of what `"validate"` is supposed to
  achieve. Fixed by using ASR's own start (not the original's) once
  validated — user's own proposed design, explicitly confirmed NOT
  equivalent to `"replace"` before implementing: `"replace"` trusts any
  confident ASR match unconditionally; this still requires ASR to
  already agree with the original within tolerance FIRST, so a real
  mismatch (wrong occurrence of a repeated phrase, hallucination, etc.)
  is still correctly rejected and left for `interpolate_fallback`, same
  guardrail as before. After the fix, re-compared against the same
  `truth.txt`: 62.9%/88.7% within 100ms/150ms — beats BOTH the old
  `"validate"` AND `"replace"` at every meaningful tolerance, combining
  ASR's own precision with `"validate"`'s safety net against wild
  mismatches.

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

### Long-segment rewindowing (unconditional, no CLI/GUI off-switch as of 2026-08-17)

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
zero regressions before shipping unconditionally everywhere (2026-08-17:
the `--rewindow-long-segments`/`--no-rewindow-long-segments` off-switch
that used to exist in both `main.py` and `realign.py` is gone — this
much validation earned it core-behavior status, not just a default); a
low-confidence
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

1. **Force-align known gaps — shipped, unconditional.**
   `transcription.force_align_words_in_window` (real wav2vec2 CTC
   forced alignment of KNOWN text into a bounded audio window), adapted
   from USKMaker's own `realign_gap_windows`. Wired into both
   `realign.py` (`_force_align_unconfident_runs`) and `main.py`'s
   standard fallback path (`lyrics_lookup.
   recover_dropped_reference_words`). No CLI/GUI off-switch as of
   2026-08-17 (was `--force-align-gaps`/`--no-force-align-gaps`,
   `config.FORCE_ALIGN_GAPS`) — this much validation earned it
   core-behavior status.
2. **Note over-segmentation — investigated, NOT shipped.** Confirmed
   real: pass-1's raw note count on Stars was ~2.8x the true (OMR)
   count, and untexted `~` continuation-note rate varies wildly by song
   (0-33%) under our pitch-first architecture vs. USKMaker's much lower
   rate (forced-alignment-first, pitch extracted per-syllable
   afterward). See `NOTE_MERGE_SEMITONES` in Shipped Defaults above for
   why the obvious fix (widen the merge threshold) was rejected.
3. **Melisma-tail merge — shipped, unconditional in `main.py` only.**
   `usdx_writer._merge_connected_melisma_tails`: a beat-adjacent,
   exact-same-pitch `~` continuation note immediately following a real
   syllable is redundant and gets folded into the preceding note
   (chains across several connected same-pitch `~` notes; never crosses
   a LineBreak). Runs at the INTEGER BEAT level (post-quantization), not
   float-seconds, since quantization itself can change what counts as
   adjacent. Deliberately NOT applied in `realign.py` — its own contract
   (never add/remove/reorder a note) would be violated by deleting a
   note. No CLI/GUI off-switch as of 2026-08-17 (was
   `--merge-connected-melisma`/`--no-merge-connected-melisma`,
   `config.MERGE_CONNECTED_MELISMA_TAILS`).

All three, plus `retry_low_quality_asr`, validated together (not just in
isolation) across 6 real gen+realign runs — zero regressions.

## Shared cross-module functions (re-test every listed caller when touched)

Logic genuinely identical across modules lives in ONE place instead of
being re-derived (and re-buggy) per caller. When editing any function
below, re-run/re-validate every listed caller, not just the one you were
working in — a fix or regression in the shared function silently affects
all of them.

- **`lrc_timing.find_cursor_window_match`** (added 2026-08-14): the
  forward-only cursor/tight-vs-wide window search + its two quality
  guards (min-match-count floor, span guard) used by every "find these
  target tokens somewhere after `cursor` in a haystack word stream"
  mechanism. Callers: `lrc_timing.match_asr_to_lrc_lines` (one anchor per
  LRC line) and `realign.match_words_to_asr` (every individual existing
  word's own position, plus a fuzzy-ratio `"replace"`-block fallback for
  ASR mishearing). Deliberately does NOT own opcode interpretation
  (marking positions, advancing the caller's own cursor) — that stays
  per-caller since the two callers want genuinely different things from
  a match. The `lenient_min_matches` param exists because the two
  callers need genuinely DIFFERENT floors for a non-strict (tight/fresh)
  window match, discovered the same day this was extracted:
  `match_asr_to_lrc_lines`'s own `mall_no_recovery` regression test needs
  2 (a lone stray "the" must not validate the 2-word target "the end");
  `match_words_to_asr`'s own `sla_confident` regression test needs 1 (a
  single real match is sometimes the ONLY confident evidence available
  at all, e.g. a chunk where only 1 of 4 words was ever transcribed, and
  must still anchor). Don't assume one universal floor is correct if you
  touch this — check both callers' own regression tests.
  Re-test: `test_dry_run.py`'s `match_asr_to_lrc_lines`/
  `two_tier_time_calibration` block AND its `realign: alignment-only
  mode` block (esp. the Pink Pony Club / Our Lady Peace / `sla_confident`
  regression tests); real-audio sanity check via
  `scan_mwta_regression.py`-style replay of a cached `[DEBUG LOG].txt`'s
  `RAW ASR TRANSCRIPT` section for at least Chicago, Video Games, and
  Our Lady Peace.
- **`text_normalize.normalize_word`** (added 2026-08-14): lowercase +
  curly-to-straight-apostrophe fold + strip everything except
  letters/digits/apostrophes. Was 6 independent per-module copies
  (`lyrics_lookup.py`, `lrc_timing.py`, `mxl_lrc_generator.py`,
  `realign.py`, `musicxml_reference.py`, `verify_existing_song.py`)
  before consolidation; the curly-apostrophe fix (see the 2026-08-15
  "reconcile_line_structure" entry below) had already been independently
  applied to the first 4 and had silently drifted out of sync with the
  last 2, which still carried the original bug (a real hand-authored
  file's curly `'` desyncing against a straight-ASCII `'` on the other
  side of a comparison) until this consolidation folded them onto the
  same fixed implementation too. `verification.py`'s OWN `_normalize`
  (strips only leading/trailing punctuation, keeps internal characters)
  is deliberately NOT part of this — genuinely different semantics for a
  genuinely different comparison, not another drifted copy.
  Re-test: `test_dry_run.py` in full (this function is exercised
  transitively by nearly every text-matching test in the suite); for the
  2 previously-unfixed callers specifically, a targeted real-lyric check
  (parse a real sandbox `.txt` with a known curly apostrophe, e.g. David
  Bowie - Absolute Beginners' "I’ve", and confirm it now normalizes
  identically to its straight-apostrophe equivalent) rather than a full
  pipeline re-run.
- **`lrc_timing.lrc_line_window`** (added 2026-08-14): trivial
  `[line's own start, next line's own start)` helper (last line gets a
  flat +5.0s fallback width) — was 3 byte-identical copies
  (`mxl_lrc_generator.place_words_via_asr`, `realign.
  match_words_to_asr_windowed`, and an inlined copy inside
  `reconcile_line_structure`'s `"lrc_merge"` branch) before
  consolidation. Low-risk (pure arithmetic, no judgment calls), but kept
  in this list since it's still a real 3x-duplicated pattern.
  Re-test: `test_dry_run.py` in full — no dedicated test targets this
  helper in isolation, it's exercised transitively through all 3
  callers' own existing test coverage.
- **`lrc_timing.match_block_to_candidates`** (added 2026-08-14): given a
  target block's own normalized words and a caller-bounded candidate
  `Word` list, runs ONE whole-block `difflib.SequenceMatcher` and returns
  `{target_local_index: matched Word}` — an "equal" opcode matches
  directly (gated on the candidate's own confidence); a "replace" opcode
  where the TARGET side is exactly one token tries every candidate token
  in that slice for the best fuzzy-ratio match (ASR mishearing a word,
  e.g. "favors" transcribed as "favorites"), gated on both the fuzzy
  ratio and the candidate's confidence. Callers: `mxl_lrc_generator.
  place_words_via_asr`'s own Pass 1 and `realign.
  match_words_to_asr_windowed` — both had independently implemented this
  exact opcode-walk (confirmed by `match_words_to_asr_windowed`'s own
  docstring: "mirroring mxl_lrc_generator.place_words_via_asr's Pass 1
  exactly"). Deliberately NOT shared with `realign.match_words_to_asr`'s
  own `_apply_match` — that one is built on `find_cursor_window_match`
  (REVERSED SequenceMatcher argument order: candidate first, target
  second) and returns a candidate-side offset for its own forward
  cursor, neither of which apply to these 2 independent, non-cursored
  callers. (A third former caller, `realign.rematch_local_gaps`, was
  removed 2026-08-15 as a net regression — see "Removed / rejected
  approaches" above.)
  Re-test: `test_dry_run.py`'s `place_words_via_asr` block (esp. the
  "favors"~"favorites" fuzzy-mishearing and next-line-spillover cases)
  and `match_words_to_asr_windowed`'s own block.
- **`lrc_timing.words_in_time_window`** (added 2026-08-14): trivial
  `[t0 - slack, t1 + slack]` ASR-word filter — was 2 copies (same 2
  callers as `match_block_to_candidates` above, since each builds its
  own `asr_in_window` right before calling it). Purely mechanical, kept
  as its own function so the slop convention (currently a flat 0.5s
  default) can't drift between callers unnoticed.
  Re-test: `test_dry_run.py` in full — exercised transitively through
  both callers' own existing test coverage, same as `lrc_line_window`.

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
