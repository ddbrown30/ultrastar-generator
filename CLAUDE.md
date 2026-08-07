# ultrastar_generator

Generates UltraStar Deluxe `.txt` karaoke song files from a raw audio
file (mp3/ogg/oga): isolates vocals, detects sung notes from the audio
itself, transcribes lyrics, fits the lyrics onto the detected notes, and
writes a spec-compliant `.txt`.

Full narrative history of every bug found and fixed is in `README.md`
(the "v1 note" through "v9 note" callouts) — read that before assuming
something is broken; it may be a known, already-worked-through issue.
Run `python test_dry_run.py` before and after any change — it's a
synthetic-data regression suite (no audio/models needed) covering every
bug fixed so far, and it's fast.

## Architecture: pitch/timing-first, lyrics-second

This is the core design principle and the source of most past bugs when
violated:

1. **Pass 1** (`note_detection.py`): detects notes (start, end, pitch)
   from the isolated vocal audio ALONE. No knowledge of lyrics. Must be
   internally non-overlapping (`_ensure_nonoverlapping` is a hard,
   explicit guarantee at the end of `detect_notes()` — if it ever has to
   fix something, that's a real bug, not routine).
2. **Transcription** (`transcription.py`): WhisperX (preferred, forced
   alignment) or faster-whisper, for lyric text + rough word timing.
3. **Lyrics correction + phrasing** (`lyrics_lookup.py`): fetches
   lyrics.ovh, does whole-sequence alignment (difflib, WER-style) against
   ASR words to fix mistranscriptions, and tags each word with a
   reference *line id* — every line break in the source lyrics becomes a
   forced `-` break in the output (`phrasing.py`), overriding gap-based
   heuristics when available.
4. **Pass 2** (`lyric_alignment.py`): fits words onto pass-1's notes.
   **Never changes note timing/pitch for words that got real notes** —
   only assigns text. Only fallback words (zero notes in their zone)
   get a synthesized note, and even then it borrows the nearest existing
   pass-1 note's pitch rather than re-analyzing audio fresh (see "Lessons
   learned" below for why).
5. Multi-word **matched reference lines** get their notes distributed
   *proportionally by syllable count*, not by trusting each interior
   word's own ASR timestamp — individual in-line ASR timing turned out to
   be unreliable enough that one bad timestamp could swallow a whole
   passage into one word's melisma.
6. Non-overlap is enforced twice more: `postprocess.enforce_monotonic`
   (seconds-level, preserves *given* word order — does NOT sort by
   timestamp, that was itself a past bug) and `usdx_writer.py`
   (integer-beat level — the space the bug actually showed up in once,
   since sub-beat gaps can collide after quantization).

## Diagnostics (use these before speculating about a bug)

- **Pass-1 debug file**: every run writes
  `<Artist> - <Title> [PASS1 DEBUG].txt` by default (`--no-pass1-debug`
  to skip) — same notes/timing as the real output, but each note's text
  is its own note name (e.g. "G#3") instead of a lyric. Load it in the
  UltraStar editor to check pass 1 in total isolation from lyric-fitting.
  **Always diff this against the real output first** when something
  looks wrong — it immediately tells you whether the bug is in
  `note_detection.py` or in `lyric_alignment.py`/`lyrics_lookup.py`.
- **Console output**: pass 1 logs frame/voicing stats, merge-pass note
  counts, spike-removal counts. Pass 2 logs how many words matched notes
  directly vs. needed a fallback (and lists them — a long fallback list
  means pass 1 is under-detecting, not that pass 2 is broken). The
  lyrics step logs every correction it made, e.g. `"is" -> "his" (at
  143.60s)`, and how many reference lines it found (zero means lookup
  silently failed for that title and it fell back to gap-based phrasing).
- `--quiet` suppresses the verbose logging if it's in the way.

## Lessons learned (don't reintroduce these)

- **Never run pYIN on a tiny, isolated audio clip.** It needs real
  context to be accurate. This caused two separate bugs: bad pitches
  when pass 1 originally analyzed per-word clips (fixed by analyzing the
  whole track once), and later a fabricated note when the pass-2
  fallback path ran a fresh pYIN call on a ~0.1s word clip (fixed by
  borrowing the nearest pass-1 note's pitch instead).
- **Don't trust individual ASR word timestamps for fine-grained
  boundaries**, even with WhisperX forced alignment. Reliable enough for
  coarse anchoring (which line/word region something belongs to), not
  reliable enough to trust as an exact boundary between two adjacent
  words — one bad interior timestamp can swallow a large chunk of
  otherwise-correct notes into the wrong word.
- **Don't sort by timestamp as a "harmless" cleanup pass.** It isn't —
  if anything upstream ever produces a note whose timestamp doesn't
  match its true reading-order position (e.g. a fallback note), sorting
  by time scrambles word order. Trust the given order; only push
  overlaps forward.
- **pYIN's voicing decision reflects periodicity, not loudness.**
  Digitally "silent" audio can still read as confidently voiced (noise,
  resampling artifacts). Always gate on energy (RMS) separately —
  and use BOTH a relative threshold (vs. the track's own loud sections)
  AND an absolute dBFS floor, since a purely relative threshold does
  nothing on a stretch that's uniformly quiet with no loud reference to
  compare against.
- **A short note that jumps far from both neighbors, whose neighbors
  match each other, is very likely a glitch, not music.** This is
  handled (`_remove_pitch_spikes`), but if new artifacts of this shape
  show up, check whether the thresholds (`--spike-max-duration`,
  `--spike-jump-semitones`) need tuning before assuming it's a new bug
  class.
- Every merge/threshold constant that controls "are these two notes
  really the same note" needs a **cap on total drift across a whole
  chain**, not just a per-step comparison — chain-merging stepwise
  melodic motion into one flattened note happened once already
  (`_merge_similar_adjacent`'s `group_min`/`group_max` tracking exists
  specifically to prevent this).
- **An onset with no pitch change still needs a way to split, or two
  legitimately re-attacked same-pitch notes merge into one.** Real bug:
  "fall as Lucifer" all sung on one held pitch merged into a single
  ~1.9s note, even though real per-syllable onsets AND real RMS dips
  were confirmed present in the audio at each word boundary — the
  segmenter simply never had a path to split on onset alone (an earlier
  fix deliberately disabled that, to stop consonant transients *inside*
  one note from causing spurious splits). Fixed with a middle ground:
  only a STRONG onset (top percentile of onset strength the track
  actually has) can split same-pitch audio, and only once the
  in-progress note has run long enough that the onset can't be its own
  attack (`config.REARTICULATION_STRENGTH_PERCENTILE` /
  `MIN_DURATION_BEFORE_REARTICULATION_SEC`). The resulting note is
  tagged `protected_start=True` so the very next merge pass (which would
  otherwise see "same pitch, ~0 gap" and silently re-merge it) leaves it
  alone.
- **A note's end time must come from its last INCLUDED frame, not from
  the frame index one past it.** Real, pre-existing bug found while
  debugging the above: `raw_notes` stores an exclusive `end_frame`
  (frames `[start_frame, end_frame)` belong to the note), but the
  original code computed `end_t` from `times[end_frame]` — the START of
  the frame AFTER the note, not the end of the note's own last frame.
  That made every adjacent pair of raw segments overlap by exactly one
  `frame_dur`, silently "fixed" by `_ensure_nonoverlapping` (with a
  warning) on literally every run — 49 times on one real ~200-note song.
  That guard's docstring calls this "shouldn't be possible by
  construction, and if it ever fires, that's a real bug" — it was right,
  and nobody had looked. Fixed by using `times[end_frame - 1]` instead.
  Worth remembering: a warning that fires constantly stops reading as a
  warning — if `_ensure_nonoverlapping` (or any similar hard-guarantee
  check) starts firing routinely again, don't assume it's "just how
  many overlaps happen"; go find the off-by-one.
- **Demucs separation is not bit-reproducible run to run, and for a
  tempo-ambiguous song that's enough to flip the detected BPM.**
  Confirmed in practice: two separations of the exact same input file
  (`Les Misérables - Stars.ogg`) produced same-size but different-checksum
  `vocals.wav` files (almost certainly CUDA/cuDNN non-deterministic
  algorithm selection at inference, not any intentional test-time
  randomness — `--shifts` isn't passed, so that's not it). That tiny
  waveform difference was enough for `librosa.beat.beat_track` to detect
  105.47 BPM in one run and 109.96 in the other for the same song —
  `detect_bpm` itself is deterministic given the same audio (confirmed:
  identical input, 3 repeated calls, identical result), so the whole
  output's beat grid silently shifts based on which Demucs run you
  happened to get, not on anything about the pipeline logic. Not fixed at
  the Demucs level (would mean fighting CUDA determinism settings across a
  subprocess boundary, for uncertain benefit) — instead, `main.py`'s
  `work_dir` now defaults to `<audio file's directory>/.ultrastar_work`
  (not `<output_dir>/.ultrastar_work`), so separation is cached and reused
  by its OWN directory regardless of `--output-dir` — e.g. comparing
  `--whisper-model` choices into different `--output-dir`s no longer
  triggers a fresh, independently-nondeterministic separation each time.
  If you ever see beat numbers that don't seem to line up with a previous
  run's reference notes for the same song, check `#BPM` first before
  assuming a real regression.
- **Pass 1's own CREPE inference is ALSO not bit-reproducible run to
  run, with a bigger blast radius than the Demucs case above.**
  Confirmed directly: calling `detect_notes()` twice in the same process
  on the exact same cached audio array produced 295 vs 298 notes, 283 of
  295 compared notes differing in timing/pitch — same underlying cause
  (CUDA/cuDNN non-deterministic algorithm selection), but this time
  perturbing the note sequence itself rather than just one tempo value,
  so it can silently shift which pitch/timing you get for a given song
  from run to run even with everything else (audio, code, flags) held
  identical. Partially mitigated (not fully fixed) in
  `note_detection.py`'s `_crepe_pitch`: forces
  `torch.use_deterministic_algorithms(True, warn_only=True)` and
  `cudnn.deterministic=True`/`benchmark=False`, scoped to just that call
  (restored after, via try/finally) so it can't affect Demucs — a
  separate subprocess anyway — or WhisperX/pyannote, which run later
  in-process and haven't been vetted for full deterministic-op coverage.
  `main.py` also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` at import time
  (required before any CUDA context init for cuBLAS ops to honor
  determinism). Measured: negligible performance cost (+1.2%, 156.6s ->
  158.5s for a full-track `detect_notes()` call), and no accuracy
  concern (a deterministic algorithm computes the same math via a fixed
  op order, not a different/worse one) — but the fix is incomplete:
  even with both settings, a repeat test still showed 221 of 282 notes
  differing between two back-to-back calls. The remaining instability
  doesn't produce any "no deterministic implementation" warning, so it's
  likely GPU floating-point non-associativity somewhere inside
  torchcrepe's own inference that these global flags don't reach.
  Decided (given the fix is free and strictly helps): keep it, don't
  chase the remainder further. **This means pass-1 pitch/timing output
  for a given song is still not guaranteed identical between runs** —
  same caveat as the Demucs/BPM case: check whether you're comparing
  runs before treating a small pitch/timing difference as a regression.

## Open threads / where we left off

Done:

1. **`key_correction.py` now uses `music21`** (its implementation of
   Krumhansl-Schmuckler key-finding) instead of the hand-rolled
   diatonic-scale-coverage heuristic, and is now **ON by default**
   (`--no-key-correction` to disable).
2. **CUDA is now the only supported device.** `--device` is gone;
   `separation.py`/`transcription.py` hardcode `device="cuda"`, and
   `main.py` aborts at startup if `torch.cuda.is_available()` is False.
   CPU fallback paths (`compute_type="int8"` etc.) were removed, not
   just defaulted away.
3. **CREPE (`torchcrepe`) runs alongside pYIN** in `note_detection.py`
   (`--no-crepe` to disable, `--crepe-model` for `full`/`tiny`). Per
   frame: where CREPE and pYIN agree within
   `config.CREPE_AGREEMENT_SEMITONES`, CREPE's pitch is used with a
   confidence boost; where they disagree, pYIN's own pitch is kept
   (unchanged) but downweighted via `config.CREPE_DISAGREEMENT_CONFIDENCE_SCALE`
   — deliberately never marked unvoiced, since that would itself
   fabricate a note boundary at exactly the least-trusted frames.
4. **Same-pitch re-articulation splitting + a real off-by-one overlap
   fix**, both in `note_detection.py` — see the two new "Lessons
   learned" entries above for the detail. Both came out of debugging the
   real "And if they fall as Lucifer fell" section reported against a
   hand-verified reference (opening notes G#/F#/G#/A/B/G#; that section's
   full expected beats/pitches/lyrics are worth asking about if
   revisiting this — not reproduced here since it's long).
5. **Chunk-based re-transcription verification redesigned** to actually
   check against reference lyrics, not just self-consistency
   (`verification.py`). Every ASR `Word` now carries `reference_text` —
   the specific reference-lyrics word it was aligned to, if any (set in
   `lyrics_lookup.align_words_to_reference`, including for the "uneven
   block" case that deliberately left text uncorrected before). A fresh,
   isolated recheck is resolved against `reference_text` when present
   (already-correct words are left alone; a wrong word is corrected to
   the reference when the recheck confirms it; reference wins as the
   fallback when everything disagrees) or against the word's own current
   text otherwise. Runs on **every** word by default now, not just
   pass-2-flagged-suspicious ones (`config.VERIFY_ALL_WORDS`,
   `--verify-suspicious-only` to restrict back). Still never touches
   timing/pitch.
6. **The "sword"/"Stars" note-boundary bug is fixed.** `verification.
   verify_placement` (see its own docstring for the full expand-search +
   forced-alignment mechanism) already had everything needed to detect
   this — it just discarded the confirmed answer instead of using it.
   Now, when it gets a PRECISE forced-alignment position (not just
   "confirmed somewhere in this window"), it corrects the word's own
   `(start, end)` to that position and `alignment.align_words` re-runs
   pass 3 with the fix applied — same pattern `verify_words` already uses
   for text corrections, just for timing. This is deliberately NOT
   either of the two previously-rejected heuristics (snap to nearest note
   gap; rebalance by syllable-count deficit/surplus) — both of those
   guessed a new boundary from indirect signals with no confirmation the
   guess was right; this instead uses the SAME position already
   positively confirmed present and located by forced alignment for
   detection. Only a "not found anywhere" or "confirmed in-window but not
   precisely located" result stays a warning (`PlacementWarning`) —
   genuinely nothing confident enough to act on. **Confirmed fixed on
   real audio**: a fresh `--verify-placement` run on `Les Misérables -
   Stars.ogg` placed "sword," at 58.11s (holding to ~61.95s) and "stars"
   cleanly starting right at 61.95s — matching pass-1's independently
   verified truth (58.11–61.95s / 61.95–63.08s) almost to the
   millisecond, vs. the old output's "sword" truncated to a single
   ~142ms beat with "Stars" swallowing the rest. The same run also caught
   and fixed an unrelated real mismatch ("God" assigned at 18.40s,
   actually sung at 24.54s) — not a one-off fix, the mechanism generalizes.
   Still **OFF by default** (`config.ENABLE_PLACEMENT_VERIFICATION`,
   `--verify-placement` to enable) — purely for COST (an expand-search
   re-transcription loop over every word, ~4 minutes on top of
   `verify_words`' own ~4 minutes for this one song), not reliability.

Feedback from this round, worth carrying forward: **new features should
default to ON** (not gated behind an opt-in flag) unless there's a
specific reason given for opt-in (like key-correction's original
flattening-risk concern, since resolved), and **every new/changed
feature needs a real-audio validation run** against `sandbox/Les
Misérables - Stars.ogg` before being reported as done, not just
`test_dry_run.py` — the "Stars" reference notes (opening 6 notes, and
the full "fall as Lucifer fell...sword" section) are saved in this
session's memory for that purpose.

Discussed but not yet decided/implemented:

1. **Essentia's MELODIA as a further pYIN/CREPE ensemble member or
   replacement**: purpose-built polyphonic melody extraction
   (`PredominantPitchMelodia`). Real architecture change (new heavy
   dependency) — needs an explicit go-ahead before building.
2. **MIDI database cross-checking**: considered and deprioritized — no
   reliable, API-searchable public database of vocal MIDI transcriptions
   is known to exist; would mostly add fragile scraping for something
   that'd fail silently on most songs.

## Environment notes

- Windows, venv at `E:\Projects\ultrastar_generator\venv`.
- CUDA is required — CPU support was removed entirely, not just
  defaulted away; the pipeline aborts at startup if
  `torch.cuda.is_available()` is False. WhisperX pulls in
  pyannote/torch — expect noisy-but-harmless warnings (torchcodec/ffmpeg
  version mismatches, TF32 reproducibility notices) on startup; these
  haven't been correctness issues so far.
- Demucs writes intermediate stems to `sandbox/.ultrastar_work/` —
  safe to delete between runs to reclaim disk space, and separation is
  cached/skipped if `vocals.wav` already exists there.
