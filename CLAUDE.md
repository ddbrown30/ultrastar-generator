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

0. **New optional pass 4 (`musicxml_reference.py`): confirms/corrects
   pass-3 syllable PITCH CLASS (never octave, never timing) against a
   user-supplied MusicXML/.mxl file** -- e.g. sheet music hand-downloaded
   from MuseScore (`--musicxml-reference <path>`, off unless given --
   MuseScore's own free download path was found to be actively blocked
   platform-side, see the MELODIA/Essentia thread below for the
   unrelated dependency story; this feature works from a file the user
   downloads themselves through the normal UI, no automated fetch).
   Aligns by lyric text (same whole-sequence `difflib` technique
   `lyrics_lookup.py` already uses), calibrates a per-song PITCH-CLASS
   offset (absorbs both real transpositions, e.g. +2/+5 semitones, and
   octave-notation inconsistency between different arrangements/parts of
   the same song -- confirmed on real files), then corrects any matched
   syllable that still disagrees once calibrated. Deliberately never
   touches absolute octave (sheet-music octave notation was found
   internally inconsistent even within one calibrated file) or timing --
   matches how UltraStar Deluxe itself scores (pitch class, octave-
   agnostic, confirmed by the user).

   Calibration is TWO-TIERED: try the full matched population first
   (works fine for most songs); if that doesn't clear a majority bar,
   retry using only the top half by OUR OWN note confidence, with a
   correspondingly lower bar -- a noisy upstream detector dilutes the
   full-population signal with matches where our own pitch is simply
   wrong, not real per-song ambiguity, and restricting to higher-
   confidence matches cleans that up without changing the winning
   answer. `Syllable` gained a `confidence` field (didn't exist before --
   pass 1's `NoteEvent.confidence` was being silently dropped by the time
   pass 3 built its output) to make this possible; threaded through every
   `Syllable` construction site. Found and fixed a related, unrelated
   pre-existing bug in the same pass: `postprocess.enforce_monotonic` was
   silently dropping `line_id` on every call (defaulting to `None`),
   which could have been quietly breaking reference-line-forced phrasing
   breaks -- not confirmed how much real impact this had before the fix,
   worth watching for if a phrasing regression is ever reported.

   **Real end-to-end validation** (realistic proxy pass-3 input --
   pyin's own weighted-mode pitch per ground-truth word, not clean
   ground truth -- run through the actual `apply_musicxml_reference`
   code path, compared before/after against each song's established
   ground truth): batb 59.0%->96.4% (+37.4pp), stars 57.1%->87.9%
   (+30.8pp), Bare Necessities 48.0%->62.9% (+14.9pp), Gaston
   41.1%->90.0% (+48.9pp, the largest of the four -- the two-tier
   calibration fix specifically targeted Gaston, previously the
   session's consistently worst-performing song, and it ended up
   benefiting the most). No regression on the 3 songs that already
   calibrated fine on the full population -- the confidence-tiered
   fallback only ever activates when the primary check fails.
0a. **Auto-detection of MusicXML reference file(s) for pass 4**
   (2026-08-09, same day as pass 4 itself): `file_discovery.
   find_companions` now also finds `.mxl`/`.musicxml`/`.xml` files next
   to the audio, matched by EXTENSION ALONE (unlike video/cover, which
   match the audio file's own basename -- a downloaded score keeps
   whatever name its source gave it, e.g. `beauty-and-the-beast.mxl`,
   never `<Artist> - <Title>.mxl`). `--musicxml-reference` still wins if
   given explicitly; otherwise every auto-detected file is used. If
   MULTIPLE reference files are found (or given -- not yet exposed as a
   multi-value CLI flag, only via auto-detection), ALL of them are
   applied SEQUENTIALLY (`apply_musicxml_references`, plural) --
   different arrangements of the same song often lyric-tag different,
   only partly-overlapping portions of it (confirmed on Once Upon A
   Dream: one file covered 52.6%, a different arrangement covered a
   different 25.1%), so using only one leaves real coverage the other
   file has on the table. Each file gets its own independent
   calibration; one that can't establish confident calibration is
   skipped on its own without blocking the others.
0b. **RMVPE default switch: tried, then REVERTED the same day
   (2026-08-09) after real end-to-end validation.** `pitch_primary`
   was switched from `"pyin"` to `"rmvpe"` (with `ENABLE_RMVPE = True`
   so pyin/CREPE still cross-check it) on the strength of RMVPE being
   the best single ISOLATED raw pitch source across the whole validated
   song set all session. Real end-to-end validation (actual production
   pipeline -- pyin+CREPE cross-checks either way, not isolation mode --
   across the 4-song core set) showed this was a net REGRESSION, not an
   improvement: batb 55.0%->55.0% (flat), stars 59.3%->52.7% (-6.6pp),
   sleeping_beauty 49.3%->44.1% (-5.2pp), gaston 37.6%->36.1% (-1.5pp);
   average -3.3pp, no gain on any song. **Both `pitch_primary` and
   `ENABLE_RMVPE` were reverted back to their original pyin-primary
   defaults the same day** -- see their own comments in
   `note_detection.py`/`config.py`. Same lesson as the consensus-override
   finding earlier this session, now confirmed a second time on a bigger
   decision: an isolated/proxy diagnostic (RMVPE's raw isolation-mode
   accuracy) does NOT reliably predict real end-to-end pipeline impact,
   because it can't see how the existing pipeline (segmentation
   thresholds, cross-check agreement/disagreement scaling) was tuned
   around a DIFFERENT source's noise characteristics over many prior
   sessions. **Don't re-attempt this switch on the strength of isolation
   numbers alone in a future session** -- that reasoning has now failed
   real validation twice in different forms (RMVPE's own segmentation
   retuning earlier this session hit a low ceiling for the same
   underlying reason). A real case for switching would need its own
   fresh end-to-end validation, not a re-read of the isolation-mode
   comparison table.
0c. **Follow-up controlled comparison (same day): confirmed cross-checking
   itself was the cause of 0b's regression, but also found the isolation-
   mode numbers were never a clean single-variable comparison to begin
   with.** Ran RMVPE-primary with cross-checking fully disabled
   (`cross_check_primary=False, use_crepe=False`), holding VOICING SOURCE
   constant (pyin's own voiced_flag + energy gate, same as both the
   pyin-primary baseline and the reverted rmvpe+crosscheck config -- NOT
   RMVPE's own voicing, which isolation mode uses instead). Real
   end-to-end results, same 4-song set:

   | Config | batb | stars | sleeping_beauty | gaston | Average |
   |---|---|---|---|---|---|
   | pyin-primary (current default) | 55.0% | 59.3% | 49.3% | 37.6% | 50.3% |
   | rmvpe+crosscheck (0b, reverted) | 55.0% | 52.7% | 44.1% | 36.1% | 47.0% |
   | rmvpe, NO crosscheck (pyin voicing) | 59.3% | 56.0% | 48.8% | 37.9% | 50.5% |
   | isolation rmvpe alone (RMVPE's own voicing) | 58.0% | 57.0% | 50.0% | 43.0% | 52.0% |

   Removing cross-check alone recovers +3.5pp (47.0%->50.5%) -- confirms
   the mechanism: cross-check disagreement never overrides the primary's
   pitch VALUE, only downweights confidence, but once RMVPE (the more
   accurate source) is primary, pyin/CREPE disagreeing with it is more
   often THEM being wrong, not RMVPE -- so the downweighting fires
   backwards, suppressing RMVPE's own correct frames' influence on the
   confidence-weighted per-note vote. BUT: "RMVPE primary, no cross-check,
   pyin's voicing" (50.5%) only ties the current shipped default (50.3%,
   pyin-primary) -- not a real win either way. The gap between that and
   isolation mode's 52.0% comes from a THIRD variable never isolated
   before now: RMVPE's OWN voicing decision, not just its pitch values.
   **Not yet validated as a real shipped config**: actually shipping
   `isolation_source="rmvpe"` (RMVPE's own voicing, no cross-check at
   all) as the default, rather than as a research/testing-only mode --
   the one remaining real candidate with a plausible (if modest, +1.7pp)
   edge, not yet tested end-to-end as an actual default. If picked back
   up, that's the next concrete step, not another isolation-vs-shipped
   comparison -- this one's already been run.
0d. **Fresh, real, REPRODUCIBLE validation of `isolation_source="rmvpe"`
   as an actual default (same day): a genuine positive result, unlike
   everything else tried in this whole RMVPE-default thread.** Run twice
   per song specifically to check RMVPE's own documented non-determinism
   (see "Lessons learned" above) -- both runs were IDENTICAL (0.0pp
   spread, same note counts, every song), so this result isn't noise:

   | Song | pyin-primary (today) | rmvpe-isolation (both runs) | Delta |
   |---|---|---|---|
   | batb | 55.0% | 57.9% | +2.9pp |
   | stars | 59.3% | 57.1% | -2.2pp |
   | sleeping_beauty | 49.3% | 50.2% | +0.9pp |
   | gaston | 37.6% | 42.8% | +5.2pp |
   | **Average** | **50.3%** | **52.0%** | **+1.7pp** |

   Wins on 3 of 4 songs (gaston most, +5.2pp), loses only on stars
   (-2.2pp, consistent with the "no universal winner" pattern seen all
   session for every source). Bigger and more reproducible than the
   smaller gains already rejected as not worth pursuing this session
   (RMVPE's own segmentation retuning +0.8pp, its voicing-threshold
   tuning +0.9pp) -- and unlike those, this SIMPLIFIES the pipeline
   rather than adding a new tuned constant: no CREPE computation, no
   cross-check logic at all when this path is used, and noticeably
   faster (19-26s/song here vs. 111-213s for the cross-check-heavy
   configs tested earlier in this thread). **Shipped**: `isolation_source
   ="rmvpe"` is now `main.py`'s real default (via `config.
   DEFAULT_PITCH_SOURCE = "rmvpe"` and the `--pitch-source` CLI flag,
   `choices=["rmvpe","ensemble"]`) -- `detect_notes()`'s own function
   default was deliberately left untouched so `test_dry_run.py` and ad
   hoc scripts aren't silently affected.
0e. **First real full-8-song end-to-end CLI batch run with the shipped
   defaults (rmvpe isolation-mode pitch + MXL auto-detection), comparing
   FINAL pipeline output (not a proxy) against each song's own ground
   truth.** Found and fixed one real regression from this session's own
   MXL auto-detection work: `file_discovery.find_companions` matched
   `.xml` by extension ALONE, which also matches these SingStar rips'
   own `notes.xml` companion file (a different, proprietary format --
   root tag `{http://www.singstargame.com}MELODY`, not MusicXML at all).
   `music21.converter.parse` threw on it and crashed the WHOLE run for
   every song that happens to ship one (6 of 8 in this batch -- only
   Stars and Gaston don't have a `notes.xml` sitting next to their
   audio). Fixed at the root: bare `.xml` candidates are now content-
   sniffed (`_looks_like_musicxml`, checks for a `score-partwise`/
   `score-timewise` tag near the top of the file) before being trusted;
   `.mxl`/`.musicxml` extensions remain trusted unconditionally (real
   MuseScore downloads, unambiguous). Also hardened
   `apply_musicxml_references` (plural) to catch a per-file parse
   exception and record it as `skipped_reason` instead of crashing the
   whole run -- so a future unexpected companion file degrades
   gracefully instead of taking down pass 4 entirely.

   After the fix, all 8 songs completed successfully. Final-output-vs-
   ground-truth pitch accuracy (word/syllable text aligned via the same
   whole-sequence difflib technique used throughout this project;
   exact-MIDI or pitch-class per each song's already-established
   convention -- see the ground-truth table further down):

   | Song | metric | coverage | accuracy |
   |---|---|---|---|
   | batb | exact | 72.1% | 98.0% |
   | stars (partial OMR anchors) | pitch-class | 89.0% | 88.9% |
   | sleeping_beauty_ouad | exact | 19.9% | 9.5% |
   | gaston | pitch-class | 2.6% | 20.0% |
   | little_mermaid | pitch-class | 74.6% | 59.3% |
   | sleeping_beauty_wonder | exact | 59.7% | 23.9% |
   | tarzan_son_of_man | exact | 40.1% | 88.0% |
   | jungle_book_bare_necessities | pitch-class | 60.9% | 35.7% |

   **The low scores are NOT a pitch-source regression -- they trace to
   two separate, PRE-EXISTING lyrics-text issues, unrelated to this
   session's rmvpe/MXL work, that this was the first real test exposing
   at full-pipeline scale:**
   1. `lyrics.ovh` reference-lyric lookup failed outright ("not found,
      or no network") for sleeping_beauty_ouad, sleeping_beauty_wonder,
      tarzan_son_of_man, and jungle_book_bare_necessities -- 4 of 8
      songs, all Disney-soundtrack titles that may just not be well
      indexed there. Without a reference, `verification.verify_words`'s
      `_resolve()` no-reference branch (`verification.py` ~line 144)
      blindly trusts a fresh isolated few-hundred-ms re-transcription
      over the original whole-context ASR text whenever they disagree --
      and isolated-clip ASR is a known-unreliable signal (same failure
      class as the long-documented "never run pYIN on a tiny isolated
      clip" pitch lesson above, just for text/Whisper instead of pitch).
      In practice this REPLACES already-correct short words with
      hallucinated garbage (confirmed in the logs, e.g. "I"->"Whoo-hoo!",
      "why"->"the white little"), which is why coverage against ground
      truth collapses for exactly these 4 songs. tarzan still scored
      88% accuracy despite this, so the underlying pitch detection isn't
      the problem -- the TEXT alignment used to find comparable notes is.
   2. Gaston's `lyrics.ovh` lookup returned a SPANISH-language reference
      (words like "quiero"/"verte"/"pueblo" -- confirmed in the log),
      not the English original, for artist/title "Beauty and the Beast"/
      "Gaston". `verify_words` then trusted that wrong-language reference
      as ground truth for TEXT, corrupting the whole song's lyrics.
      Explains gaston's 2.6% coverage outlier by itself.

   **FIXED, same day, see 0f below**: the `_resolve()` no-reference
   fallback trusting an isolated recheck over full-context ASR without
   any confidence gate, and the wrong-language reference problem (fixed
   via the LRCLIB migration + `reference_matches_transcript` gate, not
   by patching `lyrics_lookup.fetch_reference_lyrics`'s matching itself).
   This paragraph is kept for the historical diagnosis; don't read it as
   still-open.

   On the 3 songs where reference-lyric lookup worked AND matched the
   right language (batb, stars, little_mermaid) -- the closest to a
   clean read on the new pitch-source default's real accuracy --
   average is (98.0+88.9+59.3)/3 = 82.1%, all with real (not proxy)
   full-pipeline output. Outputs for all 8 songs kept at
   `sandbox/full-pipeline-validation/<song_key>/` (nothing suppressed --
   pass-1 debug file, debug log, and final .txt all present) for
   independent inspection.
0f. **Both 0e follow-up bugs fixed same day, plus one more found by the
   user inspecting real output (`batb`) directly -- 3 real fixes total,
   all real-audio re-validated:**
   1. **`lyrics_lookup.py` now tries LRCLIB (lrclib.net) FIRST, falling
      back to lyrics.ovh only if LRCLIB has nothing.** LRCLIB has a real
      search API (`artist_name`/`track_name` query params returning
      several candidates, picked by closeness to OUR OWN audio duration
      -- `_fetch_from_lrclib`) instead of lyrics.ovh's rigid single-shot
      path, and often includes synced (per-line-timestamped) lyrics
      (`LyricsResult.synced_lyrics`, LRC format -- fetched and stored,
      not consumed by anything yet, see the still-open item below).
      `fetch_reference_lyrics` now returns a `LyricsResult`
      (`plain_lyrics`/`synced_lyrics`/`source`), not a raw string.
      **Also added `reference_matches_transcript`** (difflib ratio
      between the fetched reference's word vocabulary and the ASR
      transcript's own vocabulary, `config.
      REFERENCE_LYRICS_MIN_MATCH_RATIO = 0.25`) as a source-independent
      safety net -- rejects a wrong-song/wrong-language reference
      before it's ever trusted, regardless of which source answered.
      Real validation, live network, all 8 songs: LRCLIB found real
      lyrics for 6 of 8 (up from 3 of 8 under lyrics.ovh alone) --
      including recovering the correct ENGLISH lyrics for Gaston (the
      0e Spanish-lyrics bug). Only sleeping_beauty_wonder and
      jungle_book_bare_necessities still have no reference anywhere
      (confirmed absent from both sources, not a bug).
   2. **`verification.py`'s `_resolve()` no-reference branch no longer
      blindly replaces already-correct text.** It used to trust ANY
      disagreeing isolated few-hundred-ms recheck over the original
      full-context ASR word whenever no reference existed to adjudicate
      -- exactly the "never trust inference from a tiny isolated clip"
      failure class already documented above for pitch, just for
      ASR/text. Now: log the disagreement for visibility, but keep the
      more reliable full-context text when nothing can confirm the
      recheck. `test_dry_run.py` updated to match (word 0 in the
      verify_words test now stays unchanged, `replaced=False`).
   3. **`phrasing.py`'s reference-line priority was never actually
      absolute -- found by the user inspecting real batb output.**
      `build_lines` OR'd a raw gap check (`gap >= MIN_LINE_GAP_SEC`)
      into `force_break` unconditionally, even when `Syllable.line_id`
      confirmed the next word was still part of the SAME reference
      line -- contradicting the module's own docstring. Real, confirmed
      case: "Just a little change" (one reference line) has an audible
      pause before "change" long enough to trip the gap heuristic, so
      it was being split into "Just a little" / "change" as two output
      lines; same for "Both a little scared", "Ever just as sure",
      "Certain as the sun", "Song as old as rhyme" -- all one-line
      reference phrases, all spuriously split. Fixed: when both the
      current line and the next word have a KNOWN, MATCHING line_id,
      nothing breaks the line early except the pre-existing
      implausible-length safety net (1.5x `MAX_SYLLABLES_PER_LINE`) --
      the reference wins outright, even over a long gap. New regression
      test added (`test_dry_run.py`, "long silence gap WITHIN a single
      confirmed reference line"). Confirmed fixed against real batb
      output after re-running.

      **Open tension this fix creates, not yet resolved:** a reference
      LINE is a typographic convention from whoever transcribed it, not
      guaranteed to be one musical PHRASE. Confirmed real case: LRCLIB's
      Under The Sea merges "Under the sea, under the sea" into ONE line
      (lyrics.ovh had it as two) -- if that's actually sung as two
      distinct phrases with a real pause, the phrasing fix above will
      now NEVER break there, producing a run-on line where two are
      musically correct. Not fixed -- would need either a much-higher
      "this reference line is absurdly-implausibly two phrases" gap
      threshold that still respects the reference by default, or some
      other signal; flagged for the user rather than guessed at.

   Real full-8-song re-validation after all three fixes (pitch-CLASS
   accuracy -- the metric that actually matches how UltraStar Deluxe
   itself scores, octave-agnostic; see pass 4's own docstring):

   | Song | pitch-class accuracy | coverage |
   |---|---|---|
   | batb | 100.0% | 73.6% |
   | stars (partial OMR anchors) | 88.9% | 89.0% |
   | sleeping_beauty_ouad | 89.8% | 28.0% |
   | gaston | 93.6% | 64.7% |
   | little_mermaid | 38.9% | 66.5% |
   | sleeping_beauty_wonder | 28.8% | 76.6% |
   | tarzan_son_of_man | 96.6% | 60.3% |
   | jungle_book_bare_necessities | 39.0% | 71.7% |
   | **Average** | **71.95%** | |

   Gaston's fix is the standout, real-audio-confirmed result: EXACT-MIDI
   accuracy alone is meaningless for gaston/little_mermaid/jungle_book
   (their ground truth is independently known octave-ambiguous, see the
   pitch-class-only convention table above) -- but even so, gaston's
   coverage against ground truth went from 2.6% to 64.7% purely from
   fixing the Spanish-lyrics bug (nothing about pitch detection changed).

   **Two accuracy patterns identified, NEITHER caused by this round's
   fixes -- one is harmless and CLOSED, the other is real and still
   open (both corrected below after further digging same session --
   see the superseded original framing was wrong on both counts):**
   - **sleeping_beauty_ouad is a whole-song, consistent, CLEAN octave/
     register selection difference -- not a real pitch problem, and
     NOT worth pursuing further.** 56 of 58 matched-note offsets are
     an EXACT multiple of 12 semitones (41 at -12, 11 at 0, 4 at -24);
     only 2/58 are anything else. Since UltraStar Deluxe scores pitch
     CLASS, not exact octave (confirmed by the user, see pass 4's own
     docstring), a clean octave transposition is already fully
     forgiven -- that's exactly why pitch-class accuracy here is 89.8%
     (later 96.6% after the duet-merge/melisma-token comparison-script
     fixes below). This is likely `isolation_source="rmvpe"` picking a
     different (but internally consistent) octave register for this
     song's vocal range -- cosmetically odd, but harmless for real
     gameplay. Closed; don't chase this one further.
   - **little_mermaid's poor pitch-class accuracy is REAL, and is NOT
     an octave problem at all, despite superficially looking like one.**
     (An earlier pass at this data wrongly attributed it to a lyrics-
     source-driven note-reassignment regression; that was DISPROVEN the
     same session -- see the A/B test in 0g's introduction: running the
     exact same song with lyrics lookup fully disabled produced 549/549
     IDENTICAL notes, start time and pitch, vs the LRCLIB run. Pass 1/
     pass 3 pitch/timing genuinely cannot see which lyrics source was
     used, so that explanation was wrong; the swing between reported
     numbers across runs was a text-matching artifact, not a real
     pipeline difference -- see 0g.) The REAL signature, found by
     checking whether little_mermaid's errors are clean multiples of 12
     the same way ouad's are: they are NOT. The single -12 bucket alone
     (171 of ~364 matched) accounts for essentially 100% of its 47%
     "correct" count -- every other offset in its histogram (-13, -11,
     -14, -15, -10, -9, -8 semitones, roughly matching -12's own count
     combined) is NEAR an octave but not exactly one, so none of it is
     forgiven by pitch-class comparison, and none of it would be
     forgiven by real UltraStar scoring either. This is genuine pitch-
     detection inaccuracy on roughly half this song's notes, not a
     register/octave choice. **ROOT-CAUSED AND FIXED later the same
     day, see 0h below -- don't read this as still open.**

   **Not yet built**: actually consuming `LyricsResult.synced_lyrics`
   for anything. Real feasibility check done (Tarzan - Son Of Man,
   fetched fresh synced lyrics, compared LRC per-line timestamps against
   that song's own final-output line-start times): timestamps land
   within a few seconds of our own line starts throughout the song, not
   exact but clearly the same recording/timing, not noise -- confirms
   there's real signal, same spirit as musicxml_reference.py's
   calibration-offset pattern but for TIME instead of pitch. A concrete,
   scoped design was proposed to the user (calibrate a per-song time
   offset the same two-tier confidence way pass 4 does, use it to
   flag/correct badly-drifted line placements -- cheaper than
   `verify_placement`'s re-transcription loop since it needs no fresh
   ASR call) but building it was NOT started -- check whether the user
   actually asked for it before assuming this is done.
0g. **Tried and rejected (real prototype, real audio): separating lead
   vocal from backing/harmony vocals does NOT explain or fix
   little_mermaid's poor pitch accuracy.** Hypothesis going in: "Under
   The Sea" has a near-constant background chorus through most of its
   runtime (unlike the cleaner solo songs that scored well), and RMVPE
   is a monophonic pitch tracker (confirmed by reading `_rmvpe_pitch` --
   it returns exactly one `(frequency, confidence)` per frame, no
   voice-count/polyphony signal at all) that has to arbitrarily pick one
   voice whenever several sing at once -- 15-second-bucketed accuracy
   was pervasively mediocre (20-56%) across nearly the whole song, not
   localized to one passage, consistent with a near-constant confusion
   source rather than a single bad moment.

   Installed `audio-separator` (wraps UVR's models,
   `pip install audio-separator[gpu]`) and ran its best-scoring karaoke
   model, `Mel-Roformer-Karaoke-Aufr33-Viperx` (SDR 8.4 vs the older
   MDX-Net karaoke model's 5.4), on top of the existing Demucs vocals
   stem to further split lead vocal from backing/harmony. Compared
   pass-1 (`isolation_source="rmvpe"`) pitch accuracy on the lead-only
   stem against the current Demucs-only baseline, using TIME-OVERLAP
   note matching against ground truth (dominant pass-1 note pitch
   within each ground-truth note's window) -- deliberately NOT the
   lyrics/ASR text-matching methodology used elsewhere this session,
   to avoid the exact measurement-noise problem already found and
   documented in 0f (different text -> different matched subset ->
   different reported number, same underlying pitch).

   Result: pitch-class accuracy 47.0% (baseline) vs 45.2% (lead-vocal-
   isolated) -- no improvement, actually very slightly worse, and the
   offset histograms are nearly IDENTICAL between the two conditions
   (both dominated by -12, -13, -11, -14 semitones in almost the same
   proportions).

   **Two follow-up attempts, same session, same negative result --
   this is now a well-tested dead end, not a one-off:** user listened
   to the separated stems directly and reported no audible difference
   from the original; a quantitative check confirmed it
   (`corr(original, "lead") = 0.97`, the "backing" stem the model
   split out was ~4x quieter than the lead and only ~23% of the
   original's energy -- i.e. the model found little to separate in the
   first place, not that separation itself failed). Tried (1) the
   highest-SDR-scoring karaoke model available via `audio-separator`
   at the time (`MelBand Roformer | Karaoke by Gabox`, SDR 8.69 vs
   Aufr33-Viperx's 8.45) on the same Demucs vocals stem -- correlation
   with the original was 0.972, essentially identical to Aufr33-
   Viperx's 0.969: a stronger model found the same small amount of
   separable content, not more. (2) Ran the SAME Gabox model directly
   on the ORIGINAL FULL MIX instead of the pre-Demucs-isolated vocals
   stem -- its "vocals" stem correlates 0.96 with Demucs's OWN vocals
   output, meaning on a full mix this model just re-derives a plain
   vocals-vs-instrumental split (Demucs's job), not a lead-vs-backing
   split; not a different/better separation, a different problem.

   All four conditions' real pitch-class accuracy (Demucs baseline,
   Aufr33-Viperx-on-Demucs-vocals, Gabox-on-Demucs-vocals, Gabox-on-
   full-mix): 47.0% / 45.2% / 45.1% / 47.0% -- statistically flat, and
   EVERY condition shows the same dominant -12-semitone offset in
   nearly the same proportion. This conclusively rules out harmony/
   backing-vocal confusion as little_mermaid's problem -- neither model
   choice nor input pipeline (pre-isolated vocals vs. full mix) moved
   the number at all. **Don't re-attempt lead-vocal/harmony separation
   for this song -- this thread is closed, tested three different ways
   with converging null results.**

   **IMPORTANT correction, same session, caught by the user pushing
   back on the framing above:** this is NOT the same mechanism as
   sleeping_beauty_ouad's octave issue, and calling it "an octave bias"
   at all was misleading -- checked and corrected immediately after
   writing it. UltraStar Deluxe scores pitch CLASS, so a pure/clean
   octave difference (ouad's actual problem, see above) is already
   fully forgiven and harmless. little_mermaid's -12 bucket (171 of
   ~364 matched) is real and accounts for basically its entire pitch-
   class-correct count, but EVERY OTHER offset in its histogram (-13,
   -11, -14, -15, -10, -9, -8 -- roughly matching -12's own count
   combined) is near-but-not-exactly an octave, so none of THAT is
   forgiven, by pitch-class comparison or by real UltraStar scoring.
   That's the actual problem: genuine pitch-detection inaccuracy on
   roughly half this song's notes that happens to often land in the
   neighborhood of an octave below truth, not a register/octave
   SELECTION issue the way ouad's cleanly is. Framing this as "the same
   RMVPE octave-selection mechanism as ouad" (as an earlier version of
   this note did) would send a future investigation in the wrong
   direction -- ouad's fix target would be "why does RMVPE pick a
   different but internally-consistent octave here" (probably not
   worth chasing, it's harmless); little_mermaid's real question is
   "why is RMVPE's actual pitch value wrong, not just its octave" for
   about half this song's notes, still genuinely unsolved.

0h. **Root cause found (same day, deeper dig): this is a genuine acoustic
   ambiguity in rough/character vocal production, confirmed across FOUR
   independent pitch estimators, NOT a training-distribution gap in any
   one of them -- and a real, validated fix shipped for it.**

   Traced specific little_mermaid errors back to raw per-frame RMVPE
   output (cached from earlier in the session, confirmed still valid --
   RMVPE isolation mode is bit-reproducible for this song, verified
   earlier). The "wrong" pitch traces are NOT low-confidence or noisy --
   confidence during these errors (0.77 avg) is nearly as high as during
   harmless clean-octave cases (0.81), and the traces themselves are
   smooth, stable, well-tracked contours, just locked onto the wrong
   frequency. Cross-checked 3 specific flagged notes against the
   INDEPENDENT MusicXML sheet music (not just our own ground truth) --
   the sheet music agreed with ground truth exactly every time (70=70,
   65=65, 70=70), ruling out ground-truth error as the explanation.
   Checked the accompaniment stem's own detected pitch at those same
   moments -- no match to RMVPE's wrong values, ruling out simple
   instrumental bleed-through too.

   The decisive test: the user noted the recording's only unusual trait
   is the singer's vocal delivery (Sebastian's Jamaican-accented
   reggae-style singing for little_mermaid) -- prompting a check of
   jungle_book_bare_necessities too (Baloo/Phil Harris, gravelly jazz
   talk-singing, also a poor scorer). Its raw-frame error histogram is
   nearly IDENTICAL in shape to little_mermaid's (-12 dominant/harmless,
   then -13, -11, -14 as the next-biggest buckets in matching
   proportions) -- while batb (clean legit Broadway voice) shows ZERO of
   this pattern. Then tested ALL FOUR cached pitch sources for
   jungle_book (pyin, RMVPE, SwiftF0, PENN -- four different
   architectures/training data: classical DSP, U-Net+GRU trained on
   Mandarin-pop, small CNN trained on speech+synthetic, cross-domain
   speech+music CNN) -- ALL FOUR converge on the SAME wrong answer in
   the same proportions (pitch-class accuracy 37-48% across all four,
   same -12/-13/-11/-14 shape every time). Four independently-trained
   estimators agreeing on the same wrong answer rules out "pick a
   better-trained model" as a fix -- the ambiguity is in the raw
   acoustic signal itself (very likely vocal fry / non-clean glottal
   vibration common in rough/growled vocal production, which can make
   the actual dominant periodicity genuinely different from the
   "musical" pitch a human parses out through top-down melodic
   expectation), not in any one model's training distribution. Checked
   what pretrained singing-pitch models exist trained on more diverse
   vocal STYLES specifically (not just noise-robustness) -- RMVPE
   (MIR-1K/MIR_ST500/Cmedia, Mandarin-pop), PENN (MDB+PTDB, cross-domain
   speech/music), SwiftF0 (NSynth+PTDB-TUG+synthetic Mandarin TTS
   speech), ROSVOT (2024, noise-robustness-focused via MUSAN
   augmentation, not confirmed style-diverse) -- none of the readily
   available options target this specifically, consistent with the
   four-way convergence result: no pitch source in existence was going
   to fix this on its own.

   **Fix: `musicxml_reference.py` gained `force_calibration` (new
   parameter on `apply_musicxml_reference`/`apply_musicxml_references`,
   `main.py`'s `--musicxml-force-calibration` CLI flag, EXPERIMENTAL/
   off by default).** Normal pass 4 skips a song entirely when our own
   pitch can't establish confident calibration against the MXL
   reference (exactly little_mermaid's and jungle_book's case -- the
   normal bar measures agreement against pass 1, which is the actual
   problem here). `force_calibration=True` applies the best available
   offset anyway (full population, or the high-confidence subset if
   it's the stronger signal -- reuses the existing two-tier logic,
   just removes the "give up" branch) and corrects every matched
   syllable's pitch class from it, however weak the calibration
   confidence. Still never touches octave or timing, same guarantee as
   normal pass 4.

   Real end-to-end validation, full pipeline, both hard-case songs:

   | Song | baseline pitch-class | force_calibration | delta |
   |---|---|---|---|
   | little_mermaid | 37.7% | 59.3% | +21.6pp |
   | jungle_book_bare_necessities | 39.0% | 58.0% | +19.0pp |

   Coverage unchanged in both cases (this only touches pitch, never
   text-matching), so the gain is 100% real pitch-quality improvement,
   not a measurement-methodology artifact. Both songs' calibration
   confidence was genuinely weak even after forcing (little_mermaid
   40%, jungle_book 40% full-population/38% subset -- neither clears
   the normal bars), and it still won decisively over trusting pass 1
   at all. **Provably a no-op for already-well-calibrated songs**: the
   force-only code path is only ever entered when the full-population
   confidence is already below `MUSICXML_MIN_CALIBRATION_CONFIDENCE`,
   so songs where normal calibration already clears the bar (batb,
   gaston, tarzan, stars) are completely unaffected by this flag --
   not just untested, provably unreachable for them.

   **Decided and shipped, same day**: user asked for the remaining 4
   MXL-having songs to be tested too (stars, sleeping_beauty_ouad,
   gaston, tarzan_son_of_man) before deciding on a default -- full real
   end-to-end results, all 7 MXL-having songs now covered
   (sleeping_beauty_wonder has no MXL file at all, trivially unaffected):

   | Song | baseline pitch-class | force_calibration | delta |
   |---|---|---|---|
   | batb | 100.0% | 100.0% | 0 |
   | stars | 86.4% | 86.4% | 0 |
   | sleeping_beauty_ouad | 96.6% | 96.6% | 0 |
   | gaston | 93.6% | 93.6% | 0 |
   | tarzan_son_of_man | 96.6% | 98.5% | +1.9pp |
   | little_mermaid | 37.7% | 59.3% | +21.6pp |
   | jungle_book_bare_necessities | 39.0% | 58.0% | +19.0pp |

   Every song showed zero or positive change, zero regressions -- per
   the user's own stated criterion, this is now `config.
   ENABLE_MUSICXML_FORCE_CALIBRATION = True` (default), with
   `--no-musicxml-force-calibration` to opt back out. The function-level
   default in `apply_musicxml_reference`/`apply_musicxml_references` was
   also changed to pull from this config constant directly (matching
   how every other `ENABLE_*` toggle in this codebase works, e.g.
   `ENABLE_CREPE`/`ENABLE_WORD_VERIFICATION` --
   NOT kept as a hardcoded-False library default with only the CLI
   overriding it, unlike the deliberately-separate `isolation_source`/
   pitch-source case earlier in this document). `test_dry_run.py`'s
   existing calibration tests were unaffected (they either already
   clear the confidence bar normally, or hit the too-few-matches early
   exit before force_calibration would ever matter); the new ambiguous-
   population test's "normal mode" branch now passes
   `force_calibration=False` explicitly, since the function's own
   default flipped.

0i. **New validation dimension: NOTE TIMING accuracy (start-time
   deviation from ground truth), not just pitch.** Every prior
   validation this session measured pitch only -- alignment was by TEXT
   match, and once two notes were paired only their pitch got compared;
   timing was parsed but never scored. Built `compare_timing()`
   (scratchpad `compare_full_pipeline_output.py`) to close that gap:
   same text-based pairing, but scores `|out_start - gt_start|`.

   **Found and fixed a real methodology bug this immediately exposed**:
   `difflib`'s text-only alignment has no notion of time, so a repeated
   lyric/phrase (chorus hook, reprise, echo) can get ground truth's
   occurrence paired against a DIFFERENT sung instance of the same text
   in our output -- a real alignment ambiguity, not a real timing error,
   but it corrupts pitch comparison too, not just timing (a fixed-offset
   pitch agreement can survive across instances if the repeat is
   melodically identical, silently passing what should be an invalid
   comparison). A fixed absolute distance cutoff doesn't work (confirmed
   two ways): different songs repeat at different intervals, so no one
   cutoff is both loose enough for real widely-spaced songs and tight
   enough for closely-spaced ones; and a raw per-song MEDIAN isn't
   robust either -- if repeat-artifacts actually outnumber real matches
   (confirmed real case: sleeping_beauty_ouad), the median itself gets
   pulled into the wrong cluster and the filter inverts, REJECTING real
   matches and keeping artifacts. Working fix: bucket all candidate
   deltas at 200ms resolution, take the bucket with the most candidates
   as the reference point (real matches cluster tightly regardless of
   song -- confirmed under ~200ms median everywhere once artifacts are
   removed -- while artifacts spread across many different offsets
   depdepending on where in the song each repeat lands, so the tightest
   single bucket is reliably the true cluster even when outnumbered),
   then reject anything >3s from that reference.

   This fixed 7 of 8 songs cleanly. `sleeping_beauty_ouad` resisted
   every version of the fix -- hand-inspected its actual output content
   and found why: it's a genuine duet where the SAME verse is legitimately
   sung 2-3 times as call-and-response echo (confirmed both ground truth
   and our own output are individually correct), which whole-song text
   alignment fundamentally can't disambiguate. Flagged as unreliable for
   this methodology rather than chased further -- diminishing returns on
   one duet-structured song.

   **Real timing results, 7 reliable songs** (mean/median absolute
   start-time error, and % of matched notes within 500ms):

   | Song | mean | median | <=500ms |
   |---|---|---|---|
   | batb | 206ms | 130ms | 94.7% |
   | gaston | 243ms | 162ms | 92.4% |
   | little_mermaid | 130ms | 75ms | 95.8% |
   | sleeping_beauty_wonder | 223ms | 114ms | 86.5% |
   | tarzan_son_of_man | 135ms | 25ms | 95.9% |
   | jungle_book_bare_necessities | 91ms | 75ms | 100.0% |
   | stars | 49ms | 0ms | 100.0% |
   | **Average** | **154ms** | **83ms** | **95.1%** |

   Timing is a real strength of the pipeline overall -- 95% of notes
   land within half a second of ground truth.

   **Dug into the worst individual outliers (gaston, sleeping_beauty_
   wonder) and found two real, distinct, non-artifact root causes, both
   instances of an already-documented risk (architecture note: "don't
   trust individual ASR word timestamps for fine-grained boundaries"),
   just two different concrete triggers for it:**
   1. Gaston's outlier cluster at t=115-118s traces (confirmed via the
      debug log) to a single dense 13-word/~7-second `_group_words_by_
      gap` group ("one! When the rest can match nobody bites like
      Gaston! For there's no one,") -- imprecise interior ASR word
      timestamps in a passage this dense cascade into note-boundary-
      split errors for several neighboring words at once, not just one.
   2. sleeping_beauty_wonder's worst outlier ("I", ~2s early) traces to
      a large (~4s) ASR gap between word groups ("odd melody" ends
      34.17s, "I wonder," starts 38.08s) -- the zone boundary between
      them is placed at the ASR-timestamp MIDPOINT (36.13s), but a real
      sustained note appears to actually start around 35.7s, so part of
      what's musically still the previous word's note gets assigned to
      "I" instead, displacing it early.

   **Tested whether `--verify-placement` (the existing, off-by-default
   expand-search fix built for exactly this problem class) actually
   helps -- real, surprising, negative result on both songs:**

   | Song | metric | baseline | with verify-placement | delta |
   |---|---|---|---|---|
   | gaston | pitch-class | 93.6% | 92.2% | -1.4pp |
   | gaston | timing mean/median | 243/162ms | 413/176ms | worse |
   | gaston | timing <=500ms | 92.4% | 85.2% | -7.2pp |
   | sleeping_beauty_wonder | exact | 34.6% | 33.3% | -1.3pp |
   | sleeping_beauty_wonder | timing mean/median | 223/114ms | 339/119ms | worse |
   | sleeping_beauty_wonder | timing <=500ms | 86.5% | 80.4% | -6.1pp |

   `--verify-placement` DID correctly flag/fix some individual real
   problems (confirmed: several "Gaston" mentions got genuinely
   relocated to their correct position; it also correctly flagged the
   diagnosed "bites" as suspicious, just couldn't confidently relocate
   it within its search radius) -- but the NET effect on both tested
   songs was a real regression on every metric, not just noise. Likely
   mechanism (not fully confirmed): pass 3 re-running after a placement
   fix can shift NEIGHBORING words' zone/boundary calculations too,
   and the fix itself comes from the SAME kind of re-transcription-based
   text matching that's vulnerable to the same ambiguity classes (dense
   passages, repeated words -- Gaston says "Gaston" constantly) as the
   original problem. **This changes what CLAUDE.md previously said about
   `--verify-placement`** ("OFF by default... purely for COST, not
   reliability") -- that framing is no longer accurate on this evidence;
   there's a real, now-demonstrated reliability question too, not just a
   cost one. Don't turn this on by default without addressing that.

0j. **Built `lrc_timing.py` -- the LRCLIB synced-lyrics timing feature
   flagged as unbuilt in 0f/0g, now built as a DIAGNOSTIC-ONLY line-
   timing cross-check.** `apply_lrc_timing_check()`: parses LRC-format
   synced lyrics (`parse_lrc`), aligns pass-3's own lines (grouped by
   `Syllable.line_id`, same grouping `phrasing.py` uses for `-` breaks)
   against LRC's lines by text (same difflib technique used throughout),
   calibrates a per-song TIME offset (mode at 1s-bucket resolution --
   same robust-mode technique validated this session for
   `compare_full_pipeline_output.py`'s own timing comparison, not a
   plain mean/median), then FLAGS (never moves) any line whose residual
   after calibration exceeds `LRC_TIMING_FLAG_TOLERANCE_SEC` (2.0s).
   `--lrc-timing-check` / `config.ENABLE_LRC_TIMING_CHECK` (**False**,
   deliberately off by default).

   **Deliberately does NOT auto-correct, unlike pass 4's MXL pitch
   calibration** -- explicit, considered choice given 0h/0i's fresh
   lesson: `verify_placement` was built with the same good intentions
   for a related problem and produced a real, measured regression when
   validated end-to-end, despite fixing individual real cases. Shipping
   a correction here before confirming the signal itself is trustworthy
   would risk repeating that exact mistake.

   Real validation, 2 songs (both with synced lyrics available):
   - **gaston**: calibrated cleanly (+0.0s offset, 83% agreement over 24
     matched lines), flagged 2 lines near the end (t~189-191s). Ground
     truth doesn't have dense matched coverage in exactly that window to
     directly confirm those 2 specific lines, but there IS a confirmed
     real +2252ms error nearby (177.78s, same region, already found via
     0i's ground-truth timing work) -- suggestive, not conclusive.
     Notably did NOT flag the ALREADY-CONFIRMED "bites" cluster from
     0i -- expected: that was an INTERIOR-word error within an
     otherwise-correctly-anchored line, below line-level granularity.
     This is the real tradeoff of choosing line-level over word-level
     (the user's own explicit choice, given verify_placement's fresh
     regression made word-level feel riskier to attempt first).
   - **tarzan_son_of_man**: correctly found NO clear calibration (best
     candidate only 26% agreement, below the 40% bar) and skipped
     rather than guessing -- exactly the intended conservative behavior,
     and plausible given tarzan's own timing is already excellent
     (0i: 25ms median) so there may be no consistent single offset to
     find against whatever recording LRCLIB's synced version came from.

   **Only 2 data points, 1 producing flags -- not enough to validate
   the signal is trustworthy yet, and NOT enabled by default.** Next
   step if picked back up: run on more synced-lyrics songs, and
   specifically try to get denser ground-truth coverage in flagged
   regions to get a cleaner confirm/deny than gaston's sparse-coverage
   case allowed. Decision on whether to build actual correction (and
   what form it should take) is deliberately deferred until then.

0k. **Follow-up same-day run: `--lrc-timing-check` on 4 more songs
   (batb, stars, sleeping_beauty_ouad, little_mermaid) -- real, live
   LRCLIB fetches, real full pipeline. Result: all 4 ABSTAINED, none
   produced a calibration or any flags:**

   | Song | synced lines | our lines | matched by text | outcome |
   |---|---|---|---|---|
   | batb | 18 | 13 | 6 | skipped: best offset only 33% agreement (< bar) |
   | stars | 41 | 41 | 16 | skipped: best offset only 19% agreement (< bar) |
   | sleeping_beauty_ouad | 14 | 10 | 2 | skipped: only 2 matched (< 5 required) |
   | little_mermaid | 71 | 70 | 27 | skipped: best offset only 15% agreement (< bar) |

   Combined with the earlier 2-song round (0j: gaston calibrated
   cleanly at 83% agreement/24 matched; tarzan correctly abstained at
   26%), that's now **6 songs tested, 1 clean calibration (gaston),
   5 abstentions.** This is the diagnostic's conservative bucket-mode
   design working as intended -- it is refusing to guess rather than
   outputting a low-confidence offset -- but a 1-in-6 hit rate this low
   means the underlying signal (LRC-synced-lyrics line timing vs. our
   own) is USABLE far less often on real LRCLIB data than the earlier
   2-song sample suggested. Two effects are plausibly compounding, not
   yet separated:
   1. **Match-finding itself is strict and probably under-counting.**
      `apply_lrc_timing_check` matches at the WHOLE-LINE level (each
      line's tokens joined into a single string, difflib treats that
      whole string as one sequence element -- an "equal" opcode
      requires every word in the line to agree, one substitution/
      contraction/ASR slip anywhere in the line drops the whole line
      from the candidate pool). This is a stricter technique than the
      WORD-level whole-sequence alignment used everywhere else in this
      project (`lyrics_lookup.py`, `musicxml_reference.py`,
      `compare_full_pipeline_output.py`) -- none of those require a
      whole line to match atomically. Plausible real effect here, not
      yet confirmed: real true-positive line pairs are being missed
      whenever our own line-grouping (by `Syllable.line_id`) splits/
      merges lines differently than LRCLIB's own convention (already a
      confirmed real phenomenon, see 0f's "Under The Sea" merged-line
      example) or when a word inside an otherwise-correct line differs.
   2. **Among lines that DO match, agreement is often scattered rather
      than clustered** (19-33% best-bucket agreement on 3 of 4 songs
      here) -- consistent with either genuine per-line timing variance
      (no single constant offset exists against whatever recording
      LRCLIB's synced version came from) or the same repeated-line/
      wrong-instance mismatch class 0i already found and specifically
      engineered around for WORD-level timing comparison (bucketed-mode
      + 3s cutoff) -- `lrc_timing.py` already has an analogous bucketed-
      mode step, so it's already guarding against exactly this, but a
      chorus-heavy song could still have enough distinct wrong-instance
      pairings to prevent any bucket from dominating.

   **Not yet decided**: whether to invest in word-level line matching
   (denser candidate pool, mirrors the project's established alignment
   technique elsewhere) to see if match count/confidence improves, or
   to treat this low hit-rate as evidence the signal is too sparse on
   real LRCLIB data to be worth pursuing further as-is. Deliberately
   not choosing without the user's input, given `verify_placement`'s
   own precedent this session of a well-intentioned mechanism-level fix
   not translating into a real end-to-end win.

0l. **Investigated 0k directly with real data (temporary debug dump added
   to `apply_lrc_timing_check`, then reverted -- not shipped) on stars
   and little_mermaid, the two songs with the most matched lines. Found
   TWO distinct, compounding causes -- and the word-level-matching idea
   floated in 0k turned out NOT to be the fix, once tested:**

   1. **Confirmed real (recall problem): whole-line exact matching
      massively undercounts true line correspondences.** Dumping both
      sides' actual line text side-by-side showed the vast majority of
      "unmatched" our-lines are genuinely the same line as their LRC
      counterpart, just off by 1-3 words -- almost always short,
      truncated, or misheard ASR words that survived reference-lyric
      correction (`'the hu world is a mess'` vs LRC's `'the human world
      is a mess'`; `'un the sea'` vs `'under the sea'`, repeated
      throughout little_mermaid; `'they in for a worser worser fate'`
      vs `'they in for a worser fate'`). A fuzzy per-line ratio matcher
      recovered candidates for all 70/70 (little_mermaid) and 41/41
      (stars) of our own lines, vs. the current exact matcher's 27 and
      16 respectively -- confirms 0k's hypothesis on RECALL.

   2. **Found and NOT anticipated by 0k (the actual reason calibration
      still fails even with more candidates): the our-vs-LRC time delta
      is not a constant per song for either test case -- it DRIFTS
      smoothly and substantially over the song's length, at a strikingly
      similar relative rate on both:**
      - little_mermaid: delta grows from -0.5s near t=0 to -10.4s by
        t=119.5s (~-8.3% relative), then jumps to +5s for the last ~30s
        (likely a structural difference, e.g. a bridge/spoken segment
        LRCLIB's recording has that ours doesn't, or vice versa -- there's
        a ~30s gap in matched lines right where the sign flips).
      - stars: delta grows from -1.2s at t=24.6s to -12.7s by t=153.9s
        (~-8.9% relative), no flip (shorter song, simpler structure).

      Both songs drift at essentially the same ~8-9% relative rate despite
      being unrelated songs from unrelated LRCLIB entries -- too similar
      to plausibly be two independent "different recording, different
      tempo" coincidences; more likely points to something systematic in
      how a meaningful fraction of LRCLIB's synced-lyrics entries are
      timed (e.g. sourced from a sped-up/slowed-down video rip -- not
      confirmed, would need LRCLIB-side investigation to pin down further).
      **Ruled out our own pipeline as the cause**: little_mermaid's own
      note-timing accuracy against ground truth was independently
      validated as excellent in 0i (130ms mean / 75ms median / 95.8%
      within 500ms) -- a real, growing drift of 10+ seconds by mid-song
      is nothing like that, and 0i's ground-truth check has no
      relationship to LRCLIB at all, so this isn't the same error
      showing up twice.

      **This means the word-level-matching fix from 0k does NOT actually
      solve the calibration problem for these two songs, tested directly**:
      re-running the SAME bucket-mode calibration logic on the fuzzy
      matcher's fuller candidate list gave best-bucket confidence of 13%
      (little_mermaid) and 10% (stars) -- both slightly WORSE than the
      current exact matcher's 15% and 19%. More candidates just added more
      points scattered along the same drift curve; they don't cluster any
      better, because there genuinely isn't one constant offset to find.
      Confirms recall was never the bottleneck for these two songs --
      the drift is.

   **Implication**: fixing the line-matcher (0k's proposal) is real but
   insufficient on its own. A calibration model that could handle a
   *linear drift* (offset + rate, fit over matched-line deltas vs. time)
   rather than a single constant would be needed to get real use out of
   songs like these -- a materially bigger design change than "loosen the
   matcher," and one that pushes further toward eventual auto-correction
   (0j's own stated next gate) on a signal whose root cause (why the drift
   exists at all) still isn't confirmed. **Not started -- needs the user's
   direction before building**, per the same `verify_placement` caution
   already invoked in 0k: this project has one concrete precedent this
   session of a plausible-sounding mechanism-level fix not translating
   into a real win, and building a drift-tolerant calibrator is a bigger
   lift than that was.

0m. **Extended 0l's deep-dive (same debug dump, reverted again) to the
   remaining 4 tested songs (batb, sleeping_beauty_ouad, gaston, tarzan)
   -- corrects 0l's "consistent ~8-9% rate, probably systematic" theory:
   the real picture is genuinely per-song, not one universal cause.**

   - **gaston (the one song that calibrated cleanly): confirmed FLAT.**
     20/24 exact matches sit within +-0.3s of a single 0.0s offset --
     the only outliers are the 2 already-known late-song lines (0j,
     t~189-191s), a real localized error, not drift. Good control case
     any new calibration logic must not regress.
   - **tarzan: real drift, but much smaller than little_mermaid/stars**
     -- roughly -3.2% relative (vs. their ~8-9%), plus 2 clear wrong-
     repeat-instance outliers ("son of man" / "the power to be strong"
     both recur elsewhere in the song).
   - **batb: NOT a smooth drift -- a discrete STEP.** Two tight clusters
     of matches, ~+69s (t~82-86s) then a jump down to ~+14-15s (t~108-
     121s). Most likely a different edit/arrangement (an extended
     spoken/instrumental passage present in one recording but not the
     other), not a tempo mismatch -- a linear model would badly misfit
     this song; a robust/outlier-tolerant fit that can lock onto
     whichever cluster has more support is the right shape of fix, not
     "always assume one line."
   - **sleeping_beauty_ouad: still just 2 exact matches, both for a
     line that's part of the song's genuine call-and-response duet
     repeat** (already flagged unreliable for text-based alignment in
     0i) -- almost certainly wrong-instance noise, not real signal.
     Expect this song to keep correctly abstaining under any matching
     improvement.

   **Conclusion driving the actual implementation (0n)**: the fix isn't
   "always fit offset+rate" -- it's a robust estimator that tolerates
   outliers and only introduces a slope when the data genuinely supports
   one, so gaston stays a clean flat case, tarzan/little_mermaid/stars
   pick up a real slope, batb's minority cluster gets rejected as an
   outlier rather than blended into a wrong compromise line, and ouad
   keeps abstaining. All 6 songs' real our-line/lrc-line data was already
   captured in these dump runs, making it possible to prototype and
   validate the new calibration math OFFLINE against real data before
   touching the shipped pipeline again.

0n. **Shipped the drift-tolerant calibration designed in 0m, real-audio
   validated. `lrc_timing.py` rewritten:**

   1. **Matching**: the old whole-line EXACT match (`apply_lrc_timing_
      check`'s original `difflib.SequenceMatcher` over whole-line
      strings) is replaced by `_match_lines_word_level` -- one word-level
      whole-sequence alignment (still order-preserving, so still
      resistant to picking a wrong repeated-line instance), then a
      majority vote per our-line to decide which LRC line it really
      corresponds to. Fixes the undercounting root-caused in 0l (e.g.
      "the hu world is a mess" now matches LRC's "the human world is a
      mess" on the 3-of-4-words majority, where the old exact matcher
      dropped the line entirely).
   2. **Calibration is now two-tiered** (`config.LRC_TIMING_MIN_DRIFT_
      SAMPLES=10`, `LRC_TIMING_MIN_DRIFT_CONFIDENCE=0.5`, `LRC_TIMING_
      DRIFT_INLIER_TOLERANCE_SEC=1.5`, all new): tier 1 is the original
      constant-offset bucket-mode, UNCHANGED, tried first -- so a song
      that already calibrated cleanly (gaston) keeps using the exact
      same technique. Only if tier 1 fails does tier 2 (`_robust_linear_
      fit`, Theil-Sen median-of-pairwise-slopes) get a chance, gated
      stricter than tier 1 (more samples, higher inlier fraction) since
      a 2-parameter fit can trivially "fit" a handful of points.

   **Real end-to-end validation, shipped code path (not the temporary
   debug dump used for 0k-0m's investigation)**, `--lrc-timing-check` on
   gaston (regression check) + stars + tarzan (the two that should now
   calibrate per 0m's offline prediction):

   | Song | before (0j/0k) | after |
   |---|---|---|
   | gaston | constant +0.0s, 83%/24 matched | constant +0.0s, 81%/58 matched -- same tier, same offset, more candidates, no regression |
   | stars | skipped (19%/16) | **drift +0.8s, -0.0856s/LRC-s, 100%/39 matched, 0 flags** |
   | tarzan | skipped (26%, below bar) | **drift -0.5s, -0.0425s/LRC-s, 88%/32 matched, 3 flagged** (t~13.5s and two "son"/"Son" lines at t~137-139s -- both regions independently identified as real outliers during 0m's manual line-by-line inspection, not new/surprising) |

   **Follow-up same-day run completed the set -- little_mermaid, batb,
   and sleeping_beauty_ouad also confirmed on the shipped code path**,
   matching 0m's offline predictions closely:

   | Song | shipped-code result |
   |---|---|
   | little_mermaid | **drift +1.9s, -0.0712s/LRC-s, 56%/66 matched** (predicted 55%) -- 29/66 flagged, but NOT scattered noise: they resolve into two coherent secondary regions, a middle passage consistently ~-2.7s off (t~87-124s) and the late "Under the sea" reprise consistently ~+15.4s off (t~152-180s) -- both independently identified as real structural differences during 0m's manual inspection, not new |
   | batb | **still correctly skipped** -- "constant-offset best candidate +15.0s covers 31% (need 40%), drift fit only reached 31% inliers (need 50%)" -- confirms the two-cluster arrangement-edit jump still isn't fittable by either tier, exactly as 0m predicted |
   | sleeping_beauty_ouad | **constant +67.0s, 40%/5 matched** -- lands on the same borderline case flagged as a risk in 0m (this song's known duet-repeat structural ambiguity, not a new regression -- the same risk existed before under a different candidate set) |

   All 6 originally-tested songs are now confirmed on the real shipped
   code path, not just the offline prototype: 3 real wins (stars, tarzan,
   little_mermaid), 2 unchanged-safe abstentions (batb, and effectively
   sleeping_beauty_ouad given its known unreliability), 1 unchanged
   already-working case (gaston).

   `test_dry_run.py` gained a new synthetic test exercising both fixes
   at once: 11 lines with a real linear drift (spread across enough 1-
   second buckets that tier 1 must fail), one line with a deliberately
   wrong LRC word (recall check), and one genuine outlier (wrong-
   instance-style mismatch, must get flagged without dragging the fit
   off course) -- recovers slope/offset within tolerance and flags only
   the outlier. Existing tests (clean constant-offset case) unchanged
   and still pass, confirming tier 1 behavior is untouched.

   Still **OFF by default** (`config.ENABLE_LRC_TIMING_CHECK`,
   `--lrc-timing-check`), still purely DIAGNOSTIC (flags only, never
   auto-corrects) -- same reasoning as 0j: the flagging signal itself
   hasn't yet been cross-validated against real ground-truth timing
   error, and this project has one concrete `verify_placement` precedent
   this session of a well-intentioned mechanism-level fix not
   translating into a real end-to-end win. **That validation is now
   done -- see 0o, and the result is decisive: don't build
   auto-correction from this signal.**

0o. **Ground-truth cross-validation done (the gate 0j/0n called for) --
   DECISIVE NEGATIVE RESULT: flagged lines do NOT have larger real
   timing error than unflagged ones. Don't build auto-correction from
   this signal as currently designed.**

   User asked directly: "we were doing this so that we could start using
   the lyric timing to improve accuracy, correct?" -- yes, that was
   always the eventual goal, but per the module's own docstring
   auto-correction was gated behind confirming flagged lines actually
   correlate with real problems first (exactly the check that would have
   caught `verify_placement`'s regression earlier if it had existed
   then). Built that check now: word-level whole-sequence text alignment
   between our own output and each song's real ground-truth `notes.txt`
   (shipped SingStar data, same source used throughout this project's
   pitch validation), with the same repeat-instance safety filter 0i
   already established (bucket deltas at 200ms resolution, keep only
   candidates near the dominant cluster) -- scratchpad-only, not kept in
   repo (same category as `compare_full_pipeline_output.py`).

   Real results, the 2 songs (of the driftcal reruns) that actually
   produced flags with ground truth available:

   | Song | flagged mean real error | unflagged mean real error | unflagged median |
   |---|---|---|---|
   | tarzan_son_of_man | 0.03s (n=1 with a nearby GT match) | 0.05s (n=19) | 0.02s |
   | little_mermaid | 0.11s (n=15) | 0.19s (n=32) | 0.10s |

   Flagged lines are AS ACCURATE OR MORE ACCURATE than unflagged ones on
   both songs -- the opposite of what would be needed to justify
   correction. Individual flagged-line real errors (little_mermaid, all
   with a nearby GT match): 0.01-0.45s, nothing remotely close to the
   multi-second residuals `lrc_timing.py` itself reported for these same
   lines (e.g. 'Under' flagged at +15.4s residual, real error 0.08s).
   This confirms what 0m/0n's investigation already suspected but hadn't
   directly proven: the drift `lrc_timing.py` detects reflects LRCLIB's
   synced lyrics being timed to a DIFFERENT recording/arrangement than
   ours, not a defect in our own output -- our own timing was already
   correct at those exact positions.

   A 3rd song (Chicago - When You're Good to Mama, newly added to the
   test set by the user, real ground-truth `.txt` provided alongside the
   audio) calibrated cleanly with ZERO flags (constant +0.0s, 57%/44
   matched) -- no comparison possible there, but its 37 unflagged matches
   averaged 0.30s/0.12s (mean/median) real error, a sane baseline
   consistent with this project's other established timing-accuracy
   numbers (0i: 154ms/83ms average across 7 songs).

   **Conclusion: do not build auto-correction on top of the current
   `lrc_timing.py` line-level drift/offset mechanism.** The signal it
   flags is real (LRCLIB and our own recording genuinely disagree) but
   doesn't mean what correction would need it to mean (that OUR output
   is wrong). This isn't a "gate not yet cleared, try again later"
   situation -- it's evidence the current mechanism measures the wrong
   thing for this purpose. If synced-lyrics timing is revisited as an
   accuracy signal in the future, it would need a fundamentally
   different approach (e.g. something that can distinguish "LRC's
   recording differs from ours" from "our own note-to-word timing is
   wrong", which line-level offset/drift alone cannot do) -- not a
   refinement of the existing calibration mechanism. Given this, the
   diagnostic-only behavior stays as-is (it's still useful as a coarse
   "these two recordings differ here" signal, e.g. for the user's own
   manual review), but this closes out the "use LRC timing to improve
   accuracy via correction" thread as originally envisioned.

   **CORRECTION, same day, found while validating a follow-up idea
   (0p): the comparison script above had a real parsing bug that
   undercounted matches by 5-10x.** Its regex used `\s+` for the
   separator right before the note-text field, which greedily consumed
   BOTH the field separator AND (when present) the leading space that
   marks a new word in our own pipeline's output convention -- so most
   words were silently glued onto their neighbor and could then only
   ever match ground truth by coincidence. Made worse by a second,
   independent discovery: SingStar's shipped `notes.txt` files don't
   even use that leading-space convention at all -- they mark word
   starts with a TRAILING space instead (e.g. `"Oh "` / `"the "` /
   `"po"` + `"wer "`), completely different from our own pipeline's
   format. Fixed by making the parser convention-agnostic: concatenate
   the whole note-text stream into one string and split on whitespace
   wherever it falls, rather than assuming which side the marker space
   is on.

   Re-ran with the fixed parser -- match counts jumped 5-10x (tarzan
   19->164, little_mermaid 34->336, chicago 37->207), and the resulting
   accuracy numbers now land close to this project's own previously-
   established ground-truth figures (0i's compare_timing() table),
   which is good independent confirmation the fix is correct rather than
   just different:

   | Song | flagged mean real error | unflagged mean real error | unflagged median |
   |---|---|---|---|
   | tarzan_son_of_man | 0.02s (n=2) | 0.12s (n=163) | 0.02s |
   | little_mermaid | 0.21s (n=26) | 0.17s (n=333) | 0.08s |

   **The headline conclusion survives, but is less clean-cut than first
   reported.** tarzan's flagged lines are still MORE accurate than
   unflagged (even more so with the larger sample). little_mermaid
   flipped from "flagged slightly better" to "flagged slightly worse"
   on the mean (0.21s vs 0.17s) -- and with the fixed parser, 2 of its
   26 matched flagged lines now show real errors that are actually
   non-trivial (0.90s "Play", 1.16s "Under") vs. the rest sitting at
   0.01-0.49s. That's a weak positive signal buried in mostly noise, not
   the clean "flagged==accurate" result originally reported, but also
   not strong or consistent enough on its own to justify auto-correction
   -- the majority of flagged lines are still small real errors, and
   tarzan still points the other way entirely. **Verdict unchanged**:
   don't build auto-correction from this signal as currently designed.

0p. **Tried "snap zone/word boundaries to a nearby pass-1 note onset"
   (`--zone-boundary-snap`, EXPERIMENTAL) for the timing gap identified
   at the top of this thread (dense ASR passages / large ASR gaps
   causing a zone boundary to land near, but not exactly at, a real
   note onset -- e.g. gaston's "bites" cluster, sleeping_beauty_wonder's
   early "I"). REAL END-TO-END RESULT: no improvement on any of 5 tested
   songs, mild regression on most. Don't pursue further as designed.**

   Both `_assign_notes_to_groups` (between-GROUP zone boundaries) and
   `_split_notes_by_word_boundaries` (within-group WORD boundaries)
   compute a boundary as the clamped-monotonic midpoint of two ASR
   timestamps, with zero reference to where pass-1 notes actually
   begin -- confirmed via the docstrings/code as the same mechanism in
   both places. New `_snap_boundary_to_note_onset` (`lyric_alignment.py`)
   refines that raw boundary by snapping it to a pass-1 note START when
   exactly ONE exists within `config.ZONE_BOUNDARY_SNAP_RADIUS_SEC`
   (0.5s) -- deliberately conservative: zero candidates (nothing to snap
   to) or multiple (ambiguous which one is "the" boundary) both leave
   the raw ASR midpoint untouched, same "only act when confident"
   principle `verify_placement` uses. Threaded through as an opt-in
   param (`snap_boundaries`, default `config.ENABLE_ZONE_BOUNDARY_SNAP
   = False`) all the way from `main.py`'s new `--zone-boundary-snap`
   flag down to both boundary functions -- zero effect on default runs.
   Synthetic test added (`test_dry_run.py`) reproducing the
   sleeping_beauty_wonder "I" bug shape exactly: a real note onset 0.4s
   before the raw ASR-midpoint boundary reassigns correctly when
   snapping is on, is a no-op when off, and a second synthetic case
   confirms the ambiguity guard (two candidate onsets in range -> no
   snap).

   **Real end-to-end validation** (same word-level ground-truth
   comparison technique validated in 0o, same parser-bug fix applied,
   large samples): baseline (no snap) vs. `--zone-boundary-snap`, all 5
   songs with real ground truth:

   | Song | baseline mean/median/<=500ms | snap mean/median/<=500ms |
   |---|---|---|
   | batb | 205ms/135ms/94% (n=108) | 219ms/136ms/94% (n=108) |
   | tarzan_son_of_man | 121ms/25ms/96% (n=164) | 128ms/25ms/96% (n=164) |
   | little_mermaid | 165ms/75ms/94% (n=336) | 165ms/79ms/94% (n=336) |
   | chicago | 161ms/107ms/97% (n=207) | 170ms/114ms/97% (n=207) |
   | ordinary_day | 145ms/97ms/98% (n=133) | 146ms/97ms/98% (n=133) |

   **Not a single song improved.** Confirmed the snap mechanism is
   genuinely firing (diffed the two runs' `[DEBUG LOG]` NOTE-ZONE
   ASSIGNMENT sections for batb -- boundary values really do differ,
   e.g. one boundary moved from 85.0125s to 84.81668...s, matching a
   real nearby note start) and that pass-1 itself is unaffected (batb's
   two `[PASS1 DEBUG]` files are byte-identical, confirming
   `isolation_source="rmvpe"`'s documented reproducibility holds here
   too) -- so the flat-to-negative result is a real effect of the
   snapping logic, not noise from an unrelated source. Same shape of
   outcome as `verify_placement`'s own real-audio validation: a
   well-motivated, synthetically-verified mechanism that doesn't
   generalize positively once tested end-to-end -- this is now a SECOND
   independent instance of that exact pattern in this project. Likely
   explanation (not confirmed further): real audio has messier, more
   numerous onset candidates than the clean synthetic test case, so
   "exactly one confident candidate nearby" fires on more false
   positives (snapping to an onset that ISN'T really the word boundary
   -- a consonant transient, backing-track bleed, etc.) than true
   positives in practice.

   **Decision: keep `--zone-boundary-snap` in the codebase (off by
   default, as implemented) but do not invest further in this
   direction.** Consistent with `verify_placement`'s own precedent --
   kept as an available option, not deleted, but not a promising avenue
   to keep refining (tighter radius, different candidate-selection
   logic, etc.) without a fundamentally different signal than "nearby
   pass-1 onset," which this result suggests isn't precise enough on
   its own.

0q. **Investigated the reference-line-vs-musical-phrase tension flagged
   in 0f ("Under The Sea" merging into one line when it's really two
   repeated phrases) -- ACCEPTED as a known limitation, not building a
   fix.** `--zone-boundary-snap`'s own predecessor of the idea
   (cross-checking against LRCLIB's SYNCED lyrics for a different line
   split) was tried first and doesn't help: checked little_mermaid's
   own cached synced-lyrics dump directly -- LRCLIB's SYNCED version
   merges "under the sea under the sea" into one line just like its
   PLAIN version does, so there's no second LRCLIB signal to
   cross-check against for this case.

   Confirmed the bug is real and currently live: little_mermaid's actual
   output has "under the sea? Under the sea," as ONE undivided line
   (~8 syllables, under the 12-syllable 1.5x safety net) because both
   chorus repetitions share one reference line_id and `phrasing.py`'s
   `known_same_line` rule never breaks those except for the length
   safety net.

   Proposed and validated (real-data scan, not implemented) a candidate
   fix: detect an EXACT repeated multi-word sub-phrase within a
   reference line as a targeted signal (distinct from a generic gap-
   duration heuristic, which can't tell "real pause mid-phrase" from
   "two merged phrases" -- that's exactly why `known_same_line` exists
   in the first place). A naive version (any repeated 2+-word n-gram
   anywhere in the line) was scanned against 494 real lines across all
   11 test-set songs: only 2 of 28 flagged lines were genuine (little_
   mermaid's "under the sea" chorus x3, ordinary_day's "it's all right
   it's all right") -- the rest were false positives from (1) a single
   word repeated as an ad-lib/refrain run (gaston's "town town town...",
   ordinary_day's "wayheyhey..." runs, sleeping_beauty_ouad's "no no
   no") and (2) a short 2-word fragment reused inside one otherwise-
   flowing sentence (javert_suicide's "the law ... the law", stars'
   "those who ... those who"). A refined rule -- require the repeat to
   be ADJACENT and cover (nearly) the WHOLE line, with the repeated unit
   itself containing >= 2 DISTINCT words -- correctly kept only the 2
   genuine cases and rejected every false positive in the scan.

   **User's decision: leave phrasing.py as-is, mark this an accepted
   issue rather than build the refined fix.** Hand-editing the rare
   real case is an acceptable mitigation; this doesn't warrant another
   mechanism given the project's now-two-time experience (0p,
   `verify_placement`) of well-validated-looking fixes still carrying
   real risk once shipped. If picked back up later: the refined
   adjacent-and-full-line-coverage design above is already validated
   against real data and ready to implement, not just a sketch.
1. **Key correction (`key_correction.py`) was later root-caused as
   actively harmful and has been REMOVED ENTIRELY from the codebase.**
   (Superseded: this item originally reported switching it to `music21`'s
   Krumhansl-Schmuckler key-finding and enabling it by default -- that
   default was reverted the same session it was tried, see `config.py`'s
   own historical comment before removal, and the pass never came back
   on by default afterward.) Root cause, confirmed via a direct
   pass-1-vs-pass-2 diff on a real song ("Stars", real key confirmed E
   major by the user): a single global detected key applied blindly to
   every note will snap legitimate out-of-scale notes (e.g. deliberate
   modal-mixture/borrowed tones) to the wrong pitch -- dozens of notes
   changed throughout the song, including blanket-snapping every
   legitimate C-natural to B just because C isn't diatonic in E major.
   Confirmed still off/unused by any real workflow before deletion, so
   this was a clean removal: `key_correction.py` deleted; the
   `--key-correction`/`--no-key-correction` CLI flags, `--no-pass2-debug`
   flag, `[PASS2 DEBUG]` debug file, `ENABLE_KEY_CORRECTION` config
   constant, and `PipelineOptions.key_correction`/`no_pass2_debug` fields
   all removed; the GUI's "Key correction" checkbox removed. `music21`
   itself stays a dependency -- pass 4 (`musicxml_reference.py`) and
   `file_discovery.py` use it independently for MusicXML parsing.
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
   `--verify-placement` to enable). Originally reasoned to be purely a
   COST tradeoff (an expand-search re-transcription loop over every
   word, ~4 minutes on top of `verify_words`' own ~4 minutes), not a
   reliability one — **that turned out to be wrong, see 0i below**: real
   end-to-end testing on gaston and sleeping_beauty_wonder found a net
   REGRESSION on every pitch/timing metric on both songs, despite
   correctly fixing some individual real problems. There's a genuine
   reliability question here too now, not just cost — don't default
   this on without addressing it.

Feedback from this round, worth carrying forward: **new features should
default to ON** (not gated behind an opt-in flag) unless there's a
specific reason given for opt-in (like key-correction's original
flattening-risk concern, since resolved), and **every new/changed
feature needs a real-audio validation run** against `sandbox/Les
Misérables - Stars.ogg` before being reported as done, not just
`test_dry_run.py` — the "Stars" reference notes (opening 6 notes, and
the full "fall as Lucifer fell...sword" section) are saved in this
session's memory for that purpose.

## Folder-based input, mp4/avi support, embedded cover art, YouTube input,
## existing-file verification, batch mode, GUI, launcher (2026-08-08)

Large feature set, planned and tracked via a phased plan (Phase 0 through
F) approved with the user before implementation. Each phase gets its own
real-audio/real-media validation, not just `test_dry_run.py`, per this
project's own established policy above.

**Phase 0 (foundational refactor) -- DONE.** `main.py`'s CLI entry point
was a single 300-line `run(argv) -> int` doing everything from arg-parsing
through writing the final `.txt`, `sys.exit`-driven throughout -- nothing
later in this feature set (existing-file verification's early-return,
batch mode's per-song exception isolation, the GUI calling the pipeline
in-process) could be built cleanly on top of that shape. Extracted the
whole body into `run_pipeline(input_dir, output_dir, opts: config.
PipelineOptions, *, log=print) -> PipelineResult` -- never calls
`sys.exit`, catches every "expected" failure (bad path, no notes/words
detected, etc.) into `PipelineResult(success=False, error=...)` instead.
`run()` is now a thin CLI wrapper: parse args -> build `PipelineOptions`
-> call `run_pipeline` -> map to an exit code. Every `print(...)` in the
extracted body became `log(...)` (mechanical, real work across dozens of
call sites) -- deliberately did NOT thread `log` further down into
submodules (`note_detection.py`/`musicxml_reference.py`/`lyrics_lookup.py`/
`verification.py` all still `print()` directly) to avoid touching the
pitch/timing code this project has been careful with all session; the GUI
(Phase E) captures that output via `contextlib.redirect_stdout` at the
call boundary instead. `config.PipelineOptions` (had existed as dead code,
zero call sites, referencing a `genius_token`/`device` field from a
since-replaced lyrics source and a since-removed CPU fallback path) was
replaced with a version covering every real `PipelineOptions` field.
**Verified**: `test_dry_run.py` green throughout; a real-audio run through
`run_pipeline` against `sandbox/Beauty And The Beast - Beauty And The
Beast/` produced a pass-1 debug file BYTE-IDENTICAL to a pre-refactor run
(confirms `isolation_source="rmvpe"`'s documented reproducibility held,
and the refactor introduced zero behavioral drift) -- a full mocked-
everything synthetic smoke test was considered and deliberately skipped
as disproportionate effort (would need to fake Demucs/torch.cuda/
WhisperX/lyrics-fetch simultaneously) given the real-audio diff already
gives a stronger, more direct answer for a mechanical extraction.

**Phase A (folder-based input, mp4/avi, embedded cover art) -- DONE.**
The CLI's positional argument is now a FOLDER, not a single file
(`--output-dir` is now REQUIRED and must differ from the input folder --
enforced with a hard check at the top of `run_pipeline`). New
`file_discovery.resolve_primary_source(input_dir, audio_file_override)`
decides what's inside: exactly one real audio file (`config.AUDIO_EXTS`)
-> normal path unchanged; none but exactly one `.mp4` -> that file serves
as BOTH `#MP3` and `#VIDEO` directly (confirmed working in the user's
UltraStar Deluxe install -- no separate audio extraction needed for
*output*, only an internally-cached wav for our own Demucs/pass-1/
WhisperX analysis, since nothing in this codebase had ever tested feeding
Demucs an mp4 directly); none but exactly one `.avi` -> its audio track
is extracted into a real standalone mp3 (new `media_extract.py`,
generalizing `video_sync.py`'s own pre-existing ffmpeg subprocess pattern
-- that module's private `_extract_audio_wav` was deleted in favor of the
shared `media_extract.extract_audio_track`), and the avi itself becomes
`#VIDEO`; an avi with NO audio track aborts cleanly (`media_extract.
has_audio_stream`, ffprobe-based, checked BEFORE attempting extraction --
never assumes ffmpeg's own failure mode is informative enough on its
own); more than one candidate at any tier -> `AmbiguousInputError` naming
the candidates, requiring a new `--audio-file <name>` override (decided
with the user: never silently guess which file is the song). New
`song_input.resolve_song_folder` orchestrates this plus the pre-existing
`find_companions` plus a NEW fallback: if no `.jpg`/`.jpeg` companion was
found, `cover_extract.extract_embedded_cover` (new module, finally wires
up `mutagen` -- listed in `requirements.txt` since early in this project
but never actually called until now) tries ID3 APIC (mp3), MP4 `covr`
atom, FLAC's native picture list, and OGG/Opus's base64 vorbis-comment
picture block, sniffing the real image type from magic bytes rather than
trusting a container's claimed MIME type. New `output_staging.
stage_companions_to_output` copies whichever companions the output
actually references into the output folder (feature 2's requirement) --
runs once, late, and correctly copies an identical mp3_src/video_src
(the mp4-as-audio case) only ONCE despite serving both roles. New
`ResolvedInput.videogap_applicable` flag skips `estimate_videogap`
entirely (not just "no video") when the video and audio are the same
file or one was extracted directly from the other -- correlating a
signal against itself or a trimmed copy of itself is meaningless.
`work_dir` simplifies to `input_dir / ".ultrastar_work"` (directly
available now that input is already a folder, same cache-reuse guarantee
as before).

**Verified for real**, not just via `test_dry_run.py`'s new synthetic
tests (`resolve_primary_source`'s branching, `cover_extract`'s per-format
extraction + magic-byte sniffing, `output_staging`'s copy-and-dedup
logic): built real fixture files via ffmpeg (a real mp4 with an audio
track, a real avi with an audio track, a real avi with NO audio track, a
real mp3 with a mutagen-embedded ID3 cover, two real mp3s in one folder)
and ran `resolve_song_folder` against each directly -- all 5 scenarios
passed, including the avi-no-audio clean abort and the ambiguous-folder
error + `--audio-file` override. Then two REAL FULL PIPELINE runs: (a) a
regression check on `sandbox/Beauty And The Beast - Beauty And The
Beast/` through the new folder-based CLI -- pass-1 note count matched
the Phase 0 baseline exactly (182), and the output folder correctly
contained a COPY of `music.ogg` (feature 2 confirmed working, not just
staged-but-untested); (b) a real mp4-as-audio run using gaston's own real
music-video mp4 (copied, not moved, into an isolated test folder so
gaston's own real sandbox data was never touched) -- confirmed `#MP3` and
`#VIDEO` both correctly point at the same mp4 in the final `.txt`, NO
`#VIDEOGAP` line was written (confirms the degenerate-self-correlation
guard fired -- no "Estimating VIDEOGAP" log line appeared either), and
pass-1/transcription produced sane, non-degenerate numbers (518 notes,
392 words) close to this same song's own previously-recorded numbers
from earlier in this session (505 notes via its normal mp3 companion --
the small difference is consistent with a different audio re-encode
through the mp4 container plus this project's own already-documented
pass-1 non-determinism, not a sign of something broken).

**Phase B (existing-file verification) -- DONE.** New `usdx_parser.
parse_usdx_file` parses an UltraStar `.txt` back into structured
`Syllable`/`LineBreak` data -- the exact inverse of `usdx_writer.
render_song`'s grammar, using a new `tempo.beat_to_seconds` (the missing
inverse of `seconds_to_beat`). Tolerant of `,` as the BPM decimal
separator (real files in the wild use it, even though this project's own
writer only ever emits `.`) and a leading `P1`/`P2` duet marker (parses
P1 only). Fails closed (`UsdxParseError`) on anything structurally
invalid -- never partially trusts a malformed parse.

New `verify_existing_song.py`, shaped like `musicxml_reference.py`'s
calibrate-then-compare pattern but calibration-free (this compares two
timelines of the SAME audio, not two independently-timed recordings the
way `lrc_timing.py`'s LRCLIB comparison does): word-level whole-sequence
text alignment, pitch compared at PITCH CLASS (mod 12, matching how
UltraStar Deluxe itself scores), timing compared with the same
repeat-instance bucketing guard validated in 0i (a repeated chorus/line
can otherwise pair against the wrong sung instance). `verdict` is
`"COULD_NOT_VERIFY"` (never `"PASS"`) whenever too few words matched to
trust the comparison at all. New config constants
(`EXISTING_TXT_MIN_MATCHED`/`_MIN_PITCH_ACCURACY`/`_TIMING_TOLERANCE_SEC`/
`_MIN_TIMING_AGREEMENT`); **`ENABLE_EXISTING_TXT_CHECK` defaults OFF** --
the one deliberate exception to this project's "new features default on"
convention, decided with the user: unlike every other on-by-default
feature here (which only ever ADDS a correction), this one can result in
NOT writing output the user expected on a plain re-run.

Wired into `run_pipeline`: an existing file is detected early (either an
explicit `--existing-txt <path>`, always wins, no filename-matching
needed -- same convention as `--musicxml-reference`; or auto-detected via
`--existing-txt-check` matching `"<Artist> - <Title>.txt"` in the input
folder) but only actually parsed/compared LATE, after pass 3/4, once a
fully fresh syllable sequence exists. On `PASS`, the EXISTING file is
copied byte-for-byte into `output_dir` (not the freshly-built `Song`) and
`PipelineResult.regenerated = False`; on `PROBLEMS_FOUND`/
`COULD_NOT_VERIFY`, proceeds exactly as a normal run. Critically,
companion staging (Phase A) runs on BOTH branches, unconditionally --
`output_dir != input_dir` still needs a self-contained output folder even
when the existing file is kept as-is.

**Verified for real**, addressing the plan's own flagged uncertainty
about whether the default thresholds are actually calibrated against
real self-noise (pass-1 pitch detection is not fully reproducible
run-to-run even on identical audio, per this project's own documented
CREPE/RMVPE non-determinism) rather than picked by analogy: ran
`sandbox/Beauty And The Beast - Beauty And The Beast/` through the real
pipeline twice -- once normally, once again pointing `--existing-txt` at
the FIRST run's own output file. Result: 115 words matched, **100%
pitch-class accuracy, 100% timing agreement -> PASS**, the second run's
output folder ended up byte-identical to the first (confirmed via `diff`)
plus still had its own copy of the companion audio file (confirms the
"stage on both branches" rule actually holds in the real pipeline, not
just in the plan). Then ran a THIRD time pointing `--existing-txt` at a
deliberately-corrupted copy of the same file (every pitch shifted +5
semitones, a real non-multiple-of-12 error) -- correctly landed on **0%
pitch-class accuracy -> PROBLEMS_FOUND**, and correctly fell through to
writing a fresh, uncorrupted regeneration rather than keeping the bad
file. Both the PASS/keep and PROBLEMS_FOUND/regenerate paths are now
real, end-to-end confirmed, not just unit-tested.

**Phase C (YouTube input) -- DONE.** New dependency `yt-dlp` (optional,
only imported when `--youtube-url` is actually used, same
graceful-degrade convention as `whisperx`/`torchcrepe`). New
`youtube_source.download_youtube_source(url, dest_dir, audio_only)`
downloads to a deterministic filename (`youtube_download.mp3` or `.mp4`
-- deliberately NOT named from the video's own title, since that isn't a
reliable "Artist - Title" source, which is exactly why `--youtube-url`
requires `--artist`/`--title` explicitly). New CLI flags `--youtube-url`,
`--youtube-audio-only` (default ON -- matches the user's own stated
common case of not wanting the video) / `--youtube-video`.

**Key design simplification found during implementation**: the download
step just lands directly IN `input_dir`, then falls through to the exact
same folder-resolution logic Phase A already built -- an otherwise-empty
folder containing one freshly-downloaded mp3 or mp4 is auto-classified
correctly (`kind="audio"` or `"mp4_as_audio"`) with ZERO special-casing
needed in `song_input.py`/`file_discovery.py`. The original plan
considered a dedicated bypass of `resolve_primary_source`'s
auto-detection for this case; turned out to be unnecessary once actually
implemented -- natural auto-detection already does the right thing.
Wired inside `run_pipeline` itself (not just the CLI wrapper), so the
GUI (Phase E) gets YouTube support for free by setting the same
`PipelineOptions` fields, no separate code path to build there.

**Verified for real** (per the plan's own stated requirement -- a real
download, not just mocked): a new synthetic test (fake `yt_dlp` module,
same `sys.modules` injection convention this project's test suite
already uses for `whisperx`/`librosa`/`requests`) covers
`download_youtube_source`'s own logic -- deterministic output filename,
`YoutubeDownloadError` on a download failure -- without needing network
access for every test run. Then real, live downloads against a short
(19s), well-known, stable public test video: both audio-only (produced a
real 305KB mp3) and video mode (downloaded separate video+audio streams
and correctly muxed them into a 534KB mp4) worked directly. Then a full
REAL pipeline run through `--youtube-url` end to end: downloaded audio,
correctly auto-classified as `kind="audio"`, ran Demucs/pass-1/
WhisperX/pass-3 for real (transcribed actual real speech from the
video), wrote a real output `.txt` with `#MP3:youtube_download.mp3`, and
correctly staged a copy of the downloaded mp3 into the output folder
alongside it (companion staging from Phase A working correctly for a
YouTube-sourced file too, not just local ones).

**Phase D (batch mode) -- DONE.** New `batch.py`'s `run_batch(parent_dir,
output_parent_dir, opts, log=print) -> List[Tuple[str, PipelineResult]]`
runs `run_pipeline` once per IMMEDIATE subdirectory of `parent_dir`
(never the parent itself), catching even an exception `run_pipeline`
itself didn't already turn into a `PipelineResult` -- one bad song must
never abort the rest of the batch. Output mirrors the input 1:1
(`output_parent_dir/<song folder name>/`). New `--batch` CLI flag.
`work_dir`-per-song falls out of Phase 0's design for free (each
subfolder is its own `input_dir`, so each naturally gets its own
`.ultrastar_work`) -- the only new guard needed is rejecting `--batch`
together with `--work-dir` (a shared override would collide every song's
Demucs cache into one directory), and also with `--artist`/`--title`/
`--existing-txt`/`--youtube-url` (none of which make sense applied
identically across multiple different songs) -- checked and rejected
with a clear error BEFORE any processing starts, not discovered
mid-batch. `run()` exits `0` if every song succeeded, `2` (distinct from
the single-song failure code `1`) if any song failed -- lets scripts
tell "total failure" apart from "partial batch failure."

**Verified for real**: built a real parent folder with 3 subdirectories
-- 2 real songs (Chicago, Ordinary Day; audio + their already-cached
`.ultrastar_work` copied over so this didn't need to re-pay Demucs
separation) and 1 deliberately empty/broken folder -- and ran `--batch`
against it. Confirmed: the broken folder failed with a clean
`NoAudioSourceFoundError` message (not a stack trace) and the batch
CONTINUED to the next song rather than aborting; both real songs
succeeded end-to-end; the final per-song summary correctly reported
"2/3 succeeded" with a clear per-song breakdown; the output folder
correctly mirrored the input structure (`output/Chicago/`,
`output/OrdinaryDay/`, using the SUBFOLDER's own name, not the parsed
artist/title -- and no `output/BrokenEmpty/` at all, since that song
never got far enough to write anything); exit code was `2` as designed.
Also directly confirmed the CLI rejects `--batch --work-dir ...` and
`--batch --artist ... --title ...` with a clear error before any
processing starts.

**Phase E (Tkinter GUI) -- DONE.** New `ultrastar_generator/gui.py`
(stdlib `tkinter`/`ttk` only, per the user's decision -- no new
dependency). Wraps the exact same `run_pipeline`/`run_batch` the CLI
uses -- there is only ever one real pipeline implementation, the GUI is
purely a front end over it. Mode selector (single folder / batch parent
folder / YouTube URL) with dynamic field show/hide; folder pickers via
`filedialog.askdirectory`; a curated subset of the ~30 CLI flags on the
main surface (fetch-lyrics, verify-words,
verify-placement, existing-txt-check, musicxml-force-calibration,
whisper-model, pitch-source), with the rarer/experimental flags
(`--lrc-timing-check`, `--zone-boundary-snap`, `--no-video-sync`,
`--quiet`) behind a collapsible "Advanced" section rather than
cluttering the main view. CUDA availability checked once at startup
(disables Run + shows the error inline, mirroring `run()`'s own single
check). No mid-run Cancel in v1 (Demucs/WhisperX calls aren't cleanly
interruptible without significant extra plumbing) -- documented as a
known limitation. No image preview (Tkinter's native `PhotoImage` only
handles PNG/GIF/PPM, not JPEG, without adding Pillow, which isn't a
current dependency) -- out of scope unless requested later.

Runs the pipeline on a background `threading.Thread` (Tk itself isn't
thread-safe -- only the main thread may touch widgets) and captures
ALL of its output -- including `print()` calls from deep inside
pitch/timing submodules that were deliberately NOT rewired to accept a
`log` callback (see Phase 0) -- via `contextlib.redirect_stdout` at the
call boundary, feeding a `queue.Queue` that a `self.after(100, ...)`
polling loop drains on the main thread into the log `Text` widget. This
is a correction from an earlier draft of the plan, which had leaned
toward threading a `log` callback deep into submodules instead --
`redirect_stdout` at the boundary is both lower-risk (touches zero lines
in the pitch/timing code) and higher-coverage (catches everything, not
just what a threaded callback happened to reach).

**Verified for real**, per the plan's own stated requirement (a real
interactive run, not just construction): first, a smoke test constructing
the real `App` and exercising mode-switching/option-building without
entering the event loop (`update_idletasks()` needed to actually flush
Tkinter's geometry manager -- confirmed the visibility-toggle logic for
YouTube fields / the Audio-file-override field / the Advanced section all
correctly show/hide per mode, not just that they don't crash). Then a
full REAL run driven programmatically through the actual GUI mechanism
(set the real input/output folder StringVars, call the real `_on_run()`,
pump the Tk event loop in a loop while the real background thread
processed `Chicago - When You're Good to Mama` end to end, reusing its
already-cached Demucs separation from earlier in this session -- 47s
total): confirmed the Run button was disabled the instant the run
started and re-enabled only after it finished, the log widget received
6462 characters of LIVE output during the run (not just a final dump),
and the correct real output `.txt` was written with the right artist/
title/cover/background (parsed from the filename with no override
needed, same as the CLI path). Also directly launched the real GUI
process (not just constructed the `App` object) to confirm no startup
crash under `python -m ultrastar_generator.gui`.

**Phase F (launcher) -- DONE, and the whole 10-feature plan is now
complete.** New `run_gui.bat` at the repo root, following the exact same
`BATCH_DIR`/`VENV_PATH` convention `launch_env.bat`/`setup.bat` already
use (so all three stay consistent if the venv location ever changes).
Checks the venv actually exists first (clear error + `pause` pointing at
`setup.bat` if not, rather than a cryptic failure) before launching
`venv\Scripts\pythonw.exe -m ultrastar_generator.gui` via `start ""` --
`pythonw.exe`, not `python.exe`, so no console window appears alongside
the GUI window itself.

**Verified for real**: invoked `run_gui.bat` directly via PowerShell the
same way Windows Explorer would (a fresh double-click, not from an
already-activated dev shell) -- confirmed it correctly spawned a real
`pythonw` process (no console window, no manual venv activation step
needed) and cleaned up afterward.

**This closes out the full folder-based-input/mp4-avi-support/embedded-
cover/existing-file-verification/YouTube-input/batch-mode/GUI/launcher
feature set.** All 7 phases (0, A, B, C, D, E, F) done, each with its own
real-audio/real-media/real-interactive verification, not just synthetic
tests -- `test_dry_run.py` sits at 76 passing checks by the end (up from
the session's earlier baseline), still green throughout. Nothing was
committed to git during implementation (per this project's "only commit
when the user asks" convention) -- everything from Phase 0 onward is
still sitting as uncommitted working-tree changes as of this writing.

## GUI polish, key-correction removal, interactive LRCLIB lyrics
## selection (2026-08-08)

A follow-up 16-item request, planned and tracked as 7 phases (G1-G7),
addressing rough edges found using the GUI built in the previous section
plus two substantial new pieces of work. Each phase real-verified per
this project's own established policy, not just via `test_dry_run.py`
(which reached 82 passing checks by the end, still green throughout).

**Phase G1 -- key correction removed entirely (not just left
off-by-default).** `key_correction.py` deleted outright; every reference
removed from `main.py` (`--key-correction`/`--no-key-correction`/
`--no-pass2-debug` flags, the `snap_to_key` call, the `[PASS2 DEBUG]`
file write), `config.py` (`ENABLE_KEY_CORRECTION`,
`PipelineOptions.key_correction`/`no_pass2_debug`), `gui.py` (the "Key
correction" checkbox), `test_dry_run.py`, and stale prose in
`debug_log.py`/`alignment.py`/`lyric_alignment.py`/`README.md`. This was
already confirmed net-harmful and off-by-default from earlier in the
session (a single global detected key blindly snaps legitimate
out-of-scale/modal-mixture notes -- see the historical entry earlier in
this file); this phase just finished the job by deleting the dead code
rather than leaving it as an unused opt-in. Also fixed a real,
previously-unnoticed inconsistency this removal exposed: several runtime
log lines/debug-section headers (`lyric_alignment`'s own fit-words step,
`verify_words`/`verify_placement`'s re-run points) called themselves
"pass 2" even though `lyric_alignment.py`'s own docstring had always
called itself "pass 3" -- now consistently "pass 3" everywhere, since
removing key_correction (which WAS genuinely pass 2) resolves the
ambiguity rather than creating a new one.

**Phase G2 -- `--output-dir` is now optional.** `run_pipeline`'s
`output_dir` parameter is `Optional[Path] = None`; when omitted, it
defaults to `<input_dir>/Output/<Artist> - <Title>`, computed AFTER
artist/title are resolved (the input==output collision guard moved to
run after this default is computed, since a computed default can never
collide with input_dir by construction). `run_batch`'s
`output_parent_dir` is optional the same way -- when omitted, each
subfolder falls through to `run_pipeline`'s own per-song default
independently. **Real-verified**: a real CLI run against the Chicago
sandbox song omitting `--output-dir` wrote to exactly
`<input>/Output/Chicago - When You're Good to Mama/....txt` as designed.

**Phase G3 -- debug files (`[DEBUG LOG]`, `[PASS1 DEBUG]`) now write into
`<input>/.ultrastar_work`, not the output folder.** Simple path-source
change in `main.py`; companion staging (`stage_companions_to_output`)
was already scoped to only ever copy mp3/video/cover/background, so
debug files were never at risk of leaking into a copied companion set.
**Real-verified**: re-ran the same Chicago song, confirmed both debug
files landed under `.ultrastar_work` (byte-identical content, just a
different location) and the output folder held only the real output
files.

**Phase G4 -- GUI: folder-picker memory, live placeholders, a real
audio-file picker, tooltips, non-yanking log scroll.** New
`gui_settings.json` (`Path.home()/.ultrastar_generator/`) remembers the
last-used input/output folder per field key, falling back to the
directory the program was launched from. New `PlaceholderEntry` (wraps
`ttk.Entry`) shows live grey preview text for Output folder/Artist/Title
when empty and unfocused -- computed via the SAME real functions
`run_pipeline` itself uses (`file_discovery.resolve_primary_source` +
`parse_artist_title`, plus the Phase G2 default-path formula), so there
is exactly one place that knows how to compute these, not two that could
drift apart; `effective_value()` returns `None` while a placeholder is
showing, so `_build_opts()` never mistakes preview text for real input.
Placeholders live-update via a carefully one-directional `trace_add` wiring
(input_dir/audio_file -> artist+title+output previews; artist/title ->
output preview only) specifically designed to avoid a self-referential
trace loop (a `PlaceholderEntry` writing its own preview into its own
bound var must never re-trigger its own refresh). The audio-file field
gained a real `Browse...` button (`filedialog.askopenfilename`, filtered
to `AUDIO_EXTS + VIDEO_EXTS`). New reusable `Tooltip` helper (borderless
`Toplevel` on hover) applied to every control, text condensed from the
CLI's own `--help` strings so the two surfaces don't drift. Log
auto-scroll now checks `yview()[1] >= 0.99` before re-snapping to the
bottom on each new line, so scrolling up to read something during a live
run no longer gets yanked back down. **Real-verified**: a full real
pipeline run driven through the actual GUI mechanism (background thread,
`_on_run`), output folder left blank, confirmed the run wrote to the
Phase G2 default path AND the Phase G3 debug-file location together,
through the placeholder-driven flow end to end.

**Phase G5 -- intermediate-file cleanup + "Open Output Folder" button.**
New `main.delete_intermediates(work_dir)` deletes only `separated/` and
`extracted/` under a work_dir (deliberately not the whole work_dir, since
debug files now live there too post-Phase-G3) -- shared by both an
automatic path (`PipelineOptions.delete_intermediates`, wired through a
`run_pipeline`/`_run_pipeline_body` split so cleanup runs in a `finally`
regardless of which of the body's several early-return failure paths was
hit -- work_dir may be partially populated even on a failure) and a new
GUI "Delete Intermediate Files Now" button (confirmation dialog first,
operates directly on the input folder's `.ultrastar_work` without
running anything). New GUI "Open Output Folder" button opens the folder
ONE LEVEL ABOVE the actual per-song output (e.g. `.../Output/`, not
`.../Output/<Artist> - <Title>/`) via `os.startfile` -- prefers the last
real completed run's own output path (threaded back from the worker
thread through the existing log queue with a tagged tuple, not a second
unsynchronized attribute write) and falls back to computing it from
current field values otherwise. **Real-verified**: a real CLI run with
`--delete-intermediates` confirmed `separated/` was gone afterward while
both debug files survived; also caught and fixed a stale log line
("Intermediate files kept in...") that was still printed even when the
flag was set, since it's written before the wrapper's `finally` runs.

**Phase G6 -- YouTube thumbnail becomes the cover art.**
`youtube_source.download_youtube_source` now passes `writethumbnail:
True` plus an `FFmpegThumbnailsConvertor` (format `jpg`) postprocessor to
yt-dlp, then renames the resulting `youtube_download.jpg` to
`youtube_download [CO].jpg` -- matching `file_discovery.find_companions`'
own pre-existing `[CO]`-tag convention by construction (the downloaded
audio/video is always named `youtube_download.<ext>`), so the cover is
picked up with **zero new code** in `file_discovery.py`/`song_input.py`.
Best-effort: a video with no fetchable thumbnail is a silent no-op, never
a failed download. **Real-verified**: a real download (the same short,
well-known "Me at the zoo" test video used for this project's original
YouTube verification) produced a real 23KB JPEG thumbnail, correctly
renamed and picked up by `find_companions` as `.cover` with no changes
to that function at all.

**Phase G7 -- interactive LRCLIB lyrics search/disambiguation (largest
phase).** Two independent entry points, per the user's explicit
clarification during planning:
1. **Manual pre-run search** (always available, single-song mode): a new
   "Search Lyrics..." button runs a real LRCLIB search
   (`lyrics_lookup.search_lrclib`, new -- returns every raw candidate,
   unfiltered, unlike the existing auto-pick path) and opens
   `LrcLibSearchDialog` (new `gui.py` class: candidate list on the left,
   lyrics preview on the right, synced-lyrics availability noted). A
   picked candidate is stored as `self.pinned_lyrics` and shown next to
   the button with a `Clear` option; `PipelineOptions.pinned_lyrics`
   (threaded through `run_pipeline`) always wins outright over the
   automatic fetch when set, skipping the network call entirely.
2. **Automatic mid-run ambiguity prompt** (checkbox, OFF by default,
   single-song mode only, only consulted when nothing was pre-pinned):
   `fetch_reference_lyrics`/`_fetch_from_lrclib` gained an
   `on_ambiguous(real_candidates) -> Optional[LrcLibCandidate]`
   parameter -- called only when more than one "real" candidate remains
   after filtering out instrumental/no-lyrics/wildly-off-duration
   results (a new `_real_lrclib_candidates`, 3x the normal scoring
   tolerance -- generous on purpose, this is an existence check, not a
   ranking). The GUI's callback (`_make_ambiguity_callback`) runs ON the
   background pipeline thread, schedules the SAME `LrcLibSearchDialog` on
   the main thread via `self.after(0, ...)` (the only safe way to touch
   Tk widgets from another thread), and blocks the background thread on
   a `threading.Event` until the dialog closes -- `wait_window()`'s own
   nested event loop keeps the main thread fully responsive during the
   pause. A cancelled/declined dialog falls through to the normal
   automatic pick, never leaves lyrics unset.

   Batch mode NEVER triggers either mechanism (per the user's explicit
   decision) -- `_build_opts()` forces `pinned_lyrics`/
   `lyrics_ambiguity_prompt`/`lyrics_disambiguation_callback` to
   None/False/None whenever `mode == "batch"`, regardless of what the
   (disabled-in-batch-mode) checkbox/pin state happen to hold, so
   `run_batch`'s pipeline calls can never receive a callback to invoke in
   the first place -- confirmed directly via the real `_build_opts()`
   method, not a mock.

   **Real-verified, both mechanisms, full real pipeline runs, live
   network** (Chicago sandbox song -- LRCLIB genuinely returns 5 real
   candidates for it, no synthetic fixture needed): (1) manual search +
   pin -- searched for real, deliberately pinned the LAST (non-
   auto-winning) candidate, ran the full pipeline, confirmed the log
   line `"Using manually-selected lyrics: ..."` named exactly that
   candidate and that no fresh automatic fetch ever happened; (2)
   automatic prompt -- real run with the checkbox on and nothing pinned,
   confirmed the pipeline genuinely paused, the REAL dialog opened with
   the 5 real candidates, picking one (via a real Tk timer callback, not
   a mock) resumed the run to a normal successful completion.

**Nothing committed during this phase either** -- same "only commit when
asked" convention as the rest of this session's work.

**Follow-up refinements, same day, after the above was reported done:**
1. **`--output-dir`'s meaning changed: it's now the PARENT folder a
   "<Artist> - <Title>" folder gets created under, not the final folder
   itself** (superseding Phase G2's original design above). E.g.
   `--output-dir C:\output` now produces `C:\output\<Artist> - <Title>\`;
   the default (when omitted) changed to just `<input_dir>\Output` as
   that parent -- previously it was already
   `<input_dir>\Output\<Artist> - <Title>`, so the DEFAULT case's actual
   final path is unchanged, only the explicit-value case's behavior and
   the field's own meaning changed. The input==output collision guard
   moved to compare the FINAL computed folder against input_dir (not the
   given parent), since a given parent equalling input_dir is now
   perfectly fine (it just means "put my per-song folder inside my own
   input folder") -- only an exact collision of the final folder itself
   is still rejected. `batch.py` needed no logic change (it already just
   forwards a path to `run_pipeline` as `output_dir`) -- it inherits an
   extra nesting level automatically when an explicit
   `output_parent_dir` is given (`output_parent_dir/<song folder
   name>/<Artist> - <Title>/`), documented in its own docstring.
   **Real-verified**: a real CLI run with `--output-dir` pointed at a
   fresh temp folder confirmed the output landed at exactly
   `<given-folder>\<Artist> - <Title>\...`.
2. **GUI output-folder placeholder is now a fixed relative path,
   `.\Output\`** -- no longer computed from/dependent on artist/title at
   all (matches the field's new PARENT-only meaning; the
   `<Artist> - <Title>` part is created automatically downstream, not
   something the user needs to see previewed). `_open_output_folder`'s
   fallback logic updated to match: the field's own real/placeholder
   value is now used AS the target directly (no more `.parent`
   stripping, since the field no longer includes the Artist-Title
   segment).
3. **The LRCLIB search dialog now has its own editable Artist/Title
   search fields** (pre-filled from the resolved artist/title when
   known, auto-searches once on open) instead of firing a single fixed
   search before the dialog even appears -- the user can freely edit and
   re-search for anything, not just what was auto-detected. Same dialog
   class serves both entry points: the manual "Search Lyrics..." button
   now just opens the dialog (letting IT do the initial search), and the
   automatic mid-run ambiguity prompt passes its already-found
   `initial_candidates` straight in (skipping the dialog's own
   auto-search, avoiding a redundant duplicate network call) while still
   pre-filling the search fields for a manual re-search if desired.
   **Real-verified** (GUI-level, mocked network): dialog auto-searches
   using pre-filled terms on open; editing the fields and searching
   again returns and uses the NEW results, not the original ones; the
   ambiguity-prompt path's `initial_candidates` correctly skips the
   redundant auto-search while keeping the fields usable.

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
