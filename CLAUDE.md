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
   `ENABLE_CREPE`/`ENABLE_KEY_CORRECTION`/`ENABLE_WORD_VERIFICATION` --
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
