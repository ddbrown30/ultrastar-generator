# ultrastar_generator

Generates UltraStar Deluxe `.txt` karaoke song files from a raw audio
file (mp3/ogg/oga): isolates vocals, detects sung notes from the audio
itself, transcribes lyrics, fits the lyrics onto the detected notes, and
writes a spec-compliant `.txt`.

Full narrative history of every bug found and fixed is in `README.md`
(the "v1 note" through "v8 note" callouts) — read that before assuming
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

## Open threads / where we left off

Discussed but not yet decided/implemented, in rough priority order:

1. **Swap `key_correction.py`'s ad-hoc key detection for `music21`**
   (proper Krumhansl-Schmuckler key-finding) — agreed this is a real
   existing library that should replace the hand-rolled heuristic.
   `key_correction.py` is currently OFF by default (`--key-correction`
   to enable) pending this.
2. **Chunk-based re-transcription for verification**: for suspicious
   words (fallback words, or lines with anomalous syllable-to-note
   ratios), re-run ASR on a tightly cropped window around that timestamp
   and cross-check against the expected reference word. Agreed as worth
   building, not yet implemented.
3. **CREPE and/or Essentia's MELODIA as a pYIN alternative/ensemble**:
   proposed running CREPE alongside pYIN and using agreement/disagreement
   as a confidence signal for gating notes, or trying Essentia's
   `PredominantPitchMelodia` (purpose-built polyphonic melody extraction)
   as a pYIN replacement. Real architecture change (new heavy
   dependency) — needs an explicit go-ahead before building, not a
   "just try it" change.
4. **MIDI database cross-checking**: considered and deprioritized — no
   reliable, API-searchable public database of vocal MIDI transcriptions
   is known to exist; would mostly add fragile scraping for something
   that'd fail silently on most songs.

## Environment notes

- Windows, venv at `E:\Projects\ultrastar_generator\venv`.
- GPU available (`--device cuda`); WhisperX pulls in pyannote/torch —
  expect noisy-but-harmless warnings (torchcodec/ffmpeg version
  mismatches, TF32 reproducibility notices) on startup; these haven't
  been correctness issues so far.
- Demucs writes intermediate stems to `sandbox/.ultrastar_work/` —
  safe to delete between runs to reclaim disk space, and separation is
  cached/skipped if `vocals.wav` already exists there.
