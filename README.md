# UltraStar Song Generator

Generates UltraStar Deluxe `.txt` karaoke song files from a raw audio file
(`.mp3`, `.ogg`, or `.oga`), using a **pitch/timing-first, lyrics-second**
pipeline:

1. Isolate the vocal track (Demucs)
2. **Pass 1 -- audio only:** detect the actual sequence of sung notes
   (start, end, pitch) directly from the isolated vocals, with no
   involvement of lyrics at all (`note_detection.py`)
3. Transcribe the lyrics with word-level timestamps (WhisperX forced
   alignment, falling back to faster-whisper if WhisperX isn't installed)
4. **Pass 2:** fit the transcribed words onto the note grid from pass 1 --
   splitting words into syllables per note, merging syllables where the
   audio only resolved one note for several, and marking melisma
   (multiple notes on one syllable) with the `~` continuation convention
5. Group syllables into sung lines/phrases
6. Convert everything into UltraStar's beat grid and write the `.txt`,
   after a final pass that guarantees no two notes overlap

It also detects a matching cover/background image and video file next to
the audio, and (if a video is present) estimates `#VIDEOGAP` by
cross-correlating the video's own audio track against the song.

> **Why pitch/timing-first?** The first version derived note timing from
> ASR word boundaries, which are approximate by nature (and pYIN pitch
> tracking on tiny, context-starved per-word audio clips was noisy). This
> version treats the audio's own note structure as ground truth and fits
> lyrics onto it instead, which fixes overlapping notes, timing drift, and
> pitch accuracy all at the same root cause. See section 3 below.

> **v3 note (vibrato / beat-grid collisions):** the first pitch-first
> version still produced overlapping notes in practice, because natural
> vocal vibrato (a ~5-7Hz pitch wobble) was being read as a sequence of
> new notes, and some of those fragments were short enough to round onto
> the *same* beat once quantized to the file's coarse beat grid. Fixed
> with three additions: the pitch contour is now smoothed before
> segmentation, adjacent similar-pitch fragments get merged back together
> after segmentation, and the `.txt` writer now does a final hard
> collision check in the exact integer-beat space the file format uses
> (not just in continuous seconds), so a duplicate/overlapping beat is no
> longer possible regardless of what happens upstream. See section 3.

> **v4 note (word order / over-flattened pitch):** two more root causes,
> found from real output on "Stars":
> 1. Notes were assigned to whichever word had the single best time
>    overlap, independently per note. Once anything downstream re-sorted
>    by timestamp, an imprecisely-timed word could end up with a note
>    that displaced it out of reading order -- e.g. "dark" landing before
>    "He knows his way in". Fixed by assigning notes to words via a
>    monotonic timeline partition (each word owns a contiguous zone) and
>    by making the final safety-net pass trust word order instead of
>    re-deriving it from timestamps.
> 2. The adjacent-note merge pass (added in v3 to fight vibrato) was
>    comparing each new note only to the already-merged pitch, so a real
>    stepwise melodic run (several syllables each 1-2 semitones apart)
>    could chain-merge transitively into one flattened note -- this is
>    what produced both the giant 23-beat "dark" note and the reported
>    over-snapped D#/F#-should-be-E pitches. Fixed by capping the total
>    pitch range of a merged group, not just each individual step, and by
>    tightening the merge threshold. The optional key-correction pass is
>    now off by default for the same reason (it was likely compounding
>    the flattening) until it's validated against real audio on its own.

> **v5 note (diagnostics, and lyrics.ovh for both correction AND phrasing):**
> 1. Pass 1 now has its own hard, explicit non-overlap guarantee
>    (`_ensure_nonoverlapping`, which also warns if it ever has to fix
>    anything -- that would mean a real bug upstream, not a normal
>    occurrence), and a **pass-1-only debug file** gets written by default
>    alongside the real output: same audio, same timing/pitch, but every
>    note's lyric replaced with its note name (e.g. "G#3") instead of real
>    words. Load that in the UltraStar editor to check pass 1's
>    timing/pitch in complete isolation from anything pass 2 does. Lots of
>    console diagnostics were added too -- see section 3.
> 2. lyrics.ovh lookup is now used for two things, not one: correcting ASR
>    text (previously only gated on low ASR confidence, which missed
>    real-word-but-wrong mistranscriptions like "is" instead of "his" --
>    now the whole ASR word sequence is aligned against the whole
>    reference lyric sequence, the same technique used for word-error-rate
>    scoring), AND determining phrase breaks: every line break in the
>    reference lyrics is now propagated onto the matching words and forces
>    a `-` line break in the output, taking priority over the old
>    gap-based heuristic. Also fixed a real bug where the lyrics.ovh
>    request wasn't URL-encoding the artist/title, which silently 404'd on
>    anything with an accented character (e.g. "Les Mis\u00e9rables").

> **v6 note (hallucinated notes during actual silence):** the pass-1
> debug file immediately did its job -- it showed notes being generated
> during a stretch the isolated vocal track was confirmed (by ear) to be
> completely silent. Root cause: pYIN detects pitch/periodicity, not
> loudness. Near-silent audio can still contain quantization noise,
> resampling artifacts, or a faint hum with enough incidental periodicity
> to read as a confident, real-looking pitch -- so a silent instrumental
> intro (or the gap between phrases) could generate entirely fabricated
> notes. Fixed with an explicit RMS-energy gate that runs alongside
> pYIN's own voicing decision (`--silence-threshold-db` /
> `--silence-floor-db` to tune): a frame only counts as voiced if BOTH
> agree. Needed two components, not one -- a purely relative "quieter
> than the track's own loud parts" threshold turned out to fail on a
> long/entirely silent stretch, since there's no louder reference to
> compare against in that case (caught by a test, not by inspection); an
> absolute dBFS floor covers that.

> **v7 note (a bad interior word timestamp swallowing a whole line):**
> comparing the pass-1 debug file against the real output pinpointed this
> precisely -- pass 1's raw notes for a certain passage showed plenty of
> real melodic movement (many distinct pitches), but the final output had
> collapsed most of that stretch into one giant melisma on a single word
> ("Stars"), with the next several real words vanishing into it. The
> cause: within a matched reference line, individual INTERIOR words' own
> ASR timestamps were still being trusted to draw the boundaries between
> them -- and ASR word-level timing, even after WhisperX forced alignment,
> just isn't reliable enough for that on every passage. One bad interior
> timestamp could make one word's "zone" swallow a huge stretch of
> otherwise-correct notes. Fixed by changing what ASR timing is trusted
> for: a multi-word reference line now gets ONE coarse zone (from its
> first word's start to its last word's end -- much more reliable than
> any single interior boundary), and that line's notes are then split
> across its words *proportionally by syllable count*, in reading order,
> completely ignoring each interior word's individual ASR timestamp. A
> single-word "line" (or any word lyrics-lookup couldn't match to a
> reference line) still falls back to the original per-word zone
> behavior unchanged.

> **v8 note (a fabricated "bad note" traced to its exact source, plus a
> requested spike filter):** further real-run feedback found another bad
> note in the same passage -- this time confirmed, by directly comparing
> against the pass-1 debug file, to not exist in pass 1's output AT ALL.
> Root cause: a word with zero notes in its zone was falling back to a
> fresh, isolated pYIN call on its own (often very short, e.g. ~0.1s)
> ASR clip -- exactly the kind of context-starved analysis that produces
> noisy results, which is the same lesson pass 1 itself already learned
> (see the v3 note above about not running pYIN on tiny clips). Fixed by
> having the fallback path borrow the pitch of whichever pass-1 note (from
> the full, already-verified note list) is nearest in time, instead of
> manufacturing new, unverified pitch data. Separately, a spike/outlier
> filter was added as requested: a short note that jumps far in pitch
> from BOTH neighbors, where those neighbors are themselves close in
> pitch to each other, gets removed and folded into the previous note --
> tunable via `--spike-max-duration` / `--spike-jump-semitones`.

## 1. Setup (Windows)

1. **Python 3.10+**: install from [python.org](https://www.python.org/downloads/)
   (check "Add python.exe to PATH" during install).

2. **ffmpeg**: download a build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
   (the "essentials" zip is fine), unzip it somewhere, and add its `bin`
   folder to your `PATH`. Verify with:
   ```
   ffmpeg -version
   ```

3. **PyTorch** (needed by Demucs and faster-whisper): go to
   https://pytorch.org/get-started/locally/, pick "Windows / Pip / Python /
   CPU" (or a CUDA version if you have an NVIDIA GPU and want it faster),
   and run the `pip install ...` command it gives you.

4. **Everything else**:
   ```
   pip install -r requirements.txt
   ```

## 2. Usage

Put the audio file (and optionally its video/cover/background images) in a
folder, named like:

```
Bon Jovi - Its My Life.mp3
Bon Jovi - Its My Life[CO].jpg      (cover)
Bon Jovi - Its My Life[BG].jpg      (background)
Bon Jovi - Its My Life.mp4          (optional video)
```

Then run:

```
python -m ultrastar_generator "C:\Songs\Bon Jovi - Its My Life.mp3"
```

This writes `Bon Jovi - Its My Life.txt` next to the audio file. Useful
options:

```
--artist "..." --title "..."   Override artist/title instead of parsing
                                from the filename
--bpm 120                      Override auto-detected tempo
--whisper-model medium.en      Bigger/more accurate ASR model (default:
                                small.en). Try large-v3 if you have a GPU.
--device cuda                  Use an NVIDIA GPU for Demucs + Whisper
--fetch-lyrics                 Look up reference lyrics (lyrics.ovh) to
                                correct mistranscribed words (whole-sequence
                                alignment, not just low-confidence words)
                                AND to force phrase breaks at every real
                                line break in the lyrics (default: ON;
                                use --no-fetch-lyrics to disable)
--no-video-sync                Skip auto-VIDEOGAP detection
--no-whisperx                  Force faster-whisper's own word timestamps
                                instead of WhisperX forced alignment
--no-key-correction             Disable the musical-key pitch-snapping
                                polish pass (this is already the default;
                                use --key-correction to opt in)
--pitch-smooth-window 0.11     Vibrato-suppression filter window (sec);
                                raise if notes still fragment, lower if
                                fast runs get smeared together
--note-split-semitones 1.0     Pitch change needed to start a new note
--min-note-beat-fraction 0.5   Notes shorter than this fraction of a beat
                                get merged into a neighbor
--no-pass1-debug                Don't write the pass-1-only debug .txt
                                (written by default -- see section 3)
--quiet                         Suppress the verbose [pass1]/[pass2]
                                diagnostic console output
--silence-threshold-db 40       Relative silence gate: a frame this many
                                dB quieter than the track's own loud
                                sections is treated as silence/noise
--silence-floor-db -50          Absolute silence gate (dBFS); catches a
                                long/entirely silent stretch that has no
                                louder reference for the relative gate
--spike-max-duration 0.25       A note this short that jumps far from
                                both neighbors (which are close to each
                                other) is treated as a tracking glitch
--spike-jump-semitones 4        Minimum pitch jump from both neighbors
                                to count as a spike/glitch
--output-dir DIR               Write the .txt somewhere other than next to
                                the audio
--work-dir DIR                 Where separated vocal stems are cached
                                (default: <output-dir>/.ultrastar_work)
--skip-separation --vocals-path vocals.wav
                                Skip Demucs and use an existing isolated
                                vocal track (e.g. if you already ran it, or
                                want to try a different separation tool)
```

Run with `-h` for the full list.

## 3. Diagnostics & debugging

Two things are always on by default to make it possible to tell *where*
in the pipeline something went wrong, without needing to read source:

**The pass-1 debug file.** Every run writes
`<Artist> - <Title> [PASS1 DEBUG].txt` next to the real output (disable
with `--no-pass1-debug`). It's a fully valid, loadable UltraStar file --
same audio, same BPM/GAP, same note timing and pitch -- but every note's
lyric is replaced with its own note name (e.g. "G#3") instead of a real
word. Load it in the UltraStar editor and you're looking at pass 1's
output in complete isolation: if the timing/pitch looks wrong there, the
problem is in `note_detection.py` (segmentation/pYIN/tempo), not in
lyric fitting. If the debug file looks right but the real output doesn't,
the problem is in `lyric_alignment.py` or `lyrics_lookup.py` instead --
which narrows things down immediately.

**Console diagnostics.** Pass 1 prints its own stage-by-stage numbers by
default (frame count, voicing breakdown from both pYIN and the energy
gate separately, smoothing window size, onset count, note counts
before/after each merge pass, final pitch/duration range) -- silence
these with `--quiet` if they're too noisy. Pass 2 prints how many words
matched a pass-1 note directly vs. needed a fallback note (and lists the
fallback words -- if that list is long, pass 1 likely missed notes for
short/quiet words, which is a real lead worth following up on via
`--pitch-smooth-window`/`--note-split-semitones`), plus how many matched
reference lines had their notes distributed by syllable count (see
section 4 -- this is the mechanism that replaced trusting individual
interior word timestamps). The lyrics.ovh step prints how many reference
lines it found and every single word it corrected, e.g.
`"is" -> "his" (at 143.60s)`, so you can see exactly what changed and
cross-check it against the audio at that timestamp -- if you suspect
phrase breaks aren't coming from the reference lyrics, this is the first
place to check: zero reference lines found means it silently fell back to
gap-based phrasing.

## 4. How it maps to the UltraStar format

- `#MP3` is always used for the audio file tag, even for `.ogg`/`.oga`
  files, per the UltraStar spec (the tag name is just legacy; the game
  reads whatever format is there).
- `#BPM` is written as the tempo *before* UltraStar's internal x4
  multiplication (per the format spec): a beat in the note grid equals
  `60 / (BPM * 4)` seconds. This was verified against the reference
  `.txt` files you provided (all four are internally consistent with
  this formula).
- `#GAP` = the timestamp (ms) of the first detected vocal.
- `#VIDEOGAP` is only written if a video file was found and it has its own
  audio track that could be cross-correlated against the song.
- Cover/background resolution follows your rules: same-named single image
  used for both; `[CO]`/`[BG]`-tagged images used respectively when
  present.
- Note pitch = `round(69 + 12*log2(f0/440)) - 60`, i.e. MIDI note minus 60,
  exactly as the spec defines it, computed from a **single pYIN pass over
  the whole vocal track** (not tiny per-word clips -- pYIN's internal HMM
  smoothing needs real context to be accurate, which was a major source of
  bad pitches before).
- Notes are first detected purely from audio: an RMS-energy gate first
  decides which frames could plausibly contain real singing at all
  (rejecting near-silent noise/artifacts that pYIN's pure pitch/
  periodicity detection can otherwise mistake for a confident, real-
  looking pitch -- silence has no loudness-based signature pYIN
  considers, so this check is deliberately independent of it). Within
  what's left, pYIN's pitch contour is median-filtered (~110ms window) to
  suppress vocal vibrato, then split into discrete notes at silence gaps,
  onset events that also coincide with a real pitch change, and sustained
  pitch jumps of ~1 semitone or more. A merge pass then combines adjacent
  notes that are both close in pitch (within 1 semitone) AND close in
  time -- capped so the total pitch range of anything folded together
  never exceeds that same 1 semitone, which specifically prevents a real
  melodic run (several syllables each a step apart) from chain-merging
  into one flattened note. A second pass folds any note shorter than half
  a beat into whichever neighbor has the closer pitch. Each note's pitch
  is a confidence-weighted mode over its frames, not a plain average.
  This is pass 1, and none of it has any knowledge of the lyrics.
- Words are hyphenated into syllables (via `pyphen`) and then fitted onto
  the notes from pass 1 via a **monotonic timeline partition** -- but at
  the LINE level, not the word level, whenever lyrics.ovh matched a word
  to a reference line: consecutive words sharing the same reference line
  get grouped first, and it's that GROUP's overall span (first word's
  start to last word's end) that gets a zone, with boundaries at the
  midpoint between consecutive groups. Each detected note is assigned to
  whichever zone its midpoint falls in. This is deliberately not "each
  note/word picks whichever counterpart overlaps it best independently"
  -- that let ASR timing imprecision assign a note (or a whole stretch of
  notes) to the wrong word, which then read as scrambled word order or,
  worse, one bad interior word timestamp swallowing a huge stretch of a
  matched line into a single giant melisma (both reported in practice). A
  zone partition can't do the former by construction; for the latter, a
  matched line's notes are split across its words **proportionally by
  syllable count**, in reading order, rather than trusting each interior
  word's own ASR timestamp at all -- only the coarser, more reliable
  line-level span matters. A single unmatched word (no reference line, or
  lyrics lookup disabled) falls back to the original one-zone-per-word
  behavior. Within a word, syllable count is then reconciled against the
  number of notes it actually received (1:1 when they match, merged into
  contiguous chunks when the word has more syllables than notes, marked
  with a `~` continuation when a syllable is held across multiple notes).
  **Note timing and pitch always come from pass 1**, never from the ASR
  word boundaries.
- An optional final polish pass (`--key-correction` to enable; **off by
  default**) detects the song's most likely musical key from its
  pitch-class distribution and nudges clearly-out-of-key notes toward the
  nearest in-key neighbor. This is adapted from the key-correction idea
  in the open-source `ultrastar_pitch` project, reimplemented from
  scratch here. It's off by default because it can compound pitch
  flattening on genuinely chromatic/passing-tone notes -- worth
  re-enabling and comparing once the segmentation fixes above have been
  validated against real audio on their own.
- Non-overlap is enforced twice: once in continuous seconds
  (`postprocess.enforce_monotonic`, which trusts word order and only
  pushes later notes forward -- it does NOT sort by timestamp, since
  doing that once was itself a source of word-order bugs), and once more
  authoritatively in the `.txt` writer, which re-derives every note's
  integer beat position and clamps any note that would start before the
  previous one ends *in beat space* -- because two notes a few
  milliseconds apart in real time can still round onto the exact same
  beat on a coarse grid.
- Lines break on silence gaps and a max-syllables-per-line heuristic (see
  `phrasing.py` to tune `MAX_SYLLABLES_PER_LINE` / `MIN_LINE_GAP_SEC`).
- Notes held longer than `GOLDEN_NOTE_MIN_DURATION_SEC` (0.6s by default)
  are marked golden (`*`) as a rough heuristic; this is not a substitute
  for manually picking real "money notes" in the UltraStar editor.
- `--pitch-smooth-window`, `--note-split-semitones`, and
  `--min-note-beat-fraction` expose the three main segmentation knobs on
  the CLI, so they can be tuned per-song without editing source. If notes
  are still fragmenting, raise `--pitch-smooth-window` or
  `--note-split-semitones`; if fast melodic runs are getting smeared
  into one note, lower them.

## 5. Accuracy expectations & tuning

This automates a first draft, not a finished song file. Realistically:

- **Overlapping notes**: structurally prevented now at two independent
  layers -- continuous-seconds enforcement, and (critically, since that
  alone wasn't enough) an integer-beat collision check in the writer
  itself, which is the space the actual bug showed up in. If you still
  see an overlap, that's a bug worth reporting, not an occasional edge
  case to expect.
- **Note timing/pitch**: driven entirely by the whole-track pYIN pass,
  smoothed to resist vibrato, with two merge passes cleaning up residual
  fragmentation. Expect it to track real sung pitch much more closely
  than before; it will still wobble on very heavy vibrato, breathy/airy
  notes, or where source separation left instrumental bleed in the vocal
  stem -- `--pitch-smooth-window` and `--note-split-semitones` are there
  to tune against.
- **Word timing**: WhisperX's forced alignment is a large step up from
  Whisper's own decoder timestamps, but pass 2 also doesn't actually
  *need* word timing to be frame-accurate -- it only uses word boundaries
  to decide which notes belong to which word, and the note boundaries
  (from pass 1) are what actually get written. So even a somewhat-off
  word timestamp usually still lands the lyric text on the correct notes.
- **Lyrics text**: still ASR-driven, so expect occasional misheard words,
  especially on fast/dense vocals or heavy stylization. `--fetch-lyrics`
  (on by default) nudges low-confidence words toward matching text from
  an online lyrics source, but it's a heuristic correction over ASR
  output, not a guarantee -- always worth a read-through.
- **Phrasing/line breaks**: reasonable but not always "musical" -- the
  UltraStar editor is still the right place for a final pass.

Plan to open the generated file in the UltraStar Deluxe editor for a
quick pass, same as you would with any auto-timed karaoke file.

## 6. Duets

This version deliberately does **not** produce `P1`/`P2` duet files (per
the current requirements), even though `Song.parts` and the note-writing
code were structured so that adding it later is a small, contained
change: mainly, running the pipeline twice (once per vocal range/singer)
and having `usdx_writer.py` emit a `P1`/`P2` marker line before each
part's entries. The included Aladdin duet file is a good reference target
for that when you're ready for it.

## 7. Project layout

```
ultrastar_generator/
  config.py          Tunable constants/defaults
  models.py           Word / Syllable / LineBreak / Song dataclasses
  file_discovery.py   Finds companion cover/background/video files
  separation.py        Demucs vocal isolation
  note_detection.py   Pass 1: audio-only note (pitch/timing) detection,
                        with vibrato-smoothing + merge passes + a hard
                        non-overlap guarantee
  transcription.py    WhisperX (preferred) / faster-whisper word timestamps
  lyrics_lookup.py    lyrics.ovh fetch + whole-sequence word alignment
                        (text correction + per-word reference line id)
  lyric_alignment.py Pass 2: fits words onto the pass-1 note grid,
                        propagates line id, reports AlignmentStats
  key_correction.py    Optional musical-key pitch-snapping polish pass
  postprocess.py       Non-overlap enforcement (seconds-level, order-
                        preserving -- does NOT sort by timestamp)
  tempo.py             BPM detection + beat<->seconds conversion
  syllables.py         Word -> syllable hyphenation
  phrasing.py           Syllables -> lines; forces breaks on reference
                        lyric line-id changes, falls back to gap-based
  debug_output.py       Builds/writes the pass-1-only debug .txt
  video_sync.py         VIDEOGAP estimation
  alignment.py          Pass-2 + phrasing glue (pass 1 is called
                        directly by main.py now, to keep the two passes
                        clearly separated)
  usdx_writer.py        Renders/writes the final .txt; ALSO does the
                        authoritative integer-beat non-overlap check
  main.py                CLI
```

## 8. Testing notes

`test_dry_run.py` covers everything that doesn't require the ML models
themselves: file discovery, the beat-grid math (verified against your
four original reference files), syllable hyphenation, phrasing, the
writer's exact spacing conventions, and targeted regression tests for
every bug reported so far:

- overlap enforcement (seconds-level)
- notes-drive-timing, not ASR word boundaries
- melisma continuation markers and syllable merging
- **beat-grid quantization collisions** -- reproduces the exact "Stars"
  failure (near-simultaneous notes that are non-overlapping in seconds
  but round onto the same beat at a slow tempo) and confirms the writer's
  integer-beat pass resolves it
- **vibrato fragmentation** -- feeds `detect_notes()` a synthetic 1.4s
  tone wobbling +/-0.6 semitones at 6.5Hz (via a mocked pitch contour,
  since real pYIN needs real audio/libraries not available in this
  sandbox) and confirms it collapses into a single note, matching the
  "There," example from feedback
- **key correction** -- confirms an out-of-key note gets snapped, and
  this test caught a real bug during development (the original
  equidistant-neighbor handling was accidentally a no-op for every
  standard diatonic scale; fixed by tie-breaking on pitch-class frequency)

Run it with:

```
python test_dry_run.py
```

What I *still* haven't been able to test: the real Demucs/WhisperX/pYIN
pipeline end-to-end on actual audio. Even with your mp3/ogg files now
available, this sandbox has no network access, so `librosa`, `torch`,
`demucs`, `faster-whisper`, and `whisperx` can't be installed here --
only what's already present (`numpy`/`scipy`) is available. Everything
above was validated either analytically (the beat math against your
reference files) or by mocking the exact librosa calls `note_detection.py`
makes, with synthetic pitch contours built to reproduce the reported
failure modes. That's solid evidence the logic is correct, but it's not
the same as seeing it run on "Stars" for real -- if you run this version
and it's still off, the beat/vibrato numbers and the `--pitch-smooth-window`
/ `--note-split-semitones` / `--min-note-beat-fraction` flags above should
make it much faster to zero in on why.

## 9. Known limitations / TODO

- No GPU-specific tuning beyond `--device cuda`; large files on CPU-only
  machines can take several minutes per song (mostly Demucs + Whisper).
- Golden-note and freestyle/rap (`F`/`R`/`G`) note types aren't inferred
  beyond the simple "long note -> golden" heuristic.
- `#RELATIVE` notes aren't used (all timestamps are absolute from GAP, as
  recommended).
- Duet (`P1`/`P2`) generation isn't implemented yet (see above).
