"""Defaults and constants shared across the pipeline."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

# --- UltraStar note types -------------------------------------------------
NOTE_NORMAL = ":"
NOTE_GOLDEN = "*"
NOTE_FREESTYLE = "F"
NOTE_RAP = "R"
NOTE_RAP_GOLDEN = "G"

# --- Audio / file conventions ---------------------------------------------
AUDIO_EXTS = (".mp3", ".ogg", ".oga", ".m4a")
# All companion-#VIDEO-eligible extensions (file_discovery.find_companions).
VIDEO_EXTS = (".avi", ".mp4", ".mpg", ".mpeg")
# Video containers UltraStar Deluxe can use directly as #MP3 (confirmed by
# the user) when no real audio file exists -- .mp4/.mpg/.mpeg all work;
# .avi does NOT and must have its audio extracted into a real standalone
# file first (see file_discovery.resolve_primary_source's "video_as_audio"
# vs. "avi_extract" tiers, and media_extract.py).
VIDEO_DIRECT_AUDIO_EXTS = (".mp4", ".mpg", ".mpeg")
IMAGE_EXTS = (".jpg", ".jpeg")

# --- Folder-based input resolution (file_discovery.py, media_extract.py,
# cover_extract.py, song_input.py) -------------------------------------------
# When no real audio file (AUDIO_EXTS) exists in the input folder, a video
# file stands in as the audio source. mp4 is preferred over avi (it can
# serve as BOTH #MP3 and #VIDEO directly -- confirmed working in UltraStar
# Deluxe), falling back to avi (which cannot double as #MP3 -- its audio
# track must be actually EXTRACTED into a real standalone mp3 via ffmpeg
# first; see media_extract.extract_audio_track).
AVI_EXTRACTED_MP3_QUALITY = 2  # ffmpeg libmp3lame VBR scale, 0=best/largest .. 9=worst/smallest

# Embedded cover-art extraction (mutagen, cover_extract.py) only runs when
# no .jpg/.jpeg companion was already found next to the resolved audio
# source. Matches file_discovery.find_companions' own "[CO]"/"[BG]" tag
# naming convention, so a later find_companions call on the same folder
# reads an extracted cover identically to a hand-placed one.
EMBEDDED_COVER_TAG_SUFFIX = " [CO]"

# --- Pipeline defaults ------------------------------------------------------
DEFAULT_LANGUAGE = "English"

# Fallback BPM (as stored in the .txt, i.e. NOT multiplied by 4 yet) used
# only if tempo detection fails outright. 100 gives a beat resolution of
# 150ms, which is coarse but safe.
FALLBACK_BPM = 100.0

# Minimum/maximum "sane" BPM range for the txt file's #BPM value (i.e. real
# musical tempo, since UltraStar multiplies it by 4 internally). Detected
# tempo outside this range gets folded by x2/ /2 until inside it.
MIN_BPM = 60.0
MAX_BPM = 200.0

# The BPM actually WRITTEN to the .txt (and used for beat-quantization at
# write time) is this multiple of the real detected tempo -- a common
# real-world UltraStar convention for finer beat-grid resolution (confirmed
# directly: a real, hand-authored reference file for "When You're Good to
# Mama" used #BPM:300 against a real detected tempo of ~178, roughly 1.7x,
# specifically so short/dense syllables don't need much padding to reach
# the format's 1-beat-per-note minimum). Applied only at write time --
# pass 1's own audio analysis (note_detection.py) always uses the real,
# undoubled tempo, since its segmentation thresholds are tuned against real
# beat duration, not display resolution.
BPM_WRITE_MULTIPLIER = 2

# Word-level ASR model to use with faster-whisper by default. "medium.en" is
# a good accuracy/speed tradeoff on GPU; use "small.en" for less VRAM/time
# at the cost of accuracy, or "large-v3" for even better lyric accuracy at
# the cost of more VRAM/time.
DEFAULT_WHISPER_MODEL = "medium.en"

# Demucs model. htdemucs is the current general-purpose default.
DEFAULT_DEMUCS_MODEL = "htdemucs"

# Phrase-breaking heuristics
MAX_SYLLABLES_PER_LINE = 8
MIN_LINE_GAP_SEC = 0.35  # silence gap that *forces* a new line
PREFERRED_LINE_GAP_SEC = 0.15  # silence gap that's a good place to break

# Golden-note heuristic: notes held longer than this many seconds are
# marked golden ('*') instead of normal (':'). This is a rough heuristic,
# not a substitute for manual editing in the UltraStar editor.
GOLDEN_NOTE_MIN_DURATION_SEC = 0.6

# --- Note detection (pass 1: pitch/timing from audio, no lyrics involved) --
# Notes shorter than this are treated as glitches and dropped/merged.
MIN_NOTE_DURATION_SEC = 0.06
# A note's final pitch is a confidence-weighted mode over ALL its frames
# by default, INCLUDING this many leading seconds -- set > 0 to instead
# drop that many leading seconds first. PROTOTYPE (2026-08-07, not yet
# validated at full-pipeline scale): motivated by a systematic flat-pitch
# bias found via OMR cross-validation (pYIN reads flat 3x as often as
# sharp), theorized to come from legato/portamento attacks (a rising
# glide INTO a note from the previous, lower pitch) biasing the whole-
# note average down. 0.0 = fully off (original behavior) -- see
# _trim_attack's docstring for the short-note fallback that keeps this
# from ever discarding a short note's only evidence.
ATTACK_TRIM_SEC = 0.0
# Alternative/complementary to ATTACK_TRIM_SEC: drop the bottom N% of a
# note's own frames by CONFIDENCE (wherever they land) before computing
# its final pitch, instead of a fixed leading time window. PROTOTYPE
# (2026-08-07). Validated directly against Beauty and the Beast ground
# truth (bypassing detect_notes/pass-1 entirely, to isolate this from
# segmentation effects): RMVPE's raw exact-semitone match went 54% -> 59%
# at the 50th percentile; going further (75th/90th) reversed the gain
# (too few frames left). 0 = fully off (original behavior). NOT yet
# validated at full-pipeline scale or on other songs as of this writing
# -- see _confidence_floor_filter's docstring.
CONFIDENCE_FLOOR_PERCENTILE = 0.0

# Reconciles a protected_start (re-articulation) split's rounded pitch
# with its immediate predecessor when they're near-contiguous and land
# EXACTLY 1 semitone apart -- see _confidence_floor_filter's neighbor,
# note_detection.py's inline comment at the reconciliation site, for the
# full mechanism (a real bug found via RMVPE isolation-mode testing on
# Stars, 2026-08-07: natural intra-note pitch drift straddling a
# rounding boundary right where a genuine re-attack split landed, so the
# two fragments independently round to adjacent semitones even though
# the whole syllable is really one pitch). PROTOTYPE, off by default
# until validated across multiple songs.
REARTICULATION_RECONCILE_ENABLED = False
# How close together (seconds) the two fragments' boundary must be to
# reconcile -- deliberately much tighter than NOTE_MERGE_MAX_GAP_SEC so
# this can never fire on the OTHER protected_start path (resuming after
# a genuine silence gap at a strong onset), which by definition has a
# real, non-trivial gap before it.
REARTICULATION_RECONCILE_MAX_GAP_SEC = 0.02
# A pitch change of at least this many semitones (on the SMOOTHED contour,
# without a silence gap) is treated as a new note rather than vibrato.
NOTE_SPLIT_SEMITONES = 1.0
# Re-articulation at the SAME pitch (e.g. two consecutive syllables sung
# on the same held note, no pitch change at all) previously could never
# be split -- an onset alone was deliberately not enough (see
# note_detection.py's v2 changes: a bare onset used to force spurious
# splits from consonant/attack transients inside one sustained note).
# But that meant genuinely re-attacked same-pitch notes (confirmed via a
# real bug report: "fall as Lucifer" all sung on one pitch, several
# distinct onsets each with a real RMS dip beforehand -- all silently
# merged into one giant note) could never split either. The fix: only a
# STRONG onset (top percentile of onset-strength among the onsets this
# track actually has -- filters out weak consonant blips) is allowed to
# split same-pitch audio, and only once the in-progress note has already
# run for MIN_DURATION_BEFORE_REARTICULATION_SEC (so the onset marking a
# note's own attack can't immediately re-split itself).
REARTICULATION_STRENGTH_PERCENTILE = 60.0
MIN_DURATION_BEFORE_REARTICULATION_SEC = 0.08
# A too-short protected re-articulation segment gets stretched toward
# MIN_NOTE_DURATION_SEC, capped at the next raw segment's own start (see
# note_detection.py) -- if that next segment starts almost immediately,
# the stretch can be capped down to an arbitrarily tiny sliver instead of
# the intended floor. Confirmed in practice: a real onset produced a
# protected split with only 12ms of room before the next segment, which
# then propagated all the way to lyric_alignment.py as a standalone
# syllable's pitch (pass 3 correctly refused to silently absorb/discard a
# protected_start note, so the implausibly short note surfaced instead of
# being hidden). 12ms is well under typical minimum-audible-pitch
# duration. Below this floor, the split isn't a plausible distinct note
# even though the onset that triggered it was real -- drop it instead of
# keeping a sliver, same as an unprotected too-short segment.
MIN_PLAUSIBLE_REARTICULATION_DURATION_SEC = 0.03
# A re-attack's onset often lands inside (or right next to) a brief
# unvoiced dip -- the singer's articulation itself often drops below the
# voicing/energy gate for a few frames -- which already creates a natural
# split via the normal unvoiced-frame path, just not a PROTECTED one, so
# the very next merge pass silently undoes it (same pitch, small gap).
# Confirmed on a real bug report ("fall as Lucifer fell" merged into one
# note despite genuine per-syllable onsets AND RMS dips confirmed present
# at each boundary): the dips were real but brief enough (well under
# NOTE_MERGE_MAX_GAP_SEC) to get silently re-merged. A note that resumes
# from silence within this many seconds of a strong onset is protected
# the same way an in-voicing re-articulation split is. This is a
# half-width (the dilated window spans +/- this many seconds around each
# strong onset) -- kept well under half of
# MIN_DURATION_BEFORE_REARTICULATION_SEC on purpose: a wider window let
# a single onset's dilated span qualify for TWO separate splits (once
# right as the window opens, and again once the resulting first note's
# own elapsed duration cleared the minimum -- while still inside the
# same dilated span), fragmenting one note into three instead of two.
REARTICULATION_ONSET_WINDOW_SEC = 0.02
# Median-filter window (seconds) applied to the pitch contour before
# segmentation, specifically to suppress vocal vibrato (typically a
# 4-8 Hz / 125-250ms-period wobble) without blurring real note changes.
PITCH_SMOOTH_WINDOW_SEC = 0.11
# After initial segmentation, adjacent notes this close in pitch and time
# get merged into one note (fixes vibrato/noise fragmenting one sustained
# syllable into several near-identical short notes). Kept tight on
# purpose: real melodic movement between syllables is very often exactly
# 1-2 semitones, so a loose threshold here (previously 2) was chain-
# merging genuine stepwise melody into one flattened note. This is now
# also enforced as a total-range cap across the whole merged group, not
# just a per-step comparison -- see _merge_similar_adjacent.
NOTE_MERGE_SEMITONES = 1
NOTE_MERGE_MAX_GAP_SEC = 0.05
# A note shorter than this fraction of one beat (at the song's detected
# BPM) gets folded into whichever neighbor has the closer pitch, since it
# can't be meaningfully represented on the beat grid on its own anyway.
MIN_NOTE_BEATS_FRACTION = 0.5
# Text used for "extra" notes when a word has fewer syllables than the
# audio has detected notes (i.e. melisma -- one syllable held/bent across
# multiple pitches). Matches the convention seen in real UltraStar files.
MELISMA_CONTINUATION_TEXT = "~"
# Minimum gap (seconds) enforced between any two notes when resolving
# overlaps in the final assembled note/lyric sequence.
MIN_NOTE_GAP_SEC = 0.01

# Final-step cleanup (2026-08-10): a melisma-continuation note ("~") that's
# beat-adjacent (no gap) to the PRECEDING note AND at the exact same pitch
# is redundant on the beat grid -- UltraStar doesn't need a separate note
# to hold a pitch that hasn't changed and hasn't even paused; it just makes
# the chart busier to read/sing without adding real information. Real case
# (user-reported, "Barely even friends"): a word's own note followed
# immediately by a same-pitch "~" (an onset/release tracking artifact --
# see NOTE_MERGE_SEMITONES's own docstring for the same underlying pass-1
# phenomenon) got written as two separate notes instead of one longer one.
# Merges the "~" into whichever note precedes it (word syllable OR another
# "~"), extending that note's own length and dropping the "~" entirely --
# chains, so several connected same-pitch "~" notes in a row all collapse
# into one. Runs at the INTEGER BEAT level in usdx_writer.py, on exactly
# what will be written, not on the pre-quantization float-second timing --
# quantization itself can change what counts as "adjacent". Generation
# pipeline only (main.py) -- NOT applied to realign.py, which has its own
# stricter "never add/remove/reorder a note" contract this would violate.
#
# Second step, added 2026-08-10 (user's explicit request, tried as a
# further experiment on top of the above): after the same-pitch merge,
# any "~" that's STILL only 1 beat long -- it wasn't adjacent/same-pitch
# to its predecessor, so the merge above didn't touch it -- is deleted
# outright (`_remove_orphan_short_melisma_tails`), leaving a gap on the
# beat grid rather than being folded into anything. A single 1-beat "~"
# is too short to carry real melodic information and is more often
# tracking noise than a genuine melisma continuation. Same flag controls
# both steps (both are "final lyric-placement cleanup", generation
# pipeline only).
MERGE_CONNECTED_MELISMA_TAILS = True

# Pass 1's pitch source: "rmvpe" or "swiftf0" -- see
# note_detection.PITCH_SOURCES. Exactly ONE source supplies both the
# pitch value and the voicing decision, exclusively -- no cross-check, no
# ensemble, no consensus vote between sources. pYIN, CREPE, PENN, and the
# whole cross-check/consensus-override machinery that used to sit on top
# of a source were all removed (2026-08-14, user's explicit request) --
# see CLAUDE.md's "Removed / rejected approaches": a single well-chosen
# source in true isolation had already outperformed the old pyin-primary
# + CREPE/RMVPE-cross-check ensemble on real end-to-end validation
# (+1.7pp average across the 4-song set, run twice per song to rule out
# RMVPE's own documented non-determinism -- both runs were IDENTICAL).
DEFAULT_PITCH_SOURCE = "rmvpe"

RMVPE_DEVICE = "cpu"  # onnxruntime CUDA build requires CUDA 13 + cuDNN 9,
                       # incompatible with this project's torch cu126
                       # stack; DirectML crashed/hung with garbled UTF-16
                       # logging on this machine. CPU inference measured
                       # at ~5s for a full ~3min song -- fast enough that
                       # GPU acceleration isn't worth chasing further
                       # unless this graduates from prototype.

# A frame this many dB quieter than the track's own "loud" reference level
# (the 90th percentile of RMS energy, not the absolute peak, so one loud
# transient doesn't skew the reference) is treated as silence/noise,
# REGARDLESS of what the pitch source's own voicing decision says. This
# matters because a pitch/periodicity detector detects periodicity, not
# loudness -- near-silent audio can
# still contain quantization noise, resampling ringing, or a faint hum
# with enough incidental periodicity to read as a confident, real-sounding
# pitch. Without this gate, a silent instrumental intro (or the silent gap
# between phrases) can generate entirely hallucinated notes.
SILENCE_REFERENCE_PERCENTILE = 90
SILENCE_THRESHOLD_DB_BELOW_PEAK = 40.0
# A relative threshold alone isn't enough: if an entire clip (or a long
# stretch of it) is uniformly near-silent, there's no louder reference to
# be "40dB below" -- the relative comparison can't detect "this is all
# silence" on its own. This absolute floor (dBFS, i.e. relative to full
# amplitude scale = 0dB) catches that case directly: anything this quiet
# is treated as silence no matter what it's being compared to.
SILENCE_ABSOLUTE_FLOOR_DB = -50.0

# Spike/outlier note filter: a note this short, that jumps at least this
# many semitones from BOTH its immediate neighbors, where those neighbors
# are themselves close in pitch and close in time to the spike, gets
# removed and folded into the previous note. This targets isolated
# tracking glitches ("briefly reads a wildly different pitch, then
# returns to where it was") rather than real, intentional pitch movement.
SPIKE_MAX_DURATION_SEC = 0.25
SPIKE_MIN_JUMP_SEMITONES = 4.0
SPIKE_NEIGHBOR_SIMILARITY_SEMITONES = 2.0
SPIKE_MAX_NEIGHBOR_GAP_SEC = 0.15

# Trailing breath/release-artifact filter: a short note that immediately
# follows a long, sustained, HIGH-confidence note -- either contiguously
# or after a small real gap (a breath) -- but whose own confidence is far
# below that neighbor's, gets absorbed into it (its end extended) instead
# of kept as its own note. Targets a different failure signature than
# SPIKE_* above: these artifacts (breath intake, a held vowel's release
# tail, a trailing consonant) are usually pitched CLOSE to the note they
# trail, not a big jump away -- so they never trip the pitch-jump spike
# check -- and unlike vibrato/tracking-noise fragmentation, they're
# genuinely low-CONFIDENCE, which neither the spike filter nor
# _merge_short_notes currently look at. Confirmed real cases (2026-08-07,
# user-reported): a 9-beat "sky" sustain followed, after a real ~140ms
# gap, by two spurious 1-beat notes before the next word; a 9-beat "Lord"
# sustain followed, with NO gap, by one spurious 1-beat note right before
# a phrase break.
TRAILING_ARTIFACT_MAX_DURATION_SEC = 0.12
TRAILING_ARTIFACT_CONFIDENCE_RATIO = 0.6  # candidate's confidence must be
                                            # <= this fraction of the
                                            # preceding note's.
TRAILING_ARTIFACT_MAX_GAP_SEC = 0.2       # generous enough to cover a
                                            # real breath gap (~140ms
                                            # confirmed) with headroom.
TRAILING_ARTIFACT_MIN_PRECEDING_DURATION_SEC = 0.5  # the note being
                                            # trailed must itself be a
                                            # real sustained note, not
                                            # another short fragment.

# Word-level timestamp source. "whisperx" uses forced alignment (wav2vec2
# CTC) for much more accurate word boundaries than Whisper's own decoder
# timestamps; falls back to faster-whisper automatically if whisperx isn't
# installed.
PREFER_WHISPERX = True

# whisperx's own pyannote VAD (used to chunk audio before transcription+
# alignment) has no true off switch -- vad_model/vad_method is mandatory
# in whisperx.load_model(). Confirmed via a real bug report + isolated
# testing that this VAD is the actual root cause of word timestamps being
# wrong by UP TO ~6 SECONDS around sustained/held sung notes (a long held
# vowel apparently gets misjudged by VAD in a way that corrupts the
# downstream wav2vec2 alignment's context) -- forcing near-zero onset/
# offset thresholds (treating almost everything as speech) fixed a real
# case ("Stars" in a test song) from 5.88s off down to 0.15s off, the
# best of several approaches tested (also tried: switching to faster-
# whisper's own DTW timestamps, stable-ts -- both got the same case to
# ~0.3s off, still worse than this). See --whisperx-no-vad / --whisperx-vad.
WHISPERX_NO_VAD_OPTIONS = {"vad_onset": 0.01, "vad_offset": 0.01}
# On by default (--whisperx-vad to opt back into whisperx's own VAD) --
# validated end-to-end against a real song's hand-verified reference
# timing (exact match), not just the isolated word-timestamp comparison.
ENABLE_WHISPERX_NO_VAD = True

# --- Pass 3 note assignment (lyric_alignment.py) ---------------------------
# How large an ASR gap (seconds) between two consecutive words is still
# small enough to treat them as one continuous sung phrase for NOTE-
# ASSIGNMENT purposes (see _group_words_by_gap). Deliberately a SEPARATE
# constant from MIN_LINE_GAP_SEC (phrasing.py's silence threshold for
# forcing a DISPLAY line break) even though it starts at the same value --
# the two serve genuinely different purposes (which notes a word gets vs.
# where a '-' appears in the output) and reference-lyrics line breaks must
# never affect this one, confirmed as a real bug when they did (see
# lyric_alignment.py's module docstring).
NOTE_ASSIGNMENT_MAX_GAP_SEC = 0.35

# lyrics_lookup.align_words_to_reference: when an uneven replace block
# clamps several ASR words onto the SAME single reference token (see that
# function's own comment -- a hyphenated repeated-unit token like LRC's
# "Do-do-do-do-do" clamped across the several ASR words it actually
# spans), a real confirmed case had a totally unrelated ASR word ~10.6s
# later (in silence, nothing else for the whole-sequence matcher to pair
# it with) swept into the same block purely because it was the last ASR
# word before the sequence ended. A gap this large from the previous ASR
# word in the block is never the same repeated-token run, so that word is
# left unmatched (line_id inherited from context, same as any other
# unmatched word) instead of getting the block's reference text stamped
# onto it. Deliberately generous vs. normal intra-phrase gaps (usually
# well under 0.5s) so it never fires on a real, if slightly slow, repeat.
REFERENCE_CLAMP_MAX_GAP_SEC = 2.0
# Same repeat-clamp mechanism, different failure mode (real case: David Bowie
# - Magic Dance, 2026-08-10): the decoder hallucinated a real "Dance, magic,
# dance" x5 passage into ~90 garbage ASR tokens, producing an extreme uneven
# replace block (a_len >> b_len) where nearly all of them clamped onto the
# SAME single reference token ("magic,"). The repeat-clamp path cycles that
# token's own hyphenated syllables across the repeats (built for a much
# smaller case -- LRC's "Do-do-do-do-do", ~5 real repeats) -- with ~90
# repeats and only 2 syllables ("mag"/"ic"), the cursor ran off the end and
# froze on the LAST syllable, stamping "ic" onto ~89 real notes as if it
# were confirmed reference text. This many ASR words clamping onto one
# reference token is itself the signal something is wrong (real repeated
# ad-libs don't run this high) -- past this count, don't trust a fabricated
# reference_text at all; fall back to keeping the ASR word's own raw text,
# same treatment as an ordinary uneven block gets by default.
REFERENCE_CLAMP_MAX_REPEAT = 8
# align_words_to_reference drops an ASR word with NO reference counterpart
# at all (a difflib "delete" block -- real cases: a non-lyrical audio intro
# decoded as fake lyrics, a trailing hallucination after the song's real
# last word) rather than keeping it, per the user's explicit policy: if a
# word isn't in the trusted reference, it shouldn't appear in the final
# song. BUT a real, confirmed regression (Trixie Mattel - Video Games,
# 2026-08-13) showed this is only safe for SHORT runs: on a repeat-heavy
# song ("It's you, it's you, it's all for you" x4+), difflib's own global
# alignment can misclassify a LARGE stretch of genuinely-correct, real
# reference-matching content as one giant "delete" block (219 of 355 words
# in the real failing run) -- the same repeated-phrase-disambiguation
# failure class this project has hit repeatedly elsewhere (verify-
# placement, realign.py's lrc_mode, --zone-boundary-snap), just surfacing
# in the reference-text alignment itself this time, not note placement.
# A short delete-run is almost always a genuine hallucination/ad-lib the
# reference doesn't have (both real motivating cases were 1-2 words); a
# long one is far more likely a real alignment failure than 6+ consecutive
# words of genuine non-lyrical content. Past this count, fall back to the
# OLD behavior (keep the ASR words, line_id inherited from context) rather
# than risk mass-deleting real lyrics -- same "count is itself the signal"
# shape as REFERENCE_CLAMP_MAX_REPEAT above, just for a different opcode.
REFERENCE_DELETE_MAX_RUN = 5
# align_words_to_reference's default matching runs ONE global, text-only
# difflib.SequenceMatcher over the WHOLE song's ASR words vs the whole
# reference -- it has no notion of TIME, only text. When a phrase repeats
# multiple times in both the song and the reference (a chorus sung 3x),
# SequenceMatcher can pair the WRONG occurrences together (it only
# maximizes total matched-token count, not chronological correspondence):
# real confirmed case (Trixie Mattel - Video Games, 2026-08-14) -- the
# ASR's 2nd real singing of "It's you, it's you, it's all for you,
# everything I do" matched the reference's 1st occurrence, leaving the
# ASR's 1st real singing of the SAME chorus (plus everything after it, up
# to the 2nd repeat -- 219 words) as one giant unmatched "delete" block.
# Those unmatched words then inherited a stale line_id from whatever
# matched right before the gap ("go play a video game"), and since
# phrasing.py's build_lines treats a matching line_id on both sides as
# fully overriding gap-based break heuristics, this suppressed a real
# line break and produced a spurious one elsewhere in the same passage.
#
# REJECTED APPROACH (2026-08-14, don't re-attempt without new evidence):
# a full per-LINE, time-WINDOWED re-match (bound each reference line's
# own text matching to a time window around its calibrated timestamp,
# so a repeated phrase elsewhere in the song is physically excluded from
# a given line's window). Fixed the motivating Video Games bug directly,
# but caused a severe real-audio regression on the SAME song in the
# FIRST attempt (word count 353->277, F1 84.0%->14.3%, from window-
# boundary spillover splitting a line's own tail words into the next
# line's window). A SECOND attempt (gap-grouping words before assigning
# them to a window, so a contiguous phrase can never be split by a
# boundary) fixed that specific spillover but still regressed severely
# elsewhere on the same song (F1 down to 21.7%, a different repeated-
# passage cross-contamination). Two independent regressions from two
# different angles on the same underlying idea -- consistent with this
# project's own repeated-phrase-disambiguation history (CLAUDE.md: hit
# independently in at least 5 other mechanisms already). Fully removed,
# not kept around disabled.
#
# ACTUAL FIX: when True and the reference lyrics came with synced (LRC)
# timestamps, `align_words_to_reference` NEVER re-windows or re-matches
# TEXT at all -- the whole-song text match above runs completely
# unchanged. Only the existing "fill remaining unmatched words" step
# (words the text match couldn't place, line_id=None) changes: instead
# of blindly inheriting the nearest matched neighbor's line_id (which is
# what froze that whole 219-word run on one stale value), each such word
# is assigned to whichever reference line's own (calibrated -- same
# lrc_timing.two_tier_time_calibration gate as everywhere else in this
# codebase) timestamp is nearest ITS OWN real ASR time. This can't
# reintroduce the rejected approach's failure class, because it never
# touches text matching/correctness -- only which line a word that's
# ALREADY unmatched gets attributed to for break purposes. Falls back to
# the old "inherit nearest matched neighbor" rule, unchanged, whenever no
# synced lyrics are available or calibration isn't confident.
# Real-audio validated (2026-08-14): Trixie Mattel - Video Games (the
# motivating case) -- fixes the reported break bug, ground-truth F1/
# recall/precision/pitch-class accuracy all IDENTICAL before/after (this
# mechanism only ever touches line_id on already-unmatched words, never
# timing/pitch/text); Great Big Sea - Ordinary Day (forced onto the
# standard, non-MXL+LRC path via --no-mxl-lrc-primary since it has a
# valid MXL+LRC candidate that would otherwise skip this code entirely)
# -- byte-identical output with the flag on vs off, confirming a true
# no-op on a song with nothing for it to fix.
ENABLE_TIME_BASED_LINE_ASSIGNMENT = True
# When splitting a pass-1 note at a word boundary (_split_notes_by_word_
# boundaries) leaves a word's leading piece shorter than this, it's DROPPED
# instead of becoming its own syllable (see _drop_leading_slivers) -- a real
# lyric syllable shouldn't visually attach to a near-inaudible fragment left
# over from an imprecise ASR word-start timestamp (commonly an unvoiced
# consonant like s/f with no real pitch) landing a hair inside the PREVIOUS
# word's real note. The dropped time is NOT folded into the next note's
# start either (an earlier version did this, stretching the next note
# backward -- wrong for UltraStar, where a note's start tells the player
# when to begin vocalizing a pitch that doesn't actually exist yet at that
# point) -- the next piece keeps its own pass-1-detected timing untouched.
# Deliberately a separate, more generous constant from MIN_NOTE_DURATION_SEC
# (pass 1's glitch-filtering threshold, 0.06s) -- confirmed in practice that
# reusing 0.06s here was too tight: a real 63ms sliver on "fall" (in "And if
# they fall as Lucifer fell") slipped through. 0.12s is comfortably past
# that case with margin, while pass 1 legitimately produces plenty of real
# sub-100ms notes elsewhere in fast passages, so this must stay well short
# of "drop any short note" territory.
SLIVER_DROP_MAX_DURATION_SEC = 0.12

# --- Chunk-based re-transcription verification (verification.py) ----------
# A word that got zero pass-1 notes (a "fallback" word -- see
# lyric_alignment.py) is always considered suspicious.
# Padding (seconds) added on each side of a suspicious word's own ASR
# span when cropping a fresh, isolated re-transcription window. Wide
# enough to give the ASR model real context -- see "never run X on a
# tiny, isolated clip" under Lessons learned in CLAUDE.md, the same
# principle that applies to pYIN pitch analysis applies to ASR
# confidence too -- narrow enough to stay unlikely to capture a
# neighboring word's speech instead.
RECHECK_PAD_SEC = 1.0
# Text verification is on by default (--no-verify-words to disable): it
# never touches note timing/pitch, only ever swaps in word TEXT, and a
# reference-matched word is only ever replaced when the recheck actively
# CONFIRMS a different answer than what's already there (see
# verification.py's _resolve) -- low blast radius by construction.
ENABLE_WORD_VERIFICATION = True
# Verify every word, not just the ones pass 3 flagged suspicious -- the
# extra recheck calls are cheap next to Demucs/WhisperX, and this catches
# cases pass 3's own heuristics can't see (e.g. lyrics_lookup.py's
# "uneven block" case, where a word gets a reference line tagged but its
# text is deliberately left uncorrected pending a more confident check).
# --verify-suspicious-only restricts back to just the flagged words.
VERIFY_ALL_WORDS = True

# Pass 4 (optional, musicxml_reference.py): confirms/corrects pass-3
# syllable PITCH CLASS (never octave, never timing) against a
# user-supplied MusicXML file -- e.g. hand-downloaded sheet music for the
# same song. No automatic fetch exists (see CLAUDE.md's "MuseScore
# reference data" section for why: MuseScore's own free/automated
# download path was found to be actively blocked by the platform as of
# 2026-08, and no reliable alternative source was found either) -- this
# only ever runs when --musicxml-reference points at a real file.
#
# Real validation (2026-08-08, several already-validated songs, manually
# downloaded MXL files): sheet-music vocal-melody data agrees with
# trusted ground truth on PITCH CLASS 93-98% of the time once a per-song
# calibration offset is removed -- far above any of this project's own
# audio-only pitch sources. Coverage (how much of the song a given
# arrangement lyric-tags) varies a lot more, 23-91%.
#
# Minimum matched notes required before trusting a per-song calibration
# offset at all -- too few and a "clear majority" could just be chance.
MUSICXML_MIN_CALIBRATION_SAMPLES = 8
# The modal pitch-class offset among matched notes must cover at least
# this fraction of all matches, or calibration is considered too
# ambiguous to trust (real case found needing this: Gaston, merging all
# 3 vocal parts, showed a genuine bimodal split between two DIFFERENT
# raw semitone offsets that still agree at the pitch-class level, but a
# less careful check could land on a shaky plurality from actual noise).
MUSICXML_MIN_CALIBRATION_CONFIDENCE = 0.5
# When the full-population calibration above doesn't clear that bar,
# retry using only the top half of matches by OUR OWN note confidence --
# a noisy upstream detector (real case: Gaston, our baseline pyin
# accuracy only ~41%) dilutes the calibration signal with matches where
# our own pitch is simply wrong, not real per-song ambiguity; restricting
# to higher-confidence matches measurably cleans that up (confirmed:
# Gaston went from 39.5% agreement over all matches to 46.1% over the
# top half, same winning offset both times -- not a different answer,
# less noise around the same one). A LOWER bar is used for this retry on
# purpose: a plurality among an already-noise-reduced population is more
# trustworthy than the same plurality over the full one.
MUSICXML_MIN_CALIBRATION_CONFIDENCE_HIGH_CONF_SUBSET = 0.4
# A syllable's confidence after pass 4 corrects it -- boosted above
# whatever it came in with (the low-confidence case is exactly what this
# mechanism targets), but not set to 1.0: it's still calibrated-by-
# inference from a different source, not a direct pass-1 acoustic
# measurement.
MUSICXML_CORRECTED_CONFIDENCE = 0.75

# Whether pass 4 applies the best available calibration offset even when
# it doesn't clear MUSICXML_MIN_CALIBRATION_CONFIDENCE/_HIGH_CONF_SUBSET,
# rather than skipping the file entirely (see apply_musicxml_reference's
# force_calibration param). ON by default as of 2026-08-08: validated
# real end-to-end on all 7 MXL-having songs in the test set -- 4 showed
# NO change (batb, stars, gaston, sleeping_beauty_ouad -- their normal
# calibration already clears the bar on its own, so this is provably a
# no-op for them), 1 showed a small real gain (tarzan_son_of_man,
# +1.9pp), and 2 showed large real gains (little_mermaid +21.6pp,
# jungle_book_bare_necessities +19.0pp -- both had confirmed-unreliable
# pass-1 pitch for acoustic reasons -- see CLAUDE.md 0h -- where even a
# weakly-calibrated MXL reference beats trusting pass 1 at all). Zero
# regressions found across the whole set.
ENABLE_MUSICXML_FORCE_CALIBRATION = True


# --- LRC (LRCLIB synced-lyrics) line timing check (lrc_timing.py) ----------
# DIAGNOSTIC ONLY as of 2026-08-08 -- flags lines, never corrects them yet.
# See lrc_timing.py's module docstring for why: `verify_placement` was
# built with the same good intentions for a related problem and, when
# validated for real, produced a net REGRESSION on every metric on both
# songs tested, despite fixing some individual real cases. Not enabling
# this by default, or building auto-correction, until the flagging
# signal itself is cross-validated against real ground-truth timing
# error to confirm it's trustworthy.
ENABLE_LRC_TIMING_CHECK = False
# Minimum matched lines required before trusting a per-song time
# calibration offset at all -- mirrors MUSICXML_MIN_CALIBRATION_SAMPLES.
LRC_TIMING_MIN_CALIBRATION_SAMPLES = 5
# The modal per-line delta (1-second buckets) must cover at least this
# fraction of matched lines, or calibration is too ambiguous to trust.
LRC_TIMING_MIN_CALIBRATION_CONFIDENCE = 0.4
# Once calibrated, a line's residual (its own delta minus the calibration
# offset) beyond this is flagged. Line-level timestamps are inherently
# coarser than word-level pitch calibration, so this is deliberately
# more forgiving than any word-level tolerance would be.
LRC_TIMING_FLAG_TOLERANCE_SEC = 2.0
# Fallback tried only when the constant-offset check above fails: a
# robust (Theil-Sen) fit of delta = offset + slope*lrc_start, for songs
# where LRCLIB's synced timestamps drift smoothly relative to ours
# instead of sitting at one constant offset -- confirmed real on real
# audio (stars, tarzan, little_mermaid all had a genuine drift, not
# just noise around a constant -- see CLAUDE.md). Both bars are stricter
# than the constant-offset case: a 2-parameter fit can trivially match a
# handful of points exactly (e.g. any 2 points define a line), so more
# samples and a higher inlier fraction are required before trusting it.
LRC_TIMING_MIN_DRIFT_SAMPLES = 10
LRC_TIMING_MIN_DRIFT_CONFIDENCE = 0.5
# A candidate counts as an inlier to the fitted drift line if its own
# residual is within this many seconds -- same order of magnitude as the
# constant-offset case's 1-second bucket width, not the (looser)
# post-calibration flag tolerance above.
LRC_TIMING_DRIFT_INLIER_TOLERANCE_SEC = 1.5
# Tier 3, tried only when BOTH tier 1 (constant) and tier 2 (linear
# drift) fail their own confidence gates above: a song where whichever
# recording LRCLIB's synced lyrics were timed against was EDITED
# differently from ours (a repeated chorus removed, a bridge shortened,
# etc.) produces a DISCONTINUOUS drift -- the delta jumps, or the slope
# itself changes, partway through the song -- that no single constant or
# single global slope can fit, no matter how it's chosen. Two
# interchangeable implementations exist (see lrc_timing.py's own
# `_pava_isotonic`/`_piecewise_correction_anchors`): "isotonic" (PAVA
# monotonic regression -- try first, no manual spacing heuristics
# needed since PAVA pools noisy points instead of discarding them) and
# "piecewise" (direct linear interpolation between confident anchors,
# after greedily dropping any anchor that would make the correction
# non-monotonic). Config-selectable rather than always building both --
# they solve the exact same problem, and this project doesn't maintain
# two production code paths for one job (see CLAUDE.md's own "Removed /
# rejected approaches" history of exactly this kind of duplication).
LRC_TIMING_DRIFT_MODEL = "isotonic"  # "isotonic" | "piecewise"
# Tier 3's own noise filter: a candidate must be within this many seconds
# of tier 2's OWN Theil-Sen fit (even a fit that itself didn't clear
# LRC_TIMING_MIN_DRIFT_CONFIDENCE is still useful here purely as an
# outlier filter -- see the module docstring) to be trusted as real
# signal for tier 3 at all. Deliberately LOOSER than
# LRC_TIMING_DRIFT_INLIER_TOLERANCE_SEC above: tier 3 exists specifically
# to survive cases the tier-2 fit itself doesn't trust (a real regime
# change can sit several seconds off the GLOBAL fit by construction),
# so this only needs to catch raw text-matching noise / wrong-repeated-
# line-instance mismatches, not real per-segment drift.
LRC_TIMING_PIECEWISE_OUTLIER_TOLERANCE_SEC = 4.0
# Minimum anchors surviving the outlier filter (+ monotonicity, for
# "piecewise") before tier 3 is even attempted. Fewer than this can't
# distinguish a real multi-segment edit from "2 points define a line" --
# that's just tier 2 (already failed) with extra steps.
LRC_TIMING_PIECEWISE_MIN_ANCHORS = 6
# Reject the whole tier-3 attempt if any two ADJACENT surviving anchors
# (by lrc_start) are farther apart than this. Interpolating linearly
# across a huge silent/unmatched gap with only the 2 anchors bounding it
# is exactly the global-linear-drift case again, just applied locally
# with much less evidence -- 2 points can't distinguish a real step
# change from noise, so this should fall through to "uncalibrated" (same
# as tier 2 already found) rather than fabricate a locally-linear guess
# across a gap this wide.
LRC_TIMING_PIECEWISE_MAX_ANCHOR_GAP_SEC = 45.0
# REAL VALIDATION FINDING (2026-08-11, see lrc_timing.py's own module
# docstring for the full real-audio evidence) that MOTIVATED the
# refine-vs-rescue split below: tier 3's confidence number is NOT
# directly comparable to tier 1/2's on its own -- a flexible piecewise/
# isotonic fit can look confident (84% on a real tested case, David
# Bowie "Heroes" vs a choral-cover candidate) on a recording that's
# wrong for a STRUCTURAL reason (different arrangement/artist), not just
# a timing-drift reason, because more degrees of freedom make scattered
# disagreement easier to locally explain away than a rigid 2-parameter
# line can. A genuine real win was ALSO found on a different song (33%->
# 87% within 1s on Chicago's own previously-uncalibratable real
# candidate) -- tier 3 is not broken, but a HIGH tier-3 confidence alone
# (regardless of the number) can't distinguish "genuinely complex real
# drift" from "overfit to a wrong recording", since both look like a
# flexible model finding SOME fit. The fix isn't a stricter residual
# threshold (same blind spot, just moved) -- it's asking for INDEPENDENT
# evidence, checked below.
#
# `two_tier_time_calibration` now splits tier 3 into two cases based on
# whether tier 1/2's OWN best (still-rejected) fit already cleared this
# floor BEFORE tier 3 ran:
#   - "refine" (floor cleared): two independent RIGID models (constant,
#     linear) already partially agree with each other on this candidate
#     -- tier 3 is sharpening a shape they were already circling, not
#     inventing one from nothing. Proceeds exactly as before, no
#     structural check required.
#   - "rescue" (floor NOT cleared): neither rigid model found ANY real
#     support -- tier 3 succeeding here means the flexible model is
#     supplying both the hypothesis (a discontinuous drift exists) and
#     its own validation (a good-looking fit to it), with no independent
#     check that the candidate is even the right recording. Gated behind
#     `structural_check` (see `two_tier_time_calibration`'s own
#     docstring) -- declines (falls through to uncalibrated) unless the
#     caller provides a check AND it passes.
# Threshold chosen from the only 2 real "borderline" (still-rejected-by-
# tier-1/2) data points available: Heroes' wrong-recording candidate
# (tier1 21%, tier2 26% -- the max of the two is what's compared here)
# vs. Chicago's genuinely-rescuable candidate (tier1 24%, tier2 38%).
# 0.30 sits between them, correctly classifying both -- but this is a
# 2-point real sample, not a robustly-fit threshold; revisit if a real
# case lands on the wrong side of it.
LRC_TIMING_RESCUE_MIN_PRIOR_CONFIDENCE = 0.30
# Odd/even-anchor holdout validation (see lrc_timing._holdout_residual_
# sec): fits tier 3's correction on the ODD-indexed anchors only, scores
# it against the EVEN-indexed anchors' own real times -- targets the
# general "genuine drift vs. fit to noise" question (not just the wrong-
# recording case the refine/rescue split targets), cheap since the
# anchor set already exists. Computed and returned for every tier-3
# result (both refine and rescue) as a DIAGNOSTIC value, not currently a
# hard gate -- real-tested on Heroes (2026-08-11): did NOT by itself
# flag the wrong-recording case (holdout residual was NOT dramatically
# worse than Chicago's own genuine-rescue case), so it doesn't close the
# gap the refine/rescue split targets on its own. Kept because it's
# cheap and may still catch a different failure shape (a fit that's
# locally smooth but doesn't generalize between its own anchors) that
# the refine/rescue split, which only looks at tier 1/2's prior
# confidence, can't see -- see lrc_timing.py's own docstring for the
# real numbers.
LRC_TIMING_HOLDOUT_MIN_ANCHORS = 4


# --- Existing-file verification (verify_existing_song.py) ------------------
# When an existing "<Artist> - <Title>.txt" is found in the input folder (or
# --existing-txt points at one), this compares its own pitch/timing against
# a FRESH pipeline run of the same audio, and only overwrites it if real
# problems are found. OFF by default -- decided with the user: unlike this
# project's other on-by-default features (which only ever ADD a correction
# on top of what's already there), this one can result in NOT writing
# output the user expected on a plain re-run, which is a qualitatively
# different kind of surprise than "one more automatic fix."
ENABLE_EXISTING_TXT_CHECK = False
EXISTING_TXT_MIN_MATCHED = 10
# NOT YET EMPIRICALLY VALIDATED -- picked by analogy to this project's other
# calibration/agreement bars, not measured against this pipeline's own
# run-to-run self-agreement. Before trusting these for real, run the same
# already-validated song's pipeline twice and measure how much the
# pipeline agrees with ITSELF (pass-1 pitch detection is not fully
# reproducible run-to-run even on identical audio -- see CLAUDE.md's
# CREPE/RMVPE non-determinism notes), so these bars are calibrated against
# real self-noise rather than guessed.
EXISTING_TXT_MIN_PITCH_ACCURACY = 0.85
EXISTING_TXT_TIMING_TOLERANCE_SEC = 0.5
EXISTING_TXT_MIN_TIMING_AGREEMENT = 0.85

# Coverage gate: what fraction of EACH side's own word-start syllables must
# text-match the other side at all before the comparison is trusted. Added
# after a real, confirmed blind spot (2026-08-09 MXL+LRC bug-hunt session):
# words that fail to text-match (e.g. garbled/wrong output text) simply
# never appear in the pitch/timing candidate list at all, so a real ~10%
# failure rate was silently invisible to pitch_class_accuracy/
# timing_within_tolerance_pct, which are only ever computed over whatever
# DID match. Low coverage on EITHER side now fails the gate even if the
# matched subset's own accuracy looks perfect.
EXISTING_TXT_MIN_COVERAGE = 0.85


# --- Reference lyrics (lyrics_lookup.py) ------------------------------------
# LRCLIB (lrclib.net) is tried first -- community-sourced, has a real search
# API (artist_name/track_name, not lyrics.ovh's rigid /artist/title path),
# and often has synced (per-line-timestamped) lyrics. lyrics.ovh is kept as
# a fallback for whatever LRCLIB doesn't have.
#
# LRCLIB's /api/search can return several same-title candidates (different
# recordings/albums, or occasionally a wrong-language mistag). Preferred
# candidate = closest match to our audio's own duration; a candidate whose
# duration is off by more than this is heavily penalized rather than
# outright excluded (still usable as a last resort if nothing better exists).
LRCLIB_DURATION_TOLERANCE_SEC = 60.0

# Real case that motivated this: Gaston's lyrics.ovh lookup silently
# returned a SPANISH-language reference for an English song (same
# artist/title string, wrong recording) -- verify_words then trusted it as
# ground truth for TEXT and corrupted the whole song. Any reference source
# can make this mistake, not just lyrics.ovh, so the gate lives here,
# independent of which source answered: the fetched reference's own word
# vocabulary must overlap the ASR transcript's vocabulary by at least this
# much (difflib.SequenceMatcher.ratio() over normalized word-token
# sequences) before it's trusted at all. A right-language, right-song
# reference and a same-language ASR transcript of the same audio should
# share the large majority of their words; a wrong-song/wrong-language
# reference shares almost none.
REFERENCE_LYRICS_MIN_MATCH_RATIO = 0.25


# --- MXL+LRC primary generation (mxl_lrc_generator.py) ----------------------
# Real, ground-truth-validated (2026-08-08/09 session, Chicago - "When
# You're Good to Mama"): MXL for pitch + LRCLIB synced-lyrics line starts as
# real-time anchors + real ASR to place words within a line beats every
# audio-only approach tried this session -- 100% pitch-class accuracy, 99.0%
# timing within 500ms, 92ms mean error, with ZERO CREPE/pYIN pass-1 needed.
# Shipped as the default generation path per the user's explicit decision.
ENABLE_MXL_LRC_PRIMARY = True

# Candidate selection stays intentionally permissive (see
# mxl_lrc_generator.py's module docstring for why) -- the real validity gate
# is MXL_LRC_MIN_ASR_PLACEMENT_RATE below, not these upfront filters. Two
# real wrong-recording candidates (BATB, Les Miserables - Stars) BOTH passed
# generous duration+content filtering this session; tightening these numbers
# would not have caught either case.
MXL_LRC_DURATION_TOLERANCE_SEC = 15.0
MXL_LRC_MIN_CONTENT_MATCH_RATIO = 0.3

# The decisive quality gate: does our OWN audio's real transcript actually
# agree with the matched LRC candidate's line timings? A wrong-recording
# candidate's line timestamps don't correspond to what's really sung at
# those moments in OUR audio, so this collapses -- a ground-truth-free
# signal that falls out of the placement step itself. Initial conservative
# thresholds, calibrated against this session's one real positive case
# (Chicago: 92% ASR-placed, 2/139 non-monotonic fixes) -- real negative-case
# validation (BATB/Stars) is this feature's own launch verification, not
# assumed; revisit these numbers if that validation suggests otherwise.
MXL_LRC_MIN_ASR_PLACEMENT_RATE = 0.5
MXL_LRC_MAX_NONMONOTONIC_RATE = 0.1

# A text match alone isn't enough to trust an ASR word's timestamp -- real
# case found this session: 'hen' text-matched correctly but whisperx's own
# alignment confidence was 0.003 (essentially "no idea"), and its timestamp
# was genuinely wrong (0.77s off) as a result, independent of any
# beat-quantization issue. Below this confidence, a match is treated as if
# it never happened (falls through to the MXL-tempo-estimated placement
# below) rather than trusting a low-confidence position blindly. Matches
# this project's own existing "LOW SCORE" convention for flagging whisperx
# output (transcription.py's debug logging uses the same 0.3 cutoff).
MXL_LRC_MIN_ASR_WORD_CONFIDENCE = 0.3

# assign_words_to_lines' clean-text substitution: an MXL word that lands in
# a 1:1 "replace" slot against exactly one LRC word (in the whole-sequence
# alignment -- i.e. difflib's own global alignment already decided these
# two are the best positional fit, anchored by correctly-matched context on
# both sides) is trusted as the SAME word with an OCR/spelling difference
# if their character-level similarity clears this ratio -- not just an
# exact string match. Real cases confirmed to need this (empirically
# measured difflib.SequenceMatcher ratios): "systern"~"system" (0.769),
# "eystern"~"system" (0.615, the lowest of the real cases found -- sets
# the floor here), "matizon"~"matron" (0.769), "recause"~"because" (0.857).
MXL_LRC_FUZZY_TEXT_MIN_RATIO = 0.6

# Largest MXL-word-block/LRC-word-block size `assign_words_to_lines` will
# attempt a whole-block fuzzy match on (see its own docstring) when a
# "replace" opcode isn't a clean 1:1 shape -- e.g. MXL's OCR merged two
# real words into one token, or split one real (hyphenated) word into
# two. Bounded on both sides by real matches either way, so this is never
# an unconstrained text search; the cap just keeps the block-level ratio
# comparison meaningful (a very long block's ratio gets less reliable as
# a "same content" signal) and keeps `_distribute_words_to_slots`'s
# merge/split fallbacks from producing something too coarse to be useful.
MXL_LRC_BLOCK_MAX_WORDS = 6

# Default real-seconds-per-quarter-note rate used to estimate a fallback
# word's own duration from its MXL note value when NO local tempo anchor
# is available at all (should be rare -- every line has its own LRC-based
# (t0, t1) window to derive a rate from; this is a last-resort floor).
MXL_LRC_DEFAULT_QUARTER_NOTE_SEC = 0.3


# --- realign.py "validate" strategy (PROTOTYPE, 2026-08-09) ----------------
# A word's own ORIGINAL start (after the single global GAP/drift correction
# computed by compute_gap_calibration) is trusted EXACTLY -- position AND
# length left completely untouched -- if a confident whole-song ASR match
# for that word lands within this many seconds of that expected position.
# Only START proximity is checked, never END (see validate_words_against_asr's
# own docstring for why -- ASR is known-unreliable at the end of a
# sustained/held note). NOT yet empirically validated against real ground
# truth beyond informal spot-checks -- picked by analogy to this project's
# other word-level timing-agreement tolerances (e.g. EXISTING_TXT_TIMING_
# TOLERANCE_SEC=0.5, MXL_LRC's own ASR-match acceptance), not measured.
REALIGN_VALIDATE_TOLERANCE_SEC = 0.3

# --- transcription.py long-segment re-windowing (PROTOTYPE, 2026-08-09) ----
# Real bug (David Bowie - Magic Dance): a whisper DECODER segment can
# declare a much longer span than its own text plausibly needs (real case:
# 20.1s for an 8-word line, "In 9 hours and 23 minutes, you'll be mine.")
# -- a symptom of the decoder silently DROPPING real intervening content
# (the "Jump Magic Jump!..." section actually sung in there). Forcing
# wav2vec2 to align the short remaining text against the WHOLE oversized
# window produces a confidently-wrong placement: real case, crammed into
# the first ~3s of the window (mean word score 0.56), nowhere near the
# true ~114-118s position (user confirmed by ear).
#
# Real, validated fix: for any segment at least this long, sweep smaller
# fixed-width candidate windows across its own declared span (step below)
# and keep whichever gives the best mean word score, PROVIDED it beats the
# baseline (whole-segment) score by at least the margin below -- otherwise
# keep the baseline alignment untouched. Real validation: a FINE (1s step)
# sweep found a sharp, correct peak (mean score 0.751) at the user's
# confirmed position that a coarser first attempt (2s step) skipped right
# over, and that peak beat every other candidate -- including a
# plausible-looking false local optimum (0.649) -- by a wide, unambiguous
# margin (the false peak only beat the baseline by ~0.08-0.09, well under
# the margin below, which is why a threshold this size should reject it).
# `REWINDOW_MIN_SCORE_IMPROVEMENT` is a first estimate from this ONE real
# case (roughly half the true peak's 0.19 margin over baseline) -- not
# yet validated across multiple songs.
#
# ON by default everywhere (2026-08-10) -- real-validated across 12 total
# runs (8 realign-mode songs, 4 full generation-pipeline songs), zero
# regressions, 6 genuine verified fixes carried through to the actual
# written output (not just the intermediate ASR score). See CLAUDE.md's
# "Long-segment re-windowing" section for the full validation history.
# `--no-rewindow-long-segments` (CLI, both realign.py and main.py) opts
# back out if ever needed.
REWINDOW_ENABLED = True
REWINDOW_MIN_SEGMENT_DURATION_SEC = 10.0
REWINDOW_CANDIDATE_WIDTH_SEC = 10.0
REWINDOW_STEP_SEC = 1.0
REWINDOW_MIN_SCORE_IMPROVEMENT = 0.10


# --- ASR quality retry (PROTOTYPE, 2026-08-10) -------------------------------
# Real case that motivated this ("Gold"): WhisperX can, on a given run,
# silently drop/garble whole passages rather than just mistranscribing a
# word here and there -- decoder hallucination (see transcription.py's
# rewindow section) or an unlucky sampling that recognizes far fewer real
# words than actually exist. Rather than special-casing "completely
# missing lines," this reuses whichever real per-word match-rate metric a
# code path ALREADY computes for its own "does this file/candidate really
# match this audio" gate -- realign.py's `RealignQuality.anchor_rate` and
# mxl_lrc_generator.py's `MxlLrcQuality.asr_placement_rate` both already
# share `MXL_LRC_MIN_ASR_PLACEMENT_RATE` as their existing warn/reject
# bar, so that's reused as-is here too: if a run would already be warned
# about or rejected for a low match rate, retry the whole transcription
# with `RETRY_ASR_MODEL` once before accepting that verdict. A per-word
# match-rate against real ground truth (an existing file's own text, or a
# matched LRC/MXL candidate) catches partial degradation too, not just
# fully-missing lines -- garbled decoding, wrong-language ASR, and
# hallucinated segments all show up as a depressed match rate the same
# way completely-dropped lines do.
#
# Only fires when the current model isn't already `RETRY_ASR_MODEL` (no
# point re-running the same model twice), and the retry is only ever
# ACCEPTED if its own match rate beats the original's -- if the bigger
# model does no better (e.g. the original file's lyrics genuinely don't
# match this audio at all), the original transcription's result is kept.
#
# PROTOTYPE -- needs real-audio validation before being trusted the way
# `REWINDOW_ENABLED` now is. ON by default per this project's own
# convention (a feature that only spends extra time when a run already
# looks bad is low-risk by construction, and only ever pays that cost
# once per song) -- `--no-retry-low-quality-asr` opts out.
RETRY_ASR_MODEL = "large-v3"
RETRY_LOW_QUALITY_ASR = True

# The standard (non-MXL) pass-1-4 fallback path has no existing-file/MXL
# candidate to measure a real match rate against -- the closest already-
# computed, ground-truth-ish signal there is how well the ASR
# transcript's OWN vocabulary overlaps a fetched reference-lyrics
# candidate that has ALREADY cleared `REFERENCE_LYRICS_MIN_MATCH_RATIO`
# (0.25, "is this even the right song"). This threshold is deliberately
# higher than that one -- it isn't asking "is this the right song," it's
# asking "did ASR transcribe it well" -- first estimate, not yet
# validated across multiple songs.
RETRY_ASR_MIN_REFERENCE_MATCH_RATIO = 0.6

# Per-PASSAGE trigger, added 2026-08-10 after real validation found the
# whole-song metrics above miss a localized failure: a single passage can
# be completely dropped by ASR (Trixie Mattel - Gold: the reference's own
# "Doo doo doo doo doo. They start to play." produces a real difflib
# 'insert' block -- reference words with ZERO corresponding ASR word --
# at one repeat of the chorus, while an aggregate 326-word transcript
# stays well above `RETRY_ASR_MIN_REFERENCE_MATCH_RATIO` because the rest
# of the song transcribed fine) or a real passage's TIMING can be dragged
# badly wrong by one bad decoder segment even though the overall anchor
# rate stays above `MXL_LRC_MIN_ASR_PLACEMENT_RATE` (David Bowie - Magic
# Dance with `small.en`: a hallucinated "Dance Magic Dance Dance Magic
# Dance..." loop segment left a real "Slap that baby... I saw my baby"
# passage ~12-14s late in the realigned output, while the song-wide
# anchor rate held at 58%). Both are the SAME underlying shape --
# something real and contiguous going wrong in one place, invisible to a
# whole-song average -- just surfaced through different existing
# machinery per code path (`lyrics_lookup.largest_unmatched_reference_run`
# for main.py's reference-alignment path; realign.py's own `confident`
# per-word array, already computed by `match_words_to_asr`, for the
# longest consecutive False run). Both are first estimates, sized to
# clearly clear Gold's ~9-word dropped run and comparable-scale runs,
# without firing on an ordinary single-word ad-lib gap -- not yet broadly
# validated across multiple songs the way the whole-song metrics are
# starting to be.
RETRY_ASR_MIN_UNMATCHED_REFERENCE_RUN = 5
RETRY_ASR_MIN_UNCONFIDENT_RUN = 5

# --- Force-align known-text gaps (PROTOTYPE, 2026-08-10) --------------------
# Adapted from UltraStarKaraokeMaker (github.com/walterfr/UltraStarKaraokeMaker,
# python-sidecar/pipeline/align.py's `realign_gap_windows`) after comparing
# its real output on "Trixie Mattel - Gold" against ours -- it correctly
# recovered a "Do-do-do-do-do" backing-vocal passage our own pipeline
# dropped outright, via a real wav2vec2 CTC forced alignment of the KNOWN
# missing text restricted to the audio window between the nearest measured
# neighbors. Structurally different from (and more reliable than) two
# things already in this codebase: `RETRY_ASR_MODEL` (retries the WHOLE
# song with a bigger model, probabilistic -- confirmed on a real Gold run
# that the retry can be accepted for improving the aggregate while NOT
# actually fixing the specific passage that triggered it) and a local
# text-search rematch (re-searches ASR's OWN already-decoded words in a
# window -- useless if the decoder produced zero words there at all,
# which is exactly this failure mode; tried as `realign.
# rematch_local_gaps`, removed 2026-08-16, see CLAUDE.md's "Removed /
# rejected approaches" -- net regression, this forced-alignment approach
# is the one that actually works for this failure mode). Forced alignment
# doesn't care whether the decoder ever produced these words; it directly
# measures where the GIVEN text best fits.
# See `transcription.force_align_words_in_window` for the primitive
# (validation logic ported directly from USKMaker's own function) --
# used by `realign.py` (on gaps in an existing file's own text) and
# `lyrics_lookup.py` (on reference-lyrics text ASR completely dropped).
# `MIN_WINDOW_BASE_SEC`/`MIN_WINDOW_SEC_PER_WORD` and `WINDOW_SLOP_SEC`
# are carried over from USKMaker's own values as a starting point, not
# independently re-derived.
FORCE_ALIGN_GAPS = True
FORCE_ALIGN_MIN_WINDOW_BASE_SEC = 0.10
FORCE_ALIGN_MIN_WINDOW_SEC_PER_WORD = 0.08
FORCE_ALIGN_WINDOW_SLOP_SEC = 0.5


@dataclass
class PipelineOptions:
    """Every knob `run_pipeline`/`run_batch` (main.py) need, decoupled from
    argparse -- lets the GUI (gui.py) and the CLI (main.py's `run()`) build
    this the same way and call the exact same pipeline code. Field defaults
    mirror `build_arg_parser`'s own defaults exactly; keep the two in sync
    when either changes. Supersedes an earlier, stale version of this class
    that referenced a `genius_token`/`device` field from a since-replaced
    lyrics source and a since-removed CPU fallback path -- this version was
    unused by any code before this rewrite."""
    artist: Optional[str] = None
    title: Optional[str] = None
    audio_file: Optional[str] = None  # disambiguates a folder with >1 real audio file
    work_dir: Optional[str] = None
    whisper_model: str = DEFAULT_WHISPER_MODEL
    verify_whisper_model: str = DEFAULT_WHISPER_MODEL
    demucs_model: str = DEFAULT_DEMUCS_MODEL
    bpm_override: Optional[float] = None
    skip_separation: bool = False
    vocals_path: Optional[str] = None
    fetch_lyrics: bool = True
    no_video_sync: bool = False
    no_whisperx: bool = False
    no_transcribe: bool = False  # DIAGNOSTIC (default off): skip the WhisperX
        # decoder entirely, force-align a pinned LRC candidate's own known
        # line text instead -- see transcription.force_align_reference_lyrics
    whisperx_no_vad: bool = ENABLE_WHISPERX_NO_VAD
    rewindow_long_segments: bool = REWINDOW_ENABLED
    retry_low_quality_asr: bool = RETRY_LOW_QUALITY_ASR
    force_align_gaps: bool = FORCE_ALIGN_GAPS
    time_based_line_assignment: bool = ENABLE_TIME_BASED_LINE_ASSIGNMENT
    merge_connected_melisma: bool = MERGE_CONNECTED_MELISMA_TAILS
    verify_words: bool = ENABLE_WORD_VERIFICATION
    verify_all_words: bool = VERIFY_ALL_WORDS
    musicxml_reference: Optional[str] = None
    musicxml_part: Optional[str] = None
    musicxml_force_calibration: bool = ENABLE_MUSICXML_FORCE_CALIBRATION
    lrc_timing_check: bool = ENABLE_LRC_TIMING_CHECK
    pitch_smooth_window: float = PITCH_SMOOTH_WINDOW_SEC
    note_split_semitones: float = NOTE_SPLIT_SEMITONES
    min_note_beat_fraction: float = MIN_NOTE_BEATS_FRACTION
    silence_threshold_db: float = SILENCE_THRESHOLD_DB_BELOW_PEAK
    silence_floor_db: float = SILENCE_ABSOLUTE_FLOOR_DB
    spike_max_duration: float = SPIKE_MAX_DURATION_SEC
    spike_jump_semitones: float = SPIKE_MIN_JUMP_SEMITONES
    pitch_source: str = DEFAULT_PITCH_SOURCE
    no_pass1_debug: bool = False
    no_debug_log: bool = False
    quiet: bool = False
    # New in the folder-based/batch/YouTube/GUI rework -- see CLAUDE.md.
    existing_txt_check: bool = ENABLE_EXISTING_TXT_CHECK
    existing_txt_path: Optional[str] = None
    youtube_url: Optional[str] = None
    youtube_audio_only: bool = True
    batch: bool = False
    delete_work_files: bool = False
    # Interactive LRCLIB disambiguation -- GUI only, never set by the CLI.
    # `pinned_lyrics` (actually a lyrics_lookup.LrcLibCandidate; kept as a
    # forward-reference string here to avoid a circular import, since
    # lyrics_lookup.py itself imports this module) is a manual pre-run
    # pick that always wins outright, skipping the network fetch entirely.
    # `lyrics_disambiguation_callback` is the GUI's mid-run fallback for
    # when nothing was pre-picked -- only ever consulted when
    # `lyrics_ambiguity_prompt` is on AND `pinned_lyrics` is None.
    pinned_lyrics: Optional["LrcLibCandidate"] = None
    lyrics_ambiguity_prompt: bool = False
    lyrics_disambiguation_callback: Optional[Callable[[List[Any]], Optional[Any]]] = None
    # MXL+LRC primary generation path -- see mxl_lrc_generator.py.
    # `lrclib_id`, if set, always wins over both search AND `pinned_lyrics`
    # for candidate selection everywhere a candidate is needed (this new
    # path AND the old reference-lyrics-fetch step) -- resolved once, early,
    # in main.py via lyrics_lookup.fetch_lrclib_by_id.
    mxl_lrc_primary: bool = ENABLE_MXL_LRC_PRIMARY
    lrclib_id: Optional[int] = None
    # GUI-only, never set by the CLI (which always auto-falls-back with just
    # a warning log, same convention as every other non-interactive CLI/
    # batch decision in this codebase). Called with the failure reason when
    # the MXL+LRC quality gate fails or nothing usable was found; True
    # continues with the standard audio-based fallback, False cancels the
    # whole run.
    mxl_lrc_fallback_callback: Optional[Callable[[str], bool]] = None
