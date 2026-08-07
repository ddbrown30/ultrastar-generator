"""Defaults and constants shared across the pipeline."""

from dataclasses import dataclass

# --- UltraStar note types -------------------------------------------------
NOTE_NORMAL = ":"
NOTE_GOLDEN = "*"
NOTE_FREESTYLE = "F"
NOTE_RAP = "R"
NOTE_RAP_GOLDEN = "G"

# --- Audio / file conventions ---------------------------------------------
AUDIO_EXTS = (".mp3", ".ogg", ".oga")
VIDEO_EXTS = (".avi", ".mp4")
IMAGE_EXTS = (".jpg", ".jpeg")

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

# Word-level ASR model to use with faster-whisper by default. "small.en" is
# a good accuracy/speed tradeoff on GPU; use "medium.en" or "large-v3" for
# better lyric accuracy at the cost of more VRAM/time.
DEFAULT_WHISPER_MODEL = "small.en"

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

# CREPE (torchcrepe) runs alongside pYIN as a second, independent pitch
# estimate per frame -- CREPE is a learned model and is generally more
# robust than pYIN's DSP-based approach when there's accompaniment bleed
# in the isolated vocal stem. Where the two agree (within
# CREPE_AGREEMENT_SEMITONES), CREPE's pitch is trusted and used with a
# confidence boost. Where they disagree, that frame's pitch is kept from
# pYIN (unchanged from pre-CREPE behavior) but downweighted -- NOT marked
# unvoiced, since forcing unvoiced would itself create a spurious note
# boundary at exactly the frames we're least sure about; downweighting
# instead means disagreement can suppress a frame's influence on a note's
# final reported pitch without being able to fabricate a boundary.
ENABLE_CREPE = True
DEFAULT_CREPE_MODEL = "full"  # "full" (more accurate) or "tiny" (faster)
CREPE_AGREEMENT_SEMITONES = 1.0
CREPE_AGREEMENT_CONFIDENCE_BOOST = 1.5
CREPE_DISAGREEMENT_CONFIDENCE_SCALE = 0.3

# RMVPE (rmvpe_onnx) is an optional THIRD independent pitch estimate,
# purpose-built for vocal pitch in POLYPHONIC music (i.e. explicitly
# trained to be robust to the exact kind of residual accompaniment bleed
# a Demucs-separated vocal stem can still have) -- unlike pYIN (general
# DSP-based) or CREPE (general-purpose learned model, not vocal-specific).
# Cross-checked against the current primary pitch the same way CREPE is
# (see CREPE_* above and _cross_check's docstring): agreement within
# RMVPE_AGREEMENT_SEMITONES trusts RMVPE's pitch with a confidence boost;
# disagreement keeps the current pitch but downweights it. Never marks a
# frame unvoiced, for the same reason as CREPE. PROTOTYPE (2026-08-07,
# not yet validated at full-pipeline scale) -- off by default until a
# real-audio validation run confirms it's a net improvement; see
# `pitch_primary` in note_detection.detect_notes() for swapping RMVPE in
# as the PRIMARY source (pYIN becomes the cross-check) instead of a third
# member alongside pYIN-primary.
ENABLE_RMVPE = False
RMVPE_DEVICE = "cpu"  # onnxruntime CUDA build requires CUDA 13 + cuDNN 9,
                       # incompatible with this project's torch cu126
                       # stack; DirectML crashed/hung with garbled UTF-16
                       # logging on this machine. CPU inference measured
                       # at ~5s for a full ~3min song -- fast enough that
                       # GPU acceleration isn't worth chasing further
                       # unless this graduates from prototype.
RMVPE_AGREEMENT_SEMITONES = 1.0
RMVPE_AGREEMENT_CONFIDENCE_BOOST = 1.5
RMVPE_DISAGREEMENT_CONFIDENCE_SCALE = 0.3

# A frame this many dB quieter than the track's own "loud" reference level
# (the 90th percentile of RMS energy, not the absolute peak, so one loud
# transient doesn't skew the reference) is treated as silence/noise,
# REGARDLESS of what pYIN's own pitch/voicing decision says. This matters
# because pYIN detects periodicity, not loudness -- near-silent audio can
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

# Consensus-based pitch override (final pass-1 stage, after all other
# segmentation/merge stages -- pitch only, never touches note timing).
# Diagnosed on 4 validated songs (batb, stars, sleeping_beauty, gaston):
# an ISOLATED per-note diagnostic (bypassing the real shipped pipeline --
# comparing each of {pyin, rmvpe, swiftf0, penn}'s own note-level vote
# directly to ground truth) found that unanimous agreement among
# >= CONSENSUS_MIN_AGREEING_SOURCES of them is a reliable signal (61-70%
# correct) while a vote among DISAGREEING sources is not (34-46%, worse
# than trusting a single source) -- so this deliberately never resolves
# disagreement via a vote, only overrides on unanimous agreement, leaving
# the pipeline's own decision alone otherwise.
#
# REAL end-to-end validation (actual shipped pipeline -- pyin primary +
# CREPE cross-check -- with this override applied on top, real audio, not
# the isolated diagnostic above) told a different, MIXED story: batb -0.7,
# stars -3.3 (a real regression on the song where the baseline pipeline
# was already BEST, 61.5%), sleeping_beauty +2.8, gaston +2.8 -- net
# average only +0.4pp, not the clean win the isolated diagnostic
# suggested. Likely explanation: the override only fires where an
# isolated-source vote disagrees with the CURRENT pipeline's answer, and
# on a song where the base ensemble is already resolving things well
# (stars), second-guessing it with a simpler vote (that doesn't have
# access to the base pipeline's own smoothing/merge context) does more
# harm than good; the benefit is concentrated on songs where the baseline
# is already weak. Also adds real compute cost (+15-30s/song, loading
# RMVPE/SwiftF0/PENN fresh every call). **OFF by default** -- same
# category as CONFIDENCE_FLOOR_PERCENTILE/REARTICULATION_RECONCILE_ENABLED
# above (mechanistically sound, doesn't generalize on real end-to-end
# validation). See note_detection.py's _consensus_pitch_override and
# project memory for the full validation writeup.
CONSENSUS_OVERRIDE_ENABLED = False
CONSENSUS_MIN_AGREEING_SOURCES = 2
CONSENSUS_SOURCES = ("pyin", "rmvpe", "swiftf0", "penn")

# Musical key-snapping (pass 2, inspired by the pitch-correction idea in
# the ultrastar_pitch project): detect the song's most likely key from
# pass 1's raw pitch-class distribution and nudge out-of-scale notes to
# the nearest in-key neighbor. OFF by default (again -- was briefly ON):
# root-caused as actively harmful on real material, confirmed via a
# direct pass-1-vs-pass-2 diff on a real song ("Stars", real key
# confirmed E major by the user) -- dozens of notes changed throughout
# the whole song, including blanket-snapping every legitimate C-natural
# (a deliberate modal-mixture/borrowed note, not noise) to B just because
# C isn't diatonic in E major. A single global detected key, applied
# blindly to every note, is the wrong model for harmonically
# sophisticated material -- real songs legitimately use out-of-scale
# notes on purpose. Available via --key-correction for anyone who wants
# it anyway; see [PASS2 DEBUG] to see exactly what it would change.
ENABLE_KEY_CORRECTION = False

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
# Placement verification+correction (verify_placement) is OFF by default
# (--verify-placement to enable). It now auto-corrects a word's (start,
# end) and re-runs pass 3 when it gets a PRECISE forced-alignment fix
# (the exact bug this targets: a word's ASR timestamp was badly wrong at
# a zone boundary with no acoustic pause nearby to anchor a fix on --
# "sword"/"Stars" in Les Misérables - Stars, see CLAUDE.md's open
# threads). Kept OFF by default purely for COST, not reliability: it's an
# expand-search re-transcription loop over every word (when
# VERIFY_ALL_WORDS is True), which is expensive next to the rest of the
# pipeline. Only ever corrects on a POSITIVELY CONFIRMED, precisely
# forced-aligned position -- never guesses (the two heuristic
# auto-correction ideas that WERE guesses -- snapping to the nearest note
# gap, rebalancing by syllable-count deficit/surplus -- were tried and
# rejected; see CLAUDE.md).
ENABLE_PLACEMENT_VERIFICATION = False
# Verify every word, not just the ones pass 3 flagged suspicious -- the
# extra recheck calls are cheap next to Demucs/WhisperX, and this catches
# cases pass 3's own heuristics can't see (e.g. lyrics_lookup.py's
# "uneven block" case, where a word gets a reference line tagged but its
# text is deliberately left uncorrected pending a more confident check).
# --verify-suspicious-only restricts back to just the flagged words.
VERIFY_ALL_WORDS = True
# verification.verify_placement checks whether a word's FINAL note-assigned
# position is actually where it's sung: crop a small window around that
# position, transcribe it (open vocabulary, via whisperx), and check
# whether the expected word is in the result. If not, the window expands
# and retries -- this distinguishes "the word IS in the audio, just not
# where pass 3 put it" (a real placement bug) from "nothing findable
# nearby at all". An earlier version tried forced-alignment against a wide
# window instead of this expand-and-search approach; that turned out
# unreliable in practice (whisperx.align() anchors near the window's start
# rather than truly searching it, so it produced confident-looking but
# wrong answers when the true position wasn't near the window start) --
# replaced with this approach instead.
#
# Initial search radius (seconds) on each side of the assigned position.
PLACEMENT_SEARCH_INITIAL_RADIUS_SEC = 1.0
# Radius multiplier applied each time the word isn't found.
PLACEMENT_SEARCH_GROWTH_FACTOR = 2.0
# Hard cap on search radius -- bounds cost, and a word that genuinely
# isn't findable even this far out is its own kind of finding (reported
# as "not found", not silently expanded forever).
PLACEMENT_SEARCH_MAX_RADIUS_SEC = 10.0
# Once the word IS found, its exact position is refined with a
# forced-alignment pass over that SAME confirmed window (well-conditioned
# now, since the text is already known to genuinely be in that window --
# unlike the abandoned approach above). If that refined position still
# differs from the assigned position by more than this many seconds, it's
# flagged as a mismatch. Not zero -- alignment has its own small jitter,
# and this shouldn't cry wolf over sub-second imprecision, only real
# multi-beat placement errors.
PLACEMENT_MISMATCH_TOLERANCE_SEC = 1.0



@dataclass
class PipelineOptions:
    artist: str = None
    title: str = None
    output_dir: str = None
    whisper_model: str = DEFAULT_WHISPER_MODEL
    demucs_model: str = DEFAULT_DEMUCS_MODEL
    bpm_override: float = None
    skip_separation: bool = False
    vocals_path: str = None
    fetch_lyrics: bool = True
    genius_token: str = None
    device: str = "cuda"  # only "cuda" is supported
    keep_intermediate: bool = True
    work_dir: str = None
