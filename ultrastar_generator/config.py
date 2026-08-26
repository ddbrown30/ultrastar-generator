"""Defaults and constants shared across the pipeline."""

from dataclasses import dataclass
from typing import Callable, Optional

# --- UltraStar note types -------------------------------------------------
NOTE_NORMAL = ":"
NOTE_GOLDEN = "*"
NOTE_FREESTYLE = "F"
NOTE_RAP = "R"
NOTE_RAP_GOLDEN = "G"

# --- Audio / file conventions ---------------------------------------------
AUDIO_EXTS = (".mp3", ".ogg", ".oga", ".m4a")
VIDEO_EXTS = (".avi", ".mp4", ".mpg", ".mpeg")
# .mp4/.mpg/.mpeg can serve as #MP3 directly; .avi cannot and needs its
# audio extracted into a real standalone file first (media_extract.py).
VIDEO_DIRECT_AUDIO_EXTS = (".mp4", ".mpg", ".mpeg")
IMAGE_EXTS = (".jpg", ".jpeg")

# --- Folder-based input resolution (file_discovery.py, media_extract.py,
# cover_extract.py, song_input.py) -------------------------------------------
AVI_EXTRACTED_MP3_QUALITY = 2  # ffmpeg libmp3lame VBR scale, 0=best .. 9=worst

# Cover-art fallback tag suffix -- matches file_discovery's "[CO]"/"[BG]"
# companion-file convention, so a fetched/extracted cover is reused on rerun.
COVER_TAG_SUFFIX = " [CO]"

# --- Pipeline defaults ------------------------------------------------------
DEFAULT_LANGUAGE = "English"

FALLBACK_BPM = 100.0  # used only if tempo detection fails outright

# Sane BPM range for #BPM; detected tempo outside this is folded by x2//2.
MIN_BPM = 60.0
MAX_BPM = 200.0

# Written #BPM is this multiple of the real detected tempo, for finer
# beat-grid resolution (write-time only; pass 1's own segmentation uses the real tempo).
BPM_WRITE_MULTIPLIER = 2

# faster-whisper model; "medium.en" balances accuracy/speed on GPU.
DEFAULT_WHISPER_MODEL = "medium.en"

DEFAULT_DEMUCS_MODEL = "htdemucs"

# Phrase-breaking heuristics
MAX_SYLLABLES_PER_LINE = 8
MIN_LINE_GAP_SEC = 0.35  # silence gap that *forces* a new line
PREFERRED_LINE_GAP_SEC = 0.15  # silence gap that's a good place to break

GOLDEN_NOTE_MIN_DURATION_SEC = 0.6  # notes held at least this long are marked golden ('*')

# --- Note detection (pass 1: pitch/timing from audio, no lyrics involved) --
MIN_NOTE_DURATION_SEC = 0.06  # shorter notes are dropped/merged as glitches
# Leading seconds to drop before computing a note's pitch, to exclude a
# rising attack/portamento from biasing the average. 0.0 = off.
ATTACK_TRIM_SEC = 0.0
# Drop the bottom N% of a note's own frames by confidence before computing
# its pitch, instead of a fixed leading window. 0 = off.
CONFIDENCE_FLOOR_PERCENTILE = 0.0

# Ambiguity-gated, Krumhansl-Kessler key-profile pitch-CLASS refinement --
# RMVPE only (SwiftF0 has no comparable multi-bin salience output).
ENABLE_AMBIGUITY_KEY_TIEBREAK = True
# Relative salience-mass margin between the top-2 candidate pitch classes
# below which a note is ambiguous enough for the key tie-break.
AMBIGUITY_MARGIN_THRESHOLD = 0.35

# Reconciles a re-articulation split's rounded pitch with its predecessor
# when they're near-contiguous and exactly 1 semitone apart. Off by default.
REARTICULATION_RECONCILE_ENABLED = False
REARTICULATION_RECONCILE_MAX_GAP_SEC = 0.02  # max gap (sec) between fragments to reconcile
# Pitch change (semitones), on the smoothed contour with no silence gap,
# treated as a new note rather than vibrato.
NOTE_SPLIT_SEMITONES = 1.0
# Same-pitch re-articulation (e.g. two syllables on one held note) needs a
# strong onset (top percentile of onset strength) to split, and only once
# the in-progress note has run at least this long.
REARTICULATION_STRENGTH_PERCENTILE = 60.0
MIN_DURATION_BEFORE_REARTICULATION_SEC = 0.08
# A too-short protected re-articulation segment stretched below this floor
# is dropped instead of kept as a sliver.
MIN_PLAUSIBLE_REARTICULATION_DURATION_SEC = 0.03
# Half-width (sec) of the window around a strong onset that also protects
# a silence-based split from being merged away by the next pass.
REARTICULATION_ONSET_WINDOW_SEC = 0.02
# Median-filter window (sec) on the pitch contour, to suppress vibrato
# without blurring real note changes.
PITCH_SMOOTH_WINDOW_SEC = 0.11
# Adjacent notes this close in pitch/time get merged (fixes vibrato/noise
# fragmenting one syllable). Also capped as a total range across the whole
# merged chain, not just per-step -- see _merge_similar_adjacent.
NOTE_MERGE_SEMITONES = 1
NOTE_MERGE_MAX_GAP_SEC = 0.05
# Notes shorter than this fraction of one beat get folded into whichever neighbor has the closer pitch.
MIN_NOTE_BEATS_FRACTION = 0.5
MELISMA_CONTINUATION_TEXT = "~"  # text for extra notes when a melisma outnumbers its syllables
MIN_NOTE_GAP_SEC = 0.01  # minimum gap (sec) enforced between notes when resolving overlaps

# main.py-only melisma cleanup, post-quantization (usdx_writer.py): merges
# a beat-adjacent same-pitch "~" note into its predecessor, then drops any
# "~" still 1 beat long. Not applied in realign.py, which never adds/removes notes.

# Pass 1's pitch source: "rmvpe" or "swiftf0" -- supplies both pitch and
# voicing exclusively, no cross-check/ensemble.
DEFAULT_PITCH_SOURCE = "rmvpe"

# CPU: onnxruntime's CUDA build conflicts with this project's torch stack; CPU inference is fast enough.
RMVPE_DEVICE = "cpu"

# A frame this many dB below the track's 90th-percentile RMS level is
# treated as silence regardless of the pitch source's own voicing decision
# (periodicity != loudness).
SILENCE_REFERENCE_PERCENTILE = 90
SILENCE_THRESHOLD_DB_BELOW_PEAK = 40.0
# Absolute dBFS floor for silence, for when a whole clip is uniformly quiet (no louder reference to compare against).
SILENCE_ABSOLUTE_FLOOR_DB = -50.0

# Isolated pitch-tracking-glitch filter: a short note that jumps far from
# both close-in-pitch, close-in-time neighbors is removed and folded into the previous note.
SPIKE_MAX_DURATION_SEC = 0.25
SPIKE_MIN_JUMP_SEMITONES = 4.0
SPIKE_NEIGHBOR_SIMILARITY_SEMITONES = 2.0
SPIKE_MAX_NEIGHBOR_GAP_SEC = 0.15

# Trailing breath/release-artifact filter: a short, low-confidence note
# right after a long high-confidence note is absorbed into it. Targets
# breath/consonant tails (pitch-close but low-confidence), unlike SPIKE_* above.
TRAILING_ARTIFACT_MAX_DURATION_SEC = 0.12
TRAILING_ARTIFACT_CONFIDENCE_RATIO = 0.6  # candidate confidence must be <= this fraction of the preceding note's
TRAILING_ARTIFACT_MAX_GAP_SEC = 0.2  # covers a real breath gap
TRAILING_ARTIFACT_MIN_PRECEDING_DURATION_SEC = 0.5  # preceding note must itself be genuinely sustained

# "whisperx" forced alignment gives more accurate word boundaries than
# Whisper's own decoder timestamps; falls back to faster-whisper if unavailable.
PREFER_WHISPERX = True

# whisperx's own VAD has no true off switch; near-zero onset/offset
# thresholds avoid it misjudging a sustained vowel and corrupting alignment context.
WHISPERX_NO_VAD_OPTIONS = {"vad_onset": 0.01, "vad_offset": 0.01}
ENABLE_WHISPERX_NO_VAD = True  # on by default; --whisperx-vad opts back into whisperx's own VAD

# --- Pass 3 note assignment (lyric_alignment.py) ---------------------------
# Max ASR gap (sec) between words still treated as one phrase for note
# assignment -- separate from MIN_LINE_GAP_SEC (display line breaks).
NOTE_ASSIGNMENT_MAX_GAP_SEC = 0.35

# lyrics_lookup.align_words_to_reference: in a clamped repeated-token
# replace block, a word this far from the previous one is too far to be
# the same repeat run and is left unmatched instead.
REFERENCE_CLAMP_MAX_GAP_SEC = 2.0
# This many ASR words clamped onto one reference token signals a decoder
# hallucination; past this, fall back to the ASR word's own raw text.
REFERENCE_CLAMP_MAX_REPEAT = 8
# A delete-block (ASR word with no reference counterpart) up to this long
# is dropped as a real hallucination; a longer run is more likely a
# matching failure than real non-lyrical content, so it's kept instead.
REFERENCE_DELETE_MAX_RUN = 5

# When synced LRC lyrics are available, an unmatched word's line is
# assigned by nearest calibrated LRC line timestamp instead of inheriting
# the previous matched word's line_id (avoids spurious/suppressed line breaks).

# A word's leading piece shorter than this after note-boundary splitting
# is dropped rather than kept as its own syllable (usually an unvoiced
# consonant bleeding in from the previous word's ASR timestamp).
# Deliberately more generous than MIN_NOTE_DURATION_SEC (pass 1's own glitch floor).
SLIVER_DROP_MAX_DURATION_SEC = 0.12

# --- Reference-text override (verification.py) -----------------------------
# apply_reference_text replaces a reference-tagged word's ASR text with the trusted reference text.

# Pass 4 (optional): corrects pass-3 syllable PITCH CLASS (never octave or
# timing) against a user-supplied MusicXML file.
MUSICXML_MIN_CALIBRATION_SAMPLES = 8  # minimum matched notes before trusting a per-song calibration offset
# Modal pitch-class offset must cover at least this fraction of matches to be trusted.
MUSICXML_MIN_CALIBRATION_CONFIDENCE = 0.5
# Retry threshold using only the top half of matches by our own note
# confidence (lower bar since that population is already noise-reduced).
MUSICXML_MIN_CALIBRATION_CONFIDENCE_HIGH_CONF_SUBSET = 0.4
# Confidence assigned to a pass-4-corrected syllable -- boosted, but not
# 1.0 (inferred from a different source, not a direct pass-1 measurement).
MUSICXML_CORRECTED_CONFIDENCE = 0.75

# Apply the best available calibration offset even below the confidence
# bars above, rather than skipping the file. On by default.
ENABLE_MUSICXML_FORCE_CALIBRATION = True


# --- LRC (LRCLIB synced-lyrics) line timing check (lrc_timing.py) ----------
# Diagnostic only -- flags lines, never corrects them.
ENABLE_LRC_TIMING_CHECK = False
LRC_TIMING_MIN_CALIBRATION_SAMPLES = 5  # minimum matched lines before trusting a per-song offset
# Modal per-line delta (1s buckets) must cover at least this fraction of lines to be trusted.
LRC_TIMING_MIN_CALIBRATION_CONFIDENCE = 0.4
LRC_TIMING_FLAG_TOLERANCE_SEC = 2.0  # a line's post-calibration residual beyond this is flagged

# Tier 2 fallback: robust (Theil-Sen) linear-drift fit, tried when the
# constant-offset check fails. Stricter bars than tier 1, since a
# 2-parameter fit can trivially match a handful of points.
LRC_TIMING_MIN_DRIFT_SAMPLES = 10
LRC_TIMING_MIN_DRIFT_CONFIDENCE = 0.5
LRC_TIMING_DRIFT_INLIER_TOLERANCE_SEC = 1.5  # residual within this counts as an inlier to the fitted line

# Tier 3 (tried only if tiers 1 and 2 both fail): handles discontinuous
# drift, e.g. a candidate edited differently from our recording. Two
# interchangeable models -- "isotonic" (PAVA monotonic regression) or
# "piecewise" (linear interpolation between confident anchors).
LRC_TIMING_DRIFT_MODEL = "isotonic"  # "isotonic" | "piecewise"
# Tier 3's own noise filter: max distance (sec) from tier 2's own fit to
# be trusted as real signal -- looser than the tier-2 inlier tolerance.
LRC_TIMING_PIECEWISE_OUTLIER_TOLERANCE_SEC = 4.0
LRC_TIMING_PIECEWISE_MIN_ANCHORS = 6  # minimum surviving anchors before tier 3 is attempted
# Reject tier 3 if any two adjacent surviving anchors are farther apart
# than this -- too little evidence to interpolate across that gap.
LRC_TIMING_PIECEWISE_MAX_ANCHOR_GAP_SEC = 45.0

# Tier 3 splits into "refine" (tiers 1/2 already partially agree -- runs
# unconditionally) vs. "rescue" (neither found support -- gated behind an
# optional structural_check, since a flexible fit can look confident even
# on a wrong/different recording).
LRC_TIMING_RESCUE_MIN_PRIOR_CONFIDENCE = 0.30
# Odd/even-anchor holdout validation for tier 3 fits; diagnostic only, not currently a hard gate.
LRC_TIMING_HOLDOUT_MIN_ANCHORS = 4


# --- Existing-file verification (verify_existing_song.py) ------------------
# Compares an existing .txt's pitch/timing against a fresh run of the same
# audio. Not wired into any pipeline's own default flow.
EXISTING_TXT_MIN_MATCHED = 10
# NOT empirically validated -- picked by analogy, not measured against real run-to-run self-noise.
EXISTING_TXT_MIN_PITCH_ACCURACY = 0.85
EXISTING_TXT_TIMING_TOLERANCE_SEC = 0.5
EXISTING_TXT_MIN_TIMING_AGREEMENT = 0.85

# Minimum fraction of each side's own words that must text-match the other
# side, or the comparison isn't trusted (unmatched words never enter the
# pitch/timing stats, so low coverage can hide behind a perfect-looking accuracy).
EXISTING_TXT_MIN_COVERAGE = 0.85


# --- Reference lyrics (lyrics_lookup.py) ------------------------------------
# LRCLIB is the only reference-lyrics source; a candidate must have synced
# (per-line-timestamped) lyrics to count at all.
LRCLIB_DURATION_TOLERANCE_SEC = 60.0  # duration mismatch beyond this is penalized, not excluded

# Fetched reference's vocabulary overlap with the ASR transcript must
# clear this ratio, or it's rejected as a wrong-song/wrong-language reference.
REFERENCE_LYRICS_MIN_MATCH_RATIO = 0.25


# --- MXL+LRC primary generation (mxl_lrc_generator.py) ----------------------
# Default generation path for songs with both a MusicXML score and synced
# LRC lyrics: MXL for pitch, LRC line starts as time anchors, ASR to place
# words within a line -- skips audio-only pass 1-4 entirely.
ENABLE_MXL_LRC_PRIMARY = True

# Candidate selection stays permissive -- the real gate is MXL_LRC_MIN_ASR_PLACEMENT_RATE below.
MXL_LRC_MIN_CONTENT_MATCH_RATIO = 0.3

# select_lrc_candidate's ranking, once artist-match is tied, only lets duration proximity
# override content-match ratio for candidates within the SAME coarse ratio bucket -- a large
# ratio gap always wins outright, duration is never a hard filter at all (real bug, Beauty and
# the Beast, 2026-08-25): our own artist tag is a show/movie TITLE, which is a substring of
# nearly every cast recording's artist string, so `_artist_matches` alone can't decisively
# separate the real matching arrangement from a structurally different one (e.g. a
# "Finale"/reprise cut with a shortened middle section); and our own SingStar-ripped audio can
# legitimately run 30+ seconds shorter than an OST candidate's own full-length duration while
# still being the correct, best-content-matching candidate -- a hard duration cutoff excluded
# it from scoring entirely, before ratio ever got a say.
MXL_LRC_CANDIDATE_RATIO_TIE_BUCKET = 0.25

# Decisive quality gate: does our own audio's ASR transcript agree with
# the matched LRC candidate's line timings?
MXL_LRC_MIN_ASR_PLACEMENT_RATE = 0.5
MXL_LRC_MAX_NONMONOTONIC_RATE = 0.1

# Below this confidence, a text match's timestamp isn't trusted; falls
# through to the MXL-tempo-estimated placement instead.
MXL_LRC_MIN_ASR_WORD_CONFIDENCE = 0.3

# A 1:1-matched MXL/LRC word pair is trusted as an OCR/spelling variant of
# the same word above this character-level similarity ratio.
MXL_LRC_FUZZY_TEXT_MIN_RATIO = 0.6

# Largest word-block size attempted for whole-block fuzzy matching (OCR
# merge/split cases), bounded on both sides by real matches either way. Real case that needed
# raising this from 6 to 10 (2026-08-25, Weird Al Yankovic - "Nature Trail to Hell"): the MXL
# notated "homicidal maniac" as 7 separate single-syllable "words" (ho/mi/ci/dal/man/i/ac) --
# still gated by MXL_LRC_FUZZY_TEXT_MIN_RATIO afterward (their concatenation is an exact-ratio
# match for the real word), so raising the SIZE cap doesn't loosen the actual acceptance bar.
MXL_LRC_BLOCK_MAX_WORDS = 10

# Last-resort real-seconds-per-quarter-note rate when no local tempo anchor is available at all.
MXL_LRC_DEFAULT_QUARTER_NOTE_SEC = 0.3

# The first LRC line sets #GAP for the whole file, so an error there has a
# much larger blast radius than elsewhere. If a direct real-ASR anchor for
# line 0 disagrees with the calibrated value by more than this many
# seconds, the direct anchor wins; every other line uses normal calibration.
GAP_ANCHOR_OVERRIDE_TOLERANCE_SEC = 0.5


# --- realign.py "validate" strategy ----------------------------------------
# A word's GAP/drift-corrected original start is trusted exactly if a
# confident whole-song ASR match lands within this many seconds of it.
# Only start proximity is checked (ASR is unreliable at a held note's end).
REALIGN_VALIDATE_TOLERANCE_SEC = 0.3

# --- transcription.py long-segment re-windowing ----------------------------
# A decoder segment can silently drop real content while still declaring a
# long span, forcing wav2vec2 to align the short remaining text into an
# oversized window and misplace it. Fix: for segments at least this long,
# sweep fixed-width candidate windows and keep whichever wins by mean word
# score, if it beats the baseline by the margin below. Unconditional.
REWINDOW_MIN_SEGMENT_DURATION_SEC = 10.0
REWINDOW_CANDIDATE_WIDTH_SEC = 10.0
REWINDOW_STEP_SEC = 1.0
REWINDOW_MIN_SCORE_IMPROVEMENT = 0.10


# --- ASR quality retry -------------------------------------------------------
# WhisperX can silently drop/garble whole passages on a given run. When a
# match-rate metric a code path already computes looks bad, retry once
# with RETRY_ASR_MODEL and keep the retry only if it scores better.
# On by default; --no-retry-low-quality-asr opts out.
RETRY_ASR_MODEL = "large-v3"
RETRY_LOW_QUALITY_ASR = True

# Standard (non-MXL) path fallback signal: ASR transcript vs. fetched
# reference-lyrics vocabulary overlap. Higher bar than
# REFERENCE_LYRICS_MIN_MATCH_RATIO (that asks "right song?"; this asks "transcribed well?").
RETRY_ASR_MIN_REFERENCE_MATCH_RATIO = 0.6

# Per-passage trigger: catches a localized failure (a dropped passage or
# badly-mistimed run) invisible to a whole-song aggregate metric.
RETRY_ASR_MIN_UNMATCHED_REFERENCE_RUN = 5
RETRY_ASR_MIN_UNCONFIDENT_RUN = 5

# --- Force-align known-text gaps (unconditional, no CLI/GUI off-switch) -----
# Recovers a passage the ASR decoder dropped entirely via wav2vec2 CTC
# forced alignment of the known missing text, restricted to the window
# between the nearest measured neighbors. See transcription.force_align_words_in_window.
FORCE_ALIGN_MIN_WINDOW_BASE_SEC = 0.10
FORCE_ALIGN_MIN_WINDOW_SEC_PER_WORD = 0.08
FORCE_ALIGN_WINDOW_SLOP_SEC = 0.5


@dataclass
class PipelineOptions:
    """Knobs for run_pipeline/run_batch, decoupled from argparse so the GUI
    and CLI build the same options and call the same pipeline code. Keep
    field defaults in sync with build_arg_parser's own defaults."""
    artist: Optional[str] = None
    title: Optional[str] = None
    audio_file: Optional[str] = None  # disambiguates a folder with >1 real audio file
    work_dir: Optional[str] = None
    whisper_model: str = DEFAULT_WHISPER_MODEL
    demucs_model: str = DEFAULT_DEMUCS_MODEL
    bpm_override: Optional[float] = None
    skip_separation: bool = False
    vocals_path: Optional[str] = None
    fetch_lyrics: bool = True
    fetch_cover: bool = True
    no_video_sync: bool = False
    no_whisperx: bool = False
    no_transcribe: bool = False  # diagnostic: skip the WhisperX decoder, force-align a pinned LRC candidate's known text instead
    whisperx_no_vad: bool = ENABLE_WHISPERX_NO_VAD
    retry_low_quality_asr: bool = RETRY_LOW_QUALITY_ASR
    musicxml_reference: Optional[str] = None
    musicxml_part: Optional[str] = None
    lrc_timing_check: bool = ENABLE_LRC_TIMING_CHECK
    pitch_smooth_window: float = PITCH_SMOOTH_WINDOW_SEC
    note_split_semitones: float = NOTE_SPLIT_SEMITONES
    min_note_beat_fraction: float = MIN_NOTE_BEATS_FRACTION
    silence_threshold_db: float = SILENCE_THRESHOLD_DB_BELOW_PEAK
    silence_floor_db: float = SILENCE_ABSOLUTE_FLOOR_DB
    spike_max_duration: float = SPIKE_MAX_DURATION_SEC
    spike_jump_semitones: float = SPIKE_MIN_JUMP_SEMITONES
    ambiguity_key_tiebreak: bool = ENABLE_AMBIGUITY_KEY_TIEBREAK
    ambiguity_margin_threshold: float = AMBIGUITY_MARGIN_THRESHOLD
    pitch_source: str = DEFAULT_PITCH_SOURCE
    no_pass1_debug: bool = False
    no_debug_log: bool = False
    quiet: bool = False
    youtube_url: Optional[str] = None
    youtube_audio_only: bool = True
    batch: bool = False
    delete_work_files: bool = False
    # GUI only, never set by the CLI. Manual pre-run pick that always wins,
    # skipping the network fetch entirely. Forward-referenced as a string
    # to avoid a circular import with lyrics_lookup.py.
    pinned_lyrics: Optional["LrcLibCandidate"] = None
    # MXL+LRC primary generation path. lrclib_id, if set, always wins over
    # search and pinned_lyrics for candidate selection everywhere.
    mxl_lrc_primary: bool = ENABLE_MXL_LRC_PRIMARY
    lrclib_id: Optional[int] = None
    # GUI only. Called with the failure reason when the MXL+LRC quality
    # gate fails; True continues with the standard audio fallback, False cancels the run.
    mxl_lrc_fallback_callback: Optional[Callable[[str], bool]] = None
    # GUI only. Called when no valid synced-lyrics LRCLIB candidate is
    # found; True continues with pure ASR, False cancels the run. Not
    # consulted when pinned_lyrics/lrclib_id already resolved a candidate, or when fetch_lyrics is off.
    no_lrc_fallback_callback: Optional[Callable[[str], bool]] = None
    # GUI only (CLI uses Ctrl+C). Polled via check_cancelled() at stage boundaries; True raises PipelineCancelled.
    cancel_requested: Optional[Callable[[], bool]] = None


class PipelineCancelled(Exception):
    """Raised when opts.cancel_requested() returns True, caught at each
    run_*_pipeline's own top level as a normal success=False result."""


def check_cancelled(cancel_requested: Optional[Callable[[], bool]]) -> None:
    if cancel_requested is not None and cancel_requested():
        raise PipelineCancelled()
