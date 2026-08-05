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
# a good accuracy/speed tradeoff on CPU; use "medium.en" or "large-v3" for
# better lyric accuracy if you have a GPU.
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
# A pitch change of at least this many semitones (on the SMOOTHED contour,
# without a silence gap) is treated as a new note rather than vibrato.
NOTE_SPLIT_SEMITONES = 1.0
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

# Musical key-snapping (inspired by the pitch-correction idea in the
# ultrastar_pitch project): after final notes are assembled, optionally
# detect the song's most likely key from the pitch-class distribution and
# nudge clearly off-key notes to the nearest in-key neighbor. OFF by
# default: real feedback showed it (combined with over-aggressive note
# merging) flattening genuine chromatic/passing-tone melodic movement
# toward nearby in-key notes. Available via --key-correction once the
# more impactful segmentation/merge fixes have been validated against
# real audio.
ENABLE_KEY_CORRECTION = False

# Word-level timestamp source. "whisperx" uses forced alignment (wav2vec2
# CTC) for much more accurate word boundaries than Whisper's own decoder
# timestamps; falls back to faster-whisper automatically if whisperx isn't
# installed.
PREFER_WHISPERX = True



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
    device: str = "cpu"  # "cpu" or "cuda"
    keep_intermediate: bool = True
    work_dir: str = None
