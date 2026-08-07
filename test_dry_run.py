"""Exercises everything that doesn't require ML models / real audio, plus
targeted regression tests for the reported bugs:

  1. Overlapping notes must never occur in the final output.
  2. lyric_alignment fits words onto audio-detected notes, not the other
     way around (so word-timing imprecision can't distort pitch/timing).
  3. Melisma (fewer syllables than notes) and merged syllables (more
     syllables than notes) are both handled without creating overlaps.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultrastar_generator.file_discovery import find_companions, parse_artist_title
from ultrastar_generator.tempo import beat_duration_ms, seconds_to_beat, seconds_to_beat_length
from ultrastar_generator.syllables import hyphenate
from ultrastar_generator.models import Syllable, LineBreak, Song, Word
from ultrastar_generator.phrasing import build_lines
from ultrastar_generator.usdx_writer import render_song
from ultrastar_generator.postprocess import enforce_monotonic
from ultrastar_generator.note_detection import NoteEvent
from ultrastar_generator.lyric_alignment import align_words_to_notes
import numpy as np

print("--- file_discovery ---")
audio = Path("test_sandbox/Bon Jovi - Its My Life.mp3")
artist, title = parse_artist_title(audio)
assert (artist, title) == ("Bon Jovi", "Its My Life"), (artist, title)
comp = find_companions(audio)
assert comp.video and comp.video.name == "Bon Jovi - Its My Life.mp4"
assert comp.cover and "[CO]" in comp.cover.name
assert comp.background and "[BG]" in comp.background.name
print("OK:", artist, title, comp)

print("\n--- tempo math vs. real reference files ---")
assert abs(beat_duration_ms(300) - 50.0) < 1e-6
assert seconds_to_beat(23.0, 23000, 300) == 0
assert seconds_to_beat_length(0.400, 300) == 8
t = 8300 + 200 * beat_duration_ms(120)
assert seconds_to_beat(t / 1000.0, 8300, 120) == 200
assert seconds_to_beat(48.030, 48030, 382.7) == 0
print("OK: beat math matches all 3 spot-checks")

print("\n--- syllables (regex fallback, no pyphen installed here) ---")
for w in ["reciprocity", "mothering", "cowboy", "I'm", "highway,", "always"]:
    parts = hyphenate(w)
    assert "".join(parts) == w, (w, parts)
print("OK")

print("\n--- BUG REGRESSION 1: enforce_monotonic removes overlaps ---")
overlapping = [
    Syllable("A", 0.0, 1.0, 0, True),
    Syllable("B", 0.5, 1.5, 2, True),   # overlaps A by 0.5s
    Syllable("C", 1.2, 1.3, 4, True),   # starts before B ends
    Syllable("D", 5.0, 4.9, 1, True),   # end < start (degenerate)
]
fixed = enforce_monotonic(overlapping)
for i in range(1, len(fixed)):
    assert fixed[i].start >= fixed[i - 1].end, f"OVERLAP: {fixed[i-1]} then {fixed[i]}"
for s in fixed:
    assert s.end > s.start, f"non-positive duration: {s}"
print(f"OK: {len(overlapping)} input notes -> {len(fixed)} non-overlapping notes:")
for s in fixed:
    print(f"    {s.start:.3f}-{s.end:.3f}  {s.text!r}")

print("\n--- BUG REGRESSION 2/3: lyric_alignment fits words onto NOTE timing, not word timing ---")
# Simulate: ASR thinks "hello" spans 0.0-1.0s with sloppy timing, but the
# audio-only note detector found the *real* two notes at 0.05-0.45 and
# 0.50-0.95 (typical case: ASR word boundary is a rough guess; the note
# grid is closer to truth). Final syllable timing must come from the
# notes, not from the word.
words = [Word(text="hello", start=0.0, end=1.0, confidence=0.9)]
notes = [
    NoteEvent(start=0.05, end=0.45, pitch=3),
    NoteEvent(start=0.50, end=0.95, pitch=7),
]
y = np.zeros(16000)  # dummy audio; median_pitch_in_span fallback won't be hit here
syllables, _stats = align_words_to_notes(words, notes, y, 16000)
assert len(syllables) == 2, syllables
assert syllables[0].start == 0.05 and syllables[0].end == 0.45, syllables[0]
assert syllables[1].start == 0.50 and syllables[1].end == 0.95, syllables[1]
assert syllables[0].midi_note == 3 and syllables[1].midi_note == 7
print("OK: syllable timing/pitch came from the note grid, e.g.:")
for s in syllables:
    print(f"    {s.start:.3f}-{s.end:.3f}  pitch={s.midi_note}  {s.text!r}")

print("\n--- Melisma: 1 syllable held across 3 notes gets '~' continuation ---")
words2 = [Word(text="oh", start=0.0, end=1.5, confidence=0.9)]
notes2 = [
    NoteEvent(start=0.0, end=0.4, pitch=0),
    NoteEvent(start=0.4, end=0.8, pitch=2),
    NoteEvent(start=0.8, end=1.5, pitch=4),
]
syls2, _stats2 = align_words_to_notes(words2, notes2, y, 16000)
assert [s.text for s in syls2] == ["oh", "~", "~"], syls2
print("OK:", [(s.text, s.midi_note) for s in syls2])

print("\n--- Merge: word has more syllables than detected notes ---")
words3 = [Word(text="wonderful", start=0.0, end=1.0, confidence=0.9)]
notes3 = [
    NoteEvent(start=0.0, end=0.5, pitch=0),
    NoteEvent(start=0.5, end=1.0, pitch=3),
]
syls3, _stats3 = align_words_to_notes(words3, notes3, y, 16000)
assert len(syls3) == 2, syls3
assert "".join(s.text for s in syls3) == "wonderful"
print("OK:", [s.text for s in syls3])

print("\n--- BUG REGRESSION (round 2): beat-grid quantization can't create duplicate/overlapping beats ---")
# This reproduces the exact failure mode from the "Stars" output: two
# notes non-overlapping in seconds by only a few ms can round to the SAME
# beat once quantized, at a slow-ish beat grid. BPM 105.47 -> beat_ms
# ~= 142ms (matches the real file). Build several near-simultaneous
# syllables and confirm the written beats never collide.
bpm_stars = 105.47
gap_stars_ms = 7767
close_syls = [
    Syllable("There,", 7.767, 8.100, -5, is_word_start=True),
    Syllable("~", 8.110, 8.150, -4, is_word_start=False),   # ~10ms gap, tiny note
    Syllable("~", 8.155, 8.200, -5, is_word_start=False),   # ~5ms gap, tiny note
    Syllable("~", 8.205, 8.260, -4, is_word_start=False),
]
song_stars = Song(title="Stars", artist="Les Miserables", mp3="x.ogg",
                   bpm=bpm_stars, gap_ms=gap_stars_ms, entries=close_syls)
txt_stars = render_song(song_stars)
note_lines = [l for l in txt_stars.splitlines() if l[:1] in (":", "*")]
starts = [int(l.split()[1]) for l in note_lines]
lengths = [int(l.split()[2]) for l in note_lines]
ends = [s + n for s, n in zip(starts, lengths)]
for i in range(1, len(starts)):
    assert starts[i] >= ends[i - 1], f"BEAT COLLISION: note {i-1} ends {ends[i-1]}, note {i} starts {starts[i]}\n{txt_stars}"
print(f"OK: {len(starts)} notes quantized to beats {list(zip(starts, lengths))}, no collisions")

print("\n--- BUG REGRESSION (round 2): vibrato must not fragment a sustained note ---")
# Mock librosa so detect_notes() sees a pitch contour that wobbles +/-1
# semitone at ~6.5Hz (a realistic vibrato rate) for 1.4s, matching the
# "There," example from feedback where the correct output is ONE note.
import sys as _sys, types as _types

sr = 22050
hop = 256
frame_dur = hop / sr
dur_s = 1.4
n_frames = int(dur_s / frame_dur)
t = np.arange(n_frames) * frame_dur
vibrato_hz = 6.5
base_midi = 56.0  # G#3
wobble = 0.6 * np.sin(2 * np.pi * vibrato_hz * t)  # +/-0.6 semitone
midi_contour = base_midi + wobble
f0_contour = 440.0 * 2 ** ((midi_contour - 69) / 12)

fake_librosa = _types.ModuleType("librosa")
fake_librosa.pyin = lambda y, fmin, fmax, sr, frame_length, hop_length, fill_na=np.nan: (
    f0_contour, np.ones(n_frames, dtype=bool), np.full(n_frames, 0.95)
)
fake_librosa.times_like = lambda x, sr, hop_length: np.arange(len(x)) * (hop_length / sr)


class _FakeOnset:
    @staticmethod
    def onset_detect(y, sr, hop_length, backtrack, units):
        return np.array([0.0])  # only the very first frame is an onset

    @staticmethod
    def onset_strength(y, sr, hop_length):
        # Flat/zero everywhere -- irrelevant to these tests (they're not
        # about same-pitch re-articulation splitting), but note_detection
        # always calls this now, so every fake needs to answer it.
        return np.zeros(max(1, len(y) // hop_length + 1))


fake_librosa.onset = _FakeOnset()


class _FakeFeature:
    @staticmethod
    def rms(y, frame_length, hop_length):
        # This test is specifically about vibrato/segmentation behavior,
        # not the energy gate (that has its own dedicated test below) --
        # report constant "loud" energy for every frame so the energy
        # gate never rejects anything here.
        return np.full((1, n_frames), 1.0).reshape(1, -1)


fake_librosa.feature = _FakeFeature()
_sys.modules["librosa"] = fake_librosa

# Re-import fresh so it picks up the fake librosa inside the function body
import importlib
import ultrastar_generator.note_detection as note_detection_mod
importlib.reload(note_detection_mod)

notes = note_detection_mod.detect_notes(np.zeros(1000), sr, bpm=105.47)
print(f"Detected {len(notes)} note(s) for a {dur_s}s vibrato-wobbling tone:")
for n in notes:
    print(f"    {n.start:.3f}-{n.end:.3f}  pitch={n.pitch}")
assert len(notes) == 1, f"expected vibrato to merge into 1 note, got {len(notes)}: {notes}"
assert notes[0].pitch == base_midi - 60, notes[0].pitch
assert (notes[0].end - notes[0].start) > dur_s * 0.8, "merged note lost too much duration"
print("OK: vibrato collapsed into a single sustained note")

print("\n--- BUG REGRESSION (round 5): silence must not produce hallucinated notes ---")
# Reproduces the reported bug directly: pYIN can report confident,
# real-looking pitch on audio that's actually silent (quantization noise,
# resampling artifacts, a faint hum all have enough incidental
# periodicity to fool a pure pitch/periodicity detector). The energy gate
# must reject these regardless of what pYIN's voicing flag says.
n_frames_silent = 200
fake_librosa_silent = _types.ModuleType("librosa")
# pYIN confidently reports a real-looking pitch (G#3) as "voiced" for the
# ENTIRE clip, exactly like the reported bug.
fake_librosa_silent.pyin = lambda y, fmin, fmax, sr, frame_length, hop_length, fill_na=np.nan: (
    np.full(n_frames_silent, 440.0 * 2 ** ((56 - 69) / 12)),  # G#3
    np.ones(n_frames_silent, dtype=bool),
    np.full(n_frames_silent, 0.95),
)
fake_librosa_silent.times_like = lambda x, sr, hop_length: np.arange(len(x)) * (hop_length / sr)
fake_librosa_silent.onset = _FakeOnset()


class _FakeFeatureSilent:
    @staticmethod
    def rms(y, frame_length, hop_length):
        # The actual audio is silent -- near-zero RMS throughout, same as
        # what the user found when checking vocals.wav directly.
        return np.full((1, n_frames_silent), 1e-9)


fake_librosa_silent.feature = _FakeFeatureSilent()
_sys.modules["librosa"] = fake_librosa_silent
importlib.reload(note_detection_mod)

silent_notes = note_detection_mod.detect_notes(np.zeros(4410), 22050, bpm=105.47, verbose=True)
print(f"Detected {len(silent_notes)} note(s) in genuinely silent audio (pYIN said 100% voiced)")
assert len(silent_notes) == 0, f"energy gate failed to reject hallucinated notes on silence: {silent_notes}"
print("OK: silence correctly produced zero notes despite pYIN reporting confident voicing")

print("\n--- energy gate: real (loud) singing still passes through untouched ---")
# A track with a silent intro (matching the real "Stars" bug report:
# nothing should exist before the singing starts) followed by genuinely
# loud singing -- confirms the gate rejects the silent part but does NOT
# also reject the real content once it starts.
n_silent = 80
n_loud = 80
n_frames_mixed = n_silent + n_loud
fake_librosa_mixed = _types.ModuleType("librosa")
fake_librosa_mixed.pyin = lambda y, fmin, fmax, sr, frame_length, hop_length, fill_na=np.nan: (
    np.full(n_frames_mixed, 440.0 * 2 ** ((56 - 69) / 12)),  # G#3 throughout
    np.ones(n_frames_mixed, dtype=bool),
    np.full(n_frames_mixed, 0.95),
)
fake_librosa_mixed.times_like = lambda x, sr, hop_length: np.arange(len(x)) * (hop_length / sr)
fake_librosa_mixed.onset = _FakeOnset()


class _FakeFeatureMixed:
    @staticmethod
    def rms(y, frame_length, hop_length):
        return np.concatenate([np.full(n_silent, 1e-9), np.full(n_loud, 0.2)]).reshape(1, -1)


fake_librosa_mixed.feature = _FakeFeatureMixed()
_sys.modules["librosa"] = fake_librosa_mixed
importlib.reload(note_detection_mod)

mixed_notes = note_detection_mod.detect_notes(np.zeros(4410), 22050, bpm=105.47, verbose=False)
assert len(mixed_notes) == 1, f"expected exactly the loud section as one note, got {mixed_notes}"
silent_duration_sec = n_silent * (256 / 22050)
assert mixed_notes[0].start >= silent_duration_sec - 0.05, \
    f"note started during the silent section: {mixed_notes[0]}"
print(f"OK: silent lead-in correctly produced no note; real singing starting at "
      f"{mixed_notes[0].start:.2f}s (silent section was {silent_duration_sec:.2f}s) was preserved")

print("\n--- key_correction: obvious out-of-key note gets snapped (now pass 2, operates on NoteEvent, no lyrics involved) ---")
from ultrastar_generator.key_correction import snap_to_key
# Heavily-weighted C major scale (each in-key pitch class repeated 3x so
# the key detector isn't confused by a single outlier) plus ONE clear
# outlier at pitch class 6 (F#), which isn't in C major and sits between
# F(5, in-key) and G(7, in-key).
in_key = [0, 2, 4, 5, 7, 9, 11] * 3
c_major_notes = [
    NoteEvent(start=i * 0.5, end=i * 0.5 + 0.4, pitch=pc)
    for i, pc in enumerate(in_key)
]
outlier_idx = len(c_major_notes)
c_major_notes.append(NoteEvent(start=outlier_idx * 0.5, end=outlier_idx * 0.5 + 0.4, pitch=6))
snapped = snap_to_key(c_major_notes)
assert snapped[outlier_idx].pitch in (5, 7), snapped[outlier_idx].pitch
print(f"OK: outlier pitch 6 snapped to {snapped[outlier_idx].pitch}")

print("\n--- BUG REGRESSION (round 3): word order is preserved even with imprecise/fallback timing ---")
# Reproduces the reported "He knows his way in the dark" scrambling:
# pass-1 detected notes that (due to imprecise ASR word spans) could
# best-overlap the wrong word under the old algorithm, and the old
# enforce_monotonic re-sorted by timestamp, which could then reorder the
# words themselves. The fix: note->word assignment now uses a monotonic
# zone partition, and enforce_monotonic no longer sorts -- it trusts the
# given (word/reading) order and only pushes overlaps forward.
words_order = [
    Word(text="He", start=10.50, end=10.70, confidence=0.9),
    Word(text="knows", start=10.70, end=10.95, confidence=0.9),
    Word(text="his", start=10.95, end=11.15, confidence=0.9),
    Word(text="way", start=11.15, end=11.45, confidence=0.9),
    Word(text="in", start=11.45, end=11.60, confidence=0.9),
    Word(text="the", start=11.60, end=11.75, confidence=0.9),
    # ASR mistimed "dark"'s span to start EARLIER than it should
    # (overlapping "the"/"in") -- exactly the kind of imprecision that
    # broke the old max-overlap assignment.
    Word(text="dark", start=11.50, end=13.20, confidence=0.9),
]
notes_order = [
    NoteEvent(start=10.50, end=10.68, pitch=-6),   # He
    NoteEvent(start=10.70, end=10.90, pitch=-8),   # knows
    NoteEvent(start=10.95, end=11.10, pitch=-9),   # his
    NoteEvent(start=11.15, end=11.40, pitch=-11),  # way
    NoteEvent(start=11.45, end=11.58, pitch=-8),   # in
    NoteEvent(start=11.60, end=11.73, pitch=-9),   # the
    NoteEvent(start=11.75, end=11.90, pitch=-9),   # dark
]
dummy_y2 = np.zeros(16000)
result, _stats_order = align_words_to_notes(words_order, notes_order, dummy_y2, 16000)
# Check reading order of actual words (ignore melisma "~" filler notes,
# which are an expected side effect of this test's deliberately-bad ASR
# timing for "dark", not a reordering bug).
result_words = [s.text.strip() for s in result if s.text.strip() != "~"]
assert result_words == ["He", "knows", "his", "way", "in", "the", "dark"], result_words
for i in range(1, len(result)):
    assert result[i].start >= result[i - 1].end, f"still overlapping/reordered at {i}: {result}"
print("OK: word order preserved:", result_words)

print("\n--- BUG REGRESSION (round 3): merge no longer flattens real stepwise melody ---")
from ultrastar_generator.note_detection import _merge_similar_adjacent
stepwise = [
    NoteEvent(start=0.00, end=0.20, pitch=-6),
    NoteEvent(start=0.21, end=0.40, pitch=-7),
    NoteEvent(start=0.41, end=0.60, pitch=-8),
    NoteEvent(start=0.61, end=0.80, pitch=-9),
]
merged = _merge_similar_adjacent(stepwise, max_pitch_diff=1, max_gap=0.05)
pitches_out = [n.pitch for n in merged]
assert len(merged) > 1, f"a real 3-semitone melodic run got flattened into one note: {pitches_out}"
assert set(pitches_out) != {pitches_out[0]}, f"all notes collapsed to the same pitch: {pitches_out}"
print(f"OK: 4-note stepwise run (-6,-7,-8,-9) stayed as {len(merged)} notes: {pitches_out}")

# But genuine vibrato-scale noise (tiny alternation around one pitch)
# should still collapse.
noisy_same_note = [
    NoteEvent(start=0.00, end=0.15, pitch=-4),
    NoteEvent(start=0.16, end=0.30, pitch=-5),
    NoteEvent(start=0.31, end=0.45, pitch=-4),
    NoteEvent(start=0.46, end=0.60, pitch=-5),
]
merged2 = _merge_similar_adjacent(noisy_same_note, max_pitch_diff=1, max_gap=0.05)
assert len(merged2) == 1, f"expected vibrato-scale noise to merge into 1 note, got {merged2}"
print(f"OK: noisy single-note run merged into {len(merged2)} note: pitch={merged2[0].pitch}")


print("\n--- phrasing + writer end-to-end on synthetic data ---")
syls = [
    Syllable("This", 0.0, 0.25, 0, True),
    Syllable(" is", 0.30, 0.55, 2, True),
    Syllable(" a", 0.60, 0.70, 3, True),
    Syllable(" test", 0.75, 1.10, 5, True),
    Syllable("Sec", 2.00, 2.20, 7, True),
    Syllable("ond", 2.20, 2.45, 7, False),
    Syllable(" line", 2.50, 2.90, 8, True),
]
entries = build_lines(syls)
kinds = [type(e).__name__ for e in entries]
assert "LineBreak" in kinds

song = Song(
    title="Test Song", artist="Test Artist", mp3="Test Artist - Test Song.mp3",
    bpm=120.0, gap_ms=8300,
    entries=[
        Syllable("This", 8.300, 8.550, 0, is_word_start=True),
        Syllable("is", 8.600, 8.850, 2, is_word_start=True),
        LineBreak(start=8.850, end=9.000),
        Syllable("Sec", 9.000, 9.200, 4, is_word_start=True),
        Syllable("ond", 9.200, 9.450, 4, is_word_start=False),
    ],
)
txt = render_song(song)
assert txt.startswith("#TITLE:Test Song\n")
assert "#BPM:120\n" in txt
assert "#GAP:8300\n" in txt
assert txt.strip().endswith("E")
assert "\n: 0 " in txt
print(txt)
print("OK: writer output structurally correct")

print("\nALL DRY-RUN CHECKS PASSED")

print("\n--- pitch_to_note_name conversions ---")
from ultrastar_generator.pitch import ultrastar_pitch_to_note_name
cases = {-4: "G#3", 0: "C4", 4: "E4", 3: "D#4", 1: "C#4", 6: "F#4", -9: "D#3"}
for pitch, expected in cases.items():
    got = ultrastar_pitch_to_note_name(pitch)
    assert got == expected, f"pitch {pitch}: expected {expected}, got {got}"
print("OK:", cases)

print("\n--- note_detection hard non-overlap guarantee catches a synthetic bad case ---")
from ultrastar_generator.note_detection import _ensure_nonoverlapping
bad_notes = [
    NoteEvent(start=0.0, end=1.0, pitch=0),
    NoteEvent(start=0.5, end=1.5, pitch=2),  # overlaps -- shouldn't be possible, but check the net
]
fixed_notes = _ensure_nonoverlapping(bad_notes, verbose=False)
for i in range(1, len(fixed_notes)):
    assert fixed_notes[i].start >= fixed_notes[i - 1].end
print("OK: synthetic overlap caught and fixed:", [(n.start, n.end) for n in fixed_notes])

print("\n--- debug_output: pass-1-only file is a valid, loadable UltraStar txt ---")
from ultrastar_generator.debug_output import build_pass1_debug_song
from ultrastar_generator.usdx_writer import render_song as _render
debug_notes = [
    NoteEvent(start=0.00, end=0.40, pitch=-4),
    NoteEvent(start=0.40, end=0.80, pitch=-2),
    NoteEvent(start=2.00, end=2.30, pitch=0),   # gap before this -> should get a line break
]
debug_song = build_pass1_debug_song(debug_notes, "Artist", "Title", "Artist - Title.mp3", 120.0, gap_ms=0)
debug_txt = _render(debug_song)
assert "#TITLE:Title [PASS1 DEBUG]" in debug_txt
assert "G#3" in debug_txt and "A#3" in debug_txt and "C4" in debug_txt
assert debug_txt.count("\n- ") >= 1, "expected at least one line break for the 1.2s gap"
assert debug_txt.strip().endswith("E")
print("OK: pass-1 debug file renders correctly, note names present, line break inserted for the gap")

print("\n--- lyrics_lookup.parse_lyrics_lines filters annotations/credits/blank lines ---")
from ultrastar_generator.lyrics_lookup import parse_lyrics_lines
raw = "[Verse 1]\nThere, out in the darkness\n\nA fugitive running\n[Chorus]\nfallen from grace\nParoles de la chanson par LyricFind\n"
lines = parse_lyrics_lines(raw)
assert lines == ["There, out in the darkness", "A fugitive running", "fallen from grace"], lines
print("OK:", lines)

print("\n--- lyrics_lookup.align_words_to_reference: fixes 'is'->'his' and tags line_id ---")
from ultrastar_generator.lyrics_lookup import align_words_to_reference, alignment_diff_summary
ref_lines_test = [
    "He knows his way in the dark,",
    "But mine is the way of the Lord",
]
asr_words = [
    Word(text="He", start=0.0, end=0.2, confidence=0.9),
    Word(text="knows", start=0.2, end=0.4, confidence=0.9),
    Word(text="is", start=0.4, end=0.6, confidence=0.9),   # mis-heard "his"
    Word(text="way", start=0.6, end=0.8, confidence=0.9),
    Word(text="in", start=0.8, end=1.0, confidence=0.9),
    Word(text="the", start=1.0, end=1.2, confidence=0.9),
    Word(text="dark", start=1.2, end=1.4, confidence=0.9),
    Word(text="But", start=2.0, end=2.2, confidence=0.9),
    Word(text="mine", start=2.2, end=2.4, confidence=0.9),
    Word(text="is", start=2.4, end=2.6, confidence=0.9),
    Word(text="the", start=2.6, end=2.8, confidence=0.9),
    Word(text="way", start=2.8, end=3.0, confidence=0.9),
]
aligned = align_words_to_reference(asr_words, ref_lines_test)
assert aligned[2].text == "his", f"expected 'is' corrected to 'his', got {aligned[2].text!r}"
diffs = alignment_diff_summary(asr_words, aligned)
assert any('"is" -> "his"' in d for d in diffs), diffs
# line_id: first 7 words on line 0, remaining words on line 1
assert all(w.line_id == 0 for w in aligned[:7]), [w.line_id for w in aligned[:7]]
assert all(w.line_id == 1 for w in aligned[7:]), [w.line_id for w in aligned[7:]]
# timing must be untouched by the text correction
assert aligned[2].start == 0.4 and aligned[2].end == 0.6
print("OK: corrected words:", diffs)
print("OK: line ids:", [w.line_id for w in aligned])

print("\n--- phrasing forces a break exactly on a line_id change (even with no silence gap) ---")
line_syls = [
    Syllable("He", 0.0, 0.2, 4, True, line_id=0),
    Syllable(" knows", 0.2, 0.4, 4, True, line_id=0),
    # zero-gap transition straight into line 1 -- must still break
    Syllable("But", 0.4, 0.6, 4, True, line_id=1),
    Syllable(" mine", 0.6, 0.8, 4, True, line_id=1),
]
line_entries = build_lines(line_syls)
kinds2 = [type(e).__name__ for e in line_entries]
assert kinds2.count("LineBreak") == 1, kinds2
break_pos = kinds2.index("LineBreak")
assert [type(e).__name__ for e in line_entries[:break_pos]] == ["Syllable", "Syllable"]
print("OK: line break inserted exactly at the line_id change:", kinds2)

print("\n--- BUG REGRESSION (round 6): a bad interior ASR timestamp no longer swallows a whole line's notes ---")
# Reproduces the reported "Stars" bug directly: within a matched
# reference line, one interior word's ASR timing is badly wrong ("Stars"
# is reported as a tiny sliver) -- under the OLD per-word-zone algorithm
# this would dump a huge stretch of real, musically-distinct notes onto
# "Stars" as one giant melisma. The current algorithm instead SPLITS
# notes at each word's own ASR start/end boundary (see
# lyric_alignment._split_notes_by_word_boundaries), so a note spanning
# more than one word's zone gets cut into same-pitch pieces rather than
# handed whole to whichever word's timestamp happens to contain its
# midpoint -- a tiny/bad timestamp can only ever claim the sliver of
# note-time that actually falls inside it. Gaps between words are kept
# small (<= config.NOTE_ASSIGNMENT_MAX_GAP_SEC) so grouping (purely
# gap-based, see lyric_alignment._group_words_by_gap) keeps them as ONE
# group -- a real multi-second gap between interior words of the same
# line would now correctly be treated as a real phrase boundary, not
# this bug; the real "Stars" bug's actual raw ASR data (found later, see
# [[project-stars-reference-notes]]) looked exactly like this: several
# words compressed with SMALL gaps between them, not one big gap.
line_words = [
    Word(text="Stars", start=34.2, end=34.7, confidence=0.9, line_id=5),        # 1 syllable
    Word(text="in", start=34.8, end=34.9, confidence=0.9, line_id=5),           # 1 syllable, compressed but same phrase
    Word(text="your", start=34.9, end=35.1, confidence=0.9, line_id=5),         # 1 syllable
    Word(text="multitudes", start=35.1, end=36.7, confidence=0.9, line_id=5),   # 3 syllables
]
# 12 real, evenly-spaced notes spanning the whole line -- exactly the
# kind of musically-distinct content seen in the actual pass-1 debug file
# for this passage.
line_notes = [NoteEvent(start=34.2 + i * 0.45, end=34.2 + i * 0.45 + 0.4, pitch=i % 5) for i in range(12)]

line_syllables, line_stats = align_words_to_notes(line_words, line_notes, np.zeros(16000), 16000)
assert line_stats.lines_word_boundary_split == 1, line_stats
# Track note-piece-count per ORIGINAL word (by word-start order), not by
# syllable text -- "multitudes" hyphenates into multiple differently-
# texted syllables, so grouping by text alone would misattribute them.
counts_by_word = {}
word_idx = -1
for s in line_syllables:
    if s.is_word_start:
        word_idx += 1
    key = line_words[word_idx].text
    counts_by_word[key] = counts_by_word.get(key, 0) + 1
print("Notes per word:", counts_by_word)
# "Stars" (a tiny 0.5s ASR span) must NOT dominate -- it should only get
# the note pieces whose time actually falls within its own boundary
# (here: one whole leading note plus a sliver split off the next one),
# nowhere close to swallowing most of the line the way the old bug did.
# "multitudes" (spans 35.1-36.7s, most of the line) should get most of
# the notes/pieces. A word can end up with MORE pieces than the original
# note count since a single note can now be split across several words.
assert counts_by_word.get("Stars", 0) <= 4, f"'Stars' still dominating the line: {counts_by_word}"
assert counts_by_word.get("multitudes", 0) >= counts_by_word.get("Stars", 0), counts_by_word
assert sum(counts_by_word.values()) >= 12
print("OK: line notes split by each word's own ASR boundary, not swallowed by one bad interior timestamp")

print("\n--- BUG REGRESSION (round 7): isolated pitch spike gets removed ---")
from ultrastar_generator.note_detection import _remove_pitch_spikes
# Reproduces the reported "The" bug's shape directly: a brief, isolated
# jump to a very different pitch that then returns to the surrounding
# pitch. Neighbors are close in time and pitch to each other; the spike
# is short and far from both.
spiky = [
    NoteEvent(start=0.00, end=0.30, pitch=0),
    NoteEvent(start=0.30, end=0.35, pitch=8),    # spike: 50ms, 8 semitones away
    NoteEvent(start=0.36, end=0.70, pitch=0),
]
despiked = _remove_pitch_spikes(spiky, max_duration=0.25, min_jump_semitones=4.0,
                                 neighbor_similarity_semitones=2.0, max_neighbor_gap=0.15)
# _remove_pitch_spikes alone absorbs the spike into the previous note,
# which can leave it newly adjacent to an identical-pitch note on the
# other side (detect_notes runs one more merge pass afterward to clean
# that up -- tested separately below at the full-pipeline level).
assert len(despiked) == 2, f"expected the spike absorbed, leaving 2 same-pitch notes: {despiked}"
assert despiked[0].pitch == 0 and despiked[1].pitch == 0
assert despiked[0].end == 0.35 and despiked[1].start == 0.36
print("OK: spike absorbed into the previous note:", despiked)

# A genuinely different short note between two DIFFERENT-pitched
# neighbors must NOT be treated as a spike (neighbors aren't "the same
# pitch as before").
real_short_note = [
    NoteEvent(start=0.00, end=0.30, pitch=0),
    NoteEvent(start=0.30, end=0.35, pitch=8),
    NoteEvent(start=0.36, end=0.70, pitch=5),   # different from the first neighbor
]
kept = _remove_pitch_spikes(real_short_note, max_duration=0.25, min_jump_semitones=4.0,
                             neighbor_similarity_semitones=2.0, max_neighbor_gap=0.15)
assert len(kept) == 3, f"a real note between two DIFFERENT pitches should not be treated as a spike: {kept}"
print("OK: short note between two different-pitched neighbors correctly kept:", [n.pitch for n in kept])

print("\n--- BUG REGRESSION (round 7): fallback pitch borrows nearest pass-1 note, not fresh noisy analysis ---")
# Reproduces the "The" bug's OTHER half: a word with zero notes in its
# own zone now borrows the pitch of the nearest pass-1 note (from the
# FULL note list) instead of running a fresh, isolated pitch read.
# Needs surrounding words so zone partitioning actually excludes the
# nearby notes from "the"'s own zone (with only one word in the whole
# list, everything trivially falls in its one zone).
fallback_words = [
    Word(text="start", start=90.0, end=94.9, confidence=0.9, line_id=None),
    Word(text="the", start=100.0, end=100.15, confidence=0.9, line_id=None),
    Word(text="end", start=105.1, end=110.0, confidence=0.9, line_id=None),
]
all_real_notes = [
    NoteEvent(start=94.0, end=96.0, pitch=0),    # lands in "start"'s zone
    NoteEvent(start=103.0, end=106.0, pitch=0),  # lands in "end"'s zone
]
fb_syllables, fb_stats = align_words_to_notes(fallback_words, all_real_notes, np.zeros(16000), 16000)
the_syllable = next(s for s in fb_syllables if s.text.strip() == "the")
assert fb_stats.words_with_fallback == 1, fb_stats
assert fb_stats.fallback_used_neighbor == 1
assert fb_stats.fallback_used_fresh_analysis == 0
assert the_syllable.midi_note == 0, the_syllable
print(f"OK: fallback word borrowed pitch {the_syllable.midi_note} from the nearest pass-1 note "
      f"(neighbor-borrow count={fb_stats.fallback_used_neighbor}, fresh-analysis count={fb_stats.fallback_used_fresh_analysis})")

print("\n--- BUG REGRESSION (round 7): spike removal end-to-end through detect_notes() collapses fully ---")
# Same spike shape as above, but through the real detect_notes() pipeline
# (mocked librosa, same technique as the earlier vibrato test), to
# confirm the post-spike-removal re-merge actually produces ONE note.
n_pre, n_spike, n_post = 40, 15, 40
n_frames_spike_test = n_pre + n_spike + n_post
frame_dur_test = 256 / 22050
base_midi_spike = 60.0  # C4 (pitch 0)
spike_midi_val = 68.0   # 8 semitones away
contour = np.concatenate([
    np.full(n_pre, base_midi_spike),
    np.full(n_spike, spike_midi_val),
    np.full(n_post, base_midi_spike),
])
f0_spike_test = 440.0 * 2 ** ((contour - 69) / 12)

fake_librosa_spike = _types.ModuleType("librosa")
fake_librosa_spike.pyin = lambda y, fmin, fmax, sr, frame_length, hop_length, fill_na=np.nan: (
    f0_spike_test, np.ones(n_frames_spike_test, dtype=bool), np.full(n_frames_spike_test, 0.9),
)
fake_librosa_spike.times_like = lambda x, sr, hop_length: np.arange(len(x)) * (hop_length / sr)
fake_librosa_spike.onset = _FakeOnset()


class _FakeFeatureSpike:
    @staticmethod
    def rms(y, frame_length, hop_length):
        return np.full((1, n_frames_spike_test), 0.2)  # uniformly loud -- energy gate isn't the point here


fake_librosa_spike.feature = _FakeFeatureSpike()
_sys.modules["librosa"] = fake_librosa_spike
importlib.reload(note_detection_mod)

spike_pipeline_notes = note_detection_mod.detect_notes(np.zeros(4410), 22050, bpm=105.47, verbose=False)
print(f"Detected {len(spike_pipeline_notes)} note(s) for a base-spike-base contour "
      f"({n_spike * frame_dur_test * 1000:.0f}ms spike)")
assert len(spike_pipeline_notes) == 1, f"expected full end-to-end collapse to 1 note, got {spike_pipeline_notes}"
assert spike_pipeline_notes[0].pitch == 0, spike_pipeline_notes[0]
print("OK: end-to-end pipeline fully collapsed the spike, no trace of it left:", spike_pipeline_notes)

print("\n--- BUG REGRESSION: a strong onset backed by a real energy dip splits two "
      "same-pitch re-articulated notes (previously always merged -- no pitch change "
      "ever gets a split, no matter how strong the onset) ---")
n_pre_rearticulate = 30
n_post_rearticulate = 30
n_frames_rearticulate = n_pre_rearticulate + n_post_rearticulate
same_midi = 60.0  # C4 throughout -- no pitch change at all
f0_rearticulate = np.full(n_frames_rearticulate, 440.0 * 2 ** ((same_midi - 69) / 12))

fake_librosa_rearticulate = _types.ModuleType("librosa")
fake_librosa_rearticulate.pyin = lambda y, fmin, fmax, sr, frame_length, hop_length, fill_na=np.nan: (
    f0_rearticulate, np.ones(n_frames_rearticulate, dtype=bool), np.full(n_frames_rearticulate, 0.9),
)
fake_librosa_rearticulate.times_like = lambda x, sr, hop_length: np.arange(len(x)) * (hop_length / sr)


class _FakeOnsetRearticulate:
    @staticmethod
    def onset_detect(y, sr, hop_length, backtrack, units):
        # A single, real re-attack roughly halfway through -- well past
        # config.MIN_DURATION_BEFORE_REARTICULATION_SEC into the note.
        onset_frame = n_pre_rearticulate
        return np.array([onset_frame * hop_length / sr])

    @staticmethod
    def onset_strength(y, sr, hop_length):
        # Single onset -> trivially at the top percentile among itself.
        env = np.zeros(n_frames_rearticulate)
        env[n_pre_rearticulate] = 5.0
        return env


fake_librosa_rearticulate.onset = _FakeOnsetRearticulate()


class _FakeFeatureRearticulate:
    @staticmethod
    def rms(y, frame_length, hop_length):
        return np.full((1, n_frames_rearticulate), 0.2)  # loud throughout


fake_librosa_rearticulate.feature = _FakeFeatureRearticulate()
_sys.modules["librosa"] = fake_librosa_rearticulate
importlib.reload(note_detection_mod)

rearticulate_notes = note_detection_mod.detect_notes(np.zeros(4410), 22050, bpm=105.47, verbose=False)
print(f"Detected {len(rearticulate_notes)} note(s) for a same-pitch re-articulation:", rearticulate_notes)
assert len(rearticulate_notes) == 2, \
    f"expected the strong onset to split same-pitch re-articulation into 2 notes, got {rearticulate_notes}"
assert rearticulate_notes[0].pitch == rearticulate_notes[1].pitch == 0, rearticulate_notes
print("OK: strong onset + energy dip split two same-pitch notes that a bare pitch-only check would have merged")

print("\n--- weak onset with no pitch change still does NOT split (avoids reintroducing "
      "the old 'consonant transient inside one note' spurious-split bug) ---")


class _FakeOnsetWeak:
    @staticmethod
    def onset_detect(y, sr, hop_length, backtrack, units):
        # Two onsets: a much STRONGER decoy very early (frame 3 -- before
        # config.MIN_DURATION_BEFORE_REARTICULATION_SEC has elapsed, so it
        # can't split anything itself) that raises the percentile bar high
        # enough that the frame-30 onset (the one actually inside the
        # sustained note) no longer clears it.
        return np.array([3 * hop_length / sr, n_pre_rearticulate * hop_length / sr])

    @staticmethod
    def onset_strength(y, sr, hop_length):
        env = np.zeros(n_frames_rearticulate)
        env[3] = 10.0
        env[n_pre_rearticulate] = 1.0
        return env


fake_librosa_rearticulate.onset = _FakeOnsetWeak()
_sys.modules["librosa"] = fake_librosa_rearticulate
importlib.reload(note_detection_mod)

weak_onset_notes = note_detection_mod.detect_notes(np.zeros(4410), 22050, bpm=105.47, verbose=False)
print(f"Detected {len(weak_onset_notes)} note(s) for a same-pitch run with a uniformly weak onset:",
      weak_onset_notes)
assert len(weak_onset_notes) == 1, \
    f"a uniformly weak onset (no strength signal to distinguish it) should NOT split, got {weak_onset_notes}"
print("OK: weak/undifferentiated onset did not spuriously split a sustained note")

print("\n--- CREPE cross-check: agreement uses CREPE's (more accurate) pitch ---")
import torch as _torch

n_frames_crepe = 40


class _FakeOnsetNone:
    @staticmethod
    def onset_detect(y, sr, hop_length, backtrack, units):
        return np.array([])

    @staticmethod
    def onset_strength(y, sr, hop_length):
        return np.zeros(n_frames_crepe)


def _make_fake_librosa_crepe(pyin_midi: float):
    f0 = np.full(n_frames_crepe, 440.0 * 2 ** ((pyin_midi - 69) / 12))
    fake = _types.ModuleType("librosa")
    fake.pyin = lambda y, fmin, fmax, sr, frame_length, hop_length, fill_na=np.nan: (
        f0, np.ones(n_frames_crepe, dtype=bool), np.full(n_frames_crepe, 0.9),
    )
    fake.times_like = lambda x, sr, hop_length: np.arange(len(x)) * (hop_length / sr)
    fake.onset = _FakeOnsetNone()

    class _Feat:
        @staticmethod
        def rms(y, frame_length, hop_length):
            return np.full((1, n_frames_crepe), 0.2)

    fake.feature = _Feat()
    return fake


# pYIN says midi 60.4 (rounds to 60) throughout; CREPE agrees closely
# (61.0, within config.CREPE_AGREEMENT_SEMITONES) -- final pitch should
# follow CREPE's value (61), not pYIN's.
_sys.modules["librosa"] = _make_fake_librosa_crepe(60.4)
fake_torchcrepe_agree = _types.ModuleType("torchcrepe")
_crepe_hz_agree = 440.0 * 2 ** ((61.0 - 69) / 12)
fake_torchcrepe_agree.predict = lambda audio, sr, hop_length, fmin, fmax, model, batch_size, device, return_periodicity: (
    _torch.full((1, n_frames_crepe), _crepe_hz_agree), _torch.full((1, n_frames_crepe), 0.9),
)
_sys.modules["torchcrepe"] = fake_torchcrepe_agree
importlib.reload(note_detection_mod)

crepe_agree_notes = note_detection_mod.detect_notes(np.zeros(4410), 22050, bpm=105.47, verbose=True, use_crepe=True)
print(f"Detected {len(crepe_agree_notes)} note(s):", crepe_agree_notes)
assert len(crepe_agree_notes) == 1, crepe_agree_notes
assert crepe_agree_notes[0].pitch == 1, f"expected CREPE's pitch (61 -> relative 1) to be used, got {crepe_agree_notes[0].pitch}"
print("OK: CREPE's pitch used over pYIN's when they agree")

print("\n--- CREPE cross-check: disagreement keeps pYIN's pitch, downweights confidence, "
      "and does NOT fabricate a note split ---")
_sys.modules["librosa"] = _make_fake_librosa_crepe(60.0)
fake_torchcrepe_disagree = _types.ModuleType("torchcrepe")
_crepe_hz_agree_part = 440.0 * 2 ** ((60.0 - 69) / 12)
_crepe_hz_disagree_part = 440.0 * 2 ** ((75.0 - 69) / 12)  # 15 semitones off pYIN -- clear disagreement


def _predict_disagree(audio, sr, hop_length, fmin, fmax, model, batch_size, device, return_periodicity):
    pitch = _torch.full((1, n_frames_crepe), _crepe_hz_agree_part)
    pitch[0, 30:] = _crepe_hz_disagree_part
    periodicity = _torch.full((1, n_frames_crepe), 0.9)
    return pitch, periodicity


fake_torchcrepe_disagree.predict = _predict_disagree
_sys.modules["torchcrepe"] = fake_torchcrepe_disagree
importlib.reload(note_detection_mod)

crepe_disagree_notes = note_detection_mod.detect_notes(np.zeros(4410), 22050, bpm=105.47, verbose=True, use_crepe=True)
print(f"Detected {len(crepe_disagree_notes)} note(s):", crepe_disagree_notes)
assert len(crepe_disagree_notes) == 1, \
    f"disagreement alone must not fabricate a note boundary, got {crepe_disagree_notes}"
assert crepe_disagree_notes[0].pitch == 0, \
    f"disagreeing frames should keep pYIN's pitch (majority vote still 0), got {crepe_disagree_notes[0].pitch}"
import ultrastar_generator.config as config_mod
# First 30 frames AGREE with CREPE too (both at pYIN's own pitch), so
# they get the confidence BOOST just like the all-agreeing test above;
# only the last 10 frames disagree and get downweighted relative to that
# -- so the mixed case's overall confidence should land measurably below
# the all-agreeing case's (1.35, asserted above), not because agreement
# "helps" less here but because 10/40 frames dropped from boosted to
# downweighted.
assert crepe_disagree_notes[0].confidence < crepe_agree_notes[0].confidence, \
    (crepe_disagree_notes[0].confidence, crepe_agree_notes[0].confidence)
_downweighted = 0.9 * config_mod.CREPE_DISAGREEMENT_CONFIDENCE_SCALE
assert crepe_disagree_notes[0].confidence > _downweighted, \
    "mixed confidence should sit above the fully-downweighted floor (some frames still boosted)"
print(f"OK: disagreement kept pYIN's pitch and stayed one note, but confidence "
      f"({crepe_disagree_notes[0].confidence:.3f}) was measurably lower than the all-agreeing "
      f"case ({crepe_agree_notes[0].confidence:.3f})")
del _sys.modules["torchcrepe"]

print("\n--- lyric_alignment flags suspicious words: any word whose own ASR span gets zero note pieces ---")
susp_words = [
    Word(text="clearly", start=0.0, end=0.5, confidence=0.9),  # gets a real note -> not suspicious
    Word(text="mumbled", start=2.0, end=2.3, confidence=0.9),  # zero notes -> fallback -> suspicious
    Word(text="multitudinous", start=5.0, end=5.6, confidence=0.9, line_id=0),  # matched line, gets the note -> not suspicious
    Word(text="word", start=5.6, end=5.8, confidence=0.9, line_id=0),          # same line, no note overlaps it -> suspicious
]
susp_notes = [
    NoteEvent(start=0.0, end=0.5, pitch=0, confidence=1.0),
    # nothing near 2.0-2.3s -> "mumbled" is a fallback
    NoteEvent(start=5.0, end=5.3, pitch=0, confidence=1.0),  # only 1 note for a 2-word line -- entirely
    # within "multitudinous"'s own span, so word-boundary splitting correctly gives none of it to "word"
]
_, susp_stats = align_words_to_notes(susp_words, susp_notes, np.zeros(4410), 22050)
assert 1 in susp_stats.suspicious_word_indices, susp_stats.suspicious_word_indices
assert 3 in susp_stats.suspicious_word_indices, susp_stats.suspicious_word_indices
assert 2 not in susp_stats.suspicious_word_indices, susp_stats.suspicious_word_indices
assert 0 not in susp_stats.suspicious_word_indices, susp_stats.suspicious_word_indices
print(f"OK: suspicious word indices correctly identified: {sorted(set(susp_stats.suspicious_word_indices))}")

print("\n--- verification: chunk re-transcription resolves against reference lyrics, "
      "not just self-consistency ---")
import ultrastar_generator.verification as verification_mod


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _SequencedFakeASRModel:
    """Deterministic fake: returns a fixed sequence of 'rechecked' texts,
    one per call, in the order verify_words() processes words (sorted
    index order) -- lets the test drive each of _resolve()'s branches
    independently."""
    _responses = ["rumbled", "multitudinous", "totally different", "echo", "echo"]

    def __init__(self, *a, **k):
        self._iter = iter(self._responses)

    def transcribe(self, audio, language=None, batch_size=None):
        return {"segments": [{"text": next(self._iter), "start": 0.0, "end": 1.0}]}


def _fake_load_model_verify_words(model_name, device=None, compute_type=None, language=None, vad_options=None):
    return _SequencedFakeASRModel()


fake_whisperx_verify_words = _types.ModuleType("whisperx")
fake_whisperx_verify_words.load_model = _fake_load_model_verify_words
_sys.modules["whisperx"] = fake_whisperx_verify_words
verification_mod.model_cache.reset()
# Earlier tests replaced sys.modules["librosa"] with note_detection-specific
# fakes that don't implement .resample(); verification.py needs the real
# thing, so drop the fake and let it re-import genuinely.
_sys.modules.pop("librosa", None)

verify_test_words = [
    # 0: no reference at all; recheck disagrees with current text -> replaced with recheck.
    Word(text="mumbled", start=0.0, end=0.3, confidence=0.9, line_id=None, reference_text=None),
    # 1: current text does NOT match reference, but the recheck CONFIRMS the
    # reference (case-insensitively) -> replaced with the reference's own text
    # (not the recheck's raw casing), fixing lyrics_lookup's "uneven block" case.
    Word(text="multitude", start=10.0, end=10.3, confidence=0.9, line_id=0, reference_text="Multitudinous"),
    # 2: current text wrong, recheck ALSO doesn't match reference (3-way
    # disagreement) -> falls back to trusting the reference anyway.
    Word(text="Stray", start=20.0, end=20.3, confidence=0.9, line_id=0, reference_text="Stars"),
    # 3: current text already matches reference -> left alone regardless of
    # what the recheck (mock says "echo") hears.
    Word(text="Stars", start=30.0, end=30.3, confidence=0.9, line_id=0, reference_text="Stars"),
    # 4: no reference; recheck agrees with current text -> left alone.
    Word(text="echo", start=40.0, end=40.3, confidence=0.9, line_id=None, reference_text=None),
]
y_fake = np.zeros(22050 * 42, dtype=np.float32)
new_words, verify_results = verification_mod.verify_words(
    verify_test_words, [0, 1, 2, 3, 4], y_fake, 22050, "small.en", verbose=True,
)
assert new_words[0].text == "rumbled", new_words[0]
assert new_words[1].text == "Multitudinous", new_words[1]  # reference's own text, not the recheck's raw casing
assert new_words[2].text == "Stars", new_words[2]           # forced to reference despite no confirmation
assert new_words[3].text == "Stars", new_words[3]           # untouched -- already matched reference
assert new_words[4].text == "echo", new_words[4]            # untouched -- recheck agreed
assert [r.replaced for r in verify_results] == [True, True, True, False, False], verify_results
print("OK: verification correctly resolved all 5 cases against reference lyrics:",
      [w.text for w in new_words])
del _sys.modules["whisperx"]
verification_mod.model_cache.reset()

print("\n--- verification._word_spans_from_syllables: reconstructs each word's FINAL "
      "note-assigned span from its syllable run ---")
span_words = [
    Word(text="Lucifer", start=10.0, end=10.5, confidence=0.9),
    Word(text="fell", start=11.0, end=11.3, confidence=0.9),
]
span_syllables = [
    Syllable(text="Lu", start=10.1, end=10.3, midi_note=0, is_word_start=True),
    Syllable(text="cifer", start=10.3, end=10.6, midi_note=0, is_word_start=False),
    Syllable(text="fell", start=10.7, end=11.4, midi_note=0, is_word_start=True),
]
spans = verification_mod._word_spans_from_syllables(span_words, span_syllables)
assert spans == [(10.1, 10.6), (10.7, 11.4)], spans
print("OK: word spans correctly reconstructed from multi-syllable runs:", spans)

print("\n--- verification.verify_placement: crops a small window at each word's FINAL "
      "note-assigned position, expands it until the expected word is found, then refines "
      "the exact position with forced alignment over that confirmed window (the real "
      "'Stars' bug: text was correct, but pass 3 put it far from where it's really sung) ---")


class _ExpandSearchFakeASRModel:
    """Deterministic fake: 'Stars' isn't found until the search window has
    grown enough to reach its real position (~60.0s, far from where pass 3
    assigned it); 'your' is found immediately since pass 3 assigned it
    correctly. One response per transcribe() call, in call order."""
    _responses = ["", "", "", "the great Stars shine", "your"]

    def __init__(self, *a, **k):
        self._iter = iter(self._responses)

    def transcribe(self, audio, language=None, batch_size=None):
        return {"segments": [{"text": next(self._iter)}]}


def _fake_load_model(model_name, device=None, compute_type=None, language=None, vad_options=None):
    return _ExpandSearchFakeASRModel()


def _fake_load_align_model(language_code=None, device=None):
    return object(), {}


_fake_align_responses = {
    # Window at the point "Stars" is finally found is (44.7, 60.7) -- see
    # the radius math in the comment below -- so a relative offset of
    # 15.3s is absolute 60.0s, matching where it's really sung.
    "the great Stars shine": [
        {"word": "the", "start": 14.0, "end": 14.2, "score": 0.9},
        {"word": "great", "start": 14.3, "end": 14.6, "score": 0.9},
        {"word": "Stars", "start": 15.3, "end": 15.6, "score": 0.9},
        {"word": "shine", "start": 15.7, "end": 16.0, "score": 0.9},
    ],
    # Window at (79.5, 81.5); relative 1.0s is absolute 80.5s, exactly
    # matching pass 3's (correct) assignment.
    "your": [
        {"word": "your", "start": 1.0, "end": 1.3, "score": 0.9},
    ],
}


def _fake_align(segments, model, metadata, audio, device=None):
    words_out = _fake_align_responses.get(segments[0]["text"], [])
    return {"segments": [{"words": words_out}]}


fake_whisperx = _types.ModuleType("whisperx")
fake_whisperx.load_model = _fake_load_model
fake_whisperx.load_align_model = _fake_load_align_model
fake_whisperx.align = _fake_align
_sys.modules["whisperx"] = fake_whisperx
verification_mod.model_cache.reset()

expand_words = [
    # "Stars" mis-assigned notes at 52.7s; really sung at ~60.0s. Initial
    # search radius 1.0s doubles each miss (1, 2, 4, 8) -- at radius 8.0
    # the window (44.7, 60.7) finally reaches 60.0s. The fake aligner
    # returns a precise per-word hit for "Stars", so this should be
    # AUTO-CORRECTED, not just flagged.
    Word(text="Stars", start=80.0, end=80.4, confidence=0.9, line_id=0, reference_text="Stars"),
    # "your" correctly assigned -- found on the very first (radius 1.0) try.
    Word(text="your", start=80.5, end=80.8, confidence=0.9, line_id=0, reference_text="your"),
]
expand_syllables = [
    Syllable(text="Stars", start=52.7, end=53.0, midi_note=0, is_word_start=True, line_id=0),
    Syllable(text="your", start=80.5, end=80.8, midi_note=0, is_word_start=True, line_id=0),
]
y_fake_expand = np.zeros(16000 * 100, dtype=np.float32)
expand_words_out, expand_corrections, expand_warnings = verification_mod.verify_placement(
    expand_words, expand_syllables, [0, 1], y_fake_expand, 16000, "small.en", verbose=True,
)
assert expand_warnings == [], expand_warnings
assert [c.word_index for c in expand_corrections] == [0], expand_corrections
assert expand_corrections[0].word_text == "Stars", expand_corrections[0]
assert abs(expand_corrections[0].new_start - 60.0) < 0.01, expand_corrections[0]
assert abs(expand_words_out[0].start - 60.0) < 0.01, expand_words_out[0]
assert expand_words_out[1].start == 80.5, expand_words_out[1]  # "your" untouched
print("OK: expand-search placement check correctly auto-corrected the mis-assigned word, "
      "left the correctly-assigned one untouched:",
      [(c.word_index, c.word_text, c.new_start) for c in expand_corrections])
del _sys.modules["whisperx"]
verification_mod.model_cache.reset()

print("\n--- verification.verify_placement: a word genuinely not findable anywhere in the "
      "search radius stays a WARNING, never an auto-correction ---")


class _NeverFoundFakeASRModel:
    """Every transcribe() call returns empty text -- the expected word is
    never in the result, no matter how far the search window grows."""
    def __init__(self, *a, **k):
        pass

    def transcribe(self, audio, language=None, batch_size=None):
        return {"segments": [{"text": ""}]}


def _fake_load_model_never(model_name, device=None, compute_type=None, language=None, vad_options=None):
    return _NeverFoundFakeASRModel()


fake_whisperx_never = _types.ModuleType("whisperx")
fake_whisperx_never.load_model = _fake_load_model_never
fake_whisperx_never.load_align_model = _fake_load_align_model
fake_whisperx_never.align = _fake_align
_sys.modules["whisperx"] = fake_whisperx_never
verification_mod.model_cache.reset()

never_found_words = [
    Word(text="ghost", start=10.0, end=10.4, confidence=0.9, line_id=0, reference_text="ghost"),
]
never_found_syllables = [
    Syllable(text="ghost", start=10.0, end=10.4, midi_note=0, is_word_start=True, line_id=0),
]
y_fake_never = np.zeros(16000 * 30, dtype=np.float32)
never_words_out, never_corrections, never_warnings = verification_mod.verify_placement(
    never_found_words, never_found_syllables, [0], y_fake_never, 16000, "small.en", verbose=True,
)
assert never_corrections == [], never_corrections
assert [w.word_index for w in never_warnings] == [0], never_warnings
assert never_words_out[0].start == 10.0, never_words_out[0]  # untouched -- nothing confident to act on
print("OK: a genuinely unfindable word stayed a warning, word list left untouched:",
      [(w.word_index, w.word_text) for w in never_warnings])
del _sys.modules["whisperx"]
verification_mod.model_cache.reset()




