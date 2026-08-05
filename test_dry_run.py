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

print("\n--- key_correction: obvious out-of-key note gets snapped ---")
from ultrastar_generator.key_correction import snap_to_key
# Heavily-weighted C major scale (each in-key pitch class repeated 3x so
# the key detector isn't confused by a single outlier) plus ONE clear
# outlier at pitch class 6 (F#), which isn't in C major and sits between
# F(5, in-key) and G(7, in-key).
in_key = [0, 2, 4, 5, 7, 9, 11] * 3
c_major_syls = [
    Syllable(chr(97 + i), i * 0.5, i * 0.5 + 0.4, pc, True)
    for i, pc in enumerate(in_key)
]
outlier_idx = len(c_major_syls)
c_major_syls.append(Syllable("x", outlier_idx * 0.5, outlier_idx * 0.5 + 0.4, 6, True))
snapped = snap_to_key(c_major_syls)
assert snapped[outlier_idx].midi_note in (5, 7), snapped[outlier_idx].midi_note
print(f"OK: outlier pitch 6 snapped to {snapped[outlier_idx].midi_note}")

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
# reference line, one interior word's ASR timing is badly wrong (here,
# "Stars" is reported as a tiny sliver, and "in" doesn't start until much
# later than it should) -- under the OLD per-word-zone algorithm this
# would dump a huge stretch of real, musically-distinct notes onto
# "Stars" as one giant melisma. The new algorithm distributes a matched
# line's notes by syllable count instead of trusting each interior word's
# own timestamp.
line_words = [
    Word(text="Stars", start=34.2, end=34.7, confidence=0.9, line_id=5),        # 1 syllable
    Word(text="in", start=38.0, end=38.1, confidence=0.9, line_id=5),           # 1 syllable, badly late
    Word(text="your", start=38.1, end=38.3, confidence=0.9, line_id=5),         # 1 syllable
    Word(text="multitudes", start=38.3, end=39.9, confidence=0.9, line_id=5),   # 3 syllables
]
# 12 real, evenly-spaced notes spanning the whole line -- exactly the
# kind of musically-distinct content seen in the actual pass-1 debug file
# for this passage.
line_notes = [NoteEvent(start=34.2 + i * 0.45, end=34.2 + i * 0.45 + 0.4, pitch=i % 5) for i in range(12)]

line_syllables, line_stats = align_words_to_notes(line_words, line_notes, np.zeros(16000), 16000)
assert line_stats.lines_syllable_distributed == 1, line_stats
# Track note-count per ORIGINAL word (by word-start order), not by
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
# "Stars" (1/6 of the line's 6 total syllables) must NOT dominate --
# roughly proportional to syllable count (1,1,1,3 syllables out of 6
# total -> roughly 2,2,2,6 notes out of 12), and specifically nowhere
# close to swallowing most of the line the way the old bug did.
assert counts_by_word.get("Stars", 0) <= 4, f"'Stars' still dominating the line: {counts_by_word}"
assert counts_by_word.get("multitudes", 0) >= counts_by_word.get("Stars", 0), counts_by_word
assert sum(counts_by_word.values()) == 12
print("OK: line notes distributed by syllable count, not by one bad interior ASR timestamp")


