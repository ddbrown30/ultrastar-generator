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

from ultrastar_generator.file_discovery import (find_companions, parse_artist_title,
                                                 resolve_artist_title, resolve_primary_source,
                                                 AmbiguousInputError, NoAudioSourceFoundError)
from ultrastar_generator.tempo import beat_duration_ms, seconds_to_beat, seconds_to_beat_length, beat_to_seconds
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

print("\n--- file_discovery.resolve_artist_title: falls back to the input FOLDER's name "
      "when the audio file's own name doesn't parse (real case: a ripped/downloaded song "
      "keeps a generic filename like 'music.ogg' while its folder is 'Artist - Title') ---")
import tempfile as _tempfile_artist_title
with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    normal_folder = Path(_tmp) / "some_folder"
    normal_folder.mkdir()
    parseable_audio = normal_folder / "Bon Jovi - Its My Life.mp3"
    artist2, title2 = resolve_artist_title(parseable_audio, normal_folder)
    assert (artist2, title2) == ("Bon Jovi", "Its My Life"), (artist2, title2)
print("OK: when the audio filename parses fine, resolve_artist_title uses it directly (folder name ignored)")

with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    named_folder = Path(_tmp) / "Bon Jovi - Its My Life"
    named_folder.mkdir()
    unparseable_audio = named_folder / "music.mp3"
    artist3, title3 = resolve_artist_title(unparseable_audio, named_folder)
    assert (artist3, title3) == ("Bon Jovi", "Its My Life"), (artist3, title3)
print("OK: when the audio filename doesn't parse, resolve_artist_title falls back to the folder's own name")

with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    unparseable_folder = Path(_tmp) / "random_folder_name"
    unparseable_folder.mkdir()
    unparseable_audio2 = unparseable_folder / "music.mp3"
    artist4, title4 = resolve_artist_title(unparseable_audio2, unparseable_folder)
    assert (artist4, title4) == (None, None), (artist4, title4)
print("OK: when NEITHER the audio filename nor the folder name parse, resolve_artist_title "
      "returns (None, None) rather than raising")

# A dot inside the folder name must not be misread as a file extension
# (Path.stem would strip it; resolve_artist_title uses the folder's raw
# .name specifically to avoid that).
with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    dotted_folder = Path(_tmp) / "Mr. Roboto - Styx"
    dotted_folder.mkdir()
    unparseable_audio3 = dotted_folder / "track01.mp3"
    artist5, title5 = resolve_artist_title(unparseable_audio3, dotted_folder)
    assert (artist5, title5) == ("Mr. Roboto", "Styx"), (artist5, title5)
print("OK: a dot inside the folder name (e.g. 'Mr. Roboto - Styx') is NOT misread as a file "
      "extension when falling back to the folder name")
print("OK:", artist, title, comp)

print("\n--- file_discovery.find_companions: falls back to a single unambiguous video/image "
      "even when its name doesn't match the audio file's basename (real case: a SingStar-style "
      "rip with audio 'music.ogg' + unrelated 'video.mpg'/'cover.jpg' names) ---")
with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    rip_folder = Path(_tmp) / "Beauty And The Beast - Beauty And The Beast"
    rip_folder.mkdir()
    rip_audio = rip_folder / "music.ogg"
    rip_audio.write_bytes(b"fake-audio")
    (rip_folder / "video.mpg").write_bytes(b"fake-video")
    (rip_folder / "cover.jpg").write_bytes(b"fake-cover")
    rip_comp = find_companions(rip_audio)
    assert rip_comp.video is not None and rip_comp.video.name == "video.mpg", rip_comp.video
    assert rip_comp.cover is not None and rip_comp.cover.name == "cover.jpg", rip_comp.cover
    assert rip_comp.background is not None and rip_comp.background.name == "cover.jpg", rip_comp.background
print("OK: a single non-matching-name video (.mpg) and image are both picked up via the "
      "unambiguous-single-candidate fallback")

with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    # TWO non-matching, untagged images -- genuinely ambiguous, must NOT
    # guess (same principle as the existing "multiple untagged images"
    # case, just extended to the no-basename-match case too).
    ambiguous_folder = Path(_tmp) / "Some Song"
    ambiguous_folder.mkdir()
    ambiguous_audio = ambiguous_folder / "track.mp3"
    ambiguous_audio.write_bytes(b"fake-audio")
    (ambiguous_folder / "poster.jpg").write_bytes(b"fake-1")
    (ambiguous_folder / "screenshot.jpg").write_bytes(b"fake-2")
    ambiguous_comp = find_companions(ambiguous_audio)
    assert ambiguous_comp.cover is None and ambiguous_comp.background is None, ambiguous_comp
print("OK: two non-matching, untagged images stays ambiguous -- correctly finds nothing rather than "
      "guessing which one is the cover")

print("\n--- file_discovery: MusicXML reference files matched by EXTENSION, not basename "
      "(unlike video/cover -- a downloaded score keeps its own source filename) ---")
mxl_a = audio.parent / "some-random-arrangement-name.mxl"
mxl_b = audio.parent / "another-arrangement.musicxml"
for p in (mxl_a, mxl_b):
    if not p.exists():
        p.write_text("<!-- placeholder for file-discovery test, not real MusicXML -->", encoding="utf-8")
comp2 = find_companions(audio)
assert [p.name for p in comp2.musicxml] == sorted([mxl_a.name, mxl_b.name], key=str.lower), comp2.musicxml
print("OK: both differently-named reference files found, sorted deterministically:",
      [p.name for p in comp2.musicxml])

print("\n--- file_discovery.resolve_primary_source: folder-based input resolution ---")
import tempfile as _tempfile_fd

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "Artist - Title.mp3").write_bytes(b"")
    path, kind = resolve_primary_source(d)
    assert (path.name, kind) == ("Artist - Title.mp3", "audio"), (path, kind)
print("OK: single real audio file -> kind='audio'")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "Artist - Title.mp3").write_bytes(b"")
    (d / "Artist - Title (alt take).ogg").write_bytes(b"")
    try:
        resolve_primary_source(d)
        assert False, "should have raised AmbiguousInputError"
    except AmbiguousInputError as e:
        assert "Artist - Title.mp3" in str(e) and "alt take" in str(e), e
print("OK: two real audio files -> AmbiguousInputError naming both candidates")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "Artist - Title.mp3").write_bytes(b"")
    (d / "Artist - Title (alt take).ogg").write_bytes(b"")
    path, kind = resolve_primary_source(d, audio_file_override="Artist - Title (alt take).ogg")
    assert (path.name, kind) == ("Artist - Title (alt take).ogg", "audio"), (path, kind)
print("OK: --audio-file override resolves the ambiguity")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "video.mp4").write_bytes(b"")
    path, kind = resolve_primary_source(d)
    assert (path.name, kind) == ("video.mp4", "video_as_audio"), (path, kind)
print("OK: no real audio, one mp4 -> kind='video_as_audio'")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "video.mpg").write_bytes(b"")
    path, kind = resolve_primary_source(d)
    assert (path.name, kind) == ("video.mpg", "video_as_audio"), (path, kind)
print("OK: no real audio, one mpg -> kind='video_as_audio' (UltraStar Deluxe accepts .mpg as #MP3 directly)")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "video.mpeg").write_bytes(b"")
    path, kind = resolve_primary_source(d)
    assert (path.name, kind) == ("video.mpeg", "video_as_audio"), (path, kind)
print("OK: no real audio, one mpeg -> kind='video_as_audio'")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "old_recording.avi").write_bytes(b"")
    path, kind = resolve_primary_source(d)
    assert (path.name, kind) == ("old_recording.avi", "avi_extract"), (path, kind)
print("OK: no real audio, no direct-audio video, one avi -> kind='avi_extract' "
      "(UltraStar Deluxe can NOT use .avi as #MP3 directly, unlike mp4/mpg/mpeg)")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "video.mp4").write_bytes(b"")
    (d / "old_recording.avi").write_bytes(b"")
    path, kind = resolve_primary_source(d)
    assert (path.name, kind) == ("video.mp4", "video_as_audio"), (path, kind)
print("OK: mp4 present alongside an avi -> mp4 wins (preferred over avi)")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "video.mp4").write_bytes(b"")
    (d / "video.mpg").write_bytes(b"")
    try:
        resolve_primary_source(d)
        assert False, "should have raised AmbiguousInputError"
    except AmbiguousInputError:
        pass
print("OK: two different direct-audio-usable video files (mp4 + mpg) -> AmbiguousInputError, "
      "not a silent guess")

with _tempfile_fd.TemporaryDirectory() as d:
    d = Path(d)
    (d / "cover.jpg").write_bytes(b"")
    try:
        resolve_primary_source(d)
        assert False, "should have raised NoAudioSourceFoundError"
    except NoAudioSourceFoundError:
        pass
print("OK: nothing usable in the folder -> NoAudioSourceFoundError")

print("\n--- tempo math vs. real reference files ---")
assert abs(beat_duration_ms(300) - 50.0) < 1e-6
assert seconds_to_beat(23.0, 23000, 300) == 0
assert seconds_to_beat_length(0.400, 300) == 8
t = 8300 + 200 * beat_duration_ms(120)
assert seconds_to_beat(t / 1000.0, 8300, 120) == 200
assert seconds_to_beat(48.030, 48030, 382.7) == 0
print("OK: beat math matches all 3 spot-checks")

for t_sec, gap_ms, bpm in [(23.0, 23000, 300), (48.030, 48030, 382.7), (100.456, 8300, 120)]:
    beat = seconds_to_beat(t_sec, gap_ms, bpm)
    round_tripped = beat_to_seconds(beat, gap_ms, bpm)
    # round-trip through an integer beat necessarily loses sub-beat precision --
    # must land back within one beat's own duration, not exactly at t_sec.
    assert abs(round_tripped - t_sec) <= beat_duration_ms(bpm) / 1000.0 + 1e-9, (t_sec, round_tripped)
print("OK: beat_to_seconds round-trips seconds_to_beat within one beat's own duration")

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

print("\n--- lyrics_lookup.reference_matches_transcript: rejects a wrong-song/wrong-language "
      "reference before it's ever trusted (real case: Gaston's lyrics.ovh lookup silently "
      "returned Spanish lyrics for an English song) ---")
from ultrastar_generator.lyrics_lookup import reference_matches_transcript
matching_words = [Word(text=w, start=float(i), end=float(i) + 0.5, confidence=0.9)
                   for i, w in enumerate(["He", "knows", "his", "way", "in", "the", "dark"])]
assert reference_matches_transcript(ref_lines_test, matching_words) is True
wrong_language_words = [Word(text=w, start=float(i), end=float(i) + 0.5, confidence=0.9)
                         for i, w in enumerate(["quiero", "verte", "otro", "modelo", "patron"])]
assert reference_matches_transcript(ref_lines_test, wrong_language_words) is False
print("OK: right-language reference accepted, wrong-language reference rejected")

print("\n--- lyrics_lookup._fetch_from_lrclib: picks the best candidate by duration closeness, "
      "excludes instrumental/lyric-less candidates, breaks ties toward a synced-lyrics candidate ---")
from ultrastar_generator.lyrics_lookup import _fetch_from_lrclib, fetch_reference_lyrics


class _FakeLRCLIBResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeRequestsModule:
    """Deterministic fake for the `requests` module -- lyrics_lookup.py
    does `import requests` lazily inside each fetch function, so
    installing this in sys.modules before calling is enough."""
    def __init__(self, search_payload=None, search_status=200, ovh_payload=None, ovh_status=200,
                 get_by_id_payload=None, get_by_id_status=200):
        self.search_payload = search_payload
        self.search_status = search_status
        self.ovh_payload = ovh_payload
        self.ovh_status = ovh_status
        self.get_by_id_payload = get_by_id_payload  # a single dict, for /api/get/<id> -- distinct
        self.get_by_id_status = get_by_id_status     # shape from /api/search's list response
        self.urls_requested = []

    def get(self, url, params=None, timeout=None):
        self.urls_requested.append(url)
        if "lrclib.net/api/get/" in url:
            return _FakeLRCLIBResponse(self.get_by_id_payload, self.get_by_id_status)
        if "lrclib.net" in url:
            return _FakeLRCLIBResponse(self.search_payload, self.search_status)
        return _FakeLRCLIBResponse(self.ovh_payload, self.ovh_status)


lrclib_candidates = [
    {"trackName": "Gaston", "artistName": "Beauty and the Beast", "duration": 60,
     "instrumental": True, "plainLyrics": None, "syncedLyrics": None},  # excluded: instrumental
    {"trackName": "Gaston", "artistName": "Beauty and the Beast", "duration": 300,
     "instrumental": False, "plainLyrics": "way off duration", "syncedLyrics": None},  # far duration
    {"trackName": "Gaston", "artistName": "Beauty and the Beast", "duration": 178,
     "instrumental": False, "plainLyrics": "close duration, no sync", "syncedLyrics": None},
    {"trackName": "Gaston", "artistName": "Beauty and the Beast", "duration": 179,
     "instrumental": False, "plainLyrics": "close duration, synced",
     "syncedLyrics": "[00:01.00]line one\n[00:05.00]line two"},
]
_sys.modules["requests"] = _FakeRequestsModule(search_payload=lrclib_candidates)
best = _fetch_from_lrclib("Beauty and the Beast", "Gaston", duration_sec=180.0)
assert best is not None and best.source == "lrclib"
assert best.plain_lyrics == "close duration, synced", best.plain_lyrics
assert best.synced_lyrics == "[00:01.00]line one\n[00:05.00]line two", best.synced_lyrics
print("OK: correct candidate chosen among instrumental/far-duration/synced-tiebreak options")

print("\n--- lyrics_lookup.search_lrclib: returns ALL raw candidates unfiltered (for the manual search UI) ---")
from ultrastar_generator.lyrics_lookup import search_lrclib, LrcLibCandidate
_sys.modules["requests"] = _FakeRequestsModule(search_payload=lrclib_candidates)
all_candidates = search_lrclib("Beauty and the Beast", "Gaston")
assert len(all_candidates) == 4, len(all_candidates)  # including the instrumental one
assert all(isinstance(c, LrcLibCandidate) for c in all_candidates)
assert any(c.instrumental for c in all_candidates), "instrumental candidate must NOT be filtered out here"
assert all_candidates[3].synced_lyrics == "[00:01.00]line one\n[00:05.00]line two"
print("OK: search_lrclib returns every candidate as-is, including the instrumental one, for a human to browse")

print("\n--- lyrics_lookup.search_lrclib: q= does a broader free-text search INSTEAD of artist/title ---")


class _ParamRecordingRequestsModule:
    """Records exactly what params were sent, to confirm q= is used alone
    (not combined with artist_name/track_name) when given."""
    def __init__(self, payload):
        self.payload = payload
        self.last_params = None

    def get(self, url, params=None, timeout=None):
        self.last_params = params
        return _FakeLRCLIBResponse(self.payload)


fake_q_module = _ParamRecordingRequestsModule(lrclib_candidates[:1])
_sys.modules["requests"] = fake_q_module
q_results = search_lrclib(q="some broad free-text query")
assert fake_q_module.last_params == {"q": "some broad free-text query"}, fake_q_module.last_params
assert len(q_results) == 1
print("OK: search_lrclib(q=...) sends only 'q', not artist_name/track_name")

fake_at_module = _ParamRecordingRequestsModule(lrclib_candidates[:1])
_sys.modules["requests"] = fake_at_module
search_lrclib("Some Artist", "Some Title")
assert fake_at_module.last_params == {"artist_name": "Some Artist", "track_name": "Some Title"}, \
    fake_at_module.last_params
print("OK: search_lrclib(artist, title) without q still sends artist_name/track_name as before")
del _sys.modules["requests"]

print("\n--- lyrics_lookup._fetch_from_lrclib: on_ambiguous callback (GUI disambiguation path) ---")
# Dedicated fixture (config.LRCLIB_DURATION_TOLERANCE_SEC == 60.0, so the
# ambiguity filter's 3x-tolerance cutoff is 180s): duration_sec=100 makes
# the 350s candidate's diff (250s) clearly exceed 180s -> excluded from
# "real" candidates, while the two ~100s candidates (diff 5s/10s) stay in.
ambiguous_candidates = [
    {"trackName": "Song", "artistName": "Artist", "duration": 30,
     "instrumental": True, "plainLyrics": None, "syncedLyrics": None},  # excluded: instrumental
    {"trackName": "Song", "artistName": "Artist", "duration": 350,
     "instrumental": False, "plainLyrics": "wildly different recording", "syncedLyrics": None},  # excluded: duration
    {"trackName": "Song", "artistName": "Artist", "duration": 105,
     "instrumental": False, "plainLyrics": "candidate A", "syncedLyrics": None},
    {"trackName": "Song", "artistName": "Artist", "duration": 110,
     "instrumental": False, "plainLyrics": "candidate B", "syncedLyrics": None},
]
_sys.modules["requests"] = _FakeRequestsModule(search_payload=ambiguous_candidates)
seen_real_candidates = []


def _pick_second(candidates):
    seen_real_candidates.append(candidates)
    return candidates[1]  # deliberately NOT the auto-pick winner (candidate A scores higher: closer duration)


chosen = _fetch_from_lrclib("Artist", "Song", duration_sec=100.0, on_ambiguous=_pick_second)
assert len(seen_real_candidates) == 1
real_seen = seen_real_candidates[0]
assert len(real_seen) == 2, [c.plain_lyrics for c in real_seen]
assert {c.plain_lyrics for c in real_seen} == {"candidate A", "candidate B"}
assert chosen is not None and chosen.plain_lyrics == "candidate B"
print("OK: on_ambiguous is offered only the filtered 'real' candidates (instrumental and wildly-off-duration "
      "both excluded), and its choice is used directly -- even when it's NOT what automatic scoring would pick")

_sys.modules["requests"] = _FakeRequestsModule(search_payload=ambiguous_candidates)


def _decline(candidates):
    return None  # user cancelled the popup


declined = _fetch_from_lrclib("Artist", "Song", duration_sec=100.0, on_ambiguous=_decline)
assert declined is not None and declined.plain_lyrics == "candidate A", declined
print("OK: on_ambiguous returning None (user cancelled) falls through to the normal automatic pick")

# A single-real-candidate case must NOT invoke on_ambiguous at all -- no ambiguity to resolve.
_sys.modules["requests"] = _FakeRequestsModule(search_payload=[ambiguous_candidates[0], ambiguous_candidates[2]])
was_called = []
_fetch_from_lrclib("Artist", "Song", duration_sec=105.0, on_ambiguous=lambda c: was_called.append(c) or None)
assert not was_called, "on_ambiguous must not fire when there's only one real candidate"
print("OK: on_ambiguous is never invoked when there's only one real (non-instrumental) candidate")
del _sys.modules["requests"]

print("\n--- lyrics_lookup.fetch_reference_lyrics: falls back to lyrics.ovh when LRCLIB has nothing ---")
_sys.modules["requests"] = _FakeRequestsModule(
    search_payload=[], ovh_payload={"lyrics": "fallback lyrics text"},
)
fallback_result = fetch_reference_lyrics("Some Artist", "Some Title", duration_sec=120.0)
assert fallback_result is not None and fallback_result.source == "lyrics.ovh", fallback_result
assert fallback_result.plain_lyrics == "fallback lyrics text", fallback_result
print("OK: empty LRCLIB search result correctly fell back to lyrics.ovh")
del _sys.modules["requests"]

print("\n--- lyrics_lookup: id threads through search_lrclib, and fetch_lrclib_by_id fetches directly ---")
from ultrastar_generator.lyrics_lookup import fetch_lrclib_by_id
id_candidates = [
    {"id": 111, "trackName": "Song", "artistName": "Artist", "duration": 100,
     "instrumental": False, "plainLyrics": "words", "syncedLyrics": None},
]
_sys.modules["requests"] = _FakeRequestsModule(search_payload=id_candidates)
searched = search_lrclib("Artist", "Song")
assert len(searched) == 1 and searched[0].id == 111, searched
print("OK: search_lrclib now captures LRCLIB's own numeric id (previously silently discarded)")

_sys.modules["requests"] = _FakeRequestsModule(get_by_id_payload={
    "id": 37066985, "trackName": "When You're Good to Mama", "artistName": "Taye Diggs/Queen Latifah",
    "albumName": "Chicago", "duration": 200.0, "instrumental": False,
    "plainLyrics": "words", "syncedLyrics": "[00:01.00]line one",
})
by_id = fetch_lrclib_by_id(37066985)
assert by_id is not None and by_id.id == 37066985 and by_id.track_name == "When You're Good to Mama", by_id
assert by_id.synced_lyrics == "[00:01.00]line one", by_id
print("OK: fetch_lrclib_by_id fetches ONE specific entry directly, bypassing search/scoring")

_sys.modules["requests"] = _FakeRequestsModule(get_by_id_status=404, get_by_id_payload=None)
missing = fetch_lrclib_by_id(999999999)
assert missing is None, missing
print("OK: fetch_lrclib_by_id returns None on a non-200 response (bad/missing id), never raises")
del _sys.modules["requests"]

print("\n--- mxl_lrc_generator: MXL for pitch + LRC line anchors + ASR word placement ---")
from ultrastar_generator.mxl_lrc_generator import (
    MxlWord, assign_words_to_lines, place_words_via_asr, build_syllables,
    MxlLrcQuality, config as mxl_lrc_config,
)
from ultrastar_generator.models import Word as _Word

# A tiny two-line "song": line 0 "hello world", line 1 "good bye now".
mlg_words = [
    MxlWord(text="hello", norm="hello", offset=0.0, syllables=[(0.0, 1.0, 64, "hello")]),
    MxlWord(text="world", norm="world", offset=1.0, syllables=[(1.0, 1.0, 65, "world")]),
    MxlWord(text="good", norm="good", offset=4.0, syllables=[(4.0, 1.0, 67, "good")]),
    MxlWord(text="bye", norm="bye", offset=5.0, syllables=[(5.0, 1.0, 69, "bye")]),
    MxlWord(text="now", norm="now", offset=6.0, syllables=[(6.0, 1.0, 71, "now")]),
]
mlg_lrc_lines = [(10.0, "hello world"), (20.0, "good bye now")]

word_lines = assign_words_to_lines(mlg_words, mlg_lrc_lines)
assert word_lines == [0, 0, 1, 1, 1], word_lines
print("OK: assign_words_to_lines correctly tags each word with its own LRC line index")

# ASR confidently catches "hello"/"world"/"bye" at real times close to (but not
# exactly at) the printed line starts; "good" and "now" are missing from ASR
# entirely (must fall back to proportional placement within their own line).
mlg_asr = [
    _Word(text="hello", start=10.2, end=10.5),
    _Word(text="world", start=10.6, end=10.9),
    _Word(text="bye", start=20.5, end=20.8),
]
starts, ends, quality = place_words_via_asr(mlg_words, word_lines, mlg_lrc_lines, mlg_asr)
assert starts[0] == 10.2 and starts[1] == 10.6, starts  # ASR-placed
assert starts[3] == 20.5, starts                         # ASR-placed ("bye")
assert 20.0 <= starts[2] <= starts[3], starts             # "good" fell back, proportional, before "bye"
assert starts[4] >= starts[3], starts                     # "now" fell back, still after "bye"
assert quality.n_asr_placed == 3 and quality.n_fallback == 2, quality
assert quality.asr_placement_rate == 3 / 5
# ENDs for ASR-matched words use the ASR's OWN reported duration directly --
# NOT stretched to the next word's start (the real "hen."/3.1s, "The"/7.1s
# bug this was built to fix).
assert abs(ends[0] - 10.5) < 1e-9, ends[0]   # "hello" keeps its own 0.3s ASR duration
assert abs(ends[1] - 10.9) < 1e-9, ends[1]   # "world" keeps its own 0.3s ASR duration, doesn't reach "good" (20.0)
print("OK: place_words_via_asr uses real ASR timestamps AND durations where confidently matched (never "
      "stretching a word across what should be a real rest), falls back to MXL-note-value/local-tempo "
      "estimated placement and duration otherwise")

# Confidence gating: a text match with LOW confidence must be treated as no
# match at all (real case this was built for: a 0.003-confidence match had a
# genuinely wrong timestamp, independent of anything else in the pipeline).
low_conf_asr = [
    _Word(text="hello", start=10.2, end=10.5, confidence=0.9),
    _Word(text="world", start=10.6, end=10.9, confidence=0.05),  # text matches, but confidence too low to trust
]
lc_starts, lc_ends, lc_quality = place_words_via_asr(mlg_words[:2], [0, 0], mlg_lrc_lines, low_conf_asr)
assert lc_starts[0] == 10.2, lc_starts             # trusted (high confidence)
assert lc_starts[1] != 10.6, lc_starts             # NOT trusted -- fell back instead of using the low-confidence match
assert lc_quality.n_asr_placed == 1 and lc_quality.n_fallback == 1, lc_quality
print("OK: a text match below MXL_LRC_MIN_ASR_WORD_CONFIDENCE is treated as unmatched, not trusted blindly")

# Non-monotonic clamp: a deliberately out-of-order ASR match must not produce
# a backward jump in the final output.
bad_asr = [
    _Word(text="hello", start=10.2, end=10.5),
    _Word(text="world", start=9.6, end=9.9),  # earlier than "hello" -- wrong/out of order (still inside the window)
]
bad_starts, bad_ends, bad_quality = place_words_via_asr(mlg_words[:2], [0, 0], mlg_lrc_lines, bad_asr)
assert bad_starts[1] >= bad_starts[0], bad_starts
assert bad_quality.non_monotonic_fix_count == 1, bad_quality
print("OK: an out-of-order ASR match gets clamped to non-decreasing order, and counted for the quality gate")

syllables_out = build_syllables(mlg_words, starts, ends, word_lines)
assert len(syllables_out) == 5
assert syllables_out[0].line_id == 0 and syllables_out[2].line_id == 1
assert all(syllables_out[i].start <= syllables_out[i + 1].start for i in range(4))
print("OK: build_syllables tags line_id from assign_words_to_lines and produces monotonic syllable starts")

print("\n--- mxl_lrc_generator: quality gate correctly rejects a wrong-recording-style result ---")
# Mirrors the real BATB/Stars failure this session found: a candidate that
# passes duration+content filtering but whose LRC line timings don't
# correspond to what our own audio actually says -- ASR barely matches.
n_words_gate = 10
low_quality = MxlLrcQuality(n_words=n_words_gate, n_asr_placed=2, n_fallback=8, non_monotonic_fix_count=0)
assert low_quality.asr_placement_rate < mxl_lrc_config.MXL_LRC_MIN_ASR_PLACEMENT_RATE
high_quality = MxlLrcQuality(n_words=n_words_gate, n_asr_placed=9, n_fallback=1, non_monotonic_fix_count=0)
assert high_quality.asr_placement_rate >= mxl_lrc_config.MXL_LRC_MIN_ASR_PLACEMENT_RATE
print("OK: MxlLrcQuality.asr_placement_rate correctly separates a low-confidence (wrong-recording-style) "
      "result from a high-confidence one, against the real shipped threshold")

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

print("\n--- BUG REGRESSION: a long silence gap WITHIN a single confirmed reference line no longer "
      "forces a spurious mid-line break (real case: \"Just a little change\" was being split into "
      "\"Just a little\" / \"change\" because of an audible pause before \"change\", even though "
      "both words shared the same reference line_id) ---")
same_line_gap_syls = [
    Syllable("Just", 0.0, 0.2, 4, True, line_id=0),
    Syllable(" a", 0.2, 0.4, 4, True, line_id=0),
    Syllable(" little", 0.4, 0.6, 4, True, line_id=0),
    # long gap before the next word (well over MIN_LINE_GAP_SEC's 0.35s),
    # but SAME line_id -- must NOT break.
    Syllable(" change", 1.5, 2.0, 4, True, line_id=0),
]
same_line_entries = build_lines(same_line_gap_syls)
assert all(type(e).__name__ == "Syllable" for e in same_line_entries), \
    [type(e).__name__ for e in same_line_entries]
print("OK: no break inserted despite the long gap, since line_id confirmed it's still one line:",
      [type(e).__name__ for e in same_line_entries])

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

print("\n--- zone-boundary snapping (EXPERIMENTAL, config.ENABLE_ZONE_BOUNDARY_SNAP): a real pass-1 "
      "note onset near the raw ASR-midpoint boundary reassigns a note to the correct word when enabled, "
      "and is a no-op (matches current default behavior) when disabled -- reproduces the sleeping_beauty_"
      "wonder 'I' bug shape: a real sustained note starts before the crude ASR-gap-midpoint suggests ---")
snap_words = [
    Word(text="Odd", start=0.0, end=2.0, confidence=0.9, line_id=20),
    Word(text="I", start=6.0, end=6.5, confidence=0.9, line_id=21),
]
# Raw boundary = midpoint(2.0, 6.0) = 4.0. A real note starts at 3.6s (the
# genuine, pass-1-detected onset of "I"'s singing) -- its own MIDPOINT
# (3.85) falls on the WRONG side of the raw 4.0 boundary, so without
# snapping it's misassigned to "Odd" instead of "I".
snap_notes = [
    NoteEvent(start=0.0, end=1.0, pitch=5),
    NoteEvent(start=1.0, end=2.0, pitch=5),
    NoteEvent(start=3.6, end=4.1, pitch=7),   # the ambiguous one -- real onset for "I"
    NoteEvent(start=4.6, end=6.3, pitch=7),   # unambiguously "I" either way
]


def _counts_by_word(words, syllables):
    counts = {}
    word_idx = -1
    for s in syllables:
        if s.is_word_start:
            word_idx += 1
        counts[words[word_idx].text] = counts.get(words[word_idx].text, 0) + 1
    return counts


unsnapped, _ = align_words_to_notes(snap_words, snap_notes, np.zeros(16000), 16000, snap_boundaries=False)
unsnapped_counts = _counts_by_word(snap_words, unsnapped)
assert unsnapped_counts == {"Odd": 3, "I": 1}, unsnapped_counts
print("OK: snapping disabled (default) -- unchanged from current behavior:", unsnapped_counts)

snapped, _ = align_words_to_notes(snap_words, snap_notes, np.zeros(16000), 16000,
                                   snap_boundaries=True, snap_radius_sec=0.5)
snapped_counts = _counts_by_word(snap_words, snapped)
assert snapped_counts == {"Odd": 2, "I": 2}, snapped_counts
print("OK: snapping enabled -- boundary snapped to the real 3.6s note onset, "
      "correctly reassigning it to \"I\":", snapped_counts)

# --- ambiguity guard: TWO onset candidates in range -> no snap (can't tell which is "the" boundary) ---
ambig_notes = [
    NoteEvent(start=0.0, end=1.0, pitch=5),
    NoteEvent(start=1.0, end=2.0, pitch=5),
    NoteEvent(start=3.6, end=4.1, pitch=7),
    NoteEvent(start=4.3, end=6.3, pitch=7),   # a SECOND onset candidate within 0.5s of the 4.0 boundary
]
ambig_snapped, _ = align_words_to_notes(snap_words, ambig_notes, np.zeros(16000), 16000,
                                         snap_boundaries=True, snap_radius_sec=0.5)
ambig_counts = _counts_by_word(snap_words, ambig_snapped)
assert ambig_counts == {"Odd": 3, "I": 1}, ambig_counts
print("OK: two competing onset candidates in range -> left the raw ASR-midpoint boundary alone "
      "(ambiguous, not confidently correctable):", ambig_counts)

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
    # 0: no reference at all; recheck disagrees with current text -> kept
    # (an isolated recheck is a less reliable signal than the original
    # full-context ASR text, and there's no reference to confirm the
    # disagreement either way -- see verification.py's _resolve()).
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
assert new_words[0].text == "mumbled", new_words[0]         # untouched -- no reference to confirm the recheck
assert new_words[1].text == "Multitudinous", new_words[1]  # reference's own text, not the recheck's raw casing
assert new_words[2].text == "Stars", new_words[2]           # forced to reference despite no confirmation
assert new_words[3].text == "Stars", new_words[3]           # untouched -- already matched reference
assert new_words[4].text == "echo", new_words[4]            # untouched -- recheck agreed
assert [r.replaced for r in verify_results] == [False, True, True, False, False], verify_results
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

print("\n--- musicxml_reference.apply_musicxml_reference: calibrates at the PITCH-CLASS level "
      "(absorbs a per-song transposition) and corrects only where confident ---")
import tempfile as _tempfile
from ultrastar_generator.musicxml_reference import apply_musicxml_reference

_PC_TO_STEP_ALTER = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]

def _make_mxl(words_midi, path):
    notes_xml = ""
    for text, midi in words_midi:
        step, alter = _PC_TO_STEP_ALTER[midi % 12]
        octave = midi // 12 - 1
        alter_xml = f"<alter>{alter}</alter>" if alter else ""
        notes_xml += (
            f'<note><pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>'
            f'<duration>4</duration><type>quarter</type>'
            f'<lyric><syllabic>single</syllabic><text>{text}</text></lyric></note>'
        )
    xml = (
        '<?xml version="1.0"?><score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>'
        f'<part id="P1"><measure number="1">{notes_xml}</measure></part></score-partwise>'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)

# Our own detection has these 6 words a semitone flat of the "true" pitch
# a MusicXML reference (transposed up 3 semitones from OUR octave/key
# convention) would show -- calibration should land on pitch-class +3,
# and correct the one word ("qux") that's off by more than just the
# transposition.
our_syllables = [
    Syllable(text="foo", start=0.0, end=0.5, midi_note=0, is_word_start=True),   # 60 abs
    Syllable(text="bar", start=0.5, end=1.0, midi_note=2, is_word_start=True),   # 62 abs
    Syllable(text="baz", start=1.0, end=1.5, midi_note=4, is_word_start=True),   # 64 abs
    Syllable(text="qux", start=1.5, end=2.0, midi_note=5, is_word_start=True),   # 65 abs -- will be WRONG
    Syllable(text="quux", start=2.0, end=2.5, midi_note=7, is_word_start=True),  # 67 abs
    Syllable(text="corge", start=2.5, end=3.0, midi_note=9, is_word_start=True), # 69 abs
]
# MXL: same words, all transposed +3 from our octave/key EXCEPT "qux",
# which disagrees with our pitch by a real (non-calibration) error too.
mxl_words_midi = [
    ("foo", 63), ("bar", 65), ("baz", 67), ("qux", 70), ("quux", 70), ("corge", 72),
]
with _tempfile.TemporaryDirectory() as tmpdir:
    mxl_path = f"{tmpdir}/fake_song.musicxml"
    _make_mxl(mxl_words_midi, mxl_path)
    corrected, mxl_stats = apply_musicxml_reference(
        our_syllables, mxl_path, min_calibration_samples=4, verbose=True,
    )
    assert mxl_stats.skipped_reason is None, mxl_stats.skipped_reason
    assert mxl_stats.calibration_offset == 3, mxl_stats.calibration_offset
    corrected_texts = {c.text for c in mxl_stats.corrections}
    assert corrected_texts == {"qux"}, corrected_texts
    # qux: our_pc=(65+60)%12=5, target_pc=(70-3)%12=7 -> diff=+2 -> new=65+2=67
    assert corrected[3].midi_note == 7, corrected[3]
    # everything else (already correct once transposition is accounted for) untouched
    for i in (0, 1, 2, 4, 5):
        assert corrected[i].midi_note == our_syllables[i].midi_note, (i, corrected[i])
    print("OK: calibrated to +3 semitones (pitch-class) and corrected only the genuinely "
          "wrong word, leaving already-correct-after-calibration words untouched:",
          [(c.text, c.old_pitch, c.new_pitch) for c in mxl_stats.corrections])

    # --- too few matches: should skip, not guess ---
    _, few_stats = apply_musicxml_reference(
        our_syllables[:2], mxl_path, min_calibration_samples=4, verbose=False,
    )
    assert few_stats.skipped_reason is not None
    assert few_stats.corrections == []
    print("OK: too few matched notes correctly skipped calibration:", few_stats.skipped_reason)

    # --- force_calibration: a genuinely ambiguous population (4 different
    # offsets, each covering exactly 1/4 of matches -- below both the
    # full-population AND high-confidence-subset bars either way) is
    # skipped normally, but force_calibration=True applies the best
    # available offset anyway rather than giving up -- built for songs
    # where our own pass-1 pitch is confirmed unreliable for acoustic
    # reasons, so even a weak MXL-based calibration beats none at all.
    ambiguous_our = [
        Syllable(text=w, start=float(i), end=float(i) + 0.4, midi_note=0, is_word_start=True)
        for i, w in enumerate(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"])
    ]
    # offsets in order: 2,5,7,9,2,5,7,9 -- each covers exactly 2/8 = 25%,
    # below both MUSICXML_MIN_CALIBRATION_CONFIDENCE (50%) and
    # _HIGH_CONF_SUBSET (40%), for the full population AND the top half.
    ambiguous_mxl = list(zip(
        ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"],
        [62, 65, 67, 69, 62, 65, 67, 69],
    ))
    with _tempfile.TemporaryDirectory() as tmpdir2:
        amb_path = f"{tmpdir2}/ambiguous_song.musicxml"
        _make_mxl(ambiguous_mxl, amb_path)

        _, normal_stats = apply_musicxml_reference(
            ambiguous_our, amb_path, min_calibration_samples=4, force_calibration=False, verbose=False,
        )
        assert normal_stats.skipped_reason is not None, "expected a skip without force_calibration"
        assert normal_stats.corrections == []

        forced, forced_stats = apply_musicxml_reference(
            ambiguous_our, amb_path, min_calibration_samples=4, force_calibration=True, verbose=True,
        )
        assert forced_stats.skipped_reason is None, forced_stats.skipped_reason
        assert forced_stats.calibration_offset is not None
        assert len(forced_stats.corrections) > 0, "expected force_calibration to apply SOME correction"
        print(f"OK: normal mode skipped an ambiguous population "
              f"({normal_stats.skipped_reason}), force_calibration applied offset "
              f"{forced_stats.calibration_offset:+d} anyway and corrected "
              f"{len(forced_stats.corrections)}/8 syllables")

print("\n--- musicxml_reference.apply_musicxml_references (plural): applies multiple reference "
      "files SEQUENTIALLY so coverage accumulates across files with different, only partly "
      "overlapping lyric coverage -- real case: Once Upon A Dream, two arrangements covering "
      "different fractions of the song ---")
from ultrastar_generator.musicxml_reference import apply_musicxml_references

multi_syllables = [
    Syllable(text="alpha", start=0.0, end=0.5, midi_note=0, is_word_start=True),    # 60
    Syllable(text="bravo", start=0.5, end=1.0, midi_note=2, is_word_start=True),    # 62
    Syllable(text="charlie", start=1.0, end=1.5, midi_note=6, is_word_start=True),  # 66 -- WRONG (should be 64)
    Syllable(text="delta", start=1.5, end=2.0, midi_note=5, is_word_start=True),    # 65
    Syllable(text="echo", start=2.0, end=2.5, midi_note=7, is_word_start=True),     # 67
    Syllable(text="foxtrot", start=2.5, end=3.0, midi_note=6, is_word_start=True),  # 66 -- WRONG (should be 69)
    Syllable(text="golf", start=3.0, end=3.5, midi_note=11, is_word_start=True),    # 71
]
with _tempfile.TemporaryDirectory() as tmpdir2:
    # File 1 covers only the first 4 words, transposed +3.
    path1 = f"{tmpdir2}/arrangement_one.musicxml"
    _make_mxl([("alpha", 63), ("bravo", 65), ("charlie", 67), ("delta", 68)], path1)
    # File 2 covers only the LAST 3 words (no overlap with file 1), transposed +5.
    path2 = f"{tmpdir2}/arrangement_two.musicxml"
    _make_mxl([("echo", 72), ("foxtrot", 74), ("golf", 76)], path2)

    corrected2, stats_list = apply_musicxml_references(
        multi_syllables, [path1, path2], min_calibration_samples=3, verbose=False,
    )
    assert len(stats_list) == 2, stats_list
    assert stats_list[0].calibration_offset == 3, stats_list[0].calibration_offset
    assert stats_list[1].calibration_offset == 5, stats_list[1].calibration_offset
    corrected_texts = {c.text for s in stats_list for c in s.corrections}
    assert corrected_texts == {"charlie", "foxtrot"}, corrected_texts
    # charlie: our_pc=(66+60)%12=6, target_pc=(67-3)%12=4 -> diff=-2 -> new=66-2=64
    assert corrected2[2].midi_note == 4, corrected2[2]
    # foxtrot: our_pc=(66+60)%12=6, target_pc=(74-5)%12=9 -> diff=+3 -> new=66+3=69
    assert corrected2[5].midi_note == 9, corrected2[5]
    for i in (0, 1, 3, 4, 6):
        assert corrected2[i].midi_note == multi_syllables[i].midi_note, (i, corrected2[i])
    print("OK: two non-overlapping reference files each calibrated independently and corrected "
          "only their own genuinely-wrong word, coverage accumulating across both files:",
          [(c.text, c.old_pitch, c.new_pitch) for s in stats_list for c in s.corrections])

print("\n--- lrc_timing.apply_lrc_timing_check: calibrates a per-song TIME offset against LRCLIB "
      "synced lyrics (mirrors musicxml_reference's pitch calibration, but for line start time), "
      "flags a line that disagrees even after calibration -- DIAGNOSTIC ONLY, never moves anything ---")
from ultrastar_generator.lrc_timing import apply_lrc_timing_check, parse_lrc

lrc_syllables = [
    Syllable(text="hello", start=10.0, end=10.4, midi_note=0, is_word_start=True, line_id=0),
    Syllable(text="world", start=10.5, end=10.9, midi_note=0, is_word_start=True, line_id=0),
    Syllable(text="how", start=15.0, end=15.4, midi_note=0, is_word_start=True, line_id=1),
    Syllable(text="are", start=15.5, end=15.9, midi_note=0, is_word_start=True, line_id=1),
    Syllable(text="you", start=16.0, end=16.4, midi_note=0, is_word_start=True, line_id=1),
    # "goodbye now" -- deliberately drifted far beyond what calibration explains
    Syllable(text="goodbye", start=25.0, end=25.4, midi_note=0, is_word_start=True, line_id=2),
    Syllable(text="now", start=25.5, end=25.9, midi_note=0, is_word_start=True, line_id=2),
    Syllable(text="see", start=30.0, end=30.4, midi_note=0, is_word_start=True, line_id=3),
    Syllable(text="you", start=30.5, end=30.9, midi_note=0, is_word_start=True, line_id=3),
    Syllable(text="soon", start=31.0, end=31.4, midi_note=0, is_word_start=True, line_id=3),
    Syllable(text="thanks", start=35.0, end=35.4, midi_note=0, is_word_start=True, line_id=4),
    Syllable(text="a", start=35.5, end=35.9, midi_note=0, is_word_start=True, line_id=4),
    Syllable(text="lot", start=36.0, end=36.4, midi_note=0, is_word_start=True, line_id=4),
]
# LRC lines are consistently 2.0s EARLIER than our own assigned starts,
# except "goodbye now" which is 10.0s earlier (an 8.0s residual after
# the +2.0s calibration is removed -- should get flagged).
lrc_text = (
    "[00:08.00]hello world\n"
    "[00:13.00]how are you\n"
    "[00:15.00]goodbye now\n"
    "[00:28.00]see you soon\n"
    "[00:33.00]thanks a lot\n"
)
parsed_lrc = parse_lrc(lrc_text)
assert parsed_lrc == [
    (8.0, "hello world"), (13.0, "how are you"), (15.0, "goodbye now"),
    (28.0, "see you soon"), (33.0, "thanks a lot"),
], parsed_lrc

lrc_stats = apply_lrc_timing_check(lrc_syllables, lrc_text, verbose=True)
assert lrc_stats.skipped_reason is None, lrc_stats.skipped_reason
assert abs(lrc_stats.calibration_offset_sec - 2.0) < 1e-6, lrc_stats.calibration_offset_sec
assert lrc_stats.n_matched_lines == 5, lrc_stats.n_matched_lines
flagged_texts = {f.text for f in lrc_stats.flags}
assert flagged_texts == {"goodbye"}, flagged_texts
assert abs(lrc_stats.flags[0].delta_sec - 8.0) < 1e-6, lrc_stats.flags[0].delta_sec
# never modifies the syllables themselves -- diagnostic only
assert lrc_syllables[5].start == 25.0, lrc_syllables[5]
print(f"OK: calibrated to {lrc_stats.calibration_offset_sec:+.1f}s and flagged only the genuinely "
      f"drifted line: {[(f.text, f.delta_sec) for f in lrc_stats.flags]}")

# --- too few matched lines: should skip, not guess ---
few_lines_stats = apply_lrc_timing_check(lrc_syllables[:2], lrc_text, verbose=False)
assert few_lines_stats.skipped_reason is not None
assert few_lines_stats.flags == []
print("OK: too few matched lines correctly skipped calibration:", few_lines_stats.skipped_reason)

# --- real per-song DRIFT (not just a constant offset) -- confirmed on real
# audio (stars, tarzan, little_mermaid all showed this, see CLAUDE.md 0k-0m):
# delta = offset + slope*lrc_start, spread widely enough that no single
# 1-second bucket covers the required fraction, so the constant-offset
# tier must fail before the robust drift-fit tier is tried. Also folds in
# a word-level-recall check: line 3's LRC text has one word deliberately
# wrong ("gamma3" instead of "beta3"), which the old whole-line-exact
# match would have dropped entirely -- majority-vote word-level matching
# should still recover it as a candidate. One genuine outlier (a
# wrong-instance-style mismatch) should get flagged without dragging the
# robust fit off course.
drift_syllables = []
for i in range(11):
    t0 = 10.5 * i + 5.0
    drift_syllables.append(Syllable(text=f"alpha{i}", start=t0, end=t0 + 0.4, midi_note=0, is_word_start=True, line_id=200 + i))
    drift_syllables.append(Syllable(text=f"beta{i}", start=t0 + 0.5, end=t0 + 0.9, midi_note=0, is_word_start=True, line_id=200 + i))
outlier_start = 200.0
drift_syllables.append(Syllable(text="alpha11", start=outlier_start, end=outlier_start + 0.4, midi_note=0, is_word_start=True, line_id=211))
drift_syllables.append(Syllable(text="beta11", start=outlier_start + 0.5, end=outlier_start + 0.9, midi_note=0, is_word_start=True, line_id=211))


def _lrc_ts(t):
    mm, ss = int(t) // 60, t - (int(t) // 60) * 60
    return f"[{mm:02d}:{ss:05.2f}]"


drift_lrc_lines = []
for i in range(11):
    words = f"alpha{i} beta{i}" if i != 3 else f"alpha{i} gamma{i}"  # line 3: one word deliberately wrong
    drift_lrc_lines.append(f"{_lrc_ts(10.0 * i)}{words}")
drift_lrc_lines.append(f"{_lrc_ts(110.0)}alpha11 beta11")
drift_lrc_text = "\n".join(drift_lrc_lines) + "\n"

drift_stats = apply_lrc_timing_check(drift_syllables, drift_lrc_text, verbose=True)
assert drift_stats.skipped_reason is None, drift_stats.skipped_reason
assert drift_stats.calibration_kind == "drift", drift_stats.calibration_kind
assert abs(drift_stats.calibration_slope - 0.05) < 0.01, drift_stats.calibration_slope
assert abs(drift_stats.calibration_offset_sec - 5.0) < 0.5, drift_stats.calibration_offset_sec
assert drift_stats.n_matched_lines == 12, drift_stats.n_matched_lines  # includes line 3 via word-level recall
flagged_texts = {f.text for f in drift_stats.flags}
assert flagged_texts == {"alpha11"}, flagged_texts
print(f"OK: constant-offset tier correctly failed on real drift (deltas spread across many buckets), "
      f"robust fit recovered slope={drift_stats.calibration_slope:+.4f}, offset={drift_stats.calibration_offset_sec:+.1f}s, "
      f"flagged only the genuine outlier: {flagged_texts}")

print("\n--- cover_extract: embedded-cover-art extraction (mutagen) ---")
from ultrastar_generator import cover_extract

_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 50
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

assert cover_extract._sniff_image_ext(_JPEG_BYTES) == ".jpg"
assert cover_extract._sniff_image_ext(_PNG_BYTES) == ".png"
assert cover_extract._sniff_image_ext(b"not an image") is None
print("OK: magic-byte sniffing correctly identifies jpg/png and rejects garbage")


class _FakeAPIC:
    def __init__(self, data):
        self.data = data


class _FakeID3Tags:
    def __init__(self, apics):
        self._apics = apics

    def getall(self, key):
        return self._apics if key == "APIC" else []


assert cover_extract._from_id3(_FakeID3Tags([_FakeAPIC(_JPEG_BYTES)])) == _JPEG_BYTES
assert cover_extract._from_id3(_FakeID3Tags([])) is None
print("OK: ID3 APIC extraction reads the first embedded picture's raw bytes")

assert cover_extract._from_mp4({"covr": [_PNG_BYTES]}) == _PNG_BYTES
assert cover_extract._from_mp4({}) is None
print("OK: MP4 'covr' atom extraction works")

from mutagen.flac import Picture as _FlacPicture
import base64 as _base64
_pic = _FlacPicture()
_pic.data = _JPEG_BYTES
_pic.mime = "image/jpeg"
_pic.type = 3
_block = _base64.b64encode(_pic.write()).decode("ascii")
assert cover_extract._from_vorbis_comment_block({"metadata_block_picture": [_block]}) == _JPEG_BYTES
assert cover_extract._from_vorbis_comment_block({}) is None
print("OK: OGG/Opus base64 vorbis-comment picture block round-trips correctly")

with _tempfile.TemporaryDirectory() as d:
    d = Path(d)
    fake_audio = d / "Some Artist - Some Song.mp3"
    fake_audio.write_bytes(b"not a real audio file")  # extraction must fail gracefully, not crash
    result = cover_extract.extract_embedded_cover(fake_audio, d / "out")
    assert result is None, result
print("OK: a file mutagen can't parse at all -> None, never raises")

with _tempfile.TemporaryDirectory() as d:
    d = Path(d)
    fake_audio = d / "Some Artist - Some Song.mp3"
    fake_audio.write_bytes(b"")
    _orig_extract_raw = cover_extract._extract_raw_picture
    cover_extract._extract_raw_picture = lambda p: _JPEG_BYTES  # bypass mutagen's own container validation
    try:
        out_path = cover_extract.extract_embedded_cover(fake_audio, d / "out")
    finally:
        cover_extract._extract_raw_picture = _orig_extract_raw  # restore the real function
    assert out_path is not None and out_path.name == "Some Artist - Some Song [CO].jpg", out_path
    assert out_path.read_bytes() == _JPEG_BYTES
print(f"OK: extracted cover written with find_companions' own [CO] tag convention: {out_path.name}")

print("\n--- output_staging.stage_companions_to_output ---")
from ultrastar_generator.output_staging import stage_companions_to_output

with _tempfile.TemporaryDirectory() as root:
    root = Path(root)
    in_dir = root / "in"
    out_dir = root / "out"
    in_dir.mkdir()
    mp3 = in_dir / "song.mp3"
    mp3.write_bytes(b"mp3-bytes")
    video = in_dir / "song.mp4"
    video.write_bytes(b"video-bytes")
    cover = in_dir / "song [CO].jpg"
    cover.write_bytes(b"cover-bytes")

    staged = stage_companions_to_output(out_dir, mp3_src=mp3, video_src=video, cover_src=cover)
    assert staged.mp3 == "song.mp3" and staged.video == "song.mp4" and staged.cover == "song [CO].jpg"
    assert staged.background is None
    assert (out_dir / "song.mp3").read_bytes() == b"mp3-bytes"
    assert (out_dir / "song.mp4").read_bytes() == b"video-bytes"
    assert (out_dir / "song [CO].jpg").read_bytes() == b"cover-bytes"
print("OK: mp3/video/cover copied into the output folder under their own basenames")

with _tempfile.TemporaryDirectory() as root:
    root = Path(root)
    in_dir = root / "in"
    out_dir = root / "out"
    in_dir.mkdir()
    mp4 = in_dir / "song.mp4"
    mp4.write_bytes(b"mp4-bytes")

    staged = stage_companions_to_output(out_dir, mp3_src=mp4, video_src=mp4)
    assert staged.mp3 == staged.video == "song.mp4"
    assert len(list(out_dir.iterdir())) == 1, "identical mp3/video source must only be copied ONCE"
print("OK: identical mp3_src/video_src (mp4-as-audio case) copied exactly once, both roles reference it")

print("\n--- main.delete_intermediates: scoped to separated/+extracted/, leaves debug files alone ---")
import tempfile as _tempfile_delint
from ultrastar_generator.main import delete_intermediates as _delete_intermediates
with _tempfile_delint.TemporaryDirectory() as tmp:
    work_dir = Path(tmp) / ".ultrastar_work"
    (work_dir / "separated" / "htdemucs" / "song").mkdir(parents=True)
    (work_dir / "separated" / "htdemucs" / "song" / "vocals.wav").write_bytes(b"fake-vocals")
    (work_dir / "extracted").mkdir(parents=True)
    (work_dir / "extracted" / "audio.mp3").write_bytes(b"fake-extracted-audio")
    debug_log = work_dir / "Some Artist - Some Song [DEBUG LOG].txt"
    debug_log.write_text("debug content", encoding="utf-8")
    pass1_debug = work_dir / "Some Artist - Some Song [PASS1 DEBUG].txt"
    pass1_debug.write_text("pass1 debug content", encoding="utf-8")

    _delete_intermediates(work_dir)

    assert not (work_dir / "separated").exists(), "separated/ should be deleted"
    assert not (work_dir / "extracted").exists(), "extracted/ should be deleted"
    assert debug_log.is_file(), "debug log must survive delete_intermediates"
    assert pass1_debug.is_file(), "pass-1 debug file must survive delete_intermediates"
print("OK: delete_intermediates removes separated/+extracted/ but leaves debug files under work_dir alone")

# A work_dir with no separated/extracted subfolders at all must not raise.
with _tempfile_delint.TemporaryDirectory() as tmp:
    empty_work_dir = Path(tmp) / ".ultrastar_work"
    empty_work_dir.mkdir()
    _delete_intermediates(empty_work_dir)  # must not raise
print("OK: delete_intermediates on a work_dir with nothing to delete is a silent no-op")

print("\n--- usdx_parser.parse_usdx_file: round-trips usdx_writer.render_song's own grammar ---")
from ultrastar_generator.usdx_parser import parse_usdx_file, UsdxParseError, ParsedSong

roundtrip_syllables = [
    Syllable(text="Hello", start=1.000, end=1.400, midi_note=3, is_word_start=True),
    Syllable(text="world", start=1.500, end=1.900, midi_note=5, is_word_start=True),
    Syllable(text="a", start=3.000, end=3.100, midi_note=0, is_word_start=True),
    Syllable(text="gain", start=3.100, end=3.500, midi_note=2, is_word_start=False),
]
roundtrip_entries = [
    roundtrip_syllables[0], roundtrip_syllables[1],
    LineBreak(start=2.000, end=3.000),
    roundtrip_syllables[2], roundtrip_syllables[3],
]
roundtrip_song = Song(
    title="Round Trip Test", artist="Test Artist", mp3="test.mp3",
    bpm=200.0, gap_ms=1000, entries=roundtrip_entries,
)
with _tempfile.TemporaryDirectory() as d:
    out_path = Path(d) / "Test Artist - Round Trip Test.txt"
    from ultrastar_generator.usdx_writer import write_song
    write_song(roundtrip_song, out_path)
    parsed = parse_usdx_file(out_path)

assert parsed.title == "Round Trip Test" and parsed.artist == "Test Artist"
assert abs(parsed.bpm - 200.0) < 1e-6 and parsed.gap_ms == 1000
parsed_syllables = [e for e in parsed.entries if isinstance(e, Syllable)]
assert [s.text for s in parsed_syllables] == ["Hello", "world", "a", "gain"]
assert [s.is_word_start for s in parsed_syllables] == [True, True, True, False]
assert [s.midi_note for s in parsed_syllables] == [3, 5, 0, 2]
for orig, rt in zip(roundtrip_syllables, parsed_syllables):
    # round-tripped through integer beats -- must land within one beat's
    # own duration at this BPM (50ms @ 200bpm*4), not bit-exact.
    assert abs(orig.start - rt.start) < 0.06, (orig, rt)
line_breaks = [e for e in parsed.entries if isinstance(e, LineBreak)]
assert len(line_breaks) == 1 and abs(line_breaks[0].start - 2.0) < 0.06
print("OK: parse_usdx_file round-trips render_song's own output exactly (text, pitch, word-starts, "
      "timing within one beat's quantization)")

print("\n--- usdx_parser.parse_usdx_file: a line's first word with NO leading space (a real, valid "
      "external-file convention -- the line break itself already marks the word boundary) still "
      "gets is_word_start=True, not silently merged onto the previous line's last word ---")
with _tempfile.TemporaryDirectory() as d:
    real_world_path = Path(d) / "Test Artist - Real World Convention.txt"
    # Mirrors a real confirmed case: "- 48" / ": 57 2 4 Keep" (no leading
    # space on "Keep", even though it's a genuine new word right after the
    # line break) -- external authoring tools commonly omit the redundant
    # leading space here, unlike this project's own render_song (which
    # always includes it, see usdx_writer.py).
    real_world_path.write_text(
        "#TITLE:Real World Convention\n#ARTIST:Test Artist\n#BPM:200\n#GAP:1000\n"
        ": 0 4 3 idol\n"  # single-syllable word ending a line, no leading space needed on ITS side
        "- 4\n"
        ": 4 2 3 Keep\n"      # first syllable of the NEXT line -- no leading space
        ": 6 2 3 ing\n"       # continuation syllable of "Keeping"
        ": 8 2 3  you\n"      # next word, correctly has a leading space
        "E\n",
        encoding="utf-8",
    )
    parsed_rw = parse_usdx_file(real_world_path)
    rw_syllables = [e for e in parsed_rw.entries if isinstance(e, Syllable)]
    assert [s.text for s in rw_syllables] == ["idol", "Keep", "ing", "you"], rw_syllables
    assert [s.is_word_start for s in rw_syllables] == [True, True, False, True], \
        [s.is_word_start for s in rw_syllables]
print("OK: 'Keep' (first syllable right after a line break, no leading space) correctly parses as "
      "is_word_start=True, not merged into 'idol' from the previous line")

with _tempfile.TemporaryDirectory() as d:
    garbage_path = Path(d) / "garbage.txt"
    garbage_path.write_text("#TITLE:Bad\n#ARTIST:Bad\n#BPM:200\n#GAP:0\nthis is not a valid note line\nE\n",
                             encoding="utf-8")
    try:
        parse_usdx_file(garbage_path)
        assert False, "should have raised UsdxParseError"
    except UsdxParseError:
        pass
print("OK: a structurally invalid file raises UsdxParseError (fails closed, never a partial parse)")

with _tempfile.TemporaryDirectory() as d:
    missing_bpm_path = Path(d) / "missing_bpm.txt"
    missing_bpm_path.write_text("#TITLE:Bad\n#ARTIST:Bad\n#GAP:0\n: 0 1 0 hi\nE\n", encoding="utf-8")
    try:
        parse_usdx_file(missing_bpm_path)
        assert False, "should have raised UsdxParseError"
    except UsdxParseError:
        pass
print("OK: a missing required tag (#BPM) raises UsdxParseError")

print("\n--- verify_existing_song: compares an existing .txt's pitch/timing against a fresh pipeline run ---")
from ultrastar_generator.verify_existing_song import verify_existing_song

def _mk_word_syllables(words_start_pitch, start_offset=0.0):
    """words_start_pitch: [(text, start_sec, midi_note), ...] -- each its own word-start syllable."""
    return [Syllable(text=t, start=s + start_offset, end=s + start_offset + 0.3, midi_note=p, is_word_start=True)
            for t, s, p in words_start_pitch]

base_words = [(f"word{i}", float(i), (i * 3) % 12) for i in range(15)]

# Case 1: existing file matches the fresh run closely -> PASS
existing_ok = ParsedSong(title="T", artist="A", bpm=200.0, gap_ms=0,
                          entries=_mk_word_syllables(base_words))
fresh_ok = _mk_word_syllables(base_words, start_offset=0.02)  # trivial jitter, well within tolerance
result = verify_existing_song(existing_ok, fresh_ok, min_matched=10, verbose=True)
assert result.verdict == "PASS", result
print(f"OK: closely-matching existing file -> PASS ({result.pitch_class_accuracy:.0%} pitch, "
      f"{result.timing_within_tolerance_pct:.0%} timing)")

# Case 2: existing file's pitches are all wrong (shifted by a non-multiple-of-12 amount) -> PROBLEMS_FOUND
wrong_pitch_words = [(t, s, (p + 5) % 12) for t, s, p in base_words]
existing_wrong_pitch = ParsedSong(title="T", artist="A", bpm=200.0, gap_ms=0,
                                   entries=_mk_word_syllables(wrong_pitch_words))
result = verify_existing_song(existing_wrong_pitch, fresh_ok, min_matched=10, verbose=True)
assert result.verdict == "PROBLEMS_FOUND", result
assert result.pitch_class_accuracy < 0.5, result.pitch_class_accuracy
print(f"OK: existing file with wrong pitches -> PROBLEMS_FOUND ({result.pitch_class_accuracy:.0%} pitch accuracy)")

# Case 3: existing file's timing is way off (several seconds late) -> PROBLEMS_FOUND
late_words = [(t, s + 5.0, p) for t, s, p in base_words]
existing_late = ParsedSong(title="T", artist="A", bpm=200.0, gap_ms=0,
                            entries=_mk_word_syllables(late_words))
result = verify_existing_song(existing_late, fresh_ok, min_matched=10, verbose=True)
assert result.verdict == "PROBLEMS_FOUND", result
assert result.timing_within_tolerance_pct < 0.5, result.timing_within_tolerance_pct
print(f"OK: existing file with badly-off timing -> PROBLEMS_FOUND "
      f"({result.timing_within_tolerance_pct:.0%} timing agreement)")

# Case 4: too few words in common -> COULD_NOT_VERIFY, never a false PASS
tiny_existing = ParsedSong(title="T", artist="A", bpm=200.0, gap_ms=0,
                            entries=_mk_word_syllables(base_words[:3]))
result = verify_existing_song(tiny_existing, fresh_ok, min_matched=10, verbose=False)
assert result.verdict == "COULD_NOT_VERIFY", result
print(f"OK: too few matched words -> COULD_NOT_VERIFY, not a false PASS ({result.reason})")

print("\n--- youtube_source.download_youtube_source (fake yt_dlp module, no real network) ---")
from ultrastar_generator.youtube_source import YoutubeDownloadError


class _FakeYoutubeDL:
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def download(self, urls):
        # Simulate yt-dlp actually producing the expected output file, PLUS
        # (if writethumbnail is set, as it now always is) a converted
        # thumbnail -- real yt-dlp's FFmpegThumbnailsConvertor always lands
        # it at "<outtmpl base>.jpg" regardless of audio/video mode.
        out_tmpl = self.opts["outtmpl"]
        ext = "mp4" if "merge_output_format" in self.opts else "mp3"
        Path(out_tmpl % {"ext": ext}).write_bytes(b"fake-downloaded-bytes")
        if self.opts.get("writethumbnail"):
            Path(out_tmpl % {"ext": "jpg"}).write_bytes(b"fake-thumbnail-bytes")


class _FakeYoutubeDLFails:
    def __init__(self, opts):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def download(self, urls):
        raise RuntimeError("Video unavailable")


class _FakeYtDlpModule:
    def __init__(self, cls):
        self.YoutubeDL = cls


import sys as _sys
_sys.modules["yt_dlp"] = _FakeYtDlpModule(_FakeYoutubeDL)
from ultrastar_generator.youtube_source import download_youtube_source
with _tempfile.TemporaryDirectory() as d:
    out = download_youtube_source("https://example.com/fake", Path(d), audio_only=True)
    assert out.name == "youtube_download.mp3" and out.read_bytes() == b"fake-downloaded-bytes"
    cover = Path(d) / "youtube_download [CO].jpg"
    assert cover.is_file() and cover.read_bytes() == b"fake-thumbnail-bytes"
    assert not (Path(d) / "youtube_download.jpg").exists(), "raw thumbnail should be renamed, not left behind"
with _tempfile.TemporaryDirectory() as d:
    out = download_youtube_source("https://example.com/fake", Path(d), audio_only=False)
    assert out.name == "youtube_download.mp4"
    assert (Path(d) / "youtube_download [CO].jpg").is_file()
print("OK: successful download resolves to the expected deterministic filename (mp3 audio-only / mp4 video), "
      "and the thumbnail is renamed to the [CO]-tagged cover convention find_companions already knows")


class _FakeYoutubeDLNoThumbnail(_FakeYoutubeDL):
    def download(self, urls):
        # Simulate yt-dlp succeeding at the real download but failing to
        # find/convert a thumbnail for this particular video -- must not
        # fail the overall download.
        out_tmpl = self.opts["outtmpl"]
        ext = "mp4" if "merge_output_format" in self.opts else "mp3"
        Path(out_tmpl % {"ext": ext}).write_bytes(b"fake-downloaded-bytes")


_sys.modules["yt_dlp"] = _FakeYtDlpModule(_FakeYoutubeDLNoThumbnail)
with _tempfile.TemporaryDirectory() as d:
    out = download_youtube_source("https://example.com/fake", Path(d), audio_only=True)
    assert out.is_file()
    assert not (Path(d) / "youtube_download [CO].jpg").exists()
    assert not (Path(d) / "youtube_download.jpg").exists()
print("OK: a video with no fetchable thumbnail still downloads successfully (thumbnail rename is a silent no-op)")

_sys.modules["yt_dlp"] = _FakeYtDlpModule(_FakeYoutubeDLFails)
with _tempfile.TemporaryDirectory() as d:
    try:
        download_youtube_source("https://example.com/fake", Path(d), audio_only=True)
        assert False, "should have raised YoutubeDownloadError"
    except YoutubeDownloadError as e:
        assert "Video unavailable" in str(e), e
print("OK: a download failure (network/private/removed video) raises YoutubeDownloadError, never a raw exception")
del _sys.modules["yt_dlp"]
