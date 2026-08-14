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

print("\n--- file_discovery.resolve_artist_title: the INPUT FOLDER's own name is the sole "
      "source now (folder-based input) -- a file inside can be named anything at all, real "
      "case: a ripped/downloaded song keeps a generic filename like 'music.ogg' while its "
      "folder is 'Artist - Title' ---")
import tempfile as _tempfile_artist_title
with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    named_folder = Path(_tmp) / "Bon Jovi - Its My Life"
    named_folder.mkdir()
    unparseable_audio = named_folder / "music.mp3"
    artist3, title3 = resolve_artist_title(unparseable_audio, named_folder)
    assert (artist3, title3) == ("Bon Jovi", "Its My Life"), (artist3, title3)
print("OK: the folder name is used regardless of what the audio file itself is named")

with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    normal_folder = Path(_tmp) / "some_folder"
    normal_folder.mkdir()
    parseable_audio = normal_folder / "Bon Jovi - Its My Life.mp3"
    artist2, title2 = resolve_artist_title(parseable_audio, normal_folder)
    assert (artist2, title2) == (None, None), (artist2, title2)
print("OK: even when the AUDIO FILE's own name would parse fine, a non-parseable folder name "
      "still returns (None, None) -- the audio filename is never consulted at all")

with _tempfile_artist_title.TemporaryDirectory() as _tmp:
    unparseable_folder = Path(_tmp) / "random_folder_name"
    unparseable_folder.mkdir()
    unparseable_audio2 = unparseable_folder / "music.mp3"
    artist4, title4 = resolve_artist_title(unparseable_audio2, unparseable_folder)
    assert (artist4, title4) == (None, None), (artist4, title4)
print("OK: a folder name that doesn't parse returns (None, None) rather than raising")

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
      "extension")
print("OK:", artist, title, comp)

print("\n--- file_discovery.headline_case: minor words lowercased unless first/last, but "
      "ALL CAPS or unusually-cased words are left completely untouched ---")
from ultrastar_generator.file_discovery import headline_case
assert headline_case("Beauty And The Beast") == "Beauty and the Beast"
assert headline_case("Under The Sea") == "Under the Sea"
assert headline_case("the lion king") == "The Lion King"  # first word always capitalized
assert headline_case("A Bug's Life") == "A Bug's Life"  # first word "A" stays capitalized
assert headline_case("Kill It With Fire, Or Not") == "Kill It with Fire, or Not"  # last word always capitalized
assert headline_case("KPop Demon Hunters") == "KPop Demon Hunters"  # mixed-case word untouched
assert headline_case("SHOUT AND WHISPER") == "SHOUT AND WHISPER"  # ALL CAPS words untouched
assert headline_case("aND weird CaSe") == "aND Weird CaSe"  # unusual casing untouched, but a
                                                                # normal (simple-case) word still
                                                                # gets normalized regardless of
                                                                # its neighbors' casing
assert headline_case("Don't Stop Believin'") == "Don't Stop Believin'"  # apostrophes preserved
print("OK: 'Beauty And The Beast' -> 'Beauty and the Beast', 'KPop'/'AND'/'aND' all left "
      "untouched, first/last word always capitalized")

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
# FLOORS, never rounds up (2026-08-10, user's explicit request: undershoot
# note length rather than overshoot). beat_duration_ms(300)=50ms exactly;
# 427ms/50ms = 8.54 beats -- round() would give 9 (overshoot), floor must give 8.
assert seconds_to_beat_length(0.427, 300) == 8, seconds_to_beat_length(0.427, 300)
# Still guarantees at least 1 beat even for a near-zero duration.
assert seconds_to_beat_length(0.001, 300) == 1
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

print("\n--- usdx_writer._merge_connected_melisma_tails (2026-08-10): a beat-adjacent, same-pitch '~' "
      "melisma-continuation note gets folded into the note before it instead of staying a separate note "
      "-- real user-reported example ('Barely even friends') ---")
from ultrastar_generator.usdx_writer import _merge_connected_melisma_tails, render_song as _render_merge

mcm_input = [
    ("syl", 261, 1, 1, "Bare", True, ":"),
    ("syl", 263, 1, 3, "ly", False, ":"),
    ("syl", 264, 3, 3, "~", False, ":"),        # same pitch (3) as "ly", adjacent -> merges into "ly"
    ("syl", 268, 1, 5, " even", True, ":"),
    ("syl", 269, 1, 6, "~", False, ":"),        # different pitch than "even" (6 vs 5) -- stays separate...
    ("syl", 270, 1, 6, "~", False, ":"),        # ...but THIS one is same pitch as the previous '~' (6==6),
                                                 # adjacent -> the two '~' notes merge into one
    ("syl", 272, 2, 8, " friends", True, ":"),
    ("syl", 274, 3, 8, "~", False, ":"),        # same pitch (8) as "friends", adjacent -> merges into "friends"
]
mcm_out = _merge_connected_melisma_tails(mcm_input)
assert mcm_out == [
    ("syl", 261, 1, 1, "Bare", True, ":"),
    ("syl", 263, 4, 3, "ly", False, ":"),       # 1+3=4
    ("syl", 268, 1, 5, " even", True, ":"),
    ("syl", 269, 2, 6, "~", False, ":"),        # 1+1=2, still untexted
    ("syl", 272, 5, 8, " friends", True, ":"),  # 2+3=5
], mcm_out
print("OK:", mcm_out)

mcm_gap_input = [
    ("syl", 0, 2, 4, "held", True, ":"),
    ("syl", 3, 2, 4, "~", False, ":"),   # same pitch, but NOT adjacent (0+2=2 != 3) -- a real gap/pause -> no merge
]
assert _merge_connected_melisma_tails(mcm_gap_input) == mcm_gap_input, _merge_connected_melisma_tails(mcm_gap_input)
print("OK: a same-pitch '~' separated by even a 1-beat gap is left alone (not a genuine continuation)")

mcm_pitch_input = [
    ("syl", 0, 2, 4, "held", True, ":"),
    ("syl", 2, 2, 5, "~", False, ":"),   # adjacent, but DIFFERENT pitch -> no merge
]
assert _merge_connected_melisma_tails(mcm_pitch_input) == mcm_pitch_input
print("OK: an adjacent but different-pitch '~' is left alone (a real pitch change, not noise)")

mcm_linebreak_input = [
    ("syl", 0, 2, 4, "held", True, ":"),
    ("break", 2, 2),
    ("syl", 2, 2, 4, "~", False, ":"),   # same pitch, "adjacent" only by ignoring the LineBreak -- must NOT merge
]
assert _merge_connected_melisma_tails(mcm_linebreak_input) == mcm_linebreak_input
print("OK: a '~' right after a LineBreak never merges backward across it, even at the same pitch")

# End-to-end option wiring: render_song(merge_connected_melisma=True/False) actually changes the output.
mcm_song = Song(
    # bpm=240 -> beat = 1/(240*4/60) = 0.0625s exactly; all timestamps below are
    # exact multiples of that, so quantization can't introduce a rounding-tie gap
    # (an earlier version of this test picked bpm/timestamps that landed exactly
    # on a beat's midpoint, which Python's banker's rounding resolved differently
    # for the two adjacent notes and spuriously introduced a 1-beat gap between
    # them -- not a bug in the merge logic itself, just an unlucky test input).
    title="T", artist="A", mp3="a.mp3", bpm=240.0, gap_ms=0,
    entries=[
        Syllable("Bare", 0.0, 0.25, 1, is_word_start=True),
        Syllable("ly", 0.25, 0.50, 3, is_word_start=False),
        Syllable("~", 0.50, 1.50, 3, is_word_start=False),
    ],
)
mcm_txt_on = _render_merge(mcm_song, merge_connected_melisma=True)
mcm_txt_off = _render_merge(mcm_song, merge_connected_melisma=False)
assert "~" not in mcm_txt_on, mcm_txt_on
assert "~" in mcm_txt_off, mcm_txt_off
assert mcm_txt_on != mcm_txt_off
print("OK: render_song(merge_connected_melisma=True) actually removes the redundant '~'; "
      "=False (the function's own default) leaves it untouched")

print("\n--- usdx_writer._remove_orphan_short_melisma_tails (2026-08-10): a '~' that's STILL only "
      "1 beat long after the same-pitch merge above is deleted outright (leaves a gap, not merged) ---")
from ultrastar_generator.usdx_writer import _remove_orphan_short_melisma_tails

orphan_input = [
    ("syl", 0, 2, 4, "held", True, ":"),
    ("syl", 2, 1, 6, "~", False, ":"),   # adjacent but DIFFERENT pitch (6 vs 4) -- merge pass leaves this alone,
                                          # but it's still only 1 beat long -> this pass deletes it
    ("syl", 4, 2, 4, "held", True, ":"),
]
orphan_out = _remove_orphan_short_melisma_tails(orphan_input)
assert orphan_out == [
    ("syl", 0, 2, 4, "held", True, ":"),
    ("syl", 4, 2, 4, "held", True, ":"),
], orphan_out
print("OK: a 1-beat, different-pitch '~' is deleted outright, leaving a gap:", orphan_out)

orphan_multi_beat_input = [
    ("syl", 0, 2, 4, "held", True, ":"),
    ("syl", 2, 2, 6, "~", False, ":"),   # 2 beats -- NOT a 1-beat orphan, must survive untouched
]
assert _remove_orphan_short_melisma_tails(orphan_multi_beat_input) == orphan_multi_beat_input
print("OK: a '~' longer than 1 beat is never touched by this pass, only the same-pitch merge above can shrink it")

orphan_texted_input = [
    ("syl", 0, 1, 4, "a", True, ":"),    # a real, 1-beat WORD syllable (not '~') -- must never be deleted
]
assert _remove_orphan_short_melisma_tails(orphan_texted_input) == orphan_texted_input
print("OK: a genuine 1-beat WORD syllable (not the melisma-continuation placeholder) is left alone")

# End-to-end: render_song(merge_connected_melisma=True) chains BOTH steps -- a same-pitch adjacent
# '~' still gets folded in (not deleted), while a different-pitch 1-beat orphan '~' disappears entirely.
orphan_song = Song(
    title="T", artist="A", mp3="a.mp3", bpm=240.0, gap_ms=0,
    entries=[
        Syllable("held", 0.0, 0.125, 4, is_word_start=True),   # 2 beats
        Syllable("~", 0.125, 0.1875, 7, is_word_start=False),  # 1 beat, DIFFERENT pitch -> orphan, deleted
        Syllable("next", 0.1875, 0.3125, 4, is_word_start=True),  # 2 beats
        Syllable("~", 0.3125, 0.375, 4, is_word_start=False),  # 1 beat, SAME pitch as "next" -> merged, not deleted
    ],
)
orphan_txt_on = _render_merge(orphan_song, merge_connected_melisma=True)
orphan_txt_off = _render_merge(orphan_song, merge_connected_melisma=False)
assert orphan_txt_on.count("~") == 0, orphan_txt_on
assert orphan_txt_off.count("~") == 2, orphan_txt_off
print("OK: end-to-end, merge_connected_melisma=True both merges the same-pitch '~' AND deletes the "
      "different-pitch 1-beat orphan '~', leaving zero '~' in the output; =False leaves both untouched")

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

print("\n--- lyrics_lookup.align_words_to_reference: a LONG unmatched (delete) run is KEPT, not "
      "dropped -- real regression (Trixie Mattel - Video Games, 2026-08-13): a repeat-heavy song's "
      "own repeated chorus ('It's you, it's you, it's all for you' x4+) made difflib's global "
      "alignment misclassify 219 of 355 REAL, correctly-transcribed words as one giant delete "
      "block, which the unconditional drop then mass-deleted from the final file. Past "
      "config.REFERENCE_DELETE_MAX_RUN words in one run, dropping is more likely to destroy real "
      "content than remove genuine hallucination -- fall back to keeping them (old behavior) ---")
long_delete_ref_lines = ["Swinging in the backyard"]
# 8 unmatched ASR words in a row (> REFERENCE_DELETE_MAX_RUN=5) representing
# content genuinely absent from this tiny reference -- the function can't
# tell "real alignment failure" from "genuinely long non-lyrical passage"
# apart, which is exactly why the cap treats a long run conservatively.
long_delete_words = [
    Word(text=w, start=float(i), end=float(i) + 0.2, confidence=0.9)
    for i, w in enumerate(["It's", "you,", "it's", "you,", "it's", "all", "for", "you"])
] + [
    Word(text="Swinging", start=10.0, end=10.2, confidence=0.9),
    Word(text="in", start=10.3, end=10.4, confidence=0.9),
    Word(text="the", start=10.5, end=10.6, confidence=0.9),
    Word(text="backyard", start=10.7, end=11.0, confidence=0.9),
]
long_delete_aligned = align_words_to_reference(long_delete_words, long_delete_ref_lines)
assert len(long_delete_aligned) == len(long_delete_words), \
    f"expected all {len(long_delete_words)} words kept, got {len(long_delete_aligned)}"
assert [w.text for w in long_delete_aligned[:8]] == ["It's", "you,", "it's", "you,", "it's", "all", "for", "you"]
print(f"OK: a {8}-word unmatched run (> REFERENCE_DELETE_MAX_RUN) is kept in full, not dropped:",
      [w.text for w in long_delete_aligned])

print("\n--- lyrics_lookup.align_words_to_reference: repeat-clamp caps + wraps instead of freezing "
      "(real case: David Bowie - Magic Dance, 2026-08-10 -- decoder hallucinated a 'Dance, magic, dance' "
      "x5 passage into ~90 garbage ASR tokens, all clamped onto the SAME reference token; the syllable "
      "cursor froze on the last syllable ('ic') and verify_words then stamped that onto ~89 real notes as "
      "if it were confirmed reference text) ---")
# Below the cap: repeats should WRAP across the reference token's own syllables
# instead of freezing on the last one once the cursor runs out.
wrap_ref_lines = ["magic"]
wrap_asr_words = [Word(text="dance", start=float(i), end=float(i) + 0.2, confidence=0.9) for i in range(4)]
wrap_aligned = align_words_to_reference(wrap_asr_words, wrap_ref_lines)
wrap_texts = [w.reference_text for w in wrap_aligned]
assert wrap_texts == ["mag", "ic", "mag", "ic"], wrap_texts
print("OK: below the cap, syllables wrap:", wrap_texts)

# Above the cap: this many ASR words clamping onto one reference token is
# itself the hallucination signal -- don't fabricate a reference_text at all,
# keep the ASR word's own (still garbage, but at least not falsely
# "confirmed") text untouched.
runaway_ref_lines = ["magic"]
runaway_asr_words = [Word(text="ic", start=float(i), end=float(i) + 0.2, confidence=0.9) for i in range(20)]
runaway_aligned = align_words_to_reference(runaway_asr_words, runaway_ref_lines)
assert all(w.reference_text is None for w in runaway_aligned), [w.reference_text for w in runaway_aligned]
assert all(w.text == "ic" for w in runaway_aligned)
assert all(w.line_id == 0 for w in runaway_aligned), [w.line_id for w in runaway_aligned]
import ultrastar_generator.config as _config_mod
print(f"OK: {len(runaway_asr_words)} words clamping onto one reference token (> "
      f"config.REFERENCE_CLAMP_MAX_REPEAT={_config_mod.REFERENCE_CLAMP_MAX_REPEAT}) leaves reference_text "
      f"unset, keeps ASR's own text, still tags line_id for phrase grouping")

print("\n--- lyrics_lookup.align_words_to_reference: repeat-clamp gap guard is checked against BOTH "
      "GLOBAL neighbors (not block-relative, and not just the previous one) -- real confirmed bug "
      "(Trixie Mattel - Video Games, 2026-08-13): a non-lyrical audio intro hallucinated as 'You're "
      "welcome.' landed as the song's first 2 ASR words, clamped onto reference word 0 ('Swingin'') "
      "along with the real 'Swinging' ~16s later; a backward-only gap check correctly rejected "
      "'welcome.' but let 'You're' (the very first word, no previous neighbor to compare against) "
      "through with a bogus reference_text that verification.py's own fallback then trusted and used "
      "to overwrite it. Hallucinated words are now flagged word.dropped=True -- STILL KEPT in the "
      "returned sequence (not omitted), because removing a word from the sequence entirely let a "
      "NEIGHBORING real word's pass-1 note zone silently swallow its ~16s of notes instead (real "
      "confirmed regression, Video Games, 2026-08-14 -- see Word.dropped's own docstring); "
      "lyric_alignment.py is what actually keeps a dropped word's text/notes out of the final output. "
      "The real, correctly-transcribed 'Swinging' (last word of THIS opcode block, clamped onto the "
      "same reference token) must NOT be flagged dropped, because its real close neighbor 'in' sits "
      "just outside the block, in the NEXT opcode -- a block-relative-only neighbor check would have "
      "wrongly flagged it too ---")
leading_outlier_ref_lines = ["Swingin'", "in"]
leading_outlier_words = [
    Word(text="You're", start=9.595, end=10.035, confidence=0.9),
    Word(text="welcome.", start=12.476, end=13.636, confidence=0.9),
    Word(text="Swinging", start=26.058, end=26.4, confidence=0.9),
    Word(text="in", start=26.5, end=26.6, confidence=0.9),
]
leading_outlier_aligned = align_words_to_reference(leading_outlier_words, leading_outlier_ref_lines)
assert [w.text for w in leading_outlier_aligned] == ["You're", "welcome.", "Swinging", "in"], \
    [w.text for w in leading_outlier_aligned]
assert [w.dropped for w in leading_outlier_aligned] == [True, True, False, False], \
    [(w.text, w.dropped) for w in leading_outlier_aligned]
print("OK: 'You're'/'welcome.' (no real reference correspondence, isolated in time even from each "
      "other) are flagged dropped but still kept in sequence; 'Swinging' (real word, close global "
      "neighbor 'in' just outside its own opcode block) is NOT flagged:",
      [(w.text, w.dropped) for w in leading_outlier_aligned])

print("\n--- lyrics_lookup.align_words_to_reference: an ASR word with NO reference counterpart at "
      "all is flagged dropped (word.dropped=True), not kept as visible text -- real case (Trixie "
      "Mattel - Video Games, 2026-08-13): WhisperX decoded a non-lyrical audio intro as 'You're... "
      "welcome.', which used to become the song's own first two 'lyric' words even though nothing in "
      "the reference remotely matches them; a real trailing hallucination ('you' after the song's "
      "last real word) had the same problem. Still KEPT in the returned sequence (not omitted) -- see "
      "Word.dropped's own docstring for why omitting it broke downstream note-zone boundaries; "
      "lyric_alignment.py is what actually excludes a dropped word's text/notes from the final "
      "output ---")
drop_ref_lines = ["Swinging in the backyard"]
drop_asr_words = [
    Word(text="You're", start=9.6, end=10.0, confidence=0.9),     # leading hallucination, no ref match
    Word(text="welcome.", start=12.5, end=13.6, confidence=0.9),  # leading hallucination, no ref match
    Word(text="Swinging", start=26.0, end=26.5, confidence=0.9),
    Word(text="in", start=26.6, end=26.8, confidence=0.9),
    Word(text="the", start=26.9, end=27.0, confidence=0.9),
    Word(text="backyard", start=27.1, end=27.6, confidence=0.9),
    Word(text="you", start=220.0, end=223.7, confidence=0.9),     # trailing hallucination, no ref match
]
drop_aligned = align_words_to_reference(drop_asr_words, drop_ref_lines)
assert len(drop_aligned) == len(drop_asr_words), \
    f"expected all {len(drop_asr_words)} words kept (flagged, not omitted), got {len(drop_aligned)}"
visible_texts = [w.text for w in drop_aligned if not w.dropped]
assert visible_texts == ["Swinging", "in", "the", "backyard"], visible_texts
dropped_texts = [w.text for w in drop_aligned if w.dropped]
assert dropped_texts == ["You're", "welcome.", "you"], dropped_texts
drop_diffs = alignment_diff_summary(drop_asr_words, drop_aligned)
assert any('"You\'re" -> [DROPPED' in d for d in drop_diffs), drop_diffs
assert any('"welcome." -> [DROPPED' in d for d in drop_diffs), drop_diffs
assert any('"you" -> [DROPPED' in d for d in drop_diffs), drop_diffs
print("OK: leading+trailing hallucinated words with no reference match are flagged dropped (excluded "
      "from visible text) and reported, but still occupy their slot in the sequence:", visible_texts,
      dropped_texts)
print("OK: alignment_diff_summary reports drops explicitly:", drop_diffs)

print("\n--- lyrics_lookup.align_words_to_reference: a word inside an UNEVEN replace block (a real, "
      "if imprecisely-mapped, reference correspondence -- NOT the same as having none at all) is "
      "still kept, only a clean 'delete' block (zero correspondence anywhere) is dropped ---")
uneven_ref_lines = ["double-edged knife"]
uneven_asr_words = [
    Word(text="double", start=0.0, end=0.2, confidence=0.9),
    Word(text="edged", start=0.2, end=0.4, confidence=0.9),
    Word(text="kide", start=0.4, end=0.6, confidence=0.9),  # OCR/ASR garble of "knife", still kept
]
uneven_aligned = align_words_to_reference(uneven_asr_words, uneven_ref_lines)
assert len(uneven_aligned) == 3, [w.text for w in uneven_aligned]
print("OK: uneven-block words are kept (not dropped), only zero-correspondence words are:",
      [w.text for w in uneven_aligned])

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

print("\n--- lyrics_lookup.largest_unmatched_reference_run: measures the LARGEST contiguous run of "
      "reference words with NO corresponding ASR word at all -- real case (Trixie Mattel - Gold, "
      "2026-08-10): a whole chorus repeat's 'Do-do-do-do-do' backing vocal produced ZERO ASR words at one "
      "occurrence while the rest of a 306-word real transcript (including this SAME phrase correctly "
      "transcribed at a LATER repeat) was fine -- hidden from reference_match_ratio's own aggregate (89.3%, "
      "well above the retry bar). LRCLIB writes the repeat as ONE hyphenated token, not 5 space-separated "
      "words -- `_tokenize_lines` splits on '-' (2026-08-10 fix) specifically so this counts as 5 reference "
      "words missing, not 1 -- confirmed against the real fetched reference + real parsed ASR debug-log "
      "output for this exact song: largest_unmatched_reference_run went from 1 (pre-fix, split into two "
      "even-smaller 1-token gaps by the correctly-matched 'They start to play' line sitting between them) "
      "to 7 (post-fix) ---")
from ultrastar_generator.lyrics_lookup import largest_unmatched_reference_run
lur_ref_lines = [
    "Will you grow from those cold blood wrongs",
    "when those old love songs start to play",
    "Do-do-do-do-do",
    "They start to play",
    "Do-do-do-do-do",
]
lur_words_dropped = [Word(text=w, start=float(i), end=float(i) + 0.3, confidence=0.9) for i, w in enumerate(
    ["Will", "you", "grow", "from", "those", "cold", "blood", "wrongs",
     "when", "those", "old", "love", "songs", "start", "to", "play",
     "They", "start", "to", "play"]
)]
assert largest_unmatched_reference_run(lur_ref_lines, lur_words_dropped) == 5, \
    largest_unmatched_reference_run(lur_ref_lines, lur_words_dropped)
lur_words_present = lur_words_dropped[:16] + [
    Word(text=w, start=16.0 + i * 0.3, end=16.3 + i * 0.3, confidence=0.9)
    for i, w in enumerate(["Do", "do", "do", "do", "do"])
] + lur_words_dropped[16:]
assert largest_unmatched_reference_run(lur_ref_lines, lur_words_present) == 5, \
    largest_unmatched_reference_run(lur_ref_lines, lur_words_present)  # the SECOND "Do-do-do-do-do" still missing
lur_words_both_present = lur_words_present + [
    Word(text=w, start=20.0 + i * 0.3, end=20.3 + i * 0.3, confidence=0.9)
    for i, w in enumerate(["Do", "do", "do", "do", "do"])
]
assert largest_unmatched_reference_run(lur_ref_lines, lur_words_both_present) == 0, \
    largest_unmatched_reference_run(lur_ref_lines, lur_words_both_present)
print("OK: a hyphenated 'Do-do-do-do-do' reference passage (5 real sung words, ONE written token) with zero "
      "ASR words scores a run of 5, not 1 -- with only ONE of its two real occurrences transcribed, the "
      "still-missing occurrence still scores 5; with both transcribed, scores 0")

print("\n--- lyrics_lookup._tokenize_lines: splits a hyphenated token into separate words generally, not "
      "just for the run-detection case above -- benefits align_words_to_reference's own alignment too ---")
from ultrastar_generator.lyrics_lookup import _tokenize_lines
tl_norm, tl_orig, tl_line_ids = _tokenize_lines(["Do-do-do-do-do", "well-known fact"])
assert tl_orig == ["Do", "do", "do", "do", "do", "well", "known", "fact"], tl_orig
assert tl_line_ids == [0, 0, 0, 0, 0, 1, 1, 1], tl_line_ids
print("OK:", tl_orig)

print("\n--- transcription.force_align_words_in_window (PROTOTYPE, 2026-08-10, adapted from "
      "UltraStarKaraokeMaker's realign_gap_windows): forces KNOWN text onto an audio window via a real "
      "wav2vec2 CTC call -- validates the result before ever trusting it (word count, timestamps present, "
      "within window, monotonic), never applies a partial/ambiguous result ---")
import sys as _sys_fa
import types as _types_fa


class _FakeWhisperXAlignModule:
    def __init__(self):
        self.align_fn = None

    def align(self, segments, align_model, metadata, audio, device=None, return_char_alignments=False):
        return self.align_fn(segments[0])


_fake_whisperx_fa = _FakeWhisperXAlignModule()
_sys_fa.modules["whisperx"] = _fake_whisperx_fa
from ultrastar_generator.transcription import force_align_words_in_window

# (a) clean success: 3 words, all measured, monotonic, inside the window.
_fake_whisperx_fa.align_fn = lambda seg: {"segments": [{"words": [
    {"word": "Do", "start": 10.0, "end": 10.2, "score": 0.7},
    {"word": "do", "start": 10.2, "end": 10.4, "score": 0.6},
    {"word": "do", "start": 10.4, "end": 10.6, "score": 0.65},
]}]}
fa_result = force_align_words_in_window(["Do", "do", "do"], 10.0, 11.0, None, None, None)
assert fa_result is not None and len(fa_result) == 3, fa_result
assert fa_result[0][:2] == (10.0, 10.2) and fa_result[2][:2] == (10.4, 10.6), fa_result
print("OK: clean forced-alignment result accepted, per-word (start, end, score) returned in order")

# (b) word-count mismatch (e.g. whisperx expanded/collapsed a token) -> rejected, None.
_fake_whisperx_fa.align_fn = lambda seg: {"segments": [{"words": [
    {"word": "Do", "start": 10.0, "end": 10.2, "score": 0.7},
]}]}
assert force_align_words_in_window(["Do", "do", "do"], 10.0, 11.0, None, None, None) is None
print("OK: word-count mismatch (asked for 3, got 1) -> rejected rather than guessing a mapping")

# (c) a word placed outside the window (beyond slop) -> rejected, None.
_fake_whisperx_fa.align_fn = lambda seg: {"segments": [{"words": [
    {"word": "Do", "start": 10.0, "end": 10.2, "score": 0.7},
    {"word": "do", "start": 10.2, "end": 10.4, "score": 0.6},
    {"word": "do", "start": 20.0, "end": 20.2, "score": 0.6},  # way past window_end=11.0 + slop
]}]}
assert force_align_words_in_window(["Do", "do", "do"], 10.0, 11.0, None, None, None) is None
print("OK: a word landing well outside the window -> rejected")

# (d) non-monotonic output (a later word starting before an earlier one) -> rejected, None.
_fake_whisperx_fa.align_fn = lambda seg: {"segments": [{"words": [
    {"word": "Do", "start": 10.4, "end": 10.6, "score": 0.7},
    {"word": "do", "start": 10.0, "end": 10.2, "score": 0.6},  # earlier than the word before it
]}]}
assert force_align_words_in_window(["Do", "do"], 10.0, 11.0, None, None, None) is None
print("OK: non-monotonic word order -> rejected")

# (e) missing timestamp on one word -> rejected, None.
_fake_whisperx_fa.align_fn = lambda seg: {"segments": [{"words": [
    {"word": "Do", "start": None, "end": None, "score": 0.0},
]}]}
assert force_align_words_in_window(["Do"], 10.0, 11.0, None, None, None) is None
print("OK: a word with no measured timestamp -> rejected")

# (f) window too short for the given word count -> rejected WITHOUT even calling whisperx.align.
def _fa_boom(seg):
    raise AssertionError("whisperx.align must not be called when the window is too short to bother")
_fake_whisperx_fa.align_fn = _fa_boom
assert force_align_words_in_window(["one", "two", "three", "four", "five"], 10.0, 10.05, None, None, None) is None
print("OK: a window too short for the word count -> rejected before ever calling whisperx.align")

# (g) whisperx.align itself raising -> caught, None, never crashes the pipeline.
def _fa_raises(seg):
    raise RuntimeError("simulated alignment backtrack failure")
_fake_whisperx_fa.align_fn = _fa_raises
assert force_align_words_in_window(["Do"], 10.0, 11.0, None, None, None) is None
print("OK: whisperx.align() raising is caught, not propagated")

del _sys_fa.modules["whisperx"]

print("\n--- lyrics_lookup.recover_dropped_reference_words (PROTOTYPE, 2026-08-10): splices force-aligned "
      "words into a copy of the ASR word list at each reference 'insert' gap -- real case (Trixie Mattel - "
      "Gold): recovers a whole 'Do-do-do-do-do' passage ASR produced zero words for ---")
from ultrastar_generator.lyrics_lookup import recover_dropped_reference_words
import ultrastar_generator.transcription as transcription_mod_fa
import ultrastar_generator.model_cache as model_cache_mod_fa

rdr_ref_lines = ["Will you grow from those cold blood wrongs", "Do-do-do-do-do", "They start to play"]
rdr_words = [
    Word(text="Will", start=0.0, end=0.3), Word(text="you", start=0.3, end=0.5),
    Word(text="grow", start=0.5, end=0.8), Word(text="from", start=0.8, end=1.0),
    Word(text="those", start=1.0, end=1.3), Word(text="cold", start=1.3, end=1.5),
    Word(text="blood", start=1.5, end=1.8), Word(text="wrongs", start=1.8, end=2.0),
    # <-- "Do-do-do-do-do" (5 words after hyphen-splitting) completely missing here -->
    Word(text="They", start=5.0, end=5.3), Word(text="start", start=5.3, end=5.6),
    Word(text="to", start=5.6, end=5.7), Word(text="play", start=5.7, end=6.0),
]
_orig_force_align_fa = transcription_mod_fa.force_align_words_in_window
_orig_align_model_fa = model_cache_mod_fa.get_whisperx_align_model
transcription_mod_fa.force_align_words_in_window = lambda words_text, w0, w1, *a, **kw: [
    (w0 + i * 0.4, w0 + i * 0.4 + 0.3, 0.5) for i in range(len(words_text))
]
model_cache_mod_fa.get_whisperx_align_model = lambda *a, **kw: (None, None)
_sys_fa.modules["whisperx"] = _types_fa.SimpleNamespace(load_audio=lambda path: [0] * 160000)  # 10s @ 16kHz
rdr_new_words, rdr_n = recover_dropped_reference_words(rdr_ref_lines, rdr_words, Path("dummy.wav"))
assert rdr_n == 5, rdr_n
assert len(rdr_new_words) == len(rdr_words) + 5, len(rdr_new_words)
rdr_recovered = [w for w in rdr_new_words if w.text in ("Do", "do")]
assert len(rdr_recovered) == 5 and [w.text for w in rdr_recovered] == ["Do", "do", "do", "do", "do"], \
    [w.text for w in rdr_recovered]
# recovered words must land strictly between "wrongs" (ends 2.0) and "They" (starts 5.0) -- the real gap window
assert all(2.0 <= w.start < 5.0 for w in rdr_recovered), [(w.start, w.end) for w in rdr_recovered]
assert [w.text for w in rdr_words] == ["Will", "you", "grow", "from", "those", "cold", "blood", "wrongs",
                                        "They", "start", "to", "play"], \
    "the original words list must never be mutated"
print(f"  OK: {rdr_n} words recovered ('Do-do-do-do-do' split into 5), spliced into the gap between "
      f"'wrongs' and 'They' with real (fake, in this test) timing; original word list left untouched")

# force_align_words_in_window returning None (couldn't recover) -> gap stays dropped, words unchanged.
transcription_mod_fa.force_align_words_in_window = lambda *a, **kw: None
rdr_noop_words, rdr_noop_n = recover_dropped_reference_words(rdr_ref_lines, rdr_words, Path("dummy.wav"))
assert rdr_noop_n == 0 and len(rdr_noop_words) == len(rdr_words), (rdr_noop_n, len(rdr_noop_words))
print("  OK: an unrecoverable gap (no usable alignment result) is left dropped, not force-inserted")

transcription_mod_fa.force_align_words_in_window = _orig_force_align_fa
model_cache_mod_fa.get_whisperx_align_model = _orig_align_model_fa

print("\n--- transcription.force_align_reference_lyrics (DIAGNOSTIC, 2026-08-10, --no-transcribe): builds "
      "the ENTIRE word list by force-aligning a pinned LRC candidate's own KNOWN per-line text, never "
      "running the WhisperX decoder at all -- motivated by the David Bowie - Magic Dance case where the "
      "decoder hallucinated a real repeated 'Dance, magic, dance' passage into ~90 garbage tokens ---")
from ultrastar_generator.transcription import force_align_reference_lyrics

_orig_force_align_fr = transcription_mod_fa.force_align_words_in_window
_orig_align_model_fr = model_cache_mod_fa.get_whisperx_align_model
transcription_mod_fa.force_align_words_in_window = lambda words_text, w0, w1, *a, **kw: [
    (w0 + i * 0.3, w0 + i * 0.3 + 0.25, 0.9) for i in range(len(words_text))
]
model_cache_mod_fa.get_whisperx_align_model = lambda *a, **kw: (None, None)
_sys_fa.modules["whisperx"] = _types_fa.SimpleNamespace(load_audio=lambda path: [0] * 160000)

fr_synced = "[00:10.00]hello world\n[00:15.00]second line here\n"
fr_words = force_align_reference_lyrics(Path("dummy.wav"), fr_synced, audio_duration=20.0)
assert [w.text for w in fr_words] == ["hello", "world", "second", "line", "here"], [w.text for w in fr_words]
# first line's window is [10.0, 15.0) (next line's own timestamp), second line's is [15.0, 20.0) (audio_duration)
assert all(10.0 <= w.start < 15.0 for w in fr_words[:2]), [(w.start, w.end) for w in fr_words[:2]]
assert all(15.0 <= w.start < 20.0 for w in fr_words[2:]), [(w.start, w.end) for w in fr_words[2:]]
print(f"  OK: {len(fr_words)} words built purely from KNOWN LRC line text + forced alignment, windowed "
      f"per-line (no decoder output anywhere in this path)")

# A line whose force-alignment fails is skipped entirely (not guessed/interpolated here).
transcription_mod_fa.force_align_words_in_window = lambda words_text, w0, w1, *a, **kw: (
    None if w0 == 10.0 else [(w0 + i * 0.3, w0 + i * 0.3 + 0.25, 0.9) for i in range(len(words_text))]
)
fr_partial = force_align_reference_lyrics(Path("dummy.wav"), fr_synced, audio_duration=20.0)
assert [w.text for w in fr_partial] == ["second", "line", "here"], [w.text for w in fr_partial]
print("  OK: a line whose forced alignment fails is skipped entirely, not force-guessed")

# No synced lyrics at all -> empty list, never crashes.
assert force_align_reference_lyrics(Path("dummy.wav"), "", audio_duration=20.0) == []
print("  OK: empty/no synced lyrics -> empty word list")

transcription_mod_fa.force_align_words_in_window = _orig_force_align_fr
model_cache_mod_fa.get_whisperx_align_model = _orig_align_model_fr
del _sys_fa.modules["whisperx"]

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
    MxlLrcQuality, _text_for_mxl_syllables, config as mxl_lrc_config,
)
from ultrastar_generator.lrc_timing import two_tier_time_calibration
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

word_lines, word_clean_text = assign_words_to_lines(mlg_words, mlg_lrc_lines)
assert word_lines == [0, 0, 1, 1, 1], word_lines
assert word_clean_text == ["hello", "world", "good", "bye", "now"], word_clean_text
print("OK: assign_words_to_lines correctly tags each word with its own LRC line index AND its own "
      "matched clean LRC token text")

# Fuzzy matching: an MXL word that's close-but-not-exact to a single LRC
# word, anchored by correctly-matched context on both sides (a real 1:1
# "replace" slot), is a real confirmed case -- "systern" for "system" --
# and must get the CLEAN text, not stay stuck on OCR garbage.
fuzzy_words = [
    MxlWord(text="the", norm="the", offset=0.0, syllables=[(0.0, 1.0, 60, "the")]),
    MxlWord(text="systern", norm="systern", offset=1.0,
            syllables=[(1.0, 0.5, 60, "sys"), (1.5, 0.5, 61, "tern")]),
    MxlWord(text="works", norm="works", offset=2.0, syllables=[(2.0, 1.0, 60, "works")]),
]
fuzzy_lines = [(10.0, "the system works")]
_, fuzzy_clean = assign_words_to_lines(fuzzy_words, fuzzy_lines)
assert fuzzy_clean == ["the", "system", "works"], fuzzy_clean
print("OK: an OCR-garbled MXL word in a 1:1 replace slot gets fuzzy-matched to the clean LRC text "
      "('systern' -> 'system'), not left stuck on the MXL's own OCR garbage")

# But a GENUINELY different word in the same kind of slot (not just an OCR
# spelling variant) must NOT be fuzzy-matched -- the ratio gate has to
# actually reject low-similarity pairs, not just be a formality.
unrelated_words = [
    MxlWord(text="the", norm="the", offset=0.0, syllables=[(0.0, 1.0, 60, "the")]),
    MxlWord(text="xyz", norm="xyz", offset=1.0, syllables=[(1.0, 1.0, 60, "xyz")]),
    MxlWord(text="works", norm="works", offset=2.0, syllables=[(2.0, 1.0, 60, "works")]),
]
_, unrelated_clean = assign_words_to_lines(unrelated_words, fuzzy_lines)
assert unrelated_clean == ["the", None, "works"], unrelated_clean
print("OK: a genuinely unrelated word in the same kind of slot is correctly REJECTED by the similarity "
      "ratio gate, not fuzzy-matched just because it landed in a replace slot")

# BUG REGRESSION (real cases, "Great Big Sea - Ordinary Day", lrclib id
# 6210269): a REPLACE block that isn't a clean 1:1 shape used to be left
# entirely unmatched, even when both sides are anchored by real matches
# and the whole block's content is clearly the same, just OCR-garbled or
# word-segmented differently than the real lyric. Three real shapes, all
# from this one song's actual MXL:
from ultrastar_generator.mxl_lrc_generator import _distribute_words_to_slots

# 1: N -- one MXL word (OCR-merged) covers TWO real LRC words ("winnes"
# for "win now"). Only one display slot exists, so the two real words are
# joined with a space into that one slot.
merge_words = [
    MxlWord(text="I", norm="i", offset=0.0, syllables=[(0.0, 1.0, 60, "I")]),
    MxlWord(text="winnes", norm="winnes", offset=1.0, syllables=[(1.0, 1.5, 69, "winnes")]),
    MxlWord(text="and", norm="and", offset=2.5, syllables=[(2.5, 1.0, 60, "and")]),
]
merge_lines = [(10.0, "I win now and")]
_, merge_clean = assign_words_to_lines(merge_words, merge_lines)
assert merge_clean == ["I", "win now", "and"], merge_clean

# N: N (same count, individually too garbled) -- "stomty"+"in" for
# "stop"+"trying," -- positional 1:1 once the WHOLE block is compared.
same_count_words = [
    MxlWord(text="won't", norm="wont", offset=0.0, syllables=[(0.0, 1.0, 60, "won't")]),
    MxlWord(text="stomty", norm="stomty", offset=1.0,
            syllables=[(1.0, 0.5, 67, "stom"), (1.5, 1.0, 67, "ty")]),
    MxlWord(text="in", norm="in", offset=2.5, syllables=[(2.5, 3.0, 69, "in")]),
    MxlWord(text="Oh", norm="oh", offset=5.5, syllables=[(5.5, 1.0, 60, "Oh")]),
]
same_count_lines = [(10.0, "won't stop trying, oh")]
_, same_count_clean = assign_words_to_lines(same_count_words, same_count_lines)
assert same_count_clean == ["won't", "stop", "trying,", "oh"], same_count_clean

# N: M -- three MXL words for two real (one hyphenated) LRC words
# ("double"+"edged"+"kide" for "double-edged"+"knife,") -- fewer real
# words than slots, recovered by splitting the hyphenated word first
# rather than falling straight to melisma-padding.
split_words = [
    MxlWord(text="a", norm="a", offset=0.0, syllables=[(0.0, 1.0, 60, "a")]),
    MxlWord(text="double", norm="double", offset=1.0, syllables=[(1.0, 1.0, 60, "double")]),
    MxlWord(text="edged", norm="edged", offset=2.0, syllables=[(2.0, 1.0, 60, "edged")]),
    MxlWord(text="kide", norm="kide", offset=3.0, syllables=[(3.0, 1.0, 60, "kide")]),
    MxlWord(text="but", norm="but", offset=4.0, syllables=[(4.0, 1.0, 60, "but")]),
]
split_lines = [(10.0, "a double-edged knife, but")]
_, split_clean = assign_words_to_lines(split_words, split_lines)
assert split_clean == ["a", "double", "edged", "knife,", "but"], split_clean
print("OK: assign_words_to_lines recovers a MULTI-word replace block anchored by real matches on both "
      "sides -- 1-MXL-word-merges-2-real-words, same-count-but-individually-garbled, and "
      "fewer-real-words-than-slots-via-hyphen-split, all real 'Ordinary Day' cases")

# A genuinely unrelated multi-word block (not just OCR noise) must still
# be rejected -- the block-level ratio gate has to actually reject, not
# just be a formality the way the 1:1 gate already is.
unrelated_block_words = [
    MxlWord(text="the", norm="the", offset=0.0, syllables=[(0.0, 1.0, 60, "the")]),
    MxlWord(text="zzz", norm="zzz", offset=1.0, syllables=[(1.0, 1.0, 60, "zzz")]),
    MxlWord(text="qqq", norm="qqq", offset=2.0, syllables=[(2.0, 1.0, 60, "qqq")]),
    MxlWord(text="works", norm="works", offset=3.0, syllables=[(3.0, 1.0, 60, "works")]),
]
unrelated_block_lines = [(10.0, "the completely unrelated text works")]
_, unrelated_block_clean = assign_words_to_lines(unrelated_block_words, unrelated_block_lines)
assert unrelated_block_clean[1] is None and unrelated_block_clean[2] is None, unrelated_block_clean
print("OK: a genuinely unrelated multi-word block is correctly rejected by the block-level similarity "
      "ratio gate too, not fuzzy-matched just because it's bounded by real anchors")

# _distribute_words_to_slots directly: the merge (more real words than
# slots) and melisma-pad (fewer real words than slots, no hyphen to
# split) fallback shapes, isolated from the whole-block-matching path.
assert _distribute_words_to_slots(["stop", "trying,"], 2) == ["stop", "trying,"]
assert _distribute_words_to_slots(["win", "now"], 1) == ["win now"]
assert _distribute_words_to_slots(["double-edged", "knife,"], 3) == ["double", "edged", "knife,"]
assert _distribute_words_to_slots(["go"], 3) == ["go", mxl_lrc_config.MELISMA_CONTINUATION_TEXT,
                                                  mxl_lrc_config.MELISMA_CONTINUATION_TEXT]
print("OK: _distribute_words_to_slots handles more-words-than-slots (merge), fewer (hyphen-split, then "
      "melisma-pad), and equal counts (direct positional) correctly in isolation")

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

# BUG REGRESSION (real case: Chicago "favors" OCR'd as "favere" in the MXL,
# but transcribed correctly by ASR) -- matching on the raw MXL norm alone
# missed this word entirely, even though assign_words_to_lines had already
# resolved a clean "favors" for it. place_words_via_asr must reuse that
# clean text for its own ASR matching, not just for display.
garbled_mxl_words = [
    MxlWord(text="hello", norm="hello", offset=0.0, syllables=[(0.0, 1.0, 64, "hello")]),
    MxlWord(text="favere", norm="favere", offset=1.0, syllables=[(1.0, 0.5, 64, "fa"), (1.5, 1.0, 64, "vors")]),
]
garbled_word_lines = [0, 0]
garbled_clean_text = ["hello", "favors"]  # as assign_words_to_lines would have resolved it
garbled_lrc_lines = [(10.0, "hello favors")]
garbled_asr = [
    _Word(text="hello", start=10.0, end=10.3),
    _Word(text="favors", start=11.0, end=11.6),
]
g_starts, g_ends, g_quality = place_words_via_asr(
    garbled_mxl_words, garbled_word_lines, garbled_lrc_lines, garbled_asr, word_clean_text=garbled_clean_text)
assert g_starts[1] == 11.0 and g_ends[1] == 11.6, (g_starts[1], g_ends[1])
assert g_quality.n_asr_placed == 2 and g_quality.n_fallback == 0, g_quality
# Without the clean text (old behavior), a DOUBLY-garbled word -- MXL OCR'd
# it as "favere" AND ASR separately mis-transcribed it as "favorites" (real
# case: the user's own re-run) -- must NOT match, since "favere"~"favorites"
# (ratio 0.53) falls below the fuzzy threshold even with the fuzzy-replace
# fallback active. Confirms clean-text reuse is still doing real work of its
# own, not just fuzzy matching alone.
mishear_asr = [
    _Word(text="hello", start=10.0, end=10.3),
    _Word(text="favorites", start=11.0, end=11.6),
]
g2_starts, g2_ends, g2_quality = place_words_via_asr(
    garbled_mxl_words, garbled_word_lines, garbled_lrc_lines, mishear_asr)
assert g2_quality.n_asr_placed == 1 and g2_quality.n_fallback == 1, g2_quality
print("OK: place_words_via_asr matches an MXL word against ASR using its already-resolved CLEAN text "
      "(\"favere\"->\"favors\") when available, recovering a real confident ASR match that raw-OCR-norm "
      "matching alone would miss entirely")

# BUG REGRESSION (real case: the user's own re-run mis-transcribed "favors"
# as "favorites" -- a real ASR mishearing, independent of any MXL OCR
# issue) -- even the clean text ("favors") doesn't EXACTLY match ASR's own
# output ("favorites") here, so the exact-match fix above isn't enough on
# its own; a close-but-not-identical 1:1 pairing must still be trusted via
# the same fuzzy-ratio technique assign_words_to_lines already uses for
# display text. Reuses the same `mishear_asr` data used above to prove
# clean-text-reuse alone isn't enough for THIS word -- fuzzy matching
# against the clean text is what closes the gap.
m_starts, m_ends, m_quality = place_words_via_asr(
    garbled_mxl_words, garbled_word_lines, garbled_lrc_lines, mishear_asr, word_clean_text=garbled_clean_text)
assert m_starts[1] == 11.0 and m_ends[1] == 11.6, (m_starts[1], m_ends[1])
assert m_quality.n_asr_placed == 2 and m_quality.n_fallback == 0, m_quality
# A genuinely unrelated ASR word in the same slot must still be rejected --
# confirms this isn't just accepting any 1:1 replace pairing.
unrelated_asr = [
    _Word(text="hello", start=10.0, end=10.3),
    _Word(text="banana", start=11.0, end=11.6),
]
u_starts, u_ends, u_quality = place_words_via_asr(
    garbled_mxl_words, garbled_word_lines, garbled_lrc_lines, unrelated_asr, word_clean_text=garbled_clean_text)
assert u_quality.n_asr_placed == 1 and u_quality.n_fallback == 1, u_quality
print("OK: place_words_via_asr also trusts a close-but-not-identical 1:1 ASR pairing (\"favors\"~\"favorites\", "
      "a real ASR mishearing) via the same fuzzy-ratio technique used for display text, while still rejecting "
      "a genuinely unrelated word in the same slot")

# BUG REGRESSION (real case: the user's own re-run, "There's a lot of
# favors, I'm prepared..." -- the fuzzy-replace fix above only checked a
# CLEAN 1:1 replace block, but `asr_in_window` is time-bounded, not
# line-bounded (a deliberate +-0.5s slop so a match landing just outside
# the LRC line's own window isn't missed) -- so a word belonging to the
# NEXT line ("I'm") can spill into the same window and turn what should be
# a clean 1:1 mismatch into a 1:2 replace block (['favors'] vs
# ['favorites', 'im']), which the old `(b2 - b1) == 1` check rejected
# outright even though the correct candidate ("favorites") was sitting
# right there at the start of the block. Real debug-log-confirmed case:
# this silently fell through to nearest-anchor interpolation and produced
# a ~1.85s span for a word whose real ASR duration was 0.66s.
spillover_lrc_lines = [(10.0, "hello favors"), (12.0, "im here")]
spillover_asr = [
    _Word(text="hello", start=10.0, end=10.3),
    _Word(text="favorites", start=11.0, end=11.6),
    _Word(text="im", start=12.0, end=12.3),
]
sp_starts, sp_ends, sp_quality = place_words_via_asr(
    garbled_mxl_words, garbled_word_lines, spillover_lrc_lines, spillover_asr, word_clean_text=garbled_clean_text)
assert sp_starts[1] == 11.0 and sp_ends[1] == 11.6, (sp_starts[1], sp_ends[1])
assert sp_quality.n_asr_placed == 2 and sp_quality.n_fallback == 0, sp_quality
print("OK: place_words_via_asr still recovers a fuzzy 1:1 match when a NEXT-line word spills into the same "
      "ASR time window, turning the replace block 1:2 instead of 1:1 (real case: 'favors'~'favorites' "
      "alongside a spilled-over \"I'm\")")

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

syllables_out = build_syllables(mlg_words, starts, ends, word_lines, word_clean_text)
assert len(syllables_out) == 5
assert syllables_out[0].line_id == 0 and syllables_out[2].line_id == 1
assert all(syllables_out[i].start <= syllables_out[i + 1].start for i in range(4))
print("OK: build_syllables tags line_id from assign_words_to_lines and produces monotonic syllable starts")

print("\n--- mxl_lrc_generator: _text_for_mxl_syllables substitutes clean LRC text over MXL's own OCR text ---")
# Exact syllable-count match: clean hyphenation used directly.
exact = _text_for_mxl_syllables("system", ["sys", "tern"])  # MXL notated 2 syllables, garbled 2nd one
assert exact != ["sys", "tern"], exact  # must NOT be the garbled MXL text
assert "".join(exact) == "system", exact  # reconstructs the CLEAN word, not the garbled one
# Fewer MXL syllables than the clean word hyphenates into: merged down via chunk_to_count.
merged = _text_for_mxl_syllables("reciprocity", ["a"])  # MXL notated only 1 syllable for this word
assert len(merged) == 1 and merged[0] == "reciprocity", merged
# More MXL syllables than the clean word hyphenates into: melisma continuation markers pad the rest.
melisma = _text_for_mxl_syllables("go", ["g", "o", "o", "o"])  # MXL notated 4 notes, word is 1 syllable
assert len(melisma) == 4, melisma
assert melisma[0] == "go" and melisma[1:] == [mxl_lrc_config.MELISMA_CONTINUATION_TEXT] * 3, melisma
# No clean match at all: MXL's own raw text is the only option, used as-is.
no_match = _text_for_mxl_syllables(None, ["sys", "tern"])
assert no_match == ["sys", "tern"], no_match
print("OK: _text_for_mxl_syllables prefers clean LRC text reconciled to the MXL's own syllable count "
      "(merging or melisma-padding as needed), falling back to MXL's own raw text only when no clean "
      "match exists at all")

print("\n--- mxl_lrc_generator: nearest-anchor interpolation replaces whole-line-stretch fallback ---")
# Reproduces the real confirmed bug: an LRC line whose own window includes a
# long trailing silence (an instrumental gap before the NEXT line) used to
# stretch every un-ASR-matched word in it across the WHOLE window, pushing
# them far later than their real position. One line, 4 MXL words: "one" gets
# a real ASR match early, "two"/"three" have none (must fall back), "four"
# gets a real ASR match near the START of the line's own real content --
# even though the LRC line's own declared window extends much further
# (mimicking a long trailing rest before the next line).
anchor_words = [
    MxlWord(text="one", norm="one", offset=0.0, syllables=[(0.0, 1.0, 60, "one")]),
    MxlWord(text="two", norm="two", offset=1.0, syllables=[(1.0, 1.0, 60, "two")]),
    MxlWord(text="three", norm="three", offset=2.0, syllables=[(2.0, 1.0, 60, "three")]),
    MxlWord(text="four", norm="four", offset=3.0, syllables=[(3.0, 1.0, 60, "four")]),
]
# Line window is 0s-20s (a long trailing silence before the next line), but
# the real singing (per the ASR anchors) only spans 0s-1.6s.
anchor_lines = [(0.0, "one two three four"), (20.0, "next line")]
anchor_asr = [
    _Word(text="one", start=0.0, end=0.4),
    _Word(text="four", start=1.3, end=1.6),
]
a_word_lines, _ = assign_words_to_lines(anchor_words, anchor_lines)
a_starts, a_ends, a_quality = place_words_via_asr(anchor_words, a_word_lines, anchor_lines, anchor_asr)
# "two"/"three" must land BETWEEN "one" (0.0) and "four" (1.3) -- a locally
# sane position -- NOT stretched out toward the line's own 20s-wide window.
assert 0.0 <= a_starts[1] <= a_starts[2] <= a_starts[3], a_starts
assert a_starts[3] == 1.3, a_starts
assert a_starts[1] < 5.0 and a_starts[2] < 5.0, a_starts  # nowhere close to the 20s line-window stretch
print(f"OK: fallback words between two confident anchors interpolate from the LOCAL gap (starts={a_starts}), "
      f"not the whole line's window including its trailing silence")

print("\n--- lrc_timing: match_asr_to_lrc_lines + two_tier_time_calibration recover a systematic "
      "LRC/audio offset -- BUG REGRESSION for real 'Ordinary Day' (lrclib id 6210269) case, where our own "
      "audio has ~2.4s of extra lead-in silence vs. whichever recording LRCLIB's synced lyrics were timed "
      "against ---")
from ultrastar_generator.lrc_timing import match_asr_to_lrc_lines

# 6 LRC lines 2s apart (distinct single-word content so text matching is
# unambiguous); real ASR content for each line is a CONSTANT +3.0s later
# than the LRC line's own declared timestamp -- larger than the per-line
# window's own +-0.5s slop plus the 2s line gap, so an UNCALIBRATED window
# search genuinely misses every one of these real matches, not just some.
off_lrc_lines = [(0.0, "alpha"), (2.0, "bravo"), (4.0, "charlie"), (6.0, "delta"), (8.0, "echo"), (10.0, "foxtrot")]
off_asr = [
    _Word(text="alpha", start=3.0, end=3.3),
    _Word(text="bravo", start=5.0, end=5.3),
    _Word(text="charlie", start=7.0, end=7.3),
    _Word(text="delta", start=9.0, end=9.3),
    _Word(text="echo", start=11.0, end=11.3),
    _Word(text="foxtrot", start=13.0, end=13.3),
]
off_candidates = match_asr_to_lrc_lines(off_asr, off_lrc_lines)
assert len(off_candidates) == 6, off_candidates
assert all(abs(delta - 3.0) < 1e-9 for _, _, delta in off_candidates), off_candidates
off_offset, off_slope, off_confidence, off_kind, off_skipped, off_fn, _off_holdout = two_tier_time_calibration(off_candidates)
assert off_offset == 3.0 and off_kind == "constant" and off_confidence == 1.0, (off_offset, off_kind, off_confidence)
assert off_fn is not None and abs(off_fn(10.0) - 13.0) < 1e-9, off_fn(10.0)
print("OK: match_asr_to_lrc_lines recovers a per-line real-ASR-vs-LRC delta straight from ASR's own flat "
      "word stream, and two_tier_time_calibration confidently calibrates the constant +3.0s offset from it "
      "(correction_fn agrees: 10.0 -> 13.0)")

print("\n--- lrc_timing: THIRD tier (piecewise/isotonic) -- a DISCONTINUOUS drift (real edit difference "
      "between recordings, e.g. a chorus removed/bridge shortened) that neither tier 1 (constant) nor "
      "tier 2 (single linear slope) can fit ---")
from ultrastar_generator.lrc_timing import (
    _enforce_monotonic_anchors, _pava_isotonic, _piecewise_or_isotonic_calibration, _robust_linear_fit,
    _correction_from_anchors,
)

print("  _enforce_monotonic_anchors ('piecewise' strategy) DROPS a real monotonicity violator (an anchor "
      "whose implied real time is earlier than an already-kept earlier anchor's), 'isotonic' (PAVA) POOLS "
      "it with its neighbor instead of discarding it:")
mono_cands = [(0, 0.0, 5.0), (1, 10.0, 5.0), (2, 20.0, 5.0), (3, 30.0, -15.0), (4, 40.0, 5.0), (5, 50.0, 5.0)]
# index 3's implied real time (30 + -15 = 15.0) is EARLIER than index 2's (20+5=25.0) -- a genuine violation.
piecewise_anchors = _enforce_monotonic_anchors(mono_cands)
isotonic_anchors = _pava_isotonic(mono_cands)
assert piecewise_anchors == [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0), (40.0, 45.0), (50.0, 55.0)], piecewise_anchors
# PAVA pools the violator (30, 15) with its IMMEDIATE predecessor (20, 25) into one block
# (mean_x=25.0, mean_y=20.0) -- that merged mean (20.0) is still >= the block before it ((10,15),
# mean 15.0), so the merge doesn't cascade any further back; (0,5) and (10,15) are untouched.
assert isotonic_anchors == [(0.0, 5.0), (10.0, 15.0), (25.0, 20.0), (40.0, 45.0), (50.0, 55.0)], isotonic_anchors
print("  OK: piecewise drops the violator entirely (5 anchors, jumps straight from (20,25) to (40,45)); "
      "isotonic instead POOLS it with its predecessor (20,25) into one blended (25.0, 20.0) anchor (also "
      "5 anchors, but a smoothed value instead of an outright drop)")

print("  _correction_from_anchors: linear interpolation between anchors, extrapolation past the first/last "
      "using that boundary segment's own local slope:")
interp_fn = _correction_from_anchors([(0.0, 10.0), (10.0, 20.0), (30.0, 30.0)])
assert abs(interp_fn(5.0) - 15.0) < 1e-9, interp_fn(5.0)     # midpoint of first segment (slope 1.0)
assert abs(interp_fn(20.0) - 25.0) < 1e-9, interp_fn(20.0)   # midpoint of second segment (slope 0.5)
assert abs(interp_fn(-10.0) - 0.0) < 1e-9, interp_fn(-10.0)  # extrapolated BEFORE the first anchor, slope 1.0
assert abs(interp_fn(40.0) - 35.0) < 1e-9, interp_fn(40.0)   # extrapolated AFTER the last anchor, slope 0.5
print("  OK: interpolates exactly between anchors and extrapolates past both ends using the boundary "
      "segment's own local slope, not a flat hold or the whole-song average slope")

print("  _piecewise_or_isotonic_calibration: gated on BOTH a minimum anchor count and a maximum gap "
      "between adjacent anchors -- either failing means 'fall through to uncalibrated', not a degraded guess:")
few_cands = [(i, float(i * 10), 2.0) for i in range(5)]
few_fit = _robust_linear_fit(few_cands, 4.0)
assert _piecewise_or_isotonic_calibration(few_cands, few_fit, min_anchors=6) is None
print("  OK: fewer than min_anchors surviving candidates -> tier 3 declines (5 < 6)")

gapped_cands = [(i, float(i * 5), 2.0) for i in range(4)] + [(4, 100.0, 2.0), (5, 105.0, 2.0)]
gapped_fit = _robust_linear_fit(gapped_cands, 4.0)
assert _piecewise_or_isotonic_calibration(
    gapped_cands, gapped_fit, min_anchors=6, max_anchor_gap_sec=45.0) is None
print("  OK: 6 anchors but one adjacent gap (15.0 -> 100.0, 85s) exceeds max_anchor_gap_sec -> tier 3 "
      "declines rather than interpolate across a gap 2 points can't distinguish from noise")

reasonable_cands = [(i, float(i * 8), 2.0) for i in range(6)]
reasonable_fit = _robust_linear_fit(reasonable_cands, 4.0)
ok_result = _piecewise_or_isotonic_calibration(reasonable_cands, reasonable_fit, min_anchors=6, max_anchor_gap_sec=45.0)
assert ok_result is not None and ok_result[3] == "isotonic", ok_result
print("  OK: 6 anchors within a reasonable adjacent gap -> tier 3 succeeds (default drift_model='isotonic')")

print("  two_tier_time_calibration end-to-end: a genuine 3-segment DISCONTINUOUS drift (simulating 2 real "
      "edit-cut points) that defeats both tier 1 (no single bucket clears 40% -- 3 roughly-equal segments) "
      "and tier 2 (a single global slope badly misfits 3 flat plateaus) is recovered by tier 3 instead of "
      "reported as 'uncalibrated':")
disc_cands = []
for i in range(7):
    disc_cands.append((i, float(i * 5), 2.0))     # segment A: lrc_start 0-30, delta +2.0
for i in range(7, 14):
    disc_cands.append((i, float(i * 5), 10.0))    # segment B: lrc_start 35-65, delta +10.0 (jump)
for i in range(14, 20):
    disc_cands.append((i, float(i * 5), 20.0))    # segment C: lrc_start 70-95, delta +20.0 (jump)
disc_offset, disc_slope, disc_confidence, disc_kind, disc_skipped, disc_fn, disc_holdout = two_tier_time_calibration(disc_cands)
assert disc_kind == "isotonic", (disc_kind, disc_offset, disc_confidence, disc_skipped)
assert disc_offset is not None and disc_fn is not None
# mid-plateau points should recover their own segment's real offset closely
assert abs(disc_fn(0.0) - 2.0) < 1.0, disc_fn(0.0)      # segment A: 0 + 2.0
assert abs(disc_fn(50.0) - 60.0) < 2.0, disc_fn(50.0)    # segment B: 50 + 10.0
assert abs(disc_fn(95.0) - 115.0) < 1.0, disc_fn(95.0)   # segment C: 95 + 20.0
print(f"  OK: tier 1/2 both declined this data, tier 3 ({disc_kind}) recovered a piecewise correction "
      f"tracking all 3 segments ({disc_confidence:.0%} of candidates passed the Theil-Sen inlier filter, "
      f"holdout residual {disc_holdout}) instead of reporting 'uncalibrated'")

print("  two_tier_time_calibration: this 3-segment case is a REFINE, not a rescue -- tier 1's best (still-"
      "rejected) bucket confidence (35%) and tier 2's own Theil-Sen fit (45%) both already clear the "
      "rescue floor (30%), so no structural_check is needed for it to succeed (confirms the split doesn't "
      "regress the ordinary 'two independent rigid models already partially agree' case):")
refine_offset, refine_slope, refine_confidence, refine_kind, refine_skipped, refine_fn, _rh = two_tier_time_calibration(
    disc_cands, structural_check=None)
assert refine_kind == "isotonic" and refine_offset is not None, (refine_kind, refine_skipped)
print("  OK: refine case succeeds with no structural_check at all")

print("  two_tier_time_calibration: a genuine RESCUE case (tier 1/2 both near 0%, no independent support) "
      "declines by default (no structural_check given) even though tier 3 itself finds a fit -- this is "
      "the real Heroes-shaped failure mode this split exists to close:")
import random as _random
_rng = _random.Random(42)
rescue_cands = []
for i in range(30):
    # scattered deltas (tier1/tier2 find no real pattern) EXCEPT a clean monotonic
    # sub-pattern buried in it that a flexible isotonic/piecewise fit can still track --
    # mimics "flexible model finds SOME fit even though rigid models found nothing".
    rescue_cands.append((i, float(i * 3), (i % 5) * 7.0 + _rng.uniform(-0.1, 0.1)))
r_offset, r_slope, r_confidence, r_kind, r_skipped, r_fn, r_holdout = two_tier_time_calibration(
    rescue_cands, min_calibration_samples=5, min_drift_samples=5)
assert r_offset is None and r_kind is None, (r_offset, r_kind, r_skipped)
assert r_skipped is not None and "rescue" in r_skipped, r_skipped
print(f"  OK: declined ({r_skipped!r}) -- no structural_check means an unverified rescue is never accepted")

print("  two_tier_time_calibration: the SAME rescue case is accepted when a structural_check is provided "
      "AND it passes -- and still declined when the check itself rejects it:")
r2_offset, r2_slope, r2_confidence, r2_kind, r2_skipped, r2_fn, r2_holdout = two_tier_time_calibration(
    rescue_cands, min_calibration_samples=5, min_drift_samples=5, structural_check=lambda: None)
assert r2_offset is not None and r2_kind in ("isotonic", "piecewise"), (r2_offset, r2_kind, r2_skipped)
r3_offset, r3_slope, r3_confidence, r3_kind, r3_skipped, r3_fn, r3_holdout = two_tier_time_calibration(
    rescue_cands, min_calibration_samples=5, min_drift_samples=5,
    structural_check=lambda: "fake rejection reason for this test")
assert r3_offset is None and "fake rejection reason" in r3_skipped, r3_skipped
print("  OK: structural_check passing (returns None) -> rescue accepted; structural_check rejecting "
      "(returns a reason string) -> rescue declined, reason surfaced in skipped_reason")

print("  two_tier_time_calibration: 'piecewise' selectable via drift_model= (config.LRC_TIMING_DRIFT_MODEL "
      "default is 'isotonic', but a caller/config override must be honored):")
pw_offset, pw_slope, pw_confidence, pw_kind, pw_skipped, pw_fn, pw_holdout = two_tier_time_calibration(
    disc_cands, drift_model="piecewise")
assert pw_kind == "piecewise", pw_kind
print("  OK: drift_model='piecewise' produces kind='piecewise' on the same discontinuous data")

off_mxl_words = [
    MxlWord(text=t, norm=t, offset=float(i), syllables=[(float(i), 1.0, 60, t)])
    for i, t in enumerate(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"])
]
off_word_lines = list(range(6))
# UNCALIBRATED: raw LRC timestamps used directly -- every real ASR match
# falls outside its own line's +-0.5s window (declared line span is only
# 2s, offset is 3.0s), so this must fail almost entirely.
raw_starts, raw_ends, raw_quality = place_words_via_asr(off_mxl_words, off_word_lines, off_lrc_lines, off_asr)
assert raw_quality.asr_placement_rate < 0.5, raw_quality
# CALIBRATED: shift LRC timestamps by the recovered offset first (exactly
# what generate_from_mxl_and_lrc now does) -- every word should now match
# confidently via real ASR timing.
cal_lrc_lines = [(t + off_offset + off_slope * t, text) for t, text in off_lrc_lines]
cal_starts, cal_ends, cal_quality = place_words_via_asr(off_mxl_words, off_word_lines, cal_lrc_lines, off_asr)
assert cal_quality.asr_placement_rate == 1.0, cal_quality
assert cal_starts == [3.0, 5.0, 7.0, 9.0, 11.0, 13.0], cal_starts
print("OK: applying the recovered offset to LRC line timestamps before matching turns an almost-total ASR "
      "placement failure (uncalibrated, {:.0%}) into a full recovery (calibrated, {:.0%}) -- the real fix "
      "for the 'Ordinary Day' lead-in-silence case".format(raw_quality.asr_placement_rate, cal_quality.asr_placement_rate))

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

print("\n--- load_mxl_vocal_words: untexted continuation notes (tied hold / slurred slide) are kept, "
      "not silently dropped -- BUG REGRESSION for real 'reciprocity' (G#->C# slide+fermata) and "
      "'fa'/'favors' (tied hold undershooting duration) cases ---")
import tempfile as _tempfile
import os as _os
import music21 as _music21
from ultrastar_generator.mxl_lrc_generator import load_mxl_vocal_words

_lmvw_part = _music21.stream.Part()
_lmvw_part.partName = "Voice 1"

# "go" -- one lyric-bearing note (begin) tied to an untexted same-pitch note
# (a real sustain/hold -- must MERGE into one syllable, extended duration,
# not become a second note).
_n1 = _music21.note.Note(60, quarterLength=1.0)  # C4
_n1.lyric = "go"
_lmvw_part.append(_n1)
_n2 = _music21.note.Note(60, quarterLength=1.0)  # same pitch, tied continuation
_n2.tie = _music21.tie.Tie("stop")
_lmvw_part.append(_n2)

# "up" -- one lyric-bearing note (begin) slurred (untied) into an untexted
# DIFFERENT-pitch note immediately after -- a real slide -- must become a
# SECOND syllable entry (empty text), not be dropped.
_n3 = _music21.note.Note(62, quarterLength=1.0)  # D4
_n3.lyric = "up"
_lmvw_part.append(_n3)
_n4 = _music21.note.Note(58, quarterLength=1.0)  # different pitch, no tie
_lmvw_part.append(_n4)

# A rest, then an untexted note with no word in progress boundary-wise --
# this note follows a REST (non-contiguous), so it must NOT be glued onto
# "up"'s melisma even though no new lyric appeared in between yet.
_lmvw_part.append(_music21.note.Rest(quarterLength=2.0))
_n5 = _music21.note.Note(65, quarterLength=1.0)  # no lyric, non-contiguous
_lmvw_part.append(_n5)

# Next real word closes out "up"'s word.
_n6 = _music21.note.Note(64, quarterLength=1.0)
_n6.lyric = "down"
_lmvw_part.append(_n6)

_lmvw_score = _music21.stream.Score()
_lmvw_score.append(_lmvw_part)

with _tempfile.TemporaryDirectory() as _tmpdir:
    _mxl_path = _os.path.join(_tmpdir, "test.musicxml")
    _lmvw_score.write("musicxml", fp=_mxl_path)
    lmvw_words, lmvw_parts = load_mxl_vocal_words(_mxl_path)

assert [w.text for w in lmvw_words] == ["go", "up", "down"], [w.text for w in lmvw_words]
go_word = lmvw_words[0]
assert go_word.syllables == [(0.0, 2.0, 60, "go")], go_word.syllables
up_word = lmvw_words[1]
assert up_word.syllables == [(2.0, 1.0, 62, "up"), (3.0, 1.0, 58, "")], up_word.syllables
print("OK: tied same-pitch continuation merges into one extended-duration syllable "
      f"({go_word.syllables}); slurred different-pitch continuation becomes a real second "
      f"syllable ({up_word.syllables}); a non-contiguous untexted note (after a rest) is "
      "correctly left unattached")

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
    mp3 = in_dir / "some_track.mp3"  # deliberately NOT "<Artist> - <Title>" -- a folder-based
    mp3.write_bytes(b"mp3-bytes")     # input's own files can be named anything at all
    video = in_dir / "clip.mp4"
    video.write_bytes(b"video-bytes")
    cover = in_dir / "random_pic.jpg"
    cover.write_bytes(b"cover-bytes")
    bg = in_dir / "another_pic.png"
    bg.write_bytes(b"bg-bytes")

    staged = stage_companions_to_output(out_dir, "Some Artist", "Some Song",
                                         mp3_src=mp3, video_src=video, cover_src=cover, background_src=bg)
    assert staged.mp3 == "Some Artist - Some Song.mp3", staged.mp3
    assert staged.video == "Some Artist - Some Song.mp4", staged.video
    assert staged.cover == "Some Artist - Some Song [CO].jpg", staged.cover
    assert staged.background == "Some Artist - Some Song [BG].png", staged.background
    assert (out_dir / "Some Artist - Some Song.mp3").read_bytes() == b"mp3-bytes"
    assert (out_dir / "Some Artist - Some Song.mp4").read_bytes() == b"video-bytes"
    assert (out_dir / "Some Artist - Some Song [CO].jpg").read_bytes() == b"cover-bytes"
    assert (out_dir / "Some Artist - Some Song [BG].png").read_bytes() == b"bg-bytes"
print("OK: every companion is renamed to '<Artist> - <Title>[.ext]' in the output folder regardless "
      "of its own name in the input folder; separate cover/background images each keep their own "
      "'[CO]'/'[BG]' tag")

with _tempfile.TemporaryDirectory() as root:
    root = Path(root)
    in_dir = root / "in"
    out_dir = root / "out"
    in_dir.mkdir()
    mp4 = in_dir / "video_file.mp4"
    mp4.write_bytes(b"mp4-bytes")

    staged = stage_companions_to_output(out_dir, "Some Artist", "Some Song", mp3_src=mp4, video_src=mp4)
    assert staged.mp3 == staged.video == "Some Artist - Some Song.mp4", staged.mp3
    assert len(list(out_dir.iterdir())) == 1, "identical mp3/video source must only be copied ONCE"
print("OK: identical mp3_src/video_src (mp4-as-audio case) copied exactly once, renamed once, both "
      "roles reference it")

with _tempfile.TemporaryDirectory() as root:
    root = Path(root)
    in_dir = root / "in"
    out_dir = root / "out"
    in_dir.mkdir()
    mp3b = in_dir / "song.mp3"
    mp3b.write_bytes(b"mp3-bytes")
    pic = in_dir / "cover.jpg"
    pic.write_bytes(b"pic-bytes")

    staged = stage_companions_to_output(out_dir, "Some Artist", "Some Song",
                                         mp3_src=mp3b, cover_src=pic, background_src=pic)
    assert staged.cover == staged.background == "Some Artist - Some Song.jpg", staged.cover
    assert len(list(out_dir.iterdir())) == 2, "identical cover/background source must only be copied ONCE"
print("OK: a single image serving both cover and background roles is renamed WITHOUT a '[CO]'/'[BG]' "
      "tag (nothing to disambiguate), matching find_companions' own single-untagged-image convention")

print("\n--- main.delete_work_files: deletes the entire work_dir, debug files included (intentional) ---")
import tempfile as _tempfile_delint
from ultrastar_generator.main import delete_work_files as _delete_intermediates
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

    assert not work_dir.exists(), "the whole work_dir, including debug files, should be gone"
print("OK: delete_work_files removes the entire work_dir, debug files included")

# A work_dir that doesn't exist at all must not raise.
with _tempfile_delint.TemporaryDirectory() as tmp:
    missing_work_dir = Path(tmp) / ".ultrastar_work"
    _delete_intermediates(missing_work_dir)  # must not raise
print("OK: delete_work_files on a work_dir that doesn't exist is a silent no-op")

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

# Case 5: the matched SUBSET looks perfect (same 15 words as case 1), but the
# fresh run also has a bunch of words that never match the existing file at
# all (e.g. garbled/wrong output text) -- real bug this coverage gate was
# added for: pitch/timing accuracy alone can't see this, since a word that
# never text-matches simply never becomes a candidate in the first place.
garbled_extra = [(f"garbled{i}", 20.0 + i, 0) for i in range(10)]
fresh_with_garbage = fresh_ok + _mk_word_syllables(garbled_extra)
result = verify_existing_song(existing_ok, fresh_with_garbage, min_matched=10, verbose=True)
assert result.pitch_class_accuracy == 1.0 and result.timing_within_tolerance_pct == 1.0, result
assert result.coverage_fresh < mxl_lrc_config.EXISTING_TXT_MIN_COVERAGE, result.coverage_fresh
assert result.verdict == "PROBLEMS_FOUND", result  # must NOT report PASS despite perfect matched-subset accuracy
assert set(result.unmatched_fresh) == {f"garbled{i}" for i in range(10)}, result.unmatched_fresh
print(f"OK: perfect pitch/timing on the matched subset does NOT mean PASS when coverage is low "
      f"(coverage_fresh={result.coverage_fresh:.0%}, {len(result.unmatched_fresh)} unmatched word(s)) -- "
      f"the real bug this gate catches (a real ~10% failure rate was previously invisible)")

# Case 6: a text-matched pair with a WILDLY wrong timing delta (real shape:
# a repeated line/chorus pairing against the wrong sung instance -- see
# CLAUDE.md's "repeated-phrase disambiguation" lessons) must count directly
# against timing_within_tolerance_pct/recall/precision, not vanish from the
# denominator. Real confirmed bug (2026-08-14, user's own catch): an EARLIER
# version of this module bucketed candidate deltas and silently EXCLUDED
# any pair more than 3.0s from the dominant cluster before scoring pitch/
# timing accuracy at all -- this exact case (14/15 words correct, 1 matched
# pair 50s off) used to report 100% pitch/timing accuracy on the "guarded"
# subset, hiding the real failure. Same root cause independently confirmed
# in scratchpad/compare_video_games.py's own design notes.
mismatch_words = base_words + [("echo", 500.0, 0)]
existing_mismatch = ParsedSong(title="T", artist="A", bpm=200.0, gap_ms=0,
                                entries=_mk_word_syllables(mismatch_words))
fresh_mismatch = _mk_word_syllables(base_words, start_offset=0.02) + _mk_word_syllables([("echo", 550.0, 0)])
result = verify_existing_song(existing_mismatch, fresh_mismatch, min_matched=10, verbose=True)
assert result.n_matched == 16, result.n_matched
assert result.timing_within_tolerance_pct == 15 / 16, result.timing_within_tolerance_pct
assert any(text == "echo" for text, _, _ in result.timing_mismatches), result.timing_mismatches
assert result.recall == 15 / 16 and result.precision == 15 / 16, (result.recall, result.precision)
print(f"OK: a single 50s-off matched pair correctly drags down timing_within_tolerance_pct "
      f"({result.timing_within_tolerance_pct:.0%}, not the old bucket-excluded 100%), "
      f"recall={result.recall:.0%}, precision={result.precision:.0%} -- nothing silently excluded")

print("\n--- usdx_parser.parse_usdx_file: TRAILING-space word convention (real SingStar-shipped ground "
      "truth files, e.g. Beauty and the Beast's notes.txt) parses word boundaries correctly -- BUG "
      "REGRESSION for the leading-space-only heuristic silently merging a whole line into one word ---")
with _tempfile.TemporaryDirectory() as d:
    trailing_path = Path(d) / "Test Artist - Trailing Space.txt"
    # Mirrors real notes.txt exactly: "Bare"+"ly " -> "Barely", "e"+"ven " -> "even",
    # no leading spaces anywhere, word boundary is a TRAILING space on the
    # word's own last syllable.
    trailing_path.write_text(
        "#TITLE:Trailing Space\n#ARTIST:Test Artist\n#BPM:240\n#GAP:0\n"
        ": 546 6 61 Bare\n"
        ": 553 4 63 ly \n"
        ": 560 3 65 e\n"
        ": 564 2 66 ven \n"
        "E\n",
        encoding="utf-8",
    )
    parsed_trailing = parse_usdx_file(trailing_path)
    trailing_syllables = [e for e in parsed_trailing.entries if isinstance(e, Syllable)]
    assert [s.text for s in trailing_syllables] == ["Bare", "ly", "e", "ven"], trailing_syllables
    assert [s.is_word_start for s in trailing_syllables] == [True, False, True, False], \
        [s.is_word_start for s in trailing_syllables]
print("OK: trailing-space convention correctly parses 'Bare'+'ly'->'Barely' and 'e'+'ven'->'even' as two "
      "separate words (previously the leading-space-only check would have left is_word_start=False for "
      "everything but the very first syllable, merging the whole line into one bogus word)")

print("\n--- realign: alignment-only mode -- re-times an EXISTING file's own notes against real ASR, "
      "never touching pitch or the note sequence itself ---")
from ultrastar_generator.realign import (
    ExistingWord, extract_words, match_words_to_asr, interpolate_fallback, seed_lrc_anchors, realign_song,
)

print("  extract_words groups syllables into words using is_word_start, ignoring LineBreaks:")
rg_entries = [
    Syllable(text="Bare", start=0.0, end=0.3, midi_note=1, is_word_start=True),
    Syllable(text="ly", start=0.3, end=0.6, midi_note=3, is_word_start=False),
    LineBreak(start=0.6, end=1.0),
    Syllable(text="hi", start=1.0, end=1.2, midi_note=5, is_word_start=True),
]
rg_words = extract_words(rg_entries)
assert [w.text for w in rg_words] == ["Barely", "hi"], rg_words
assert rg_words[0].entry_indices == [0, 1] and rg_words[1].entry_indices == [3], rg_words
assert rg_words[0].orig_start == 0.0 and rg_words[0].orig_end == 0.6, rg_words[0]
print("  OK: multi-syllable word reconstructed from consecutive same-word syllables, LineBreak skipped, "
      "orig_start/orig_end span the whole word")

print("  match_words_to_asr: exact match + fuzzy match for ASR's own mishearing, real ASR timing used directly:")
mwa_existing = extract_words([
    Syllable(text="hello", start=0.0, end=0.1, midi_note=0, is_word_start=True),
    Syllable(text="favors", start=1.0, end=1.1, midi_note=0, is_word_start=True),
    Syllable(text="world", start=2.0, end=2.1, midi_note=0, is_word_start=True),
])
mwa_asr = [
    _Word(text="hello", start=10.0, end=10.3),
    _Word(text="favorites", start=11.0, end=11.6),  # ASR mishearing, same real case as mxl_lrc_generator
    _Word(text="world", start=12.0, end=12.3),
]
mwa_starts, mwa_ends, mwa_confident = match_words_to_asr(mwa_existing, mwa_asr)
assert mwa_confident == [True, True, True], mwa_confident
assert mwa_starts == [10.0, 11.0, 12.0], mwa_starts
assert mwa_ends == [10.3, 11.6, 12.3], mwa_ends
print("  OK: exact matches ('hello'/'world') AND a fuzzy ASR-mishearing match ('favors'~'favorites') all "
      "confidently anchor to ASR's own real start/end")

mwa_unrelated_asr = [_Word(text="hello", start=10.0, end=10.3), _Word(text="xyz", start=11.0, end=11.3),
                      _Word(text="world", start=12.0, end=12.3)]
_, _, mwa_unrelated_confident = match_words_to_asr(mwa_existing, mwa_unrelated_asr)
assert mwa_unrelated_confident == [True, False, True], mwa_unrelated_confident
print("  OK: a genuinely unrelated ASR word ('xyz' for 'favors') is correctly rejected, not fuzzy-matched")

print("  interpolate_fallback: two-sided rate interpolation, one-sided constant shift, degenerate-original "
      "-offset fallback, and identity fallback when no anchor exists anywhere:")
if_words = [ExistingWord(entry_indices=[], text=f"w{i}", norm=f"w{i}", orig_start=float(i), orig_end=float(i) + 0.5)
            for i in range(5)]
if_starts = [0.0, None, None, None, 8.0]
if_ends = [0.5, None, None, None, 8.5]
if_confident = [True, False, False, False, True]
n_interp, n_kept = interpolate_fallback(if_words, if_starts, if_ends, if_confident)
assert n_interp == 3 and n_kept == 0, (n_interp, n_kept)
assert if_starts == [0.0, 2.0, 4.0, 6.0, 8.0], if_starts  # rate = (8-0)/(4-0) = 2.0/orig-second
assert [round(e - s, 6) for s, e in zip(if_starts, if_ends)] == [0.5, 1.0, 1.0, 1.0, 0.5], list(zip(if_starts, if_ends))
print("    OK: two confident anchors -> proportional rate interpolation using ORIGINAL offsets purely as "
      "relative position (duration scaled by the same local rate)")

deg_words = [ExistingWord(entry_indices=[], text=f"w{i}", norm=f"w{i}", orig_start=0.0, orig_end=0.5)
             for _ in range(5)]
deg_starts = [5.0, None, None, None, 5.0]
deg_ends = [5.5, None, None, None, 5.5]
deg_confident = [True, False, False, False, True]
interpolate_fallback(deg_words, deg_starts, deg_ends, deg_confident)
assert deg_starts[1] == deg_starts[2] == deg_starts[3] == 5.0, deg_starts
print("    OK: degenerate original offsets (e.g. a flat list of equal-length placeholder notes, no real "
      "rhythm information to interpolate from) fall back to a constant shift instead of a divide-by-zero "
      "or nonsense rate")

one_sided_words = [ExistingWord(entry_indices=[], text=f"w{i}", norm=f"w{i}", orig_start=float(i), orig_end=float(i) + 0.5)
                    for i in range(5)]
one_sided_starts = [1.0, None, None, None, None]
one_sided_ends = [1.5, None, None, None, None]
one_sided_confident = [True, False, False, False, False]
interpolate_fallback(one_sided_words, one_sided_starts, one_sided_ends, one_sided_confident)
assert one_sided_starts == [1.0, 2.0, 3.0, 4.0, 5.0], one_sided_starts  # constant +1.0 shift, not extrapolated rate
print("    OK: only ONE anchor available -> constant shift from that anchor, not a rate extrapolated from "
      "a single data point")

none_words = [ExistingWord(entry_indices=[], text=f"w{i}", norm=f"w{i}", orig_start=float(i) * 10, orig_end=float(i) * 10 + 0.5)
              for i in range(3)]
none_starts = [None, None, None]
none_ends = [None, None, None]
none_confident = [False, False, False]
n_interp2, n_kept2 = interpolate_fallback(none_words, none_starts, none_ends, none_confident)
assert n_kept2 == 3 and n_interp2 == 0, (n_interp2, n_kept2)
assert none_starts == [0.0, 10.0, 20.0] and none_ends == [0.5, 10.5, 20.5], (none_starts, none_ends)
print("    OK: NO anchor anywhere in the whole song -> original timing kept completely unchanged, never "
      "guessed from nothing (this mode always has a safe fallback, unlike mxl_lrc_generator's equivalent)")

print("  BUG REGRESSION (real case: David Bowie - I'm Afraid of Americans, a song with many near-"
      "identical repeated short lines): a fallback word's own interpolated estimate must NEVER drag "
      "an already-CONFIDENT neighbor's own real match forward -- confirmed real case: a mis-aimed LRC "
      "line window (windowed matching picked the wrong occurrence of a repeated line) put a CONFIDENT "
      "anchor chronologically EARLIER than the confident anchor before it (a real inversion between two "
      "'confident' matches), which sends interpolate_fallback's own rate formula negative -- its "
      "degenerate fallback (shift from the earlier anchor alone, ignoring the later one entirely) then "
      "overshot PAST several subsequent genuinely-confident matches, and the old forward-only clamp "
      "flattened every one of them to that one wrong value:")
cb_words = [ExistingWord(entry_indices=[], text=f"w{i}", norm=f"w{i}", orig_start=float(i), orig_end=float(i) + 0.5)
            for i in range(5)]
# w0 confident at real t=10.0. w2/w3/w4 confident at real t=5.0/5.3/5.6 --
# EARLIER than w0, despite coming AFTER it in the word sequence (the real
# inversion a mis-aimed line window produces). w1 (fallback, between w0 and
# w2) computes a negative rate from this and overshoots to 11.0 -- worse
# than either anchor.
cb_starts = [10.0, None, 5.0, 5.3, 5.6]
cb_ends = [10.5, None, 5.2, 5.5, 5.8]
cb_confident = [True, False, True, True, True]
n_interp, n_kept = interpolate_fallback(cb_words, cb_starts, cb_ends, cb_confident)
assert cb_starts[2:5] == [5.0, 5.3, 5.6], cb_starts  # untouched -- these are CONFIDENT, never overwritten
assert cb_starts[1] == 5.0, cb_starts  # the overshot fallback estimate (11.0) is pulled back to the next confident value
print("  OK: confident words w2/w3/w4 keep their own real values exactly, completely unaffected by w1's "
      "overshot estimate (11.0 -> pulled back to 5.0); a real word0-vs-w1 inversion (10.0 then 5.0) can "
      "still remain -- one of the two confident ANCHORS is itself wrong here, which this pass can't "
      "resolve -- but it no longer drags w2/w3/w4 down with it")

print("  BUG REGRESSION (real case: David Bowie - I'm Afraid Of Americans, 'Johnny wants a brain, Johnny "
      "wants to suck on a coke' -- a repeated phrase elsewhere in the song stole the whole-song match, "
      "leaving this real run unmatched and landing compressed/wrong via blind interpolation): "
      "rematch_local_gaps retries an unmatched run using ONLY the ASR words bounded between its nearest "
      "confident neighbors, so a same-text decoy far outside that window can't be picked instead:")
from ultrastar_generator.realign import rematch_local_gaps
rlg_words = extract_words([
    Syllable(text="hello", start=0.0, end=0.5, midi_note=0, is_word_start=True),
    Syllable(text="echo", start=1.0, end=1.5, midi_note=0, is_word_start=True),
    Syllable(text="echo", start=2.0, end=2.5, midi_note=0, is_word_start=True),
    Syllable(text="world", start=3.0, end=3.5, midi_note=0, is_word_start=True),
])
rlg_asr = [
    _Word(text="echo", start=100.0, end=100.3),   # decoy: same text, way outside the [10.5, 20.0] window
    _Word(text="hello", start=10.0, end=10.5),
    _Word(text="echo", start=12.0, end=12.3),      # real first 'echo', inside the window
    _Word(text="echo", start=15.0, end=15.3),      # real second 'echo', inside the window
    _Word(text="world", start=20.0, end=20.5),
]
rlg_starts = [10.0, None, None, 20.0]
rlg_ends = [10.5, None, None, 20.5]
rlg_confident = [True, False, False, True]
rlg_n = rematch_local_gaps(rlg_words, rlg_asr, rlg_starts, rlg_ends, rlg_confident)
assert rlg_n == 2, rlg_n
assert rlg_confident == [True, True, True, True], rlg_confident
assert rlg_starts == [10.0, 12.0, 15.0, 20.0], rlg_starts
print("  OK: both 'echo' occurrences recovered from their real, IN-WINDOW ASR timestamps (12.0/15.0) -- "
      "the far-away same-text decoy at t=100 was never a candidate")

rlg2_words = extract_words([
    Syllable(text="mystery", start=1.0, end=1.5, midi_note=0, is_word_start=True),
    Syllable(text="hello", start=2.0, end=2.5, midi_note=0, is_word_start=True),
])
rlg2_asr = [_Word(text="hello", start=10.0, end=10.5)]  # 'mystery' genuinely never transcribed at all
rlg2_starts = [None, 10.0]
rlg2_ends = [None, 10.5]
rlg2_confident = [False, True]
rlg2_n = rematch_local_gaps(rlg2_words, rlg2_asr, rlg2_starts, rlg2_ends, rlg2_confident)
assert rlg2_n == 0 and rlg2_confident == [False, True], (rlg2_n, rlg2_confident)
print("  OK: a word genuinely absent from the ASR window is left unmatched (for interpolate_fallback), "
      "not force-matched to something implausible")

print("  _force_align_unconfident_runs (PROTOTYPE, 2026-08-10, adapted from UltraStarKaraokeMaker): forces "
      "the EXISTING file's own text for a still-unconfident run onto the audio window between its nearest "
      "confident neighbors -- unlike rematch_local_gaps, doesn't need ASR to have transcribed anything there "
      "at all, so it can recover a run rematch_local_gaps genuinely can't (real case: 'mystery' above):")
from ultrastar_generator.realign import _force_align_unconfident_runs
import ultrastar_generator.transcription as transcription_mod_rg
import ultrastar_generator.model_cache as model_cache_mod_rg

fa_words = extract_words([
    Syllable(text="Do-", start=0.0, end=0.5, midi_note=0, is_word_start=True),
    Syllable(text="mystery", start=1.0, end=1.5, midi_note=0, is_word_start=True),
    Syllable(text="hello", start=2.0, end=2.5, midi_note=0, is_word_start=True),
])
fa_starts = [None, None, 10.0]
fa_ends = [None, None, 10.5]
fa_confident = [False, False, True]
_orig_fawiw = transcription_mod_rg.force_align_words_in_window
_orig_align_model_rg = model_cache_mod_rg.get_whisperx_align_model
transcription_mod_rg.force_align_words_in_window = lambda words_text, w0, w1, *a, **kw: [
    (w0 + i * 0.3, w0 + i * 0.3 + 0.25, 0.55) for i in range(len(words_text))
]
model_cache_mod_rg.get_whisperx_align_model = lambda *a, **kw: (None, None)
_sys_fa.modules["whisperx"] = _types_fa.SimpleNamespace(load_audio=lambda path: [0] * 160000)  # 10s @ 16kHz
fa_n = _force_align_unconfident_runs(fa_words, fa_starts, fa_ends, fa_confident, Path("dummy.wav"))
assert fa_n == 2, fa_n
assert fa_confident == [True, True, True], fa_confident
assert fa_starts[0] == 0.0 and fa_starts[1] == 0.3, fa_starts  # window = [0.0 (song start), 10.0)
print("  OK: a run with NO neighboring confident word before it (song start) still gets a real window "
      "([0.0, next confident word's start)) and both words are recovered")

# force_align_words_in_window returning None (window unusable) -> run stays unconfident for interpolation.
transcription_mod_rg.force_align_words_in_window = lambda *a, **kw: None
fa_confident2 = [False, False, True]
fa_n2 = _force_align_unconfident_runs(fa_words, list(fa_starts), list(fa_ends), fa_confident2, Path("dummy.wav"))
assert fa_n2 == 0 and fa_confident2 == [False, False, True], (fa_n2, fa_confident2)
print("  OK: an unrecoverable run (no usable alignment result) is left unconfident, not force-marked anyway")

# a fully-confident word list never even tries to load audio (early return, no whisperx call at all).
def _fa_rg_boom(*a, **kw):
    raise AssertionError("force_align_words_in_window must not be called when nothing is unconfident")
transcription_mod_rg.force_align_words_in_window = _fa_rg_boom
fa_none_n = _force_align_unconfident_runs(fa_words, [0.0, 1.0, 10.0], [0.5, 1.5, 10.5], [True, True, True],
                                           Path("dummy.wav"))
assert fa_none_n == 0
print("  OK: no unconfident words at all -> returns immediately, never touches the audio/align model")

transcription_mod_rg.force_align_words_in_window = _orig_fawiw
model_cache_mod_rg.get_whisperx_align_model = _orig_align_model_rg
del _sys_fa.modules["whisperx"]

print("  seed_lrc_anchors: LRCLIB line starts fill in an anchor for the first not-yet-confident word of "
      "each matched line (a forced/pinned candidate skips network + candidate-selection filtering):")
from ultrastar_generator.lyrics_lookup import LrcLibCandidate
sla_existing = extract_words([
    Syllable(text="alpha", start=0.0, end=1.0, midi_note=0, is_word_start=True),
    Syllable(text="bravo", start=1.0, end=2.0, midi_note=0, is_word_start=True),
    Syllable(text="charlie", start=2.0, end=3.0, midi_note=0, is_word_start=True),
    Syllable(text="delta", start=3.0, end=4.0, midi_note=0, is_word_start=True),
])
sla_asr = [_Word(text="alpha", start=10.05, end=10.3)]  # bravo/charlie/delta never transcribed at all
sla_starts, sla_ends, sla_confident = match_words_to_asr(sla_existing, sla_asr)
assert sla_confident == [True, False, False, False], sla_confident
sla_candidate = LrcLibCandidate(
    track_name="T", artist_name="A", album_name="", duration=None,
    plain_lyrics="alpha bravo charlie delta",
    synced_lyrics="[00:10.00]alpha bravo\n[00:20.00]charlie delta\n",
    instrumental=False, id=999,
)
sla_result = seed_lrc_anchors(sla_existing, sla_asr, sla_starts, sla_ends, sla_confident,
                               "A", "T", 100.0, forced_candidate=sla_candidate)
assert sla_result is not None and sla_result.n_seeded == 1, sla_result
assert sla_confident == [True, False, True, False], sla_confident  # "charlie" seeded (first word of line 2)
assert sla_starts[2] == 20.0, sla_starts  # this song's single ASR anchor isn't enough evidence to calibrate
                                            # (needs >= LRC_TIMING_MIN_CALIBRATION_SAMPLES), so the line's
                                            # own raw timestamp is used uncalibrated
print("  OK: 'charlie' (first word of the second LRC line) gets a real-time anchor from the line's own "
      "timestamp; 'bravo'/'delta' (not line-first) are untouched, left for interpolation")

print("  seed_lrc_anchors returns None (no mutation) when no usable candidate exists -- confirmed real "
      "case: Beauty and the Beast has NO valid LRCLIB candidate at all:")
sla_none_starts, sla_none_ends, sla_none_confident = match_words_to_asr(sla_existing, sla_asr)
_sys.modules["requests"] = _FakeRequestsModule(search_payload=[])  # deterministic empty search, no real network
sla_none_result = seed_lrc_anchors(sla_existing, sla_asr, sla_none_starts, sla_none_ends, sla_none_confident,
                                    "A", "T", 100.0, forced_candidate=None)
assert sla_none_result is None, sla_none_result
assert sla_none_confident == [True, False, False, False], sla_none_confident  # unchanged
print("  OK: no usable LRC candidate -> starts/ends/confident left completely untouched, falls through to "
      "ASR-only interpolation")

print("  match_words_to_asr_windowed (PROTOTYPE 'windowed' lrc_mode): LRC line starts window the ASR "
      "search per-line -- a same-text ASR word far outside the line's own window is correctly ignored, "
      "unlike whole-song matching which has no time information to reject it with:")
from ultrastar_generator.realign import match_words_to_asr_windowed, prepare_lrc

mww_existing = extract_words([
    Syllable(text="alpha", start=0.0, end=1.0, midi_note=0, is_word_start=True),
    Syllable(text="bravo", start=1.0, end=2.0, midi_note=0, is_word_start=True),
    Syllable(text="charlie", start=2.0, end=3.0, midi_note=0, is_word_start=True),
    Syllable(text="delta", start=3.0, end=4.0, midi_note=0, is_word_start=True),
])
mww_word_lines = [0, 0, 1, 1]
mww_lrc_lines = [(10.0, "alpha bravo"), (20.0, "charlie delta")]
mww_asr = [
    _Word(text="alpha", start=100.0, end=100.3),  # decoy: right TEXT, way outside line 0's window
    _Word(text="alpha", start=10.05, end=10.3),   # the real one, inside line 0's window
    _Word(text="bravo", start=10.6, end=10.9),
    _Word(text="charlie", start=20.05, end=20.3),
]
mww_starts, mww_ends, mww_confident = match_words_to_asr_windowed(mww_existing, mww_word_lines, mww_lrc_lines, mww_asr)
assert mww_confident == [True, True, True, False], mww_confident
assert mww_starts[0] == 10.05, mww_starts  # the in-window occurrence, NOT the far-away decoy at t=100
assert mww_starts[1] == 10.6, mww_starts
assert mww_starts[2] == 20.05, mww_starts
print("  OK: 'alpha' correctly matched to its IN-WINDOW ASR occurrence (10.05s), ignoring the decoy "
      "at 100s with identical text; 'delta' (no ASR word anywhere) stays unmatched for interpolate_fallback")

print("  prepare_lrc: shared candidate-selection/calibration/line-assignment step used by BOTH LRC "
      "strategies (via a forced candidate, no real network):")
mww_prep = prepare_lrc(mww_existing, mww_asr[1:], "A", "T", 100.0, forced_candidate=sla_candidate)
assert mww_prep is not None and mww_prep.word_lines == [0, 0, 1, 1], mww_prep
print("  OK: prepare_lrc returns the calibrated lines + per-word line assignment both strategies build on")

print("  check_repeat_structure: rejects an LRC candidate whose REPEAT STRUCTURE doesn't match ours -- "
      "BUG REGRESSION (real case: David Bowie - I'm Afraid of Americans, an LRC candidate from a "
      "different edition/box-set mix with 9 extra chorus repeats): the real signal can't be a single "
      "exact-repeated LINE, since a real chorus is often split across several near-duplicate variants "
      "(e.g. 'I'm afraid of Americans'/'...of the world'/'...I can't help it'), each individually "
      "landing within tolerance on its own -- the shared distinctive WORD across all of them is needed:")
from ultrastar_generator.realign import _reconstruct_our_lines, check_repeat_structure

crs_entries = []
for _ in range(3):
    crs_entries.append(Syllable(text="chorus", start=0.0, end=0.1, midi_note=0, is_word_start=True))
    crs_entries.append(Syllable(text="one", start=0.1, end=0.2, midi_note=0, is_word_start=True))
    crs_entries.append(LineBreak(start=0.2, end=0.3))
for _ in range(3):
    crs_entries.append(Syllable(text="chorus", start=0.0, end=0.1, midi_note=0, is_word_start=True))
    crs_entries.append(Syllable(text="two", start=0.1, end=0.2, midi_note=0, is_word_start=True))
    crs_entries.append(LineBreak(start=0.2, end=0.3))
crs_our_lines = _reconstruct_our_lines(crs_entries)
assert crs_our_lines == ["chorus one"] * 3 + ["chorus two"] * 3, crs_our_lines

crs_lrc_matching = ["chorus one"] * 3 + ["chorus two"] * 3
assert check_repeat_structure(crs_our_lines, crs_lrc_matching) is None
print("  OK: matching repeat structure ('chorus' appears 6x total on both sides, split the same way "
      "across both variants) -> not rejected")

crs_lrc_mismatched = ["chorus one"] * 4 + ["chorus two"] * 4  # 8x vs our 6x -- structurally different
crs_rejection = check_repeat_structure(crs_our_lines, crs_lrc_mismatched)
assert crs_rejection is not None and "chorus" in crs_rejection, crs_rejection
print("  OK: mismatched repeat structure ('chorus' 6x vs 8x, split across TWO near-duplicate line "
      "variants each individually within tolerance on its own) -> correctly rejected via the shared "
      "distinctive word, not just the single most-repeated exact line")

assert check_repeat_structure(["alpha", "bravo", "charlie"], ["anything", "goes", "here"]) is None
print("  OK: no repeated line in our own file at all -> nothing to check, never rejects")

print("  prepare_lrc: a repeat-structure-mismatched candidate is rejected outright (via our_lines/log), "
      "even though it would otherwise be perfectly usable (forced candidate, so selection/duration/"
      "content filters don't apply):")
crs_prep_words = extract_words(crs_entries)
crs_lrc_text = "\n".join(f"[00:{i:02d}.00]chorus one" for i in range(4)) + "\n" + \
               "\n".join(f"[00:{i + 10:02d}.00]chorus two" for i in range(4))
crs_forced = LrcLibCandidate(
    track_name="T", artist_name="A", album_name="", duration=None,
    plain_lyrics="chorus one chorus two", synced_lyrics=crs_lrc_text,
    instrumental=False, id=555,
)
crs_prep_log = []
crs_prep_result = prepare_lrc(crs_prep_words, [], "A", "T", 100.0, forced_candidate=crs_forced,
                               our_lines=crs_our_lines, log=crs_prep_log.append)
assert crs_prep_result is None, crs_prep_result
assert any("rejected" in line for line in crs_prep_log), crs_prep_log
print("  OK: prepare_lrc itself rejects the candidate (returns None, same as 'no candidate found') and "
      "logs the specific reason when our_lines/log are provided")

print("  realign_song end-to-end with lrc_mode='windowed': same forced candidate, output stays consistent "
      "with the windowed matcher's own per-line window (not just whole-song ASR matching):")
rsw_existing = ParsedSong(
    title="Windowed", artist="A", bpm=60.0, gap_ms=0,
    entries=[
        Syllable(text="alpha", start=0.0, end=1.0, midi_note=1, is_word_start=True),
        Syllable(text="bravo", start=1.0, end=2.0, midi_note=2, is_word_start=True),
        Syllable(text="charlie", start=2.0, end=3.0, midi_note=3, is_word_start=True),
        Syllable(text="delta", start=3.0, end=4.0, midi_note=4, is_word_start=True),
    ],
    raw_tags={"TITLE": "Windowed", "ARTIST": "A", "BPM": "60", "GAP": "0"},
)
rsw_log = []
rsw_result = realign_song(rsw_existing, mww_asr, artist="A", title="T", audio_duration=100.0,
                           use_lrc=True, lrc_mode="windowed", forced_lrc_candidate=sla_candidate, log=rsw_log.append)
assert rsw_result.success, rsw_result.error
rsw_syllables = [e for e in rsw_result.song.entries if isinstance(e, Syllable)]
assert abs(rsw_syllables[0].start - 10.05) < 1e-6, rsw_syllables[0]  # picked the in-window "alpha", not the decoy
assert [s.midi_note for s in rsw_syllables] == [1, 2, 3, 4], rsw_syllables  # pitch still never touched
print("  OK: realign_song(lrc_mode='windowed') end-to-end also correctly rejects the far-away decoy via "
      "the LRC line window, pitch untouched")

print("  BUG REGRESSION (real case: David Bowie - Heroes): lrc_mode='seed' must ALSO refuse to seed "
      "anchors from an LRC candidate whose time calibration failed -- a wrong-recording candidate's "
      "uncalibrated line timestamp seeded as an anchor can land LATER than several already-correctly-"
      "ASR-matched neighbors, and interpolate_fallback's forward-only monotonic clamp then drags those "
      "correct neighbors forward to match the bad anchor, corrupting real matches, not just filling a "
      "genuine gap. Confirmed real case: an LRCLIB candidate for 'Heroes' by a completely different "
      "artist (a choral cover, calibration_confidence=0.0) seeded 'Then' at a time LATER than 'Just for "
      "one day', which was already correctly ASR-matched -- corrupting that whole passage:")
bg_existing = extract_words([
    Syllable(text="alpha", start=0.0, end=1.0, midi_note=0, is_word_start=True),
    Syllable(text="bravo", start=1.0, end=2.0, midi_note=0, is_word_start=True),
    Syllable(text="charlie", start=2.0, end=3.0, midi_note=0, is_word_start=True),
    Syllable(text="delta", start=3.0, end=4.0, midi_note=0, is_word_start=True),
])
# Only 1 real ASR match to the LRC's own line text -- not enough samples for
# two_tier_time_calibration to trust ANY offset (same as the real Bowie case:
# calibration_confidence=0.0, not just "borderline").
bg_asr = [_Word(text="alpha", start=10.05, end=10.3)]
bg_starts, bg_ends, bg_confident = match_words_to_asr(bg_existing, bg_asr)
assert bg_confident == [True, False, False, False], bg_confident

bg_existing_song = ParsedSong(
    title="T", artist="A", bpm=60.0, gap_ms=0,
    entries=[
        Syllable(text="alpha", start=0.0, end=1.0, midi_note=1, is_word_start=True),
        Syllable(text="bravo", start=1.0, end=2.0, midi_note=2, is_word_start=True),
        Syllable(text="charlie", start=2.0, end=3.0, midi_note=3, is_word_start=True),
        Syllable(text="delta", start=3.0, end=4.0, midi_note=4, is_word_start=True),
    ],
    raw_tags={"TITLE": "T", "ARTIST": "A", "BPM": "60", "GAP": "0"},
)
bg_log = []
bg_result = realign_song(bg_existing_song, bg_asr, artist="A", title="T", audio_duration=100.0,
                          use_lrc=True, lrc_mode="seed", forced_lrc_candidate=sla_candidate, log=bg_log.append)
assert bg_result.success, bg_result.error
assert bg_result.quality.n_lrc_seeded == 0, bg_result.quality
assert any("NOT trusted as a seed anchor" in line for line in bg_log), bg_log
print("  OK: lrc_mode='seed' also declines to seed from an uncalibrated candidate (0 seeded, not 1) -- "
      "'charlie' (first word of line 2) is left for interpolate_fallback instead of a raw, untrusted "
      "line timestamp")

print("  realign_song: 'windowed' mode's own low anchor rate (under a CONFIDENTLY-calibrated candidate) "
      "auto-falls-back to 'seed' mode instead of just warning -- a low rate here means the per-line window "
      "itself is mis-targeted (e.g. words with no LRC line of their own get bucketed into the wrong line's "
      "narrow window), not that whole-song ASR can't find these words at all:")
wsf_entries = [
    Syllable(text=t, start=float(i), end=float(i) + 0.5, midi_note=0, is_word_start=True)
    for i, t in enumerate(["alpha", "bravo", "charlie", "delta", "echo",
                            "golf", "hotel", "india", "juliet", "kilo", "lima", "mike"])
]
wsf_existing_song = ParsedSong(
    title="T", artist="A", bpm=60.0, gap_ms=0, entries=wsf_entries,
    raw_tags={"TITLE": "T", "ARTIST": "A", "BPM": "60", "GAP": "0"},
)
# LRC candidate only knows about 5 one-word lines (calibration needs
# >= LRC_TIMING_MIN_CALIBRATION_SAMPLES=5 matched LINES, not words) --
# 'golf'..'mike' have no LRC line of their own at all, so
# assign_words_to_lines buckets them into the LAST confirmed line
# ('echo')'s own narrow window (its own docstring: "inherit the nearest
# PRECEDING confirmed match's line").
wsf_candidate = LrcLibCandidate(
    track_name="T", artist_name="A", album_name="", duration=None,
    plain_lyrics="alpha bravo charlie delta echo",
    synced_lyrics="[00:10.00]alpha\n[00:11.00]bravo\n[00:12.00]charlie\n[00:13.00]delta\n[00:14.00]echo\n",
    instrumental=False, id=777,
)
wsf_asr = [
    # Calibration + windowed-match anchors for the 5 real LRC lines (5
    # samples, all in perfect agreement -> confident constant offset ~0).
    _Word(text="alpha", start=10.05, end=10.3), _Word(text="bravo", start=11.05, end=11.3),
    _Word(text="charlie", start=12.05, end=12.3), _Word(text="delta", start=13.05, end=13.3),
    _Word(text="echo", start=14.05, end=14.3),
    # 'golf'..'mike' really were transcribed -- just far outside the only
    # window they can be bucketed into ([13.5, 19.5], from the last LRC
    # line + its fallback "+5.0s" span since there's no next line).
    _Word(text="golf", start=200.0, end=200.3), _Word(text="hotel", start=200.35, end=200.6),
    _Word(text="india", start=200.65, end=200.9), _Word(text="juliet", start=200.95, end=201.2),
    _Word(text="kilo", start=201.25, end=201.5), _Word(text="lima", start=201.55, end=201.8),
    _Word(text="mike", start=201.85, end=202.1),
]
wsf_log = []
wsf_result = realign_song(wsf_existing_song, wsf_asr, artist="A", title="T", audio_duration=300.0,
                           use_lrc=True, lrc_mode="windowed", forced_lrc_candidate=wsf_candidate,
                           log=wsf_log.append)
assert wsf_result.success, wsf_result.error
assert any("only 42%" in line and "real anchor" in line for line in wsf_log), wsf_log
assert any("Falling back to 'seed' mode" in line for line in wsf_log), wsf_log
assert wsf_result.quality.n_asr_matched == 12, wsf_result.quality  # the 'seed' retry's own result, not windowed's 5/12
wsf_starts = {e.text: e.start for e in wsf_result.song.entries if isinstance(e, Syllable)}
assert wsf_starts["golf"] == 200.0 and wsf_starts["mike"] == 201.85, wsf_starts  # real ASR times, not squeezed
                                                                                    # into the 13.5-19.5 window
print("  OK: 'windowed' mode's own 5/12 (42%) anchor rate triggers both the low-anchor warning AND an "
      "automatic retry with lrc_mode='seed', which finds all 12/12 words via whole-song ASR matching "
      "(no time window to exclude 'golf'..'mike's real, correctly-transcribed positions)")

print("  realign_song end-to-end: a file whose notes are a degenerate flat list of equal-length "
      "placeholder notes (don't match the audio at all) gets re-timed to match real ASR, with the SAME "
      "note count/order/pitch and BPM, only start/end (and derived GAP) changed:")
rs_existing = ParsedSong(
    title="Degenerate", artist="Test Artist", bpm=60.0, gap_ms=0,
    entries=[
        Syllable(text="one", start=0.0, end=1.0, midi_note=10, is_word_start=True, note_type=":"),
        Syllable(text="two", start=1.0, end=2.0, midi_note=20, is_word_start=True, note_type=":"),
        LineBreak(start=2.0, end=2.0),
        Syllable(text="three", start=2.0, end=3.0, midi_note=30, is_word_start=True, note_type="*"),
        Syllable(text="four", start=3.0, end=4.0, midi_note=40, is_word_start=True, note_type=":"),
    ],
    raw_tags={"TITLE": "Degenerate", "ARTIST": "Test Artist", "MP3": "music.ogg", "BPM": "60", "GAP": "0"},
)
rs_asr = [
    _Word(text="one", start=10.0, end=10.4),
    _Word(text="two", start=10.5, end=10.9),
    _Word(text="three", start=11.0, end=11.4),
    _Word(text="four", start=11.5, end=11.9),
]
rs_log = []
rs_result = realign_song(rs_existing, rs_asr, use_lrc=False, log=rs_log.append)
assert rs_result.success, rs_result.error
rs_syllables = [e for e in rs_result.song.entries if isinstance(e, Syllable)]
assert [s.text for s in rs_syllables] == ["one", "two", "three", "four"], rs_syllables
assert [s.midi_note for s in rs_syllables] == [10, 20, 30, 40], rs_syllables       # pitch NEVER touched
assert [s.note_type for s in rs_syllables] == [":", ":", "*", ":"], rs_syllables  # note type NEVER touched
assert len(rs_result.song.entries) == len(rs_existing.entries), \
    (len(rs_result.song.entries), len(rs_existing.entries))                        # no note added/removed
assert rs_result.song.bpm == 60.0, rs_result.song.bpm                              # BPM never touched
for s, expected_asr in zip(rs_syllables, rs_asr):
    assert abs(s.start - expected_asr.start) < 1e-6, (s, expected_asr)
    assert abs(s.end - expected_asr.end) < 1e-6, (s, expected_asr)
assert rs_result.song.gap_ms == 10000, rs_result.song.gap_ms  # GAP re-derived from the new first syllable
rs_break = next(e for e in rs_result.song.entries if isinstance(e, LineBreak))
assert abs(rs_break.start - 10.9) < 1e-6, rs_break.start  # re-anchored to "two"'s new end
assert abs(rs_break.end - 11.0) < 1e-6, rs_break.end      # re-anchored to "three"'s new start
assert rs_result.song.mp3 == "music.ogg"  # untouched metadata carried through verbatim
assert rs_result.quality.n_asr_matched == 4 and rs_result.quality.n_kept_original == 0, rs_result.quality
rs_orig_syllables_after = [e for e in rs_existing.entries if isinstance(e, Syllable)]
assert [s.start for s in rs_orig_syllables_after] == [0.0, 1.0, 2.0, 3.0], rs_orig_syllables_after
print("  OK: the CALLER's own ParsedSong (rs_existing) is never mutated -- its syllables still show their "
      "original 0-4s placeholder timing after the call, so realign_song can safely be called twice on the "
      "same parsed object (e.g. to compare two lrc_mode strategies against each other)")
print("  OK: every note landed exactly on its real ASR timestamp (0-4s placeholder timing completely "
      "replaced by real ~10-12s audio timing), pitch/note-type/note-count/BPM untouched, GAP and the "
      "LineBreak's position both correctly re-derived from the new syllable timing")

print("  realign_song: a low anchor rate (lyrics/audio likely mismatched) still returns a usable result "
      "(never crashes/aborts -- the original file is always a safe fallback) but logs a clear warning:")
rs_bad_existing = ParsedSong(
    title="Mismatch", artist="Test Artist", bpm=120.0, gap_ms=0,
    entries=[Syllable(text=f"w{i}", start=float(i), end=float(i) + 0.5, midi_note=0, is_word_start=True)
             for i in range(10)],
    raw_tags={"TITLE": "Mismatch", "ARTIST": "Test Artist", "BPM": "120", "GAP": "0"},
)
rs_bad_asr = [_Word(text="completely", start=50.0, end=50.3), _Word(text="unrelated", start=51.0, end=51.3)]
rs_bad_log = []
rs_bad_result = realign_song(rs_bad_existing, rs_bad_asr, use_lrc=False, log=rs_bad_log.append)
assert rs_bad_result.success, rs_bad_result.error
assert rs_bad_result.quality.n_asr_matched == 0, rs_bad_result.quality
assert any("WARNING" in line and "may not match" in line for line in rs_bad_log), rs_bad_log
print("  OK: 0% real anchor rate still produces a valid (unchanged-timing) output rather than crashing, "
      "with a clear warning logged for the user to review")

print("  _retry_asr_if_low_quality (PROTOTYPE): a low anchor rate triggers a re-transcription with "
      "config.RETRY_ASR_MODEL, keeping whichever attempt has the higher anchor rate -- never fires when "
      "already using the retry model, or when the anchor rate is already above the bar:")
from ultrastar_generator.realign import _retry_asr_if_low_quality, RealignPipelineOptions
import ultrastar_generator.transcription as transcription_mod

_orig_transcribe_words = transcription_mod.transcribe_words
rq_opts = RealignPipelineOptions(whisper_model="small.en", use_lrc=False, retry_low_quality_asr=True, batch=True)

# (a) low anchor rate + a retry model that DOES find the real words, in --batch mode -> retry is accepted.
rq_good_asr = [_Word(text=f"w{i}", start=float(i) + 50.0, end=float(i) + 50.4) for i in range(10)]
transcription_mod.transcribe_words = lambda *a, **kw: rq_good_asr
rq_log = []
rq_retried = _retry_asr_if_low_quality(
    rs_bad_result, existing=rs_bad_existing, vocals_path=Path("dummy.wav"), opts=rq_opts,
    audio_duration=100.0, forced_candidate=None, debug_log=None, log=rq_log.append,
)
assert rq_retried is not rs_bad_result, "a genuine improvement must return the RETRY result, not the original"
assert rq_retried.quality.n_asr_matched == 10, rq_retried.quality
assert any(config_mod.RETRY_ASR_MODEL in line and "retrying" in line for line in rq_log), rq_log
assert any("improved" in line for line in rq_log), rq_log
print(f"  OK: original 0% anchor rate, --batch mode -> retry with '{config_mod.RETRY_ASR_MODEL}' finds all "
      f"10 words, retry result adopted")

# (a2) SAME low anchor rate, but NOT --batch mode (2026-08-10, user's explicit request) -> logs a WARNING
# suggesting --batch instead, never actually calls transcribe_words, returns the original unchanged.
def _rq_boom_not_batch(*a, **kw):
    raise AssertionError("transcribe_words must not be called outside --batch mode")
transcription_mod.transcribe_words = _rq_boom_not_batch
rq_opts_not_batch = RealignPipelineOptions(whisper_model="small.en", use_lrc=False, retry_low_quality_asr=True,
                                            batch=False)
rq_not_batch_log = []
rq_not_batch_result = _retry_asr_if_low_quality(
    rs_bad_result, existing=rs_bad_existing, vocals_path=Path("dummy.wav"), opts=rq_opts_not_batch,
    audio_duration=100.0, forced_candidate=None, debug_log=None, log=rq_not_batch_log.append,
)
assert rq_not_batch_result is rs_bad_result, "must be a no-op (same object) outside --batch mode"
assert any("WARNING" in line and "--batch" in line for line in rq_not_batch_log), rq_not_batch_log
print("  OK: same low anchor rate, but NOT --batch mode -> logs a WARNING suggesting --batch, never "
      "actually retries, original result returned unchanged")

# (b) already using the retry model -- must never fire (and never call transcribe_words at all).
def _rq_boom(*a, **kw):
    raise AssertionError("transcribe_words must not be called when already at the retry model")
transcription_mod.transcribe_words = _rq_boom
rq_opts_already = RealignPipelineOptions(whisper_model=config_mod.RETRY_ASR_MODEL, use_lrc=False,
                                          retry_low_quality_asr=True)
rq_noop1 = _retry_asr_if_low_quality(
    rs_bad_result, existing=rs_bad_existing, vocals_path=Path("dummy.wav"), opts=rq_opts_already,
    audio_duration=100.0, forced_candidate=None, debug_log=None, log=lambda s: None,
)
assert rq_noop1 is rs_bad_result, "must be a no-op (same object) when whisper_model is already the retry model"
print(f"  OK: whisper_model already '{config_mod.RETRY_ASR_MODEL}' -> no retry attempted")

# (c) anchor rate already clears the bar -- must never fire either.
rq_noop2 = _retry_asr_if_low_quality(
    rs_result, existing=rs_existing, vocals_path=Path("dummy.wav"), opts=rq_opts,
    audio_duration=100.0, forced_candidate=None, debug_log=None, log=lambda s: None,
)
assert rq_noop2 is rs_result, "must be a no-op when the anchor rate is already above the retry bar"
print("  OK: anchor rate already above the retry bar -> no retry attempted")

print("  _retry_asr_if_low_quality per-PASSAGE trigger (PROTOTYPE, 2026-08-10): a long consecutive run of "
      "unconfident words must ALSO trigger a retry even when the whole-song anchor rate is already fine -- "
      "real case: David Bowie - Magic Dance with small.en had a 58% anchor rate (well above the bar) while "
      "one hallucinated decoder segment still left a real passage ~12-14s off in the final output:")
from ultrastar_generator.realign import RealignQuality, RealignResult

# (d) anchor rate clears the whole-song bar, but longest_unconfident_run alone clears ITS OWN bar -> fires.
rq_passage_quality = RealignQuality(n_words=20, n_asr_matched=15, longest_unconfident_run=6)
assert rq_passage_quality.anchor_rate >= config_mod.MXL_LRC_MIN_ASR_PLACEMENT_RATE, rq_passage_quality.anchor_rate
assert rq_passage_quality.longest_unconfident_run >= config_mod.RETRY_ASR_MIN_UNCONFIDENT_RUN
rq_passage_result = RealignResult(success=True, song=None, quality=rq_passage_quality)
transcription_mod.transcribe_words = lambda *a, **kw: rq_good_asr   # same 10-word perfect-match fixture as (a)
rq_passage_log = []
rq_passage_retried = _retry_asr_if_low_quality(
    rq_passage_result, existing=rs_bad_existing, vocals_path=Path("dummy.wav"), opts=rq_opts,
    audio_duration=100.0, forced_candidate=None, debug_log=None, log=rq_passage_log.append,
)
assert rq_passage_retried is not rq_passage_result, "a long unconfident run alone must trigger a retry"
assert any("consecutive word" in line and "retrying" in line for line in rq_passage_log), rq_passage_log
print("  OK: 75% anchor rate (fine) + 6 consecutive unconfident words (over the per-passage bar) -> retry "
      "fires anyway")

# (e) neither signal clears its own bar -> no-op (anchor rate fine, run just under the per-passage bar).
rq_passage_ok_quality = RealignQuality(n_words=20, n_asr_matched=15,
                                        longest_unconfident_run=config_mod.RETRY_ASR_MIN_UNCONFIDENT_RUN - 1)
rq_passage_ok_result = RealignResult(success=True, song=None, quality=rq_passage_ok_quality)
def _rq_passage_boom(*a, **kw):
    raise AssertionError("transcribe_words must not be called when neither trigger clears its bar")
transcription_mod.transcribe_words = _rq_passage_boom
rq_passage_noop = _retry_asr_if_low_quality(
    rq_passage_ok_result, existing=rs_bad_existing, vocals_path=Path("dummy.wav"), opts=rq_opts,
    audio_duration=100.0, forced_candidate=None, debug_log=None, log=lambda s: None,
)
assert rq_passage_noop is rq_passage_ok_result, "must be a no-op when neither trigger clears its own bar"
print(f"  OK: anchor rate fine + unconfident run just under the bar "
      f"({config_mod.RETRY_ASR_MIN_UNCONFIDENT_RUN - 1} < {config_mod.RETRY_ASR_MIN_UNCONFIDENT_RUN}) -> "
      f"no retry attempted")

transcription_mod.transcribe_words = _orig_transcribe_words

print("  realign: the existing file being realigned is ALWAYS treated as read-only -- never overwritten, "
      "not even if an explicit --output path resolves to the same file:")
from ultrastar_generator.realign import resolve_realign_output_path, check_output_not_existing_file

ro_existing_path = Path("C:/Songs/Some Artist - Some Song.txt")
ro_default_out = resolve_realign_output_path(ro_existing_path, None)
assert ro_default_out.name == "Some Artist - Some Song [REALIGNED].txt", ro_default_out
assert check_output_not_existing_file(ro_default_out, ro_existing_path) is None, \
    "the default output path must never collide with the existing file"
print("  OK: the default output path is a separate '[REALIGNED]' file, never the existing file itself")

ro_same_out = resolve_realign_output_path(ro_existing_path, str(ro_existing_path))
ro_error = check_output_not_existing_file(ro_same_out, ro_existing_path)
assert ro_error is not None and "read-only" in ro_error, ro_error
print("  OK: an explicit --output that resolves to the SAME path as the existing file is REFUSED "
      "(no override exists for this on purpose)")

ro_different_out = resolve_realign_output_path(ro_existing_path, "C:/Songs/somewhere/else.txt")
assert check_output_not_existing_file(ro_different_out, ro_existing_path) is None
print("  OK: an explicit --output pointing somewhere genuinely different is allowed")

print("\n--- realign: 'validate' strategy (PROTOTYPE) -- a word CONFIRMED by ASR near its own "
      "(GAP-corrected) original position is left completely untouched instead of being replaced "
      "with ASR's own value ---")
from ultrastar_generator.realign import (
    compute_gap_calibration, validate_words_against_asr, realign_song_validate, GapCalibration,
)

print("  compute_gap_calibration: 'sometimes the song file is nearly perfect but the GAP is wrong so "
      "everything is offset' -- a UNIFORM +5.0s shift across every word is detected as a single "
      "constant offset, not treated as N independent per-word corrections:")
gc_words = extract_words([
    Syllable(text="alpha", start=10.0, end=10.5, midi_note=0, is_word_start=True),
    Syllable(text="bravo", start=11.0, end=11.5, midi_note=0, is_word_start=True),
    Syllable(text="charlie", start=12.0, end=12.5, midi_note=0, is_word_start=True),
    Syllable(text="delta", start=13.0, end=13.5, midi_note=0, is_word_start=True),
    Syllable(text="echo", start=14.0, end=14.5, midi_note=0, is_word_start=True),
])
gc_asr = [_Word(text="alpha", start=15.0, end=15.5), _Word(text="bravo", start=16.0, end=16.5),
          _Word(text="charlie", start=17.0, end=17.5), _Word(text="delta", start=18.0, end=18.5),
          _Word(text="echo", start=19.0, end=19.5)]
gc = compute_gap_calibration(gc_words, gc_asr)
assert gc.offset is not None and abs(gc.offset - 5.0) < 1e-6 and gc.kind == "constant", gc
assert gc.confidence == 1.0, gc.confidence
print("  OK: a uniform +5.0s shift across all 5 words is recovered as one confident constant GAP "
      "offset, using the SAME robust two-tier calibration already validated for LRC line timing")

print("  validate_words_against_asr: a word whose (GAP-corrected) original position is confirmed by "
      "ASR is kept EXACTLY as original -- position AND length -- not replaced by ASR's own (slightly "
      "different) timestamp; a word ASR disagrees with is left unvalidated:")
gc_words2 = extract_words([
    Syllable(text="alpha", start=10.0, end=10.5, midi_note=0, is_word_start=True),
    Syllable(text="bravo", start=11.0, end=11.5, midi_note=0, is_word_start=True),
])
# A pre-built GapCalibration (bypassing compute_gap_calibration's own
# >=5-sample minimum, tested separately above) with a known +5.0s offset --
# alpha's ASR match (15.0) is within tolerance of its expected (10.0+5.0=
# 15.0); bravo's ASR match is deliberately way off (30.0, not 16.0) and
# must NOT validate.
gc2 = GapCalibration(offset=5.0, slope=0.0, confidence=1.0, kind="constant", skipped_reason=None,
                      asr_starts=[15.0, 30.0], asr_ends=[15.6, 30.6], asr_confident=[True, True])
vw_starts, vw_ends, vw_validated = validate_words_against_asr(gc_words2, gc2)
assert vw_validated == [True, False], vw_validated
assert vw_starts[0] == 15.0 and vw_ends[0] == 15.5, (vw_starts[0], vw_ends[0])  # ORIGINAL 0.5s length kept,
                                                                                  # NOT ASR's own 0.6s span
print("  OK: 'alpha' validated with its OWN original 0.5s length (10.0-10.5, shifted to 15.0-15.5), "
      "NOT overwritten by ASR's own slightly-different 15.0-15.6 span; 'bravo' (ASR disagrees) is "
      "correctly left unvalidated")

print("  compute_gap_calibration + validate_words_against_asr: a DISCONTINUOUS whole-file drift (tier 3) "
      "is calibrated and applied via correction_fn, not the offset+slope fallback (which would badly "
      "misfit a real step change):")
gc3_words = extract_words(
    [Syllable(text=f"w{i}", start=float(i * 5), end=float(i * 5) + 0.3, midi_note=0, is_word_start=True)
     for i in range(20)]
)
gc3_asr = []
for i in range(20):
    orig = float(i * 5)
    delta = 2.0 if i < 7 else (10.0 if i < 14 else 20.0)  # same 3-segment shape as the lrc_timing test above
    gc3_asr.append(_Word(text=f"w{i}", start=orig + delta, end=orig + delta + 0.3))
gc3 = compute_gap_calibration(gc3_words, gc3_asr)
assert gc3.kind == "isotonic" and gc3.correction_fn is not None, gc3
vw3_starts, _vw3_ends, vw3_validated = validate_words_against_asr(gc3_words, gc3, tolerance_sec=1.5)
# a word in each segment should validate near its own segment's real offset (via correction_fn), not a
# single global offset/slope compromise that would misfit segments B/C badly
assert vw3_validated[0] and abs(vw3_starts[0] - 2.0) < 1.5, (vw3_validated[0], vw3_starts[0])
assert vw3_validated[10] and abs(vw3_starts[10] - 60.0) < 2.0, (vw3_validated[10], vw3_starts[10])
assert vw3_validated[18] and abs(vw3_starts[18] - 110.0) < 1.5, (vw3_validated[18], vw3_starts[18])
print(f"  OK: compute_gap_calibration recovered kind={gc3.kind!r} for a 3-segment discontinuous whole-file "
      f"drift, and validate_words_against_asr's per-word expected_start correctly tracked each segment via "
      f"correction_fn (not a single misfit offset+slope)")

print("  GapCalibration manually constructed WITHOUT a correction_fn (e.g. older/test code) still works "
      "via the offset+slope fallback in validate_words_against_asr -- backward compatible:")
gc4 = GapCalibration(offset=5.0, slope=0.0, confidence=1.0, kind="constant", skipped_reason=None,
                      asr_starts=[15.0], asr_ends=[15.5], asr_confident=[True])
assert gc4.correction_fn is None
vw4_starts, _vw4_ends, vw4_validated = validate_words_against_asr(
    extract_words([Syllable(text="alpha", start=10.0, end=10.5, midi_note=0, is_word_start=True)]), gc4)
assert vw4_validated == [True] and vw4_starts[0] == 15.0, (vw4_validated, vw4_starts)
print("  OK: correction_fn=None (default) falls back to the offset+slope formula, unchanged from before "
      "this session's tier-3 addition")

print("  realign_song_validate end-to-end: an already-correct file whose ONLY problem is a wrong GAP -- "
      "every word's own relative timing/length is preserved EXACTLY, just uniformly shifted, and pitch/"
      "note-count/BPM are untouched (same invariants as realign_song). Needs >= 5 agreeing words for "
      "two_tier_time_calibration's own minimum-sample gate to trust a single offset at all:")
rv_existing = ParsedSong(
    title="GapOnly", artist="Test Artist", bpm=120.0, gap_ms=0,
    entries=[
        Syllable(text="one", start=10.0, end=10.4, midi_note=5, is_word_start=True, note_type=":"),
        Syllable(text="two", start=11.0, end=11.9, midi_note=7, is_word_start=True, note_type="*"),
        LineBreak(start=11.9, end=12.0),
        Syllable(text="three", start=12.0, end=12.3, midi_note=9, is_word_start=True, note_type=":"),
        Syllable(text="four", start=13.0, end=13.2, midi_note=2, is_word_start=True, note_type=":"),
        Syllable(text="five", start=14.0, end=14.6, midi_note=4, is_word_start=True, note_type=":"),
    ],
    raw_tags={"TITLE": "GapOnly", "ARTIST": "Test Artist", "BPM": "120", "GAP": "0"},
)
rv_asr = [_Word(text="one", start=18.0, end=18.4), _Word(text="two", start=19.0, end=19.7),
          _Word(text="three", start=20.0, end=20.2), _Word(text="four", start=21.0, end=21.3),
          _Word(text="five", start=22.0, end=22.9)]
rv_log = []
rv_result = realign_song_validate(rv_existing, rv_asr, use_lrc=False, log=rv_log.append)
assert rv_result.success, rv_result.error
rv_syllables = [e for e in rv_result.song.entries if isinstance(e, Syllable)]
# GAP offset should be exactly +8.0s (18-10, 19-11, 20-12, 21-13, 22-14 all agree).
assert [round(s.start, 6) for s in rv_syllables] == [18.0, 19.0, 20.0, 21.0, 22.0], rv_syllables
# Original LENGTHS preserved exactly -- NOT replaced by ASR's own (slightly different) spans.
assert [round(s.end - s.start, 6) for s in rv_syllables] == [0.4, 0.9, 0.3, 0.2, 0.6], rv_syllables
assert [s.midi_note for s in rv_syllables] == [5, 7, 9, 2, 4], rv_syllables
assert [s.note_type for s in rv_syllables] == [":", "*", ":", ":", ":"], rv_syllables
assert len(rv_result.song.entries) == len(rv_existing.entries)
assert rv_result.song.bpm == 120.0
assert rv_result.quality.n_validated == 5, rv_result.quality
print("  OK: all 5 words validated and shifted by exactly the same +8.0s GAP correction, each keeping "
      "its OWN original length rather than ASR's own slightly different span -- pitch/note-type/"
      "note-count/BPM untouched")

print("  realign_song_validate: a mostly-correct file with ONE genuinely wrong word -- the wrong word "
      "is repositioned via interpolation while its correctly-matching neighbors (>= 5 of them, so the "
      "GAP calibration itself is well-established) stay validated (untouched) around it:")
rvm_existing = ParsedSong(
    title="Mostly", artist="A", bpm=120.0, gap_ms=0,
    entries=[
        Syllable(text="one", start=0.0, end=0.4, midi_note=1, is_word_start=True),
        Syllable(text="two", start=1.0, end=1.4, midi_note=2, is_word_start=True),
        Syllable(text="three", start=2.0, end=2.4, midi_note=3, is_word_start=True),   # this one is WRONG
        Syllable(text="four", start=3.0, end=3.4, midi_note=4, is_word_start=True),
        Syllable(text="five", start=4.0, end=4.4, midi_note=5, is_word_start=True),
        Syllable(text="six", start=5.0, end=5.4, midi_note=6, is_word_start=True),
    ],
    raw_tags={"TITLE": "Mostly", "ARTIST": "A", "BPM": "120", "GAP": "0"},
)
# one/two/four/five/six all agree on a clean +10.0s offset; "three"'s real
# ASR match is a wild outlier (50.0) that must NOT be trusted as validating it.
rvm_asr = [_Word(text="one", start=10.0, end=10.4), _Word(text="two", start=11.0, end=11.4),
           _Word(text="three", start=50.0, end=50.4), _Word(text="four", start=13.0, end=13.4),
           _Word(text="five", start=14.0, end=14.4), _Word(text="six", start=15.0, end=15.4)]
rvm_result = realign_song_validate(rvm_existing, rvm_asr, use_lrc=False, log=lambda s: None)
assert rvm_result.success, rvm_result.error
rvm_syllables = [e for e in rvm_result.song.entries if isinstance(e, Syllable)]
assert rvm_syllables[1].start == 11.0 and rvm_syllables[3].start == 13.0, rvm_syllables  # validated, untouched
assert 11.0 < rvm_syllables[2].start < 13.0, rvm_syllables[2]  # repositioned BETWEEN its validated neighbors,
                                                                  # not left at the wild ASR outlier (50.0)
assert rvm_result.quality.n_validated == 5 and rvm_result.quality.n_interpolated == 1, rvm_result.quality
print("  OK: 'two'/'four' (and the other agreeing words) validated and untouched; 'three' (wild ASR "
      "outlier) correctly rejected and interpolated between its validated neighbors instead")

print("  realign: find_existing_txt_in_folder auto-detects the single .txt to realign in a folder "
      "(needed for --batch, where a single explicit --existing-txt can't apply across multiple "
      "subfolders) -- fails closed (never guesses) when zero or multiple real candidates exist:")
from ultrastar_generator.realign import find_existing_txt_in_folder, AmbiguousExistingTxtError

with _tempfile.TemporaryDirectory() as d:
    single_dir = Path(d)
    (single_dir / "Some Artist - Some Song.txt").write_text("x", encoding="utf-8")
    found = find_existing_txt_in_folder(single_dir)
    assert found.name == "Some Artist - Some Song.txt", found
print("  OK: exactly one .txt in the folder -> found directly")

with _tempfile.TemporaryDirectory() as d:
    reran_dir = Path(d)
    (reran_dir / "Some Artist - Some Song.txt").write_text("x", encoding="utf-8")
    (reran_dir / "Some Artist - Some Song [REALIGNED].txt").write_text("y", encoding="utf-8")
    found2 = find_existing_txt_in_folder(reran_dir)
    assert found2.name == "Some Artist - Some Song.txt", found2
print("  OK: a folder that already has a PREVIOUS run's own '[REALIGNED]' output still correctly "
      "picks the ORIGINAL file -- not falsely 'ambiguous', and never the REALIGNED file itself "
      "(which would compound drift across repeated batch runs)")

with _tempfile.TemporaryDirectory() as d:
    empty_dir = Path(d)
    try:
        find_existing_txt_in_folder(empty_dir)
        assert False, "should have raised AmbiguousExistingTxtError"
    except AmbiguousExistingTxtError:
        pass
print("  OK: no .txt file at all -> AmbiguousExistingTxtError, never silently skipped")

with _tempfile.TemporaryDirectory() as d:
    multi_dir = Path(d)
    (multi_dir / "one.txt").write_text("x", encoding="utf-8")
    (multi_dir / "two.txt").write_text("y", encoding="utf-8")
    try:
        find_existing_txt_in_folder(multi_dir)
        assert False, "should have raised AmbiguousExistingTxtError"
    except AmbiguousExistingTxtError:
        pass
print("  OK: two genuinely different .txt files -> AmbiguousExistingTxtError, never guesses which one")

with _tempfile.TemporaryDirectory() as d:
    named_dir = Path(d) / "Some Artist - Some Song"
    named_dir.mkdir()
    (named_dir / "Some Artist - Some Song.txt").write_text("x", encoding="utf-8")
    (named_dir / "notes_backup.txt").write_text("y", encoding="utf-8")
    found3 = find_existing_txt_in_folder(named_dir)
    assert found3.name == "Some Artist - Some Song.txt", found3
print("  OK: multiple .txt files, but exactly one matches '<folder name>.txt' -> that one is used, "
      "not treated as ambiguous")

with _tempfile.TemporaryDirectory() as d:
    unrelated_dir = Path(d) / "Some Artist - Some Song"
    unrelated_dir.mkdir()
    (unrelated_dir / "one.txt").write_text("x", encoding="utf-8")
    (unrelated_dir / "two.txt").write_text("y", encoding="utf-8")
    try:
        find_existing_txt_in_folder(unrelated_dir)
        assert False, "should have raised AmbiguousExistingTxtError"
    except AmbiguousExistingTxtError:
        pass
print("  OK: multiple .txt files, NONE matching the folder's own name -> still AmbiguousExistingTxtError "
      "(the folder-name match is a narrow disambiguation, not a fallback 'guess something' rule)")

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

print("\n--- CLI smoke test: both entry points' build_arg_parser() must not crash "
      "(catches a missing config constant an argparse default references, e.g. FORCE_ALIGN_GAPS "
      "went missing from config.py once this session without any other test catching it) ---")
from ultrastar_generator.main import build_arg_parser as _main_build_arg_parser
_main_build_arg_parser().parse_args(["dummy_input_dir"])
print("OK: main.py's build_arg_parser() builds and parses without error")
from ultrastar_generator.realign import build_arg_parser as _realign_build_arg_parser
_realign_build_arg_parser().parse_args(["dummy_input_dir"])
print("OK: realign.py's build_arg_parser() builds and parses without error")
from ultrastar_generator.pitch_refresh import build_arg_parser as _pitch_refresh_build_arg_parser
_pitch_refresh_build_arg_parser().parse_args(["dummy_input_dir"])
print("OK: pitch_refresh.py's build_arg_parser() builds and parses without error")

print("\n--- transcription._split_segment_text (PROTOTYPE, 2026-08-10, config.REWINDOW_SPLIT_ENABLED): "
      "splits a long decoder segment's own text into sub-phrases for split-rewindowing ---")
from ultrastar_generator.transcription import _split_segment_text

punct_split = _split_segment_text(" Johnny's in America. Johnny wants a brain! Johnny wants to know?")
assert punct_split == ["Johnny's in America.", "Johnny wants a brain!", "Johnny wants to know?"], punct_split
print("OK: splits cleanly on sentence-ending punctuation:", punct_split)

single_sentence = _split_segment_text(" Just one short sentence.")
assert single_sentence == ["Just one short sentence."], single_sentence
print("OK: a single sentence (no internal punctuation split, under the word-count fallback "
      "threshold) is returned as one whole piece")

# No punctuation at all (a repeated-chorus run, the real motivating case) -- falls back
# to fixed word-count chunks (config.REWINDOW_SPLIT_FALLBACK_WORDS, default 8).
no_punct = "I'm afraid of Americans I'm afraid of the world I'm afraid I can't help it I'm afraid I can't"
no_punct_split = _split_segment_text(no_punct)
assert len(no_punct_split) > 1, no_punct_split
reconstructed = " ".join(no_punct_split)
assert reconstructed == no_punct, (reconstructed, no_punct)
assert all(len(p.split()) <= config_mod.REWINDOW_SPLIT_FALLBACK_WORDS for p in no_punct_split), no_punct_split
print("OK: punctuation-less repeated text falls back to fixed word-count chunks, "
      "reconstructing the original text exactly:", no_punct_split)

print("\n--- pitch_refresh: pitch-only refresh of an existing usdx timing base (same basic idea as "
      "the external ultrastar_pitch/usp tool, see CLAUDE.md/project memory) ---")
import ultrastar_generator.pitch_refresh as pitch_refresh_mod
from ultrastar_generator.pitch_refresh import (
    refresh_song_pitch, compute_pitch_class_predictions, resolve_pitch_refresh_output_path,
    find_existing_txt_in_folder as pr_find_existing_txt_in_folder, _KeyNudge, OUTPUT_MARKER,
)


def _fake_pitch_source_for_refresh_test(y, sr, hop_length, frame_length, fmin, fmax, n_frames, **kwargs):
    """A deterministic stand-in for a real PITCH_SOURCES entry: note0
    (0-1s) is voiced at pitch class 3, note1 (1-2s) is left COMPLETELY
    UNVOICED (to exercise the nearest-neighbor fallback), note2/note3
    (2-4s) are voiced at pitch class 9."""
    frame_dur = hop_length / sr
    times = np.arange(n_frames) * frame_dur
    midi = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    for i, t in enumerate(times):
        if 0.0 <= t < 1.0:
            midi[i], conf[i], voiced[i] = 60 + 3, 0.9, True
        elif 2.0 < t < 4.0:
            # strictly > 2.0 (not >=) so the inclusive right boundary of note1's own
            # [1.0, 2.0] search window (see _pred_for_note's side="right") can't pick up
            # a stray voiced frame exactly AT t=2.0 and accidentally give note1 a real
            # (wrong) prediction instead of exercising the unvoiced-fallback path.
            midi[i], conf[i], voiced[i] = 60 + 9, 0.9, True
        # 1.0 <= t <= 2.0 (note1 and the boundary frame right after it): left unvoiced on purpose
    return midi, conf, voiced


pitch_refresh_mod.PITCH_SOURCES["_fake_test_source"] = _fake_pitch_source_for_refresh_test
try:
    pr_entries = [
        Syllable(text="Al", start=0.0, end=1.0, midi_note=60, is_word_start=True),   # pc0 -> expect pc3
        Syllable(text="pha", start=1.0, end=2.0, midi_note=61, is_word_start=False),  # pc1, unvoiced -> fallback
        LineBreak(start=2.0),
        Syllable(text="Bra", start=2.0, end=3.0, midi_note=62, is_word_start=True),   # pc2 -> expect pc9
        Syllable(text="vo", start=3.0, end=4.0, midi_note=74, is_word_start=False),   # pc2 (74%12) -> expect pc9
    ]
    pr_song = ParsedSong(title="Test Song", artist="Test Artist", bpm=200.0, gap_ms=1234,
                          entries=pr_entries, raw_tags={"MP3": "audio.mp3"})
    # Real sr + the module's own default hop_length/frame_length (256/2048), matching what
    # refresh_song_pitch itself uses -- big enough (4.5s) to clear the frame_length floor.
    sr_fake = 16000
    y_fake = np.zeros(int(4.5 * sr_fake))

    preds = compute_pitch_class_predictions(
        pr_entries, y_fake, sr_fake, pitch_source="_fake_test_source",
        attack_trim_sec=0.0, confidence_floor_percentile=0.0, voicing_threshold=None,
    )
    assert preds == [3, None, 9, 9], preds
    print("OK: compute_pitch_class_predictions returns the fake source's own per-note pitch class, "
          "None for the note with zero voiced frames")

    result_song = refresh_song_pitch(
        pr_song, y_fake, sr_fake, pitch_source="_fake_test_source",
        attack_trim_sec=0.0, confidence_floor_percentile=0.0, voicing_threshold=None, key_nudge=False,
    )
    result_notes = [e for e in result_song.entries if isinstance(e, Syllable)]
    assert len(result_notes) == 4, result_notes
    assert [n.text for n in result_notes] == ["Al", "pha", "Bra", "vo"]
    assert [(n.start, n.end) for n in result_notes] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    assert [n.is_word_start for n in result_notes] == [True, False, True, False]
    assert sum(1 for e in result_song.entries if isinstance(e, LineBreak)) == 1, "LineBreak must survive untouched"
    print("OK: refresh_song_pitch never touches timing, text, note count, note order, or line breaks")

    new_pcs = [n.midi_note % 12 for n in result_notes]
    assert new_pcs == [3, 3, 9, 9], new_pcs
    print("OK: every note's pitch CLASS was replaced with the (fallback-filled) prediction:", new_pcs)

    new_midis = [n.midi_note for n in result_notes]
    # octave preserved: only the pitch-CLASS component changes, same convention as usp itself
    assert new_midis == [63, 63, 69, 81], new_midis
    print("OK: each note's OCTAVE is preserved -- only the pitch-class digit changed "
          f"(orig midi [60,61,62,74] -> {new_midis})")
    print("OK: note1 (no voiced frames at all) borrowed pitch class 3 from its nearest scored "
          "neighbor (note0, tied-distance with note2 -- ties resolve backward) rather than being "
          "left unset or crashing")

    assert result_song.gap_ms == 1234 and result_song.bpm == 200.0 and result_song.mp3 == "audio.mp3"
    assert result_song.title == "Test Song" and result_song.artist == "Test Artist"
    print("OK: GAP/BPM/other header tags carried through from the existing file completely "
          "untouched (pitch_refresh reuses realign._song_from_existing for this, not a re-implementation)")
finally:
    del pitch_refresh_mod.PITCH_SOURCES["_fake_test_source"]

print("\n--- pitch_refresh: the existing file is ALWAYS treated as read-only, same guarantee as "
      "realign.py (reuses realign.check_output_not_existing_file directly, not re-implemented) ---")
default_out = resolve_pitch_refresh_output_path(Path("song/Artist - Title.txt"), None)
assert default_out.name == f"Artist - Title {OUTPUT_MARKER}.txt", default_out
print(f"OK: the default output path is a separate '{OUTPUT_MARKER}' file, never the existing file itself")

import tempfile as _tempfile_pr
with _tempfile_pr.TemporaryDirectory() as _tmp_pr:
    _tmp_pr = Path(_tmp_pr)
    same_path = _tmp_pr / "Artist - Title.txt"
    same_path.write_text("dummy")
    guard = pitch_refresh_mod.check_output_not_existing_file(same_path, same_path)
    assert guard is not None, "must refuse to overwrite the existing file"
    print("OK: an --output that resolves to the SAME path as the existing file is refused")

print("\n--- pitch_refresh: find_existing_txt_in_folder auto-detection excludes BOTH this module's "
      "own '[PITCH REFRESHED]' output AND realign.py's '[REALIGNED]' output, not just its own ---")
with _tempfile_pr.TemporaryDirectory() as _tmp_pr2:
    _tmp_pr2 = Path(_tmp_pr2)
    original = _tmp_pr2 / "My Song.txt"
    original.write_text("dummy")
    (_tmp_pr2 / f"My Song {OUTPUT_MARKER}.txt").write_text("dummy")
    (_tmp_pr2 / "My Song [REALIGNED].txt").write_text("dummy")
    found = pr_find_existing_txt_in_folder(_tmp_pr2, exclude_markers=pitch_refresh_mod._EXCLUDE_MARKERS)
    assert found == original, found
print("OK: a folder with a previous pitch-refresh output AND a previous realign output still "
      "correctly picks the ORIGINAL file, never either module's own prior output as the next input")

print("\n--- pitch_refresh._KeyNudge: vendored port of usp's own StochasticPostprocessor (OFF by "
      "default in this module -- see key_nudge=False default and project memory on why) ---")
key0 = _KeyNudge.detect_key([0] * 10)
assert key0 == 0, key0
print(f"OK: a pitch-class distribution concentrated entirely at class 0 detects pseudo-key {key0} "
      "(argmax of that key table column, deterministic)")
nudged = _KeyNudge.correct(0, [1])
assert nudged == [2], nudged
unnudged = _KeyNudge.correct(0, [0])
assert unnudged == [0], unnudged
print("OK: an out-of-key pitch class (1, zero probability under key 0) is nudged +-1 semitone toward "
      "whichever neighbor scores higher (2), while an in-key class (0) is left untouched")
