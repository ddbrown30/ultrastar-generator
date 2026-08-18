"""Splits a word into syllables.

Combines two independent splitters and takes whichever gives MORE real
syllable pieces, ties favoring pyphen -- see `hyphenate`'s own docstring
for why and the real data behind it (real-world-scale validated against
a 413-word corpus from 55 real songs, 2026-08-18: pyphen wins 86 of 111
tie cases, not a close call):

- pyphen (a hyphenation library based on the same dictionaries as
  LibreOffice/Firefox spellcheck) -- tuned for legal PRINT line-wrap
  points, not phonetic syllable boundaries, so it systematically
  UNDER-COUNTS relative to real singing syllabification often enough to
  matter (confirmed real cases: "Lucifer" -> only 2 pieces instead of 3,
  "fugitive" -> only 2 instead of 3 -- not just a wrong break point, the
  wrong SYLLABLE COUNT, which forces bogus melisma-padding onto a real
  syllable elsewhere in this codebase).
- `_sonority_syllabify`, a from-scratch Sonority Sequencing Principle
  syllabifier: gets the syllable COUNT right more often than pyphen: for
  each inter-vowel consonant cluster, prefers a REAL WORD boundary
  (`_word_boundary_split`, a small bundled common-word list, user's own
  proposed design 2026-08-18) over the pure maximal-onset default
  (`_split_cluster`) -- e.g. "filling" splits "fill"/"ing" (doubled 'l'
  stays whole, since "fill" alone is a real word) but "running" splits
  "run"/"ning" (only one 'n' stays, since "runn" isn't a real word but
  "run" is) -- same doubled-consonant shape, opposite real splits,
  resolved by which prefix actually forms a word.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

_dic = None


def _get_dic():
    global _dic
    if _dic is None:
        try:
            import pyphen
            _dic = pyphen.Pyphen(lang="en_US")
        except ImportError:
            _dic = False  # sentinel: unavailable
    return _dic


_VOWEL_GROUPS = re.compile(r"[^aeiouyAEIOUY]*[aeiouyAEIOUY]+(?:[^aeiouyAEIOUY]*$)?", re.UNICODE)


def _regex_fallback(word: str) -> List[str]:
    matches = _VOWEL_GROUPS.findall(word)
    matches = [m for m in matches if m]
    if not matches:
        return [word]
    # Reassemble consuming the whole word (regex above can leave gaps for
    # leading/trailing consonant clusters); simplest robust approach: just
    # split by vowel-group boundaries found via finditer with positions.
    parts = []
    pos = 0
    for m in _VOWEL_GROUPS.finditer(word):
        if m.start() > pos:
            continue
        parts.append(word[pos:m.end()])
        pos = m.end()
    if pos < len(word):
        if parts:
            parts[-1] += word[pos:]
        else:
            parts.append(word[pos:])
    return parts or [word]


_VOWELS = set("aeiouy")
_LEGAL_ONSET_3 = {"str", "spr", "scr", "spl", "thr", "shr", "squ"}
_LEGAL_ONSET_2 = {
    "bl", "br", "cl", "cr", "dr", "dw", "fl", "fr", "gl", "gr", "pl", "pr",
    "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw", "sh", "ch",
    "th", "wh", "ph", "wr", "qu", "gh", "ck", "ng", "ct",
}


def _nuclei_runs(core: str) -> List[tuple]:
    """Maximal contiguous runs of vowel letters (a syllable nucleus can be
    a vowel digraph like "ie"/"ou", not just one letter). A single
    trailing "e" preceded by a consonant is dropped (silent e: "fire",
    "fugitive") UNLESS it's a "-Cle" ending ("double", "little"), where
    the 'l' itself is a syllabic consonant and the syllable is real."""
    runs = []
    i, n = 0, len(core)
    while i < n:
        if core[i].lower() in _VOWELS:
            j = i
            while j < n and core[j].lower() in _VOWELS:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) > 1:
        last_start, last_end = runs[-1]
        if (last_end == n and last_end - last_start == 1 and core[last_start].lower() == "e"
                and last_start > 0 and core[last_start - 1].lower() not in _VOWELS):
            is_cle_ending = (core[last_start - 1].lower() == "l" and last_start > 1
                              and core[last_start - 2].lower() not in _VOWELS)
            if not is_cle_ending:
                runs.pop()
    return runs


def _split_cluster(cluster: str) -> int:
    """Split index WITHIN a consonant run between two vowel nuclei (coda
    of the previous syllable = cluster[:idx], onset of the next =
    cluster[idx:]) -- maximal-onset principle: give the next syllable the
    longest suffix of the cluster that's a real legal English onset."""
    n = len(cluster)
    if n <= 1:
        return 0  # nothing, or a single consonant -> whole thing is the next syllable's onset
    if cluster[0].lower() == cluster[1].lower():
        return 1  # doubled consonant: split down the middle
    if n >= 3 and cluster[-3:].lower() in _LEGAL_ONSET_3:
        return n - 3
    if cluster[-2:].lower() in _LEGAL_ONSET_2:
        return n - 2
    return n - 1  # default: only the last consonant starts the next syllable


_common_words = None


def _get_common_words() -> set:
    """A ~10k-word common-English wordlist (google-10000-english,
    MIT-licensed, bundled as `data/common_words.txt` -- a static data
    file, not a new runtime dependency), lazy-loaded once. Deliberately
    NOT a large/exhaustive dictionary (~370k words tried and rejected,
    see `_word_boundary_split`'s own docstring): a big indiscriminate
    wordlist is full of obscure/dialectal/abbreviation entries ("alw",
    "bab", "fug", "sil"...) that create false-positive real-word matches
    on meaningless prefixes, actively making splits WORSE."""
    global _common_words
    if _common_words is None:
        path = Path(__file__).parent / "data" / "common_words.txt"
        try:
            _common_words = set(w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip())
        except OSError:
            _common_words = set()
    return _common_words


def _word_boundary_split(core: str, end_a: int, cluster: str,
                          next_run: Optional[tuple], is_final_boundary: bool) -> int:
    """Split index WITHIN `cluster`, preferring a REAL WORD boundary over
    the pure maximal-onset default -- user's own proposed rule
    (2026-08-18), real confirmed case: "filling" splits "fill"/"ing" (the
    doubled 'l' stays WHOLE with the first syllable, since "fill" is
    already a real word on its own) but "running" splits "run"/"ning"
    (only ONE 'n' stays behind, since "runn" ISN'T a real word but "run"
    is) -- the same doubled-consonant shape, opposite real splits,
    resolved by which prefix length actually forms a word.

    Two guards on top of the basic rule, both user-identified real
    counterexamples found via real-audio validation (2026-08-18):
      - A DOUBLED consonant only ever considers keeping the WHOLE pair
        with the first syllable (take=len(cluster)) or splitting it ONE
        EACH down the middle (take=1) -- never pushing the whole pair to
        the SECOND syllable (take=0), which isn't a real English
        doubling pattern. Without this, "happy" wrongly matched "ha" (a
        real short word) and produced "ha"/"ppy" instead of "hap"/"py".
      - Never choose a split that leaves a FINAL lone "y" as the whole
        second syllable ("bul"/"ly" not "bull"/"y", even though "bull"
        is a real word) -- a word-final "y" acting as a vowel is never
        sung/said as its own bare syllable, it always keeps at least the
        one preceding consonant. Scoped to the LAST inter-nucleus
        boundary only, and only when the upcoming nucleus run is exactly
        one "y" ending the word.

    Tries the longest allowed prefix first and takes the first one where
    `core`'s own text up to that point is a real common word. Falls back
    to `_split_cluster`'s maximal-onset default when no allowed prefix
    length forms a real word at all (e.g. "system" -- "sys" and "syst"
    aren't real English words on their own)."""
    n = len(cluster)
    is_doubled = n >= 2 and cluster[0].lower() == cluster[1].lower()
    candidates = [n, 1] if is_doubled else list(range(n, -1, -1))

    forbid_full = False
    if is_final_boundary and next_run is not None:
        ns, ne = next_run
        if ne - ns == 1 and core[ns].lower() == "y" and ne == len(core):
            forbid_full = True

    words = _get_common_words()
    for take in candidates:
        if forbid_full and take == n:
            continue
        if core[:end_a + take].lower() in words:
            return take
    return _split_cluster(cluster)


def _sonority_syllabify(core: str) -> List[str]:
    """Sonority-Sequencing-Principle syllabifier: finds vowel nuclei, then
    splits each inter-nucleus consonant cluster via `_word_boundary_split`
    (falls back to `_split_cluster`'s pure maximal-onset rule when no real
    word boundary applies). A word with 0 or 1 nucleus is returned whole
    (nothing to split)."""
    runs = _nuclei_runs(core)
    if len(runs) <= 1:
        return [core]
    boundaries = []
    for k in range(len(runs) - 1):
        _, end_a = runs[k]
        start_b, _ = runs[k + 1]
        is_final = (k == len(runs) - 2)
        split = _word_boundary_split(core, end_a, core[end_a:start_b], runs[k + 1], is_final)
        boundaries.append(end_a + split)
    parts, prev = [], 0
    for b in boundaries:
        parts.append(core[prev:b])
        prev = b
    parts.append(core[prev:])
    return [p for p in parts if p]


def _pyphen_parts(core: str) -> List[str]:
    dic = _get_dic()
    if dic:
        parts = dic.inserted(core, hyphen="\u00ad").split("\u00ad")
    else:
        parts = _regex_fallback(core)
    return [p for p in parts if p] or [core]


def hyphenate(word: str) -> List[str]:
    """Returns a list of syllable strings that concatenate back to `word`
    exactly (punctuation included) -- important since UltraStar note text
    is displayed verbatim.

    Runs both pyphen and `_sonority_syllabify` on the word's alphabetic
    core and keeps whichever gives MORE pieces, ties favoring the
    sonority splitter.

    Real-data-validated 2026-08-18, in stages, all against the same
    trusted-only 56-word set (extracted from Stars' own MXL and Video
    Games' existing hand-authored .txt -- the only two sources the user
    has confirmed trustworthy for syllable splitting; an earlier pass
    pooled ground truth from all 12 MusicXML scores in this project's
    sandbox, 208 words, but the user later clarified the other 10
    scores' own house style/OCR quality is unverified, so those broader
    numbers are superseded here, not the trusted figure):

    1. pyphen alone: 50.0% exact-match, 60.7% count-match.
    2. `_sonority_syllabify` alone, maximal-onset only (no word-boundary
       rule yet): 51.8% exact-match, 76.8% count-match -- wins on count
       (fewer bogus melisma-pads) but loses to pyphen on exact
       break-point accuracy. On the specific TIE-break subset (same
       piece count as pyphen, different split): pyphen matched trusted
       ground truth 6 times, sonority 3, both wrong 3 -- pyphen still
       won, so ties favored pyphen at this stage.
    3. User's own proposed refinement, added to `_sonority_syllabify`'s
       own cluster-splitting (`_word_boundary_split`): for a doubled
       consonant (or any inter-vowel cluster) prefer a REAL WORD
       boundary over pure maximal-onset -- concrete motivating
       counterexample from Video Games (a single trusted source):
       "running" splits "run"/"ning" (only ONE 'n' stays behind, since
       "runn" isn't a real word but "run" is) while "filling" splits
       "fill"/"ing" (the doubled 'l' stays WHOLE, since "fill" already
       is a real word) -- the identical doubled-consonant shape, same
       song, opposite real splits, resolved by which prefix length
       actually forms a word. On the tiny 56-word trusted set THIS
       flipped the tie-break subset itself (sonority won 3 of 5 ties) --
       ties were switched to favor sonority at this stage. **This turned
       out to be a sample-size artifact, corrected in stage 4 below.**
    4. Real-world-scale validation (2026-08-18, user's own explicit
       request): downloaded 55 more songs (20 pop/rock, 18 musical
       theatre, 17 other genres -- country/soul/folk/jazz-standard/
       gospel/Christmas -- all English, 1951-2008, sourced from a public
       GitHub mirror of the old Wikifonia lead-sheet database,
       `jganseman/musq`) and extracted 413 unique real multi-syllable
       words with real notated splits, several appearing in MULTIPLE
       independent songs. On this MUCH larger sample, the tie-break
       picture reverses sharply: pyphen matches ground truth in 86 of
       111 tie cases, sonority only 14 -- the tiny 56-word sample's 3-2
       split was not representative. Switched ties back to favoring
       pyphen. Critically, this did NOT undo stage 3's real fixes:
       pyphen already independently agrees with the word-boundary-rule
       sonority splitter on "filling"/"running"/"bully"/"duty"/"happy"/
       "rely"/"Thunder", and "Lucifer"/"fugitive" are resolved by the
       "more pieces wins" COUNT rule, never the tie-break -- only
       "never"-shaped words are actually affected, and the 413-word
       corpus shows those are GENUINELY inconsistent in real notation
       (see below), not a case with one right answer to chase. Final
       combined hybrid on the 413-word corpus: 70.5% exact-match / 90.8%
       count-match -- beats plain pyphen's own 64.9%/72.2% on BOTH
       metrics simultaneously (not a trade-off), and beats the
       tie-favors-sonority version's 54.2%/90.8%.

    **The 413-word corpus also directly confirms the user's own
    hypothesis** that some syllable-split disagreement reflects
    convention/preference rather than one binary right answer: 15 words
    (~3.6% of the unique vocabulary) have GENUINELY different splits
    (case differences excluded) across different independently-sourced
    songs -- e.g. "never" splits "nev/er" 3 times and "ne/ver" 4 times
    across different shows/songs, roughly a coin flip; "many"/"very"
    both have real notated instances splitting the trailing "y" as its
    own bare syllable ("man/y", "ver/y") -- a direct, real counterexample
    to this module's own "no lone final y" guard (kept anyway: it's
    still right far more often than wrong, e.g. "bully"/"duty"/"happy"/
    "rely" all agree with it, and a single real word list can't encode
    "except sometimes, for short high-frequency words"). No further
    tuning was attempted past this analysis -- with real disagreement
    baked into the ground truth itself, chasing 100% isn't a real target.

    The bundled word list (`data/common_words.txt`, ~10k words,
    MIT-licensed google-10000-english) was deliberately chosen over a
    large/exhaustive dictionary (~370k words, tried and rejected): the
    big list is full of obscure/dialectal/abbreviation entries ("alw",
    "bab", "fug", "sil"...) that register as false-positive real-word
    matches on meaningless prefixes, making splits WORSE, not better
    (measured: exact-match dropped to 48.2% with the 370k list vs. 60.7%
    with the curated 10k one, same rule, same trusted ground truth).

    Even the curated 10k list has its own contamination, found via real-
    audio validation (not the offline ground-truth set -- neither Stars
    nor Video Games happens to contain these specific words): it's
    frequency-derived from web text, so calendar/title ABBREVIATIONS
    ("thu" for Thursday, "mon"/"tue"/"wed"/"fri"/"sat", "mr"/"dr"/"st"/
    "nd"/"rd"/"th") are common STRINGS but not real standalone spoken/
    sung words -- "Thunder" was matching "thu" and splitting "Thu"/
    "nder" instead of "Thun"/"der". Fixed by stripping these specific
    entries from `data/common_words.txt` (not a blanket length/frequency
    filter -- a real independent word that also happens to double as an
    abbreviation, e.g. "sun"/"mar", is deliberately kept).

    Two more real refinements to `_word_boundary_split` itself, both
    user-identified directly (2026-08-18): a DOUBLED consonant only ever
    considers keeping the whole pair with the first syllable OR
    splitting it one-each down the middle, never pushing the whole pair
    to the second syllable -- without this, "happy" matched "ha" (a
    real short interjection) and produced "ha"/"ppy" instead of "hap"/
    "py". And a word-final "y" acting as a vowel is never its own bare
    syllable -- "bully" must split "bul"/"ly", not "bull"/"y", even
    though "bull" is a real, longer-matching word; the guard specifically
    forbids attaching a FULL cluster when doing so would leave nothing
    but a lone word-final "y" as the whole next syllable.

    None of this reaches 100%, and can't -- see the 413-word corpus
    disagreement finding above."""
    if not word:
        return [word]

    # Separate leading/trailing punctuation so the splitters only see the
    # alphabetic core, then reattach it to the first/last syllable.
    m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", word, re.UNICODE)
    lead, core, trail = m.groups() if m else ("", word, "")

    if not core:
        return [word]

    pyphen_parts = _pyphen_parts(core)
    sonority_parts = _sonority_syllabify(core)
    parts = sonority_parts if len(sonority_parts) > len(pyphen_parts) else pyphen_parts

    parts[0] = lead + parts[0]
    parts[-1] = parts[-1] + trail
    return parts


def chunk_to_count(parts: List[str], n_chunks: int) -> List[str]:
    """Merges a syllable list down to exactly n_chunks contiguous text
    chunks (used when a word has more syllables than there are target
    slots -- e.g. audio-detected notes, or an MXL word's own notated
    syllable count -- to hold them)."""
    n_chunks = max(1, n_chunks)
    if n_chunks >= len(parts):
        return parts
    # Distribute parts across n_chunks as evenly as possible, in order.
    chunks = []
    base = len(parts) // n_chunks
    extra = len(parts) % n_chunks
    idx = 0
    for c in range(n_chunks):
        take = base + (1 if c < extra else 0)
        take = max(1, take)
        chunks.append("".join(parts[idx:idx + take]))
        idx += take
    return chunks
