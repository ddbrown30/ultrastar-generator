"""Splits a word into syllables. Runs pyphen (print line-wrap hyphenation, tends to under-count real
singing syllables) and `_sonority_syllabify` (Sonority Sequencing Principle, prefers a real-word
boundary over pure maximal-onset), keeping whichever gives more pieces, ties favoring pyphen."""

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
            _dic = False  # unavailable
    return _dic


_VOWEL_GROUPS = re.compile(r"[^aeiouyAEIOUY]*[aeiouyAEIOUY]+(?:[^aeiouyAEIOUY]*$)?", re.UNICODE)


def _regex_fallback(word: str) -> List[str]:
    matches = _VOWEL_GROUPS.findall(word)
    matches = [m for m in matches if m]
    if not matches:
        return [word]
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
    """Maximal contiguous vowel-letter runs (nuclei). Drops a trailing silent 'e' unless it's a
    "-Cle" ending ("double", "little"), where the 'l' is a real syllabic consonant."""
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
    """Split index within a consonant cluster between two vowel nuclei, via the maximal-onset
    principle (longest legal English onset goes to the next syllable)."""
    n = len(cluster)
    if n <= 1:
        return 0
    if cluster[0].lower() == cluster[1].lower():
        return 1  # doubled consonant: split down the middle
    if n >= 3 and cluster[-3:].lower() in _LEGAL_ONSET_3:
        return n - 3
    if cluster[-2:].lower() in _LEGAL_ONSET_2:
        return n - 2
    return n - 1


_common_words = None


def _get_common_words() -> set:
    """Lazy-loaded ~10k-word common-English wordlist (`data/common_words.txt`) -- deliberately small,
    not exhaustive, to avoid obscure entries producing false-positive splits."""
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
    """Split index within `cluster`, preferring a real-word boundary over the maximal-onset default
    (e.g. "filling" -> "fill"/"ing", "running" -> "run"/"ning"). A doubled consonant only ever keeps
    the whole pair or splits it evenly (never pushes it whole to the next syllable). Never leaves a
    final lone "y" as its own syllable. Falls back to `_split_cluster` if no prefix is a real word."""
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
    """Sonority-Sequencing-Principle syllabifier: finds vowel nuclei, splits each inter-nucleus
    cluster via `_word_boundary_split`. A word with 0 or 1 nucleus is returned whole."""
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
    """Returns syllable strings that concatenate back to `word` exactly, including punctuation."""
    if not word:
        return [word]

    # Strip leading/trailing punctuation before splitting; reattach to the first/last syllable.
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
    """Merges a syllable list down to exactly n_chunks contiguous text chunks."""
    n_chunks = max(1, n_chunks)
    if n_chunks >= len(parts):
        return parts
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
