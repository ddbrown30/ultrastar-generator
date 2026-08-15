"""Word-text normalization shared by every text-matching mechanism in
this codebase (LRC/ASR/reference-lyrics/ground-truth comparison).

Extracted 2026-08-14: the exact same normalization logic had drifted into
6 independent per-module copies (`lyrics_lookup.py`, `lrc_timing.py`,
`mxl_lrc_generator.py`, `realign.py`, `musicxml_reference.py`,
`verify_existing_song.py`). A real bug -- curly apostrophes (U+2019, e.g.
a real hand-authored existing file's own "Johnny's") silently DELETED
instead of folded to match a straight-ASCII equivalent on the other side
of a comparison, desyncing an otherwise byte-identical line/word -- had
already been found and fixed independently in 4 of the 6 copies
(`lyrics_lookup.py`, `lrc_timing.py`, `mxl_lrc_generator.py`,
`realign.py`) before this consolidation. The other two
(`musicxml_reference.py`, `verify_existing_song.py`) had NOT received
that fix and carried the same latent bug -- only found BY this
consolidation. Use this in any new text-matching mechanism instead of
writing another copy.

`verification.py` has its OWN, deliberately different `_normalize`
(strips only leading/trailing punctuation, keeps internal characters
untouched) -- not folded in here, since it serves a genuinely different
comparison (fuzzy-ratio matching where internal word structure matters),
not another drifted copy of this one."""
import re


def normalize_word(s: str) -> str:
    """Lowercase, fold curly apostrophes to straight, strip everything
    else except letters/digits/apostrophes. Apostrophes are deliberately
    kept (not stripped like other punctuation) -- they carry real
    information for a contraction ("it's" vs "its") that discarding them
    entirely would lose."""
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", s)
