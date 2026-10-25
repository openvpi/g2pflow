"""Contextual edge metadata derived from UniDic token readings.

The model consumes per-edge source/prior/confidence features that the
training-time pipeline produced from an in-context G2P frontend.  At
inference fugashi/UniDic supplies the same signals: the token kana
reading restores the word-level features, and a decomposition over the
pack's single-character entries restores the character-level ones.
"""

from .lexicon import hira, is_kanji

CONTEXT_METADATA = ("pyopenjtalk", 0.75, 0.90)


def _is_hira_or_long(ch: str) -> bool:
    return "\u3041" <= ch <= "\u3096" or ch == "\u30fc"


def char_readings_from_table(table: dict) -> dict[str, frozenset[str]]:
    """Single-character pack readings, the decomposition vocabulary."""
    readings: dict[str, frozenset[str]] = {}
    for surface, entries in table.items():
        if len(surface) == 1 and is_kanji(surface):
            readings[surface] = frozenset(entry[0] for entry in entries)
    return readings


def decompose_reading(surface: str, reading: str,
                      char_readings: dict[str, frozenset[str]],
                      cache: dict | None = None,
                      max_paths: int = 20000) -> tuple[tuple[int, str], ...]:
    """Per-character readings of a kanji token via DP over pack entries.

    Returns only positions whose reading is identical across every
    successful decomposition; ambiguous or unalignable tokens yield ().
    """
    if cache is not None:
        cached = cache.get((surface, reading))
        if cached is not None:
            return cached

    n, m = len(surface), len(reading)
    reach = [[False] * (m + 1) for _ in range(n + 1)]
    reach[0][0] = True
    for i in range(n):
        char = surface[i]
        kanji = char_readings.get(char, ()) if is_kanji(char) else None
        for j in range(m + 1):
            if not reach[i][j]:
                continue
            if kanji:
                for candidate in kanji:
                    if reading.startswith(candidate, j):
                        reach[i + 1][j + len(candidate)] = True
            elif j < m and reading[j] == char and _is_hira_or_long(char):
                reach[i + 1][j + 1] = True

    result: tuple[tuple[int, str], ...] = ()
    if n and reach[n][m]:
        per_position: dict[int, set[str]] = {}
        stack = [(n, m, ())]
        pops = 0
        while stack:
            i, j, acc = stack.pop()
            pops += 1
            if pops > max_paths:
                per_position = {}
                break
            if i == 0 and j == 0:
                for index, value in acc:
                    per_position.setdefault(index, set()).add(value)
                continue
            char = surface[i - 1]
            if is_kanji(char):
                for candidate in char_readings.get(char, ()):
                    k = len(candidate)
                    if (k <= j and reach[i - 1][j - k]
                            and reading[j - k:j] == candidate):
                        stack.append((i - 1, j - k, ((i - 1, candidate),) + acc))
            elif j >= 1 and reach[i - 1][j - 1] and reading[j - 1] == char \
                    and _is_hira_or_long(char):
                stack.append((i - 1, j - 1, acc))
        result = tuple((index, next(iter(values)))
                       for index, values in sorted(per_position.items())
                       if len(values) == 1)

    if cache is not None and len(cache) < 200000:
        cache[(surface, reading)] = result
    return result


def build_context_metadata(segments, char_readings: dict[str, frozenset[str]],
                           cache: dict | None = None) -> dict:
    """Metadata overrides keyed by (start, end, reading) for pack edges."""
    metadata: dict[tuple[int, int, str], tuple[str, float, float]] = {}
    for segment in segments:
        kana = getattr(segment, "kana", "")
        if not kana or not any(is_kanji(ch) for ch in segment.surface):
            continue
        reading = hira(kana)
        if not reading:
            continue
        metadata[(segment.start, segment.end, reading)] = CONTEXT_METADATA
        for index, char_reading in decompose_reading(
                segment.surface, reading, char_readings, cache):
            start = segment.start + index
            metadata[(start, start + 1, char_reading)] = CONTEXT_METADATA
    return metadata
