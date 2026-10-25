"""Candidate reading lattice: normalization, edges, lexicon pack and filters."""

import re
import unicodedata

from .constants import MAX_EDGES, MAX_SPAN_READINGS, MAX_SURFACE_LEN

# ---------------------------------------------------------------- normalization

KANJI_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002fa1f々〆ヶ]"
)


def normalize_surface(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("|", "")


def hira(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch for ch in text
    )


def is_kanji(ch: str) -> bool:
    return bool(ch and KANJI_RE.fullmatch(ch))


def _char_type(char: str) -> int:
    if is_kanji(char):
        return 1
    code = ord(char)
    if 0x3040 <= code <= 0x309F:
        return 2
    if 0x30A0 <= code <= 0x30FF:
        return 3
    if char.isascii() and char.isalpha():
        return 4
    if char.isdigit():
        return 5
    return 6


# ---------------------------------------------------------------- edges


class Edge:
    __slots__ = ("start", "end", "surface", "reading", "source",
                 "prior", "confidence", "locked")

    def __init__(self, start, end, surface, reading, source="unknown",
                 prior=0.0, confidence=0.0, locked=False):
        self.start = start
        self.end = end
        self.surface = surface
        self.reading = reading
        self.source = source
        self.prior = prior
        self.confidence = confidence
        self.locked = locked

    def key(self):
        return (self.start, self.end, self.reading,
                "copy" if self.source == "copy" else "")


SOURCE_TO_ID = {
    "rules": 0, "haqumei": 1, "lyric_memory": 2, "kanjidic": 3,
    "pyopenjtalk": 4, "gold": 5, "copy": 6, "unknown": 7, "jmdict": 8,
    "unidic": 9, "generated": 10, "alnum": 11, "yomogi_dict": 12,
}
_PRIORITY_ORDER = (
    "gold", "lyric_memory", "alnum", "rules", "haqumei", "pyopenjtalk",
    "jmdict", "unidic", "yomogi_dict", "kanjidic", "copy")
SOURCE_PRIORITY = {
    name: len(_PRIORITY_ORDER) - index
    for index, name in enumerate(_PRIORITY_ORDER)
}

_DAKUTEN = dict(zip("かきくけこさしすせそたちつてとはひふへほう",
                    "がぎぐげござじずぜぞだぢづでどばびぶべぼゔ"))
_HANDAKU = dict(zip("はひふへほ", "ぱぴぷぺぽ"))
_WORD_SOURCES = frozenset({"jmdict", "unidic", "yomogi_dict", "pyopenjtalk"})
_PUNCT_OR_SPACE = frozenset(" \t\r\n、。！？!?…・~～-—")


# ---------------------------------------------------------------- lexicon pack


class UnifiedLexicon:
    """Port of UnifiedLexiconProvider (pack + rules + prefix scan)."""

    def __init__(self, table: dict, *, use_copy: bool = False):
        self.table = {k: tuple(v) for k, v in table.items()}
        self._lower_table: dict[str, tuple] = {}
        for k, v in self.table.items():
            low = k.lower()
            if low not in self._lower_table:
                self._lower_table[low] = v
        self.use_copy = use_copy

    @staticmethod
    def _entries(item) -> list[tuple[str, str, float, float]]:
        r, src, p, c = item if isinstance(item, tuple) and len(item) == 4 \
            else (item, "alnum", 1.2, 0.9)
        return [(r, src, p, c)]

    def build(self, text: str) -> list[Edge]:
        text = normalize_surface(text)
        n = len(text)
        edges: list[Edge] = []
        seen: set[tuple[int, int, str]] = set()

        for match in re.finditer(r"[A-Za-z0-9]+", text):
            span_start, span_end = match.span()
            word = match.group()
            entries = self.table.get(word) or self._lower_table.get(word.lower())
            if entries:
                for item in entries:
                    for r, src, p, c in self._entries(item):
                        if (span_start, span_end, r) not in seen:
                            seen.add((span_start, span_end, r))
                            edges.append(Edge(span_start, span_end, word, r,
                                              src, p, c))
            else:
                for k in range(len(word) - 1, 1, -1):
                    prefix = word[:k]
                    sub_entries = (self.table.get(prefix)
                                   or self._lower_table.get(prefix.lower()))
                    if sub_entries:
                        for item in sub_entries:
                            for r, src, p, c in self._entries(item):
                                if (span_start, span_start + k, r) not in seen:
                                    seen.add((span_start, span_start + k, r))
                                    edges.append(Edge(span_start, span_start + k,
                                                      prefix, r, src, p, c))
                        break

        for i in range(n):
            limit = min(MAX_SURFACE_LEN, n - i)
            for k in range(1, limit + 1):
                sub = text[i:i + k]
                if sub.isascii() and sub.isalnum():
                    continue
                entries = self.table.get(sub)
                if entries is not None:
                    for item in entries:
                        r, src, p, c = item if isinstance(item, tuple) and len(item) == 4 \
                            else (item, "jmdict", 0.85, 0.82)
                        if (i, i + k, r) not in seen:
                            seen.add((i, i + k, r))
                            edges.append(Edge(i, i + k, sub, r, src, p, c))

        if self.use_copy:
            edges.extend(copy_edges(text))
        return filter_truncated_okurigana(edges, text)


def copy_edges(text: str) -> list[Edge]:
    return [Edge(i, i + 1, ch, ch, "copy", prior=-1.0)
            for i, ch in enumerate(text)]


# ------------------------------------------------- composite provider chain


def _rank(edge: Edge) -> tuple[int, float, float]:
    return (SOURCE_PRIORITY.get(edge.source, -1), edge.confidence, edge.prior)


def expand_word_variants(edges: list[Edge], text: str) -> list[Edge]:
    out = list(edges)
    seen = {(e.start, e.end, e.reading) for e in edges}
    for edge in edges:
        if (edge.source not in _WORD_SOURCES or edge.locked
                or edge.end - edge.start < 2 or len(edge.reading) < 2):
            continue
        if edge.start == 0:
            continue
        if text and text[edge.start - 1] in _PUNCT_OR_SPACE:
            continue
        if edge.surface and edge.surface[0].isdigit():
            continue
        first = edge.reading[0]
        for mapped in (_DAKUTEN.get(first), _HANDAKU.get(first)):
            if not mapped:
                continue
            reading = mapped + edge.reading[1:]
            marker = (edge.start, edge.end, reading)
            if marker not in seen:
                seen.add(marker)
                out.append(Edge(edge.start, edge.end, edge.surface, reading,
                                edge.source, max(0.0, edge.prior - 0.4),
                                edge.confidence * 0.85))
    return out


def filter_truncated_okurigana(edges: list[Edge], text: str) -> list[Edge]:
    longer_spans: dict[tuple[int, str], int] = {}
    for e in edges:
        key = (e.start, e.reading)
        if key not in longer_spans or e.end > longer_spans[key]:
            longer_spans[key] = e.end
    filtered = []
    for e in edges:
        max_end = longer_spans.get((e.start, e.reading), e.end)
        if max_end > e.end:
            ext = text[e.end:max_end]
            if ext and all("\u3040" <= ch <= "\u30ff" for ch in ext):
                continue
        filtered.append(e)
    return filtered


def filter_rendaku_traps(edges: list[Edge]) -> list[Edge]:
    by_span: dict[tuple[int, int], list[int]] = {}
    for index, edge in enumerate(edges):
        by_span.setdefault((edge.start, edge.end), []).append(index)
    drop: set[int] = set()
    for indices in by_span.values():
        readings = {edges[i].reading for i in indices
                    if edges[i].source in _WORD_SOURCES}
        if len(readings) < 2:
            continue
        for i in indices:
            edge = edges[i]
            if edge.source not in _WORD_SOURCES or edge.locked:
                continue
            reading = edge.reading
            for k in range(1, len(reading)):
                voiced = _DAKUTEN.get(reading[k]) or _HANDAKU.get(reading[k])
                if voiced and reading[:k] + voiced + reading[k + 1:] in readings:
                    drop.add(i)
                    break
    if not drop:
        return edges
    return [edge for index, edge in enumerate(edges) if index not in drop]


def _has_complete_path(edges: list[Edge], length: int) -> bool:
    reachable = [False] * (length + 1)
    reachable[0] = True
    by_start: dict[int, list[int]] = {}
    for edge in edges:
        if 0 <= edge.start < edge.end <= length:
            by_start.setdefault(edge.start, []).append(edge.end)
    for start in range(length):
        if reachable[start]:
            for end in by_start.get(start, ()):
                reachable[end] = True
    return reachable[length]


def repair_complete_path(edges: list[Edge], text: str) -> list[Edge]:
    if _has_complete_path(edges, len(text)):
        return edges
    present = {(e.start, e.end, e.reading, e.source) for e in edges}
    for edge in copy_edges(text):
        key = (edge.start, edge.end, edge.reading, edge.source)
        if key not in present:
            edges.append(edge)
            present.add(key)
    if not _has_complete_path(edges, len(text)):
        raise ValueError("candidate graph has no complete 0..L path")
    return edges


def _hira_char(ch: str) -> bool:
    return "ぁ" <= ch <= "ゖ"


def derive_okurigana_stems(edges: list[Edge], text: str) -> list[Edge]:
    existing = {(e.start, e.end, e.reading) for e in edges}
    derived: list[Edge] = []
    for e in edges:
        surf = text[e.start:e.end]
        if len(surf) < 2:
            continue
        k = len(surf)
        while k > 0 and _hira_char(surf[k - 1]):
            k -= 1
        okuri = surf[k:]
        stem = surf[:k]
        if not okuri or not stem or not is_kanji(stem[0]):
            continue
        if any(_hira_char(ch) for ch in stem):
            continue
        if not e.reading.endswith(okuri):
            continue
        stem_reading = e.reading[:-len(okuri)]
        if not stem_reading:
            continue
        key = (e.start, e.start + k, stem_reading)
        if key in existing:
            continue
        existing.add(key)
        derived.append(Edge(e.start, e.start + k, stem, stem_reading,
                            e.source, e.prior, e.confidence, False))
    return edges + derived


def _prune_key(edge: Edge):
    priority, confidence, prior = _rank(edge)
    return (-priority, -confidence, -prior, edge.start, edge.end,
            edge.reading, edge.source)


def prune(edges: list[Edge], text: str) -> list[Edge]:
    mandatory = {(e.start, e.end, e.reading, e.source)
                 for e in copy_edges(text)}
    keep = list(edges)
    spans: dict[tuple[int, int], list[Edge]] = {}
    for edge in keep:
        spans.setdefault((edge.start, edge.end), []).append(edge)
    keep = []
    for span in sorted(spans):
        ranked = sorted(spans[span], key=_prune_key)
        copies = [e for e in ranked
                  if (e.start, e.end, e.reading, e.source) in mandatory]
        others = [e for e in ranked if e not in copies]
        selected = others[:MAX_SPAN_READINGS]
        reserve = max(1, MAX_SPAN_READINGS // 2)
        fallback = [e for e in others
                    if e.source == "kanjidic" and e not in selected]
        for edge in fallback:
            if sum(1 for e in selected if e.source == "kanjidic") >= reserve:
                break
            replaceable = [e for e in selected if e.source != "kanjidic"]
            if not replaceable:
                break
            selected[selected.index(replaceable[-1])] = edge
        keep.extend(copies)
        keep.extend(selected)
    copies = [e for e in keep
              if (e.start, e.end, e.reading, e.source) in mandatory]
    others = [e for e in keep if e not in copies]
    copy_keys = {(e.start, e.end, e.reading, e.source)
                 for e in copies if e.source == "copy"}
    for edge in copy_edges(text):
        key = (edge.start, edge.end, edge.reading, edge.source)
        if key not in copy_keys:
            copies.append(edge)
            copy_keys.add(key)
    budget = max(0, MAX_EDGES - len(copies))
    keep = copies + sorted(others, key=_prune_key)[:budget]
    return keep


def build_edges(text: str, lexicon: UnifiedLexicon,
                context_meta: dict | None = None) -> list[Edge]:
    text = normalize_surface(text)
    edges = lexicon.build(text)
    if context_meta:
        # Applied before variant/stem derivation so derived edges inherit
        # the contextual metadata, mirroring the training-time provider.
        for edge in edges:
            override = context_meta.get((edge.start, edge.end, edge.reading))
            if override is not None:
                edge.source, edge.prior, edge.confidence = override
    edges = expand_word_variants(edges, text)
    unique: dict[tuple, Edge] = {}
    for edge in edges:
        old = unique.get(edge.key())
        if old is None or _rank(edge) > _rank(old):
            unique[edge.key()] = edge
    result = list(unique.values())
    result = prune(result, text)
    result = filter_truncated_okurigana(result, text)
    result = filter_rendaku_traps(result)
    result = repair_complete_path(result, text)
    result = derive_okurigana_stems(result, text)
    return sorted(result,
                  key=lambda e: (e.start, e.end, -e.prior, e.reading, e.source))
