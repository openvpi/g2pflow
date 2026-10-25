"""ONNX scoring and decoding runtime for kashi-g2p."""

import lzma
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "japanese-kashi requires optional dependencies. Install them with: "
        'python -m pip install "g2pflow[kashi]"'
    ) from exc

from .constants import MAX_COMPONENTS, MAX_EDGES, READING_LEN, SEQ_LEN
from .context import build_context_metadata, char_readings_from_table
from .decoder import decode_beam
from .lexicon import (
    Edge,
    SOURCE_TO_ID,
    UnifiedLexicon,
    _char_type,
    build_edges,
    hira,
    normalize_surface,
)

_BUNDLE_MAGIC = "##kashi-g2p v1"
_BUNDLE_SECTION = "##section "
_BUNDLE_END = "##end"


# ---------------------------------------------------------------- windows


def safe_windows(text: str, edges: list[Edge]) -> list[tuple[int, int]]:
    if len(text) <= SEQ_LEN:
        return [(0, len(text))]
    windows = []
    start = 0
    while start < len(text):
        limit = min(len(text), start + SEQ_LEN)
        end = limit
        if limit < len(text):
            while True:
                crossing = [e for e in edges
                            if e.start < end < e.end and e.end > start]
                if not crossing:
                    break
                moved = min(e.start for e in crossing)
                if moved >= end:
                    raise RuntimeError("safe window boundary did not advance")
                end = moved
        if end <= start:
            raise ValueError(
                "provider edge exceeds the window limit; no safe split")
        windows.append((start, end))
        start = end
    return windows


def local_edges(text: str, edges: list[Edge], start: int, end: int) -> list[Edge]:
    result = []
    for edge in edges:
        if start <= edge.start and edge.end <= end:
            result.append(Edge(edge.start - start, edge.end - start,
                               edge.surface, edge.reading, edge.source,
                               edge.prior, edge.confidence, edge.locked))
        elif edge.start < end and start < edge.end:
            raise RuntimeError("internal error: a window bisected an edge")
    if not result:
        raise ValueError("no edges for a safe window")
    return result


# ---------------------------------------------------------------- runtime


def _load_bundle(bundle_path: Path):
    """Load (pack, prior, components, vocab) from a lexicon.txz bundle."""
    with lzma.open(bundle_path, "rb") as fh:
        raw = fh.read()
    table: dict[str, tuple] = {}
    prior: dict[tuple[str, str], int] = {}
    components: dict[str, list] = {}
    vocab: list[str] = []
    section = None
    last_surface = None
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("##"):
            if line in (_BUNDLE_MAGIC, _BUNDLE_END):
                continue
            if line.startswith(_BUNDLE_SECTION):
                section = line[len(_BUNDLE_SECTION):]
                last_surface = None
                continue
            raise ValueError(f"bad header line in {bundle_path}: {line!r}")
        if section == "pack":
            surface, reading, source, bonus, weight = line.split("\t")
            entry = (reading, source, float(bonus), float(weight))
            if surface == last_surface:
                table[surface] += (entry,)
            else:
                table[surface] = (entry,)
                last_surface = surface
        elif section == "prior":
            surface, reading, row = line.split("\t")
            prior[(surface, reading)] = int(row)
        elif section == "components":
            char, _, ids = line.partition("\t")
            components[char] = [int(v) for v in ids.split(",")] if ids else []
        elif section == "vocab":
            vocab.append(line)
    return table, prior, components, vocab


class G2POnnxRuntime:
    def __init__(self, *, model_path: str | Path,
                 bundle_path: str | Path, prefer_dml: bool = False):
        table, prior, components, vocab = _load_bundle(Path(bundle_path))
        self.table = table
        self.prior_index = prior
        self.components = components
        self.vocab = {token: index for index, token in enumerate(vocab)}
        self.unk_id = int(self.vocab.get("[UNK]", 1))
        self.lexicon = UnifiedLexicon(self.table, use_copy=False)
        providers = ort.get_available_providers()
        use = ["DmlExecutionProvider", "CPUExecutionProvider"] \
            if prefer_dml and "DmlExecutionProvider" in providers \
            else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=use)
        self.provider_name = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]
        self._reading_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._char_readings: dict[str, frozenset[str]] | None = None
        self._decomposition_cache: dict[tuple[str, str],
                                        tuple[tuple[int, str], ...]] = {}

    def _context_metadata(self, segments) -> dict:
        if self._char_readings is None:
            self._char_readings = char_readings_from_table(self.table)
        return build_context_metadata(
            segments, self._char_readings, self._decomposition_cache)

    def _features(self, text: str):
        unk = self.unk_id
        input_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
        attention = np.zeros((1, SEQ_LEN), dtype=bool)
        for index, char in enumerate(text[:SEQ_LEN]):
            input_ids[0, index] = self.vocab.get(char, unk)
            attention[0, index] = True
        components = np.zeros((1, SEQ_LEN, MAX_COMPONENTS), dtype=np.int64)
        char_types = np.zeros((1, SEQ_LEN), dtype=np.int64)
        for index, char in enumerate(text[:SEQ_LEN]):
            char_types[0, index] = _char_type(char)
            for j, value in enumerate(self.components.get(char, ())[:MAX_COMPONENTS]):
                components[0, index, j] = value
        return input_ids, attention, components, char_types

    def _reading_tokens(self, reading: str) -> tuple[np.ndarray, np.ndarray]:
        """Token ids for one reading string, cached (readings repeat heavily)."""
        cached = self._reading_cache.get(reading)
        if cached is None:
            ids = np.zeros((READING_LEN,), dtype=np.int64)
            mask = np.zeros((READING_LEN,), dtype=bool)
            for j, char in enumerate(hira(reading)[:READING_LEN]):
                ids[j] = self.vocab.get(char, self.unk_id)
                mask[j] = True
            if len(self._reading_cache) < 200000:
                self._reading_cache[reading] = (ids, mask)
            return ids, mask
        return cached

    def _edge_tensors(self, edges: list[Edge]):
        count = len(edges)
        if count > MAX_EDGES:
            raise ValueError(f"{count} edges exceed the candidate cap ({MAX_EDGES})")
        starts = np.zeros((1, count), dtype=np.int64)
        ends = np.zeros((1, count), dtype=np.int64)
        reading_ids = np.zeros((1, count, READING_LEN), dtype=np.int64)
        reading_mask = np.zeros((1, count, READING_LEN), dtype=bool)
        mask = np.ones((1, count), dtype=bool)
        source_ids = np.zeros((1, count), dtype=np.int64)
        prior = np.zeros((1, count), dtype=np.float32)
        pack_ids = np.zeros((1, count), dtype=np.int64)
        locked = np.zeros((1, count), dtype=bool)
        for index, edge in enumerate(edges):
            starts[0, index] = edge.start
            ends[0, index] = edge.end
            mask[0, index] = True
            source_ids[0, index] = SOURCE_TO_ID.get(edge.source, 7)
            prior[0, index] = edge.prior
            pack_ids[0, index] = self.prior_index.get(
                (edge.surface, edge.reading), 0)
            ids_row, mask_row = self._reading_tokens(edge.reading)
            reading_ids[0, index] = ids_row
            reading_mask[0, index] = mask_row
        return dict(edge_start=starts, edge_end=ends,
                    edge_reading_ids=reading_ids,
                    edge_reading_mask=reading_mask,
                    edge_mask=mask, edge_source_ids=source_ids,
                    edge_prior=prior, edge_pack_ids=pack_ids,
                    edge_locked=locked)

    def score(self, text: str, edges: list[Edge]) -> np.ndarray:
        input_ids, attention, components, char_types = self._features(text)
        feeds = self._edge_tensors(edges)
        feeds.update(input_ids=input_ids, attention_mask=attention,
                     component_ids=components, char_type_ids=char_types)
        scores = self.session.run(["edge_scores"], feeds)[0]
        return scores[0][:len(edges)]

    def predict_words(self, text: str, segments, *, beam_size: int = 16) -> list[dict]:
        """Per-word N-best readings with boundaries frozen to ``segments``.
        Candidates are the top ``beam_size`` paths through each segment's
        sub-lattice, ranked by the in-context model score.
        """
        text = normalize_surface(text)
        if not text:
            return []
        global_edges = build_edges(
            text, self.lexicon, self._context_metadata(segments))
        windows = safe_windows(text, global_edges)
        scored = []
        for wstart, wend in windows:
            local = local_edges(text, global_edges, wstart, wend)
            scored.append((wstart, wend, local,
                           self.score(text[wstart:wend], local)))

        words: list[dict] = []
        for segment in segments:
            start, end = segment.start, segment.end
            candidates: list[dict] = []
            sub_edges: list[Edge] = []
            sub_scores: list[float] = []
            for wstart, wend, local, scores in scored:
                if wend <= start or end <= wstart:
                    continue
                offset = start - wstart
                for j, edge in enumerate(local):
                    if offset <= edge.start and edge.end <= end - wstart:
                        sub_edges.append(Edge(
                            edge.start - offset, edge.end - offset,
                            edge.surface, edge.reading, edge.source,
                            edge.prior, edge.confidence, edge.locked))
                        sub_scores.append(float(scores[j]))
            if sub_edges:
                try:
                    paths = decode_beam(sub_scores, sub_edges, end - start,
                                        beam_size)
                except ValueError:
                    paths = []
                seen: set[str] = set()
                for path_score, indices in paths:
                    reading = "".join(sub_edges[i].reading for i in indices)
                    if reading not in seen:
                        seen.add(reading)
                        candidates.append({"reading": reading,
                                           "score": path_score})
                    if len(candidates) >= beam_size:
                        break
            words.append({"surface": segment.surface, "start": start,
                          "end": end, "candidates": candidates})
        return words
