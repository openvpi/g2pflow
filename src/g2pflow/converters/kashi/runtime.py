"""ONNX scoring and decoding runtime for kashi-g2p."""

import lzma
from itertools import product
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "japanese-kashi requires optional dependencies. Install them with: "
        'python -m pip install "g2pflow[kashi]"'
    ) from exc

from .constants import MAX_EDGES, READING_LEN, SEQ_LEN
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


def windows_from_segments(segments, length: int) -> list[tuple[int, int]]:
    """Split ``[0, length)`` at fixed-word boundaries into <=SEQ_LEN windows.

    Window edges are segment boundaries whenever possible, so no fixed word is
    bisected.  A segment longer than the model window is split and decoded
    chunk by chunk.
    """
    if length <= 0:
        return []
    if length <= SEQ_LEN:
        return [(0, length)]
    windows: list[tuple[int, int]] = []
    start = 0
    for segment in segments:
        seg_start, seg_end = int(segment.start), int(segment.end)
        if seg_end <= start:
            continue
        if seg_end - start <= SEQ_LEN:
            continue
        while seg_start - start >= SEQ_LEN:
            windows.append((start, start + SEQ_LEN))
            start += SEQ_LEN
        if seg_start > start:
            windows.append((start, seg_start))
            start = seg_start
        while seg_end - start > SEQ_LEN:
            windows.append((start, start + SEQ_LEN))
            start += SEQ_LEN
    while length - start > SEQ_LEN:
        windows.append((start, start + SEQ_LEN))
        start += SEQ_LEN
    if start < length:
        windows.append((start, length))
    for window_start, window_end in windows:
        if not 0 <= window_start < window_end <= length:
            raise ValueError("segment window is outside the input")
        if window_end - window_start > SEQ_LEN:
            raise ValueError(
                "segment window exceeds the model limit; segmentation is "
                "not contiguous")
    return windows


def _segment_contained(edges: list[Edge], segments, start: int,
                       end: int) -> list[Edge]:
    """Drop edges that cross a fixed-word boundary."""
    segment_end = [0] * (end - start)
    for segment in segments:
        seg_start = max(int(segment.start), start)
        seg_end = min(int(segment.end), end)
        for position in range(seg_start - start, seg_end - start):
            segment_end[position] = seg_end - start
    return [edge for edge in edges
            if 0 <= edge.start < len(segment_end)
            and segment_end[edge.start] > 0
            and edge.end <= segment_end[edge.start]]


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
    """Load (pack, prior, vocab) from a lexicon.txz bundle."""
    with lzma.open(bundle_path, "rb") as fh:
        raw = fh.read()
    table: dict[str, tuple] = {}
    prior: dict[tuple[str, str], int] = {}
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
        elif section == "vocab":
            vocab.append(line)
    return table, prior, vocab


class G2POnnxRuntime:
    def __init__(self, *, model_path: str | Path,
                 bundle_path: str | Path, prefer_dml: bool = False):
        table, prior, vocab = _load_bundle(Path(bundle_path))
        self.table = table
        self.prior_index = prior
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

    def _features(self, text: str):
        unk = self.unk_id
        input_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
        attention = np.zeros((1, SEQ_LEN), dtype=bool)
        for index, char in enumerate(text[:SEQ_LEN]):
            input_ids[0, index] = self.vocab.get(char, unk)
            attention[0, index] = True
        char_types = np.zeros((1, SEQ_LEN), dtype=np.int64)
        for index, char in enumerate(text[:SEQ_LEN]):
            char_types[0, index] = _char_type(char)
        return input_ids, attention, char_types

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
        input_ids, attention, char_types = self._features(text)
        feeds = self._edge_tensors(edges)
        feeds.update(input_ids=input_ids, attention_mask=attention,
                     char_type_ids=char_types)
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
        # One graph per fixed-word window: a global graph can exceed MAX_EDGES.
        scored = []
        for wstart, wend in windows_from_segments(segments, len(text)):
            local = _segment_contained(
                build_edges(text[wstart:wend], self.lexicon),
                segments, wstart, wend)
            if local:
                scored.append((wstart, wend, local,
                               self.score(text[wstart:wend], local)))
            else:
                scored.append((wstart, wend, [], np.zeros(0)))

        words: list[dict] = []
        for segment in segments:
            start, end = segment.start, segment.end
            if not str(segment.surface).strip():
                # Whitespace has no MeCab node and gets no reading.
                continue
            candidates: list[dict] = []
            chunks: list[tuple[list[Edge], list[float]]] = []
            for wstart, wend, local, scores in scored:
                if wend <= start or end <= wstart:
                    continue
                offset = start - wstart
                chunk_edges: list[Edge] = []
                chunk_scores: list[float] = []
                for j, edge in enumerate(local):
                    if offset <= edge.start and edge.end <= end - wstart:
                        chunk_edges.append(Edge(
                            edge.start - offset, edge.end - offset,
                            edge.surface, edge.reading, edge.source,
                            edge.prior, edge.confidence, edge.locked))
                        chunk_scores.append(float(scores[j]))
                if chunk_edges:
                    chunks.append((chunk_edges, chunk_scores))
            if len(chunks) == 1:
                chunk_edges, chunk_scores = chunks[0]
                try:
                    paths = decode_beam(chunk_scores, chunk_edges,
                                        end - start, beam_size)
                except ValueError:
                    paths = []
                seen: set[str] = set()
                for path_score, indices in paths:
                    reading = "".join(chunk_edges[i].reading
                                      for i in indices)
                    if reading not in seen:
                        seen.add(reading)
                        candidates.append({"reading": reading,
                                           "score": path_score})
                    if len(candidates) >= beam_size:
                        break
            elif chunks:
                # Chunks are scored by separate forward passes, so decode them
                # independently and join the best paths left to right.
                combos: list[tuple[float, str]] = [(0.0, "")]
                for chunk_edges, chunk_scores in chunks:
                    low = min(edge.start for edge in chunk_edges)
                    high = max(edge.end for edge in chunk_edges)
                    shifted = [Edge(edge.start - low, edge.end - low,
                                    edge.surface, edge.reading, edge.source,
                                    edge.prior, edge.confidence, edge.locked)
                               for edge in chunk_edges]
                    try:
                        paths = decode_beam(chunk_scores, shifted,
                                            high - low, beam_size)
                    except ValueError:
                        paths = []
                    if not paths:
                        combos = []
                        break
                    chunk_readings = [
                        (float(path_score),
                         "".join(shifted[i].reading for i in indices))
                        for path_score, indices in paths[:2]]
                    combos = [
                        (base_score + chunk_score, base + reading)
                        for base_score, base in combos
                        for chunk_score, reading in chunk_readings]
                    combos.sort(key=lambda item: -item[0])
                    combos = combos[:beam_size]
                seen = set()
                for path_score, reading in combos:
                    if reading in seen:
                        continue
                    seen.add(reading)
                    candidates.append({"reading": reading,
                                       "score": path_score})
            words.append({"surface": segment.surface, "start": start,
                          "end": end, "candidates": candidates})
        return words
