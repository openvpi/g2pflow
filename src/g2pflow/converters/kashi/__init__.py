"""Japanese G2P converter backed by the kashi-g2p ONNX model.
UniDic fixes the word boundaries; each word exposes beam-search N-best
readings through the Kana phoneme stage.
"""

from __future__ import annotations

import os
import shutil
import unicodedata
import urllib.request
from itertools import product
from pathlib import Path

from ...registry import converter
from ..base import (Converter, G2PConversionError, G2PReading, G2PWord,
                    compose_word)
from ..japanese import JapaneseKanaConverter, _is_japanese_char
from .lexicon import hira, normalize_surface
from .segmentation import UnidicSegmenter
from ..text import find_run, is_kana

_RELEASE_TAG = "v1.1.0"
_RELEASE_BASE = ("https://github.com/Xiantaidu/kashi-g2p/releases/"
                 f"download/{_RELEASE_TAG}")
MODEL_URL = f"{_RELEASE_BASE}/model.onnx"
LEXICON_URL = f"{_RELEASE_BASE}/lexicon.txz"


def _default_cache_dir() -> Path:
    return Path(os.environ.get("G2PFLOW_CACHE_DIR", "")) \
        if os.environ.get("G2PFLOW_CACHE_DIR") \
        else Path.home() / ".cache" / "g2pflow"



_REQUIRED_MODEL_INPUTS = frozenset({
    "input_ids", "attention_mask", "char_type_ids", "edge_start", "edge_end",
    "edge_reading_ids", "edge_reading_mask", "edge_mask", "edge_source_ids",
    "edge_prior", "edge_pack_ids", "edge_locked",
})


def _model_inputs(path: Path) -> frozenset[str] | None:
    """Input names of an ONNX file, or None when it cannot be read."""
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        return None
    try:
        session = ort.InferenceSession(str(path),
                                       providers=["CPUExecutionProvider"])
    except Exception:  # noqa: BLE001 - truncated/corrupt cache entry
        return None
    return frozenset(item.name for item in session.get_inputs())


def _is_stale(path: Path) -> bool:
    """Whether an existing model file predates the current input signature."""
    return path.is_file() and not _model_is_current(path)


def _install(url: str, path: Path, *, stale=None) -> None:
    """Download *url* and move it into place only after the transfer succeeded.

    With *stale*, skip when a concurrent worker already refreshed the file.
    """
    if stale is not None and not stale(path):
        return
    print(f"[g2pflow] downloading {url} ...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".staged")
    try:
        _download(url, staged)
        path.with_suffix(path.suffix + ".inputs").unlink(missing_ok=True)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _model_is_current(path: Path) -> bool:
    expected = ",".join(sorted(_REQUIRED_MODEL_INPUTS))
    stamp = path.with_suffix(path.suffix + ".inputs")
    try:
        if path.is_file() \
                and stamp.read_text(encoding="utf-8").strip() == expected:
            return True
    except OSError:
        pass
    names = _model_inputs(path)
    if names is None or names != _REQUIRED_MODEL_INPUTS:
        return False
    try:
        stamp.write_text(expected + "\n", encoding="utf-8")
    except OSError:
        pass
    return True


def _download(url: str, path: Path, *, attempts: int = 3) -> None:
    """Stream *url* to *path*, retrying and verifying Content-Length."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = path.with_suffix(path.suffix + f".part{attempt}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total = response.headers.get("Content-Length")
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(response, fh)
            if total is not None and tmp.stat().st_size != int(total):
                raise IOError(f"truncated download: {url} "
                              f"({tmp.stat().st_size} of {total} bytes)")
            tmp.replace(path)
            return
        except Exception as exc:  # noqa: BLE001 - retry any transfer error
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                print(f"[g2pflow] download failed ({exc}); retrying "
                      f"({attempt}/{attempts}) ...", flush=True)
    raise IOError(f"could not download {url}") from last_error


def _kana_only(text: str) -> bool:
    return bool(text) and all(is_kana(ch) or ch == "ー" for ch in text)


def _normalize_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """NFKC the run and remember each normalized char's source span.

    Combining marks stay with their base character.
    """
    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and (
                unicodedata.combining(text[end])
                or text[end] in "\u3099\u309a"):
            end += 1
        for char in normalize_surface(text[index:end]):
            normalized.append(char)
            starts.append(index)
            ends.append(end)
        index = end
    return "".join(normalized), starts, ends


@converter(id="japanese-kashi", language="ja,jpn")
class JapaneseKashiConverter(Converter):
    """Japanese G2P readings via the kashi-g2p ONNX model.
    *dict_path*: romaji-to-phoneme dictionary; *model_path* / *lexicon_path* /
    *cache_dir* / *unidic_dir*: auto-download resources; *beam_size* caps
    candidates per word; 1 yields the best reading only.
    """

    def __init__(self, dict_path: str, *,
                 model_path: str | None = None,
                 lexicon_path: str | None = None,
                 cache_dir: str | None = None,
                 prefer_dml: bool = False,
                 beam_size: int = 16,
                 unidic_dir: str | None = None) -> None:
        self._kana = JapaneseKanaConverter(dict_path=dict_path)
        cache = Path(cache_dir) if cache_dir else _default_cache_dir() / "kashi"
        cache = cache / _RELEASE_TAG
        # Only files this package placed in its own cache may be replaced.
        # Caller-supplied paths belong to the caller.
        self._model_is_managed = model_path is None
        self._lexicon_is_managed = lexicon_path is None
        self._model_path = Path(model_path) if model_path else cache / "model.onnx"
        self._lexicon_path = Path(lexicon_path) if lexicon_path else cache / "lexicon.txz"
        self._prefer_dml = prefer_dml
        self._beam_size = max(1, int(beam_size))
        self._unidic_dir = unidic_dir
        self._runtime = None
        self._segmenter = None

    def find(self, text: str) -> tuple[int, int] | None:
        # Japanese-script runs only, mirroring japanese-mecab.
        return find_run(text, _is_japanese_char)

    def _get_segmenter(self):
        if self._segmenter is None:
            self._segmenter = UnidicSegmenter(self._unidic_dir)
        return self._segmenter

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        # Caller-supplied files are never touched. Only cache entries this
        # package manages may be refreshed when out of date.
        user_missing = [
            path for path, managed in (
                (self._model_path, self._model_is_managed),
                (self._lexicon_path, self._lexicon_is_managed))
            if not managed and not path.is_file()]
        if user_missing:
            raise FileNotFoundError(
                "kashi model file not found: "
                + ", ".join(str(path) for path in user_missing))
        if self._needs_download():
            try:
                from filelock import FileLock
            except ModuleNotFoundError as exc:
                if exc.name != "filelock":
                    raise
                raise ImportError(
                    'kashi model download requires filelock. Install it with: '
                    'python -m pip install "g2pflow[kashi]"'
                ) from exc
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            # Serialize first-use downloads across spawned workers.
            with FileLock(str(self._model_path.with_suffix(".download.lock"))):
                if self._needs_download():
                    self._fetch_managed()
        from .runtime import G2POnnxRuntime
        self._runtime = G2POnnxRuntime(
            model_path=self._model_path,
            bundle_path=self._lexicon_path,
            prefer_dml=self._prefer_dml,
        )
        return self._runtime

    def _needs_download(self) -> bool:
        """Whether a package-managed file is missing or out of date."""
        if self._model_is_managed:
            if not self._model_path.is_file() or _is_stale(self._model_path):
                return True
        if self._lexicon_is_managed and not self._lexicon_path.is_file():
            return True
        return False

    def _fetch_managed(self) -> None:
        """Download into place, keeping any existing file until success."""
        if self._model_is_managed:
            if not self._model_path.is_file():
                _install(MODEL_URL, self._model_path)
            elif _is_stale(self._model_path):
                print("[g2pflow] cached model is from an older release; "
                      "re-downloading ...", flush=True)
                _install(MODEL_URL, self._model_path, stale=_is_stale)
        if self._lexicon_is_managed and not self._lexicon_path.is_file():
            _install(LEXICON_URL, self._lexicon_path)

    def _reading(self, kana: str) -> G2PReading:
        kana_words = self._kana._convert(kana)
        alternatives = [
            [path for reading in word.readings for path in reading.paths]
            for word in kana_words
        ]
        paths = []
        seen = set()
        for parts in product(*alternatives):
            path = [group for part in parts for group in part]
            key = tuple((group.script, tuple(group.phonemes)) for group in path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
        return G2PReading(paths=paths)

    def _kana_readings(self, surface: str, pron: str) -> list[G2PReading]:
        candidates = []
        if _kana_only(pron):
            candidates.append(pron)
        candidates.append(surface)
        readings = []
        seen = set()
        for candidate in candidates:
            try:
                reading = self._reading(candidate)
            except KeyError:
                continue
            key = tuple(tuple((group.script, tuple(group.phonemes))
                              for group in path)
                        for path in reading.paths)
            if key not in seen:
                seen.add(key)
                readings.append(reading)
        if not readings:
            raise G2PConversionError([surface])
        return readings[:self._beam_size]

    def __getstate__(self) -> dict:
        # Native ONNX/fugashi handles are rebuilt lazily after unpickling.
        state = dict(self.__dict__)
        state["_runtime"] = None
        state["_segmenter"] = None
        return state

    def _kanji_readings(self, surface: str, candidates: list[dict],
                        kana: str, pron: str) -> list[G2PReading]:
        """Model candidates converted like japanese-mecab.
        A candidate matching UniDic's kana uses UniDic's pron; others are kept
        verbatim.
        """
        readings = []
        seen = set()
        for candidate in candidates:
            reading = candidate["reading"]
            if not _kana_only(reading):
                continue
            if kana and pron and _kana_only(pron) \
                    and hira(reading) == hira(kana):
                reading = pron
            if reading in seen:
                continue
            seen.add(reading)
            try:
                readings.append(self._reading(reading))
            except Exception:
                continue
        if not readings:
            raise G2PConversionError([surface])
        return readings[:self._beam_size]

    def _convert(self, text: str) -> list[G2PWord]:
        if not text:
            return []
        normalized, raw_starts, raw_ends = _normalize_with_offsets(text)
        if not normalized:
            return []
        segments = self._get_segmenter().segment(normalized)
        info = {segment.start: segment for segment in segments}
        result = []
        last_raw_end = 0
        for word in self._get_runtime().predict_words(
                normalized, segments, beam_size=self._beam_size):
            start, end = word["start"], word["end"]
            raw_start = raw_starts[start]
            raw_end = raw_ends[end - 1]
            if raw_start < last_raw_end:
                # Keep normalized text when one source char became several.
                surface_text = normalized[start:end]
            else:
                surface_text = text[raw_start:raw_end]
                last_raw_end = raw_end
            segment = info.get(start)
            surface = word["surface"]
            if all(is_kana(ch) for ch in surface):
                # Kana words take UniDic's contextual pron (は->ワ); the
                # literal reading stays available as a second candidate.
                result.append(G2PWord(
                    text=surface_text,
                    readings=self._kana_readings(
                        surface, segment.pron if segment else "")))
                continue
            result.append(G2PWord(
                text=surface_text,
                readings=self._kanji_readings(
                    surface, word["candidates"],
                    segment.kana if segment else "",
                    segment.pron if segment else "")))
        return result

    def _convert_word(self, text: str) -> G2PWord | None:
        """Convert a caller-fixed word as one unit, like japanese-mecab."""
        if not text:
            return None
        normalized, _starts, _ends = _normalize_with_offsets(text)
        if not normalized:
            return None
        prons = self._get_segmenter().whole_word_readings(normalized)
        readings = []
        for pron in prons:
            try:
                readings.append(self._reading(pron))
            except KeyError:
                continue
        if not readings:
            # Same fallback as japanese-mecab when no whole word exists.
            return compose_word(text, self._convert(normalized))
        return G2PWord(text=text, readings=readings)
