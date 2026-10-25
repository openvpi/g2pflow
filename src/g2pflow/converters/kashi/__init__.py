"""Japanese G2P converter backed by the kashi-g2p ONNX model.
UniDic fixes the word boundaries; each word exposes beam-search N-best
readings through the Kana phoneme stage.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from itertools import product
from pathlib import Path

from ...registry import converter
from ..base import Converter, G2PConversionError, G2PReading, G2PWord
from ..japanese import JapaneseKanaConverter, _is_japanese_char
from .notation import fold_notation
from .segmentation import UnidicSegmenter
from ..text import find_run, is_kana

_RELEASE_BASE = ("https://github.com/Xiantaidu/kashi-g2p/releases/"
                 "download/v1.0.0")
MODEL_URL = f"{_RELEASE_BASE}/model.onnx"
LEXICON_URL = f"{_RELEASE_BASE}/lexicon.txz"


def _default_cache_dir() -> Path:
    return Path(os.environ.get("G2PFLOW_CACHE_DIR", "")) \
        if os.environ.get("G2PFLOW_CACHE_DIR") \
        else Path.home() / ".cache" / "g2pflow"


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


@converter(id="japanese-kashi", language="ja,jpn")
class JapaneseKashiConverter(Converter):
    """Japanese G2P readings via the kashi-g2p ONNX model.
    *dict_path*: romaji-to-phoneme dictionary; *model_path* / *lexicon_path* /
    *cache_dir* / *unidic_dir*: resources (auto-download); *beam_size*: N-best per word (16).
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
        missing = [path for path in (self._model_path, self._lexicon_path)
                   if not path.is_file()]
        if missing:
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
                for path, url in ((self._model_path, MODEL_URL),
                                  (self._lexicon_path, LEXICON_URL)):
                    if not path.is_file():
                        print(f"[g2pflow] downloading {url} ...", flush=True)
                        _download(url, path)
        from .runtime import G2POnnxRuntime
        self._runtime = G2POnnxRuntime(
            model_path=self._model_path,
            bundle_path=self._lexicon_path,
            prefer_dml=self._prefer_dml,
        )
        return self._runtime

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
        if pron and all(is_kana(ch) or ch == "ー" for ch in pron):
            candidates.append(pron)
        candidates.append(surface)
        readings = []
        seen = set()
        for candidate in candidates:
            try:
                reading = self._reading(fold_notation(candidate))
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
        return readings

    def _convert(self, text: str) -> list[G2PWord]:
        if not text:
            return []
        segments = self._get_segmenter().segment(text)
        prons = {segment.start: segment.pron for segment in segments}
        result = []
        for word in self._get_runtime().predict_words(
                text, segments, beam_size=self._beam_size):
            surface = word["surface"]
            if all(is_kana(ch) for ch in surface):
                # Kana words take UniDic's contextual pron (は->ワ); the
                # literal reading stays available as a second candidate.
                result.append(G2PWord(
                    text=surface,
                    readings=self._kana_readings(
                        surface, prons.get(word["start"], ""))))
                continue
            readings = []
            seen = set()
            for candidate in word["candidates"]:
                reading = candidate["reading"]
                if not all(is_kana(ch) or ch == "ー" for ch in reading):
                    continue  # drop candidates still containing unresolved kanji
                if reading in seen:
                    continue
                seen.add(reading)
                try:
                    readings.append(self._reading(fold_notation(reading)))
                except Exception:
                    continue  # skip candidates the Kana dictionary cannot voice
            if not readings:
                raise G2PConversionError([surface])
            result.append(G2PWord(text=surface, readings=readings))
        return result
