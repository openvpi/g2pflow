"""Japanese G2P converter backed by the kashi-g2p ONNX model.

UniDic fixes the word boundaries; the model supplies beam-search N-best
readings, and the shared Japanese reading flow turns them into phonemes.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import urllib.request
from pathlib import Path

from .._kana import KanaReading, KanaSegment
from ..japanese import JapaneseReadingConverter
from .segmentation import UnidicSegmenter

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
    except ImportError:
        return None
    try:
        session = ort.InferenceSession(str(path),
                                       providers=["CPUExecutionProvider"])
    except Exception:
        return None
    return frozenset(item.name for item in session.get_inputs())


def _is_stale(path: Path) -> bool:
    """Whether an existing model file predates the current input signature."""
    return path.is_file() and not _model_is_current(path)


def _install(url: str, path: Path, *, stale=None) -> None:
    """Download *url* and move it into place once the transfer succeeded."""
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


class JapaneseKashiConverter(JapaneseReadingConverter):
    """Japanese G2P readings via the kashi-g2p ONNX model.
    *dict_path*: romaji-to-phoneme dictionary; *model_path* / *lexicon_path* /
    *cache_dir* / *unidic_dir*: auto-download resources; *beam_size* caps
    candidates per word.
    """

    def __init__(self, dict_path: str, *,
                 model_path: str | None = None,
                 lexicon_path: str | None = None,
                 cache_dir: str | None = None,
                 prefer_dml: bool = False,
                 beam_size: int = 16,
                 unidic_dir: str | None = None) -> None:
        self._beam_size = max(1, int(beam_size))
        # Kana words keep their contextual UniDic pronunciation.
        super().__init__(dict_path, max_readings=self._beam_size,
                         include_surface=True)
        cache = Path(cache_dir) if cache_dir else _default_cache_dir() / "kashi"
        cache = cache / _RELEASE_TAG
        # Only files this package placed in its own cache may be replaced.
        self._model_is_managed = model_path is None
        self._lexicon_is_managed = lexicon_path is None
        self._model_path = Path(model_path) if model_path else cache / "model.onnx"
        self._lexicon_path = Path(lexicon_path) if lexicon_path else cache / "lexicon.txz"
        self._prefer_dml = prefer_dml
        self._unidic_dir = unidic_dir
        self._runtime = None
        self._segmenter = None

    def _get_segmenter(self):
        if self._segmenter is None:
            self._segmenter = UnidicSegmenter(self._unidic_dir)
        return self._segmenter

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        # Fail before fetching ~141 MB the runtime could never load.
        if importlib.util.find_spec("onnxruntime") is None:
            raise ImportError(
                "japanese-kashi requires optional dependencies. Install them "
                'with: python -m pip install "g2pflow[kashi]"')
        # Caller-supplied files are never touched.
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

    def _segments(self, text: str) -> list[KanaSegment]:
        return self._get_segmenter().segment(text)

    def _readings(self, text: str,
                  segments: list[KanaSegment]) -> list[list[KanaReading]]:
        return [
            [KanaReading(candidate["reading"])
             for candidate in word["candidates"]]
            for word in self._get_runtime().predict_words(
                text, segments, beam_size=self._beam_size)
        ]

    def _word_readings(self, text: str) -> list[KanaReading]:
        return [KanaReading(pron)
                for pron in self._get_segmenter().whole_word_readings(text)]

    def __getstate__(self) -> dict:
        # Native ONNX/fugashi handles are rebuilt lazily after unpickling.
        state = dict(self.__dict__)
        state["_runtime"] = None
        state["_segmenter"] = None
        return state
