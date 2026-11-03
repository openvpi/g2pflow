"""Download and cache the kashi-g2p model and lexicon."""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

RELEASE_TAG = "v1.1.0"
_RELEASE_BASE = f"https://github.com/Xiantaidu/kashi-g2p/releases/download/{RELEASE_TAG}"
MODEL_URL = f"{_RELEASE_BASE}/model.onnx"
LEXICON_URL = f"{_RELEASE_BASE}/lexicon.txz"


def default_cache_dir() -> Path:
    configured = os.environ.get("G2PFLOW_CACHE_DIR")
    return Path(configured) if configured else Path.home() / ".cache" / "g2pflow"


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


def model_needs_download(path: Path) -> bool:
    """Whether a managed model is missing or has an incompatible input signature."""
    return not path.is_file() or not _model_is_current(path)


def install(url: str, path: Path, *, needed) -> None:
    """Download *url* into *path* once, serialized per target file."""
    if not needed(path):
        return
    from filelock import FileLock

    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix(path.suffix + ".download.lock"))):
        # Another instance may have finished the same download while waiting.
        if not needed(path):
            return
        print(f"[g2pflow] downloading {url} ...", flush=True)
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
        if path.is_file() and stamp.read_text(encoding="utf-8").strip() == expected:
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
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                print(f"[g2pflow] download failed ({exc}); retrying "
                      f"({attempt}/{attempts}) ...", flush=True)
    raise IOError(f"could not download {url}") from last_error
