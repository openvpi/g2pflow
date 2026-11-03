from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import filelock
import pytest

from g2pflow.converters import japanese
from g2pflow.converters.japanese import JapaneseKashiConverter
from g2pflow.converters.kashi import resources, runtime


@pytest.fixture
def backend(monkeypatch):
    downloads, loads = [], []
    find_spec = japanese.importlib.util.find_spec
    monkeypatch.setattr(
        japanese.importlib.util, "find_spec",
        lambda name: object() if name == "onnxruntime" else find_spec(name),
    )

    def download(url, path):
        downloads.append(url)
        path.write_bytes(b"model" if url == resources.MODEL_URL else b"lexicon")

    def load(**kwargs):
        loads.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(resources, "_download", download)
    monkeypatch.setattr(resources, "_model_inputs", lambda path: (
        resources._REQUIRED_MODEL_INPUTS
        if path.is_file() and path.read_bytes() == b"model" else None
    ))
    monkeypatch.setattr(runtime, "G2POnnxRuntime", load)
    return SimpleNamespace(downloads=downloads, loads=loads, download=download)


@pytest.mark.parametrize("cached_model", [None, b"incompatible", b"model"])
def test_managed_resources_are_cached(kana_dict, tmp_path, backend, cached_model):
    instance = JapaneseKashiConverter(str(kana_dict), cache_dir=str(tmp_path))
    assert backend.downloads == backend.loads == []
    if cached_model is not None:
        instance._model_path.parent.mkdir(parents=True)
        instance._model_path.write_bytes(cached_model)
        instance._lexicon_path.write_bytes(b"lexicon")
    loaded = instance._get_runtime()
    assert instance._get_runtime() is loaded
    assert len(backend.loads) == 1
    assert instance._model_path.read_bytes() == b"model"
    assert instance._lexicon_path.read_bytes() == b"lexicon"
    assert backend.downloads == (
        [resources.MODEL_URL, resources.LEXICON_URL] if cached_model is None
        else [resources.MODEL_URL] if cached_model != b"model" else []
    )


def test_explicit_resources_are_loaded_without_download(kana_dict, tmp_path, backend):
    model, lexicon = tmp_path / "custom.onnx", tmp_path / "custom.txz"
    model.write_bytes(b"caller model")
    lexicon.write_bytes(b"caller lexicon")
    instance = JapaneseKashiConverter(
        str(kana_dict), model_path=str(model), lexicon_path=str(lexicon), prefer_dml=True,
    )
    loaded = instance._get_runtime()
    assert (loaded.model_path, loaded.bundle_path, loaded.prefer_dml) == (model, lexicon, True)
    assert backend.downloads == []
    assert model.read_bytes() == b"caller model"
    assert lexicon.read_bytes() == b"caller lexicon"


@pytest.mark.parametrize("option", ["model_path", "lexicon_path"])
def test_missing_explicit_resource_raises(kana_dict, tmp_path, backend, option):
    instance = JapaneseKashiConverter(str(kana_dict), **{option: str(tmp_path / "missing")})
    with pytest.raises(FileNotFoundError):
        instance._get_runtime()
    assert backend.downloads == backend.loads == []


def test_missing_runtime_reports_installation_hint(kana_dict, tmp_path, backend, monkeypatch):
    monkeypatch.setattr(japanese.importlib.util, "find_spec", lambda name: None)
    instance = JapaneseKashiConverter(str(kana_dict), cache_dir=str(tmp_path))
    with pytest.raises(ImportError, match=r"g2pflow\[all\]"):
        instance._get_runtime()
    assert backend.downloads == []


def test_failed_transfer_preserves_cached_resource(tmp_path, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"cached model")

    def fail(url, staged):
        staged.write_bytes(b"partial transfer")
        raise OSError("transfer interrupted")

    monkeypatch.setattr(resources, "_download", fail)
    with pytest.raises(OSError, match="transfer interrupted"):
        resources.install(resources.MODEL_URL, target, needed=lambda path: True)
    assert target.read_bytes() == b"cached model"
    assert not target.with_suffix(".onnx.staged").exists()


def test_shared_resource_download_is_serialized(kana_dict, tmp_path, backend, monkeypatch):
    native_lock = filelock.FileLock
    entered, release, second_attempt = Event(), Event(), Event()
    guard = Lock()
    lock_paths = []

    class ObservedLock:
        def __init__(self, path):
            self.path = path
            self.lock = native_lock(path)

        def __enter__(self):
            with guard:
                lock_paths.append(self.path)
                if len(lock_paths) == 2:
                    second_attempt.set()
            return self.lock.__enter__()

        def __exit__(self, *args):
            return self.lock.__exit__(*args)

    def download(url, path):
        backend.download(url, path)
        entered.set()
        assert release.wait(10)

    instances = []
    for name in ("a.onnx", "b.onnx"):
        model = tmp_path / name
        model.write_bytes(b"caller model")
        instances.append(JapaneseKashiConverter(
            str(kana_dict), model_path=str(model), cache_dir=str(tmp_path / "cache"),
        ))
    monkeypatch.setattr(filelock, "FileLock", ObservedLock)
    monkeypatch.setattr(resources, "_download", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(instances[0]._get_runtime)
        try:
            assert entered.wait(10)
            second = pool.submit(instances[1]._get_runtime)
            assert second_attempt.wait(10)
            assert lock_paths[0] == lock_paths[1]
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    assert Counter(backend.downloads) == {resources.LEXICON_URL: 1}
    assert instances[0]._lexicon_path.read_bytes() == b"lexicon"
