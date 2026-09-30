import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import g2pflow
from g2pflow import (
    G2PPipelineConfig, build_pipeline_from_config, discover_plugins,
    get_converter, get_preprocessor, register_plugin_paths,
)


def plugin_source(identifier, phone="CUSTOM"):
    return f'''from g2pflow import converter, preprocessor, Preprocessor
from g2pflow.converters.simple import PassthroughConverter

@preprocessor(id="{identifier}")
class Clean(Preprocessor):
    def process(self, fragments):
        return [fragment.strip() for fragment in fragments]

@converter(id="{identifier}", language="en")
class Custom(PassthroughConverter):
    def _convert(self, text):
        words = super()._convert(text)
        for word in words:
            word.readings[0].paths[0][0].phonemes = ["{phone}"]
        return words
'''


def write_plugin(root, identifier, phone="CUSTOM", filename="custom.py"):
    root.mkdir(exist_ok=True, parents=True)
    (root / filename).write_text(plugin_source(identifier, phone), encoding="utf-8")


def test_external_both_kinds_and_no_sys_path_changes(tmp_path):
    write_plugin(tmp_path, "external")
    before = sys.path[:]
    register_plugin_paths(tmp_path)
    assert sys.path == before
    config = G2PPipelineConfig.model_validate({
        "preprocessors": [{"id": "external"}], "converters": [{"id": "external"}],
    })
    word = build_pipeline_from_config(config).convert(" hello ")[0]
    assert (word.text, word.language) == ("hello", "en")
    assert word.readings[0].paths[0][0].phonemes == ["CUSTOM"]


def test_duplicate_path_and_repeat_scans_keep_class_identity(tmp_path):
    write_plugin(tmp_path, "once")
    register_plugin_paths(tmp_path, tmp_path / ".")
    cls = get_converter("once")
    register_plugin_paths(str(tmp_path))
    discover_plugins()
    assert get_converter("once") is cls
    (tmp_path / "custom.py").write_text("raise RuntimeError('must not reload')", encoding="utf-8")
    discover_plugins()
    assert get_converter("once") is cls
    (tmp_path / "custom.py").unlink()
    discover_plugins()
    assert get_converter("once") is cls


def test_new_modules_are_discovered_explicitly(tmp_path):
    register_plugin_paths(tmp_path)
    write_plugin(tmp_path, "added")
    with pytest.raises(KeyError):
        get_converter("added")
    discover_plugins()
    assert get_preprocessor("added").__module__ == get_converter("added").__module__


def test_different_roots_with_same_filenames_are_isolated(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    write_plugin(first, "one", "ONE")
    write_plugin(second, "two", "TWO")
    register_plugin_paths(first, second)
    assert get_converter("one").__module__ != get_converter("two").__module__
    assert get_converter("one")().convert("x")[0].readings[0].paths[0][0].phonemes == ["ONE"]
    assert get_converter("two")().convert("x")[0].readings[0].paths[0][0].phonemes == ["TWO"]


def test_scan_order_is_path_order_then_filename(tmp_path):
    roots = [tmp_path / "zroot", tmp_path / "aroot"]
    for index, root in enumerate(roots):
        write_plugin(root, f"{index}-z", filename="z.py")
        write_plugin(root, f"{index}-a", filename="a.py")
    register_plugin_paths(*roots)
    selected = [name for name in g2pflow.list_converters() if name[0].isdigit()]
    assert selected == ["0-a", "0-z", "1-a", "1-z"]


def test_package_relative_imports_and_no_recursive_import(tmp_path):
    package = tmp_path / "bundle"
    package.mkdir()
    (tmp_path / "helpers.py").write_text("PHONE = 'RELATIVE'\n", encoding="utf-8")
    (package / "__init__.py").write_text("from . import component\n", encoding="utf-8")
    source = plugin_source("package").replace('["CUSTOM"]', '[PHONE]')
    (package / "component.py").write_text("from ..helpers import PHONE\n" + source, encoding="utf-8")
    (package / "unused.py").write_text("raise RuntimeError('not imported')", encoding="utf-8")
    (tmp_path / "__init__.py").write_text("raise RuntimeError('root is not a package')", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not Python", encoding="utf-8")
    register_plugin_paths(tmp_path)
    discover_plugins()
    assert get_converter("package")().convert("x")[0].readings[0].paths[0][0].phonemes == ["RELATIVE"]
    assert not any(name.endswith(".unused") for name in sys.modules)


def test_relative_helper_import_is_not_executed_again_by_scan(tmp_path):
    (tmp_path / "a.py").write_text("from .z import Custom\n", encoding="utf-8")
    write_plugin(tmp_path, "helper", filename="z.py")
    register_plugin_paths(tmp_path)
    assert get_converter("helper") is not None


def test_invalid_paths_and_import_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        register_plugin_paths(tmp_path / "missing")
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        register_plugin_paths(path)
    (tmp_path / "broken.py").write_text("import g2pflow_test_missing_dependency\n", encoding="utf-8")
    with pytest.raises(ModuleNotFoundError, match="g2pflow_test_missing_dependency"):
        register_plugin_paths(tmp_path)


def test_duplicate_component_does_not_override_builtin(tmp_path):
    original = get_converter("passthrough")
    write_plugin(tmp_path, "passthrough")
    with pytest.raises(ValueError, match="already registered"):
        register_plugin_paths(tmp_path)
    assert get_converter("passthrough") is original


def test_duplicate_external_id_errors(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    write_plugin(first, "duplicate", "FIRST")
    write_plugin(second, "duplicate", "SECOND")
    register_plugin_paths(first)
    original = get_converter("duplicate")
    with pytest.raises(ValueError, match="already registered"):
        register_plugin_paths(second)
    assert get_converter("duplicate") is original


def _spawn_worker(paths, serialized, connection):
    try:
        register_plugin_paths(*paths)
        pipeline = pickle.loads(serialized)
        word = pipeline.convert(" hello ")[0]
        config = G2PPipelineConfig.model_validate({"converters": [{"id": "worker"}]})
        rebuilt = build_pipeline_from_config(config).convert("again")[0]
        connection.send((word.text, word.readings[0].paths[0][0].phonemes, rebuilt.text))
    except Exception as exc:
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


def test_spawn_registration_and_pipeline_pickle(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_plugin(first, "other")
    write_plugin(second, "worker", "CHILD")
    roots = [str(first), str(second)]
    register_plugin_paths(*roots)
    config = G2PPipelineConfig.model_validate({
        "preprocessors": [{"id": "worker"}], "converters": [{"id": "worker"}],
    })
    serialized = pickle.dumps(build_pipeline_from_config(config))
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_worker, args=(roots, serialized, child))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "Spawned plugin worker did not finish"
        assert parent.recv() == ("hello", ["CHILD"], "again")
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


def test_builtin_directory_discovery_and_rescan(tmp_path):
    package = tmp_path / "g2pflow"
    shutil.copytree(Path(g2pflow.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    initial = package / "converters" / "new_builtin.py"
    initial.write_text(
        "from ..registry import converter\nfrom .simple import PassthroughConverter\n"
        "converter(id='new-builtin')(type('NewBuiltin', (PassthroughConverter,), {}))\n",
        encoding="utf-8",
    )
    code = '''from pathlib import Path
import g2pflow
assert g2pflow.get_converter('new-builtin')
folder = Path(g2pflow.__file__).parent / 'preprocessors'
(folder / 'added.py').write_text(
    "from ..registry import preprocessor\\nfrom .simple import LowercasePreprocessor\\n"
    "preprocessor(id='added')(type('Added', (LowercasePreprocessor,), {}))\\n",
    encoding='utf-8',
)
g2pflow.discover_plugins()
assert g2pflow.get_preprocessor('added')
g2pflow.discover_plugins()
'''
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True,
                   capture_output=True, text=True, timeout=30)
