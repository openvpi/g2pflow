import os
import sys
from pathlib import Path

import pytest

import g2pflow
from g2pflow import discovery, registry


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", help="Run real-resource tests")
    parser.addoption("--run-distribution", action="store_true", help="Test built distributions")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, option in (("integration", "--run-integration"),
                               ("distribution", "--run-distribution")):
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"Enable with {option}"))


@pytest.fixture(autouse=True)
def isolated_registration():
    preprocessors = registry._preprocessor_registry.copy()
    converters = registry._converter_registry.copy()
    paths = discovery._plugin_paths[:]
    loaded = discovery._loaded_files.copy()
    modules = set(sys.modules)
    yield
    registry._preprocessor_registry.clear()
    registry._preprocessor_registry.update(preprocessors)
    registry._converter_registry.clear()
    registry._converter_registry.update(converters)
    discovery._plugin_paths[:] = paths
    discovery._loaded_files.clear()
    discovery._loaded_files.update(loaded)
    for name in set(sys.modules) - modules:
        if name == "g2pflow._plugins" or name.startswith("g2pflow._plugins."):
            sys.modules.pop(name, None)
    if "g2pflow._plugins" not in sys.modules and hasattr(g2pflow, "_plugins"):
        delattr(g2pflow, "_plugins")


@pytest.fixture
def dictionary(tmp_path):
    def write(content, name="dictionary.txt"):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path
    return write


@pytest.fixture
def kana_dict(dictionary):
    return dictionary(
        "a\ta\ni\ti\nu\tu\ne\te\no\to\n"
        "ka\tk a\nki\tk i\nko\tk o\nkya\tky a\n"
        "ne\tn e\nto\tt o\ncl\tcl\nn\tN\nga\tg a\n"
        "su\ts u\nzu\tz u\nsa\ts a\nshi\tsh i\n"
        "yo\ty o\nde\td e\n", "kana.txt",
    )


@pytest.fixture
def external_resource():
    def required(name):
        value = os.environ.get(name)
        if not value:
            pytest.fail(f"{name} is required for explicitly enabled resource tests")
        path = Path(value)
        if not path.is_dir():
            pytest.fail(f"{name} is not a directory: {path}")
        return path
    return required
