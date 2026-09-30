import os
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.distribution


@pytest.fixture(scope="module")
def artifacts():
    root = Path(__file__).resolve().parent.parent
    folder = Path(os.environ.get("G2PFLOW_DIST_DIR", root / "dist")).resolve()
    wheels = list(folder.glob("g2pflow-*.whl"))
    sources = list(folder.glob("g2pflow-*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Build exactly one wheel and one sdist first"
    return wheels[0], sources[0]


def test_archive_contents(artifacts):
    wheel, source = artifacts
    with zipfile.ZipFile(wheel) as archive:
        files = archive.namelist()
        assert "g2pflow/py.typed" in files
        assert "g2pflow/encoding.py" not in files
        assert not any(name.startswith(("tests/", "lib/", "dictionaries/", "assets/")) for name in files)
        for language in ("mandarin", "cantonese"):
            for resource in ("word.txt", "phrases_dict.txt", "phrases_map.txt", "trans_word.txt", "user_dict.txt", "License.txt"):
                assert f"g2pflow/converters/cpp_pinyin/dicts/{language}/{resource}" in files
        assert any(name.endswith("/licenses/licenses/cpp-pinyin-LICENSE.txt") for name in files)
        assert any(name.endswith("/licenses/licenses/cpp-kana-LICENSE.txt") for name in files)
        metadata = archive.read(next(name for name in files if name.endswith(".dist-info/METADATA"))).decode()
        assert "Requires-Python: >=3.10" in metadata
        assert "Provides-Extra: ja" in metadata
        assert "Provides-Extra: lstm" in metadata
        requirements = [line.removeprefix("Requires-Dist: ")
                        for line in metadata.splitlines() if line.startswith("Requires-Dist: ")]
        assert [requirement for requirement in requirements if ";" not in requirement] == ["pydantic<3,>=2.11"]
        assert any(requirement.startswith("numpy;") and 'extra == "lstm"' in requirement for requirement in requirements)
        assert any(requirement.startswith("filelock;") and 'extra == "ja"' in requirement for requirement in requirements)
        assert "License-Expression: MIT AND Apache-2.0 AND CC-BY-SA-3.0 AND CC-BY-SA-4.0" in metadata
    with tarfile.open(source) as archive:
        files = archive.getnames()
        assert any(name.endswith("/tests/test_discovery.py") for name in files)
        assert any(name.endswith("/docs/plugins.md") for name in files)


def run(*args, cwd, env):
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("artifact_kind", ["wheel", "sdist"])
def test_clean_install_and_resources(artifacts, tmp_path, artifact_kind):
    wheel, source = artifacts
    root = Path(__file__).resolve().parent.parent
    wheelhouse = Path(os.environ.get("G2PFLOW_WHEELHOUSE", root / "wheelhouse")).resolve()
    assert wheelhouse.is_dir(), "Prepare an offline wheelhouse as described in docs/development.md"
    env = dict(os.environ, PYTHONNOUSERSITE="1", PIP_NO_INDEX="1", PIP_FIND_LINKS=str(wheelhouse))
    env.pop("PYTHONPATH", None)
    env.pop("PIP_TARGET", None)
    env.pop("PIP_PREFIX", None)
    if artifact_kind == "sdist":
        rebuilt = tmp_path / "rebuilt"
        run(sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir", "--wheel-dir", str(rebuilt),
            str(source), cwd=tmp_path, env=env)
        wheel = next(rebuilt.glob("g2pflow-*.whl"))
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(str(python), "-m", "pip", "install", str(wheel), cwd=tmp_path, env=env)
    run(str(python), "-m", "pip", "check", cwd=tmp_path, env=env)
    code = '''import importlib.util
import json
import sys
from pathlib import Path
import g2pflow
from g2pflow import G2PPipelineConfig, build_pipeline_from_config, register_plugin_paths
assert Path(g2pflow.__file__).is_relative_to(Path(sys.prefix))
assert len(g2pflow.list_converters()) == 8
assert len(g2pflow.list_preprocessors()) == 4
for name in ('numpy', 'filelock', 'torch', 'lightning', 'onnxruntime', 'fugashi', 'unidic'):
    assert importlib.util.find_spec(name) is None, name
    assert name not in sys.modules, name
dictionary = Path('phones.txt')
dictionary.write_text('ni\\tn i\\nhao\\th ao\\n', encoding='utf-8')
config = G2PPipelineConfig.model_validate({'converters': [{
    'id': 'chinese-pinyin', 'kwargs': {'dict_path': str(dictionary)},
}]})
words = build_pipeline_from_config(config).convert('你好')
assert [word.readings[0].paths[0][0].phonemes for word in words] == [['n', 'i'], ['h', 'ao']]
kana_dict = Path('kana.txt')
kana_dict.write_text('ja\\tj a\\n', encoding='utf-8')
from g2pflow.converters.japanese import JapaneseKanaConverter, JapaneseMecabConverter
assert JapaneseKanaConverter(str(kana_dict)).convert('ぢゃ')[0].readings[0].paths[0][0].phonemes == ['j', 'a']
assert JapaneseMecabConverter(str(kana_dict)).find('猫') == (0, 1)
model_dir = Path('model')
model_dir.mkdir()
(model_dir / 'char.json').write_text(json.dumps({'<unk>': 0, 'a': 1}), encoding='utf-8')
(model_dir / 'phonemes.json').write_text(json.dumps({'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}), encoding='utf-8')
from g2pflow.converters.lstm import LSTMConverter
lstm = LSTMConverter(model_path=str(model_dir), dict_path=str(dictionary))
assert lstm.convert('ni')[0].readings[0].paths[0][0].phonemes == ['n', 'i']
plugins = Path('external plugins')
plugins.mkdir()
(plugins / 'custom.py').write_text(
    "from g2pflow import converter\\n"
    "from g2pflow.converters.simple import PassthroughConverter\\n"
    "@converter(id='installed-plugin')\\n"
    "class Installed(PassthroughConverter):\\n    pass\\n", encoding='utf-8')
register_plugin_paths(plugins)
config = G2PPipelineConfig.model_validate({'converters': [{'id': 'installed-plugin'}]})
assert build_pipeline_from_config(config).convert('installed')[0].text == 'installed'
assert not any(name == 'lib' or name.startswith('lib.') for name in sys.modules)
'''
    run(str(python), "-c", code, cwd=tmp_path, env=env)
