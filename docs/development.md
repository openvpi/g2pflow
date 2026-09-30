# Development and validation

Use a separate Python 3.10+ environment and install the checkout:

```shell
pip install -e ".[dev]"
pytest --cov=g2pflow --cov-report=term-missing --cov-report=xml
```

The default suite is offline. Small handwritten dictionaries and deterministic
backend doubles exercise routing, configuration, plugin discovery, candidate
structure, MeCab/UniDic lifecycle and LSTM beam behavior. Plugin tests include a
real `spawn` process. Tests write their fixtures into pytest's temporary folders.
The `dev` extra includes NumPy to exercise the LSTM numerical code. Distribution
tests separately verify installations with only the base Pydantic dependency.
Python comments and docstrings use ASCII; text fixtures intentionally include
non-ASCII input.

## Real-resource integration tests

Install `g2pflow[all]` and set these environment variables:

| Variable | Directory contents |
| --- | --- |
| `G2PFLOW_TEST_DICT_DIR` | `ds-zh-pinyin-lite.txt`, `jyutping_dict.txt`, `japanese_dict_full.txt`, `ds_cmudict-07b.txt` |
| `G2PFLOW_TEST_LSTM_MODEL` | LstmG2p English `encoder.onnx`, `decoder.onnx`, `char.json`, `phonemes.json` |
| `G2PFLOW_TEST_UNIDIC_DIR` | Full UniDic dictionary with `sys.dic` and `mecabrc` |

```shell
pytest --run-integration -m integration
```

These explicitly requested tests fail if resources are absent, and never
download them during the test. They check the reference pronunciation mappings
and LstmG2p model described above, plus serialization after native initialization.
Regular tests do not depend on these files.

## Build and validate distributions

```shell
python -m build
python -m twine check dist/*
python -m pip download --only-binary=:all: --dest wheelhouse dist/g2pflow-0.1.0-py3-none-any.whl "setuptools>=77.0.3" wheel
pytest --run-distribution -m distribution
```

Preparing `wheelhouse` is the network-enabled preparation step. Distribution
tests then use `--no-index` behavior to install into fresh environments and
rebuild the wheel from the sdist. They verify dependencies, packaged resources,
licensing files, standalone operation and external plugins outside the checkout.
The fresh environments do not inherit the developer environment's packages.

Defaults are `dist` and `wheelhouse` in the repository root. Set
`G2PFLOW_DIST_DIR` or `G2PFLOW_WHEELHOUSE` to use other locations. Keep one wheel
and one sdist version in the artifact directory for this validation.

## CI

GitHub Actions runs the offline suite on Ubuntu, Windows and macOS with Python
3.10 and 3.12. A separate job installs all optional backends and runs the suite
without downloading models or UniDic. A distribution job builds wheel/sdist,
checks metadata, prepares dependency wheels and exercises clean installations.
Coverage reports and distribution archives are uploaded as workflow artifacts.
There is no publication workflow.
