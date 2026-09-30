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

PFML conformance and round-trip checks run in the default offline suite. They
cover language scopes, fixed converter boundaries, lower-level shorthand,
complete multi-candidate trees, malformed/incomplete markup, silent-branch
filtering, and lossless serialization of successful outputs and language markers.
They also cover normalization through the framework's public converter entry
points. Randomized tree fixtures use a fixed seed and require no extra testing
dependency.

## Build and validate distributions

```shell
python -m build
python -m twine check dist/*
python -m pip download --only-binary=:all: --dest wheelhouse dist/g2pflow-0.3.1-py3-none-any.whl "setuptools>=77.0.3" wheel
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

GitHub Actions runs the offline suite on Ubuntu with Python 3.10 through 3.14,
and on Windows and macOS with Python 3.10 and 3.14. A separate Python 3.12 job
installs all optional backends and runs the suite without downloading models or
UniDic. A distribution job builds wheel/sdist, checks metadata, prepares dependency
wheels and exercises clean installations. Coverage reports and distribution
archives are uploaded as workflow artifacts.

The test workflow runs for branch pushes, pull requests and manual dispatches.
It is also reusable by the release workflow, which tests the release's tagged
commit and publishes the resulting verified distribution artifacts.

## PyPI publishing setup

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
with GitHub Actions OIDC. The GitHub environment is named `pypi` and permits
deployment from `v*` tags. No PyPI API token is stored in the repository.

For a new PyPI project, sign into the account that will own the package and add
a pending publisher at https://pypi.org/manage/account/publishing/:

| Field | Value |
| --- | --- |
| PyPI project name | `g2pflow` |
| GitHub owner | `openvpi` |
| Repository name | `g2pflow` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

Enter the workflow filename only, without `.github/workflows/`. The first
successful publication creates the PyPI project and converts the pending
publisher into a normal publisher. For an existing project, configure the same
publisher under that project's publishing settings.

## Publish a version

1. Update `project.version` in `pyproject.toml`, commit and push the change.
2. Open https://github.com/openvpi/g2pflow/releases/new and create a tag matching
   the package version exactly, prefixed with `v`, on the intended commit.
   For version `0.3.1`, use the tag `v0.3.1`.
3. Publish the GitHub Release. A saved draft does not publish to PyPI.
4. The `Publish to PyPI` workflow validates the tag, runs the complete test and
   build workflow on that tagged commit, then uploads its wheel and sdist.
5. Check the completed workflow and https://pypi.org/project/g2pflow/.

Both published releases and published prereleases trigger this workflow; use a
matching package version such as `0.3.1rc1` and tag `v0.3.1rc1` for prereleases.
Uploading uses the artifacts from the successful checks in that same run and
does not rebuild them in the publishing job. Only the publishing job has the
OIDC permission. PyPI does not allow replacing an already uploaded file; publish
corrections under a new version.
