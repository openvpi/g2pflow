# Development and validation

Use a separate Python environment compatible with `requires-python` in
[pyproject.toml](../pyproject.toml) and install the checkout:

```shell
pip install -e ".[dev]"
pytest --cov=g2pflow --cov-report=term-missing --cov-report=xml
```

The default suite runs offline with temporary fixtures and deterministic backend
doubles. Real-resource integration tests and distribution tests are explicit
opt-ins.

## Real-resource integration tests

Install `g2pflow[all]` and set these environment variables:

| Variable | Directory contents |
| --- | --- |
| `G2PFLOW_TEST_DICT_DIR` | `ds-zh-pinyin-lite.txt`, `jyutping_dict.txt`, `japanese_dict_full.txt`, `ds_cmudict-07b.txt` |
| `G2PFLOW_TEST_LSTM_MODEL` | LstmG2p English `encoder.onnx`, `decoder.onnx`, `char.json`, `phonemes.json` |
| `G2PFLOW_TEST_KASHI_MODEL` | kashi-g2p `model.onnx` and `lexicon.txz` |
| `G2PFLOW_TEST_UNIDIC_DIR` | Full UniDic dictionary with `sys.dic` and `mecabrc` |

```shell
pytest --run-integration -m integration
```

These explicitly requested tests fail if resources are absent and never
download them during the test.
Regular tests do not depend on these files.

## Build and validate distributions

```shell
python -m build
python -m twine check dist/*
python -m pip wheel --wheel-dir wheelhouse ".[all]" setuptools wheel
pytest --run-distribution -m distribution
```

Preparing `wheelhouse` downloads dependencies and builds wheels for packages
distributed as source archives. Distribution
tests then use `--no-index` behavior to install into fresh environments and
rebuild the wheel from the sdist. They verify dependencies, packaged resources,
licensing files, standalone operation and external plugins outside the checkout.
The fresh environments do not inherit the developer environment's packages.

Defaults are `dist` and `wheelhouse` in the repository root. Set
`G2PFLOW_DIST_DIR` or `G2PFLOW_WHEELHOUSE` to use other locations. Keep one wheel
and one sdist version in the artifact directory for this validation.

## CI

The [test workflow](../.github/workflows/tests.yml) defines the platform and
Python-version matrix, dependency installation, checks and uploaded artifacts.
Tests run without downloading dictionaries or models.

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
   `project.version` from `pyproject.toml` exactly, prefixed with `v`, on the
   intended commit.
3. Publish the GitHub Release. A saved draft does not publish to PyPI.
4. The `Publish to PyPI` workflow validates the tag, runs the complete test and
   build workflow on that tagged commit, then uploads its wheel and sdist.
5. Check the completed workflow and https://pypi.org/project/g2pflow/.

Both published releases and published prereleases trigger this workflow. For
prereleases, include the prerelease suffix from `project.version` in the tag.
Uploading uses the artifacts from the successful checks in that same run and
does not rebuild them in the publishing job. Only the publishing job has the
OIDC permission. PyPI does not allow replacing an already uploaded file; publish
corrections under a new version.
