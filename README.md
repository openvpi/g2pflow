# g2pflow

Composable multilingual grapheme-to-phoneme pipelines for Python 3.10+.

g2pflow preserves alternative readings, complete pronunciation paths and the
pronunciation groups within each path. Converters control word boundaries and
can be combined in priority order for mixed-language text.

## Install

```shell
pip install g2pflow
pip install "g2pflow[ja]"    # MeCab and UniDic
pip install "g2pflow[lstm]"  # ONNX inference for LSTM converters
pip install "g2pflow[all]"   # Both optional backends
```

These are the intended PyPI installation commands. Before publication, install
this checkout with `pip install -e .` (or `pip install -e ".[all,dev]"`).

The base package depends only on Pydantic. NumPy and ONNX Runtime are loaded for
LSTM OOV inference; filelock is loaded only when UniDic needs downloading.
The `lstm` and `ja` extras install these respective optional dependencies.

## Quick start

This example needs no downloaded dictionaries or models:

```python
from g2pflow import G2PPipelineConfig, build_pipeline_from_config

config = G2PPipelineConfig.model_validate({
    "preprocessors": [{"id": "lowercase"}],
    "converters": [{
        "id": "characters",
        "language": "en",
        "kwargs": {"mapping": {"h": ["HH"], "i": ["AY"]}},
    }],
})
pipeline = build_pipeline_from_config(config)
word = pipeline.convert("Hi", languages=["en"])[0]
assert word.text == "hi"
assert word.readings[0].paths[0][0].phonemes == ["HH", "AY"]
```

`G2PPipeline` also accepts preprocessor and converter instances directly.
Configuration is validated by independent Pydantic models; there is no YAML
loader, training configuration system or command-line interface.

## Output

- `G2PWord(text, language, readings)` is a converter-defined semantic word.
- `G2PReading(paths)` groups the complete realizations of one reading.
- `G2PPath` is an ordered `list[G2PGroup]`.
- `G2PGroup(script, phonemes)` associates a pronunciation script with phones.

For example, a reading of Japanese `猫` may contain a path with groups
`("ne", ["n", "e"])` and `("ko", ["k", "o"])`. Different paths can have
different group boundaries. `paths=[]` has no candidates; `paths=[[]]` has one
empty pronunciation. The package does not select candidates against audio or
encode phones into model vocabulary IDs.

## Components and resources

| Converter ID | Purpose | Caller-supplied resources |
| --- | --- | --- |
| `dictionary` | Word-to-phoneme dictionary | `dict_path` |
| `characters` | Character-to-phoneme mapping | `mapping` |
| `passthrough` | Return each token as its own phone | None |
| `chinese-pinyin` | Mandarin text through pinyin | Script-to-phone `dict_path` |
| `yue-jyutping` | Cantonese text through jyutping | Script-to-phone `dict_path` |
| `japanese-kana` | Kana through romaji | Script-to-phone `dict_path` |
| `japanese-mecab` | Japanese word segmentation and readings | Script-to-phone `dict_path`; optional `unidic_dir` |
| `lstm` | Dictionary lookup and ONNX OOV inference | `model_path`; optional `dict_path` |

Preprocessors: `filter-punctuation`, `strip-whitespace`, `lowercase`, and
`remove-accents`.

Mandarin and Cantonese character/phrase reading data are bundled. The caller
chooses the phoneme inventory through external pronunciation dictionaries.
English ONNX models are external. The Japanese backend manages UniDic's
first-use download and can also use a preinstalled directory. See
[resources and converter details](docs/usage.md).

## External plugins

```python
from g2pflow import register_plugin_paths

register_plugin_paths("./my_plugins", "../shared_plugins")
```

Each directory may contain both preprocessors and converters, registered with
`@preprocessor` and `@converter`. Registration scans immediately. Call
`discover_plugins()` to find newly added modules; existing modules load once
per process. Paths are supplied only through Python, not through configuration.
See [plugin discovery and multiprocessing](docs/plugins.md).

## Development

```shell
pip install -e ".[dev]"
pytest --cov=g2pflow --cov-report=term-missing
python -m build
python -m twine check dist/*
```

Default tests run offline with small fixtures and controlled backend doubles.
Real-resource tests and clean-install distribution tests are explicit opt-ins;
see [development and validation](docs/development.md).

## Licensing

Project code uses the [MIT license](LICENSE). Redistributed code and
data retain their respective licenses; the distribution's license expression
also includes Apache-2.0, CC-BY-SA-3.0 and CC-BY-SA-4.0.
See [third-party notices](docs/third-party.md).
