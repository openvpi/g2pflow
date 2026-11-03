# g2pflow

Composable multilingual grapheme-to-phoneme pipelines for Python 3.10+.

g2pflow preserves alternative readings, complete pronunciation paths and the
pronunciation groups within each path. Converters control word boundaries and
can be combined in priority order for mixed-language text.

## Install

```shell
pip install g2pflow
pip install "g2pflow[all]"  # Include optional runtime dependencies
```

From a checkout, install with `pip install -e .` or `pip install -e ".[all,dev]"`.

The UniDic dictionary is loaded on first conversion and downloaded if missing.
Neural models are loaded when inference is needed.

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
Configuration models validate Python dictionaries. See
[configuration and construction](docs/usage.md#configuration-and-construction).

## PFML markup

PFML (Pronunciation Flow Markup Language) adds language scopes, fixed word
boundaries and complete pronunciation candidate trees to ordinary text:

```python
from g2pflow import G2PPipeline, to_pfml

# Direct phonemes need no dictionaries, models or converters.
words = G2PPipeline().convert_pfml(
    '<scope language="ja"><word script="ka" phonemes="k a">か</word></scope>'
)
assert words[0].readings[0].paths[0][0].phonemes == ["k", "a"]
assert G2PPipeline().convert_pfml(to_pfml(words)) == words
```

`<scope language="ja">東京へ行く</scope>` routes a whole text range to Japanese
G2P; `<word language="zh">重庆</word>` fixes one output word. Automatic text
requires a pipeline configured for the corresponding language. Direct results
must provide all final phonemes; they preserve multiple readings, paths and
group boundaries without invoking G2P or filtering candidates. Missing outer
containers can be omitted, down to `<phoneme>ong</phoneme>`.

There is no required root tag. `convert()` still handles plain text, while
`convert_pfml()` handles XML fragments. Python outputs remain `G2PWord` objects;
`to_pfml()` serializes successful outputs losslessly, including all pronunciation
candidates and language markers. See the [PFML 1.0 specification and API](docs/pfml.md).

## Output

- `G2PWord(text, language, readings)` is a converter-defined semantic word.
- `G2PReading(paths)` groups the complete realizations of one reading.
- `G2PPath` is an ordered `list[G2PGroup]`.
- `G2PGroup(script, phonemes)` associates a pronunciation script with phones.

For example, a reading of Japanese `猫` may contain a path with groups
`("ne", ["n", "e"])` and `("ko", ["k", "o"])`. Different paths can have
different group boundaries. The framework filters silent groups and paths,
omits words left without pronunciation, and fills missing display labels.
Missing readings or path candidates are conversion failures. The package does
not select candidates against audio or encode phones into model vocabulary IDs.

## Components and resources

Converters determine word boundaries and readings. Pronunciation dictionaries
let the caller choose the phoneme inventory. See the
[component and resource reference](docs/usage.md#components-and-resources)
for converter IDs, preprocessors, resource requirements and download options.

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
