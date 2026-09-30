# Plugin discovery

Built-in `preprocessors` and `converters` directories are scanned on import.
Adding a Python component module does not require editing a component list.
External plugins use the same decorators and registries.

## Write and register a plugin

Place this in `my_plugins/custom.py`:

```python
from g2pflow import (
    Converter, G2PGroup, G2PReading, G2PWord, Preprocessor,
    converter, preprocessor,
)

@preprocessor(id="custom-strip")
class CustomStrip(Preprocessor):
    def process(self, fragments):
        return [fragment.strip() for fragment in fragments]

@converter(id="custom-word", language="en")
class CustomWord(Converter):
    def find(self, text):
        return (0, len(text)) if text else None

    def _convert(self, text):
        return [G2PWord(text=text, readings=[G2PReading(paths=[[
            G2PGroup(script=text, phonemes=["CUSTOM"]),
        ]])])]
```

Then register the directory before constructing the pipeline:

```python
from g2pflow import (
    G2PPipelineConfig, build_pipeline_from_config, register_plugin_paths,
)

register_plugin_paths("my_plugins")
config = G2PPipelineConfig.model_validate({
    "preprocessors": [{"id": "custom-strip"}],
    "converters": [{"id": "custom-word"}],
})
pipeline = build_pipeline_from_config(config)
assert pipeline.convert("hello")[0].readings[0].paths[0][0].phonemes == ["CUSTOM"]
```

Converter `find()` returns the earliest accepted, non-empty half-open slice
using Python string indices. It should not perform pronunciation inference.
`_convert()` receives a claimed run and can emit, merge, split or omit words.
`preprocessors()` optionally returns local preprocessors for claimed runs.

The base class owns the public `convert()` and `convert_word()` methods. They
normalize results automatically, including direct calls outside a pipeline:
zero-phone groups and empty paths are removed, parents left empty are omitted,
and missing display labels are filled from surviving pronunciations. A word
with no readings, a reading with no paths, or an empty phoneme symbol raises
`G2PConversionError`. Implement the backend hooks instead of overriding these
public methods; no manual normalization call is needed.

PFML `<word>` inputs additionally use `accepts_word(text)` and
`_convert_word(text) -> G2PWord | None`. By default, the former requires `find()`
to cover the whole input and the latter accepts zero or one raw result from
`_convert()`. Override these hooks for custom fixed-unit handling such as phrase
lookup. A silent unit is omitted; splitting a fixed word raises
`G2PWordBoundaryError`. The pipeline does not silently merge plugin outputs.
When extending another backend hook, call its corresponding protected method
(for example, `super()._convert(text)`). See the
[PFML converter contract](pfml.md#python-api-and-converter-contract).

Advanced converters can subclass `LexiconConverter` or
`PronunciationScriptConverter` from `g2pflow.converters.paradigm`, or
`PronunciationScriptDictionaryConverter` from `g2pflow.converters.dictionary`.

## Scanning rules

- `register_plugin_paths(*paths)` accepts directory strings or `Path` objects,
  resolves them against the current working directory, saves them in order and
  immediately scans. `~` is expanded; `@` has no special meaning here.
- Paths are process-local and never appear in `G2PPipelineConfig`.
- Each directory is scanned in name order for direct `.py` files (except
  `__init__.py`) and child packages with `__init__.py`. Packages explicitly
  import their own nested component modules. Other files are ignored.
- Both component kinds can live together. Same-kind duplicate IDs raise
  `ValueError`, including collisions with built-ins. There is no override mode.
- Different roots get separate private module namespaces, so `custom.py` in
  two roots can coexist. Use relative imports for neighboring helpers, such as
  `from .helpers import normalize`. Global `sys.path` is not changed.
- Re-registering a path or rescanning does not re-execute loaded modules.
  `discover_plugins()` discovers newly added modules in built-in and external
  directories. Factories use the registry as it stands; explicitly rescan after
  adding files in a running process.
- Changes to loaded modules, removals and component replacement require a new
  process. There is no watcher, unload API or code reload.
- Invalid paths and import errors propagate. Imports retain ordinary Python
  side effects; discovery is not a transaction or a sandbox. After a failing
  plugin initialization, fix the plugin and restart the process.

## Spawned processes

Every spawned process must register the same roots in the same order before
using external component IDs or unpickling instances of their classes. Module
names are assigned by root registration order. Prefer passing root paths and
configuration dictionaries to workers and constructing pipelines there.

For a process pool, register paths in its initializer before submitting pickled
pipeline tasks. For a directly spawned process, pass configuration to the child
instead of passing a plugin instance among its startup arguments: startup
arguments are unpickled before the target function runs. Standard Python
pickling still requires custom component state itself to be picklable.
