# Pipelines and resources

## Configuration and construction

`PreprocessorConfig` contains `id` and `kwargs={}`. `ConverterConfig` adds
`language=None`. `G2PPipelineConfig` contains `preprocessors=[]` and
`converters=[]`. They inherit directly from Pydantic `BaseModel`; validation,
serialization and default handling follow Pydantic 2.

```python
from g2pflow import G2PPipelineConfig, build_pipeline_from_config

config = G2PPipelineConfig.model_validate({
    "preprocessors": [{"id": "filter-punctuation"}, {"id": "lowercase"}],
    "converters": [{
        "id": "dictionary",
        "language": "en",
        "kwargs": {"dict_path": "@english.txt"},
    }],
})
pipeline = build_pipeline_from_config(config, root_path="resources")
words = pipeline.convert("Hello world", languages=["en"])
```

The corresponding single-component factories are
`build_preprocessor_from_config()` and `build_converter_from_config()`.
All three factories accept `root_path`. Every string beginning with `@` inside
kwargs (including nested lists and dictionaries) is resolved against that root.
Ordinary relative paths remain relative to the current working directory.
The input configuration is not mutated.

YAML users can pass their loader's resulting dictionary to `model_validate()`.
Use the pipeline mapping directly; no `binarizer.g2p` wrapper, inheritance or
scope parameter is part of g2pflow.

## Routing and languages

For per-range language scopes, fixed word boundaries, direct multi-candidate
pronunciations and lossless serialization, see [PFML 1.0](pfml.md).
`pipeline.convert_pfml(source, language="zh")` uses the same converters with
explicit language routing. Scoped converters must declare that language;
configure `language` on neutral dictionary/LSTM converters before using them
inside a language scope.

Global preprocessors transform `[text]` into fragments. Within each fragment,
the earliest range claimed by a higher-priority converter is reserved for that
converter. Its left remainder goes to lower priorities; its right remainder
continues at the same priority. Unclaimed non-whitespace text is collected in
text order and raises `G2PConversionError` before conversion begins.

Each converter applies its own preprocessors to its claimed runs and can emit
zero or more words. Conversion errors propagate; they do not trigger fallback
to a lower-priority converter. The pipeline preserves the resulting word text,
candidate ordering and grouping. The converter base class filters silent groups
and paths, omits parents left empty, and fills missing display labels. A result
with no readings or a reading with no candidate paths raises `G2PConversionError`.
The same rules apply to direct converter calls and PFML results.

`languages=[...]` filters active converters. A converter's language is a tuple
of aliases, `None`, or `Language.ANY`. The first matching registered alias is
written onto each output word (the first registered alias if no filter is
provided). `None` is language-neutral. `Language.ANY`, used by `characters` and
`passthrough`, leaves language selection to downstream consumers. A non-null
configuration `language` overrides the registered language on that instance.

Defaults: Mandarin `zh,zho,cmn`; Cantonese `yue`; Japanese `ja,jpn`;
dictionary and LSTM converters are language-neutral. Specify `languages=["ja"]`
when Chinese and Japanese converters coexist and Japanese should claim kanji.
No active converter raises `ValueError`, including an empty converter list.

## Pronunciation dictionaries

Dictionary files are UTF-8, with a key, a tab, and space-separated phones:

```text
hello	hh ax l ow
hello(1)	hh eh l ow
world	w er l d
```

`DictionaryConverter` is case-insensitive and treats trailing `(N)` or ` (N)`
as pronunciation variants. Duplicate entries preserve their order. The shared
`load_pronunciation_dict()` accumulates duplicate keys without performing that
suffix normalization. Entries without phones are skipped.

Mandarin, Cantonese and Kana converters take a required `dict_path` mapping
pinyin, jyutping or romaji to the desired phonemes. Those mappings are not
bundled: the calling application controls its phoneme inventory. Unsupported
script keys raise `KeyError`. Mandarin/Cantonese prefer phrase-context readings,
then preserve other supported readings in order.

Kana conversion keeps digraphs together, normalizes katakana to hiragana, and
maps through the romaji table. Long-vowel marks and the standalone handakuten
produce no standalone phonemes and are omitted from output.
`double_written_sokuon=False` looks up a sokuon as `cl` separately from the
following kana. With `True`, the sokuon and following consonant-leading kana
form one pronunciation unit and use a combined dictionary key: for example,
`った` uses `tta`, and `っきゃ` uses `kkya`. Skipped long-vowel marks and standalone
handakuten do not interrupt that lookup. A trailing sokuon or one before a vowel
continues to use `cl`.

The option changes only pronunciation-script spelling and grouping. Every
phoneme sequence, including all alternatives, comes unchanged from the dictionary
entry; no consonant phoneme is inserted or doubled. For example, `tta\tcl t a`
produces `script="tta"` and `phonemes=["cl", "t", "a"]`. A missing combined key
raises `KeyError` instead of falling back to separate keys or guessed phonemes.

## Japanese MeCab and UniDic

Install `g2pflow[ja]` for fugashi, UniDic and filelock. `japanese-mecab` accepts `dict_path`, `nbest=32`,
`double_written_sokuon=False` and `unidic_dir=None`. MeCab supplies word surfaces
and UniDic's `pron` readings; g2pflow converts each complete reading into grouped
pronunciation paths using the same Kana implementation.

Importing g2pflow or constructing this converter does not load MeCab or download
UniDic. Filelock is imported only when a default dictionary download is needed;
an existing dictionary or explicit directory does not require it at runtime.
The first conversion initializes the native tagger:

1. With `unidic_dir`, load that directory without downloading a default dictionary.
2. Otherwise use `unidic.DICDIR`, checking for `sys.dic` and `mecabrc`.
3. If either is missing, acquire the parent directory's `download.lock`, check
   again, then call UniDic's downloader. This serializes spawned workers.

The default download needs network access and a writable UniDic installation
directory. To prepare offline use ahead of time:

```shell
python -m unidic download
```

Alternatively provide an existing full UniDic directory through `unidic_dir`.
Download and initialization errors propagate. No small dictionary is substituted
automatically. Missing backend packages produce an installation hint for
`g2pflow[ja]`.

N-best candidates must cover the whole word as one node. Empty, placeholder and
duplicate readings are discarded. Pure Kana without readings falls back to the
Kana converter; unreadable non-Kana words raise `G2PConversionError`.

## LSTM ONNX inference

Install `g2pflow[lstm]` for NumPy and ONNX Runtime. The converter accepts `model_path`, `dict_path=None`,
and `beam_size=16`. Its model directory must contain `encoder.onnx`,
`decoder.onnx`, `char.json` and `phonemes.json`, in the
[LstmG2p](https://github.com/wolfgitpr/LstmG2p) format. No model is bundled or
automatically downloaded.

Vocabulary JSON is read during construction. NumPy and ONNX Runtime are loaded
only on the first OOV inference, when native sessions are created. Dictionary
hits work without either optional package. Beam
results are ordered by mean log probability including EOS in the length, with
special tokens removed and duplicate phone sequences collapsed in rank order.
Use `beam_size=1` for greedy decoding. Native sessions and MeCab taggers are
excluded when pickling and recreated when needed in the receiving process.
