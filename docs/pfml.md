# PFML 1.0

PFML (Pronunciation Flow Markup Language) is an XML-based input and interchange
format for g2pflow. It specifies language scopes, fixed word boundaries, and
complete pronunciation candidate trees. The suggested extension is `.pfml`.
This document defines version 1.0; the parser and `tests/test_pfml.py` implement
its content model and conformance examples.

See the [tag reference](#tag-reference) for each element's attributes, allowed
contents, placement and examples.

## Conversion model

Each word is either **automatic** (source text goes through G2P) or **direct**
(all final phonemes are supplied and no G2P is performed). There is no candidate
filtering, partial phoneme inference, or mixing of automatic and direct
candidates within one word. A script without its final phonemes is an error.

Both forms use the same output model:

```text
word
  text                       source text or a filled display label
  language                   application-defined ID, Language.ANY, or None
  readings[]                 alternative readings
    paths[]                  complete alternative realizations of one reading
      groups[]               ordered pronunciation groups within one path
        script               pronunciation-script label
        phonemes[]           ordered, atomic phoneme symbols
```

`G2PWord.text` and `G2PWord.language` correspond to the `text` and `language`
attributes on `word`. Missing source labels are filled for
direct results, while an omitted language inherits the surrounding scope.
Simple automatic words can also supply their text as character
data. [Serialization](#serialization) preserves both fields and the complete
candidate tree of successful, normalized outputs, including the distinct `None`
and `Language.ANY` language values.

Paths can have different group counts and boundaries. Explicit readings, paths,
groups and phonemes keep their order and duplicates. Groups from different
paths are never recombined by the PFML parser.

Every output word must reach the phoneme level. The framework removes groups
with zero phonemes, then empty paths, readings and words. Silent units and
punctuation removed by preprocessing leave no output placeholders; callers that
need the complete source retain it themselves. A backend result with no readings
or a reading with no candidate paths is a conversion failure, not a successful
silent result. Phoneme symbols must be non-empty strings.

Python conversion returns `list[G2PWord]`, whose dataclasses represent the
complete candidate tree. PFML is the external representation. `PFMLDocument`
is a normalized input plan containing `PFMLText` requests and direct `G2PWord`
results, with language scopes and omitted containers resolved.

## Fragments and language scopes

PFML is an XML 1.0 **fragment**, with text and elements allowed as siblings
without an enclosing root element. XML declarations,
document types, processing instructions and XML namespaces are not part of
PFML 1.0. Comments are ignored and CDATA is ordinary character data. Unknown
elements, attributes and invalid nesting are errors.

```xml
今天说<scope language="en">hello world</scope>，然后读<scope language="ja">東京へ行く</scope>。
```

`scope` sets a language scope without fixing word boundaries. It can contain
ordinary text, nested scopes, words and direct pronunciation fragments. `word`
may override the inherited language. Resolution follows the nearest explicit
declaration; use an outer `scope` to set the default language for the fragment.
An empty `language=""` clears inherited language information. With no effective
language, the normal pipeline routing policy and optional `languages` filter
apply.

```xml
<scope language="zh">我住在<word>重庆</word>，喜欢<scope language="en">New York</scope>。</scope>
```

Language IDs are opaque strings defined by the application. PFML imposes no
language-code standard or format requirement; `zh`, `yue`, `ja` and `en` are
examples, not a prescribed vocabulary. IDs are preserved without trimming,
case folding, canonicalization or dialect fallback. Explicit scopes select
only converters whose registered/configured language tuple contains that exact
ID. Language-neutral and `Language.ANY` converters must
be explicitly bound to a language in configuration to participate in a scope.
Unsupported language scopes fail instead of falling through to another language.

The pipeline's `languages` argument filters automatic text with **no effective
language declaration**, including text whose inherited language was cleared.
An explicit PFML language declaration takes precedence over that filter;
`languages` is not a whitelist for the whole document. Direct results need no
converter for their declared language, and the filter does not change their
language metadata.

Literal `<` and `&` must be escaped as `&lt;` and `&amp;`, or placed in CDATA.
`pipeline.convert()` remains a plain-text entry point, so existing text with XML
looking content is never silently interpreted as PFML. `convert_pfml()` always
parses PFML, including text-only fragments.

## Tag reference

PFML 1.0 supports the following six elements. Authoring and serialization share
one content model.

| Element | Purpose | Attributes |
| --- | --- | --- |
| [`scope`](#scope) | A language scope around mixed text and markup | `language` |
| [`word`](#word) | One fixed source word or complete result | `text`, `language`, `script`, `phonemes`, `language-kind` |
| [`reading`](#reading) | One reading containing complete path candidates | None |
| [`path`](#path) | One complete, ordered pronunciation path | None |
| [`group`](#group) | A pronunciation-script label and its phonemes | `script`, `phonemes` |
| [`phoneme`](#phoneme) | One final phoneme symbol | `symbol`, `language` |

Element and attribute names are case-sensitive. Only the listed attributes
are accepted on each element; attributes do not transfer between element
types. For example, `group` has no `language` attribute. Language IDs themselves
are application-defined strings, as described under
[language scopes](#fragments-and-language-scopes).

Within a pronunciation container, direct children must all be at the same
supported level. Whitespace between structural children is formatting.
The [omitted-container rules](#omitted-containers-and-label-filling) define how
missing levels are filled and how standalone candidate containers are handled.
Phoneme symbols in the examples are illustrative; PFML does not check them
against a dictionary or model inventory.

### scope

**Purpose:** set a language scope without fixing word boundaries.

**Placement:** at fragment level or inside another `scope`.

**Contents:** ordinary text and any of the six supported elements, in order.
An empty scope is allowed and produces no words.

| Attribute | Required | Meaning and default |
| --- | --- | --- |
| `language` | No | Language ID for automatic text and words in this scope. Omission inherits the surrounding scope; `language=""` clears it. Inner declarations override it. |

```xml
<scope language="zh">你好，<scope language="en">hello world</scope>。</scope>
```

Both text ranges are automatically segmented by the corresponding G2P backend.
The nested scope does not turn `hello world` into one fixed word. Contiguous
ordinary text with the same effective language is coalesced, including across
equivalent scopes. A scope cannot appear inside a word or pronunciation container.

### word

**Purpose:** represent one `G2PWord`, either automatically converted from source
text or supplied as a complete pronunciation result.

**Placement:** at fragment level or inside a `scope`. Adjacent words are separate
output words.

**Contents:** source character data when there are no child elements, or
pronunciation children at one level: `reading`, `path`, `group` or `phoneme`.
When there are child elements, source text can only be supplied with the `text`
attribute.

| Attribute | Required | Meaning and default |
| --- | --- | --- |
| `text` | Conditional | Maps to `G2PWord.text`. An automatic word needs non-whitespace source text, either here or as character data. For direct results, an omitted or empty value is filled from surviving pronunciation labels. |
| `language` | No | Maps to `G2PWord.language`. Omission inherits the surrounding scope; an empty value clears it to `None`. |
| `script` | No | Compact single-group script label. Requires the `phonemes` attribute. An omitted or empty label is filled from that group's phonemes. |
| `phonemes` | No | Compact, whitespace-separated final phoneme names. Its presence makes the word direct and creates one reading, one path and one group. `phonemes=""` supplies silence and the word is omitted from results. Cannot accompany pronunciation children. |
| `language-kind` | No; direct results only | The only value is `any`, encoding `Language.ANY` independently of surrounding language context. Cannot accompany a `language` attribute. |

An automatic word with both model attributes:

```xml
<word text="你" language="zh"/>
```

The character-data form, `<word language="zh">你</word>`, is also allowed.
Providing both a `text` attribute and non-whitespace source character data is
an error. An automatic word fixes one output word, including internal spaces,
and preserves its source in `G2PWord.text` when a pronunciation remains.
Preprocessors receive a copy that they may rewrite or remove, but must not split.

A direct word with structured pronunciation children:

```xml
<word text="你" language="zh">
  <group script="ni">
    <phoneme symbol="n"/>
    <phoneme symbol="i"/>
  </group>
</word>
```

The same single-group result can be written compactly:

```xml
<word text="你" language="zh" script="ni" phonemes="n i"/>
```

Direct words bypass all global/local preprocessors and converters. All declared
candidates must contain final phonemes; a `script` without `phonemes` cannot
request G2P completion. Word-level pronunciation attributes and pronunciation
children cannot be mixed. Nested words and scopes are not allowed.

Automatic text runs and explicit words are separate conversion units. Context
within a unit is available to its backend; PFML 1.0 does not pass neighboring
units into backend inference. Backend and boundary failures do not trigger
fallback to another converter.

### reading

**Purpose:** represent one `G2PReading`, containing the complete realizations
of one reading.

**Placement:** inside a `word`, or standalone at fragment/scope level.

**Contents:** `path` children, or `group`/`phoneme` children when intermediate
containers are omitted. All direct children must use the same level.

**Attributes:** none.

```xml
<word text="重" language="zh">
  <reading><group script="chong" phonemes="ch ong"/></reading>
  <reading><group script="zhong" phonemes="zh ong"/></reading>
</word>
```

This word has two alternative readings, each with one implicit path. Multiple
paths inside a reading are alternative realizations of that same reading.
Groups or phonemes directly inside it instead form one implicit path.

Standalone readings each form their own implicit word; adjacent standalone
readings do not become alternatives. An empty `<reading/>` is invalid because
final phonemes are missing. A reading whose explicitly supplied pronunciation
is entirely silent is omitted from results.

### path

**Purpose:** represent one `G2PPath`, an ordered list of pronunciation groups.

**Placement:** inside a `reading`; inside a `word` with one implicit reading;
or standalone at fragment/scope level.

**Contents:** an ordered sequence of `group` children, or `phoneme` children
with one implicit group. All direct children must use the same level.

**Attributes:** none.

```xml
<reading>
  <path><group script="chong" phonemes="ch ong"/></path>
  <path><group script="chong" phonemes="ch o ng"/></path>
</reading>
```

This creates one implicit word containing one reading with two alternative
paths. Different paths may have different group counts and boundaries. Groups
inside a path are ordered parts of that path, not independent candidates.

Standalone paths each form their own implicit word. An empty `<path/>` is
invalid because final phonemes are missing. A path containing only explicitly
silent groups, such as `<path><group phonemes=""/></path>`, is omitted from results.

### group

**Purpose:** represent one `G2PGroup`, pairing a script label with an ordered
phoneme sequence.

**Placement:** inside a `path`, `reading` or `word`, or at fragment/scope level.
Missing outer containers are filled. Adjacent groups in one content run form
an ordered path.

**Contents:** `phoneme` children, or no child elements when the `phonemes`
attribute supplies the sequence. Direct character data is not allowed except
for formatting whitespace.

| Attribute | Required | Meaning and default |
| --- | --- | --- |
| `script` | No | Maps to `G2PGroup.script`. Omission or an empty value fills it by joining the final phoneme names with spaces. |
| `phonemes` | Conditional | Whitespace-separated final phoneme names, used when there are no `phoneme` children. `phonemes=""` explicitly means zero phonemes and the group is omitted from results. Cannot accompany children. |

Attribute form:

```xml
<group script="ni" phonemes="zh/n zh/i"/>
```

Equivalent element form:

```xml
<group script="ni">
  <phoneme language="zh" symbol="n"/>
  <phoneme language="zh" symbol="i"/>
</group>
```

Both produce `script="ni"` and `phonemes=["zh/n", "zh/i"]`. Names supplied in
the `phonemes` attribute are used verbatim; the surrounding word's language
does not add a prefix to them.

A group must supply either the attribute or at least one phoneme child.
`<group script="ni"/>` is incomplete and invalid. Explicitly empty sequences are
filtered before labels are filled; their script labels do not contribute to
an implicit word's text.

### phoneme

**Purpose:** supply one final string in `G2PGroup.phonemes`.

**Placement:** inside a `group`, `path`, `reading` or `word`, or at fragment/scope
level. Missing outer containers are filled. Adjacent phonemes in one content
run form an ordered sequence.

**Contents:** a symbol as character data when the `symbol` attribute is absent;
otherwise only optional formatting whitespace. Child elements are never allowed.

| Attribute | Required | Meaning and default |
| --- | --- | --- |
| `symbol` | Conditional | One non-empty, atomic phoneme string, preserved literally. Omission uses character data instead; that shorthand must contain no whitespace. Use separate elements for multiple symbols. |
| `language` | No | An explicit prefix for this phoneme only: a non-empty value produces `language/symbol`. Omission or an empty value leaves the symbol unchanged. This attribute is not inherited from word/scope elements or caller context and does not change `G2PWord.language`. |

```xml
<phoneme language="zh" symbol="ong"/>
```

This supplies the full name `zh/ong`. At fragment level it also fills one word,
reading, path and group, with `script="zh/ong"` and `text="zh/ong"`.

| Equivalent or alternate spelling | Resulting phoneme string |
| --- | --- |
| `<phoneme language="zh">ong</phoneme>` | `zh/ong` |
| `<phoneme symbol="zh/ong"/>` | `zh/ong` |
| `<phoneme symbol="ong"/>` | `ong` |
| `<phoneme>ong</phoneme>` | `ong` |

A `symbol` attribute and non-whitespace character data cannot both be supplied.
`<phoneme/>` and `<phoneme symbol=""/>` are invalid. A group's `phonemes=""`
specifies a silent sequence, which is omitted from results. Whitespace inside
an explicit `symbol` attribute belongs to that one symbol; it is never split.

Serialization preserves full phoneme names as complete strings, without
guessing how to split them back into language and symbol attributes.

## Omitted containers and label filling

```xml
<phoneme>ong</phoneme>
```

This is one implicit word, reading, path and group, with `phonemes=["ong"]`,
`script="ong"` and `text="ong"`. Missing containers are always **singletons**;
they never introduce candidate choices.

The content-bearing levels are `word` (source text), `group` (pronunciation
script) and `phoneme` (final symbols). The `reading` and `path` elements organize
candidates.

| Supplied siblings | At fragment/scope level | Inside a pronunciation parent |
| --- | --- | --- |
| `phoneme` | One ordered sequence in an implicit word | One ordered phoneme sequence |
| `group` | One ordered path in an implicit word | Ordered groups in one path |
| `path` | Each path becomes a separate implicit word | Alternative paths in one reading |
| `reading` | Each reading becomes a separate implicit word | Alternative readings in one word |
| `word` | Separate output words | Not allowed |

Candidate relationships require a pronunciation parent: multiple readings in
a `word`, or multiple paths in a `reading` (possibly an implicit reading inside
an explicit `word`). A `scope` only supplies language context and does not create
a candidate relationship.

Within a word, reading or path, all pronunciation children MUST be at the same
level. For example, a path and a phoneme cannot be siblings inside one word.
Write an explicit path around the phoneme to disambiguate that structure.
Wrappers cannot be repeated backwards or nested within their own level.

At fragment/scope level, a maximal run of adjacent `phoneme` elements or adjacent
`group` elements forms one implicit word. Formatting whitespace between them
is ignored. Non-whitespace ordinary text, an explicit word, a scope boundary or
a change of element level ends that run. Standalone `reading` and `path`
elements are legal but independent: each gets its own missing outer containers,
even when several are adjacent. Scopes are only allowed at fragment/scope level,
not inside a word.

```xml
<word>
  <path><phoneme>ch</phoneme><phoneme>ong</phoneme></path>
  <path><phoneme>ch</phoneme><phoneme>o</phoneme><phoneme>ng</phoneme></path>
</word>
```

This omits one reading and each path's group while retaining both complete paths.

Without the enclosing word, the same two paths are two separate words:

```xml
<path><phoneme>ch</phoneme><phoneme>ong</phoneme></path>
<path><phoneme>ch</phoneme><phoneme>o</phoneme><phoneme>ng</phoneme></path>
```

Each word has one reading with one path. Standalone readings follow the same
rule, while preserving any explicitly enclosed path alternatives within each
reading.

After silent branches are removed, absent or empty upper labels are filled
mechanically:

1. `group.script`: join that group's phonemes with one ASCII space.
2. A path's display label: join its group scripts with one space.
3. A reading's display label: join its path labels with ` | `.
4. `word.text`: join its reading labels with ` | ` as well.

Explicit non-empty labels are preserved. Display separators are never parsed
back as pronunciation data: the actual candidate tree remains authoritative.
Filling uses every surviving branch, not just the first candidate. Language is inherited
independently and is never inferred from pronunciation labels.

## Serialization

`to_pfml()` emits ordinary direct words with explicit containers and labels:

```xml
<word text="猫" language="ja"><reading><path><group script="ne"><phoneme symbol="n"/><phoneme symbol="e"/></group><group script="ko"><phoneme symbol="k"/><phoneme symbol="o"/></group></path></reading></word>
```

Serialization uses the same normalization as conversion: it removes silent
branches and fills empty labels without mutating the supplied objects. Missing
readings or missing path candidates raise `G2PConversionError`. Successfully
converted output is already normalized, so its word text, language, candidate
order, duplicate values and group boundaries survive the round trip:

```python
from g2pflow import G2PPipeline, to_pfml

serialized = to_pfml(words)
restored = G2PPipeline().convert_pfml(serialized)
assert restored == words
```

Each serialized word explicitly sets its language: `language="..."` for an ID,
`language=""` for `None`, or `language-kind="any"` for `Language.ANY`. It therefore
keeps its language even if the fragment is later placed inside a scope. The enum
marker remains distinct from a literal string
such as `"*"`; an empty language string is normalized to `None`.

Phonemes use `symbol` attributes to preserve each atomic string, including
embedded whitespace, without imposing an inventory or splitting language
prefixes.

Serialization escapes XML syntax and carriage returns. Strings containing
characters XML 1.0 cannot represent fail explicitly. The round-trip guarantee
is about the output data model, not original PFML spelling, indentation,
attribute order, comments or omitted-container choices.

## Python API and converter contract

```python
from g2pflow import G2PPipeline, parse_pfml, to_pfml
from g2pflow.converters.simple import CharPhonemeConverter

converter = CharPhonemeConverter({"h": ["HH"], "i": ["AY"]})
converter.language = ("en",)
pipeline = G2PPipeline(converters=[converter])
words = pipeline.convert_pfml('<scope language="en"><word>hi</word></scope>')
assert words[0].readings[0].paths[0][0].phonemes == ["HH", "AY"]
plan = parse_pfml('<scope language="en"><word>hi</word><phoneme>sil</phoneme></scope>')
assert len(plan.parts) == 2
assert G2PPipeline().convert_pfml(to_pfml(words)) == words
```

`parse_pfml(source, *, language=None)` validates the whole fragment and returns a
`PFMLDocument` without invoking G2P. Its optional `language` argument supplies
the initial inherited language. `PFMLText.fixed` identifies a fixed word;
other text requests retain automatic segmentation. `PFMLError` reports syntax
and content-model failures. The pipeline parses the complete input before it
invokes any converter.

`convert_pfml(source, *, languages=None)` executes that normalized plan and
returns ordinary `G2PWord` objects. An outer `scope` sets the default language.
The `languages` argument uses the same converter-filtering policy as `convert()`
for automatic text without an effective language declaration. Manual-only inputs
work with an empty pipeline.

`to_pfml(words)` returns a rootless fragment of direct words.

`Converter.convert(text) -> list[G2PWord]` and
`Converter.convert_word(text) -> G2PWord | None` are framework entry points.
Both normalize backend results: silent branches disappear, empty labels are
filled, and incomplete candidates fail with `G2PConversionError`. This also
applies when a converter is called directly, outside a pipeline. Backends
implement `_convert(text)` and optionally `_convert_word(text)`; they do not
override the public entry points or call normalization helpers themselves.

For automatic fixed words, `accepts_word(text)` is a read-only claim check that
by default requires `find()` to cover the whole input. The default
`_convert_word()` calls `_convert()` and accepts zero or one raw word. Zero
words or a wholly silent result become `None`; multiple words raise
`G2PWordBoundaryError`. Override the claim check and/or `_convert_word()` when
ordinary segmentation cannot respect the requested unit. Fixed-word
preprocessors may remove a unit but may not split it into multiple fragments.

Dictionary/lexicon converters look up the complete unit. Character converters
process every character in it. Mandarin/Cantonese/Kana converters combine their
independent internal units into a word, keeping reading combinations outside
path combinations. MeCab first tries whole-unit readings, then composes its
ordinary internally segmented units. This composition is implemented by those
backends, not used as a generic fallback for arbitrary plugins.
