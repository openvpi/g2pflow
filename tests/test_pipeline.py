import pytest

from g2pflow import (
    Converter, G2PConversionError, G2PGroup, G2PPipeline, G2PReading, G2PWord,
    Language, Preprocessor,
)
from g2pflow.converters.base import resolve_language
from g2pflow.converters.simple import PassthroughConverter
from g2pflow.preprocessors.simple import (
    FilterPunctuation, LowercasePreprocessor, RemoveAccentsPreprocessor,
    StripWhitespacePreprocessor,
)


class LiteralConverter(Converter):
    def __init__(self, literal, calls, label=None):
        self.literal = literal
        self.calls = calls
        self.label = label or literal

    def find(self, text):
        start = text.find(self.literal)
        return None if start < 0 else (start, start + len(self.literal))

    def convert(self, text):
        self.calls.append((self.label, text))
        return [G2PWord(text, readings=[G2PReading(paths=[[
            G2PGroup(self.label, [self.label]),
        ]])])]


def test_priority_claims_do_not_cross_and_output_is_in_text_order():
    calls = []
    pipeline = G2PPipeline(converters=[
        LiteralConverter("bc", calls), LiteralConverter("ab", calls),
        LiteralConverter("a", calls), LiteralConverter("d", calls),
    ])
    words = pipeline.convert("abcd bc")
    assert [word.text for word in words] == ["a", "bc", "d", "bc"]
    assert calls == [("a", "a"), ("bc", "bc"), ("d", "d"), ("bc", "bc")]


def test_unrecognized_ranges_are_reported_before_any_conversion():
    calls = []
    pipeline = G2PPipeline(converters=[LiteralConverter("ok", calls)])
    with pytest.raises(G2PConversionError) as exc:
        pipeline.convert("bad ok worse ok last")
    assert exc.value.unconverted_tokens == ["bad ", " worse ", " last"]
    assert calls == []
    assert pipeline.convert(" \t ") == []


@pytest.mark.parametrize("match", [(-1, 1), (0, 0), (2, 1), (0, 4)])
def test_invalid_claim_ranges(match):
    class Invalid(LiteralConverter):
        def find(self, text):
            return match
    with pytest.raises(ValueError, match="invalid range"):
        G2PPipeline(converters=[Invalid("x", [])]).convert("abc")


def test_conversion_failure_does_not_fall_back():
    class Broken(LiteralConverter):
        def convert(self, text):
            raise RuntimeError("backend failed")
    with pytest.raises(RuntimeError, match="backend failed"):
        G2PPipeline(converters=[Broken("x", []), PassthroughConverter()]).convert("x")


def test_empty_conversion_keeps_claim_reserved():
    class Omit(LiteralConverter):
        def convert(self, text):
            return []
    pipeline = G2PPipeline(converters=[Omit("bc", []), PassthroughConverter()])
    assert [word.text for word in pipeline.convert("abc")] == ["a"]


def test_local_preprocessing_can_split_rewrite_and_omit():
    class Local(Preprocessor):
        def process(self, fragments):
            assert fragments == ["ab"]
            return ["a", "", "b"]

    class Split(LiteralConverter):
        language = ("en",)

        def preprocessors(self):
            return [Local()]

        def convert(self, text):
            return [G2PWord(text.upper()), G2PWord(text * 2, language="wrong")]

    result = G2PPipeline(converters=[Split("ab", [])]).convert("ab")
    assert [word.text for word in result] == ["A", "aa", "B", "bb"]
    assert [word.language for word in result] == ["en"] * 4


def test_global_fragments_cannot_be_rejoined_by_routing():
    pipeline = G2PPipeline(
        preprocessors=[FilterPunctuation(), StripWhitespacePreprocessor(), LowercasePreprocessor()],
        converters=[LiteralConverter("ab", []), PassthroughConverter()],
    )
    assert [word.text for word in pipeline.convert(" A,B! ")] == ["a", "b"]


def test_language_filter_and_markers():
    english = LiteralConverter("x", [], "english")
    english.language = ("en", "eng")
    japanese = LiteralConverter("x", [], "japanese")
    japanese.language = ("ja", "jpn")
    pipeline = G2PPipeline(converters=[english, japanese])
    assert pipeline.convert("x")[0].language == "en"
    assert pipeline.convert("x", languages=["jpn"])[0].language == "jpn"
    with pytest.raises(ValueError, match="No converter"):
        pipeline.convert("x", languages=["zh"])
    with pytest.raises(ValueError, match="No converter"):
        G2PPipeline().convert("")
    assert G2PPipeline(converters=[PassthroughConverter()]).convert("x")[0].language is Language.ANY


@pytest.mark.parametrize("tag,languages,expected", [
    (None, ["en"], None), (Language.ANY, ["en"], Language.ANY),
    (("zh", "cmn"), None, "zh"), (("zh", "cmn"), ["cmn", "zh"], "zh"),
    (("zh",), ["en"], None),
])
def test_language_resolution(tag, languages, expected):
    assert resolve_language(tag, languages) == expected


def test_word_structure_and_candidate_identity_are_preserved():
    words = [G2PWord("merged", readings=[G2PReading(paths=[
        [G2PGroup("a", ["A"]), G2PGroup("b", ["B"])],
        [G2PGroup("ab", ["C"])], [],
    ]), G2PReading(paths=[])])]

    class Output(LiteralConverter):
        def convert(self, text):
            return words

    result = G2PPipeline(converters=[Output("x", [])]).convert("x")
    assert result[0] is words[0]
    assert result[0].readings[0].paths[-1] == []
    assert result[0].readings[1].paths == []
    assert [len(path) for path in result[0].readings[0].paths] == [2, 1, 0]


def test_preprocessors_unicode_and_apostrophes():
    parts = FilterPunctuation().process(["Hi，猫! don't well-known..."])
    assert parts == ["Hi", "猫", " don't well-known"]
    assert StripWhitespacePreprocessor().process([" ", " x "]) == ["x"]
    assert LowercasePreprocessor().process(["HELLO", "猫"]) == ["hello", "猫"]
    assert RemoveAccentsPreprocessor().process(["café résumé naïve"]) == ["cafe resume naive"]
