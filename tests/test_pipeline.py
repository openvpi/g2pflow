from xml.sax.saxutils import escape

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

    def _convert(self, text):
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
        def _convert(self, text):
            raise RuntimeError("backend failed")
    with pytest.raises(RuntimeError, match="backend failed"):
        G2PPipeline(converters=[Broken("x", []), PassthroughConverter()]).convert("x")


def test_empty_conversion_keeps_claim_reserved():
    class Omit(LiteralConverter):
        def _convert(self, text):
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

        def _convert(self, text):
            return [G2PWord(value, language="wrong", readings=[G2PReading([[
                G2PGroup(value, [value]),
            ]])]) for value in (text.upper(), text * 2)]

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
        G2PPipeline().convert("x")
    assert G2PPipeline(converters=[PassthroughConverter()]).convert("x")[0].language is Language.ANY


@pytest.mark.parametrize("entrypoint", ["convert", "convert_pfml"])
@pytest.mark.parametrize("text", ["", " \t\r\n"])
def test_empty_input_needs_no_converter_or_preprocessing(entrypoint, text):
    class Unexpected(Preprocessor):
        def process(self, fragments):
            raise AssertionError("empty input requires no preprocessing")

    pipeline = G2PPipeline(preprocessors=[Unexpected()])
    assert getattr(pipeline, entrypoint)(text, languages=["en"]) == []


@pytest.mark.parametrize("languages", [None, ["en"], ["jpn"]])
@pytest.mark.parametrize("text", ["X & <word>X</word>", " X,X! "])
def test_plain_text_and_pfml_character_data_share_routing_and_preprocessing(text, languages):
    calls = []
    english = LiteralConverter("x", calls, "english")
    english.language = ("en", "eng")
    japanese = LiteralConverter("x", calls, "japanese")
    japanese.language = ("ja", "jpn")
    pipeline = G2PPipeline(
        preprocessors=[LowercasePreprocessor(), FilterPunctuation()],
        converters=[english, japanese, PassthroughConverter()],
    )
    plain_words = pipeline.convert(text, languages=languages)
    plain_calls = calls[:]
    calls.clear()
    assert pipeline.convert_pfml(escape(text), languages=languages) == plain_words
    assert calls == plain_calls


def test_plain_text_keeps_xml_syntax_and_characters_xml_cannot_represent():
    text = '<word language="ja">x</word>&\0\r\n'
    pipeline = G2PPipeline(converters=[LiteralConverter(text, [])])
    assert pipeline.convert(text)[0].text == text


@pytest.mark.parametrize("entrypoint", ["convert", "convert_pfml"])
def test_automatic_text_requires_an_active_converter(entrypoint):
    pipeline = G2PPipeline()
    with pytest.raises(ValueError, match="No converter"):
        getattr(pipeline, entrypoint)("x")


@pytest.mark.parametrize("tag,languages,expected", [
    (None, ["en"], None), (Language.ANY, ["en"], Language.ANY),
    (("zh", "cmn"), None, "zh"), (("zh", "cmn"), ["cmn", "zh"], "zh"),
    (("zh",), ["en"], None),
])
def test_language_resolution(tag, languages, expected):
    assert resolve_language(tag, languages) == expected


def test_viable_candidate_structure_is_preserved_and_silent_branches_are_removed():
    words = [G2PWord("merged", readings=[G2PReading(paths=[
        [G2PGroup("a", ["A"]), G2PGroup("b", ["B"])],
        [G2PGroup("ab", ["C"])], [],
    ]), G2PReading(paths=[[]])])]

    class Output(LiteralConverter):
        def _convert(self, text):
            return words

    result = G2PPipeline(converters=[Output("x", [])]).convert("x")
    assert [len(path) for path in result[0].readings[0].paths] == [2, 1]
    assert len(result[0].readings) == 1
    assert words[0].readings[0].paths[-1] == []
    assert words[0].readings[1].paths == [[]]


@pytest.mark.parametrize("word", [
    G2PWord("x"), G2PWord("x", readings=[G2PReading([])]),
    G2PWord("x", readings=[G2PReading([[G2PGroup("x", [""])]])]),
])
def test_incomplete_converter_results_fail_without_fallback(word):
    class Output(LiteralConverter):
        def _convert(self, text):
            return [word]
    pipeline = G2PPipeline(converters=[Output("x", []), PassthroughConverter()])
    with pytest.raises(G2PConversionError):
        Output("x", []).convert("x")
    with pytest.raises(G2PConversionError):
        Output("x", []).convert_word("x")
    with pytest.raises(G2PConversionError):
        pipeline.convert("x")
    with pytest.raises(G2PConversionError):
        pipeline.convert_pfml('<word text="x"/>')


def test_empty_groups_are_removed_and_labels_are_filled():
    class Output(LiteralConverter):
        def _convert(self, text):
            return [G2PWord("", readings=[G2PReading([
                [G2PGroup("silent", [])],
                [G2PGroup("silent", []), G2PGroup("", ["A"])],
                [G2PGroup("", ["A"])],
            ])]), G2PWord("silent", readings=[G2PReading([[]])])]
    expected = [G2PWord("A | A", readings=[G2PReading([
        [G2PGroup("A", ["A"])], [G2PGroup("A", ["A"])],
    ])])]
    converter = Output("x", [])
    assert converter.convert("x") == expected
    assert G2PPipeline(converters=[converter]).convert("x") == expected


def test_framework_normalizes_fixed_backend_hooks():
    class Fixed(LiteralConverter):
        def _convert_word(self, text):
            return G2PWord("", readings=[G2PReading([
                [], [G2PGroup("", ["A"])],
            ])])
    assert Fixed("x", []).convert_word("x") == G2PWord("A", readings=[
        G2PReading([[G2PGroup("A", ["A"])]])])


def test_punctuation_only_fixed_word_is_omitted():
    pipeline = G2PPipeline(preprocessors=[FilterPunctuation()], converters=[PassthroughConverter()])
    assert pipeline.convert("！。") == []
    assert pipeline.convert_pfml('<word text="！。"/>') == []


def test_preprocessors_unicode_and_apostrophes():
    parts = FilterPunctuation().process(["Hi，猫! don't well-known..."])
    assert parts == ["Hi", "猫", " don't well-known"]
    assert StripWhitespacePreprocessor().process([" ", " x "]) == ["x"]
    assert LowercasePreprocessor().process(["HELLO", "猫"]) == ["hello", "猫"]
    assert RemoveAccentsPreprocessor().process(["café résumé naïve"]) == ["cafe resume naive"]
