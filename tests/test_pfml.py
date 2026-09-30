import random
from xml.sax.saxutils import quoteattr

import pytest

from g2pflow import (
    Converter, G2PConversionError, G2PGroup, G2PPipeline, G2PReading, G2PWord,
    G2PWordBoundaryError, Language, PFMLError, PFMLText, Preprocessor,
    parse_pfml, to_pfml,
)
from g2pflow.converters.chinese import PinyinConverter
from g2pflow.converters.dictionary import DictionaryConverter
from g2pflow.converters.japanese import JapaneseKanaConverter
from g2pflow.converters.paradigm import LexiconConverter
from g2pflow.converters.simple import CharPhonemeConverter, PassthroughConverter
from g2pflow.preprocessors.simple import FilterPunctuation, LowercasePreprocessor


def result(source, **kwargs):
    return G2PPipeline().convert_pfml(source, **kwargs)


def tagged(converter, language):
    converter.language = (language,)
    return converter


def test_bare_phonemes_and_omitted_containers():
    expected = [G2PWord("ch ong", readings=[G2PReading([[G2PGroup("ch ong", ["ch", "ong"])]])])]
    leaves = "<phoneme>ch</phoneme>\n  <phoneme>ong</phoneme>"
    for source in [leaves, f"<group>{leaves}</group>", f"<path>{leaves}</path>",
                   f"<reading>{leaves}</reading>", f"<word>{leaves}</word>"]:
        assert result(source) == expected
    assert len(result("<word phonemes='ch'/><word phonemes='ong'/>")) == 2


@pytest.mark.parametrize("source,full_name", [
    ('<phoneme language="zh" symbol="ong"/>', "zh/ong"),
    ('<phoneme language="zh">ong</phoneme>', "zh/ong"),
    ('<phoneme symbol="ong"/>', "ong"),
    ('<phoneme symbol="ong">\n  </phoneme>', "ong"),
    ('<phoneme language="" symbol="ong"/>', "ong"),
    ('<phoneme symbol="zh/ong"/>', "zh/ong"),
    ('<phoneme>zh/ong</phoneme>', "zh/ong"),
])
def test_phoneme_symbol_and_language_attributes(source, full_name):
    expected = [G2PWord(full_name, readings=[G2PReading([[G2PGroup(full_name, [full_name])]])])]
    assert result(source) == expected
    assert result(f"<group>{source}</group>") == expected
    assert result(to_pfml(expected)) == expected


def test_phoneme_language_prefix_is_local_and_independent_of_word_scope():
    source = ('<scope language="cmn"><word text="你"><group script="ni">'
              '<phoneme language="zh" symbol="n"/><phoneme symbol="i"/>'
              '<phoneme language="en" symbol="AH0"/>'
              '</group></word></scope>')
    words = result(f'<scope language="ja">{source}</scope>')
    assert words == [G2PWord("你", "cmn", [G2PReading([[
        G2PGroup("ni", ["zh/n", "i", "en/AH0"]),
    ]])])]
    assert result(f'<scope language="fr">{to_pfml(words)}</scope>') == words


def test_phoneme_symbol_attribute_preserves_literal_content():
    source = ('<word text=""><reading><path><group script="">'
              '<phoneme symbol=" A&#13;&#10;&#9;B "/>'
              '<phoneme language="zh" symbol="ong"/>'
              '</group></path></reading></word>')
    words = result(source)
    assert words[0].readings[0].paths[0][0].phonemes == [" A\r\n\tB ", "zh/ong"]
    assert result(to_pfml(words)) == words


def test_all_candidates_groups_and_order_are_preserved():
    source = """<word language="cmn" text="重">
      <reading>
        <path><group script="chong" phonemes="ch ong"/></path>
        <path><group script="ch" phonemes="ch"/><group script="ong" phonemes="o ng"/></path>
        <path><group script="chong" phonemes="ch ong"/></path>
      </reading>
      <reading><phoneme>zh</phoneme><phoneme>ong</phoneme></reading>
    </word>"""
    word = result(source)[0]
    assert word == G2PWord("重", "cmn", [
        G2PReading([
            [G2PGroup("chong", ["ch", "ong"])],
            [G2PGroup("ch", ["ch"]), G2PGroup("ong", ["o", "ng"])],
            [G2PGroup("chong", ["ch", "ong"])],
        ]),
        G2PReading([[G2PGroup("zh ong", ["zh", "ong"])]])])
    assert result(to_pfml([word])) == [word]


def test_wrapped_candidates_and_content_sequences_keep_their_relationships():
    paths = result("<word><path><phoneme>A</phoneme></path><path><phoneme>B</phoneme></path></word>")
    assert paths[0].readings == [G2PReading([[G2PGroup("A", ["A"])], [G2PGroup("B", ["B"])]])]
    groups = result("<group phonemes='A'/><group phonemes='B'/>")
    assert groups[0].readings == [G2PReading([[G2PGroup("A", ["A"]), G2PGroup("B", ["B"])]])]
    readings = result("<word><reading><phoneme>A</phoneme></reading><reading><phoneme>B</phoneme></reading></word>")
    assert len(readings[0].readings) == 2
    assert paths[0].text == "A | B"
    assert readings[0].text == "A | B"


@pytest.mark.parametrize("tag", ["reading", "path"])
@pytest.mark.parametrize("scope", ["{}", "<scope>{}</scope>", '<scope language="cmn">{}</scope>'])
def test_standalone_candidate_containers_are_independent_words(tag, scope):
    siblings = f"<{tag}><phoneme>A</phoneme></{tag}>\n  <{tag}><phoneme>B</phoneme></{tag}>"
    words = result(scope.format(siblings))
    assert [word.text for word in words] == ["A", "B"]
    assert [word.readings for word in words] == [
        [G2PReading([[G2PGroup("A", ["A"])]])],
        [G2PReading([[G2PGroup("B", ["B"])]])],
    ]
    if "language" in scope:
        assert [word.language for word in words] == ["cmn", "cmn"]


def test_standalone_readings_still_preserve_their_internal_path_candidates():
    source = ('<reading><path><phoneme>A</phoneme></path><path><phoneme>B</phoneme></path></reading>'
              '<reading><phoneme>C</phoneme></reading>')
    words = result(source)
    assert [word.text for word in words] == ["A | B", "C"]
    assert [len(word.readings[0].paths) for word in words] == [2, 1]


@pytest.mark.parametrize("source,text,script", [
    ('<word script="ka" phonemes="k a">か</word>', "か", "ka"),
    ('<word phonemes="k a">か</word>', "か", "k a"),
    ('<word phonemes="k a"/>', "k a", "k a"),
    ('<word text="" script="" phonemes="k a"/>', "k a", "k a"),
    ('<word script="ka" phonemes="k a"/>', "ka", "ka"),
    ('<word text="か" script="ka" phonemes="k a"/>', "か", "ka"),
    ('<word text="か"><group script="ka"><phoneme>k</phoneme><phoneme>a</phoneme></group></word>', "か", "ka"),
])
def test_authoring_backfill(source, text, script):
    assert result(source) == [G2PWord(text, readings=[G2PReading([[G2PGroup(script, ["k", "a"])]])])]


def test_explicit_empty_phones_are_omitted():
    assert result('<group script="silent" phonemes=""/>') == []
    assert result('<word text="silent" phonemes=""/>') == []
    source = ('<word><reading><group script="silent" phonemes=""/></reading>'
              '<reading><path><group phonemes=""/></path>'
              '<path><group phonemes=""/><group phonemes="A"/></path></reading></word>')
    assert result(source) == [G2PWord("A", readings=[G2PReading([[G2PGroup("A", ["A"])]])])]


@pytest.mark.parametrize("middle", [
    '<word phonemes=""/>', '<group phonemes=""/>', '<reading><group phonemes=""/></reading>',
    '<word phonemes=""/><scope/>',
])
def test_omitting_silent_markup_keeps_surrounding_conversion_units_separate(middle):
    pipeline = G2PPipeline(converters=[PassthroughConverter()])
    words = pipeline.convert_pfml(f'a{middle}<scope>b</scope>')
    assert [word.text for word in words] == ["a", "b"]


def test_parser_normalizes_scopes_without_running_g2p():
    document = parse_pfml('hi<scope> there</scope><word>New York</word>'
                          '<scope language="ja"><word>猫</word></scope>', language="en")
    assert document.parts == [PFMLText("hi there", "en"), PFMLText("New York", "en", True),
                              PFMLText("猫", "ja", True)]


def test_text_attribute_can_supply_an_automatic_fixed_word():
    document = parse_pfml('<word text="你"/><scope language="en"><word text="New York"/></scope>')
    assert document.parts == [PFMLText("你", fixed=True), PFMLText("New York", "en", True)]
    pipeline = G2PPipeline(converters=[tagged(PassthroughConverter(), "en")])
    word = pipeline.convert_pfml('<word language="en" text="New York"/>')[0]
    assert word.text == "New York"
    assert word.readings[0].paths[0][0].phonemes == ["New York"]


def test_group_phoneme_elements_and_attribute_are_equivalent():
    attribute = '<word text="你"><group script="ni" phonemes="n i"/></word>'
    elements = '<word text="你"><group script="ni"><phoneme>n</phoneme><phoneme>i</phoneme></group></word>'
    assert result(attribute) == result(elements)


def test_all_candidate_display_separators_are_pipes():
    source = ('<word><reading><path><phoneme>A</phoneme></path><path><phoneme>B</phoneme></path></reading>'
              '<reading><phoneme>C</phoneme></reading></word>')
    assert result(source)[0].text == "A | B | C"


def test_text_and_explicit_words_end_implicit_result_runs():
    parts = parse_pfml('<phoneme>A</phoneme>text<phoneme>B</phoneme>'
                       '<word phonemes="C"/><phoneme>D</phoneme>').parts
    assert [part.text for part in parts] == ["A", "text", "B", "C", "D"]
    assert isinstance(parts[1], PFMLText)


def test_scopes_resolve_ambiguous_scripts_and_preserve_automatic_segmentation():
    chinese = tagged(CharPhonemeConverter({"学": ["X"], "生": ["S"]}), "cmn")
    japanese = tagged(CharPhonemeConverter({"学": ["G"], "生": ["S"]}), "ja")
    english = tagged(PassthroughConverter(), "en")
    pipeline = G2PPipeline(converters=[PassthroughConverter(), chinese, japanese, english])
    words = pipeline.convert_pfml('<scope language="cmn">学生'
                                  '<scope language="ja"><word>学生</word></scope>'
                                  '<scope language="en">one two</scope>学生</scope>')
    assert [word.text for word in words] == ["学", "生", "学生", "one", "two", "学", "生"]
    assert [word.language for word in words] == ["cmn", "cmn", "ja", "en", "en", "cmn", "cmn"]
    assert words[2].readings[0].paths[0][0].phonemes == ["G", "S"]
    with pytest.raises(ValueError, match="PFML language"):
        pipeline.convert_pfml('<scope language="fr">bonjour</scope>')
    with pytest.raises(ValueError, match="PFML language"):
        G2PPipeline(converters=[PassthroughConverter()]).convert_pfml('<scope language="en">hi</scope>')


def test_word_language_overrides_scope_and_empty_language_clears_it():
    pipeline = G2PPipeline(converters=[tagged(PassthroughConverter(), "en"),
                                      tagged(PassthroughConverter(), "ja")])
    words = pipeline.convert_pfml('<scope language="ja"><word language="en">x</word>'
                                  '<scope language="">y</scope>z</scope>')
    assert [word.language for word in words] == ["en", "en", "ja"]
    assert pipeline.convert_pfml("x", languages=["ja"])[0].language == "ja"
    assert result('<scope language="en"><scope language="ja">'
                  '<phoneme>X</phoneme></scope></scope>')[0].language == "ja"


@pytest.mark.parametrize("source", ["x", '<word text="x"/>'])
def test_languages_filter_only_applies_to_automatic_text_without_an_effective_language(source):
    pipeline = G2PPipeline(converters=[tagged(PassthroughConverter(), "ja"),
                                      tagged(PassthroughConverter(), "en")])
    fragment = (source + f'<scope language="ja">{source}'
                f'<scope language="">{source}</scope></scope>'
                '<word text="direct" language="ja" phonemes="DIRECT"/>')
    words = pipeline.convert_pfml(fragment, languages=["en"])
    assert [word.language for word in words] == ["en", "ja", "en", "ja"]
    assert words[-1].readings[0].paths == [[G2PGroup("DIRECT", ["DIRECT"])]]


@pytest.mark.parametrize("language_id", ["zh", "custom_zh.v2", "Voice ID", " voice-id ", "voice:alpha/beta", "voice&<id>"])
def test_language_ids_are_opaque_application_strings(language_id):
    encoded = quoteattr(language_id)
    converter = tagged(PassthroughConverter(), language_id)
    pipeline = G2PPipeline(converters=[converter])
    scoped = pipeline.convert_pfml(f'<scope language={encoded}><word text="x"/></scope>')
    assert scoped[0].language == language_id
    assert pipeline.convert_pfml(f'<word text="x" language={encoded}/>') == scoped
    direct = result(f'<phoneme language={encoded} symbol="ong"/>')
    assert direct[0].readings[0].paths[0][0].phonemes == [f"{language_id}/ong"]
    assert result(to_pfml(scoped + direct)) == scoped + direct


def test_language_routing_does_not_case_fold_application_ids():
    upper = tagged(CharPhonemeConverter({"x": ["UPPER"]}), "Voice")
    lower = tagged(CharPhonemeConverter({"x": ["LOWER"]}), "voice")
    pipeline = G2PPipeline(converters=[upper, lower])
    words = pipeline.convert_pfml('<scope language="voice">x</scope><scope language="Voice">x</scope>')
    assert [word.readings[0].paths[0][0].phonemes for word in words] == [["LOWER"], ["UPPER"]]
    with pytest.raises(ValueError, match="PFML language"):
        pipeline.convert_pfml('<scope language="VOICE">x</scope>')


def test_same_latin_spelling_can_route_to_pinyin_or_english(dictionary):
    mandarin = tagged(DictionaryConverter(str(dictionary("hang\tH ANG\n", "pinyin.txt"))), "cmn")
    english = tagged(DictionaryConverter(str(dictionary("hang\tHH AE NG\n", "english.txt"))), "en")
    pipeline = G2PPipeline(converters=[english, mandarin])
    words = pipeline.convert_pfml('<scope language="cmn">hang</scope><scope language="en">hang</scope>')
    assert [w.readings[0].paths[0][0].phonemes for w in words] == [["H", "ANG"], ["HH", "AE", "NG"]]


def test_lexicon_fixed_phrase_uses_whole_lookup_and_whole_oov_input(dictionary):
    calls = []

    class Lexicon(LexiconConverter):
        def find(self, text):
            return 0, len(text)

        def infer_oov(self, token):
            calls.append(token)
            return [["OOV"], []]

    converter = Lexicon(str(dictionary("New York\tN Y\n")))
    pipeline = G2PPipeline(converters=[converter])
    words = pipeline.convert_pfml('<word>New York</word><word>Los Angeles</word>')
    assert calls == ["Los Angeles"]
    assert words[0].readings[0].paths[0][0].phonemes == ["N", "Y"]
    assert words[1].readings == [G2PReading([[G2PGroup("Los Angeles", ["OOV"])]])]


def test_fixed_dictionary_phrase_and_original_surface(dictionary):
    converter = tagged(DictionaryConverter(str(dictionary("new york\tN Y\nnew york\tNY\n"))), "en")
    pipeline = G2PPipeline(preprocessors=[LowercasePreprocessor()], converters=[converter])
    word = pipeline.convert_pfml('<word language="en">New York</word>')[0]
    assert word.text == "New York"
    assert [p[0].phonemes for p in word.readings[0].paths] == [["N", "Y"], ["NY"]]
    assert word.readings[0].paths[0][0].script == "new york"


def test_fixed_chinese_keeps_phrase_readings_and_composes_whole_paths(dictionary):
    converter = PinyinConverter(str(dictionary("chong\tCH ONG\nchong\tCH O NG\nzhong\tZH ONG\nqing\tQ ING\n")))
    word = G2PPipeline(converters=[converter]).convert_pfml('<word language="cmn">重庆</word>')[0]
    assert word.text == "重庆"
    assert [[g.script for g in r.paths[0]] for r in word.readings] == [["chong", "qing"], ["zhong", "qing"]]
    assert [len(r.paths) for r in word.readings] == [2, 1]
    assert all(len(p) == 2 for r in word.readings for p in r.paths)


def test_fixed_kana_composes_dictionary_variants(dictionary):
    converter = JapaneseKanaConverter(str(dictionary("ne\tn e\nne\tN E\nko\tk o\nko\tK O\n")))
    word = G2PPipeline(converters=[converter]).convert_pfml('<word language="ja">ねこ</word>')[0]
    assert word.text == "ねこ"
    assert len(word.readings) == 1
    assert len(word.readings[0].paths) == 4


def test_manual_results_bypass_preprocessing_and_backend():
    class Broken(Preprocessor):
        def process(self, fragments):
            raise AssertionError("manual results must not be preprocessed")
    pipeline = G2PPipeline(preprocessors=[Broken()])
    word = pipeline.convert_pfml('<word phonemes="A B">Original!</word>')[0]
    assert word.text == "Original!"
    assert word.readings[0].paths[0][0].phonemes == ["A", "B"]


def test_fixed_boundary_failures_do_not_fall_back():
    class Split(Converter):
        def find(self, text):
            return 0, len(text)

        def _convert(self, text):
            return [G2PWord(t, readings=[G2PReading([[G2PGroup(t, [t])]])]) for t in ("a", "b")]
    pipeline = G2PPipeline(converters=[Split(), PassthroughConverter()])
    with pytest.raises(G2PWordBoundaryError, match="returned 2 words"):
        pipeline.convert_pfml("<word>ab</word>")
    assert len(pipeline.convert_pfml("ab")) == 2
    pipeline = G2PPipeline(preprocessors=[FilterPunctuation()], converters=[PassthroughConverter()])
    with pytest.raises(G2PWordBoundaryError, match="split"):
        pipeline.convert_pfml("<word>a,b</word>")
    with pytest.raises(G2PConversionError):
        G2PPipeline(converters=[CharPhonemeConverter({"a": ["A"]})]).convert_pfml("<word>ab</word>")


def test_converter_local_preprocessors_cannot_split_fixed_words():
    class Local(PassthroughConverter):
        def preprocessors(self):
            return [FilterPunctuation()]
    with pytest.raises(G2PWordBoundaryError):
        G2PPipeline(converters=[Local()]).convert_pfml("<word>a,b</word>")


def test_single_word_backend_can_use_the_default_fixed_word_hook():
    class Single(Converter):
        def find(self, text):
            return 0, len(text)

        def _convert(self, text):
            return [G2PWord("rewritten", readings=[G2PReading([[], [G2PGroup("a", ["A"])]])])]
    word = G2PPipeline(converters=[Single()]).convert_pfml("<word>source</word>")[0]
    assert word.text == "source"
    assert word.readings[0].paths == [[G2PGroup("a", ["A"])]]


def test_silent_fixed_words_are_omitted(kana_dict):
    pipeline = G2PPipeline(converters=[JapaneseKanaConverter(str(kana_dict))])
    assert pipeline.convert_pfml('<word language="ja" text="ー゜"/>') == []
    words = pipeline.convert_pfml('<word language="ja" text="カー"/>')
    assert words[0].text == "カー"
    assert words[0].readings[0].paths == [[G2PGroup("ka", ["k", "a"])]]


@pytest.mark.parametrize("source", [
    '<word script="xing">行</word>', '<group script="ka"/>', '<group/>',
    '<reading/>', '<path/>', '<word><reading/></word>', '<phoneme/>',
    '<phoneme>ch ong</phoneme>', '<word/>', '<word><text/></word>',
    '<word><text>x</text><text>y</text></word>',
    '<word><phoneme>X</phoneme>source</word>',
    '<word text="x">x</word>', '<word text="x">source<phoneme>X</phoneme></word>',
    '<word text="x"><text>x</text><phoneme>X</phoneme></word>',
    '<word><text>x</text><phoneme>X</phoneme></word>',
    '<word><phoneme>X</phoneme><path><phoneme>Y</phoneme></path></word>',
    '<group phonemes="X"><phoneme>Y</phoneme></group>',
    '<word phonemes="X"><phoneme>Y</phoneme></word>',
    '<word mode="filter">x</word>', '<word language-kind="any">x</word>',
    '<scope lang="en">x</scope>', '<g2p>x</g2p>', '<any/>',
    '<phoneme language="zh"/>', '<phoneme symbol=""/>',
    '<phoneme symbol="ong">ong</phoneme>', '<phoneme symbol="ong"><phoneme>X</phoneme></phoneme>',
    '<scope><word>x</scope>', 'x & y', '<?xml version="1.0"?><word>x</word>',
    '<!DOCTYPE x [<!ENTITY a "injected">]>&a;', '<?process x?>',
    '<word mode="exact"/>', '<word mode="exact"><text/><phoneme>X</phoneme></word>',
    '<word mode="exact" text=""><phoneme>X</phoneme></word>',
    '<word mode="exact" text="" language-kind="any" language="en"/>',
    '<word mode="exact" text=""><reading><path><group/></path></reading></word>',
    '<word text="x" language-kind="any" language="zh" phonemes="X"/>',
    '<word text="x" language-kind="unknown" phonemes="X"/>',
    '<alternatives><phoneme>A</phoneme><phoneme>B</phoneme></alternatives>',
])
def test_invalid_or_incomplete_input_never_runs_g2p(source):
    class Unexpected(PassthroughConverter):
        def find(self, text):
            raise AssertionError("the whole document must be validated first")
    with pytest.raises(PFMLError):
        G2PPipeline(converters=[Unexpected()]).convert_pfml("valid text " + source)


def test_xml_escaping_comments_cdata_and_plain_text_entry_point():
    source = '<word phonemes="X">&lt;a&gt;&amp;<![CDATA[<b>]]><!-- ignored --></word>'
    assert result(source)[0].text == "<a>&<b>"
    pipeline = G2PPipeline(converters=[PassthroughConverter()])
    assert pipeline.convert('<word>x</word>')[0].text == '<word>x</word>'
    assert pipeline.convert_pfml("&lt;x&gt;&amp;")[0].text == "<x>&"
    assert result(" \n<!-- comment -->\t") == []


def test_ordinary_pfml_round_trip_preserves_complete_candidates_and_language_types():
    readings = [G2PReading([
        [G2PGroup("a\r\nb", ["A B", "\t\n\r", "<&>\"'", "猫😀"])],
        [G2PGroup("other", ["O"])],
    ]), G2PReading([[G2PGroup("third", ["T"])]])]
    words = [G2PWord("a\r\nb\tc & <x>", language, readings)
             for language in [None, Language.ANY, "*", "en", "a<&\r\n\tb"]]
    serialized = to_pfml(words)
    assert '<text' not in serialized
    assert 'mode=' not in serialized
    assert 'language=""' in serialized
    assert 'symbol=' in serialized
    assert result(serialized) == words
    # Serialized language markers override foreign scopes without a special mode.
    assert result('<scope language="en"><scope language="ja">'
                  + serialized + '</scope></scope>') == words
    assert to_pfml(result(serialized)) == serialized
    assert result(to_pfml([])) == []


def test_serialization_applies_the_same_filtering_and_backfill_as_conversion():
    words = [G2PWord("", "", [G2PReading([
        [], [G2PGroup("", []), G2PGroup("", ["A"])],
    ])]), G2PWord("silent", readings=[G2PReading([[]])])]
    assert result(to_pfml(words)) == [G2PWord("A", readings=[G2PReading([[G2PGroup("A", ["A"])]])])]
    assert words[0].text == ""  # Normalization does not mutate caller data.
    assert to_pfml(words[1:]) == ""


@pytest.mark.parametrize("word", [G2PWord("x"), G2PWord("x", readings=[G2PReading([])])])
def test_serialization_rejects_failed_conversions(word):
    with pytest.raises(G2PConversionError):
        to_pfml([word])


def test_generated_candidate_trees_round_trip_without_flattening():
    rng = random.Random(42)
    strings = ["a", "a b", "猫", "😀", "<>&\"'", "\r\n\t"]
    for _ in range(80):
        words = [G2PWord(rng.choice(strings), rng.choice([None, Language.ANY, "en"]), [
            G2PReading([[
                G2PGroup(rng.choice(strings), [rng.choice(strings) for _ in range(rng.randrange(1, 4))])
                for _ in range(rng.randrange(1, 4))
            ] for _ in range(rng.randrange(1, 4))])
            for _ in range(rng.randrange(1, 4))
        ]) for _ in range(rng.randrange(4))]
        assert result(to_pfml(words)) == words


@pytest.mark.parametrize("text", ["\0", "\x01", "\ud800", "\uffff"])
def test_unrepresentable_xml_characters_fail_before_serializing(text):
    with pytest.raises(PFMLError, match="XML 1.0"):
        to_pfml([G2PWord(text, readings=[G2PReading([[G2PGroup("a", ["A"])]])])])
