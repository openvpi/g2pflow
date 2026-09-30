import pytest

from g2pflow import G2PConversionError, G2PGroup, G2PPipeline, Language
from g2pflow.converters.chinese import JyutpingConverter, PinyinConverter
from g2pflow.converters.dictionary import DictionaryConverter, load_pronunciation_dict
from g2pflow.converters.japanese import JapaneseKanaConverter
from g2pflow.converters.paradigm import LexiconConverter, PronunciationScriptConverter
from g2pflow.converters.simple import CharPhonemeConverter, PassthroughConverter
from g2pflow.converters.text import find_run, split_words, word_spans


def phones(word):
    return [[phone for group in path for phone in group.phonemes]
            for reading in word.readings for path in reading.paths]


def test_dictionary_variants_and_loader(dictionary):
    path = dictionary("Hello\thh ax l ow\nhello(1)\thh eh l ow\nhello (2)\th e l o\n"
                      "hello\tH\nempty\n\n")
    assert load_pronunciation_dict(path)["hello"] == [["H"]]
    converter = DictionaryConverter(str(path))
    assert converter.find("say HELLO") == (4, 9)
    assert converter.find("helloworld") is None
    assert phones(converter.convert("HELLO")[0]) == [
        ["hh", "ax", "l", "ow"], ["hh", "eh", "l", "ow"], ["h", "e", "l", "o"], ["H"],
    ]
    with pytest.raises(KeyError, match="find should"):
        converter.convert("unknown")


def test_passthrough_and_character_empty_paths():
    assert [word.text for word in PassthroughConverter().convert("one 猫")] == ["one", "猫"]
    converter = CharPhonemeConverter({"a": ["A", "AA"], "b": [], "c": ["C"]})
    assert converter.language is Language.ANY
    assert converter.find("zz abc") == (3, 6)
    assert converter.find("az") is None
    assert phones(converter.convert("abc")[0]) == [["A", "AA", "C"]]
    assert converter.convert("b") == []
    assert converter.convert_word("b") is None


def test_word_boundaries_and_runs():
    text = "hello猫きゃ カー 한 don't-test"
    assert split_words(text) == ["hello", "猫", "きゃ", "カ", "ー", "한", "don't-test"]
    assert [text[a:b] for a, b in word_spans(text)] == split_words(text)
    assert find_run("xx123a45", str.isdigit) == (2, 5)
    assert find_run("abc", str.isdigit) is None


def test_lexicon_lookup_then_oov(dictionary):
    class Lexicon(LexiconConverter):
        def find(self, text):
            return (0, len(text)) if text else None

        def infer_oov(self, token):
            return [["NEW"], []]
    converter = Lexicon(str(dictionary("known\tK\nknown\tN\n")))
    assert phones(converter.convert("known")[0]) == [["K"], ["N"]]
    assert phones(converter.convert("missing")[0]) == [["NEW"]]
    assert phones(Lexicon().convert("x")[0])[0] == ["NEW"]


def test_script_readings_and_paths_are_deduplicated_without_flattening():
    class Script(PronunciationScriptConverter):
        def find(self, text):
            return (0, len(text))

        def text_to_scripts(self, words):
            return [["first", "first", "second"] for _ in words]

        def script_to_paths(self, script):
            if script == "second":
                return [[]]
            return [[G2PGroup("a", ["A"]), G2PGroup("b", ["B"])],
                    [G2PGroup("a", ["A"]), G2PGroup("b", ["B"])],
                    [G2PGroup("ab", ["A", "B"])]]

    word = Script().convert("word")[0]
    assert len(word.readings) == 1
    assert [len(path) for path in word.readings[0].paths] == [2, 1]

    class Broken(Script):
        def text_to_scripts(self, words):
            return []
    with pytest.raises(ValueError, match="word count"):
        Broken().convert("word")

    class NoPaths(Script):
        def script_to_paths(self, script):
            return []
    with pytest.raises(G2PConversionError, match="no candidate paths"):
        NoPaths().convert("word")

    class NoReadings(Script):
        def text_to_scripts(self, words):
            return [[] for _ in words]
    with pytest.raises(G2PConversionError, match="no readings"):
        NoReadings().convert("word")


def test_mandarin_phrase_priority_and_alternatives(dictionary):
    converter = PinyinConverter(str(dictionary("chong\tCH ONG\nzhong\tZH ONG\nqing\tQ ING\n")))
    words = converter.convert("重庆")
    assert [word.text for word in words] == ["重", "庆"]
    assert words[0].readings[0].paths[0][0].script == "chong"
    assert {reading.paths[0][0].script for reading in words[0].readings} == {"chong", "zhong"}
    assert converter.find("abc重庆xyz") == (3, 5)
    assert converter.find("abc") is None


def test_unsupported_chinese_readings_propagate(dictionary):
    converter = PinyinConverter(str(dictionary("unused\tU\n")))
    with pytest.raises(KeyError, match="not found"):
        converter.convert("你")


def test_cantonese_bundled_readings(dictionary):
    converter = JyutpingConverter(str(dictionary("nei\tn ei\nhou\th ou\n")))
    assert [phones(word)[0] for word in converter.convert("你好")] == [["n", "ei"], ["h", "ou"]]


def test_kana_digraphs_katakana_and_empty_pronunciation(kana_dict):
    converter = JapaneseKanaConverter(str(kana_dict))
    words = converter.convert("キャットー゜")
    assert [word.text for word in words] == ["キャ", "ッ", "ト"]
    assert [phones(word) for word in words] == [[["ky", "a"]], [["cl"]], [["t", "o"]]]
    assert converter.convert("ー゜") == []
    assert converter.convert_word("ー゜") is None
    assert phones(converter.convert_word("カー")) == [["k", "a"]]
    assert converter.find("wordきゃっと猫") == (4, 8)


@pytest.mark.parametrize("text,script", [
    ("った", "tta"), ("ッタ", "tta"), ("っきゃ", "kkya"),
])
def test_double_written_sokuon_looks_up_one_complete_dictionary_key(dictionary, text, script):
    path = dictionary(f"{script}\tCLOSURE ONSET VOWEL\n{script}\tGEM VOWEL\n")
    converter = JapaneseKanaConverter(str(path), double_written_sokuon=True)
    words = converter.convert(text)
    assert len(words) == 1
    assert words[0].text == text
    assert words[0].readings[0].paths == [
        [G2PGroup(script, ["CLOSURE", "ONSET", "VOWEL"])],
        [G2PGroup(script, ["GEM", "VOWEL"])],
    ]


def test_double_written_sokuon_skips_empty_scripts_and_preserves_trailing_cl(dictionary):
    path = dictionary("kka\tGEM_K A\ncl\tCLOSURE\n")
    converter = JapaneseKanaConverter(str(path), double_written_sokuon=True)
    words = converter.convert("っーかっ")
    assert [word.text for word in words] == ["っーか", "っ"]
    assert [phones(word) for word in words] == [[["GEM_K", "A"]], [["CLOSURE"]]]


def test_sokuon_before_vowel_keeps_the_cl_dictionary_key(dictionary):
    path = dictionary("cl\tCLOSURE\na\tVOWEL\n")
    converter = JapaneseKanaConverter(str(path), double_written_sokuon=True)
    assert [phones(word) for word in converter.convert("っあ")] == [
        [["CLOSURE"]], [["VOWEL"]],
    ]


def test_sokuon_disabled_uses_separate_dictionary_entries(dictionary):
    path = dictionary("cl\tCLOSURE\nta\tONSET VOWEL\n")
    converter = JapaneseKanaConverter(str(path))
    words = converter.convert("った")
    assert [word.text for word in words] == ["っ", "た"]
    assert [phones(word) for word in words] == [[["CLOSURE"]], [["ONSET", "VOWEL"]]]


def test_double_written_sokuon_requires_the_combined_key_without_phoneme_fallback(dictionary):
    path = dictionary("cl\tCLOSURE\nta\tONSET VOWEL\n")
    converter = JapaneseKanaConverter(str(path), double_written_sokuon=True)
    with pytest.raises(KeyError, match="tta"):
        converter.convert("った")
    with pytest.raises(KeyError, match="not found"):
        converter.script_to_paths("t")


def test_pfml_fixed_kana_word_keeps_combined_script_and_dictionary_candidates(dictionary):
    path = dictionary("ka\tK A\ntta\tCLOSURE T A\ntta\tGEM_T A\n")
    converter = JapaneseKanaConverter(str(path), double_written_sokuon=True)
    word = G2PPipeline(converters=[converter]).convert_pfml('<word text="かった" language="ja"/>')[0]
    assert word.text == "かった"
    assert word.readings[0].paths == [
        [G2PGroup("ka", ["K", "A"]), G2PGroup("tta", ["CLOSURE", "T", "A"])],
        [G2PGroup("ka", ["K", "A"]), G2PGroup("tta", ["GEM_T", "A"])],
    ]


@pytest.mark.parametrize("forms,script,expected", [
    (("ぢゃ", "ヂャ"), "ja", ["j", "a"]),
    (("ぢゅ", "ヂュ"), "ju", ["j", "u"]),
    (("ぢぇ", "ヂェ"), "je", ["j", "e"]),
    (("ぢょ", "ヂョ"), "jo", ["j", "o"]),
    (("づぁ", "ヅァ"), "za", ["z", "a"]),
    (("づぉ", "ヅォ"), "zo", ["z", "o"]),
])
def test_voiced_kana_variants_use_existing_dictionary_keys(dictionary, forms, script, expected):
    path = dictionary("ja\tj a\nju\tj u\nje\tj e\njo\tj o\nza\tz a\nzo\tz o\n")
    converter = JapaneseKanaConverter(str(path))
    for text in forms:
        words = converter.convert(text)
        assert len(words) == 1
        assert words[0].text == text
        assert words[0].readings[0].paths == [[G2PGroup(script, expected)]]
