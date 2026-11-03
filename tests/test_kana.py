import pytest

from g2pflow import G2PGroup, G2PPipeline
from g2pflow.converters._kana import parse_kana
from g2pflow.converters.japanese import JapaneseKanaConverter, KanaReading
from g2pflow.preprocessors.simple import NormalizeUnicodePreprocessor


@pytest.mark.parametrize("text,expected", [
    ("きゃっと", [("きゃ", ("kya",)), ("っ", ("cl",)), ("と", ("to",))]),
    ("あぁねぇ", [("あ", ("a",)), ("ぁ", ("a",)), ("ね", ("ne",)), ("ぇ", ("e",))]),
    ("ぐゎ", [("ぐゎ", ("gwa",))]),
    ("ゟヿ", [("ゟ", ("yo", "ri")), ("ヿ", ("ko", "to"))]),
    ("ヷヸヹヺ", [("ヷ", ("va",)), ("ヸ", ("vi",)), ("ヹ", ("ve",)), ("ヺ", ("vo",))]),
    ("すゞめ", [("すゞ", ("su", "zu")), ("め", ("me",))]),
    ("いすゞ", [("い", ("i",)), ("すゞ", ("su", "zu"))]),
    ("カヾ", [("カヾ", ("ka", "ga"))]),
    ("がゝ", [("がゝ", ("ga", "ka"))]),
    ("ぱゝ", [("ぱゝ", ("pa", "ha"))]),
    ("きゃゞ", [("きゃゞ", ("kya", "gya"))]),
    ("ぎゃゝ", [("ぎゃゝ", ("gya", "kya"))]),
    ("かゞゝ", [("かゞゝ", ("ka", "ga", "ka"))]),
    ("ヿヽ", [("ヿヽ", ("ko", "to", "to"))]),
])
def test_sequence_keys_and_source_spans(text, expected):
    units = parse_kana(text)
    assert [(text[unit.start:unit.end], unit.scripts) for unit in units] == expected
    assert "".join(text[unit.start:unit.end] for unit in units) == text


@pytest.mark.parametrize("text,expected", [
    ("っきゃ", [("っきゃ", ("kkya",))]),
    ("っーかっ", [("っーか", ("kka",)), ("っ", ("cl",))]),
    ("っあ", [("っ", ("cl",)), ("あ", ("a",))]),
    ("っっか", [("っ", ("cl",)), ("っか", ("kka",))]),
    ("っかゝ", [("っかゝ", ("kka", "ka"))]),
    ("っヿ", [("っヿ", ("kko", "to"))]),
])
def test_sokuon_with_contextual_and_multiple_key_units(text, expected):
    units = parse_kana(text, double_written_sokuon=True)
    assert [(text[unit.start:unit.end], unit.scripts) for unit in units] == expected


@pytest.mark.parametrize("text", ["ゝ", "ゞめ", "ヽ", "ヾ", "あゞ", "\u3097", "か\u3099"])
def test_unresolved_kana_is_an_error(text):
    with pytest.raises(KeyError):
        parse_kana(text)


def test_multiple_key_character_preserves_word_and_dictionary_alternatives(dictionary):
    converter = JapaneseKanaConverter(str(dictionary("ko\tK O\nko\tKO\nto\tT O\n")))
    words = converter.convert("ヿ")
    assert len(words) == 1 and words[0].text == "ヿ"
    assert words[0].readings[0].paths == [
        [G2PGroup("ko", ["K", "O"]), G2PGroup("to", ["T", "O"])],
        [G2PGroup("ko", ["KO"]), G2PGroup("to", ["T", "O"])],
    ]
    assert converter.reading("ヿ") == words[0].readings[0]


def test_reading_state_is_local_to_each_call(dictionary):
    converter = JapaneseKanaConverter(str(dictionary("ka\tKA\nga\tGA\nsa\tSA\nza\tZA\n")))
    for text, keys in [("かゞ", ["ka", "ga"]), ("さゞ", ["sa", "za"])]:
        assert [group.script for group in converter.reading(text).paths[0]] == keys
    with pytest.raises(KeyError, match="preceding syllable"):
        converter.reading("ゞ")


def test_preprocessing_and_fixed_word_boundaries(dictionary):
    converter = JapaneseKanaConverter(str(dictionary("ga\tGA\nka\tKA\n")))
    pipeline = G2PPipeline([NormalizeUnicodePreprocessor()], [converter])
    raw = "か\u3099ゝ"
    word = pipeline.convert_pfml(f"<word>{raw}</word>")[0]
    assert word.text == raw
    assert [group.script for group in word.readings[0].paths[0]] == ["ga", "ka"]
    with pytest.raises(KeyError, match="preceding syllable"):
        pipeline.convert_pfml("<word>か</word><word>ゞ</word>")


@pytest.mark.parametrize("reading,expected", [
    (KanaReading("きょう", "キョー"), ["kyo"]),
    (KanaReading("ちょうど", "チョード"), ["cho", "do"]),
    (KanaReading("おもう", "オモウ"), ["o", "mo", "u"]),
    (KanaReading("は", "ワ"), ["wa"]),
    (KanaReading("は"), ["ha"]),
    (KanaReading("きょう"), ["kyo", "u"]),
])
def test_spelling_and_explicit_pronunciation_are_distinct(dictionary, reading, expected):
    path = dictionary("kyo\tKY O\ncho\tCH O\ndo\tD O\no\tO\nmo\tM O\nu\tU\nwa\tW A\nha\tH A\n")
    converter = JapaneseKanaConverter(str(path))
    assert [group.script for group in converter.reading(reading).paths[0]] == expected
