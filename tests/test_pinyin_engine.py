import pytest

from g2pflow.converters.cpp_pinyin import PinyinEngine, apply_tone
from g2pflow.converters.cpp_pinyin.constants import (
    STYLE_BOPOMOFO, STYLE_FIRST_LETTER, STYLE_NORMAL, STYLE_TONE, STYLE_TONE2, STYLE_TONE3,
)
from g2pflow.converters.cpp_pinyin.dict_loader import (
    encode_phrase_key, load_phrase_dict, load_phrase_map, load_trans_dict,
    load_user_dict, load_word_dict,
)


def test_dictionary_loaders_skip_malformed_lines(dictionary, tmp_path):
    words = dictionary("\ninvalid\n:pron\n字:zì,zi\n好:hǎo\n")
    assert load_word_dict(words) == {"字": [["zì"], ["zi"]], "好": [["hǎo"]]}
    phrases = dictionary("好字:hǎo,zì\n好字:hao,zi\n坏:huài\n坏词:huài\ninvalid\n", "phrases.txt")
    key = encode_phrase_key(list("好字"))
    assert key == (ord("好") << 16) | ord("字")
    assert load_phrase_dict(phrases) == {key: [["hǎo", "zì"], ["hao", "zi"]]}
    mapping = dictionary("\n字:1\n好:2\ninvalid\n", "map.txt")
    assert load_phrase_map(mapping) == {"字", "好"}
    trans = dictionary("\n體:体\n:字\n字:\ninvalid\n", "trans.txt")
    assert load_trans_dict(trans) == {"體": "体"}
    user = dictionary("好字:hao3 zi4\n坏:huai4\n坏词:huai4\ninvalid\n", "user.txt")
    assert load_user_dict(user) == {key: [["hao3", "zi4"]]}
    assert load_user_dict(tmp_path / "absent.txt") == {}


@pytest.mark.parametrize("polyphonic,phrase,expected", [
    ("甲", "甲乙", ["x", "y", "ma", "ma"]),
    ("乙", "甲乙", ["x", "y", "ma", "ma"]),
    ("丙", "甲乙丙", ["x", "y", "z", "ma"]),
    ("丙", "甲乙丙丁", ["x", "y", "z", "w"]),
])
def test_phrase_matching_directions(tmp_path, polyphonic, phrase, expected):
    (tmp_path / "word.txt").write_text("\n".join(f"{char}:mā,má" for char in "甲乙丙丁"), encoding="utf-8")
    (tmp_path / "phrases_map.txt").write_text(f"{polyphonic}:1\n", encoding="utf-8")
    (tmp_path / "phrases_dict.txt").write_text(f"{phrase}:" + ",".join("xyzw"[:len(phrase)]), encoding="utf-8")
    (tmp_path / "trans_word.txt").write_text("體:甲\n", encoding="utf-8")
    engine = PinyinEngine(tmp_path)
    assert engine.is_ready()
    assert engine.query_raw(list("甲乙丙丁"), style=STYLE_NORMAL) == [[value] for value in expected]
    assert engine.has_polyphonic(polyphonic)
    assert engine.simplify(["體", "?"]) == ["甲", "?"]
    assert engine.readings("體") == ["ma1", "ma2"]
    assert engine.default_reading("體", style=STYLE_NORMAL) == "ma"
    assert engine.readings("?") == ["?"]
    assert engine.default_reading("?") == "?"
    assert engine.query_raw(["?"], style=STYLE_NORMAL) == [["?"]]


def test_user_phrases_and_polyphonic_default(tmp_path):
    for name, data in {
        "word.txt": "甲:mā,má\n乙:yǐ\n",
        "phrases_map.txt": "甲:1\n", "phrases_dict.txt": "",
        "trans_word.txt": "", "user_dict.txt": "甲乙:ga1 yi3\n",
    }.items():
        (tmp_path / name).write_text(data, encoding="utf-8")
    engine = PinyinEngine(tmp_path)
    assert engine.query_raw(["甲"]) == [["ma1"]]
    assert engine.query_raw(list("甲乙")) == [["ga1"], ["yi3"]]


@pytest.mark.parametrize("text,style,kwargs,expected", [
    ("zhōng", STYLE_TONE3, {}, "zhong1"),
    ("nǚ", STYLE_TONE3, {}, "nv3"),
    ("nǚ", STYLE_TONE3, {"v_to_u": True}, "nü3"),
    ("le", STYLE_TONE3, {"neutral_tone5": False}, "le"),
    ("le", STYLE_TONE3, {}, "le5"),
    ("neoi5", STYLE_NORMAL, {}, "neoi"),
    ("nv3", STYLE_NORMAL, {"v_to_u": True}, "nü"),
    ("zhōng", STYLE_FIRST_LETTER, {}, "z"),
    ("", STYLE_FIRST_LETTER, {}, ""),
    ("ma3", STYLE_BOPOMOFO, {}, "ㄇㄚˇ"),
    ("ma1", STYLE_TONE, {}, "mā"),
    ("le5", STYLE_TONE, {}, "le"),
    ("m3", STYLE_TONE, {}, "m"),
    ("hao3", STYLE_TONE2, {}, "ha3o"),
    ("m3", STYLE_TONE2, {}, "m3"),
    ("le5", STYLE_TONE2, {}, "le"),
    ("ma3", -1, {}, "ma"),
])
def test_tone_formats(text, style, kwargs, expected):
    assert apply_tone(text, style, **kwargs) == expected


def test_default_bundled_engine():
    engine = PinyinEngine()
    assert engine.is_ready()
    assert engine.default_reading("你") == "ni3"
