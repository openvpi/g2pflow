"""Offline tests for the japanese-kashi converter (no model download)."""

import pytest

from g2pflow import G2PConversionError, G2PGroup, G2PPipeline
from g2pflow.converters.kashi import MODEL_URL, LEXICON_URL, JapaneseKashiConverter
from g2pflow.converters.kashi.segmentation import Segment


class _FakeSegmenter:
    def __init__(self, segments_by_text):
        self.segments_by_text = segments_by_text
        self.calls = []

    def segment(self, text):
        self.calls.append(text)
        return [Segment(*item)
                for item in self.segments_by_text.get(text, [])]


class _FakeRuntime:
    def __init__(self, words_by_text):
        self.words_by_text = words_by_text
        self.calls = []

    def predict_words(self, text, segments, *, beam_size=16):
        self.calls.append((text, len(segments), beam_size))
        words = []
        cursor = 0
        for surface, readings in self.words_by_text.get(text, []):
            words.append({
                "surface": surface, "start": cursor,
                "end": cursor + len(surface),
                "candidates": [{"reading": reading, "score": -float(index)}
                               for index, reading in enumerate(
                                   readings[:beam_size])],
            })
            cursor += len(surface)
        return words


def phones(word):
    return [[phone for group in path for phone in group.phonemes]
            for reading in word.readings for path in reading.paths]


@pytest.fixture
def toshi_dict(dictionary):
    return dictionary(
        "a\ta\ni\ti\nu\tu\ne\te\no\to\nzu\tz u\nshi\tsh i\n"
        "hi\th i\nga\tg a\nka\tk a\nne\tn e\nko\tk o\nru\tr u\n", "toshi.txt",
    )


@pytest.fixture
def kashi_converter(kana_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    return JapaneseKashiConverter(str(kana_dict), model_path=str(model),
                                  lexicon_path=str(lexicon))


def test_release_urls():
    assert MODEL_URL.endswith("v1.0.0/model.onnx")
    assert LEXICON_URL.endswith("v1.0.0/lexicon.txz")


def test_find_claims_japanese_runs_only(kashi_converter):
    assert kashi_converter.find("今日は良い天気ですね。") == (0, 10)
    assert kashi_converter.find("I do Love もう止められない") == (10, 18)
    assert kashi_converter.find("1人で行く") == (1, 5)
    assert kashi_converter.find("Hello world") is None
    assert kashi_converter.find("") is None


def test_fixed_segmentation_words(kashi_converter):
    segments = [(0, 1, "猫"), (1, 2, "が"), (2, 4, "好き")]
    fake_seg = _FakeSegmenter({"猫が好き": segments})
    fake_rt = _FakeRuntime({"猫が好き": [("猫", ["ねこ"]), ("が", ["が"]),
                                        ("好き", ["すき"])]})
    kashi_converter._segmenter = fake_seg
    kashi_converter._runtime = fake_rt
    words = kashi_converter.convert("猫が好き")
    assert [word.text for word in words] == ["猫", "が", "好き"]
    assert [phones(word) for word in words] == [
        [["n", "e", "k", "o"]],
        [["g", "a"]],
        [["s", "u", "k", "i"]],
    ]
    assert fake_seg.calls == ["猫が好き"]
    assert fake_rt.calls == [("猫が好き", 3, 16)]


def test_word_candidates_match_mecab_nbest_format(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"東": [(0, 1, "東")]})
    converter._runtime = _FakeRuntime({"東": [("東", ["ひがし", "あず"])]})
    words = converter.convert("東")
    assert len(words) == 1
    assert words[0].text == "東"
    assert [reading.paths for reading in words[0].readings] == [
        [[G2PGroup("hi", ["h", "i"]), G2PGroup("ga", ["g", "a"]),
          G2PGroup("shi", ["sh", "i"])]],
        [[G2PGroup("a", ["a"]), G2PGroup("zu", ["z", "u"])]],
    ]
    assert phones(words[0]) == [["h", "i", "g", "a", "sh", "i"],
                                ["a", "z", "u"]]


def test_duplicate_candidates_are_deduped(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"東": [(0, 1, "東")]})
    converter._runtime = _FakeRuntime({"東": [("東", ["ひがし", "ひがし", "あず"])]})
    words = converter.convert("東")
    assert len(words[0].readings) == 2


def test_beam_size_is_configurable(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon), beam_size=4)
    fake_rt = _FakeRuntime({"東": [("東", ["ひがし", "あず", "ひがし",
                                          "あず", "ひがし"])]})
    converter._segmenter = _FakeSegmenter({"東": [(0, 1, "東")]})
    converter._runtime = fake_rt
    words = converter.convert("東")
    assert len(words[0].readings) == 2
    assert fake_rt.calls == [("東", 1, 4)]


def test_long_vowels_folded_to_match_mecab(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    kyo_dict = dictionary("kyo\tk y o\nu\tu\n", "kyo.txt")
    converter = JapaneseKashiConverter(str(kyo_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"今日": [(0, 2, "今日")]})
    converter._runtime = _FakeRuntime({"今日": [("今日", ["きょう"])]})
    words = converter.convert("今日")
    assert len(words) == 1
    # きょう -> きょー -> kyo (ー dropped), same as MeCab's UniDic pron キョー
    assert phones(words[0]) == [["k", "y", "o"]]


def test_kana_words_use_whole_word_readings(kashi_converter):
    kashi_converter._segmenter = _FakeSegmenter({"キャット": [(0, 4, "キャット")]})
    kashi_converter._runtime = _FakeRuntime({"キャット": [("キャット", ["キャット"])]})
    words = kashi_converter.convert("キャット")
    assert [word.text for word in words] == ["キャット"]
    assert phones(words[0]) == [["ky", "a", "cl", "t", "o"]]


def test_unreadable_kanji_raises_like_mecab(kashi_converter):
    kashi_converter._segmenter = _FakeSegmenter({"珍": [(0, 1, "珍")]})
    kashi_converter._runtime = _FakeRuntime({"珍": [("珍", ["珍"])]})
    with pytest.raises(G2PConversionError):
        kashi_converter.convert("珍")


def test_candidates_with_kanji_are_dropped_whole(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"光る": [(0, 2, "光る")]})
    converter._runtime = _FakeRuntime(
        {"光る": [("光る", ["ひかる", "光る"])]})
    words = converter.convert("光る")
    assert len(words) == 1
    assert len(words[0].readings) == 1  # the kanji-bearing candidate is gone
    assert phones(words[0]) == [["h", "i", "k", "a", "r", "u"]]


def test_unknowable_low_rank_candidates_are_skipped(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(dictionary("hi\th i\nka\tk a\n",
                                                    "tiny.txt")),
                                       model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"光か": [(0, 2, "光か")]})
    converter._runtime = _FakeRuntime(
        {"光か": [("光か", ["ひか", "ぐわ"])]})
    words = converter.convert("光か")
    assert len(words) == 1
    assert len(words[0].readings) == 1  # ぐわ needs a missing dict key, skipped
    assert phones(words[0]) == [["h", "i", "k", "a"]]


def test_kana_words_handle_small_kana(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("ne\tn e\ne\te\n", "fallback.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"ねぇ": [(0, 2, "ねぇ")]})
    converter._runtime = _FakeRuntime({"ねぇ": [("ねぇ", ["ねぇ"])]})
    words = converter.convert("ねぇ")
    assert [word.text for word in words] == ["ねぇ"]
    assert phones(words[0]) == [["n", "e", "e"]]


def test_kana_particles_use_unidic_pronunciation(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("wa\tw a\nha\th a\ne\te\nhe\th e\n", "particles.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    for surface, pron, expected in (
            ("は", "ワ", [["w", "a"], ["h", "a"]]),
            ("へ", "エ", [["e"], ["h", "e"]])):
        converter._segmenter = _FakeSegmenter({surface: [(0, 1, surface, pron)]})
        converter._runtime = _FakeRuntime({surface: [(surface, [surface])]})
        words = converter.convert(surface)
        assert [word.text for word in words] == [surface]
        assert phones(words[0]) == expected


def test_kana_words_keep_unidic_boundaries(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("sa\ts a\nyo\ty o\nna\tn a\nra\tr a\n", "kana.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter(
        {"さよなら": [(0, 4, "さよなら", "サヨナラ")]})
    converter._runtime = _FakeRuntime(
        {"さよなら": [("さよなら", ["さよなら"])]})
    words = converter.convert("さよなら")
    assert [word.text for word in words] == ["さよなら"]
    assert len(words[0].readings) == 1
    assert phones(words[0]) == [["s", "a", "y", "o", "n", "a", "r", "a"]]


def test_kana_word_dictionary_gap_is_a_conversion_error(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("ne\tn e\n", "gap.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"ねぇ": [(0, 2, "ねぇ")]})
    converter._runtime = _FakeRuntime({"ねぇ": [("ねぇ", ["ねぇ"])]})
    with pytest.raises(G2PConversionError):
        converter.convert("ねぇ")


def test_empty_reading_omits_word(kashi_converter):
    kashi_converter._segmenter = _FakeSegmenter({"": []})
    kashi_converter._runtime = _FakeRuntime({"": []})
    assert kashi_converter.convert("") == []


def test_runtime_reuses_cached_instances(kashi_converter):
    fake_rt = _FakeRuntime({"あ": [("あ", ["あ"])]})
    fake_seg = _FakeSegmenter({"あ": [(0, 1, "あ")]})
    kashi_converter._runtime = fake_rt
    kashi_converter._segmenter = fake_seg
    assert kashi_converter._get_runtime() is fake_rt
    assert kashi_converter._get_segmenter() is fake_seg


@pytest.mark.integration
def test_real_model_end_to_end(kana_dict, tmp_path):
    """Opt-in: downloads the real model (~106 MB) on first use."""
    converter = JapaneseKashiConverter(str(kana_dict), cache_dir=str(tmp_path))
    words = converter.convert("今日は良い天気ですね")
    assert words and words[0].readings


def test_pipeline_config_integration(kana_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    from g2pflow import G2PPipelineConfig, build_pipeline_from_config
    pipeline = build_pipeline_from_config(G2PPipelineConfig.model_validate({
        "converters": [{
            "id": "japanese-kashi",
            "language": "ja",
            "kwargs": {"dict_path": str(kana_dict),
                       "model_path": str(model),
                       "lexicon_path": str(lexicon)},
        }],
    }))
    pipeline._converters[0]._segmenter = _FakeSegmenter({"ネコが": [(0, 2, "ネコ"), (2, 3, "が")]})
    pipeline._converters[0]._runtime = _FakeRuntime({"ネコが": [("ネコ", ["ねこ"]), ("が", ["が"])]})
    words = pipeline.convert("ネコが", languages=["ja"])
    assert [word.text for word in words] == ["ネコ", "が"]


# ------------------------------------------------- context metadata


def test_decompose_reading_splits_compound():
    from g2pflow.converters.kashi.context import decompose_reading
    chars = {"汽": frozenset({"き"}), "船": frozenset({"せん"})}
    assert decompose_reading("汽船", "きせん", chars) == ((0, "き"), (1, "せん"))
    assert decompose_reading("汽船", "ふね", chars) == ()


def test_decompose_reading_drops_ambiguous_positions():
    from g2pflow.converters.kashi.context import decompose_reading
    chars = {"灯": frozenset({"いた", "い"}), "火": frozenset({"く", "たく"}),
             "刀": frozenset({"し"})}
    # いたくし = (いた,く,し) or (い,たく,し): only 刀 is unambiguous.
    assert decompose_reading("灯火刀", "いたくし", chars) == ((2, "し"),)


def test_decompose_reading_okurigana_consumes_itself():
    from g2pflow.converters.kashi.context import decompose_reading
    chars = {"泳": frozenset({"およ"})}
    assert decompose_reading("泳いで", "およいで", chars) == ((0, "およ"),)


def test_build_context_metadata_word_and_char_keys():
    from g2pflow.converters.kashi.context import (CONTEXT_METADATA,
                                                  build_context_metadata)
    chars = {"汽": frozenset({"き"}), "船": frozenset({"せん"})}
    segments = [Segment(3, 5, "汽船", "", "キセン"),
                Segment(5, 6, "の", "", "ノ")]
    meta = build_context_metadata(segments, chars)
    assert meta == {
        (3, 5, "きせん"): CONTEXT_METADATA,
        (3, 4, "き"): CONTEXT_METADATA,
        (4, 5, "せん"): CONTEXT_METADATA,
    }


def test_build_context_metadata_ignores_kana_only_and_kanaless():
    from g2pflow.converters.kashi.context import build_context_metadata
    chars = {"船": frozenset({"せん"})}
    assert build_context_metadata([Segment(0, 3, "さよなら")], chars) == {}
    assert build_context_metadata(
        [Segment(0, 2, "汽船", "", "")], chars) == {}


def test_build_edges_applies_context_metadata_before_variants():
    from g2pflow.converters.kashi.context import CONTEXT_METADATA
    from g2pflow.converters.kashi.lexicon import UnifiedLexicon, build_edges
    pack = UnifiedLexicon({
        "済み": (("すみ", "yomogi_dict", 0.6, 0.75),),
        "あ": (("あ", "copy", -1.0, 0.0),),
    })
    meta = {(1, 3, "すみ"): CONTEXT_METADATA}
    edges = {(e.start, e.end, e.reading): e
             for e in build_edges("あ済み", pack, meta)}
    word = edges[(1, 3, "すみ")]
    assert (word.source, word.prior, word.confidence) == \
        ("pyopenjtalk", 0.75, 0.9)
    variant = edges[(1, 3, "ずみ")]
    # the rendaku variant inherits the contextual metadata
    assert variant.source == "pyopenjtalk"
    assert variant.prior == pytest.approx(0.35)
    assert variant.confidence == pytest.approx(0.765)
    plain = {(e.start, e.end, e.reading): e
             for e in build_edges("あ済み", pack)}
    assert plain[(1, 3, "すみ")].source == "yomogi_dict"
