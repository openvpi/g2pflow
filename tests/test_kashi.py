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
        return [Segment(start, end, surface)
                for start, end, surface in self.segments_by_text.get(text, [])]


class _FakeRuntime:
    def __init__(self, words_by_text):
        self.words_by_text = words_by_text
        self.calls = []

    def predict_words(self, text, segments, *, beam_size=16):
        self.calls.append((text, len(segments), beam_size))
        return [{"surface": surface, "start": 0, "end": len(surface),
                 "candidates": [{"reading": reading, "score": -float(index)}
                                for index, reading in enumerate(
                                    readings[:beam_size])]}
                for surface, readings in self.words_by_text.get(text, [])]


def phones(word):
    return [[phone for group in path for phone in group.phonemes]
            for reading in word.readings for path in reading.paths]


@pytest.fixture
def toshi_dict(dictionary):
    return dictionary(
        "a\ta\ni\ti\nu\tu\ne\te\no\to\nzu\tz u\nshi\tsh i\n"
        "hi\th i\nga\tg a\nne\tn e\nko\tk o\n", "toshi.txt",
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
    converter._segmenter = _FakeSegmenter({"キョウ": [(0, 3, "キョウ")]})
    converter._runtime = _FakeRuntime({"キョウ": [("キョウ", ["きょう"])]})
    words = converter.convert("キョウ")
    assert len(words) == 1
    # きょう -> きょー -> kyo (ー dropped), same as MeCab's UniDic pron キョー
    assert phones(words[0]) == [["k", "y", "o"]]


def test_unreadable_kana_falls_back_to_kana_units(kashi_converter):
    # copy-only reading of a kana surface: split into kana units like MeCab
    kashi_converter._segmenter = _FakeSegmenter({"キャット": [(0, 4, "キャット")]})
    kashi_converter._runtime = _FakeRuntime({"キャット": [("キャット", ["キャット"])]})
    words = kashi_converter.convert("キャット")
    assert [word.text for word in words] == ["キャ", "ッ", "ト"]


def test_unreadable_kanji_raises_like_mecab(kashi_converter):
    kashi_converter._segmenter = _FakeSegmenter({"珍": [(0, 1, "珍")]})
    kashi_converter._runtime = _FakeRuntime({"珍": [("珍", ["珍"])]})
    with pytest.raises(G2PConversionError):
        kashi_converter.convert("珍")


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
