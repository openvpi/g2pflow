from types import SimpleNamespace

import pytest

from g2pflow import G2PConversionError, G2PPipeline, G2PPipelineConfig, build_pipeline_from_config
from g2pflow.converters.japanese import JapaneseKashiConverter, KanaSegment


class ModelDouble:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    def predict_words(self, text, segments, *, beam_size):
        self.calls.append((text, segments, beam_size))
        return [
            {
                "surface": segment.surface, "start": segment.start, "end": segment.end,
                "candidates": [{"reading": reading, "score": -index}
                               for index, reading in enumerate(
                                   self.candidates.get(segment.surface, [])[:beam_size])],
            }
            for segment in segments
        ]


def attach_backend(instance, segments, candidates, *, fixed=()):
    instance._segmenter = SimpleNamespace(
        segment=lambda text: segments,
        whole_word_readings=lambda text: list(fixed),
    )
    instance._runtime = ModelDouble(candidates)
    return instance._runtime


@pytest.fixture
def kashi(tmp_path, kana_dict):
    def create(path=kana_dict, **kwargs):
        return JapaneseKashiConverter(
            str(path), model_path=str(tmp_path / "model.onnx"),
            lexicon_path=str(tmp_path / "lexicon.txz"), **kwargs,
        )
    return create


def scripts(word):
    return [[group.script for group in reading.paths[0]] for reading in word.readings]


def test_model_candidates_and_contextual_readings(kashi, dictionary):
    instance = kashi(dictionary(
        "kyo\tKYO\nko\tKO\nn\tN\nni\tNI\nchi\tCHI\n"
        "wa\tWA\nha\tHA\nne\tNE\n",
    ))
    segments = [
        KanaSegment(0, 2, "今日", "キョー", "キョウ"),
        KanaSegment(2, 3, "は", "ワ", "ハ"),
        KanaSegment(3, 4, "猫", "ネコ", "ネコ"),
    ]
    model = attach_backend(instance, segments, {
        "今日": ["きょう", "こんにち", "今日"],
        "猫": ["ねこ", "ねこ"],
    })
    words = instance.convert("今日は猫")
    assert [word.text for word in words] == ["今日", "は", "猫"]
    assert [scripts(word) for word in words] == [
        [["kyo"], ["ko", "n", "ni", "chi"]], [["wa"], ["ha"]], [["ne", "ko"]],
    ]
    assert model.calls == [("今日は猫", segments, 16)]


def test_source_units_reach_the_model_with_complete_boundaries(kashi, dictionary):
    instance = kashi(dictionary("ha\tHA\nwa\tWA\n"), beam_size=1)
    model = attach_backend(instance, [
        KanaSegment(0, 1, "は", "ワ", "ハ"), KanaSegment(1, 2, "ゝ"),
    ], {})
    word, = instance.convert("はゝ")
    assert word.text == "はゝ"
    assert scripts(word) == [["ha", "ha"]]
    assert model.calls == [("はゝ", [KanaSegment(0, 2, "はゝ")], 1)]


@pytest.mark.parametrize("beam_size", [1, 2])
@pytest.mark.parametrize("whole_word", [False, True])
def test_fixed_words_and_candidate_limits(kashi, dictionary, beam_size, whole_word):
    instance = kashi(dictionary(
        "to\tTO\nkyo\tKYO\na\tA\nzu\tZU\nma\tMA\nmi\tMI\nya\tYA\nko\tKO\n",
    ), beam_size=beam_size)
    model = attach_backend(instance, [
        KanaSegment(0, 2, "東京", "トーキョー", "トウキョウ"),
        KanaSegment(2, 3, "都", "ト", "ト"),
    ], {"東京": ["とうきょう", "あずま"], "都": ["と", "みやこ"]},
        fixed=["トーキョート", "アズマミヤコ"] if whole_word else [])
    word, = G2PPipeline(converters=[instance]).convert_pfml('<word language="ja">東京都</word>')
    assert word.text == "東京都"
    assert word.language == "ja"
    assert len(word.readings) == beam_size
    assert scripts(word)[0] == ["to", "kyo", "to"]
    assert bool(model.calls) is not whole_word


def test_unreadable_model_output_is_a_conversion_error(kashi):
    instance = kashi()
    attach_backend(instance, [KanaSegment(0, 1, "語")], {"語": ["語"]})
    with pytest.raises(G2PConversionError) as error:
        instance.convert("語")
    assert error.value.unconverted_tokens == ["語"]


def test_pipeline_configuration_and_preprocessing(kana_dict, tmp_path):
    pipeline = build_pipeline_from_config(G2PPipelineConfig.model_validate({
        "preprocessors": [{"id": "normalize-unicode"}],
        "converters": [{"id": "japanese-kashi", "language": "ja", "kwargs": {
            "dict_path": str(kana_dict), "model_path": str(tmp_path / "model.onnx"),
            "lexicon_path": str(tmp_path / "lexicon.txz"),
        }}],
    }))
    attach_backend(pipeline._converters[0], [KanaSegment(0, 1, "が", "ガ", "ガ")], {})
    word, = pipeline.convert("か\u3099")
    assert word.text == "が"
    assert word.language == "ja"
    assert word.readings[0].paths[0][0].phonemes == ["g", "a"]
