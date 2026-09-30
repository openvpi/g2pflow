import pickle

import pytest

from g2pflow import G2PPipelineConfig, build_pipeline_from_config


pytestmark = pytest.mark.integration


def flatten(word):
    return [[phone for group in path for phone in group.phonemes]
            for reading in word.readings for path in reading.paths]


@pytest.mark.parametrize("language,identifier,filename,text,expected", [
    ("zh", "chinese-pinyin", "ds-zh-pinyin-lite.txt", "你好", [["n", "i"], ["h", "ao"]]),
    ("yue", "yue-jyutping", "jyutping_dict.txt", "你好", [["n", "ei"], ["h", "ou"]]),
    ("ja", "japanese-kana", "japanese_dict_full.txt", "きゃっと", [["ky", "a"], ["cl"], ["t", "o"]]),
    ("ja", "japanese-kana", "japanese_dict_full.txt", "ぢゃぢゅぢぇぢょづぁづぉ",
     [["j", "a"], ["j", "u"], ["j", "e"], ["j", "o"], ["z", "a"], ["z", "o"]]),
    ("en", "dictionary", "ds_cmudict-07b.txt", "hello world", [["hh", "ax", "l", "ow"], ["w", "er", "l", "d"]]),
])
def test_real_dictionaries(external_resource, language, identifier, filename, text, expected):
    dictionary = external_resource("G2PFLOW_TEST_DICT_DIR") / filename
    config = G2PPipelineConfig.model_validate({"converters": [{
        "id": identifier, "language": language, "kwargs": {"dict_path": str(dictionary)},
    }]})
    words = build_pipeline_from_config(config).convert(text, languages=[language])
    assert [flatten(word)[0] for word in words] == expected


def test_real_mecab_and_pickle(external_resource):
    dictionary = external_resource("G2PFLOW_TEST_DICT_DIR") / "japanese_dict_full.txt"
    unidic = external_resource("G2PFLOW_TEST_UNIDIC_DIR")
    config = G2PPipelineConfig.model_validate({"converters": [{
        "id": "japanese-mecab", "kwargs": {"dict_path": str(dictionary), "unidic_dir": str(unidic)},
    }]})
    pipeline = build_pipeline_from_config(config)
    words = pipeline.convert("猫が好きです", languages=["ja"])
    assert [word.text for word in words] == ["猫", "が", "好き", "です"]
    assert flatten(words[0])[0] == ["n", "e", "k", "o"]
    assert pickle.loads(pickle.dumps(pipeline)).convert("猫が好きです", languages=["ja"]) == words


def test_real_onnx_and_pickle(external_resource):
    dictionary = external_resource("G2PFLOW_TEST_DICT_DIR") / "ds_cmudict-07b.txt"
    model = external_resource("G2PFLOW_TEST_LSTM_MODEL")
    config = G2PPipelineConfig.model_validate({"converters": [{
        "id": "lstm", "language": "en", "kwargs": {
            "dict_path": str(dictionary), "model_path": str(model), "beam_size": 4,
        },
    }]})
    pipeline = build_pipeline_from_config(config)
    words = pipeline.convert("gorpingly", languages=["en"])
    paths = flatten(words[0])
    assert paths[0] == ["g", "ao", "r", "p", "ih", "ng", "l", "iy"]
    assert len(paths) == len({tuple(path) for path in paths}) == 4
    assert pickle.loads(pickle.dumps(pipeline)).convert("gorpingly", languages=["en"]) == words
