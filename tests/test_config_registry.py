from pathlib import Path

import pytest
from pydantic import ValidationError

from g2pflow import (
    Converter, ConverterConfig, G2PPipelineConfig, Language, Preprocessor,
    PreprocessorConfig, build_converter_from_config, build_pipeline_from_config,
    build_preprocessor_from_config, converter, get_converter, get_preprocessor,
    list_converters, list_preprocessors, preprocessor,
)
from g2pflow.registry import parse_language


def test_public_configs_are_independent_and_have_fresh_defaults():
    first, second = G2PPipelineConfig(), G2PPipelineConfig()
    first.converters.append(ConverterConfig(id="passthrough"))
    assert second.model_dump() == {"preprocessors": [], "converters": []}
    assert first.model_dump()["converters"] == [
        {"id": "passthrough", "language": None, "kwargs": {}},
    ]
    assert "plugin_paths" not in G2PPipelineConfig.model_fields
    assert not hasattr(first, "resolve")
    a, b = ConverterConfig(id="a"), ConverterConfig(id="b")
    a.kwargs["x"] = 1
    assert b.kwargs == {}


@pytest.mark.parametrize("model,data", [
    (ConverterConfig, {}), (PreprocessorConfig, {}),
    (ConverterConfig, {"id": "x", "kwargs": []}),
    (ConverterConfig, {"id": "x", "language": []}),
    (G2PPipelineConfig, {"converters": [{}]}),
])
def test_invalid_configuration(model, data):
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_builtin_components_are_discovered():
    assert set(list_converters()) == {
        "characters", "chinese-pinyin", "dictionary", "japanese-kana",
        "japanese-mecab", "lstm", "passthrough", "yue-jyutping",
    }
    assert set(list_preprocessors()) == {
        "filter-punctuation", "lowercase", "strip-whitespace", "remove-accents",
    }


def test_registration_and_duplicate_ids():
    @converter(id="custom", language="en, eng")
    class Custom(Converter):
        def find(self, text):
            return None

        def convert(self, text):
            return []

    @preprocessor(id="custom")
    class Processor(Preprocessor):
        def process(self, fragments):
            return fragments

    assert get_converter("custom") is Custom
    assert get_preprocessor("custom") is Processor
    assert Custom.language == ("en", "eng")
    with pytest.raises(ValueError, match="already registered"):
        converter(id="custom")(Custom)
    with pytest.raises(ValueError, match="already registered"):
        preprocessor(id="custom")(Processor)
    with pytest.raises(KeyError):
        get_converter("not-registered")


@pytest.mark.parametrize("value,expected", [
    (None, None), (Language.ANY, Language.ANY),
    ("zh, cmn", ("zh", "cmn")), ("", ("",)),
])
def test_language_parsing(value, expected):
    assert parse_language(value) == expected


def test_language_override_is_instance_local():
    first = build_converter_from_config(ConverterConfig(id="passthrough", language="en"))
    second = build_converter_from_config(ConverterConfig(id="passthrough"))
    assert first.language == ("en",)
    assert second.language is Language.ANY
    assert get_converter("passthrough").language is Language.ANY


def test_nested_path_resolution_preserves_configuration(tmp_path):
    @converter(id="capture")
    class Capture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    config = ConverterConfig(id="capture", kwargs={
        "direct": "@a.txt", "nested": [{"path": "@../b.txt"}, 3, None],
        "relative": "plain.txt",
    })
    before = config.model_dump()
    result = build_converter_from_config(config, root_path=tmp_path)
    assert result.kwargs == {
        "direct": str((tmp_path / "a.txt").resolve()),
        "nested": [{"path": str((tmp_path / "../b.txt").resolve())}, 3, None],
        "relative": "plain.txt",
    }
    assert config.model_dump() == before


def test_preprocessor_kwargs_and_paths(tmp_path):
    @preprocessor(id="capture")
    class Capture:
        def __init__(self, path):
            self.path = path

    result = build_preprocessor_from_config(
        PreprocessorConfig(id="capture", kwargs={"path": "@file.txt"}), root_path=tmp_path,
    )
    assert Path(result.path) == tmp_path / "file.txt"


def test_factory_builds_pipeline_and_cwd_relative_resources(dictionary, monkeypatch):
    path = dictionary("hello\thh eh l ow\n")
    monkeypatch.chdir(path.parent)
    config = G2PPipelineConfig.model_validate({
        "preprocessors": [{"id": "lowercase"}],
        "converters": [{"id": "dictionary", "language": "en",
                        "kwargs": {"dict_path": path.name}}],
    })
    result = build_pipeline_from_config(config, root_path="unused").convert("HELLO")
    assert result[0].text == "hello"
    assert result[0].language == "en"
    assert result[0].readings[0].paths[0][0].phonemes == ["hh", "eh", "l", "ow"]
