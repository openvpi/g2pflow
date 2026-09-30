"""Composable multilingual grapheme-to-phoneme pipelines."""

from .api import (
    build_converter_from_config,
    build_pipeline_from_config,
    build_preprocessor_from_config,
)
from .config import ConverterConfig, G2PPipelineConfig, PreprocessorConfig
from .converters.base import (
    Converter, G2PConversionError, G2PGroup, G2PPath, G2PReading, G2PWord,
)
from .discovery import discover_plugins, register_plugin_paths
from .pipeline import G2PPipeline
from .preprocessors.base import Preprocessor
from .registry import (
    Language,
    converter,
    get_converter,
    get_preprocessor,
    list_converters,
    list_preprocessors,
    preprocessor,
)

__all__ = [
    "Converter", "ConverterConfig", "G2PConversionError", "G2PGroup",
    "G2PPath", "G2PPipeline", "G2PPipelineConfig", "G2PReading", "G2PWord",
    "Language", "Preprocessor", "PreprocessorConfig",
    "build_converter_from_config", "build_pipeline_from_config",
    "build_preprocessor_from_config", "converter", "discover_plugins",
    "get_converter", "get_preprocessor", "list_converters",
    "list_preprocessors", "preprocessor", "register_plugin_paths",
]
