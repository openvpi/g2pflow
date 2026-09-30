from pathlib import Path
from typing import Any

from .config import (
    ConverterConfig,
    G2PPipelineConfig,
    PreprocessorConfig,
)
from .converters.base import Converter
from .pipeline import G2PPipeline
from .preprocessors.base import Preprocessor
from .registry import get_converter, get_preprocessor, parse_language


def _resolve_path_refs(obj: Any, root: Path) -> Any:
    if isinstance(obj, str) and obj.startswith("@"):
        return str((root / obj[1:]).resolve())
    if isinstance(obj, dict):
        return {k: _resolve_path_refs(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_path_refs(v, root) for v in obj]
    return obj


def build_preprocessor_from_config(
    config: PreprocessorConfig, *, root_path: str | Path = ""
) -> Preprocessor:
    cls = get_preprocessor(config.id)
    kwargs = _resolve_path_refs(config.kwargs, Path(root_path))
    return cls(**kwargs)


def build_converter_from_config(
    config: ConverterConfig, *, root_path: str | Path = ""
) -> Converter:
    cls: type[Converter] = get_converter(config.id)
    kwargs = _resolve_path_refs(config.kwargs, Path(root_path))
    instance = cls(**kwargs)

    if config.language is not None:
        instance.language = parse_language(config.language)

    return instance


def build_pipeline_from_config(
    config: G2PPipelineConfig, *, root_path: str | Path = ""
) -> G2PPipeline:
    root = Path(root_path)
    return G2PPipeline(
        preprocessors=[build_preprocessor_from_config(pc, root_path=root) for pc in config.preprocessors],
        converters=[build_converter_from_config(cc, root_path=root) for cc in config.converters],
    )
