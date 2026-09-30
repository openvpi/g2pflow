"""Standalone configuration models for G2P construction."""

from typing import Any

from pydantic import BaseModel, Field


class PreprocessorConfig(BaseModel):
    id: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ConverterConfig(BaseModel):
    id: str
    language: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class G2PPipelineConfig(BaseModel):
    preprocessors: list[PreprocessorConfig] = Field(default_factory=list)
    converters: list[ConverterConfig] = Field(default_factory=list)
