"""Faithful port of cpp-pinyin G2P engine.

The Mandarin and Cantonese converters in ``chinese.py`` delegate to
``PinyinEngine``. This package also exposes the tone conversion utilities.
"""

from .engine import PinyinEngine
from .tones import apply_tone, STYLE_TONE3, STYLE_NORMAL, STYLE_TONE2, STYLE_FIRST_LETTER, STYLE_BOPOMOFO

__all__ = [
    "PinyinEngine",
    "apply_tone",
    "STYLE_TONE3",
    "STYLE_NORMAL",
    "STYLE_TONE2",
    "STYLE_FIRST_LETTER",
    "STYLE_BOPOMOFO",
]
