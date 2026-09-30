"""Paradigm base classes for G2P converters."""

from abc import ABC, abstractmethod

from .base import Converter, G2PGroup, G2PPath, G2PWord, G2PReading, compose_word
from .text import split_words


class LexiconConverter(Converter, ABC):
    """Converter for languages where the writing system doubles as the
    pronunciation script (most alphabetical languages).

    Known words are looked up in a pronunciation dictionary loaded from
    *dict_path*.  Out-of-vocabulary words are handled by
    ``_infer_oov``, which subclasses implement with language-specific
    letter-to-sound rules.
    """

    def __init__(self, dict_path: str | None = None) -> None:
        super().__init__()
        self._dict: dict[str, list[list[str]]] = {}
        if dict_path is not None:
            from .dictionary import load_pronunciation_dict
            self._dict = load_pronunciation_dict(dict_path)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def infer_oov(self, token: str) -> list[list[str]]:
        """Infer phoneme sequences for an out-of-vocabulary token.

        Called when *token* is not found in the pronunciation dictionary.
        Returns one or more alternative phoneme sequences.
        """
        ...

    # ------------------------------------------------------------------
    # Converter interface
    # ------------------------------------------------------------------

    def _convert(self, text: str) -> list[G2PWord]:
        return [self._convert_word(token) for token in split_words(text)]

    def accepts_word(self, text: str) -> bool:
        return text in self._dict or super().accepts_word(text)

    def _convert_word(self, text: str) -> G2PWord:
        pronunciations = self._dict.get(text)
        paths = ([list(p) for p in pronunciations] if pronunciations is not None
                 else self.infer_oov(text))
        return G2PWord(text=text, readings=[G2PReading(paths=[
            [G2PGroup(script=text, phonemes=p)] if p else [] for p in paths
        ])])


class PronunciationScriptConverter(Converter, ABC):
    """Converter for writing systems that use a decoupled *pronunciation script*.

    Two-phase convert:

    1. ``text_to_scripts``  --  text tokens are rendered into pronunciation-
       script tokens (pinyin, jyutping, romaji, ...).
    2. ``script_to_paths``  --  each script token is mapped to one or
       more phoneme sequences (typically via dictionary lookup).
    """

    @abstractmethod
    def text_to_scripts(self, words: list[str]) -> list[list[str]]:
        """Convert text tokens to pronunciation-script tokens.

        Each inner list holds the alternative script representations for
        one input token.  A single-reading token has a one-element inner
        list.
        """
        ...

    @abstractmethod
    def script_to_paths(self, script: str) -> list[G2PPath]:
        """Map a reading script to complete paths with group script labels."""
        ...

    def _tokenize(self, text: str) -> list[str]:
        """Split source text into units with individual pronunciation scripts."""
        return split_words(text)

    def _convert(self, text: str) -> list[G2PWord]:
        words = self._tokenize(text)
        scripts_per_token = self.text_to_scripts(words)
        if len(scripts_per_token) != len(words):
            raise ValueError("text_to_scripts must preserve word count.")
        result: list[G2PWord] = []
        for token, scripts in zip(words, scripts_per_token):
            readings: list[G2PReading] = []
            seen_words: set[str] = set()
            for s in scripts:
                if s in seen_words:
                    continue
                seen_words.add(s)
                seen_paths: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
                paths: list[G2PPath] = []
                for path in self.script_to_paths(s):
                    key = tuple((group.script, tuple(group.phonemes)) for group in path)
                    if key not in seen_paths:
                        seen_paths.add(key)
                        paths.append(path)
                readings.append(G2PReading(paths=paths))
            result.append(G2PWord(text=token, readings=readings))
        return result

    def _convert_word(self, text: str) -> G2PWord | None:
        return compose_word(text, self._convert(text))
