from ..registry import Language, converter
from .base import Converter, G2PGroup, G2PWord, G2PReading
from .text import split_words, word_spans


@converter(id="passthrough", language=Language.ANY)
class PassthroughConverter(Converter):
    """Catch-all converter that returns each token as its own phoneme.
    Typically placed last in a chain as a fallback for unconverted tokens."""

    def find(self, text: str) -> tuple[int, int] | None:
        return next(word_spans(text), None)

    def _convert(self, text: str) -> list[G2PWord]:
        return [self._convert_word(t) for t in split_words(text)]

    def accepts_word(self, text: str) -> bool:
        return bool(text)

    def _convert_word(self, text: str) -> G2PWord:
        return G2PWord(text=text, readings=[G2PReading(paths=[
            [G2PGroup(script=text, phonemes=[text])],
        ])])


@converter(id="characters", language=Language.ANY)
class CharPhonemeConverter(Converter):
    """One-to-one character-to-phoneme mapping.
    Each character in a token is mapped to one or more phonemes."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def find(self, text: str) -> tuple[int, int] | None:
        for begin, end in word_spans(text):
            if all(c in self._mapping for c in text[begin:end]):
                return begin, end
        return None

    def _convert(self, text: str) -> list[G2PWord]:
        return [self._convert_word(token) for token in split_words(text)]

    def accepts_word(self, text: str) -> bool:
        return bool(text) and all(c in self._mapping for c in text)

    def _convert_word(self, text: str) -> G2PWord:
        phonemes = [phone for char in text for phone in self._mapping[char]]
        path = [G2PGroup(script=text, phonemes=phonemes)] if phonemes else []
        return G2PWord(text=text, readings=[G2PReading(paths=[path])])
