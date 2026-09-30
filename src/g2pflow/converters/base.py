from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import TypeAlias, final

from ..preprocessors.base import Preprocessor
from ..registry import Language


@dataclass
class G2PGroup:
    """A contiguous pronunciation group and its pronunciation script."""

    script: str
    phonemes: list[str]


G2PPath: TypeAlias = list[G2PGroup]


@dataclass
class G2PReading:
    """Complete legal phoneme realizations of one reading."""

    paths: list[G2PPath] = field(default_factory=list)


@dataclass
class G2PWord:
    """One output word with complete pronunciation candidates.

    Silent units are omitted from results. Fixed PFML words keep their source
    boundaries; otherwise converters determine word text and boundaries.
    """

    text: str
    language: str | Language | None = None
    readings: list[G2PReading] = field(default_factory=list)


class Converter(ABC):
    language: tuple[str, ...] | Language | None = None

    @abstractmethod
    def find(self, text: str) -> tuple[int, int] | None:
        """Return the earliest accepted non-empty slice, or None.

        Offsets use Python string indices in the supplied text. Matching
        does not perform pronunciation inference.
        """
        ...

    # noinspection PyMethodMayBeStatic
    def preprocessors(self) -> list[Preprocessor]:
        """Transform a claimed input run; word text and count may change."""
        return []

    @final
    def convert(self, text: str) -> list[G2PWord]:
        """Convert and normalize a run, retaining only complete non-empty paths."""
        return normalize_words(self._convert(text))

    @abstractmethod
    def _convert(self, text: str) -> list[G2PWord]:
        """Backend hook producing raw words; filtering belongs to the framework."""
        ...

    def accepts_word(self, text: str) -> bool:
        """Whether this converter can claim an entire pre-segmented word.

        Override for units (such as dictionary phrases) that find() normally
        segments. This method must not perform pronunciation inference.
        """
        return bool(text) and self.find(text) == (0, len(text))

    @final
    def convert_word(self, text: str) -> G2PWord | None:
        """Convert and normalize one fixed unit, omitting silent results."""
        word = self._convert_word(text)
        if word is not None and not isinstance(word, G2PWord):
            raise G2PWordBoundaryError(text, "_convert_word() must return a G2PWord or None")
        return normalize_word(word) if word is not None else None

    def _convert_word(self, text: str) -> G2PWord | None:
        """Backend hook for a fixed unit.

        The default supports converters that emit at most one word.
        Backends with internal segmentation must implement this explicitly.
        """
        words = self._convert(text)
        if len(words) > 1:
            raise G2PWordBoundaryError(text, f"{type(self).__name__} returned {len(words)} words")
        return words[0] if words else None


class G2PWordBoundaryError(ValueError):
    def __init__(self, text: str, reason: str) -> None:
        self.text = text
        super().__init__(f"Cannot preserve the fixed word {text!r}: {reason}")


def compose_word(text: str, words: list[G2PWord]) -> G2PWord | None:
    """Compose independent backend units, preserving reading/path nesting.

    Only converters that define their units as independent should call this.
    It is not a general pipeline fallback for a violated word boundary.
    """
    if not words:
        return None
    readings = []
    for combination in product(*(word.readings for word in words)):
        paths = [
            [G2PGroup(group.script, list(group.phonemes)) for path in parts for group in path]
            for parts in product(*(reading.paths for reading in combination))
        ]
        readings.append(G2PReading(paths))
    return G2PWord(text, readings=readings)


class G2PConversionError(Exception):
    def __init__(self, unconverted_tokens: list[str], *, reason: str | None = None) -> None:
        self.unconverted_tokens = unconverted_tokens
        self.reason = reason
        super().__init__(
            f"The following tokens could not be converted "
            f"by any converter in the chain: {unconverted_tokens}"
            + (f" ({reason})" if reason else "")
        )


def normalize_word(word: G2PWord) -> G2PWord | None:
    """Remove silent branches and fill labels without changing viable choices.

    Missing candidate lists are conversion failures. Explicit empty paths or
    groups are silent; they disappear along with any parents left empty.
    The input objects are not mutated.
    """
    if not word.readings:
        raise G2PConversionError([word.text], reason="the result has no readings")
    readings = []
    for reading in word.readings:
        if not reading.paths:
            raise G2PConversionError([word.text], reason="a reading has no candidate paths")
        paths = []
        for path in reading.paths:
            groups = []
            for group in path:
                if not group.phonemes:
                    continue
                if any(not isinstance(phone, str) or not phone for phone in group.phonemes):
                    raise G2PConversionError([word.text], reason="phoneme symbols must be non-empty strings")
                groups.append(G2PGroup(group.script or " ".join(group.phonemes), list(group.phonemes)))
            if groups:
                paths.append(groups)
        if paths:
            readings.append(G2PReading(paths))
    if not readings:
        return None
    text = word.text or " | ".join(
        " ".join(group.script for group in path)
        for reading in readings for path in reading.paths
    )
    return G2PWord(text, word.language or None, readings)


def normalize_words(words: list[G2PWord]) -> list[G2PWord]:
    return [result for word in words if (result := normalize_word(word)) is not None]


def resolve_language(
    language: tuple[str, ...] | Language | None,
    languages: list[str] | None,
) -> str | Language | None:
    """Resolve explicit tags, preserving ANY for vocabulary-based resolution."""
    if language is None:
        return None
    if language is Language.ANY:
        return Language.ANY
    if not languages:
        return language[0]
    for tag in language:
        if tag in languages:
            return tag
    return None
