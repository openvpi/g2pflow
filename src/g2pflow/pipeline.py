from dataclasses import replace

from .converters.base import (
    Converter, G2PConversionError, G2PWord, G2PWordBoundaryError, resolve_language,
)
from .pfml import PFMLDocument, PFMLText, parse_pfml
from .preprocessors.base import Preprocessor
from .registry import Language


class G2PPipeline:
    def __init__(
        self,
        preprocessors: list[Preprocessor] | None = None,
        converters: list[Converter] | None = None,
    ) -> None:
        self._preprocessors = preprocessors or []
        self._converters = converters or []

    def convert(
        self, text: str, *, languages: list[str] | None = None,
    ) -> list[G2PWord]:
        """Convert literal text as one automatic input fragment."""
        return self._convert_document(PFMLDocument([PFMLText(text)]), languages)

    def convert_pfml(
        self, source: str, *, languages: list[str] | None = None,
    ) -> list[G2PWord]:
        """Convert a PFML fragment, preserving all non-silent direct candidates.

        Use an outer scope element to set a default language. Scoped languages
        select only converters explicitly registered/configured for that
        language. The languages filter applies to unscoped automatic text as
        in convert(); explicit language declarations take precedence.
        Complete direct results require neither converters nor preprocessors.
        """
        return self._convert_document(parse_pfml(source), languages)

    def _convert_document(
        self, document: PFMLDocument, languages: list[str] | None,
    ) -> list[G2PWord]:
        language_set = set(languages) if languages else None
        result: list[G2PWord] = []
        for part in document.parts:
            if isinstance(part, G2PWord):
                result.append(part)
                continue
            if not part.text.strip():
                continue
            if part.language is None:
                active = [c for c in self._converters
                          if language_set is None or c.language is None
                          or c.language is Language.ANY
                          or any(tag in language_set for tag in c.language)]
                requested = languages
            else:
                active = [c for c in self._converters
                          if isinstance(c.language, tuple)
                          and part.language in c.language]
                requested = [part.language]
            if not active:
                if part.language is not None:
                    raise ValueError(f"No converter matches the PFML language {part.language!r}.")
                raise ValueError("No converter matches the requested languages.")
            if part.fixed:
                word = self._convert_fixed(part.text, active, requested)
                if word is not None:
                    result.append(word)
            else:
                result.extend(self._convert_text(part.text, active, requested))
        return result

    def _convert_text(
        self, text: str, active: list[Converter], languages: list[str] | None,
    ) -> list[G2PWord]:
        fragments = [text]
        for processor in self._preprocessors:
            fragments = processor.process(fragments)

        claimed: list[tuple[int, str]] = []
        unrecognized: list[str] = []
        for fragment in fragments:
            ranges: list[tuple[int, int, int]] = []
            pending = [(0, len(fragment), 0)]
            while pending:
                begin, end, priority = pending.pop()
                if begin == end:
                    continue
                part = fragment[begin:end]
                for index in range(priority, len(active)):
                    match = active[index].find(part)
                    if match is None:
                        continue
                    left, right = match
                    if not 0 <= left < right <= len(part):
                        raise ValueError(f"{type(active[index]).__name__}.find returned invalid range {match}.")
                    ranges.append((begin + left, begin + right, index))
                    # LIFO order routes the left remainder before the right.
                    pending.append((begin + right, end, index))
                    pending.append((begin, begin + left, index + 1))
                    break
                else:
                    if part.strip():
                        unrecognized.append(part)
            claimed.extend((index, fragment[begin:end]) for begin, end, index in sorted(ranges))

        if unrecognized:
            raise G2PConversionError(unrecognized)

        result: list[G2PWord] = []
        for index, part in claimed:
            converter = active[index]
            parts = [part]
            for processor in converter.preprocessors():
                parts = processor.process(parts)
            language = resolve_language(converter.language, languages)
            for part in parts:
                if part:
                    words = converter.convert(part)
                    for word in words:
                        word.language = language or None
                    result.extend(words)
        return result

    def _convert_fixed(
        self, text: str, active: list[Converter], languages: list[str] | None,
    ) -> G2PWord | None:
        def normalize(value: str, processors: list[Preprocessor]) -> str | None:
            for processor in processors:
                parts = processor.process([value])
                if not parts or parts == [""]:
                    return None
                if len(parts) != 1:
                    raise G2PWordBoundaryError(text, f"{type(processor).__name__} split it")
                value = parts[0]
            return value

        normalized = normalize(text, self._preprocessors)
        if normalized is None:
            return None
        for converter in active:
            if converter.accepts_word(normalized):
                prepared = normalize(normalized, converter.preprocessors())
                if prepared is None:
                    return None
                word = converter.convert_word(prepared)
                if word is None:
                    return None
                return replace(word, text=text, language=resolve_language(converter.language, languages) or None)
        raise G2PConversionError([text])
