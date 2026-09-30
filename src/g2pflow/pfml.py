"""PFML 1.0 fragments, normalized input plans and lossless result serialization."""

from dataclasses import dataclass, field
from typing import Iterable, TypeAlias
from xml.etree import ElementTree as ET

from .converters.base import G2PGroup, G2PPath, G2PReading, G2PWord, normalize_word
from .registry import Language


class PFMLError(ValueError):
    """Malformed XML or a violation of the PFML content model."""


@dataclass
class PFMLText:
    """An automatic conversion request; fixed marks an explicit word boundary."""

    text: str
    language: str | None = None
    fixed: bool = False


PFMLPart: TypeAlias = PFMLText | G2PWord


@dataclass
class PFMLDocument:
    """Normalized input, containing automatic requests and complete results.

    This is a semantic plan, not a lossless XML syntax tree. All inherited
    languages and omitted pronunciation containers are resolved by the parser.
    """

    parts: list[PFMLPart] = field(default_factory=list)


_LEVELS = ("reading", "path", "group", "phoneme")


def _fail(element: ET.Element, message: str) -> None:
    raise PFMLError(f"<{element.tag}>: {message}")


def _attributes(element: ET.Element, allowed: set[str]) -> None:
    unknown = set(element.attrib) - allowed
    if unknown:
        _fail(element, f"unknown attribute(s): {', '.join(sorted(unknown))}")


def _language(element: ET.Element, inherited: str | None) -> str | None:
    value = element.get("language")
    if value is None:
        return inherited
    return value or None


def _leaf(element: ET.Element) -> str:
    if len(element):
        _fail(element, "must contain only character data")
    return element.text or ""


def _children(element: ET.Element) -> list[ET.Element]:
    if (element.text or "").strip() or any((c.tail or "").strip() for c in element):
        _fail(element, "unexpected character data; structured words use the text attribute")
    return list(element)


def _level(elements: list[ET.Element], allowed: tuple[str, ...]) -> str | None:
    if not elements:
        return None
    level = elements[0].tag
    if level not in allowed or any(e.tag != level for e in elements):
        _fail(elements[0], f"expected siblings at one of these levels: {', '.join(allowed)}")
    return level


def _phoneme(element: ET.Element) -> str:
    _attributes(element, {"language", "symbol"})
    content = _leaf(element)
    if "symbol" in element.attrib:
        if content.strip():
            _fail(element, "symbol attribute and phoneme character data cannot be combined")
        value = element.attrib["symbol"]
    else:
        value = content
    if not value or ("symbol" not in element.attrib and any(c.isspace() for c in value)):
        _fail(element, "expected one non-empty phoneme symbol; use multiple <phoneme> elements")
    language = _language(element, None)
    return f"{language}/{value}" if language else value


def _group(element: ET.Element) -> G2PGroup:
    _attributes(element, {"script", "phonemes"})
    children = _children(element)
    _level(children, ("phoneme",))
    if "phonemes" in element.attrib:
        if children:
            _fail(element, "phonemes attribute and <phoneme> children cannot be combined")
        phonemes = element.attrib["phonemes"].split()
    elif children:
        phonemes = [_phoneme(child) for child in children]
    else:
        _fail(element, 'missing final phonemes; use phonemes="" for an empty pronunciation')
    return G2PGroup(element.get("script", ""), phonemes)


def _groups(elements: list[ET.Element]) -> G2PPath:
    level = _level(elements, ("group", "phoneme"))
    if level == "phoneme":
        return [G2PGroup("", [_phoneme(e) for e in elements])]
    return [_group(e) for e in elements]


def _path(element: ET.Element) -> G2PPath:
    _attributes(element, set())
    children = _children(element)
    if not children:
        _fail(element, "missing final phonemes")
    return _groups(children)


def _paths(elements: list[ET.Element]) -> list[G2PPath]:
    level = _level(elements, ("path", "group", "phoneme"))
    if level is None or level == "path":
        return [_path(e) for e in elements]
    return [_groups(elements)]


def _reading(element: ET.Element) -> G2PReading:
    _attributes(element, set())
    children = _children(element)
    if not children:
        _fail(element, "missing final phonemes")
    return G2PReading(_paths(children))


def _readings(elements: list[ET.Element]) -> list[G2PReading]:
    level = _level(elements, _LEVELS)
    if level is None or level == "reading":
        return [_reading(e) for e in elements]
    return [G2PReading(_paths(elements))]


def _word(element: ET.Element, inherited: str | None) -> PFMLPart | None:
    _attributes(element, {"text", "language", "language-kind", "script", "phonemes"})
    children = list(element)
    if any(child.tag == "text" for child in children):
        _fail(element, "use the text attribute instead of a <text> child")
    if children:
        _children(element)
        text = element.get("text", "")
    elif "text" in element.attrib:
        if (element.text or "").strip():
            _fail(element, "text attribute and source character data cannot be combined")
        text = element.attrib["text"]
    else:
        text = element.text or ""

    kind = element.get("language-kind")
    if kind is not None:
        if kind != "any" or "language" in element.attrib:
            _fail(element, 'language-kind="any" requires an absent language attribute')
        language = Language.ANY
    else:
        language = _language(element, inherited)
    if "script" in element.attrib and "phonemes" not in element.attrib:
        _fail(element, "an intermediate script without final phonemes is not allowed")
    if "phonemes" in element.attrib:
        if children:
            _fail(element, "word pronunciation attributes cannot accompany pronunciation children")
        phonemes = element.attrib["phonemes"].split()
        script = element.get("script", "")
        readings = [G2PReading([[G2PGroup(script, phonemes)]])]
    elif children:
        readings = _readings(children)
    else:
        if kind is not None:
            _fail(element, "language-kind requires a direct pronunciation result")
        if not text.strip():
            _fail(element, "an automatic word requires non-empty source text")
        return PFMLText(text, language, fixed=True)
    return normalize_word(G2PWord(text, language, readings))


class _FragmentBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise PFMLError("PFML fragments do not allow document type declarations")

    def pi(self, target, text):
        raise PFMLError("PFML fragments do not allow processing instructions")


def parse_pfml(source: str, *, language: str | None = None) -> PFMLDocument:
    """Parse a PFML fragment without invoking any converter.

    Adjacent phonemes or groups form one implicit word. Standalone readings
    and paths each form a separate word; alternatives require a pronunciation
    parent. Whitespace between content siblings is formatting, while ordinary
    text and scopes end a run. Explicit words establish their own boundaries.
    """
    wrapper = "<_pfml_fragment>"
    try:
        root = ET.fromstring(
            wrapper + source + "</_pfml_fragment>",
            parser=ET.XMLParser(target=_FragmentBuilder()),
        )
    except ET.ParseError as error:
        line, column = error.position
        if line == 1:
            column = max(0, column - len(wrapper))
        raise PFMLError(f"Invalid PFML XML at line {line}, column {column}: {error}") from error
    document = PFMLDocument()
    merge_text = True

    def append_text(text: str | None, current_language: str | None) -> None:
        nonlocal merge_text
        if not text:
            return
        if merge_text and document.parts and isinstance(document.parts[-1], PFMLText):
            previous = document.parts[-1]
            if not previous.fixed and previous.language == current_language:
                previous.text += text
                return
        document.parts.append(PFMLText(text, current_language))
        merge_text = True

    def visit(element: ET.Element, inherited: str | None) -> None:
        nonlocal merge_text
        append_text(element.text, inherited)
        children = list(element)
        index = 0
        while index < len(children):
            child = children[index]
            if child.tag == "scope":
                _attributes(child, {"language"})
                visit(child, _language(child, inherited))
            elif child.tag == "word":
                merge_text = False
                word = _word(child, inherited)
                if word is not None:
                    document.parts.append(word)
            elif child.tag in _LEVELS:
                merge_text = False
                run = [child]
                while (child.tag in ("group", "phoneme")
                       and index + 1 < len(children)
                       and not (children[index].tail or "").strip()
                       and children[index + 1].tag == child.tag):
                    index += 1
                    run.append(children[index])
                readings = _readings(run)
                word = normalize_word(G2PWord("", inherited, readings))
                if word is not None:
                    document.parts.append(word)
                child = children[index]
            else:
                _fail(child, "unknown element in a PFML text scope")
            append_text(child.tail, inherited)
            index += 1

    visit(root, language or None)
    return document


def _xml_string(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("PFML text, language, script and phonemes must be strings")
    if any(not (c in "\t\n\r" or 0x20 <= ord(c) <= 0xD7FF
                or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF)
           for c in value):
        raise PFMLError("A result contains a character that XML 1.0 cannot represent")
    return value


def to_pfml(words: Iterable[G2PWord]) -> str:
    """Serialize normalized results as ordinary direct PFML.

    Silent branches are omitted and empty labels are filled. All viable
    candidates, their order, grouping and language markers survive a round trip.
    """
    fragments: list[str] = []
    for source_word in words:
        word = normalize_word(source_word)
        if word is None:
            continue
        element = ET.Element("word", {"text": _xml_string(word.text)})
        if word.language is Language.ANY:
            element.set("language-kind", "any")
        elif word.language is not None:
            element.set("language", _xml_string(word.language))
        else:
            element.set("language", "")
        for reading in word.readings:
            reading_element = ET.SubElement(element, "reading")
            for path in reading.paths:
                path_element = ET.SubElement(reading_element, "path")
                for group in path:
                    group_element = ET.SubElement(path_element, "group", {
                        "script": _xml_string(group.script),
                    })
                    for phoneme in group.phonemes:
                        ET.SubElement(group_element, "phoneme", {"symbol": _xml_string(phoneme)})
        # XML normalizes literal carriage returns, but not character references.
        fragments.append(ET.tostring(element, encoding="unicode").replace("\r", "&#13;"))
    return "\n".join(fragments)
