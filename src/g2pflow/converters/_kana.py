"""Parse kana sequences into source spans and romaji dictionary keys."""

from dataclasses import dataclass

from .text import is_kana


# Kana-to-romaji mappings adapted from cpp-kana's kanaToRomajiMap.
_KANA_TO_ROMAJI = {
    "っ": "cl",
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "o", "ん": "n",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "きぇ": "kye",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo", "ぎぇ": "gye",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo", "じぇ": "je",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "にぇ": "nye",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo", "ひぇ": "hye",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo", "びぇ": "bye",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo", "ぴぇ": "pye",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "みぇ": "mye",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo", "りぇ": "rye",
    "いぇ": "ye",
    "うぁ": "wa", "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "くぁ": "kwa", "くぃ": "kwi", "くぇ": "kwe", "くぉ": "kwo",
    "ぐぁ": "gwa", "ぐぃ": "gwi", "ぐぇ": "gwe", "ぐぉ": "gwo",
    "くゎ": "kwa", "ぐゎ": "gwa",
    "すぁ": "swa", "すぃ": "swi", "すぇ": "swe", "すぉ": "swo",
    "ずぁ": "zwa", "ずぃ": "zwi", "ずぇ": "zwe", "ずぉ": "zwo",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
    "てぃ": "ti", "てゅ": "tyu", "でぃ": "di", "でゅ": "dyu",
    "とぅ": "tu", "どぅ": "du",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "ぶぁ": "bwa", "ぶぃ": "bwi", "ぶぇ": "bwe", "ぶぉ": "bwo",
    "ぷぁ": "pwa", "ぷぃ": "pwi", "ぷぇ": "pwe", "ぷぉ": "pwo",
    "ぬぁ": "nwa", "ぬぃ": "nwi", "ぬぇ": "nwe", "ぬぉ": "nwo",
    "むぁ": "mwa", "むぃ": "mwi", "むぇ": "mwe", "むぉ": "mwo",
    "るぁ": "rwa", "るぃ": "rwi", "るぇ": "rwe", "るぉ": "rwo",
    "ゔ": "vu", "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
    "ヷ": "va", "ヸ": "vi", "ヹ": "ve", "ヺ": "vo",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢぇ": "je", "ぢょ": "jo",
    "づぁ": "za", "づぉ": "zo",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa", "ゕ": "ka", "ゖ": "ke",
    "ゟ": "yo ri", "ヿ": "ko to",
    "ー": "", "゜": "",
}
_SCRIPTS = {kana: tuple(romaji.split()) for kana, romaji in _KANA_TO_ROMAJI.items()}
_MAX_KANA_LENGTH = max(map(len, _SCRIPTS))
_VOICED = dict(zip("かきくけこさしすせそたちつてとはひふへほうわゐゑを",
                   "がぎぐげござじずぜぞだぢづでどばびぶべぼゔヷヸヹヺ"))
_UNVOICED = {voiced: plain for plain, voiced in _VOICED.items()}
_UNVOICED.update(zip("ぱぴぷぺぽ", "はひふへほ"))
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")


@dataclass(frozen=True)
class KanaUnit:
    """A source span whose dictionary keys must be interpreted together."""

    start: int
    end: int
    scripts: tuple[str, ...]


@dataclass(frozen=True)
class KanaReading:
    """A kana spelling and, when known, its explicit pronunciation."""

    kana: str
    pronunciation: str | None = None


@dataclass(frozen=True)
class KanaSegment:
    """A backend word span with an optional contextual UniDic reading."""

    start: int
    end: int
    surface: str
    pron: str = ""
    kana: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start


def _hiragana(text: str) -> str:
    return "".join(
        chr(ord(char) - 0x60) if 0x30A1 <= ord(char) <= 0x30F6
        else {"ヽ": "ゝ", "ヾ": "ゞ"}.get(char, char)
        for char in text
    )


def parse_kana(text: str, *, double_written_sokuon: bool = False) -> list[KanaUnit]:
    """Parse one normalized kana sequence, retaining contextual dependencies."""
    kana = _hiragana(text)
    units: list[KanaUnit] = []
    previous = ""
    previous_index = 0
    index = 0
    while index < len(kana):
        char = kana[index]
        if char in "ゝゞ":
            if not previous:
                raise KeyError(f"Kana iteration mark has no preceding syllable at {index}: {text!r}")
            plain = _UNVOICED.get(previous[0], previous[0]) + previous[1:]
            if char == "ゞ":
                voiced = _VOICED.get(plain[0])
                if voiced is None:
                    raise KeyError(f"Kana syllable cannot be voiced: {previous!r}")
                plain = voiced + plain[1:]
            first = units[previous_index]
            units[previous_index:] = [KanaUnit(first.start, index + 1,
                                               first.scripts + _SCRIPTS[plain])]
            previous = plain
            index += 1
            continue
        for size in range(min(_MAX_KANA_LENGTH, len(kana) - index), 0, -1):
            token = kana[index:index + size]
            if token in _SCRIPTS:
                break
        else:
            raise KeyError(f"Unsupported kana at {index}: {text[index]!r}")
        scripts = _SCRIPTS[token]
        units.append(KanaUnit(index, index + size, scripts))
        if scripts:
            previous = {"ゟ": "り", "ヿ": "と"}.get(token, token)
            previous_index = len(units) - 1
        index += size

    if not double_written_sokuon:
        return units
    result: list[KanaUnit] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if unit.scripts == ("cl",):
            following = index + 1
            while following < len(units) and not units[following].scripts:
                following += 1
            if following < len(units):
                next_unit = units[following]
                first, *rest = next_unit.scripts
                if first != "cl" and first[0] in _CONSONANTS:
                    result.append(KanaUnit(unit.start, next_unit.end,
                                           (first[0] + first, *rest)))
                    index = following + 1
                    continue
        result.append(unit)
        index += 1
    return result


def prepare_kana_segments(
    segments: list[KanaSegment], *, double_written_sokuon: bool = False,
) -> list[KanaSegment]:
    """Join source dependencies before querying readings for the resulting words."""
    prepared: list[KanaSegment] = []
    begin = 0
    while begin < len(segments):
        if not all(is_kana(char) for char in segments[begin].surface):
            prepared.append(segments[begin])
            begin += 1
            continue
        end = begin + 1
        while (end < len(segments)
               and segments[end - 1].end == segments[end].start
               and all(is_kana(char) for char in segments[end].surface)):
            end += 1
        run = segments[begin:end]
        try:
            units = parse_kana(
                "".join(segment.surface for segment in run),
                double_written_sokuon=double_written_sokuon,
            )
        except KeyError:
            # Backend readings may resolve an unrecognized source spelling.
            prepared.extend(run)
        else:
            offset = 0
            unit_index = 0
            for segment in run:
                while unit_index < len(units) and units[unit_index].end <= offset:
                    unit_index += 1
                if (unit_index < len(units)
                        and units[unit_index].start < offset < units[unit_index].end):
                    previous = prepared[-1]
                    # Boundary changes invalidate per-word pronunciation metadata.
                    prepared[-1] = KanaSegment(
                        previous.start, segment.end, previous.surface + segment.surface,
                    )
                else:
                    prepared.append(segment)
                offset += len(segment.surface)
        begin = end
    return prepared
