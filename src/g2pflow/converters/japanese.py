"""Japanese kana and MeCab G2P converters with shared pronunciation rules."""

from itertools import product
from pathlib import Path

from ..registry import converter
from ._kana import merge_kana_fragments, parse_kana
from .base import (
    Converter, G2PConversionError, G2PGroup, G2PPath, G2PReading, G2PWord,
    compose_word,
)
from .dictionary import load_pronunciation_dict
from .text import find_run, is_kana


@converter(id="japanese-kana", language="ja,jpn")
class JapaneseKanaConverter(Converter):
    """Parse kana sequences and look up their romaji keys in a phoneme dictionary.

    With double_written_sokuon, a closure and following syllable use one key
    such as "tta". Phonemes always come from the supplied dictionary.
    """

    def __init__(
        self,
        dict_path: str,
        *,
        double_written_sokuon: bool = False,
    ) -> None:
        self._script_dict = load_pronunciation_dict(dict_path)
        self._double_written_sokuon = double_written_sokuon

    def find(self, text: str) -> tuple[int, int] | None:
        return find_run(text, is_kana)

    def script_to_paths(self, script: str) -> list[G2PPath]:
        if script == "":
            return [[]]
        pronunciations = self._script_dict.get(script)
        if pronunciations is None:
            raise KeyError(f"Script token {script!r} not found in script-to-phoneme dict.")
        return [[G2PGroup(script, list(phones))] for phones in pronunciations]

    def _scripts_reading(self, scripts: tuple[str, ...]) -> G2PReading:
        paths: list[G2PPath] = []
        seen = set()
        for parts in product(*(self.script_to_paths(script) for script in scripts)):
            path = [group for part in parts for group in part]
            key = tuple((group.script, tuple(group.phonemes)) for group in path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
        return G2PReading(paths=paths)

    def reading(self, text: str) -> G2PReading:
        """Convert one complete kana reading, preserving dictionary alternatives."""
        units = parse_kana(text, double_written_sokuon=self._double_written_sokuon)
        return self._scripts_reading(tuple(script for unit in units for script in unit.scripts))

    def _convert(self, text: str) -> list[G2PWord]:
        return [
            G2PWord(text[unit.start:unit.end], readings=[self._scripts_reading(unit.scripts)])
            for unit in parse_kana(text, double_written_sokuon=self._double_written_sokuon)
        ]

    def _convert_word(self, text: str) -> G2PWord | None:
        return G2PWord(text, readings=[self.reading(text)]) if text else None

    def _convert_readings(self, words: list[tuple[str, list[str]]]) -> list[G2PWord]:
        """Convert backend candidates, joining only neighbors with kana dependencies."""
        joined: list[tuple[str, list[str]]] = []
        for surface, readings in words:
            if not readings and not all(is_kana(char) for char in surface):
                raise G2PConversionError([surface])
            if joined:
                previous, previous_readings = joined[-1]
                pairs = list(product(previous_readings or [previous], readings or [surface]))
                if any(
                    unit.start < len(left) < unit.end
                    for left, right in pairs
                    for unit in parse_kana(left + right,
                                           double_written_sokuon=self._double_written_sokuon)
                ):
                    joined[-1] = (
                        previous + surface,
                        list(dict.fromkeys(left + right for left, right in pairs)),
                    )
                    continue
            joined.append((surface, readings))
        result: list[G2PWord] = []
        for surface, readings in joined:
            if readings:
                result.append(G2PWord(surface, readings=[self.reading(kana) for kana in readings]))
            else:
                result.extend(self._convert(surface))
        return result

def _is_japanese_char(char: str) -> bool:
    cp = ord(char)
    return (
        is_kana(char)
        or char in "々〆〇"
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2FA1F
        or 0x30000 <= cp <= 0x323AF
    )


@converter(id="japanese-mecab", language="ja,jpn")
class JapaneseMecabConverter(Converter):
    """Segment full Japanese word forms and enumerate whole-word readings.

    MeCab supplies only kana pronunciations (UniDic's ``pron`` field).
    Romaji, phoneme groups, dictionary alternatives and long vowels are
    handled by :class:`JapaneseKanaConverter` without additional rules.
    The full PyPI ``unidic`` dictionary is loaded on first conversion and
    downloaded automatically if it has not been installed yet.
    """

    def __init__(
        self,
        dict_path: str,
        *,
        nbest: int = 32,
        double_written_sokuon: bool = False,
        unidic_dir: str | None = None,
    ) -> None:
        if isinstance(nbest, bool) or not isinstance(nbest, int) or nbest < 1:
            raise ValueError("nbest must be a positive integer.")
        self._nbest = nbest
        self._double_written_sokuon = double_written_sokuon
        self._unidic_dir = unidic_dir
        self._tagger = None
        self._kana = JapaneseKanaConverter(
            dict_path=dict_path, double_written_sokuon=double_written_sokuon,
        )

    def __getstate__(self):
        # Binarization and data loaders use spawned worker processes. MeCab's
        # native tagger cannot be pickled; each worker creates its own instance.
        state = self.__dict__.copy()
        state["_tagger"] = None
        return state

    def find(self, text: str) -> tuple[int, int] | None:
        # In particular, never take ASCII romaji/phoneme input from the
        # Japanese dictionary converter used by existing training datasets.
        return find_run(text, _is_japanese_char)

    def _get_tagger(self):
        if self._tagger is not None:
            return self._tagger
        try:
            import fugashi
            import unidic
            from unidic.download import download_version
        except ModuleNotFoundError as exc:
            if exc.name not in ("fugashi", "unidic"):
                raise
            raise ImportError(
                "japanese-mecab requires optional dependencies. Install them with: "
                'python -m pip install "g2pflow[ja]"'
            ) from exc
        if self._unidic_dir is None:
            dicdir = Path(unidic.DICDIR)
            required = (dicdir / "sys.dic", dicdir / "mecabrc")
            if not all(path.is_file() for path in required):
                try:
                    from filelock import FileLock
                except ModuleNotFoundError as exc:
                    if exc.name != "filelock":
                        raise
                    raise ImportError(
                        'UniDic download requires filelock. Install it with: '
                        'python -m pip install "g2pflow[ja]"'
                    ) from exc
                # Serialize first-use downloads across spawned workers.
                with FileLock(str(dicdir.parent / "download.lock")):
                    if not all(path.is_file() for path in required):
                        download_version()
        else:
            dicdir = Path(self._unidic_dir)
        # Explicit paths avoid system MeCab dictionaries and support spaces
        # in Windows environment paths.
        dicdir = dicdir.resolve()
        self._tagger = fugashi.Tagger(
            f'-r "{(dicdir / "mecabrc").as_posix()}" -d "{dicdir.as_posix()}"'
        )
        return self._tagger

    def _pronunciations(self, word: str) -> list[str]:
        readings: list[str] = []
        seen: set[str] = set()
        for nodes in self._get_tagger().nbestToNodeList(word, self._nbest):
            if len(nodes) != 1 or nodes[0].surface != word:
                continue
            pron = nodes[0].feature.pron
            if not pron or pron in ("*", "-") or pron in seen:
                continue
            seen.add(pron)
            readings.append(pron)
        return readings

    def _convert_word(self, text: str) -> G2PWord | None:
        pronunciations = self._pronunciations(text)
        if pronunciations:
            return G2PWord(text, readings=[self._kana.reading(pron) for pron in pronunciations])
        return compose_word(text, self._convert(text))

    def _convert(self, text: str) -> list[G2PWord]:
        if not text:
            return []
        # Snapshot surfaces before N-best calls replace MeCab's lattice.
        surfaces = [node.surface for node in self._get_tagger()(text)]
        surfaces = merge_kana_fragments(
            surfaces, double_written_sokuon=self._double_written_sokuon,
        )
        return self._kana._convert_readings([
            (surface, self._pronunciations(surface)) for surface in surfaces
        ])
