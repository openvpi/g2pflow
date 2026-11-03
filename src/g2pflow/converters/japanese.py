"""Japanese reading backends and shared kana-to-phoneme conversion."""

import importlib.util
from abc import abstractmethod
from itertools import product
from pathlib import Path

from ..registry import converter
from ._kana import KanaReading, KanaSegment, _hiragana, parse_kana, prepare_kana_segments
from .base import (
    Converter, G2PConversionError, G2PGroup, G2PPath, G2PReading, G2PWord,
    compose_word,
)
from .dictionary import load_pronunciation_dict
from .kashi import resources as kashi_resources
from .text import find_run, is_kana


def _kana_only(text: str) -> bool:
    return bool(text) and all(is_kana(char) for char in text)


@converter(id="japanese-kana", language="ja,jpn")
class JapaneseKanaConverter(Converter):
    """Convert literal kana or explicit backend pronunciations to dictionary paths.

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

    def reading(self, text: str | KanaReading) -> G2PReading:
        """Convert a literal spelling or a reading with an explicit pronunciation."""
        if isinstance(text, KanaReading):
            text = text.pronunciation if text.pronunciation is not None else text.kana
        units = parse_kana(text, double_written_sokuon=self._double_written_sokuon)
        return self._scripts_reading(tuple(script for unit in units for script in unit.scripts))

    def _convert(self, text: str) -> list[G2PWord]:
        return [
            G2PWord(text[unit.start:unit.end], readings=[self._scripts_reading(unit.scripts)])
            for unit in parse_kana(text, double_written_sokuon=self._double_written_sokuon)
        ]

    def _convert_word(self, text: str) -> G2PWord | None:
        return G2PWord(text, readings=[self.reading(text)]) if text else None

    def convert_readings(
        self,
        segments: list[KanaSegment],
        candidates: list[list[KanaReading]],
        *,
        include_surface: bool = False,
        max_readings: int | None = None,
    ) -> list[G2PWord]:
        """Resolve backend spellings using matching pronunciations and source spans.

        include_surface keeps the contextual and literal readings of kana words.
        Candidate order is preserved; unsupported candidates are discarded.
        """
        if len(segments) != len(candidates):
            raise ValueError("The backend must return candidates for every prepared segment.")
        rows: list[tuple[KanaSegment, list[str]]] = []
        for segment, readings in zip(segments, candidates):
            if include_surface and _kana_only(segment.surface):
                readings = []
                if _kana_only(segment.pron):
                    readings.append(KanaReading(segment.kana or segment.surface, segment.pron))
                readings.append(KanaReading(segment.surface, pronunciation=segment.surface))
            spellings = []
            for candidate in readings:
                spelling = candidate.kana
                if candidate.pronunciation is not None:
                    spelling = candidate.pronunciation
                elif (segment.kana and _kana_only(segment.pron)
                      and _hiragana(spelling) == _hiragana(segment.kana)):
                    spelling = segment.pron
                if _kana_only(spelling) and spelling not in spellings:
                    spellings.append(spelling)
            rows.append((segment, spellings))
        return self._render_words(rows, max_readings=max_readings)

    def _shares_unit(self, left: str, right: str) -> bool:
        try:
            units = parse_kana(left + right, double_written_sokuon=self._double_written_sokuon)
        except KeyError:
            return False
        return any(unit.start < len(left) < unit.end for unit in units)

    def _render_words(
        self, rows: list[tuple[KanaSegment, list[str]]], *, max_readings: int | None,
    ) -> list[G2PWord]:
        joined: list[tuple[KanaSegment, list[str]]] = []
        for segment, readings in rows:
            if not readings and not _kana_only(segment.surface):
                raise G2PConversionError([segment.surface])
            if joined and joined[-1][0].end == segment.start:
                previous, previous_readings = joined[-1]
                left = previous_readings or [previous.surface]
                right = readings or [segment.surface]
                if any(self._shares_unit(a, b) for a, b in product(left, right)):
                    joined[-1] = (
                        KanaSegment(previous.start, segment.end, previous.surface + segment.surface),
                        list(dict.fromkeys(a + b for a, b in product(left, right))),
                    )
                    continue
            joined.append((segment, readings))
        result: list[G2PWord] = []
        for segment, spellings in joined:
            if not spellings:
                result.extend(self._convert(segment.surface))
                continue
            readings = []
            seen = set()
            for spelling in spellings:
                try:
                    reading = self.reading(spelling)
                except KeyError:
                    continue
                key = tuple(tuple((group.script, tuple(group.phonemes)) for group in path)
                            for path in reading.paths)
                if key in seen:
                    continue
                seen.add(key)
                readings.append(reading)
                if max_readings is not None and len(readings) >= max_readings:
                    break
            if not readings:
                raise G2PConversionError([segment.surface])
            result.append(G2PWord(segment.surface, readings=readings))
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


class JapaneseReadingConverter(Converter):
    """Shared conversion flow for backends that produce Japanese kana readings."""

    def __init__(
        self, dict_path: str, *, double_written_sokuon: bool = False,
        max_readings: int | None = None, include_surface: bool = False,
    ) -> None:
        if max_readings is not None and max_readings < 1:
            raise ValueError("max_readings must be a positive integer.")
        self._double_written_sokuon = double_written_sokuon
        self._max_readings = max_readings
        self._include_surface = include_surface
        self._kana = JapaneseKanaConverter(
            dict_path, double_written_sokuon=double_written_sokuon,
        )

    def find(self, text: str) -> tuple[int, int] | None:
        return find_run(text, _is_japanese_char)

    @abstractmethod
    def _segments(self, text: str) -> list[KanaSegment]:
        """Return source words and any contextual kana/pronunciation metadata."""
        ...

    @abstractmethod
    def _readings(self, text: str, segments: list[KanaSegment]) -> list[list[KanaReading]]:
        """Return ordered candidates for the prepared source words."""
        ...

    @abstractmethod
    def _word_readings(self, text: str) -> list[KanaReading]:
        """Return candidates covering a caller-fixed word, or an empty list."""
        ...

    def _convert(self, text: str) -> list[G2PWord]:
        if not text:
            return []
        segments = prepare_kana_segments(
            [segment for segment in self._segments(text) if segment.surface.strip()],
            double_written_sokuon=self._double_written_sokuon,
        )
        return self._kana.convert_readings(
            segments, self._readings(text, segments),
            include_surface=self._include_surface, max_readings=self._max_readings,
        )

    def _convert_word(self, text: str) -> G2PWord | None:
        if not text:
            return None
        readings = self._word_readings(text)
        if readings:
            words = self._kana.convert_readings(
                [KanaSegment(0, len(text), text)], [readings], max_readings=self._max_readings,
            )
            return words[0]
        return compose_word(text, self._convert(text), max_readings=self._max_readings)


@converter(id="japanese-mecab", language="ja,jpn")
class JapaneseMecabConverter(JapaneseReadingConverter):
    """Segment full Japanese word forms and enumerate whole-word readings.

    MeCab supplies UniDic kana spellings and pronunciations. Source boundaries,
    romaji, dictionary alternatives and phoneme paths use the shared Japanese
    conversion flow.
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
        super().__init__(dict_path, double_written_sokuon=double_written_sokuon)
        self._nbest = nbest
        self._unidic_dir = unidic_dir
        self._tagger = None

    def __getstate__(self):
        # Binarization and data loaders use spawned worker processes. MeCab's
        # native tagger cannot be pickled; each worker creates its own instance.
        state = self.__dict__.copy()
        state["_tagger"] = None
        return state

    def _get_tagger(self):
        if self._tagger is not None:
            return self._tagger
        import fugashi
        import unidic
        from unidic.download import download_version

        if self._unidic_dir is None:
            dicdir = Path(unidic.DICDIR)
            required = (dicdir / "sys.dic", dicdir / "mecabrc")
            if not all(path.is_file() for path in required):
                from filelock import FileLock

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

    def _word_readings(self, text: str) -> list[KanaReading]:
        readings: list[KanaReading] = []
        for nodes in self._get_tagger().nbestToNodeList(text, self._nbest):
            if len(nodes) != 1 or nodes[0].surface != text:
                continue
            feature = nodes[0].feature
            pron = feature.pron
            if not pron or pron in ("*", "-"):
                continue
            kana = getattr(feature, "kana", "") or pron
            reading = KanaReading(kana, pron)
            if reading not in readings:
                readings.append(reading)
        return readings

    def _segments(self, text: str) -> list[KanaSegment]:
        segments: list[KanaSegment] = []
        cursor = 0
        # Snapshot nodes before N-best calls replace MeCab's lattice.
        for node in self._get_tagger()(text):
            surface = node.surface
            start = text.find(surface, cursor)
            if start < 0:
                raise ValueError(f"MeCab surface {surface!r} is not present in the input.")
            cursor = start + len(surface)
            pron = getattr(node.feature, "pron", "") or ""
            kana = getattr(node.feature, "kana", "") or ""
            segments.append(KanaSegment(start, cursor, surface, pron, kana))
        return segments

    def _readings(self, text: str, segments: list[KanaSegment]) -> list[list[KanaReading]]:
        return [self._word_readings(segment.surface) for segment in segments]


@converter(id="japanese-kashi", language="ja,jpn")
class JapaneseKashiConverter(JapaneseReadingConverter):
    """Score kana readings within UniDic word boundaries using kashi-g2p.

    Model and lexicon files are downloaded to a versioned cache on first use
    unless their paths are supplied. beam_size caps the readings per word.
    """

    def __init__(
        self,
        dict_path: str,
        *,
        model_path: str | None = None,
        lexicon_path: str | None = None,
        cache_dir: str | None = None,
        prefer_dml: bool = False,
        beam_size: int = 16,
        unidic_dir: str | None = None,
    ) -> None:
        self._beam_size = max(1, int(beam_size))
        super().__init__(dict_path, max_readings=self._beam_size, include_surface=True)
        cache = Path(cache_dir) if cache_dir else kashi_resources.default_cache_dir() / "kashi"
        cache = cache / kashi_resources.RELEASE_TAG
        self._model_is_managed = model_path is None
        self._lexicon_is_managed = lexicon_path is None
        self._model_path = Path(model_path) if model_path else cache / "model.onnx"
        self._lexicon_path = Path(lexicon_path) if lexicon_path else cache / "lexicon.txz"
        self._prefer_dml = prefer_dml
        self._unidic_dir = unidic_dir
        self._runtime = None
        self._segmenter = None

    def _get_segmenter(self):
        if self._segmenter is None:
            from .kashi.segmentation import UnidicSegmenter

            self._segmenter = UnidicSegmenter(self._unidic_dir)
        return self._segmenter

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        if importlib.util.find_spec("onnxruntime") is None:
            raise ImportError(
                "japanese-kashi requires optional dependencies. Install them with: "
                'python -m pip install "g2pflow[all]"'
            )
        missing = [
            path for path, managed in (
                (self._model_path, self._model_is_managed),
                (self._lexicon_path, self._lexicon_is_managed),
            )
            if not managed and not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "kashi resource file not found: " + ", ".join(str(path) for path in missing)
            )
        self._fetch_managed()
        from .kashi.runtime import G2POnnxRuntime

        self._runtime = G2POnnxRuntime(
            model_path=self._model_path,
            bundle_path=self._lexicon_path,
            prefer_dml=self._prefer_dml,
        )
        return self._runtime

    def _fetch_managed(self) -> None:
        if self._model_is_managed:
            kashi_resources.install(
                kashi_resources.MODEL_URL, self._model_path,
                needed=kashi_resources.model_needs_download,
            )
        if self._lexicon_is_managed:
            kashi_resources.install(
                kashi_resources.LEXICON_URL, self._lexicon_path,
                needed=lambda path: not path.is_file(),
            )

    def _segments(self, text: str) -> list[KanaSegment]:
        return self._get_segmenter().segment(text)

    def _readings(self, text: str, segments: list[KanaSegment]) -> list[list[KanaReading]]:
        return [
            [KanaReading(candidate["reading"]) for candidate in word["candidates"]]
            for word in self._get_runtime().predict_words(text, segments, beam_size=self._beam_size)
        ]

    def _word_readings(self, text: str) -> list[KanaReading]:
        return [KanaReading(pron) for pron in self._get_segmenter().whole_word_readings(text)]

    def __getstate__(self) -> dict:
        # Native handles are recreated in the receiving process.
        state = self.__dict__.copy()
        state["_runtime"] = None
        state["_segmenter"] = None
        return state
