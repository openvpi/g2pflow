"""Fixed word segmentation (UniDic boundaries) for the kashi-g2p converter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_COUNTER_HEADS = frozenset(
    "年月日時分秒人個本枚台番曲目匹通件点度階杯冊倍段歳才回歩")


@dataclass(frozen=True, slots=True)
class Segment:
    """One fixed word span in the input text."""

    start: int
    end: int
    surface: str
    pron: str = ""
    kana: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start


def _is_numeral_surface(surface: str) -> bool:
    return bool(surface) and all(char.isdigit() for char in surface)


def _is_counter_surface(surface: str) -> bool:
    return bool(surface) and (surface[0] in _COUNTER_HEADS
                              or surface[0] == "つ")


def merge_numeral_counters(segments: list[Segment],
                           text: str) -> list[Segment]:
    """Merge numeral + counter segments so numbers keep one reading unit."""
    merged: list[Segment] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        if (index + 1 < len(segments)
                and current.end == segments[index + 1].start
                and _is_numeral_surface(current.surface)
                and _is_counter_surface(segments[index + 1].surface)):
            following = segments[index + 1]
            pron = (current.pron + following.pron
                    if current.pron and following.pron else "")
            kana = (current.kana + following.kana
                    if current.kana and following.kana else "")
            merged.append(Segment(current.start, following.end,
                                  text[current.start:following.end],
                                  pron, kana))
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _tagger_args(dicdir: Path) -> str:
    rc = dicdir / "mecabrc"
    rc_path = rc if rc.is_file() else Path(os.devnull)
    return f'-r "{rc_path.as_posix()}" -d "{dicdir.as_posix()}"'


class UnidicSegmenter:
    """UniDic (MeCab) word boundaries via fugashi."""

    name = "unidic"

    def __init__(self, unidic_dir: str | Path | None = None):
        import fugashi

        if unidic_dir:
            dicdir = Path(unidic_dir)
            if not dicdir.exists():
                raise FileNotFoundError(
                    f"UniDic dictionary does not exist: {dicdir}")
            self.tagger = fugashi.Tagger(_tagger_args(dicdir))
        else:
            import unidic

            dicdir = Path(unidic.DICDIR)
            required = (dicdir / "sys.dic", dicdir / "mecabrc")
            if not all(path.is_file() for path in required):
                from filelock import FileLock
                from unidic.download import download_version

                dicdir.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(str(dicdir.parent / "download.lock")):
                    if not all(path.is_file() for path in required):
                        download_version()
            self.tagger = fugashi.Tagger(_tagger_args(dicdir))

    def whole_word_readings(self, text: str, nbest: int = 8) -> list[str]:
        """UniDic prons for analyses that keep ``text`` as a single word."""
        readings: list[str] = []
        seen: set[str] = set()
        for nodes in self.tagger.nbestToNodeList(text, nbest):
            if len(nodes) != 1 or str(nodes[0].surface) != text:
                continue
            pron = str(getattr(nodes[0].feature, "pron", "") or "")
            if not pron or pron in ("*", "-") or pron in seen:
                continue
            seen.add(pron)
            readings.append(pron)
        return readings

    def segment(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        cursor = 0
        for node in self.tagger(text):
            surface = str(node.surface)
            if not surface:
                continue
            start = text.find(surface, cursor)
            if start < 0:
                raise ValueError(
                    f"UniDic surface {surface!r} is not present in the input")
            if start > cursor:
                # MeCab drops whitespace nodes; keep each gap as its own segment.
                segments.append(Segment(cursor, start, text[cursor:start]))
            pron = str(getattr(node.feature, "pron", "") or "")
            if pron in ("*", "-"):
                pron = ""
            kana = str(getattr(node.feature, "kana", "") or "")
            if kana in ("*", "-"):
                kana = ""
            end = start + len(surface)
            segments.append(Segment(start, end, surface, pron, kana))
            cursor = end
        if cursor < len(text):
            segments.append(Segment(cursor, len(text), text[cursor:]))
        return merge_numeral_counters(segments, text)
