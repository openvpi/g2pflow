"""Fixed word segmentation (UniDic boundaries) for the kashi-g2p converter."""

from __future__ import annotations

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
        if (index + 1 < len(segments) and _is_numeral_surface(current.surface)
                and _is_counter_surface(segments[index + 1].surface)):
            following = segments[index + 1]
            merged.append(Segment(current.start, following.end,
                                  text[current.start:following.end]))
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


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
            self.tagger = fugashi.Tagger(f'-r nul -d "{dicdir.as_posix()}"')
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
            self.tagger = fugashi.Tagger(
                f'-r "{(dicdir / "mecabrc").as_posix()}" '
                f'-d "{dicdir.as_posix()}"')

    def segment(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        cursor = 0
        for node in self.tagger(text):
            surface = str(node.surface)
            if not surface:
                continue
            end = cursor + len(surface)
            segments.append(Segment(cursor, end, surface))
            cursor = end
        if cursor != len(text):
            raise ValueError(
                "UniDic surfaces do not concatenate to the input text")
        return merge_numeral_counters(segments, text)
