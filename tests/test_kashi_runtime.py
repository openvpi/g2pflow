import lzma
from types import SimpleNamespace

import numpy as np
import pytest

from g2pflow.converters._kana import KanaSegment
from g2pflow.converters.kashi.lexicon import Edge, UnifiedLexicon, build_edges
from g2pflow.converters.kashi.runtime import G2POnnxRuntime, _load_bundle, windows_from_segments
from g2pflow.converters.kashi.segmentation import merge_numeral_counters


def runtime(table):
    instance = object.__new__(G2POnnxRuntime)
    instance.lexicon = UnifiedLexicon(table)
    instance.score = lambda text, edges: [edge.prior for edge in edges]
    return instance


def test_candidates_stay_within_word_boundaries():
    instance = runtime({
        "大学": (("だいがく", "jmdict", 0.8, 0.8),),
        "生": (("せい", "jmdict", 0.8, 0.8), ("なま", "jmdict", 0.7, 0.8)),
        "大学生": (("だいがくせい", "jmdict", 0.9, 0.9),),
    })
    seen = []

    def score(text, edges):
        seen.extend(edges)
        return [edge.prior for edge in edges]

    instance.score = score
    words = instance.predict_words("大学生", [
        KanaSegment(0, 2, "大学"), KanaSegment(2, 3, "生"),
    ], beam_size=2)
    assert all(not edge.start < 2 < edge.end for edge in seen)
    assert [word["surface"] for word in words] == ["大学", "生"]
    assert [word["candidates"][0]["reading"] for word in words] == ["だいがく", "せい"]
    assert [candidate["reading"] for candidate in words[1]["candidates"]] == ["せい", "なま"]


def test_windows_preserve_word_boundaries():
    segments = [KanaSegment(i, i + 10, "あ" * 10) for i in range(0, 200, 10)]
    windows = windows_from_segments(segments, 200)
    assert windows[0][0] == 0 and windows[-1][1] == 200
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
    assert all(0 < end - start <= 64 for start, end in windows)
    assert all(any(start <= word.start < word.end <= end for start, end in windows)
               for word in segments)


@pytest.mark.parametrize("text,segments", [
    ("語" * 130, [KanaSegment(i, i + 1, "語") for i in range(130)]),
    ("あ" * 80, [KanaSegment(0, 80, "あ" * 80)]),
    ("あ い", [KanaSegment(0, 1, "あ"), KanaSegment(1, 2, " "), KanaSegment(2, 3, "い")]),
])
def test_windowed_inference_covers_the_input(text, segments):
    instance = runtime({"語": tuple(("ご" + "う" * i, "jmdict", 0.5, 0.5) for i in range(20))})

    def score(local_text, edges):
        assert len(local_text) <= 64
        assert len(edges) <= 256
        return [edge.prior for edge in edges]

    instance.score = score
    words = instance.predict_words(text, segments, beam_size=4)
    assert [(word["start"], word["end"], word["surface"]) for word in words] == [
        (segment.start, segment.end, segment.surface)
        for segment in segments if segment.surface.strip()
    ]
    assert all(word["candidates"] for word in words)
    if text == "あ" * 80:
        assert words[0]["candidates"][0]["reading"] == text


def test_rendaku_candidates_keep_both_readings():
    lexicon = UnifiedLexicon({"中島": (
        ("なかじま", "yomogi_dict", 0.6, 0.75),
        ("なかしま", "yomogi_dict", 0.6, 0.75),
    )})
    edges = {edge.reading: edge for edge in build_edges("中島", lexicon)
             if edge.start == 0 and edge.end == 2}
    assert {"なかじま", "なかしま"} <= edges.keys()
    assert edges["なかじま"].prior > edges["なかしま"].prior


def test_bundle_contents(tmp_path):
    path = tmp_path / "lexicon.txz"
    path.write_bytes(lzma.compress((
        "##kashi-g2p v1\n##section pack\n大学\tだいがく\tunidic\t1.0\t1.0\n"
        "##section prior\n大学\tだいがく\t1\n##section vocab\n大\n学\n##end\n"
    ).encode("utf-8")))
    table, prior, vocab = _load_bundle(path)
    assert table == {"大学": (("だいがく", "unidic", 1.0, 1.0),)}
    assert prior == {("大学", "だいがく"): 1}
    assert vocab == ["大", "学"]


def test_model_features_and_scores():
    instance = object.__new__(G2POnnxRuntime)
    instance.vocab = {"今": 2, "日": 3, "き": 4, "ょ": 5, "う": 6}
    instance.unk_id = 1
    instance.prior_index = {("今日", "キョウ"): 7}
    instance._reading_cache = {}
    feeds = {}

    def run(outputs, inputs):
        feeds.update(inputs)
        return [np.array([[0.75]], dtype=np.float32)]

    instance.session = SimpleNamespace(run=run)
    scores = instance.score("今日", [Edge(0, 2, "今日", "キョウ", "unidic", 0.5)])
    assert scores.tolist() == [0.75]
    assert feeds["input_ids"][0, :2].tolist() == [2, 3]
    assert feeds["attention_mask"].sum() == 2
    assert feeds["edge_start"].tolist() == [[0]]
    assert feeds["edge_end"].tolist() == [[2]]
    assert feeds["edge_reading_ids"][0, 0, :3].tolist() == [4, 5, 6]
    assert feeds["edge_reading_mask"].sum() == 3
    assert feeds["edge_pack_ids"].tolist() == [[7]]


@pytest.mark.parametrize("text,segments,expected", [
    ("3人", [KanaSegment(0, 1, "3"), KanaSegment(1, 2, "人")], ["3人"]),
    ("3 人", [KanaSegment(0, 1, "3"), KanaSegment(2, 3, "人")], ["3", "人"]),
])
def test_counter_boundaries(text, segments, expected):
    assert [segment.surface for segment in merge_numeral_counters(segments, text)] == expected
