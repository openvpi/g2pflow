"""Offline tests for the japanese-kashi converter (no model download)."""

import pytest

from g2pflow import G2PConversionError, G2PGroup, G2PPipeline
from g2pflow.converters.kashi import MODEL_URL, LEXICON_URL, JapaneseKashiConverter
from g2pflow.converters.kashi.segmentation import Segment


class _FakeSegmenter:
    def __init__(self, segments_by_text):
        self.segments_by_text = segments_by_text
        self.calls = []

    def segment(self, text):
        self.calls.append(text)
        return [Segment(*item)
                for item in self.segments_by_text.get(text, [])]

    def whole_word_readings(self, text, nbest=8):
        return []


class _FakeRuntime:
    def __init__(self, words_by_text):
        self.words_by_text = words_by_text
        self.calls = []

    def predict_words(self, text, segments, *, beam_size=16):
        self.calls.append((text, len(segments), beam_size))
        words = []
        cursor = 0
        for surface, readings in self.words_by_text.get(text, []):
            words.append({
                "surface": surface, "start": cursor,
                "end": cursor + len(surface),
                "candidates": [{"reading": reading, "score": -float(index)}
                               for index, reading in enumerate(
                                   readings[:beam_size])],
            })
            cursor += len(surface)
        return words


def phones(word):
    return [[phone for group in path for phone in group.phonemes]
            for reading in word.readings for path in reading.paths]


@pytest.fixture
def toshi_dict(dictionary):
    return dictionary(
        "a\ta\ni\ti\nu\tu\ne\te\no\to\nzu\tz u\nshi\tsh i\n"
        "hi\th i\nga\tg a\nka\tk a\nne\tn e\nko\tk o\nru\tr u\n", "toshi.txt",
    )


@pytest.fixture
def kashi_converter(kana_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    return JapaneseKashiConverter(str(kana_dict), model_path=str(model),
                                  lexicon_path=str(lexicon))


def test_release_urls():
    assert MODEL_URL.endswith("v1.1.0/model.onnx")
    assert LEXICON_URL.endswith("v1.1.0/lexicon.txz")


def test_default_download_cache_is_versioned(kana_dict, tmp_path):
    converter = JapaneseKashiConverter(str(kana_dict), cache_dir=str(tmp_path))
    assert converter._model_path == tmp_path / "v1.1.0" / "model.onnx"
    assert converter._lexicon_path == tmp_path / "v1.1.0" / "lexicon.txz"


def test_find_claims_japanese_runs_only(kashi_converter):
    assert kashi_converter.find("今日は良い天気ですね。") == (0, 10)
    assert kashi_converter.find("I do Love もう止められない") == (10, 18)
    assert kashi_converter.find("1人で行く") == (1, 5)
    assert kashi_converter.find("Hello world") is None
    assert kashi_converter.find("") is None


def test_fixed_segmentation_words(kashi_converter):
    segments = [(0, 1, "猫"), (1, 2, "が"), (2, 4, "好き")]
    fake_seg = _FakeSegmenter({"猫が好き": segments})
    fake_rt = _FakeRuntime({"猫が好き": [("猫", ["ねこ"]), ("が", ["が"]),
                                        ("好き", ["すき"])]})
    kashi_converter._segmenter = fake_seg
    kashi_converter._runtime = fake_rt
    words = kashi_converter.convert("猫が好き")
    assert [word.text for word in words] == ["猫", "が", "好き"]
    assert [phones(word) for word in words] == [
        [["n", "e", "k", "o"]],
        [["g", "a"]],
        [["s", "u", "k", "i"]],
    ]
    assert fake_seg.calls == ["猫が好き"]
    assert fake_rt.calls == [("猫が好き", 3, 16)]


def test_word_candidates_match_mecab_nbest_format(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"東": [(0, 1, "東")]})
    converter._runtime = _FakeRuntime({"東": [("東", ["ひがし", "あず"])]})
    words = converter.convert("東")
    assert len(words) == 1
    assert words[0].text == "東"
    assert [reading.paths for reading in words[0].readings] == [
        [[G2PGroup("hi", ["h", "i"]), G2PGroup("ga", ["g", "a"]),
          G2PGroup("shi", ["sh", "i"])]],
        [[G2PGroup("a", ["a"]), G2PGroup("zu", ["z", "u"])]],
    ]
    assert phones(words[0]) == [["h", "i", "g", "a", "sh", "i"],
                                ["a", "z", "u"]]


def test_duplicate_candidates_are_deduped(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"東": [(0, 1, "東")]})
    converter._runtime = _FakeRuntime({"東": [("東", ["ひがし", "ひがし", "あず"])]})
    words = converter.convert("東")
    assert len(words[0].readings) == 2


def test_beam_size_is_configurable(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon), beam_size=4)
    fake_rt = _FakeRuntime({"東": [("東", ["ひがし", "あず", "ひがし",
                                          "あず", "ひがし"])]})
    converter._segmenter = _FakeSegmenter({"東": [(0, 1, "東")]})
    converter._runtime = fake_rt
    words = converter.convert("東")
    assert len(words[0].readings) == 2
    assert fake_rt.calls == [("東", 1, 4)]


def test_long_vowels_follow_unidic_pron_like_mecab(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    kyo_dict = dictionary("kyo\tk y o\nu\tu\n", "kyo.txt")
    converter = JapaneseKashiConverter(str(kyo_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter(
        {"今日": [(0, 2, "今日", "キョー", "キョウ")]})
    converter._runtime = _FakeRuntime({"今日": [("今日", ["きょう"])]})
    words = converter.convert("今日")
    assert len(words) == 1
    assert phones(words[0]) == [["k", "y", "o"]]


def test_kana_words_use_whole_word_readings(kashi_converter):
    kashi_converter._segmenter = _FakeSegmenter({"キャット": [(0, 4, "キャット")]})
    kashi_converter._runtime = _FakeRuntime({"キャット": [("キャット", ["キャット"])]})
    words = kashi_converter.convert("キャット")
    assert [word.text for word in words] == ["キャット"]
    assert phones(words[0]) == [["ky", "a", "cl", "t", "o"]]


def test_unreadable_kanji_raises_like_mecab(kashi_converter):
    kashi_converter._segmenter = _FakeSegmenter({"珍": [(0, 1, "珍")]})
    kashi_converter._runtime = _FakeRuntime({"珍": [("珍", ["珍"])]})
    with pytest.raises(G2PConversionError):
        kashi_converter.convert("珍")


def test_candidates_with_kanji_are_dropped_whole(toshi_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(toshi_dict), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"光る": [(0, 2, "光る")]})
    converter._runtime = _FakeRuntime(
        {"光る": [("光る", ["ひかる", "光る"])]})
    words = converter.convert("光る")
    assert len(words) == 1
    assert len(words[0].readings) == 1  # the kanji-bearing candidate is gone
    assert phones(words[0]) == [["h", "i", "k", "a", "r", "u"]]


def test_unknowable_low_rank_candidates_are_skipped(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(str(dictionary("hi\th i\nka\tk a\n",
                                                    "tiny.txt")),
                                       model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"光か": [(0, 2, "光か")]})
    converter._runtime = _FakeRuntime(
        {"光か": [("光か", ["ひか", "ぐわ"])]})
    words = converter.convert("光か")
    assert len(words) == 1
    assert len(words[0].readings) == 1  # ぐわ needs a missing dict key, skipped
    assert phones(words[0]) == [["h", "i", "k", "a"]]


def test_kana_words_handle_small_kana(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("ne\tn e\ne\te\n", "fallback.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"ねぇ": [(0, 2, "ねぇ")]})
    converter._runtime = _FakeRuntime({"ねぇ": [("ねぇ", ["ねぇ"])]})
    words = converter.convert("ねぇ")
    assert [word.text for word in words] == ["ねぇ"]
    assert phones(words[0]) == [["n", "e", "e"]]


def test_kana_particles_use_unidic_pronunciation(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("wa\tw a\nha\th a\ne\te\nhe\th e\n", "particles.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    for surface, pron, expected in (
            ("は", "ワ", [["w", "a"], ["h", "a"]]),
            ("へ", "エ", [["e"], ["h", "e"]])):
        converter._segmenter = _FakeSegmenter({surface: [(0, 1, surface, pron)]})
        converter._runtime = _FakeRuntime({surface: [(surface, [surface])]})
        words = converter.convert(surface)
        assert [word.text for word in words] == [surface]
        assert phones(words[0]) == expected


def test_kana_words_keep_unidic_boundaries(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("sa\ts a\nyo\ty o\nna\tn a\nra\tr a\n", "kana.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter(
        {"さよなら": [(0, 4, "さよなら", "サヨナラ")]})
    converter._runtime = _FakeRuntime(
        {"さよなら": [("さよなら", ["さよなら"])]})
    words = converter.convert("さよなら")
    assert [word.text for word in words] == ["さよなら"]
    assert len(words[0].readings) == 1
    assert phones(words[0]) == [["s", "a", "y", "o", "n", "a", "r", "a"]]


def test_kana_word_dictionary_gap_is_a_conversion_error(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    converter = JapaneseKashiConverter(
        str(dictionary("ne\tn e\n", "gap.txt")),
        model_path=str(model), lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"ねぇ": [(0, 2, "ねぇ")]})
    converter._runtime = _FakeRuntime({"ねぇ": [("ねぇ", ["ねぇ"])]})
    with pytest.raises(G2PConversionError):
        converter.convert("ねぇ")


def test_empty_reading_omits_word(kashi_converter):
    kashi_converter._segmenter = _FakeSegmenter({"": []})
    kashi_converter._runtime = _FakeRuntime({"": []})
    assert kashi_converter.convert("") == []


def test_runtime_reuses_cached_instances(kashi_converter):
    fake_rt = _FakeRuntime({"あ": [("あ", ["あ"])]})
    fake_seg = _FakeSegmenter({"あ": [(0, 1, "あ")]})
    kashi_converter._runtime = fake_rt
    kashi_converter._segmenter = fake_seg
    assert kashi_converter._get_runtime() is fake_rt
    assert kashi_converter._get_segmenter() is fake_seg


@pytest.mark.integration
def test_real_model_end_to_end(kana_dict, tmp_path):
    """Opt-in: downloads the real model (~106 MB) on first use."""
    converter = JapaneseKashiConverter(str(kana_dict), cache_dir=str(tmp_path))
    words = converter.convert("今日は良い天気ですね")
    assert words and words[0].readings


def test_kanji_candidates_match_unidic_long_vowels_only(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    table = dictionary("ta\tt a\nme\tm e\ni\ti\nki\tk i\nbi\tb i\n",
                       "meta.txt")
    converter = JapaneseKashiConverter(str(table), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter(
        {"ため息": [(0, 3, "ため息", "タメイキ", "タメイキ")]})
    converter._runtime = _FakeRuntime(
        {"ため息": [("ため息", ["ためいき", "たび"])]})
    words = converter.convert("ため息")
    assert phones(words[0]) == [["t", "a", "m", "e", "i", "k", "i"],
                                ["t", "a", "b", "i"]]


def test_non_matching_candidate_is_not_folded(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    table = dictionary("ka\tk a\nku\tk u\nme\tm e\n", "meta2.txt")
    converter = JapaneseKashiConverter(str(table), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter(
        {"革命": [(0, 2, "革命", "カクメー", "カクメイ")]})
    converter._runtime = _FakeRuntime(
        {"革命": [("革命", ["かくめい"])]})
    words = converter.convert("革命")
    assert phones(words[0]) == [["k", "a", "k", "u", "m", "e"]]


def test_windows_follow_segment_boundaries():
    from g2pflow.converters.kashi.runtime import windows_from_segments
    segments = [Segment(i * 10, (i + 1) * 10, "x" * 10) for i in range(20)]
    windows = windows_from_segments(segments, 200)
    assert windows[0][0] == 0 and windows[-1][1] == 200
    for start, end in windows:
        assert 0 < end - start <= 64
    assert all(any(start <= s.start and s.end <= end for start, end in windows)
               for s in segments)


def test_enforce_edge_cap_keeps_copies_and_bounds_others():
    from g2pflow.converters.kashi.lexicon import Edge, enforce_edge_cap
    edges = [Edge(i, i + 1, "x", "x", "copy", -1.0) for i in range(40)]
    edges += [Edge(0, 1, "x", f"r{i}", "jmdict", 0.5) for i in range(400)]
    capped = enforce_edge_cap(edges)
    assert len(capped) == 256
    assert sum(edge.source == "copy" for edge in capped) == 40


def test_long_text_is_built_per_window_without_edge_cap_error():
    from g2pflow.converters.kashi.lexicon import UnifiedLexicon
    from g2pflow.converters.kashi.runtime import G2POnnxRuntime
    text = "明日は晴れです" * 20
    pack = {"明": (("あ", "jmdict", 0.5, 0.5),),
            "日": (("ひ", "jmdict", 0.5, 0.5),)}
    runtime = object.__new__(G2POnnxRuntime)
    runtime.lexicon = UnifiedLexicon(pack, use_copy=False)
    runtime._char_readings = {}
    runtime._decomposition_cache = {}
    runtime.score = lambda local_text, edges: [0.0] * len(edges)
    segments = [Segment(i, i + 1, char) for i, char in enumerate(text)]
    windows = len(__import__(
        "g2pflow.converters.kashi.runtime", fromlist=["windows_from_segments"]
    ).windows_from_segments(segments, len(text)))
    words = runtime.predict_words(text, segments, beam_size=4)
    assert windows > 1
    assert len(words) == len(text)


def test_rendaku_trap_is_kept_and_downweighted():
    from g2pflow.converters.kashi.lexicon import UnifiedLexicon, build_edges
    pack = {
        "中島": (("なかじま", "yomogi_dict", 0.6, 0.75),
                ("なかしま", "yomogi_dict", 0.6, 0.75)),
        "あ": (("あ", "copy", -1.0, 0.0),),
    }
    edges = {(edge.start, edge.end, edge.reading): edge
             for edge in build_edges("あ中島", UnifiedLexicon(pack))}
    unvoiced = edges[(1, 3, "なかしま")]
    voiced = edges[(1, 3, "なかじま")]
    assert unvoiced.prior == pytest.approx(0.4)
    assert voiced.prior == pytest.approx(0.6)


def test_convert_word_returns_one_fixed_word(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    table = dictionary(
        "to\tt o\nu\tu\nkyo\tky o\nta\tt a\n", "fixed.txt")
    converter = JapaneseKashiConverter(str(table), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter({"東京都": []})
    converter._runtime = _FakeRuntime(
        {"東京都": [("東京都", ["とうきょうと"])]})
    word = converter.convert_word("東京都")
    assert word is not None and word.text == "東京都"
    assert phones(word) == [["t", "o", "u", "ky", "o", "u", "t", "o"]]


def test_getstate_drops_native_handles(kashi_converter):
    kashi_converter._runtime = object()
    kashi_converter._segmenter = object()
    state = kashi_converter.__getstate__()
    assert state["_runtime"] is None
    assert state["_segmenter"] is None


def test_combining_mark_keeps_raw_offsets(dictionary, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    table = dictionary("ga\tg a\ncl\tcl\nko\tk o\nu\tu\n", "norm.txt")
    converter = JapaneseKashiConverter(str(table), model_path=str(model),
                                       lexicon_path=str(lexicon))
    converter._segmenter = _FakeSegmenter(
        {"が学校": [(0, 1, "が"), (1, 3, "学校")]})
    converter._runtime = _FakeRuntime(
        {"が学校": [("が", ["が"]), ("学校", ["がっこう"])]})
    words = converter.convert("か\u3099学校")
    assert [word.text for word in words] == ["か\u3099", "学校"]


def test_pipeline_config_integration(kana_dict, tmp_path):
    model = tmp_path / "model.onnx"
    lexicon = tmp_path / "lexicon.txz"
    model.write_bytes(b"stub")
    lexicon.write_bytes(b"stub")
    from g2pflow import G2PPipelineConfig, build_pipeline_from_config
    pipeline = build_pipeline_from_config(G2PPipelineConfig.model_validate({
        "converters": [{
            "id": "japanese-kashi",
            "language": "ja",
            "kwargs": {"dict_path": str(kana_dict),
                       "model_path": str(model),
                       "lexicon_path": str(lexicon)},
        }],
    }))
    pipeline._converters[0]._segmenter = _FakeSegmenter({"ネコが": [(0, 2, "ネコ"), (2, 3, "が")]})
    pipeline._converters[0]._runtime = _FakeRuntime({"ネコが": [("ネコ", ["ねこ"]), ("が", ["が"])]})
    words = pipeline.convert("ネコが", languages=["ja"])
    assert [word.text for word in words] == ["ネコ", "が"]


def test_merge_numeral_counters_does_not_cross_a_gap():
    from g2pflow.converters.kashi.segmentation import merge_numeral_counters
    text = "3 人"
    segments = [Segment(0, 1, "3"), Segment(2, 3, "人")]
    assert [segment.surface for segment in
            merge_numeral_counters(segments, text)] == ["3", "人"]


def test_windows_from_segments_chunks_a_trailing_gap():
    from g2pflow.converters.kashi.runtime import windows_from_segments
    segments = [Segment(0, 70, "x" * 70)]
    windows = windows_from_segments(segments, 200)
    assert windows[0][0] == 0 and windows[-1][1] == 200
    assert all(0 < end - start <= 64 for start, end in windows)


def test_enforce_edge_cap_rejects_more_copies_than_cap():
    from g2pflow.converters.kashi.lexicon import Edge, enforce_edge_cap
    edges = [Edge(i, i + 1, "x", "x", "copy", -1.0) for i in range(300)]
    with pytest.raises(ValueError, match="copy edges exceed"):
        enforce_edge_cap(edges)


def test_predict_words_skips_whitespace_segments():
    from g2pflow.converters.kashi.lexicon import UnifiedLexicon
    from g2pflow.converters.kashi.runtime import G2POnnxRuntime
    runtime = object.__new__(G2POnnxRuntime)
    runtime.lexicon = UnifiedLexicon({}, use_copy=False)
    runtime._char_readings = {}
    runtime._decomposition_cache = {}
    runtime.score = lambda local_text, edges: [0.0] * len(edges)
    segments = [Segment(0, 1, "あ"), Segment(1, 2, " "),
                Segment(2, 3, "い")]
    words = runtime.predict_words("あ い", segments, beam_size=4)
    assert [word["surface"] for word in words] == ["あ", "い"]
    assert [(word["start"], word["end"]) for word in words] == [(0, 1), (2, 3)]


def test_segment_longer_than_window_is_decoded_per_chunk():
    from g2pflow.converters.kashi.lexicon import UnifiedLexicon
    from g2pflow.converters.kashi.runtime import G2POnnxRuntime
    text = "あ" * 80
    runtime = object.__new__(G2POnnxRuntime)
    runtime.lexicon = UnifiedLexicon({}, use_copy=False)
    runtime._char_readings = {}
    runtime._decomposition_cache = {}
    runtime.score = lambda local_text, edges: [0.0] * len(edges)
    words = runtime.predict_words(text, [Segment(0, 80, text)], beam_size=4)
    assert len(words) == 1
    assert words[0]["candidates"]
    assert words[0]["candidates"][0]["reading"] == text
