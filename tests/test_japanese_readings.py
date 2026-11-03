import pytest

from g2pflow import G2PConversionError, G2PPipeline
from g2pflow.converters.japanese import JapaneseReadingConverter, KanaReading, KanaSegment


class ReadingBackend(JapaneseReadingConverter):
    language = ("ja", "jpn")

    def __init__(self, dictionary, segments, candidates, *, fixed=(), **kwargs):
        super().__init__(str(dictionary), **kwargs)
        self.segments = segments
        self.candidates = candidates
        self.fixed = list(fixed)
        self.received_segments = []

    def _segments(self, text):
        return self.segments

    def _readings(self, text, segments):
        self.received_segments = segments
        return [self.candidates.get(segment.surface, []) for segment in segments]

    def _word_readings(self, text):
        return self.fixed


def scripts(word):
    return [[group.script for group in reading.paths[0]] for reading in word.readings]


@pytest.mark.parametrize("include_surface", [False, True])
@pytest.mark.parametrize("text,parts,prons,expected", [
    ("はゝ", ["は", "ゝ"], ["ワ", ""], ["ha", "ha"]),
    ("すゞ", ["す", "ゞ"], ["ス", ""], ["su", "zu"]),
    ("ぐゎ", ["ぐ", "ゎ"], ["グ", "ワ"], ["gwa"]),
    ("グヮ", ["グ", "ヮ"], ["グ", "ワ"], ["gwa"]),
    ("きゃ", ["き", "ゃ"], ["キ", "ヤ"], ["kya"]),
])
def test_source_units_are_prepared_before_backend_readings(
    dictionary, include_surface, text, parts, prons, expected,
):
    path = dictionary("ha\tHA\nsu\tSU\nzu\tZU\ngwa\tGWA\nkya\tKYA\n")
    segments = [KanaSegment(i, i + len(part), part, pron, part)
                for i, (part, pron) in enumerate(zip(parts, prons))]
    backend = ReadingBackend(path, segments, {}, include_surface=include_surface)
    word, = backend.convert(text)
    assert word.text == text
    assert scripts(word) == [expected]
    assert backend.received_segments == [KanaSegment(0, len(text), text)]


@pytest.mark.parametrize("include_surface", [False, True])
@pytest.mark.parametrize("text,kana,pron,expected", [
    ("きょう", "キョウ", "キョー", ["kyo"]),
    ("ちょうど", "チョウド", "チョード", ["cho", "do"]),
    ("キャット", "キャット", "キャット", ["kya", "cl", "to"]),
])
def test_intact_kana_words_keep_contextual_pronunciations(
    dictionary, include_surface, text, kana, pron, expected,
):
    path = dictionary("kyo\tKYO\ncho\tCHO\ndo\tDO\nkya\tKYA\ncl\tCL\nto\tTO\nu\tU\n")
    backend = ReadingBackend(
        path, [KanaSegment(0, len(text), text, pron, kana)],
        {text: [KanaReading(kana)]}, include_surface=include_surface, max_readings=1,
    )
    assert scripts(backend.convert(text)[0]) == [expected]


def test_model_alternatives_only_use_matching_reference_pronunciation(dictionary):
    path = dictionary("kyo\tKYO\nko\tKO\nn\tN\nni\tNI\nchi\tCHI\nu\tU\n")
    backend = ReadingBackend(path, [KanaSegment(0, 2, "今日", "キョー", "キョウ")], {
        "今日": [KanaReading("きょう"), KanaReading("こんにち")],
    })
    assert scripts(backend.convert("今日")[0]) == [["kyo"], ["ko", "n", "ni", "chi"]]


def test_unsupported_candidates_do_not_poison_valid_readings(dictionary):
    path = dictionary("hi\tHI\nka\tKA\nru\tRU\n")
    backend = ReadingBackend(path, [KanaSegment(0, 2, "光る")], {
        "光る": [KanaReading("光る"), KanaReading("ぐゎー"), KanaReading("ひかる")],
    })
    assert scripts(backend.convert("光る")[0]) == [["hi", "ka", "ru"]]


def test_no_viable_kanji_reading_is_an_error(dictionary):
    backend = ReadingBackend(dictionary("ka\tKA\n"), [KanaSegment(0, 1, "語")], {
        "語": [KanaReading("語")],
    })
    with pytest.raises(G2PConversionError):
        backend.convert("語")


def test_context_does_not_cross_source_gaps(dictionary):
    backend = ReadingBackend(dictionary("ka\tKA\nga\tGA\n"), [
        KanaSegment(0, 1, "か"), KanaSegment(2, 3, "ゞ"),
    ], {})
    with pytest.raises(KeyError, match="preceding syllable"):
        backend.convert("か ゞ")


def test_candidate_limit_deduplicates_before_truncating(dictionary):
    path = dictionary("ka\tK A\nka\tKA\nki\tKI\n")
    backend = ReadingBackend(path, [], {}, max_readings=2, fixed=[
        KanaReading("か"), KanaReading("カ"), KanaReading("き"),
    ])
    word = backend.convert_word("語")
    assert scripts(word) == [["ka"], ["ki"]]
    assert len(word.readings[0].paths) == 2


def test_fixed_word_fallback_bounds_the_combined_candidates(dictionary):
    path = dictionary("ka\tKA\nki\tKI\n")
    text = "語" * 20
    backend = ReadingBackend(path, [KanaSegment(i, i + 1, "語") for i in range(len(text))], {
        "語": [KanaReading("か"), KanaReading("き")],
    }, max_readings=3)
    word, = G2PPipeline(converters=[backend]).convert_pfml(f"<word>{text}</word>")
    assert word.text == text
    assert len(word.readings) == 3
    assert all(len(reading.paths[0]) == len(text) for reading in word.readings)


def test_explicit_word_boundary_does_not_borrow_kana_context(dictionary):
    class LiteralBackend(ReadingBackend):
        def _segments(self, text):
            return [KanaSegment(0, len(text), text)]

    backend = LiteralBackend(dictionary("ka\tKA\nga\tGA\n"), [], {})
    with pytest.raises(KeyError, match="preceding syllable"):
        G2PPipeline(converters=[backend]).convert_pfml("<word>か</word><word>ゞ</word>")
