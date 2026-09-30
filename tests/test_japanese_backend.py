import builtins
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from g2pflow import G2PConversionError, G2PPipeline
from g2pflow.converters.japanese import JapaneseMecabConverter


def node(surface, pron):
    return SimpleNamespace(surface=surface, feature=SimpleNamespace(pron=pron))


class TaggerDouble:
    def __init__(self, surfaces, analyses):
        self.surfaces = surfaces
        self.analyses = analyses

    def __call__(self, text):
        return [node(surface, None) for surface in self.surfaces]

    def nbestToNodeList(self, surface, nbest):
        return self.analyses.get(surface, [])[:nbest]


@pytest.fixture
def backend(monkeypatch, tmp_path):
    dicdir = tmp_path / "Uni Dic" / "dicdir"
    dicdir.mkdir(parents=True)
    events = []
    native = TaggerDouble([], {})
    unidic = ModuleType("unidic")
    unidic.DICDIR = str(dicdir)
    downloader = ModuleType("unidic.download")

    def download():
        events.append("download")
        (dicdir / "sys.dic").touch()
        (dicdir / "mecabrc").touch()

    downloader.download_version = download
    fugashi = ModuleType("fugashi")

    def tagger(options):
        events.append(("tagger", options))
        return native

    fugashi.Tagger = tagger
    monkeypatch.setitem(sys.modules, "unidic", unidic)
    monkeypatch.setitem(sys.modules, "unidic.download", downloader)
    monkeypatch.setitem(sys.modules, "fugashi", fugashi)

    class Lock:
        def __init__(self, filename):
            assert Path(filename) == dicdir.parent / "download.lock"

        def __enter__(self):
            events.append("lock")

        def __exit__(self, *args):
            events.append("unlock")

    filelock = ModuleType("filelock")
    filelock.FileLock = Lock
    monkeypatch.setitem(sys.modules, "filelock", filelock)
    return SimpleNamespace(dicdir=dicdir, events=events, native=native,
                           download=download, downloader=downloader, fugashi=fugashi,
                           filelock=filelock)


def test_construction_does_not_initialize_or_download(kana_dict, backend):
    instance = JapaneseMecabConverter(str(kana_dict))
    assert instance._tagger is None
    assert backend.events == []
    assert instance.find("word猫かなabc") == (4, 7)
    assert instance.find("romaji") is None
    assert instance.convert("") == []
    assert backend.events == []


@pytest.mark.parametrize("existing", [(), ("sys.dic",), ("mecabrc",)])
def test_missing_default_downloads_inside_lock(kana_dict, backend, existing):
    for filename in existing:
        (backend.dicdir / filename).touch()
    instance = JapaneseMecabConverter(str(kana_dict))
    assert instance._get_tagger() is backend.native
    assert backend.events[:3] == ["lock", "download", "unlock"]
    options = backend.events[3][1]
    assert f'-r "{(backend.dicdir / "mecabrc").as_posix()}"' in options
    assert f'-d "{backend.dicdir.as_posix()}"' in options
    assert instance._get_tagger() is backend.native
    assert len(backend.events) == 4


def test_existing_default_avoids_download_and_lock(kana_dict, backend, monkeypatch):
    backend.download()
    backend.events.clear()
    monkeypatch.setitem(sys.modules, "filelock", None)
    JapaneseMecabConverter(str(kana_dict))._get_tagger()
    assert len(backend.events) == 1
    assert backend.events[0][0] == "tagger"


def test_recheck_after_another_worker_downloads(kana_dict, backend, monkeypatch):
    class RacingLock:
        def __init__(self, path):
            pass

        def __enter__(self):
            (backend.dicdir / "sys.dic").touch()
            (backend.dicdir / "mecabrc").touch()

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(backend.filelock, "FileLock", RacingLock)
    JapaneseMecabConverter(str(kana_dict))._get_tagger()
    assert "download" not in backend.events


def test_explicit_directory_never_downloads_default(kana_dict, backend, tmp_path, monkeypatch):
    explicit = tmp_path / "external dictionary"
    monkeypatch.setitem(sys.modules, "filelock", None)
    instance = JapaneseMecabConverter(str(kana_dict), unidic_dir=str(explicit))
    instance._get_tagger()
    assert len(backend.events) == 1
    assert explicit.as_posix() in backend.events[0][1]


def test_missing_filelock_reports_download_dependency(kana_dict, backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "filelock", None)
    with pytest.raises(ImportError, match=r"g2pflow\[ja\]"):
        JapaneseMecabConverter(str(kana_dict))._get_tagger()
    assert backend.events == []


def test_download_and_tagger_failures_propagate(kana_dict, backend, monkeypatch):
    def fail():
        raise OSError("download unavailable")

    monkeypatch.setattr(backend.downloader, "download_version", fail)
    with pytest.raises(OSError, match="download unavailable"):
        JapaneseMecabConverter(str(kana_dict))._get_tagger()
    assert backend.events == ["lock", "unlock"]

    def broken(options):
        raise RuntimeError("invalid dictionary")

    monkeypatch.setattr(backend.fugashi, "Tagger", broken)
    with pytest.raises(RuntimeError, match="invalid dictionary"):
        JapaneseMecabConverter(str(kana_dict), unidic_dir="explicit")._get_tagger()


@pytest.mark.parametrize("missing", ["fugashi", "unidic"])
def test_optional_dependency_install_hint(kana_dict, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError, match=r"g2pflow\[ja\]"):
        JapaneseMecabConverter(str(kana_dict))._get_tagger()


def test_transitive_missing_dependency_is_not_hidden(kana_dict, monkeypatch):
    original = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "fugashi":
            raise ModuleNotFoundError("internal backend dependency", name="internal_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModuleNotFoundError) as exc:
        JapaneseMecabConverter(str(kana_dict))._get_tagger()
    assert exc.value.name == "internal_dependency"


def test_nbest_keeps_whole_word_unique_readings_and_groups(kana_dict):
    instance = JapaneseMecabConverter(str(kana_dict))
    instance._tagger = TaggerDouble(["猫"], {"猫": [
        [node("猫", "ネコ")], [node("猫", "ネコ")],
        [node("猫", "ネカ")], [node("猫", "*")], [node("猫", "-")],
        [node("猫", "")], [node("別", "ネコ")],
        [node("猫", "ネ"), node("猫", "コ")],
    ]})
    word = instance.convert("猫")[0]
    assert word.text == "猫"
    assert [[group.script for group in reading.paths[0]] for reading in word.readings] == [
        ["ne", "ko"], ["ne", "ka"],
    ]


def test_reading_combines_dictionary_paths_without_crossing_readings(dictionary):
    path = dictionary("ne\tn e\nne\tN E\nko\tk o\nko\tK O\n")
    instance = JapaneseMecabConverter(str(path))
    instance._tagger = TaggerDouble(["猫"], {"猫": [[node("猫", "ネコ")]]})
    paths = instance.convert("猫")[0].readings[0].paths
    assert [[group.phonemes for group in candidate] for candidate in paths] == [
        [["n", "e"], ["k", "o"]], [["n", "e"], ["K", "O"]],
        [["N", "E"], ["k", "o"]], [["N", "E"], ["K", "O"]],
    ]


def test_kana_fallback_and_split_digraph(kana_dict):
    instance = JapaneseMecabConverter(str(kana_dict))
    instance._tagger = TaggerDouble(["き", "ゃ", "ー"], {})
    words = instance.convert("きゃー")
    assert [word.text for word in words] == ["きゃ"]
    assert words[0].readings[0].paths[0][0].phonemes == ["ky", "a"]


def test_silent_mecab_readings_do_not_appear_in_results(kana_dict):
    instance = JapaneseMecabConverter(str(kana_dict))
    instance._tagger = TaggerDouble(["ー"], {"ー": [[node("ー", "ー")]]})
    assert instance.convert("ー") == []
    assert instance.convert_word("ー") is None


def test_nonstandard_kana_fallback_rejoins_split_digraphs(dictionary):
    path = dictionary("ja\tj a\nju\tj u\nje\tj e\njo\tj o\nza\tz a\nzo\tz o\n")
    instance = JapaneseMecabConverter(str(path))
    text = "ぢゃヂュぢぇヂョづぁヅォ"
    instance._tagger = TaggerDouble(list(text), {})
    words = instance.convert(text)
    assert [word.text for word in words] == ["ぢゃ", "ヂュ", "ぢぇ", "ヂョ", "づぁ", "ヅォ"]
    assert [word.readings[0].paths[0][0].phonemes for word in words] == [
        ["j", "a"], ["j", "u"], ["j", "e"], ["j", "o"], ["z", "a"], ["z", "o"],
    ]


def test_missing_non_kana_reading_raises(kana_dict):
    instance = JapaneseMecabConverter(str(kana_dict))
    instance._tagger = TaggerDouble(["猫"], {})
    with pytest.raises(G2PConversionError) as exc:
        instance.convert("猫")
    assert exc.value.unconverted_tokens == ["猫"]


def test_gemination_merges_words_and_pickle_drops_native_tagger(kana_dict):
    instance = JapaneseMecabConverter(str(kana_dict), double_written_sokuon=True)
    instance._tagger = TaggerDouble(["っ", "か"], {})
    word = instance.convert("っか")[0]
    assert word.text == "っか"
    assert [group.phonemes for group in word.readings[0].paths[0]] == [["k"], ["k", "a"]]
    instance._tagger = lambda: None
    restored = pickle.loads(pickle.dumps(instance))
    assert restored._tagger is None
    assert instance._tagger is not None
    assert restored._double_written_sokuon is True


@pytest.mark.parametrize("nbest", [True, False, 0, -1, 1.5])
def test_invalid_nbest(kana_dict, nbest):
    with pytest.raises(ValueError, match="positive integer"):
        JapaneseMecabConverter(str(kana_dict), nbest=nbest)


def test_pfml_fixed_word_prefers_whole_word_candidates(kana_dict):
    instance = JapaneseMecabConverter(str(kana_dict))
    instance._tagger = TaggerDouble(["unwanted", "segmentation"], {"猫": [
        [node("猫", "ネコ")], [node("猫", "ネカ")],
    ]})
    word = G2PPipeline(converters=[instance]).convert_pfml('<word language="ja">猫</word>')[0]
    assert word.text == "猫"
    assert [[g.script for g in r.paths[0]] for r in word.readings] == [["ne", "ko"], ["ne", "ka"]]


def test_pfml_fixed_word_composes_independent_internal_units(kana_dict):
    instance = JapaneseMecabConverter(str(kana_dict))
    instance._tagger = TaggerDouble(["ね", "こ"], {})
    word = G2PPipeline(converters=[instance]).convert_pfml('<word language="ja">ねこ</word>')[0]
    assert word.text == "ねこ"
    assert [[g.script for g in p] for p in word.readings[0].paths] == [["ne", "ko"]]
