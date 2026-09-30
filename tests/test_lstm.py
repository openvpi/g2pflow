import builtins
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from g2pflow.converters.lstm import LSTMConverter


@pytest.fixture
def model_dir(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "char.json").write_text(json.dumps({"<unk>": 0, "a": 1, "b": 2}), encoding="utf-8")
    (folder / "phonemes.json").write_text(json.dumps({
        "<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3, "A": 4, "B": 5,
    }), encoding="utf-8")
    return folder


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    sessions = []

    class Session:
        def __init__(self, path):
            self.path = path
            sessions.append(path)

        def run(self, outputs, inputs):
            if Path(self.path).name == "encoder.onnx":
                calls.append(("encoder", inputs["input_ids"].copy()))
                state = np.zeros((1, 1, 1), dtype=np.float32)
                return [np.zeros((1, 1, 1)), state, state.copy()]
            previous = int(inputs["decoder_input"][0, 0])
            step = int(inputs["hidden"][0, 0, 0])
            calls.append(("decoder", previous, step))
            logits = np.full(6, -100.0)
            if step == 0:
                logits[4], logits[5] = 0, -1
            elif step == 1 and previous == 4:
                logits[3], logits[5] = 0, -2
            elif step == 1:
                logits[4], logits[3] = 0, -2
            else:
                logits[3] = 0
            return [logits.reshape(1, 1, 6), inputs["hidden"] + 1,
                    inputs["cell"] + 1, None]

    module = ModuleType("onnxruntime")
    module.InferenceSession = Session
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return calls, sessions


def test_lazy_sessions_dictionary_hits_and_word_matching(model_dir, dictionary, runtime):
    calls, sessions = runtime
    instance = LSTMConverter(model_path=str(model_dir), dict_path=str(dictionary("known\tK\nknown\tN\n")))
    assert sessions == []
    assert instance.find("unknown known") == (8, 13)
    assert instance.find("xxx ab") == (4, 6)
    assert instance.find("xxx") is None
    paths = instance.convert("known")[0].readings[0].paths
    assert [path[0].phonemes for path in paths] == [["K"], ["N"]]
    assert calls == sessions == []


def test_beam_paths_ranking_eos_and_separate_hypothesis_state(model_dir, runtime):
    calls, sessions = runtime
    instance = LSTMConverter(model_path=str(model_dir), beam_size=2)
    assert instance.infer_oov("ab") == [["A"], ["B", "A"]]
    assert len(sessions) == 2
    assert ("decoder", 4, 1) in calls
    assert ("decoder", 5, 1) in calls
    assert not any(call[:2] == ("decoder", 3) for call in calls)
    assert np.array_equal(calls[0][1], [[2, 1, 2, 3]])
    instance.infer_oov("a")
    assert len(sessions) == 2


def test_greedy_decode_and_special_tokens(model_dir, runtime):
    instance = LSTMConverter(model_path=str(model_dir), beam_size=1)
    assert instance.infer_oov("A") == [["A"]]
    assert instance._decode([0, 1, 2, 4, 3, 5, 99]) == ["A", "B"]


def test_deduplicate_paths_after_special_token_removal(model_dir, monkeypatch):
    class Encoder:
        def run(self, _, inputs):
            state = np.zeros((1, 1, 1))
            return [state, state, state]

    class Decoder:
        def run(self, _, inputs):
            scores = np.full(6, -100.0)
            if int(inputs["decoder_input"][0, 0]) == 2:
                scores[0], scores[3] = 0, -1
            else:
                scores[3] = 0
            return [scores.reshape(1, 1, 6), inputs["hidden"], inputs["cell"], None]

    instance = LSTMConverter(model_path=str(model_dir), beam_size=2)
    instance._encoder_session = Encoder()
    instance._decoder_session = Decoder()
    assert instance.infer_oov("a") == [[]]


def test_pickle_drops_native_sessions(model_dir):
    instance = LSTMConverter(model_path=str(model_dir))
    instance._encoder_session = lambda: None
    instance._decoder_session = lambda: None
    restored = pickle.loads(pickle.dumps(instance))
    assert restored._encoder_session is restored._decoder_session is None
    assert instance._encoder_session is not None
    assert restored._char_vocab == instance._char_vocab


def test_missing_runtime_hint(model_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"g2pflow\[lstm\]"):
        LSTMConverter(model_path=str(model_dir)).infer_oov("a")


def test_dictionary_hits_work_without_numpy_or_runtime(model_dir, dictionary, monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    instance = LSTMConverter(model_path=str(model_dir), dict_path=str(dictionary("known\tK\n")))
    word = instance.convert("known")[0]
    assert word.readings[0].paths[0][0].phonemes == ["K"]
    with pytest.raises(ImportError, match=r"g2pflow\[lstm\]"):
        instance.infer_oov("a")


def test_internal_runtime_import_error_is_preserved(model_dir, monkeypatch):
    original = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError("internal runtime dependency", name="internal_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ModuleNotFoundError) as exc:
        LSTMConverter(model_path=str(model_dir)).infer_oov("a")
    assert exc.value.name == "internal_dependency"


@pytest.mark.parametrize("beam_size", [True, False, 0, -1, 1.5])
def test_invalid_beam_size(model_dir, beam_size):
    with pytest.raises(ValueError, match="positive integer"):
        LSTMConverter(model_path=str(model_dir), beam_size=beam_size)
