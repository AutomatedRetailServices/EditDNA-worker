"""D-053 Section 12: dedicated tests for cutsell_worker/asr.py's
ground-truthed explicit decode-parameter wiring and the deterministic
candidate config. Covers named categories:
  1. explicit transcribe parameter wiring
  2. config fingerprint changes when behaviorally-relevant config changes
  3. config fingerprint stable when config identical
  4. temperature fallback disabled in deterministic mode
  5. VAD config explicit
  6. compute type explicit
  7. legacy config unchanged when flag OFF

A fake faster_whisper.WhisperModel records every kwarg its __init__ and
transcribe() are called with -- assertions check the ACTUAL argument dict
that crosses the call boundary, never a guessed subset."""
from __future__ import annotations

import sys
import types

import pytest

from cutsell_worker import asr
from cutsell_worker.asr import (
    DEFAULT_TEMPERATURE_LADDER,
    DEFAULT_VAD_PARAMETERS,
    DETERMINISTIC_TEMPERATURE,
    FasterWhisperASR,
    build_deterministic_asr_provider,
    load_asr_provider_from_env,
)


class _FakeWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class _FakeSegment:
    def __init__(self, text, start, end, words):
        self.text = text
        self.start = start
        self.end = end
        self.words = words


class _FakeWhisperModel:
    """Records every kwarg __init__ and transcribe() are called with."""

    instances: list["_FakeWhisperModel"] = []

    def __init__(self, model_name, *, device, compute_type):
        self.model_name = model_name
        self.init_kwargs = {"device": device, "compute_type": compute_type}
        self.transcribe_calls: list[dict] = []
        _FakeWhisperModel.instances.append(self)

    def transcribe(self, path, **kwargs):
        self.transcribe_calls.append(kwargs)
        segments = (
            _FakeSegment("hello world", 0.0, 1.0, (
                _FakeWord("hello", 0.0, 0.4),
                _FakeWord("world", 0.4, 1.0),
            )),
        )
        return segments, types.SimpleNamespace(language="en")


@pytest.fixture(autouse=True)
def _stub_faster_whisper(monkeypatch):
    _FakeWhisperModel.instances = []
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    yield


# --- Category 1: explicit transcribe parameter wiring -------------------

def test_transcribe_wires_every_explicit_decode_parameter():
    provider = FasterWhisperASR(model_name="medium")
    provider.transcribe("fake.mp4", source_asset_id="src-1")

    assert len(_FakeWhisperModel.instances) == 1
    model = _FakeWhisperModel.instances[0]
    assert model.init_kwargs == {"device": "auto", "compute_type": "auto"}

    assert len(model.transcribe_calls) == 1
    call = model.transcribe_calls[0]
    assert call["task"] == "transcribe"
    assert call["beam_size"] == 5
    assert call["best_of"] == 5
    assert call["patience"] == 1.0
    assert call["length_penalty"] == 1.0
    assert call["repetition_penalty"] == 1.0
    assert call["no_repeat_ngram_size"] == 0
    assert call["temperature"] == list(DEFAULT_TEMPERATURE_LADDER)
    assert call["compression_ratio_threshold"] == 2.4
    assert call["log_prob_threshold"] == -1.0
    assert call["no_speech_threshold"] == 0.6
    assert call["condition_on_previous_text"] is True
    assert call["prompt_reset_on_temperature"] == 0.5
    assert call["word_timestamps"] is True
    assert call["vad_filter"] is True
    assert call["clip_timestamps"] == "0"


def test_legacy_provider_passes_the_full_fallback_ladder_as_a_list():
    provider = FasterWhisperASR(model_name="medium")
    provider.transcribe("fake.mp4", source_asset_id="src-1")

    call = _FakeWhisperModel.instances[0].transcribe_calls[0]
    assert call["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# --- Category 4: temperature fallback disabled in deterministic mode ----

def test_deterministic_provider_passes_scalar_temperature_not_a_list():
    provider = build_deterministic_asr_provider("medium")
    provider.transcribe("fake.mp4", source_asset_id="src-1")

    call = _FakeWhisperModel.instances[0].transcribe_calls[0]
    # Scalar, not a single-element list -- confirms no fallback rung exists
    # for the library's decode-with-fallback loop to ever escalate to.
    assert call["temperature"] == 0.0
    assert isinstance(call["temperature"], float)


def test_deterministic_provider_reports_sampling_fallback_disabled():
    assert build_deterministic_asr_provider("medium").sampling_fallback_enabled is False
    assert FasterWhisperASR(model_name="medium").sampling_fallback_enabled is True


# --- Category 5: VAD config explicit -------------------------------------

def test_vad_parameters_are_threaded_explicit_and_match_ground_truthed_defaults():
    provider = FasterWhisperASR(model_name="medium")
    provider.transcribe("fake.mp4", source_asset_id="src-1")

    call = _FakeWhisperModel.instances[0].transcribe_calls[0]
    assert call["vad_parameters"] == {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "window_size_samples": 1024,
        "speech_pad_ms": 400,
    }
    assert call["vad_parameters"] == asr._vad_parameters_dict(DEFAULT_VAD_PARAMETERS)


# --- Category 6: compute type explicit -----------------------------------

def test_compute_type_is_threaded_explicit_into_whispermodel_init():
    provider = FasterWhisperASR(model_name="medium", compute_type="float16")
    provider.transcribe("fake.mp4", source_asset_id="src-1")

    assert _FakeWhisperModel.instances[0].init_kwargs["compute_type"] == "float16"


def test_compute_type_default_is_still_auto_unchanged():
    # D-053 Section 4: "Do NOT change it yet" -- compute_type stays "auto"
    # by default for both providers.
    assert FasterWhisperASR(model_name="medium").compute_type == "auto"
    assert build_deterministic_asr_provider("medium").compute_type == "auto"


# --- Category 7: legacy config unchanged when flag OFF -------------------

def test_load_asr_provider_from_env_flag_off_matches_bare_legacy_construction():
    provider = load_asr_provider_from_env({}, model_name="medium")
    bare = FasterWhisperASR(model_name="medium")
    assert provider.config_fingerprint().fingerprint() == bare.config_fingerprint().fingerprint()
    assert provider.temperature_ladder == DEFAULT_TEMPERATURE_LADDER
    assert provider.sampling_fallback_enabled is True


def test_load_asr_provider_from_env_deterministic_flag_on_disables_fallback():
    provider = load_asr_provider_from_env({"CUTSELL_ASR_DETERMINISTIC_CONFIG": "1"}, model_name="medium")
    assert provider.temperature_ladder == DETERMINISTIC_TEMPERATURE
    assert provider.sampling_fallback_enabled is False


def test_load_asr_provider_from_env_flag_falsey_values_stay_legacy():
    for value in ("0", "false", "no", "off", ""):
        provider = load_asr_provider_from_env({"CUTSELL_ASR_DETERMINISTIC_CONFIG": value}, model_name="medium")
        assert provider.temperature_ladder == DEFAULT_TEMPERATURE_LADDER


# --- Categories 2/3: config fingerprint changes/stable --------------------

def test_config_fingerprint_stable_for_identical_config():
    a = FasterWhisperASR(model_name="medium")
    b = FasterWhisperASR(model_name="medium")
    assert a.config_fingerprint().fingerprint() == b.config_fingerprint().fingerprint()


def test_config_fingerprint_changes_when_beam_size_changes():
    a = FasterWhisperASR(model_name="medium", beam_size=5)
    b = FasterWhisperASR(model_name="medium", beam_size=8)
    assert a.config_fingerprint().fingerprint() != b.config_fingerprint().fingerprint()


def test_config_fingerprint_changes_between_legacy_and_deterministic_temperature():
    legacy = FasterWhisperASR(model_name="medium")
    deterministic = build_deterministic_asr_provider("medium")
    assert legacy.config_fingerprint().fingerprint() != deterministic.config_fingerprint().fingerprint()


def test_config_fingerprint_records_sampling_fallback_enabled_flag():
    legacy = FasterWhisperASR(model_name="medium")
    deterministic = build_deterministic_asr_provider("medium")
    assert legacy.config_fingerprint().sampling_fallback_enabled is True
    assert deterministic.config_fingerprint().sampling_fallback_enabled is False
