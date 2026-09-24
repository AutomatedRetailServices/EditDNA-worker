"""Safety of the real ASR entry point before a paid full-engine experiment."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import gpt_whisperx_asr as gw
from cutsell_worker.universal_clean_cut_validation import _validation_asr


def test_provider_is_explicit_raw_opt_in():
    config = SimpleNamespace(asr_model="medium")
    assert _validation_asr(config, env={}).model_name == "medium"
    assert isinstance(_validation_asr(config, env={gw.PROVIDER_ENV: gw.PROVIDER}), gw.GPTWhisperXASR)
    with pytest.raises(ValueError, match="Unknown validation"):
        _validation_asr(config, env={gw.PROVIDER_ENV: "typo"})


def test_engine_fingerprint_contract_includes_requested_language():
    provider = gw.GPTWhisperXASR()
    assert provider.config_fingerprint(language_hint=None).fingerprint().startswith("asrcfg_")
    assert provider.config_fingerprint(language_hint="es").fingerprint() != provider.config_fingerprint(language_hint=None).fingerprint()


@pytest.mark.parametrize("duration,silences", [
    (366.997, ((28.0, 29.0), (60.0, 63.0), (92.0, 94.0), (200.0, 202.0))),
    (41.0, ()), (300.0, ()), (12.0, ()), (40.001, ((20.0, 22.0),)),
])
def test_windows_cover_source_once_without_oversized_alignment(duration, silences):
    windows = gw.chunk_windows(duration, silences)
    assert windows[0][0] == 0
    assert windows[-1][1] == duration
    assert all(0 < end - start <= gw.MAX_CHUNK_SEC for start, end in windows)
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
    assert gw.chunk_windows(duration, silences) == windows


def test_chunking_prefers_actual_quiet_midpoint():
    assert gw.chunk_windows(65, ((28.0, 30.0),))[0] == (0.0, 29.0)


def evidence():
    return ({"index": 3, "start": 100.0, "end": 110.0, "text": "No cuesta 23."},
            {"segments": [{"words": [{"word": "No", "start": 0.2, "end": 0.4, "score": 0.9},
                                      {"word": "cuesta", "start": 0.4, "end": 0.9},
                                      {"word": "23.", "start": 1.1, "end": 1.7}]}]})


def test_preserves_negation_numbers_and_absolute_time():
    chunk, result = evidence()
    segments = gw.checked_segments(chunk, result, "source-a")
    assert [w.text for s in segments for w in s.words] == ["No", "cuesta", "23."]
    assert segments[0].start == 100.2
    assert segments[0].end == 101.7
    assert segments[0].source_asset_id == "source-a"


@pytest.mark.parametrize("mutation", ["missing", "zero", "overlap", "outside", "nan", "text", "score"])
def test_bad_alignment_cannot_reach_the_editor(mutation):
    chunk, result = evidence()
    word = result["segments"][0]["words"][1]
    if mutation == "missing": del word["start"]
    if mutation == "zero": word["end"] = word["start"]
    if mutation == "overlap": word["start"] = 0.3
    if mutation == "outside": word["end"] = 12
    if mutation == "nan": word["start"] = float("nan")
    if mutation == "text": word["word"] = "paga"
    if mutation == "score": word["score"] = float("inf")
    with pytest.raises(gw.AlignmentEvidenceError):
        gw.checked_segments(chunk, result, "source")


@pytest.mark.parametrize("key", ["", "sk-admin-not-an-inference-key"])
def test_missing_or_admin_credentials_fail_before_media_or_api(monkeypatch, key):
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(gw.requests, "post", lambda *a, **k: pytest.fail("API must not be called"))
    with pytest.raises(RuntimeError, match="ordinary OPENAI_API_KEY"):
        gw.GPTWhisperXASR().transcribe("unused", source_asset_id="s")


def test_provider_orchestrates_real_contract_without_leaking_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-project-test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-cloud-test-key")
    runtime = tmp_path / "python"
    runtime.touch()
    monkeypatch.setattr(gw, "ALIGNER_PYTHON", str(runtime))
    monkeypatch.setattr(gw, "ALIGNER_SCRIPT", str(runtime))
    monkeypatch.setattr(gw, "detect_audio_silence_intervals", lambda *a, **k: ())
    requests_seen = []

    def post(url, **kwargs):
        requests_seen.append((url, kwargs))
        assert kwargs["data"] == [("model", "gpt-transcribe")]
        assert kwargs["files"]["file"][2] == "audio/wav"
        return SimpleNamespace(status_code=200, headers={"x-request-id": "req-test"},
                               json=lambda: {"text": " No\ncuesta 23. ", "languages": [{"code": "es"}]})

    def run(argv, **kwargs):
        if argv[0] == "ffprobe":
            return SimpleNamespace(stdout=json.dumps({"format": {"duration": 10}}))
        if argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"wav-test")
            return SimpleNamespace(returncode=0)
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "AWS_SECRET_ACCESS_KEY" not in kwargs["env"]
        manifest = json.loads(Path(argv[2]).read_text())
        assert manifest["chunks"][0]["text"] == "No cuesta 23."
        _, result = evidence()
        result["index"] = 0
        result["alignment_model"] = "test-model"
        Path(argv[3]).write_text(json.dumps({"chunks": [result], "runtime": {"test": True}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gw.requests, "post", post)
    monkeypatch.setattr(gw.subprocess, "run", run)
    provider = gw.GPTWhisperXASR()
    source = tmp_path / "video.mp4"
    source.write_bytes(b"first source bytes")
    output = provider.transcribe(str(source), source_asset_id="s")
    assert len(requests_seen) == 1
    assert output[0].text == "No cuesta 23."
    assert provider.last_audit["status"] == "passed"
    assert provider.last_audit["word_count"] == 3
    assert "fake-project" not in json.dumps(provider.last_audit)
    # The real engine re-enters ASR for pre-Freeze boundary completion.
    # It must receive the exact evidence and incur no second API pass.
    again = provider.transcribe(str(source), source_asset_id="s")
    assert again is output
    assert len(requests_seen) == 1
    assert provider.last_audit["cache_hit_count"] == 1
    # Path reuse never permits stale speech evidence.
    first_hash = provider.last_audit["source_media_sha256"]
    source.write_bytes(b"different source bytes")
    provider.transcribe(str(source), source_asset_id="s")
    assert len(requests_seen) == 2
    assert provider.last_audit["source_media_sha256"] != first_hash


def test_provider_failure_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-project-test-key")
    monkeypatch.setattr(gw, "ALIGNER_PYTHON", str(tmp_path / "missing"))
    monkeypatch.setattr(gw.requests, "post", lambda *a, **k: pytest.fail("No API call before image preflight"))
    with pytest.raises(RuntimeError, match="isolated WhisperX environment is missing"):
        gw.GPTWhisperXASR().transcribe("unused", source_asset_id="s")
