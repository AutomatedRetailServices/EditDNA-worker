"""Offline contract tests for external ASR text comparison; no API calls."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutsell_worker import asr_text_comparison as comparison


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _fake_audio(monkeypatch):
    def run(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"fake-mp3")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(comparison.subprocess, "run", run)


@pytest.mark.parametrize("provider", ["gpt-transcribe", "gpt-4o-transcribe"])
@pytest.mark.parametrize("language_hint", [None, "es"])
def test_gpt_comparison_is_text_only_and_never_selection_authority(monkeypatch, provider, language_hint):
    _fake_audio(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    seen = []
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: (
        seen.append((url, kw)), _Response({"text": "No quiero cambiar el mensaje."})
    )[1])

    result = comparison.compare_text_candidate("source.mp4", provider, language_hint=language_hint)

    assert result["text"] == "No quiero cambiar el mensaje."
    assert result["has_word_timestamps"] is False
    assert result["selection_authority"] is False
    assert seen[0][0] == "https://api.openai.com/v1/audio/transcriptions"
    expected_data = {"model": provider}
    if provider == "gpt-transcribe" and language_hint:
        expected_data["languages[]"] = language_hint
    assert seen[0][1]["data"] == expected_data


def test_nova_multilingual_comparison_keeps_english_and_spanish(monkeypatch):
    _fake_audio(monkeypatch)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-only")
    seen = []
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: (
        seen.append((url, kw)), _Response({"results": {"channels": [
            {"alternatives": [{"transcript": "It happened, y entonces continué."}]}
        ]}})
    )[1])

    result = comparison.compare_text_candidate("source.mp4", "deepgram-nova-3-multi")

    assert result["text"] == "It happened, y entonces continué."
    assert seen[0][1]["params"] == {"model": "nova-3", "language": "multi"}
    assert result["selection_authority"] is False


@pytest.mark.parametrize("provider", ["gpt-transcribe", "gpt-4o-transcribe"])
def test_missing_api_key_never_silently_falls_back(monkeypatch, provider):
    _fake_audio(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        comparison.compare_text_candidate("source.mp4", provider)


def test_gpt_transcribe_http_failure_never_becomes_a_transcript(monkeypatch):
    _fake_audio(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    import requests

    class DeniedResponse:
        def raise_for_status(self):
            raise requests.HTTPError("403 model_not_found")

        def json(self):
            pytest.fail("must not consume an error response as a transcript")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: DeniedResponse())
    with pytest.raises(requests.HTTPError, match="403"):
        comparison.compare_text_candidate("source.mp4", "gpt-transcribe")


def test_unknown_provider_rejected_before_audio_or_network(monkeypatch):
    monkeypatch.setattr(comparison.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected ffmpeg"))
    with pytest.raises(ValueError, match="unsupported"):
        comparison.compare_text_candidate("source.mp4", "unknown")
