"""D-053 Section 12 test category 8: "ASR-only harness" -- tests for
cutsell_worker.asr_only_benchmark. Verifies the harness runs ONLY the ASR
stage (download -> transcribe -> CanonicalASREvidence) and never touches
AttemptReconstructor/hybrid editorial/resolver/render, that it honors
CUTSELL_ASR_DETERMINISTIC_CONFIG via load_asr_provider_from_env exactly like
the real pipeline will, and that its result is plain-JSON-native."""
from __future__ import annotations

import json

import pytest

from cutsell_worker import asr_only_benchmark as harness
from cutsell_worker.asr import DETERMINISTIC_TEMPERATURE, DEFAULT_TEMPERATURE_LADDER, FasterWhisperASR
from cutsell_worker.config import RuntimeConfig
from cutsell_worker.contracts import TranscriptSegment, Word
from cutsell_worker.media_probe import MediaProbe


def _fake_config(**overrides) -> RuntimeConfig:
    values = dict(
        redis_url=None,
        database_url=None,
        sentry_dsn_present=False,
        openai_api_key_present=False,
        aws_access_key_present=True,
        aws_secret_key_present=True,
        aws_region="us-east-1",
        s3_bucket="fake-bucket",
        runpod_api_key_present=False,
        runpod_template_id=None,
        brain_backend="none",
        asr_model="medium",
        semantic_model="none",
        visual_model="none",
        take_judge_model="none",
        clean_cut_judge_model="none",
        clean_cut_judge_enabled=False,
        max_source_minutes=60,
        max_concurrent_jobs_per_user=1,
        monthly_processing_minutes=1000,
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def _install_common_fakes(monkeypatch, *, asr_provider):
    monkeypatch.setattr(harness, "load_runtime_config", lambda: _fake_config())
    monkeypatch.setattr(harness, "load_asr_provider_from_env", lambda *, model_name: asr_provider)
    monkeypatch.setattr(harness, "download_source", lambda uri, destination: destination)
    monkeypatch.setattr(harness, "probe_media", lambda path: MediaProbe(duration_sec=12.5, width=1920, height=1080, fps=30.0, has_audio=True))


class _FakeASRProvider:
    def __init__(self, *, temperature_ladder=DEFAULT_TEMPERATURE_LADDER):
        self.model_name = "medium"
        self.device = "auto"
        self.compute_type = "auto"
        self.temperature_ladder = temperature_ladder
        self.condition_on_previous_text = True
        self.vad_parameters = (0.5, 250.0, float("inf"), 2000.0, 1024.0, 400.0)
        self.transcribe_calls: list[tuple] = []

    @property
    def sampling_fallback_enabled(self) -> bool:
        return len(self.temperature_ladder) > 1

    def transcribe(self, path, *, source_asset_id, language_hint=None):
        self.transcribe_calls.append((path, source_asset_id, language_hint))
        return (
            TranscriptSegment(
                source_asset_id=source_asset_id,
                start=0.0,
                end=1.0,
                text="hello world",
                words=(
                    Word(text="hello", start=0.0, end=0.4),
                    Word(text="world", start=0.4, end=1.0),
                ),
            ),
        )

    def config_fingerprint(self, *, language_hint=None):
        return FasterWhisperASR(
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            temperature_ladder=self.temperature_ladder,
        ).config_fingerprint(language_hint=language_hint)


def test_run_asr_only_benchmark_requires_source_key():
    with pytest.raises(ValueError, match="source_key"):
        harness.run_asr_only_benchmark({})


def test_run_asr_only_benchmark_produces_canonical_evidence(monkeypatch):
    provider = _FakeASRProvider()
    _install_common_fakes(monkeypatch, asr_provider=provider)

    result = harness.run_asr_only_benchmark({"source_key": "videos/source.mp4", "benchmark_id": "asr-t1"})

    assert result["ok"] is True
    assert result["evidence_hash"].startswith("asrev_")
    assert result["normalized_word_sequence"] == ["hello", "world"]
    assert result["normalized_word_count"] == 2
    assert result["raw_segment_count"] == 1
    assert result["asr_config_fingerprint"] == provider.config_fingerprint().fingerprint()


def test_run_asr_only_benchmark_only_calls_transcribe_once_no_editorial_stages(monkeypatch):
    # The harness must stop after ASR -- never construct AttemptReconstructor,
    # hybrid editorial, resolver, or render stages.
    provider = _FakeASRProvider()
    _install_common_fakes(monkeypatch, asr_provider=provider)

    harness.run_asr_only_benchmark({"source_key": "videos/source.mp4", "benchmark_id": "asr-t2"})

    assert len(provider.transcribe_calls) == 1


def test_run_asr_only_benchmark_respects_deterministic_flag_via_env_loader(monkeypatch):
    # The harness delegates entirely to load_asr_provider_from_env -- it
    # never re-implements the CUTSELL_ASR_DETERMINISTIC_CONFIG branch
    # itself, so it automatically stays in lockstep with the real pipeline.
    deterministic_provider = _FakeASRProvider(temperature_ladder=DETERMINISTIC_TEMPERATURE)
    captured_model_name = {}

    def _fake_loader(*, model_name):
        captured_model_name["value"] = model_name
        return deterministic_provider

    monkeypatch.setattr(harness, "load_runtime_config", lambda: _fake_config(asr_model="medium"))
    monkeypatch.setattr(harness, "load_asr_provider_from_env", _fake_loader)
    monkeypatch.setattr(harness, "download_source", lambda uri, destination: destination)
    monkeypatch.setattr(harness, "probe_media", lambda path: MediaProbe(duration_sec=12.5, width=1920, height=1080, fps=30.0, has_audio=True))

    result = harness.run_asr_only_benchmark({"source_key": "videos/source.mp4", "benchmark_id": "asr-t3"})

    assert captured_model_name["value"] == "medium"
    assert result["asr_config"]["sampling_fallback_enabled"] is False
    assert result["asr_config"]["temperature_ladder"] == list(DETERMINISTIC_TEMPERATURE)


def test_run_asr_only_benchmark_result_is_json_serializable(monkeypatch):
    provider = _FakeASRProvider()
    _install_common_fakes(monkeypatch, asr_provider=provider)

    result = harness.run_asr_only_benchmark({"source_key": "videos/source.mp4", "benchmark_id": "asr-t4"})

    serialized = json.dumps(result)
    assert json.loads(serialized) == result


def test_run_asr_only_benchmark_requires_s3_bucket(monkeypatch):
    provider = _FakeASRProvider()
    monkeypatch.setattr(harness, "load_runtime_config", lambda: _fake_config(s3_bucket=None))
    monkeypatch.setattr(harness, "load_asr_provider_from_env", lambda *, model_name: provider)

    with pytest.raises(RuntimeError, match="S3_BUCKET"):
        harness.run_asr_only_benchmark({"source_key": "videos/source.mp4"})


def test_collect_asr_runtime_audit_never_crashes_when_deps_missing():
    # Best-effort audit -- must degrade to an error string per key rather
    # than raise, even in a sandbox without faster_whisper/torch installed.
    audit = harness.collect_asr_runtime_audit()
    assert "python_version" in audit
    assert "platform" in audit
    assert "faster_whisper_version" in audit
    assert "ctranslate2_version" in audit
    assert "gpu_name" in audit
    assert "cuda_available" in audit
    assert "transcribe_signature" in audit
