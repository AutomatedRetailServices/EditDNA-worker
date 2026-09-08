"""D-155 Phase A -- parallel perception orchestration.

Per docs/CUTSELL_DECISIONS.md D-154/D-155. `parallel_perception.py` runs
Track A (ASR), the real-signal half of Track B (`audio_silence.py`), and
Track C (`local_performance.py`) concurrently instead of sequentially --
same values, only scheduling changes. This suite formalizes the live-
execution proofs already produced by hand (see D-155's own session record)
into pytest: byte-identical values parallel vs. sequential, genuine wall-
clock overlap, deterministic name-keyed readback, and per-track failure
isolation (ASR mandatory/hard-fail; audio/visual fail-open, isolated).
"""
from __future__ import annotations

import time

import pytest

from cutsell_worker.contracts import SourceAsset, TranscriptSegment
from cutsell_worker.local_performance import LocalPerformanceResult, LocalPerformanceTimeline
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.parallel_perception import (
    TRACK_AUDIO,
    TRACK_SPEECH,
    TRACK_VISUAL,
    parallel_perception_enabled,
    run_parallel_perception,
)
from cutsell_worker.raw_understanding_map import TRACK_STATUS_FAILED, TRACK_STATUS_PASS


def _source(source_asset_id: str = "src-1") -> SourceAsset:
    return SourceAsset(
        source_asset_id=source_asset_id, project_id="p1", user_id="u1",
        original_name="a.mp4", source_order=0, duration_sec=5.0, uri="local://a.mp4",
        has_audio=True,
    )


class _FakeASR:
    def __init__(self, delay: float = 0.0, segments=None, raise_exc: Exception | None = None):
        self._delay = delay
        self._segments = segments if segments is not None else [
            TranscriptSegment(source_asset_id="src-1", start=0.0, end=1.0, text="hello")
        ]
        self._raise_exc = raise_exc

    def transcribe(self, path, *, source_asset_id, language_hint=None):
        if self._delay:
            time.sleep(self._delay)
        if self._raise_exc:
            raise self._raise_exc
        return list(self._segments)


def _local_perf_result(available: bool = True) -> LocalPerformanceResult:
    status = ProviderStatus("local_performance", True, available, "applied" if available else "provider_unavailable")
    timeline = LocalPerformanceTimeline(
        source_asset_id="src-1", observations=(), events=(),
        sampled_fps=12.0, source_fps=30.0, status=status,
    )
    return LocalPerformanceResult((timeline,), status)


# ---------------------------------------------------------------------------
# 1-2: env-flag rollback semantics.
# ---------------------------------------------------------------------------

def test_parallel_perception_enabled_defaults_true_when_env_absent():
    assert parallel_perception_enabled(env={}) is True


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off"])
def test_parallel_perception_enabled_false_for_recognized_falsy_values(value):
    assert parallel_perception_enabled(env={"CUTSELL_PARALLEL_PERCEPTION_ENABLED": value}) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_parallel_perception_enabled_true_for_recognized_truthy_values(value):
    assert parallel_perception_enabled(env={"CUTSELL_PARALLEL_PERCEPTION_ENABLED": value}) is True


# ---------------------------------------------------------------------------
# 3-5: byte-identical values, parallel vs. sequential.
# ---------------------------------------------------------------------------

def test_parallel_and_sequential_produce_identical_transcripts(monkeypatch):
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR()

    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", lambda paths: {})
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", lambda paths, target_fps=12.0: _local_perf_result())

    parallel = run_parallel_perception(sources, local_paths, asr, None, env={"CUTSELL_PARALLEL_PERCEPTION_ENABLED": "1"})
    sequential = run_parallel_perception(sources, local_paths, asr, None, env={"CUTSELL_PARALLEL_PERCEPTION_ENABLED": "0"})

    assert parallel.transcripts == sequential.transcripts
    assert parallel.audio_silence_by_source == sequential.audio_silence_by_source
    assert parallel.local_performance.status.status == sequential.local_performance.status.status


def test_sequential_mode_matches_pre_d155_call_order_values(monkeypatch):
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR(segments=[TranscriptSegment(source_asset_id="src-1", start=0.0, end=1.0, text="hi")])
    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", lambda paths: {"src-1": ()})
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", lambda paths, target_fps=12.0: _local_perf_result())

    outcome = run_parallel_perception(sources, local_paths, asr, None, env={"CUTSELL_PARALLEL_PERCEPTION_ENABLED": "0"})
    assert outcome.parallel_enabled is False
    assert [seg.text for seg in outcome.transcripts] == ["hi"]
    assert outcome.audio_silence_by_source == {"src-1": ()}


def test_deterministic_name_keyed_readback_regardless_of_which_track_declared_first(monkeypatch):
    # Track order in the `tasks` dict (visual last) must not affect the
    # returned outcome's own field identity -- always speech/audio/visual.
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR(delay=0.05)
    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", lambda paths: {})
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", lambda paths, target_fps=12.0: _local_perf_result())
    outcome = run_parallel_perception(sources, local_paths, asr, None)
    names = [t.name for t in outcome.track_timings]
    assert names == [TRACK_SPEECH, TRACK_AUDIO, TRACK_VISUAL]


# ---------------------------------------------------------------------------
# 6: genuine wall-clock overlap (formalizes the live smoke test).
# ---------------------------------------------------------------------------

def test_independent_tracks_genuinely_overlap_in_parallel_mode(monkeypatch):
    delay = 0.1
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR(delay=delay)

    def _slow_audio(paths):
        time.sleep(delay)
        return {}

    def _slow_visual(paths, target_fps=12.0):
        time.sleep(delay)
        return _local_perf_result()

    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", _slow_audio)
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", _slow_visual)

    outcome = run_parallel_perception(sources, local_paths, asr, None, env={"CUTSELL_PARALLEL_PERCEPTION_ENABLED": "1"})
    # 3 tracks x ~0.1s each sequentially would sum to ~0.3s; genuine overlap
    # keeps wall time close to one track's own duration.
    assert outcome.wall_time_ms < outcome.sum_track_time_ms * 0.8, "parallel overlap not observed"


def test_sequential_mode_wall_time_matches_sum_of_track_times(monkeypatch):
    delay = 0.05
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR(delay=delay)

    def _slow_audio(paths):
        time.sleep(delay)
        return {}

    def _slow_visual(paths, target_fps=12.0):
        time.sleep(delay)
        return _local_perf_result()

    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", _slow_audio)
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", _slow_visual)

    outcome = run_parallel_perception(sources, local_paths, asr, None, env={"CUTSELL_PARALLEL_PERCEPTION_ENABLED": "0"})
    # Sequential: wall time should be close to the sum of the 3 tracks (no overlap).
    assert outcome.wall_time_ms >= outcome.sum_track_time_ms * 0.9


# ---------------------------------------------------------------------------
# 7-10: failure semantics -- ASR mandatory hard-fail; audio/visual isolated.
# ---------------------------------------------------------------------------

def test_asr_failure_propagates_unmodified(monkeypatch):
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR(raise_exc=RuntimeError("asr boom"))
    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", lambda paths: {})
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", lambda paths, target_fps=12.0: _local_perf_result())

    with pytest.raises(RuntimeError, match="asr boom"):
        run_parallel_perception(sources, local_paths, asr, None)


def test_audio_track_failure_isolated_speech_and_visual_still_succeed(monkeypatch):
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR()

    def _raising_audio(paths):
        raise RuntimeError("audio boom")

    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", _raising_audio)
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", lambda paths, target_fps=12.0: _local_perf_result())

    outcome = run_parallel_perception(sources, local_paths, asr, None)
    assert outcome.audio_track_status == TRACK_STATUS_FAILED
    assert outcome.speech_track_status == TRACK_STATUS_PASS
    assert outcome.visual_track_status == TRACK_STATUS_PASS
    assert len(outcome.transcripts) == 1


def test_visual_track_failure_isolated_speech_and_audio_still_succeed(monkeypatch):
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR()

    def _raising_visual(paths, target_fps=12.0):
        raise RuntimeError("visual boom")

    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", lambda paths: {})
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", _raising_visual)

    outcome = run_parallel_perception(sources, local_paths, asr, None)
    assert outcome.visual_track_status == TRACK_STATUS_FAILED
    assert outcome.speech_track_status == TRACK_STATUS_PASS
    assert outcome.audio_track_status == TRACK_STATUS_PASS
    assert outcome.local_performance.status.status == "failed"


def test_visual_status_partial_when_local_performance_reports_unavailable(monkeypatch):
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR()
    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", lambda paths: {})
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", lambda paths, target_fps=12.0: _local_perf_result(available=False))

    outcome = run_parallel_perception(sources, local_paths, asr, None)
    from cutsell_worker.raw_understanding_map import TRACK_STATUS_PARTIAL
    assert outcome.visual_track_status == TRACK_STATUS_PARTIAL


# ---------------------------------------------------------------------------
# 11-12: diagnostics tail-safety + no provider/network reference.
# ---------------------------------------------------------------------------

def test_diagnostics_are_bounded_no_transcript_payload(monkeypatch):
    sources = (_source(),)
    local_paths = {"src-1": "/tmp/fake.mp4"}
    asr = _FakeASR(segments=[TranscriptSegment(source_asset_id="src-1", start=0.0, end=1.0, text="secret content")])
    monkeypatch.setattr("cutsell_worker.parallel_perception.audio_silence_events", lambda paths: {})
    monkeypatch.setattr("cutsell_worker.parallel_perception.analyze_local_performance", lambda paths, target_fps=12.0: _local_perf_result())

    outcome = run_parallel_perception(sources, local_paths, asr, None)
    diag = outcome.diagnostics()
    assert "secret content" not in repr(diag)
    assert diag["track_count"] == 3
    assert diag["tracks_failed"] == 0
    assert set(diag.keys()) == {
        "parallel_perception_enabled", "track_count", "tracks_started",
        "tracks_completed", "tracks_failed", "parallel_wall_time_ms",
        "sum_track_time_ms", "speech_track_status", "audio_track_status",
        "visual_track_status",
    }


def test_module_never_references_a_network_or_openai_provider():
    import cutsell_worker.parallel_perception as mod
    source = open(mod.__file__, encoding="utf-8").read()
    for forbidden in ("openai", "OpenAIVisualProvider", "whole_video_openai", "requests.", "httpx."):
        assert forbidden not in source, f"unexpected network/provider reference: {forbidden}"
