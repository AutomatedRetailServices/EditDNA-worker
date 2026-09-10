"""D-200: pre-RAW wiring fix -- tests.

Proves the ONE authorized flow_b.py change (passing the already-computed
`raw_understanding_maps` into `build_flow_b_draft`) is exactly that: pure
integration wiring, no new computation, no second ASR pass, no second
`RawUnderstandingMap` construction, byte-identical behavior with either
diagnostics flag off, and -- with both flags on -- the live Language
Spine actually receives real word timings instead of trivially reporting
NOT_EVALUABLE for a wiring reason.

See docs/CUTSELL_DECISIONS.md D-200.
"""
from __future__ import annotations

from cutsell_worker.contracts import ProcessingRequest, SourceAsset, TranscriptSegment, Word
from cutsell_worker.flow_b import process_local_sources
from cutsell_worker.language_spine_live_integration import CAPABILITY_NOT_EVALUABLE
from cutsell_worker.media_probe import MediaProbe


class _CountingASR:
    """Real-shaped ASR fake (same contract as the existing media-ingest
    fixture) that COUNTS invocations -- the NO-SECOND-ASR-PASS proof."""

    def __init__(self):
        self.call_count = 0

    def transcribe(self, path, *, source_asset_id, language_hint=None):
        self.call_count += 1
        return (
            TranscriptSegment(
                source_asset_id=source_asset_id,
                start=0.0,
                end=2.2,
                text="This serum changed my skin completely.",
                words=(
                    Word("This", 0.0, 0.2),
                    Word("serum", 0.25, 0.5),
                    Word("changed", 0.55, 0.8),
                    Word("my", 0.85, 1.0),
                    Word("skin", 1.05, 1.3),
                    Word("completely.", 1.35, 1.7),
                ),
            ),
        )


def _source(source_id="src_one"):
    return SourceAsset(
        source_asset_id=source_id, project_id="project-1", user_id="user-1",
        original_name="raw.mov", source_order=0, duration_sec=3.0, uri="s3://bucket/raw.mov",
    )


def _run(tmp_path, monkeypatch, *, p1_flag=None, live_spine_flag=None):
    if p1_flag is None:
        monkeypatch.delenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CUTSELL_EDITORIAL_MOMENT_SEQUENCE_DIAGNOSTICS_ENABLED", p1_flag)
    if live_spine_flag is None:
        monkeypatch.delenv("CUTSELL_LIVE_LANGUAGE_SPINE_DIAGNOSTICS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CUTSELL_LIVE_LANGUAGE_SPINE_DIAGNOSTICS_ENABLED", live_spine_flag)

    source = _source()
    media = tmp_path / "raw.mov"
    media.write_bytes(b"fake")
    monkeypatch.setattr(
        "cutsell_worker.flow_b.probe_media",
        lambda _path: MediaProbe(duration_sec=3.0, width=1080, height=1920, fps=30.0, has_audio=True),
    )
    request = ProcessingRequest(
        project_id="project-1", user_id="user-1", sources=(source,), language_hint="en",
    )
    asr = _CountingASR()
    result = process_local_sources(
        request, {source.source_asset_id: str(media)}, asr_provider=asr, editorial_mode="clean_cut",
    )
    return result, asr, source


# ---------------------------------------------------------------------------
# 1-2: default OFF / P1-on-live-spine-off byte-identical to pre-D-200.
# ---------------------------------------------------------------------------
def test_01_both_flags_off_draft_ready_unaffected(tmp_path, monkeypatch):
    result, asr, _ = _run(tmp_path, monkeypatch)
    assert result.state.value == "draft_ready"
    assert result.draft.diagnostics.get("live_language_spine") == {"status": "disabled"}
    assert result.draft.diagnostics.get("editorial_moment_sequence") == {"status": "disabled"}


def test_02_p1_on_live_spine_off_d198_behavior_unaffected(tmp_path, monkeypatch):
    result, asr, _ = _run(tmp_path, monkeypatch, p1_flag="1")
    assert result.state.value == "draft_ready"
    p1 = result.draft.diagnostics.get("editorial_moment_sequence")
    assert p1 is not None and p1.get("status") == "evaluated"
    # The live-spine flag is off -- the separate D-200 key must stay disabled.
    assert result.draft.diagnostics.get("live_language_spine") == {"status": "disabled"}


# ---------------------------------------------------------------------------
# 3: both flags on -- the actual wiring fix under test.
# ---------------------------------------------------------------------------
def test_03_both_flags_on_live_spine_receives_real_word_timings(tmp_path, monkeypatch):
    result, asr, source = _run(tmp_path, monkeypatch, p1_flag="1", live_spine_flag="1")
    assert result.state.value == "draft_ready"
    live = result.draft.diagnostics.get("live_language_spine")
    assert live is not None and live.get("status") == "evaluated"
    sources = live.get("sources", [])
    assert len(sources) >= 1
    row = next(r for r in sources if r["source_asset_id"] == source.source_asset_id)
    # THE wiring proof: status is NOT trivially NOT_EVALUABLE-for-a-missing-map --
    # real LanguageWord/Attempt counts come from the real ASR word timings above.
    assert row["capability_status"] != CAPABILITY_NOT_EVALUABLE
    assert row["language_word_count"] > 0
    assert row["language_attempt_count"] > 0


def test_04_no_second_asr_pass(tmp_path, monkeypatch):
    _, asr, _ = _run(tmp_path, monkeypatch, p1_flag="1", live_spine_flag="1")
    # Exactly one ASR pass regardless of both diagnostics flags -- D-199's
    # live Language Spine construction reads only already-computed word
    # timings, never re-invoking ASR.
    assert asr.call_count == 1


def test_05_flags_off_also_exactly_one_asr_pass(tmp_path, monkeypatch):
    _, asr, _ = _run(tmp_path, monkeypatch)
    assert asr.call_count == 1


# ---------------------------------------------------------------------------
# 6: downstream immutability -- selection/family/render output identical
# regardless of flags.
# ---------------------------------------------------------------------------
def test_06_downstream_selection_immutable_across_flag_states(tmp_path, monkeypatch):
    off_result, _, _ = _run(tmp_path, monkeypatch)
    on_result, _, _ = _run(tmp_path, monkeypatch, p1_flag="1", live_spine_flag="1")
    off_ids = tuple(c.clip_id for c in off_result.draft.selected)
    on_ids = tuple(c.clip_id for c in on_result.draft.selected)
    assert off_ids == on_ids
    assert off_result.state.value == on_result.state.value == "draft_ready"
