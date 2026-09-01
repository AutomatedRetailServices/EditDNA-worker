from pathlib import Path

from cutsell_worker.asr import ASRProvider
from cutsell_worker.contracts import ProcessingRequest, SourceAsset, TranscriptSegment, Word
from cutsell_worker.flow_b import process_local_sources
from cutsell_worker.media_probe import MediaProbe
from cutsell_worker.providers import ProviderStatus, SemanticProviderResult
from cutsell_worker.silence_analysis import silence_ratio, word_silence_gaps
from cutsell_worker.take_segmentation import segment_takes


class FakeASR:
    def transcribe(self, path, *, source_asset_id, language_hint=None):
        return (
            TranscriptSegment(
                source_asset_id=source_asset_id,
                start=0.0,
                end=2.2,
                text="This serum changed my skin",
                words=(
                    Word("This", 0.0, 0.2),
                    Word("serum", 0.25, 0.5),
                    Word("changed", 1.2, 1.5),
                    Word("my", 1.55, 1.7),
                    Word("skin", 1.8, 2.2),
                ),
            ),
        )


class BrokenSemantic:
    def classify(self, takes):
        raise RuntimeError("provider unavailable")


def _source(source_id="src_one"):
    return SourceAsset(
        source_asset_id=source_id,
        project_id="project-1",
        user_id="user-1",
        original_name="raw.mov",
        source_order=0,
        duration_sec=3.0,
        uri="s3://bucket/raw.mov",
    )


def test_word_gaps_and_segmentation_preserve_source_identity():
    source = _source()
    segments = FakeASR().transcribe("ignored", source_asset_id=source.source_asset_id)
    gaps = word_silence_gaps(segments, min_gap_sec=0.45)
    assert len(gaps) == 1
    assert gaps[0].source_asset_id == source.source_asset_id
    assert gaps[0].duration_sec == 0.7

    takes = segment_takes(segments, (source,), gaps)
    assert len(takes) == 1
    assert takes[0].source_asset_id == source.source_asset_id
    assert takes[0].signals is not None
    assert takes[0].signals.silence_ratio > 0


def test_segmentation_rejects_unregistered_transcript_source():
    segment = TranscriptSegment("unknown", 0.0, 1.0, "hello")
    try:
        segment_takes((segment,), (_source(),))
    except ValueError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("unregistered source must be rejected")


def test_real_flow_b_fails_open_when_semantic_provider_breaks_in_full_mode(tmp_path, monkeypatch):
    source = _source()
    media = tmp_path / "raw.mov"
    media.write_bytes(b"fake")

    monkeypatch.setattr(
        "cutsell_worker.flow_b.probe_media",
        lambda _path: MediaProbe(duration_sec=3.0, width=1080, height=1920, fps=30.0, has_audio=True),
    )
    request = ProcessingRequest(
        project_id="project-1",
        user_id="user-1",
        sources=(source,),
        language_hint="en",
    )
    result = process_local_sources(
        request,
        {source.source_asset_id: str(media)},
        asr_provider=FakeASR(),
        semantic_provider=BrokenSemantic(),
        editorial_mode="full",
    )
    assert result.state.value == "draft_ready"
    assert result.draft.selected
    assert result.draft.selected[0].source_asset_id == source.source_asset_id
    assert result.stage_status["semantic"]["status"] == "degraded"
    assert result.stage_status["take_segmentation"]["candidate_count"] == 1
