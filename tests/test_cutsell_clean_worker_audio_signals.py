from cutsell_worker.contracts import SourceAsset, TranscriptSegment, Word
from cutsell_worker.take_segmentation import segment_takes


def _source():
    return SourceAsset(
        source_asset_id="src-1",
        project_id="p1",
        user_id="u1",
        original_name="raw.mov",
        source_order=0,
        duration_sec=5.0,
        uri="s3://bucket/raw.mov",
    )


def test_whisper_confidence_contributes_to_audio_quality():
    strong = TranscriptSegment(
        "src-1", 0.0, 2.0, "clear creator delivery",
        words=(
            Word("clear", 0.0, 0.4, 0.98),
            Word("creator", 0.5, 1.0, 0.96),
            Word("delivery", 1.1, 1.6, 0.97),
        ),
    )
    weak = TranscriptSegment(
        "src-1", 2.0, 4.0, "unclear creator delivery",
        words=(
            Word("unclear", 2.0, 2.4, 0.35),
            Word("creator", 2.5, 3.0, 0.40),
            Word("delivery", 3.1, 3.6, 0.38),
        ),
    )
    takes = segment_takes((strong, weak), (_source(),))
    assert takes[0].signals.audio_quality > takes[1].signals.audio_quality
