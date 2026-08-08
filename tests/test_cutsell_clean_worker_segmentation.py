from cutsell_worker.contracts import SourceAsset, TranscriptSegment, Word
from cutsell_worker.take_segmentation import segment_takes


def test_internal_long_pause_creates_separate_candidate_takes():
    source = SourceAsset(
        source_asset_id="src-1",
        project_id="p1",
        user_id="u1",
        original_name="raw.mov",
        source_order=0,
        duration_sec=8.0,
        uri="s3://bucket/raw.mov",
    )
    segment = TranscriptSegment(
        "src-1", 0.0, 6.0, "first attempt second attempt",
        words=(
            Word("first", 0.0, 0.3, 0.9),
            Word("attempt", 0.4, 0.8, 0.9),
            Word("second", 2.0, 2.4, 0.9),
            Word("attempt", 2.5, 2.9, 0.9),
        ),
    )
    takes = segment_takes((segment,), (source,))
    assert len(takes) == 2
    assert [take.text for take in takes] == ["first attempt", "second attempt"]
    assert all(take.source_asset_id == "src-1" for take in takes)


def test_normal_short_pause_stays_inside_same_take():
    source = SourceAsset("src-1", "p1", "u1", "raw.mov", 0, 5.0, "s3://bucket/raw.mov")
    segment = TranscriptSegment(
        "src-1", 0.0, 2.0, "this is natural",
        words=(
            Word("this", 0.0, 0.3, 0.9),
            Word("is", 0.5, 0.7, 0.9),
            Word("natural", 0.9, 1.3, 0.9),
        ),
    )
    takes = segment_takes((segment,), (source,))
    assert len(takes) == 1
    assert takes[0].text == "this is natural"
