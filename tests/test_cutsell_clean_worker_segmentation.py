from cutsell_worker.contracts import SourceAsset, TranscriptSegment, Word
from cutsell_worker.take_segmentation import segment_takes


def _source(duration=8.0):
    return SourceAsset(
        source_asset_id="src-1",
        project_id="p1",
        user_id="u1",
        original_name="raw.mov",
        source_order=0,
        duration_sec=duration,
        uri="s3://bucket/raw.mov",
    )


def test_internal_long_pause_creates_separate_candidate_takes():
    source = _source()
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
    source = _source(5.0)
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


def test_short_open_asr_fragment_is_marked_incomplete_without_being_deleted():
    source = _source(5.0)
    segment = TranscriptSegment(
        "src-1", 1.0, 2.0, "You know",
        words=(
            Word("You", 1.0, 1.3, 0.9),
            Word("know", 1.4, 1.9, 0.9),
        ),
    )
    takes = segment_takes((segment,), (source,))
    assert len(takes) == 1
    assert takes[0].text == "You know"
    assert takes[0].complete_idea is False


def test_short_closed_line_remains_complete():
    source = _source(5.0)
    segment = TranscriptSegment(
        "src-1", 1.0, 2.0, "No way!",
        words=(
            Word("No", 1.0, 1.3, 0.9),
            Word("way!", 1.4, 1.9, 0.9),
        ),
    )
    takes = segment_takes((segment,), (source,))
    assert len(takes) == 1
    assert takes[0].complete_idea is True


def test_long_unpunctuated_speech_is_not_penalized_for_asr_punctuation_loss():
    source = _source(8.0)
    segment = TranscriptSegment(
        "src-1", 0.0, 4.0, "this is still a complete thought without punctuation",
        words=(
            Word("this", 0.0, 0.3, 0.9),
            Word("is", 0.4, 0.6, 0.9),
            Word("still", 0.7, 1.0, 0.9),
            Word("a", 1.1, 1.2, 0.9),
            Word("complete", 1.3, 1.8, 0.9),
            Word("thought", 1.9, 2.3, 0.9),
            Word("without", 2.4, 2.8, 0.9),
            Word("punctuation", 2.9, 3.6, 0.9),
        ),
    )
    takes = segment_takes((segment,), (source,))
    assert len(takes) == 1
    assert takes[0].complete_idea is True
