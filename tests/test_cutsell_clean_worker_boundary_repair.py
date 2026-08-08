from cutsell_worker.contracts import SourceAsset, TranscriptSegment, Word
from cutsell_worker.take_segmentation import segment_takes


def _source(source_id="src-1"):
    return SourceAsset(
        source_asset_id=source_id,
        project_id="p1",
        user_id="u1",
        original_name="raw.mov",
        source_order=0,
        duration_sec=20.0,
        uri="s3://bucket/raw.mov",
    )


def test_adjacent_one_word_tail_is_reattached_to_open_phrase():
    segments = (
        TranscriptSegment(
            source_asset_id="src-1",
            start=1.0,
            end=2.0,
            text="Use them for working",
            words=(
                Word("Use", 1.0, 1.2, 0.9),
                Word("them", 1.2, 1.4, 0.9),
                Word("for", 1.4, 1.6, 0.9),
                Word("working", 1.6, 2.0, 0.9),
            ),
        ),
        TranscriptSegment(
            source_asset_id="src-1",
            start=2.0,
            end=2.32,
            text="out.",
            words=(Word("out.", 2.0, 2.32, 0.9),),
        ),
    )
    takes = segment_takes(segments, (_source(),))
    assert len(takes) == 1
    assert takes[0].text == "Use them for working out."
    assert takes[0].start == 1.0
    assert takes[0].end == 2.32


def test_short_standalone_utterance_after_finished_sentence_is_preserved():
    segments = (
        TranscriptSegment(
            source_asset_id="src-1",
            start=1.0,
            end=2.0,
            text="This feels amazing.",
            words=(Word("This", 1.0, 1.2), Word("feels", 1.2, 1.5), Word("amazing.", 1.5, 2.0)),
        ),
        TranscriptSegment(
            source_asset_id="src-1",
            start=2.04,
            end=2.38,
            text="Wow!",
            words=(Word("Wow!", 2.04, 2.38),),
        ),
    )
    takes = segment_takes(segments, (_source(),))
    assert [take.text for take in takes] == ["This feels amazing.", "Wow!"]


def test_boundary_repair_never_crosses_sources():
    second = SourceAsset(
        source_asset_id="src-2",
        project_id="p1",
        user_id="u1",
        original_name="raw2.mov",
        source_order=1,
        duration_sec=20.0,
        uri="s3://bucket/raw2.mov",
    )
    segments = (
        TranscriptSegment("src-1", 1.0, 2.0, "Use them for working", (Word("working", 1.6, 2.0),)),
        TranscriptSegment("src-2", 0.0, 0.3, "out.", (Word("out.", 0.0, 0.3),)),
    )
    takes = segment_takes(segments, (_source(), second))
    assert len(takes) == 2
    assert {take.source_asset_id for take in takes} == {"src-1", "src-2"}
