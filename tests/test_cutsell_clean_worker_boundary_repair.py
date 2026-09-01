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


def test_open_leadin_fragment_is_joined_forward_before_best_take():
    segments = (
        TranscriptSegment(
            source_asset_id="src-1",
            start=1.0,
            end=1.55,
            text="So",
            words=(Word("So", 1.0, 1.55, 0.9),),
        ),
        TranscriptSegment(
            source_asset_id="src-1",
            start=1.55,
            end=3.2,
            text="one time I was in LA.",
            words=(
                Word("one", 1.55, 1.8, 0.9),
                Word("time", 1.8, 2.0, 0.9),
                Word("I", 2.0, 2.1, 0.9),
                Word("was", 2.1, 2.4, 0.9),
                Word("in", 2.4, 2.6, 0.9),
                Word("LA.", 2.6, 3.2, 0.9),
            ),
        ),
    )
    takes = segment_takes(segments, (_source(),))
    assert len(takes) == 1
    assert takes[0].text == "So one time I was in LA."
    assert takes[0].start == 1.0
    assert takes[0].end == 3.2


def test_open_micro_fragment_chain_is_coalesced_until_meaningful():
    segments = (
        TranscriptSegment("src-1", 1.0, 1.6, "I", (Word("I", 1.0, 1.6),)),
        TranscriptSegment("src-1", 1.6, 2.3, "think", (Word("think", 1.6, 2.3),)),
        TranscriptSegment(
            "src-1",
            2.3,
            4.5,
            "this was the funniest one.",
            (
                Word("this", 2.3, 2.6),
                Word("was", 2.6, 2.9),
                Word("the", 2.9, 3.1),
                Word("funniest", 3.1, 3.8),
                Word("one.", 3.8, 4.5),
            ),
        ),
    )
    takes = segment_takes(segments, (_source(),))
    assert len(takes) == 1
    assert takes[0].text == "I think this was the funniest one."


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


def test_boundary_repair_does_not_bridge_real_pause():
    segments = (
        TranscriptSegment("src-1", 1.0, 1.5, "So", (Word("So", 1.0, 1.5),)),
        TranscriptSegment(
            "src-1",
            2.0,
            4.0,
            "this is a new thought.",
            (Word("this", 2.0, 2.2), Word("is", 2.2, 2.4), Word("new", 2.8, 3.2), Word("thought.", 3.2, 4.0)),
        ),
    )
    takes = segment_takes(segments, (_source(),))
    assert [take.text for take in takes] == ["So", "this is a new thought."]


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
