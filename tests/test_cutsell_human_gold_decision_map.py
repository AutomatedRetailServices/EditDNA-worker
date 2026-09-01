from cutsell_worker.human_gold_decision_map import (
    AlignmentAnchor,
    GoldSourceChunk,
    Span,
    build_decision_map,
    coalesce_alignment_anchors,
    complement_spans,
)


def chunk(index, raw_start, raw_end, *, gold_start=None, gold_end=None, confidence=0.95):
    gold_start = raw_start if gold_start is None else gold_start
    gold_end = (gold_start + (raw_end - raw_start)) if gold_end is None else gold_end
    return GoldSourceChunk(
        index=index,
        gold_start=gold_start,
        gold_end=gold_end,
        raw_start=raw_start,
        raw_end=raw_end,
        source_offset_sec=raw_start - gold_start,
        alignment_confidence=confidence,
    )


def test_complement_spans_maps_every_human_deleted_gap():
    deleted = complement_spans(
        20.0,
        (Span(2.0, 5.0), Span(7.0, 10.0), Span(10.0, 12.0), Span(18.0, 20.0)),
    )
    assert deleted == (Span(0.0, 2.0), Span(5.0, 7.0), Span(12.0, 18.0))


def test_alignment_plateaus_split_when_human_edit_jumps_to_new_raw_source_region():
    anchors = (
        AlignmentAnchor(0.5, 10.5, 0.96),
        AlignmentAnchor(0.75, 10.75, 0.95),
        AlignmentAnchor(1.0, 11.0, 0.97),
        AlignmentAnchor(1.25, 31.25, 0.96),
        AlignmentAnchor(1.5, 31.5, 0.95),
    )
    groups = coalesce_alignment_anchors(anchors)
    assert len(groups) == 2
    assert round(groups[0][0].offset, 3) == 10.0
    assert round(groups[1][0].offset, 3) == 30.0


def test_split_engine_clips_can_jointly_cover_one_human_gold_chunk():
    report = build_decision_map(
        raw_duration_sec=30.0,
        gold_duration_sec=10.0,
        gold_chunks=(chunk(1, 10.0, 20.0, gold_start=0.0, gold_end=10.0),),
        engine_result={
            "selected": [
                {"clip_id": "a", "start": 10.0, "end": 15.0, "text": "first clean subdelivery"},
                {"clip_id": "b", "start": 15.0, "end": 20.0, "text": "second clean subdelivery"},
            ],
            "alternates": [],
            "discarded": [],
        },
    )
    row = report["human_kept"][0]
    assert row["engine_selected_coverage"] == 1.0
    assert row["selection_pass"] is True
    assert report["selection_parity"]["gold_coverage_by_engine"] == 1.0


def test_human_gold_chunk_left_as_engine_alternate_is_best_take_mismatch():
    report = build_decision_map(
        raw_duration_sec=30.0,
        gold_duration_sec=6.0,
        gold_chunks=(chunk(1, 10.0, 16.0, gold_start=0.0, gold_end=6.0),),
        engine_result={
            "selected": [{"clip_id": "wrong", "start": 20.0, "end": 26.0, "text": "wrong retry"}],
            "alternates": [{"clip_id": "human", "start": 10.0, "end": 16.0, "text": "human selected delivery"}],
            "discarded": [],
        },
    )
    row = report["human_kept"][0]
    assert row["selection_pass"] is False
    assert row["rule_candidate"] == "best_take_or_grouping_mismatch"


def test_human_deleted_raw_region_kept_by_engine_is_false_keep():
    report = build_decision_map(
        raw_duration_sec=20.0,
        gold_duration_sec=5.0,
        gold_chunks=(chunk(1, 5.0, 10.0, gold_start=0.0, gold_end=5.0),),
        engine_result={
            "selected": [
                {"clip_id": "human", "start": 5.0, "end": 10.0, "text": "gold"},
                {"clip_id": "retry", "start": 12.0, "end": 16.0, "text": "retry human removed"},
            ],
            "alternates": [],
            "discarded": [],
        },
    )
    deleted = [
        row for row in report["human_deleted"]
        if row["raw_start"] <= 12.0 and row["raw_end"] >= 16.0
    ][0]
    assert deleted["engine_false_keep"] is True
    assert deleted["rule_candidate"] == "retry_or_slack_false_keep"


def test_selection_and_boundary_parity_are_reported_separately():
    report = build_decision_map(
        raw_duration_sec=20.0,
        gold_duration_sec=5.0,
        gold_chunks=(chunk(1, 5.0, 10.0, gold_start=0.0, gold_end=5.0),),
        engine_result={
            "selected": [
                {"clip_id": "near", "start": 4.8, "end": 10.7, "text": "same content loose boundary"}
            ],
            "alternates": [],
            "discarded": [],
        },
    )
    row = report["human_kept"][0]
    assert row["selection_pass"] is True
    assert row["rule_candidate"] == "boundary_authority_mismatch"
    assert report["boundary_parity"]["boundary_measurement_count"] == 2
