from cutsell_worker.human_gold_decision_map import AlignmentAnchor, GoldSourceChunk
from cutsell_worker.human_gold_decision_map_v2 import _top_indices


def test_top_indices_orders_best_scores_first():
    import numpy as np
    values = np.array([0.1, 0.9, 0.4, 0.8, 0.2], dtype=np.float32)
    assert list(_top_indices(values, 3)) == [1, 3, 2]


def test_alignment_anchor_offset_is_source_minus_gold():
    anchor = AlignmentAnchor(gold_time=12.0, raw_time=35.5, correlation=0.95)
    assert anchor.offset == 23.5


def test_gold_source_chunk_preserves_gold_duration():
    chunk = GoldSourceChunk(
        index=1,
        gold_start=10.0,
        gold_end=15.25,
        raw_start=42.0,
        raw_end=47.25,
        source_offset_sec=32.0,
        alignment_confidence=0.9,
    )
    assert chunk.duration == 5.25
    assert chunk.raw_span.start == 42.0
    assert chunk.raw_span.end == 47.25
