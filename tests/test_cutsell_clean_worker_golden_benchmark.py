from cutsell_worker.golden_benchmark import evaluate_validation_report


def _report(selected=None, alternates=None, discarded=None):
    return {
        "source_key": "Editdna bloopers videos/example.mp4",
        "elapsed_sec": 2.0,
        "strategy": "mixed",
        "selected": selected or [],
        "alternates": alternates or [],
        "discarded": discarded or [],
        "stage_status": {
            "semantic": {"status": "applied"},
            "visual": {"status": "applied"},
            "take_judge": {"status": "provider_complete"},
        },
        "diagnostics": {"take_judge_status_counts": {"applied": 1}},
    }


def test_evaluator_accepts_valid_selected_and_same_group_alternate():
    report = _report(
        selected=[{
            "clip_id": "a", "start": 0.0, "end": 1.2, "text": "wow it is good",
            "take_group_id": "g1",
        }],
        alternates=[{
            "clip_id": "b", "start": 2.0, "end": 3.0, "text": "wow it is good",
            "take_group_id": "g1",
        }],
    )
    result = evaluate_validation_report(report)
    assert result["structural_pass"] is True
    assert result["retry_group_count"] == 1
    assert result["hard_failures"] == []
    assert result["provider_status"]["take_judge"] == "provider_complete"


def test_evaluator_flags_empty_timeline_and_clip_id_collision():
    report = _report(
        selected=[],
        alternates=[{"clip_id": "same", "start": 1, "end": 2, "text": "hello", "take_group_id": "g"}],
        discarded=[{"clip_id": "same", "start": 3, "end": 4, "text": "bad"}],
    )
    result = evaluate_validation_report(report)
    assert result["structural_pass"] is False
    assert "empty_selected_timeline" in result["hard_failures"]
    assert "clip_id_collision_across_draft_buckets" in result["hard_failures"]


def test_evaluator_warns_on_tiny_fragment_and_adjacent_duplicate_without_hard_failure():
    report = _report(selected=[
        {"clip_id": "a", "start": 0.0, "end": 0.2, "text": "out", "take_group_id": None},
        {"clip_id": "b", "start": 0.3, "end": 1.0, "text": "out", "take_group_id": None},
    ])
    result = evaluate_validation_report(report)
    assert result["structural_pass"] is True
    assert result["tiny_fragment_count"] == 1
    assert result["adjacent_duplicate_count"] == 1
    assert any(item.startswith("tiny_selected_fragments") for item in result["warnings"])
    assert any(item.startswith("adjacent_duplicate_selected_text") for item in result["warnings"])


def test_evaluator_rejects_orphan_alternate():
    report = _report(
        selected=[{"clip_id": "a", "start": 0, "end": 1, "text": "hello", "take_group_id": None}],
        alternates=[{"clip_id": "b", "start": 2, "end": 3, "text": "hello", "take_group_id": None}],
    )
    result = evaluate_validation_report(report)
    assert result["structural_pass"] is False
    assert "orphan_alternates:1" in result["hard_failures"]
