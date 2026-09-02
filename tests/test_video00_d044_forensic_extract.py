"""D-044: pure extraction-logic tests for video00_d044_forensic_extract.py.
No S3/network dependency -- operates on a small local JSON fixture."""
from __future__ import annotations

import json

from benchmarks.video00_d044_forensic_extract import extract


def _write_fixture(tmp_path):
    result = {
        "benchmark_id": "video00-modal-test",
        "selected_count": 2,
        "source_duration_sec": 360.0,
        "selected": [
            {"clip_id": "clip_A", "text": "sonografia buena toma"},
            {"clip_id": "clip_B", "text": "algo no relacionado"},
        ],
        "discarded": [{"clip_id": "clip_C", "text": "sonografia mala toma"}],
        "diagnostics": {
            "attempt_reconstruction": {"attempt_count": 3},
            "take_grouping_status": {"status": "applied"},
            "take_grouping_reason": "baseline",
            "take_group_count": 1,
            "alternate_group_count": 1,
            "take_group_members": [["clip_A", "clip_C"]],
            "semantic_idea_equivalence": {"status": "applied", "pairs": []},
            "take_judge_status_counts": {"applied": 1},
            "take_judge_groups": [
                {"clip_id": "clip_A", "text": "sonografia buena toma", "reason": "winner"},
                {"clip_id": "clip_B", "text": "algo no relacionado", "reason": "n/a"},
            ],
            "clean_cut_decisions": [
                {"clip_id": "clip_A", "reason": "keep sonografia"},
                {"clip_id": "clip_D", "reason": "unrelated"},
            ],
            "hybrid_editorial_chunks": [],
            "claim_coverage_best_take": None,
            "final_selection_retry_arbiter": None,
            "canonical_edit_plan": {"ideas": [{"idea_id": "tg_1", "winning_clip_ids": ["clip_A"]}]},
            "final_story_coherence_validation": {"status": "applied"},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return str(path)


def test_extract_pulls_selected_and_discarded_clip_id_and_text(tmp_path):
    path = _write_fixture(tmp_path)
    forensic = extract(path)
    assert forensic["selected"] == [
        {"clip_id": "clip_A", "text": "sonografia buena toma"},
        {"clip_id": "clip_B", "text": "algo no relacionado"},
    ]
    assert forensic["discarded"] == [{"clip_id": "clip_C", "text": "sonografia mala toma"}]


def test_extract_pulls_grouping_and_arbiter_diagnostics_unfiltered_without_keywords(tmp_path):
    path = _write_fixture(tmp_path)
    forensic = extract(path)
    assert forensic["take_group_members"] == [["clip_A", "clip_C"]]
    assert forensic["semantic_idea_equivalence"] == {"status": "applied", "pairs": []}
    assert forensic["canonical_edit_plan_ideas"] == [{"idea_id": "tg_1", "winning_clip_ids": ["clip_A"]}]


def test_extract_filters_take_judge_groups_by_keyword(tmp_path):
    path = _write_fixture(tmp_path)
    forensic = extract(path, keywords=["sonografia"])
    ids = [item["clip_id"] for item in forensic["take_judge_groups_filtered"]]
    assert ids == ["clip_A"]  # clip_B's text doesn't match


def test_extract_filters_clean_cut_decisions_by_keyword():
    import tempfile
    import os

    result = {
        "diagnostics": {
            "clean_cut_decisions": [
                {"clip_id": "clip_X", "reason": "keep sonografia toma"},
                {"clip_id": "clip_Y", "reason": "discard unrelated"},
            ]
        }
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(result, fh)
        forensic = extract(path, keywords=["sonografia"])
        ids = [item["clip_id"] for item in forensic["clean_cut_decisions_filtered"]]
        assert ids == ["clip_X"]
    finally:
        os.unlink(path)


def test_extract_handles_missing_diagnostics_gracefully(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    forensic = extract(str(path))
    assert forensic["selected"] == []
    assert forensic["attempt_reconstruction"] is None


def test_main_requires_result_path_argument():
    from benchmarks.video00_d044_forensic_extract import main
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["video00_d044_forensic_extract.py"]
        assert main() == 2
    finally:
        sys.argv = old_argv
