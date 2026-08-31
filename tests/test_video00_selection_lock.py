import json

from benchmarks.validate_video00_selection_lock import validate


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def lock():
    return {
        "schema_version": "cutsell.video00.selection_lock.v1",
        "baseline_run_id": 1,
        "selected_count": 3,
        "segments": [
            {"clip_id": "a", "text": "the product launched successfully"},
            {"clip_id": "b", "text": "customers loved the results"},
            {"clip_id": "c", "text": "call to action asking viewers to subscribe"},
        ],
    }


def test_identical_selection_locks(tmp_path):
    result = {"selected_count": 3, "selected": [
        {"clip_id": "a", "text": "the product launched successfully"},
        {"clip_id": "b", "text": "customers loved the results"},
        {"clip_id": "c", "text": "call to action asking viewers to subscribe"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "lock.json", lock()),
    )
    assert ok is True
    assert report["selection_locked"] is True
    assert report["error_count"] == 0


def test_benign_rechunk_does_not_break_the_lock(tmp_path):
    # D-032: the exact failure shape RAW 33402023395 hit -- a gold segment
    # got merged with its neighbor into one candidate segment, changing the
    # count without losing any content. Must not be reported as an error.
    result = {"selected_count": 2, "selected": [
        {"clip_id": "a", "text": "the product launched successfully and customers loved the results"},
        {"clip_id": "c", "text": "call to action asking viewers to subscribe"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "lock.json", lock()),
    )
    assert ok is True
    assert report["selection_locked"] is True
    assert report["actual_selected_count"] == 2
    assert report["expected_selected_count"] == 3
    relations = [row["relation"] for row in report["alignment"]]
    assert "RECHUNKED" in relations


def test_genuinely_missing_gold_content_breaks_the_lock(tmp_path):
    result = {"selected_count": 2, "selected": [
        {"clip_id": "a", "text": "the product launched successfully"},
        {"clip_id": "c", "text": "call to action asking viewers to subscribe"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "lock.json", lock()),
    )
    assert ok is False
    assert report["selection_locked"] is False
    reasons = [row["reason"] for row in report["errors"]]
    assert "missing_segment" in reasons


def test_duplicate_rendered_segment_breaks_the_lock(tmp_path):
    result = {"selected_count": 4, "selected": [
        {"clip_id": "a", "text": "the product launched successfully"},
        {"clip_id": "a2", "text": "the product launched successfully"},
        {"clip_id": "b", "text": "customers loved the results"},
        {"clip_id": "c", "text": "call to action asking viewers to subscribe"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "lock.json", lock()),
    )
    assert ok is False
    reasons = [row["reason"] for row in report["errors"]]
    assert "duplicate_segment" in reasons
