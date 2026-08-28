import json

from benchmarks.validate_video00_regression_qa import validate


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def manifest():
    return {
        "schema_version": "cutsell.video00.regression_qa.v1",
        "baseline_run_id": 1,
        "expected_selected_count": 3,
        "checks": [
            {"id": "good_a", "kind": "required_exact", "text": "Good A"},
            {"id": "bad_absent", "kind": "forbidden_contains", "text": "bad take"},
            {"id": "order", "kind": "required_order", "texts": ["Good A", "Good B", "CTA"]},
        ],
    }


def test_regression_qa_passes_clean_selection(tmp_path):
    result = {"selected": [{"text": "Good A"}, {"text": "Good B"}, {"text": "CTA"}]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest()),
    )
    assert ok is True
    assert report["failed_check_count"] == 0


def test_regression_qa_names_returned_historical_bug(tmp_path):
    result = {"selected": [{"text": "Good A"}, {"text": "bad take returned"}, {"text": "CTA"}]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest()),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "bad_absent" in ids
    assert "order" in ids


def test_regression_qa_catches_count_drift(tmp_path):
    result = {"selected": [{"text": "Good A"}, {"text": "Good B"}]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest()),
    )
    assert ok is False
    assert any(row["id"] == "selection_count_23" for row in report["failed_checks"])
