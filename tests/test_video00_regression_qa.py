import json

from benchmarks.validate_video00_regression_qa import validate


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


# Fuller placeholder phrases (not bare "Good A"/"Good B"/"CTA" labels) --
# after content-token filtering (D-032's coverage-based matching drops
# stopwords and tokens shorter than 3 chars), two texts distinguished only
# by a single letter would collapse to the same content and become
# indistinguishable to the matcher. Real transcript sentences never have
# this problem; these fixtures avoid it deliberately.
def manifest():
    return {
        "schema_version": "cutsell.video00.regression_qa.v1",
        "baseline_run_id": 1,
        "expected_selected_count": 3,
        "checks": [
            {"id": "alpha_present", "kind": "required_exact", "text": "the product launched successfully"},
            {"id": "bad_absent", "kind": "forbidden_contains", "text": "bad take"},
            {"id": "order", "kind": "required_order", "texts": [
                "the product launched successfully",
                "customers loved the results",
                "call to action asking viewers to subscribe",
            ]},
        ],
    }


def test_regression_qa_passes_clean_selection(tmp_path):
    result = {"selected": [
        {"text": "the product launched successfully"},
        {"text": "customers loved the results"},
        {"text": "call to action asking viewers to subscribe"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest()),
    )
    assert ok is True
    assert report["failed_check_count"] == 0


def test_regression_qa_names_returned_historical_bug(tmp_path):
    result = {"selected": [
        {"text": "the product launched successfully"},
        {"text": "bad take returned"},
        {"text": "call to action asking viewers to subscribe"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest()),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "bad_absent" in ids
    assert "order" in ids  # "customers loved the results" is genuinely missing


def test_regression_qa_count_drift_alone_is_a_warning_not_a_failure(tmp_path):
    # D-032: a changed segment count is not, by itself, evidence of real
    # content loss -- benign re-chunking changes count while every idea
    # stays present. Recorded as a warning; qa_pass is driven by whether
    # the actual required facts are still there, not by count equality.
    result = {"selected": [
        {"text": "the product launched successfully and customers loved the results"},
        {"text": "call to action asking viewers to subscribe"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest()),
    )
    assert ok is True
    assert report["failed_check_count"] == 0
    assert any(row["id"] == "selection_count_23" for row in report["warnings"])


def test_regression_qa_still_fails_when_a_required_fact_is_genuinely_missing(tmp_path):
    result = {"selected": [
        {"text": "the product launched successfully"},
        {"text": "some unrelated closing remark"},
    ]}
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest()),
    )
    assert ok is False
    ids = {row["id"] for row in report["failed_checks"]}
    assert "order" in ids
    # The count drift is still recorded, just as a warning, not a failure.
    assert any(row["id"] == "selection_count_23" for row in report["warnings"])
