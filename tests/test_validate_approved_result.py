import json

from benchmarks.validate_approved_result import validate


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def _approval():
    return {
        "approval_id": "approved",
        "source_sha256": "abc",
        "boundary_tolerance_sec": 0.1,
        "text_coverage_floor": 0.9,
        "output_duration_sec": 7.0,
        "output_duration_tolerance_sec": 0.1,
        "live_render_qc_status": "PASS",
        "selected": [{"start": 10.0, "end": 17.0, "text": "Entrega íntegra y útil."}],
    }


def _result():
    return {
        "source_media_sha256": "abc", "output_duration_sec": 7.05, "deliverable": True,
        "live_render_qc": {"status": "PASS"},
        "selected": [{"clip_id": "new-id", "start": 10.05, "end": 17.05, "text": "Entrega integra y util."}],
    }


def test_approved_result_accepts_new_ids_and_accent_drift(tmp_path):
    ok, report = validate(_write(tmp_path, "r.json", _result()), _write(tmp_path, "a.json", _approval()))
    assert ok and report["failures"] == []


def test_approved_result_rejects_boundary_content_and_qc_regressions(tmp_path):
    result = _result()
    result["selected"][0].update(start=10.5, text="Otra entrega.")
    result["live_render_qc"]["status"] = "FAIL"
    ok, report = validate(_write(tmp_path, "r.json", result), _write(tmp_path, "a.json", _approval()))
    assert not ok
    assert {row["kind"] for row in report["failures"]} == {"boundary", "spoken_content", "live_render_qc"}
