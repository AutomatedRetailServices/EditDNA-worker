import cutsell_worker.golden_benchmark as golden


def _report(key="Editdna bloopers videos/Bloppers1.mp4"):
    return {
        "source_key": key,
        "source_window": None,
        "source_duration_sec": 120.0,
        "analyzed_duration_sec": 120.0,
        "selected_duration_sec": 30.0,
        "elapsed_sec": 1.0,
        "strategy": "storytelling",
        "selected": [
            {
                "clip_id": "c1",
                "start": 0.0,
                "end": 30.0,
                "text": "This is the selected take",
                "semantic_role": "STORY",
                "take_group_id": "g1",
            }
        ],
        "alternates": [],
        "discarded": [],
        "diagnostics": {
            "take_grouping_provider_status": {"status": "complete"},
            "composer_provider_status": {"status": "complete"},
        },
        "stage_status": {
            "whole_video_context": {"status": "complete"},
            "semantic": {"status": "complete"},
            "visual": {"status": "complete"},
            "take_judge": {"status": "complete"},
            "clean_cut": {"status": "complete"},
        },
    }


def test_full_mode_calls_validation_without_window(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        golden,
        "_source_records",
        lambda **_kwargs: ({"key": "Editdna bloopers videos/Bloppers1.mp4", "size": 123},),
    )

    def fake_validation(key, **kwargs):
        calls.append((key, kwargs))
        return _report(key)

    monkeypatch.setattr(golden, "run_single_validation", fake_validation)
    report = golden.run_golden_benchmark(
        video_limit=1,
        mode="full",
        window_sec=60,
        preview_dir=str(tmp_path),
    )

    assert report["schema_version"] == "cutsell.golden.v2"
    assert report["mode"] == "full"
    assert report["window_sec"] is None
    assert report["total_input_duration_sec"] == 120.0
    assert report["total_selected_duration_sec"] == 30.0
    assert report["selected_to_input_ratio"] == 0.25
    assert "source_start_sec" not in calls[0][1]
    assert "source_end_sec" not in calls[0][1]


def test_bounded_mode_passes_window(monkeypatch):
    calls = []
    monkeypatch.setattr(
        golden,
        "_source_records",
        lambda **_kwargs: ({"key": "Editdna bloopers videos/Bloppers1.mp4", "size": 123},),
    )

    def fake_validation(key, **kwargs):
        calls.append((key, kwargs))
        report = _report(key)
        report["source_window"] = {"start_sec": 0.0, "end_sec": 90.0}
        report["analyzed_duration_sec"] = 90.0
        return report

    monkeypatch.setattr(golden, "run_single_validation", fake_validation)
    report = golden.run_golden_benchmark(video_limit=1, mode="bounded", window_sec=90)
    assert report["mode"] == "bounded"
    assert report["window_sec"] == 90.0
    assert calls[0][1]["source_start_sec"] == 0.0
    assert calls[0][1]["source_end_sec"] == 90.0


def test_evaluation_surfaces_new_brain_provider_statuses():
    evaluated = golden.evaluate_validation_report(_report())
    statuses = evaluated["provider_status"]
    assert statuses["whole_video"] == "complete"
    assert statuses["take_grouping"] == "complete"
    assert statuses["composer"] == "complete"
    assert evaluated["selected_to_input_ratio"] == 0.25
