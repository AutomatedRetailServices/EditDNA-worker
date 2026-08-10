from types import SimpleNamespace

import cutsell_worker.universal_clean_cut as universal


def test_universal_clean_cut_disables_editorial_layers(monkeypatch):
    captured = {}

    def fake_process(request, local_paths, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            schema_version="cutsell.v1",
            project_id="p1",
            state="draft_ready",
            draft=SimpleNamespace(),
            stage_status={"clean_cut": "context_aware_deterministic_complete"},
        )

    monkeypatch.setattr(universal, "process_local_sources", fake_process)

    result = universal.process_universal_clean_cut_sources(
        object(),
        {},
        asr_provider=object(),
        visual_provider=object(),
        take_judge_provider=object(),
        clean_cut_provider=object(),
        take_grouping_provider=object(),
        whole_video_provider=object(),
    )

    assert isinstance(captured["semantic_provider"], universal.NoopSemanticProvider)
    assert captured["composer_provider"] is None
    assert captured["draft_review_provider"] is None
    assert captured["take_grouping_provider"] is not None
    assert captured["take_judge_provider"] is not None
    assert captured["whole_video_provider"] is not None
    assert captured["visual_provider"] is not None
    assert result.stage_status["brain_mode"] == "universal_clean_cut"
    assert result.stage_status["semantic"] == "not_requested_clean_cut_only"
    assert result.stage_status["composer"] == "not_requested_clean_cut_only"
    assert result.stage_status["draft_review"] == "not_requested_clean_cut_only"
