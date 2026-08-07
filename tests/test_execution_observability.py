from worker.execution_observability import attach_semantic_execution


def _clip(clip_id, source_index=0, semantic_v2=None):
    meta = {"keep": True}
    if semantic_v2 is not None:
        meta["semantic_v2"] = semantic_v2
    return {
        "id": clip_id,
        "source_index": source_index,
        "text": "valid spoken clip",
        "meta": meta,
    }


def test_semantic_execution_reports_provider_unavailable_fallback():
    result = {
        "processed_source_indices": [0],
        "clips": [_clip("a")],
    }
    attach_semantic_execution(
        result,
        requested=True,
        provider_available=False,
    )
    execution = result["semantic_execution"]
    assert execution["requested"] is True
    assert execution["status_counts"] == {"provider_unavailable": 1}
    assert execution["clip_status_counts"] == {"provider_unavailable": 1}
    assert execution["fallback_reasons"] == {"provider_unavailable": 1}
    meta = result["clips"][0]["meta"]
    assert meta["semantic_v2_execution_status"] == "provider_unavailable"
    assert meta["semantic_v2_fallback_reason"] == "provider_unavailable"


def test_semantic_execution_reports_partial_application_and_abstention():
    result = {
        "processed_source_indices": [0],
        "clips": [
            _clip("a", semantic_v2={"applied": True, "abstain": False}),
            _clip("b", semantic_v2={"applied": False, "abstain": True}),
            _clip("c"),
        ],
    }
    attach_semantic_execution(
        result,
        requested=True,
        provider_available=True,
    )
    execution = result["semantic_execution"]
    source = execution["sources"][0]
    assert source["status"] == "partially_applied"
    assert source["semantic_results"] == 2
    assert source["applied"] == 1
    assert source["abstained"] == 1
    assert source["fallback_reasons"] == {
        "model_abstained": 1,
        "missing_semantic_result": 1,
    }
    assert execution["clip_status_counts"] == {
        "applied": 1,
        "abstained": 1,
        "classifier_no_result": 1,
    }
    assert result["clips"][0]["meta"]["semantic_v2_execution_status"] == "applied"
    assert "semantic_v2_fallback_reason" not in result["clips"][0]["meta"]
    assert result["clips"][1]["meta"]["semantic_v2_fallback_reason"] == "model_abstained"
    assert result["clips"][2]["meta"]["semantic_v2_fallback_reason"] == "missing_semantic_result"


def test_semantic_execution_reports_unapplied_result_as_safe_fallback():
    result = {
        "processed_source_indices": [0],
        "clips": [_clip("a", semantic_v2={"applied": False, "abstain": False})],
    }
    attach_semantic_execution(
        result,
        requested=True,
        provider_available=True,
    )
    source = result["semantic_execution"]["sources"][0]
    assert source["status"] == "fallback_only"
    assert source["fallback_reasons"] == {"unsafe_to_apply": 1}
    assert result["clips"][0]["meta"]["semantic_v2_execution_status"] == "fallback_only"
    assert result["clips"][0]["meta"]["semantic_v2_fallback_reason"] == "unsafe_to_apply"


def test_semantic_execution_reports_not_requested_explicitly():
    result = {
        "processed_source_indices": [0],
        "clips": [_clip("a")],
    }
    attach_semantic_execution(
        result,
        requested=False,
        provider_available=True,
    )
    source = result["semantic_execution"]["sources"][0]
    assert source["status"] == "not_requested"
    assert source["fallback_reasons"] == {}
    assert result["clips"][0]["meta"]["semantic_v2_execution_status"] == "not_requested"
    assert "semantic_v2_fallback_reason" not in result["clips"][0]["meta"]


def test_semantic_execution_is_propagated_to_editable_draft_items():
    result = {
        "processed_source_indices": [0],
        "clips": [
            _clip("a", semantic_v2={"applied": True, "abstain": False}),
            _clip("b"),
        ],
        "editable_draft": {
            "selected": [{"clip_id": "a"}],
            "alternates": [{"clip_id": "b"}],
            "discarded": [],
        },
    }
    attach_semantic_execution(
        result,
        requested=True,
        provider_available=True,
    )
    selected = result["editable_draft"]["selected"][0]
    alternate = result["editable_draft"]["alternates"][0]
    assert selected["semantic_v2_execution_status"] == "applied"
    assert "semantic_v2_fallback_reason" not in selected
    assert alternate["semantic_v2_execution_status"] == "classifier_no_result"
    assert alternate["semantic_v2_fallback_reason"] == "missing_semantic_result"
