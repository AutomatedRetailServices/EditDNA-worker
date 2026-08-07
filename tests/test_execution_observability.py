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
    assert execution["fallback_reasons"] == {"provider_unavailable": 1}


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
    source = result["semantic_execution"]["sources"][0]
    assert source["status"] == "partially_applied"
    assert source["semantic_results"] == 2
    assert source["applied"] == 1
    assert source["abstained"] == 1
    assert source["fallback_reasons"] == {
        "model_abstained": 1,
        "missing_semantic_result": 1,
    }


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
