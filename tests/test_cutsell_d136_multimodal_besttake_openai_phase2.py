"""D-136 (D-127 Phase 2): provider-backed OFFLINE evaluation tests.

Covers the 21 required targeted-test items. All tests use a FAKE injected
`client_factory` (the same dependency-injection convention every other
`*_openai.py` provider in this repo already uses for tests) -- none of
these tests make a real network call, matching the repo-wide testing
convention and this task's own bounded provider-cost authorization (real
calls are reserved for the bounded offline eval run itself, never for
the test suite).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cutsell_worker.case_b_performance_evidence import CaseBPerformanceEvidence
from cutsell_worker.multimodal_besttake_arbiter import (
    ABSTAINED,
    BEST_TAKE,
    EQUIVALENT,
    GOOD_TAKE_TRIM_ENTRY,
    GOOD_TAKE_TRIM_EXIT,
    INVALID_RESPONSE,
    MEANING_SAFETY_MISMATCH,
    PROVIDER_ERROR,
    TIMEOUT,
    UNCERTAIN,
    WOULD_INVOKE_SHADOW,
    MultimodalBestTakeGatePolicy,
    MultimodalBestTakeRequest,
    MultimodalFinalistInput,
    safe_arbitrate,
)
from cutsell_worker.multimodal_besttake_openai import (
    INSTRUCTION,
    OpenAIMultimodalBestTakeArbiter,
)
from cutsell_worker.multimodal_besttake_eval_phase2 import (
    run_phase2_eval,
    summarize_phase2_eval,
)


def _evidence(cid: str) -> CaseBPerformanceEvidence:
    return CaseBPerformanceEvidence(
        candidate_id=cid, source_asset_id="s", delivery_available=True,
        delivery_start=0.0, delivery_end=1.0, delivery_span_duration=1.0,
        delivery_events=(), delivery_event_count=0, delivery_event_duration_total=0.0,
        count_by_kind={}, duration_by_kind={}, event_density=0.0,
    )


def _finalist(cid: str, *, meaning_sufficient: bool = True, frames: tuple[str, ...] = ()) -> MultimodalFinalistInput:
    return MultimodalFinalistInput(
        candidate_id=cid, source_asset_id="s", source_start=0.0, source_end=1.0,
        transcript_text="synthetic", meaning_sufficient=meaning_sufficient,
        case_b_evidence=_evidence(cid), semantic_label="x", semantic_confidence=0.5,
        deliveryscore_summary=1.0, sampled_frame_references=frames,
    )


class _FakeResponse:
    def __init__(self, output_text: str, usage=None):
        self.output_text = output_text
        self.usage = usage


class _FakeResponsesEndpoint:
    def __init__(self, output_text: str, capture: dict | None = None, usage=None):
        self._output_text = output_text
        self._capture = capture
        self._usage = usage

    def create(self, model, input):
        if self._capture is not None:
            self._capture["model"] = model
            self._capture["input"] = input
        return _FakeResponse(self._output_text, usage=self._usage)


class _FakeClient:
    def __init__(self, output_text: str, capture: dict | None = None, usage=None):
        self.responses = _FakeResponsesEndpoint(output_text, capture=capture, usage=usage)


def _arbiter(output_text: str, capture: dict | None = None, usage=None) -> OpenAIMultimodalBestTakeArbiter:
    return OpenAIMultimodalBestTakeArbiter(client_factory=lambda: _FakeClient(output_text, capture=capture, usage=usage))


def _one_stub_jpeg(tmp_path: Path, name: str) -> str:
    # A 2x2 pixel JPEG is enough -- base64-encoding/reading it is all these
    # tests exercise, never real decoding by a real vision model.
    import subprocess
    dest = tmp_path / f"{name}.jpg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=gray:s=8x8:d=1",
         "-frames:v", "1", str(dest)],
        capture_output=True, check=True,
    )
    return str(dest)


# ---------------------------------------------------------------------------
# 1-2. Concrete provider request mapping / 2-finalist request
# ---------------------------------------------------------------------------

def test_concrete_provider_two_finalist_request_mapping(tmp_path):
    capture: dict = {}
    frame = _one_stub_jpeg(tmp_path, "a")
    request = MultimodalBestTakeRequest(
        family_id="fam", proposition_context="ctx",
        finalists=(_finalist("A", frames=(frame,)), _finalist("B", frames=(frame,))),
    )
    arbiter = _arbiter(
        json.dumps({"outcome": "BEST_TAKE", "best_take_candidate_id": "A", "confidence": 0.9, "reason": "r"}),
        capture=capture,
    )
    response = arbiter.arbitrate(request)
    assert response.outcome == "BEST_TAKE"
    assert response.best_take_candidate_id == "A"
    assert response.provider == "openai"
    assert response.available is True
    # The request content contains the instruction + both candidate ids.
    content_texts = [item.get("text", "") for item in capture["input"][0]["content"] if item["type"] == "input_text"]
    joined = " ".join(content_texts)
    assert "A" in joined and "B" in joined
    assert INSTRUCTION in content_texts


# ---------------------------------------------------------------------------
# 3. 3-finalist ceiling
# ---------------------------------------------------------------------------

def test_three_finalist_ceiling_accepted_four_rejected():
    policy = MultimodalBestTakeGatePolicy()
    three = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B"), _finalist("C")))
    four = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B"), _finalist("C"), _finalist("D")))
    from cutsell_worker.multimodal_besttake_arbiter import should_request_multimodal_arbitration
    assert should_request_multimodal_arbitration(three, policy) is True
    assert should_request_multimodal_arbitration(four, policy) is False


# ---------------------------------------------------------------------------
# 4. No full RAW in request
# ---------------------------------------------------------------------------

def test_no_full_raw_in_request_only_bounded_references(tmp_path):
    frame = _one_stub_jpeg(tmp_path, "a")
    finalist = _finalist("A", frames=(frame,))
    # Only a bounded reference (a path string to one small stub frame) is
    # ever carried -- never raw media bytes, never a full source video path.
    assert finalist.video_span_reference is None
    assert finalist.audio_span_reference is None
    assert len(finalist.sampled_frame_references) == 1
    assert Path(finalist.sampled_frame_references[0]).stat().st_size < 200_000


# ---------------------------------------------------------------------------
# 5-6. BEST_TAKE candidate validation / invalid candidate rejected
# ---------------------------------------------------------------------------

def test_best_take_candidate_validated(tmp_path):
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    arbiter = _arbiter(json.dumps({"outcome": "BEST_TAKE", "best_take_candidate_id": "A", "confidence": 0.8, "reason": "r"}))
    status, response = safe_arbitrate(arbiter, request)
    assert status == WOULD_INVOKE_SHADOW
    assert response.best_take_candidate_id == "A"


def test_invalid_candidate_id_rejected(tmp_path):
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    arbiter = _arbiter(json.dumps({"outcome": "BEST_TAKE", "best_take_candidate_id": "Z", "confidence": 0.8, "reason": "r"}))
    status, response = safe_arbitrate(arbiter, request)
    assert status == INVALID_RESPONSE
    assert response is None


# ---------------------------------------------------------------------------
# 7-10. EQUIVALENT / TRIM_ENTRY / TRIM_EXIT / UNCERTAIN accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", [EQUIVALENT, GOOD_TAKE_TRIM_ENTRY, GOOD_TAKE_TRIM_EXIT, UNCERTAIN])
def test_bounded_vocabulary_outcomes_accepted(outcome):
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    arbiter = _arbiter(json.dumps({"outcome": outcome, "best_take_candidate_id": None, "confidence": 0.4, "reason": "r"}))
    status, response = safe_arbitrate(arbiter, request)
    assert status in (ABSTAINED, WOULD_INVOKE_SHADOW)
    assert response.outcome == outcome


# ---------------------------------------------------------------------------
# 11-13. provider timeout / error / invalid response safe
# ---------------------------------------------------------------------------

class _RaisingArbiter:
    def __init__(self, exc: Exception):
        self._exc = exc

    def arbitrate(self, request):
        raise self._exc


class _FakeAPITimeoutError(Exception):
    pass


class _FakeAPIConnectionError(Exception):
    pass


def test_provider_timeout_safe():
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_RaisingArbiter(_FakeAPITimeoutError("timed out")), request)
    assert status == TIMEOUT
    assert response is None


def test_provider_error_safe():
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_RaisingArbiter(_FakeAPIConnectionError("network down")), request)
    assert status == PROVIDER_ERROR
    assert response is None


def test_invalid_response_safe(tmp_path):
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    arbiter = _arbiter("not json at all {{{")
    status, response = safe_arbitrate(arbiter, request)
    assert status == INVALID_RESPONSE
    assert response is None


# ---------------------------------------------------------------------------
# 14. meaning mismatch safe
# ---------------------------------------------------------------------------

def test_meaning_mismatch_safe():
    request = MultimodalBestTakeRequest(
        family_id="fam", proposition_context="",
        finalists=(_finalist("A", meaning_sufficient=True), _finalist("B", meaning_sufficient=False)),
    )
    arbiter = _arbiter(json.dumps({"outcome": "BEST_TAKE", "best_take_candidate_id": "B", "confidence": 0.9, "reason": "r"}))
    status, response = safe_arbitrate(arbiter, request)
    assert status == MEANING_SAFETY_MISMATCH
    assert response is not None  # response is preserved for eval visibility, never authoritative


# ---------------------------------------------------------------------------
# 15. no live pipeline import/instantiation
# ---------------------------------------------------------------------------

def test_no_live_pipeline_import_or_instantiation():
    pipeline_source = Path("cutsell_worker/pipeline.py").read_text()
    assert "multimodal_besttake_openai" not in pipeline_source
    assert "multimodal_besttake_arbiter" not in pipeline_source
    assert "OpenAIMultimodalBestTakeArbiter" not in pipeline_source
    flow_b_source = Path("cutsell_worker/flow_b.py").read_text()
    assert "multimodal_besttake_openai" not in flow_b_source
    assert "OpenAIMultimodalBestTakeArbiter" not in flow_b_source


# ---------------------------------------------------------------------------
# 16. no production winner change
# ---------------------------------------------------------------------------

def test_no_production_winner_change_structural():
    # The arbiter/eval modules never write to any structure pipeline.py
    # reads back -- confirmed by the same import-absence proof above, plus
    # every Phase2CaseResult being a plain, discarded dataclass instance.
    import tempfile
    from cutsell_worker.multimodal_besttake_eval_phase2 import run_phase2_eval

    class _AlwaysBestTakeA:
        def arbitrate(self, request):
            from cutsell_worker.multimodal_besttake_arbiter import MultimodalBestTakeResponse
            return MultimodalBestTakeResponse(
                family_id=request.family_id, outcome="BEST_TAKE",
                best_take_candidate_id=request.finalists[0].candidate_id,
                confidence=0.99, reason="r", provider="fake", model="fake",
                requested=True, available=True,
            )

    with tempfile.TemporaryDirectory() as d:
        results = run_phase2_eval(_AlwaysBestTakeA(), workdir=d)
    # Results are plain dataclasses returned to the caller -- nothing here
    # touches take_judge_groups, ranked, selected_clip_id, or any pipeline
    # module-level state (none of those symbols are imported anywhere in
    # this module or multimodal_besttake_openai.py).
    phase2_source = Path("cutsell_worker/multimodal_besttake_eval_phase2.py").read_text()
    for forbidden in ("selected_clip_id", "take_judge_groups", "ranked_takes", "winner_path"):
        assert forbidden not in phase2_source
    assert len(results) == 9


# ---------------------------------------------------------------------------
# 17. D-128 shadow unchanged
# ---------------------------------------------------------------------------

def test_d128_shadow_module_unchanged_by_phase2():
    fallback_source = Path("cutsell_worker/multimodal_besttake_fallback.py").read_text()
    assert "openai" not in fallback_source.lower()
    assert "multimodal_besttake_openai" not in fallback_source


# ---------------------------------------------------------------------------
# 18. D-123 unchanged
# ---------------------------------------------------------------------------

def test_d123_module_unaffected_by_phase2():
    # D-123's own gate lives in pipeline.py's _case_b_fast_path_conflict /
    # semantic_best_take_integrity.py -- neither references this task's
    # new modules at all.
    integrity_source = Path("cutsell_worker/semantic_best_take_integrity.py").read_text()
    assert "multimodal_besttake_openai" not in integrity_source
    assert "OpenAIMultimodalBestTakeArbiter" not in integrity_source


# ---------------------------------------------------------------------------
# 19. Boundary unchanged
# ---------------------------------------------------------------------------

def test_boundary_module_unaffected_by_phase2():
    boundary_source = Path("cutsell_worker/boundary_engine_pass.py").read_text()
    assert "multimodal_besttake_openai" not in boundary_source
    assert "OpenAIMultimodalBestTakeArbiter" not in boundary_source


# ---------------------------------------------------------------------------
# 20. Overlap fields ignored
# ---------------------------------------------------------------------------

def test_overlap_fields_never_referenced_by_phase2_modules():
    for path in ("cutsell_worker/multimodal_besttake_openai.py", "cutsell_worker/multimodal_besttake_eval_phase2.py"):
        source = Path(path).read_text()
        assert "dialogue_overlap_enabled" not in source
        assert "audio_overlap" not in source


# ---------------------------------------------------------------------------
# 21. iOS metadata non-authoritative
# ---------------------------------------------------------------------------

def test_ios_metadata_never_referenced_by_phase2_modules():
    for path in ("cutsell_worker/multimodal_besttake_openai.py", "cutsell_worker/multimodal_besttake_eval_phase2.py"):
        source = Path(path).read_text()
        assert "SourceAsset" not in source
        assert ".metadata" not in source


# ---------------------------------------------------------------------------
# Eval harness sanity (not in the 21-item list, but validates the report
# shape the DELIVERABLE depends on).
# ---------------------------------------------------------------------------

def test_phase2_eval_harness_runs_all_nine_cases_and_summarizes(tmp_path):
    from cutsell_worker.multimodal_besttake_arbiter import MultimodalBestTakeResponse

    class _AlwaysUncertain:
        def arbitrate(self, request):
            return MultimodalBestTakeResponse(
                family_id=request.family_id, outcome="UNCERTAIN",
                best_take_candidate_id=None, confidence=0.2, reason="r",
                provider="fake", model="fake", requested=True, available=True,
            )

    results = run_phase2_eval(_AlwaysUncertain(), workdir=str(tmp_path))
    assert len(results) == 9
    metrics = summarize_phase2_eval(results)
    assert metrics["total_cases"] == 9
    assert metrics["positive_cases"] == 1
    assert metrics["negative_controls"] == 8
    assert metrics["skipped_single_member"] == 1
