"""D-139: Gemini-backed `MultimodalBestTakeArbiter` -- provider-comparison
challenger against the D-138 OpenAI baseline. Covers the 20 required
targeted-test items.

All tests use a FAKE injected `session` (mirrors `hybrid_google_transport.
GoogleGeminiTransport`'s own existing testing convention for this repo's
Gemini REST transport) -- none of these tests make a real network call,
matching the repo-wide testing convention and this task's own bounded
provider-cost authorization (real calls are reserved for the bounded
offline eval run itself).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

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
    should_request_multimodal_arbitration,
    safe_arbitrate,
)
from cutsell_worker.multimodal_besttake_gemini import (
    REQUIRED_MODEL_ID,
    GeminiMultimodalBestTakeArbiter,
    list_gemini_models,
    model_supports_generate_content,
)
from cutsell_worker.multimodal_besttake_openai import INSTRUCTION


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


def _one_stub_jpeg(tmp_path: Path, name: str) -> str:
    import subprocess
    dest = tmp_path / f"{name}.jpg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=gray:s=8x8:d=1",
         "-frames:v", "1", str(dest)],
        capture_output=True, check=True,
    )
    return str(dest)


class _FakeGeminiHTTPResponse:
    def __init__(self, json_body, *, ok: bool = True):
        self._json_body = json_body
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.exceptions.HTTPError("simulated Gemini HTTP error")

    def json(self):
        return self._json_body


class _FakeGeminiSession:
    def __init__(self, *, generate_response=None, models_response=None, capture=None, raise_exc=None):
        self._generate_response = generate_response
        self._models_response = models_response
        self._capture = capture
        self._raise_exc = raise_exc

    def post(self, url, headers=None, json=None, timeout=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._capture is not None:
            self._capture["url"] = url
            self._capture["json"] = json
        return self._generate_response

    def get(self, url, headers=None, timeout=None):
        return self._models_response


def _gemini_ok_response(outcome_json: dict, usage: dict | None = None) -> _FakeGeminiHTTPResponse:
    body = {"candidates": [{"content": {"parts": [{"text": json.dumps(outcome_json)}]}}]}
    if usage is not None:
        body["usageMetadata"] = usage
    return _FakeGeminiHTTPResponse(body)


def _arbiter(session: _FakeGeminiSession) -> GeminiMultimodalBestTakeArbiter:
    return GeminiMultimodalBestTakeArbiter(api_key="fake-key", session=session)


# ---------------------------------------------------------------------------
# 1. Interface compatibility
# ---------------------------------------------------------------------------

def test_gemini_arbiter_satisfies_the_shared_protocol_shape():
    session = _FakeGeminiSession(generate_response=_gemini_ok_response(
        {"outcome": "BEST_TAKE", "best_take_candidate_id": "A", "confidence": 0.9, "reason": "CLEAR_PERFORMANCE_SUPERIORITY: r"},
    ))
    arbiter = _arbiter(session)
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    response = arbiter.arbitrate(request)
    assert response.family_id == "fam"
    assert response.provider == "gemini"
    assert response.model == REQUIRED_MODEL_ID
    assert response.requested is True
    assert response.available is True


# ---------------------------------------------------------------------------
# 2. 2-finalist request mapping
# ---------------------------------------------------------------------------

def test_two_finalist_request_mapping_includes_instruction_and_both_ids(tmp_path):
    capture: dict = {}
    frame = _one_stub_jpeg(tmp_path, "a")
    session = _FakeGeminiSession(
        generate_response=_gemini_ok_response({"outcome": "BEST_TAKE", "best_take_candidate_id": "A", "confidence": 0.9, "reason": "r"}),
        capture=capture,
    )
    arbiter = _arbiter(session)
    request = MultimodalBestTakeRequest(
        family_id="fam", proposition_context="ctx",
        finalists=(_finalist("A", frames=(frame,)), _finalist("B", frames=(frame,))),
    )
    response = arbiter.arbitrate(request)
    assert response.best_take_candidate_id == "A"
    parts = capture["json"]["contents"][0]["parts"]
    text_parts = [p["text"] for p in parts if "text" in p]
    joined = " ".join(text_parts)
    assert "A" in joined and "B" in joined
    assert INSTRUCTION in text_parts
    # image parts are inlineData blocks (camelCase, per the real Generative
    # Language API JSON convention -- see multimodal_besttake_gemini.py's
    # own D-139 real-evidence fix comment), not text.
    image_parts = [p for p in parts if "inlineData" in p]
    assert len(image_parts) == 2  # one frame per candidate
    assert image_parts[0]["inlineData"]["mimeType"] == "image/jpeg"


# ---------------------------------------------------------------------------
# 3. 3-finalist ceiling (shared gate policy, provider-agnostic)
# ---------------------------------------------------------------------------

def test_three_finalist_ceiling_accepted_four_rejected():
    policy = MultimodalBestTakeGatePolicy()
    three = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B"), _finalist("C")))
    four = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B"), _finalist("C"), _finalist("D")))
    assert should_request_multimodal_arbitration(three, policy) is True
    assert should_request_multimodal_arbitration(four, policy) is False


# ---------------------------------------------------------------------------
# 4. Canonical output normalization
# ---------------------------------------------------------------------------

def test_canonical_output_normalization_matches_shared_vocabulary():
    session = _FakeGeminiSession(generate_response=_gemini_ok_response(
        {"outcome": "GOOD_TAKE_TRIM_EXIT", "best_take_candidate_id": "A", "confidence": 0.8, "reason": "REMOVABLE_EXIT_DEFECT: r"},
    ))
    arbiter = _arbiter(session)
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    response = arbiter.arbitrate(request)
    assert response.outcome == GOOD_TAKE_TRIM_EXIT  # no Gemini-specific vocabulary


# ---------------------------------------------------------------------------
# 5-6. Supplied-candidate / BEST_TAKE validation
# ---------------------------------------------------------------------------

def test_best_take_candidate_validated():
    session = _FakeGeminiSession(generate_response=_gemini_ok_response(
        {"outcome": "BEST_TAKE", "best_take_candidate_id": "A", "confidence": 0.8, "reason": "r"},
    ))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == WOULD_INVOKE_SHADOW
    assert response.best_take_candidate_id == "A"


def test_candidate_outside_supplied_set_rejected():
    session = _FakeGeminiSession(generate_response=_gemini_ok_response(
        {"outcome": "BEST_TAKE", "best_take_candidate_id": "Z", "confidence": 0.8, "reason": "r"},
    ))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == INVALID_RESPONSE
    assert response is None


# ---------------------------------------------------------------------------
# 7-10. EQUIVALENT / UNCERTAIN / TRIM_ENTRY / TRIM_EXIT accepted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", [EQUIVALENT, GOOD_TAKE_TRIM_ENTRY, GOOD_TAKE_TRIM_EXIT, UNCERTAIN])
def test_bounded_vocabulary_outcomes_accepted(outcome):
    session = _FakeGeminiSession(generate_response=_gemini_ok_response(
        {"outcome": outcome, "best_take_candidate_id": None, "confidence": 0.5, "reason": "r"},
    ))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status in (ABSTAINED, WOULD_INVOKE_SHADOW)
    assert response.outcome == outcome


# ---------------------------------------------------------------------------
# 11. Invalid output (malformed JSON text)
# ---------------------------------------------------------------------------

def test_invalid_response_safe():
    body = {
        "candidates": [
            {"content": {"parts": [{"text": "not json at all {{{"}]}},
        ],
    }
    session = _FakeGeminiSession(generate_response=_FakeGeminiHTTPResponse(body))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == INVALID_RESPONSE
    assert response is None


def test_malformed_gemini_response_shape_also_invalid():
    # No "candidates" key at all -- _extract_text raises ValueError.
    session = _FakeGeminiSession(generate_response=_FakeGeminiHTTPResponse({}))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == INVALID_RESPONSE
    assert response is None


# ---------------------------------------------------------------------------
# 12. Provider error (real HTTP failure shape)
# ---------------------------------------------------------------------------

def test_provider_http_error_classified_as_provider_error():
    session = _FakeGeminiSession(generate_response=_FakeGeminiHTTPResponse({}, ok=False))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == PROVIDER_ERROR
    assert response is None


def test_connection_error_also_classified_as_provider_error():
    session = _FakeGeminiSession(raise_exc=requests.exceptions.ConnectionError("network down"))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == PROVIDER_ERROR
    assert response is None


# ---------------------------------------------------------------------------
# 13. Timeout/failure
# ---------------------------------------------------------------------------

def test_provider_timeout_safe():
    session = _FakeGeminiSession(raise_exc=requests.exceptions.Timeout("timed out"))
    request = MultimodalBestTakeRequest(family_id="fam", proposition_context="", finalists=(_finalist("A"), _finalist("B")))
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == TIMEOUT
    assert response is None


# ---------------------------------------------------------------------------
# 14. Meaning mismatch
# ---------------------------------------------------------------------------

def test_meaning_safety_mismatch_fails_open():
    session = _FakeGeminiSession(generate_response=_gemini_ok_response(
        {"outcome": "BEST_TAKE", "best_take_candidate_id": "B", "confidence": 0.9, "reason": "r"},
    ))
    request = MultimodalBestTakeRequest(
        family_id="fam", proposition_context="",
        finalists=(_finalist("A"), _finalist("B", meaning_sufficient=False)),
    )
    status, response = safe_arbitrate(_arbiter(session), request)
    assert status == MEANING_SAFETY_MISMATCH
    assert response is not None  # fail open: response recorded, never authoritative


# ---------------------------------------------------------------------------
# 15. No full RAW in request
# ---------------------------------------------------------------------------

def test_no_full_raw_in_request_only_bounded_references(tmp_path):
    frame = _one_stub_jpeg(tmp_path, "a")
    finalist = _finalist("A", frames=(frame,))
    assert finalist.video_span_reference is None
    assert finalist.audio_span_reference is None
    assert len(finalist.sampled_frame_references) == 1
    assert Path(finalist.sampled_frame_references[0]).stat().st_size < 200_000


# ---------------------------------------------------------------------------
# 16-18. No fixture-specific prompt words / no oracle leakage / no
# audio-hearing claim -- re-verified for THIS provider's prompt (which is
# the SAME `INSTRUCTION` constant, imported not copied, so this also proves
# the two providers cannot silently drift apart).
# ---------------------------------------------------------------------------

def test_gemini_prompt_is_the_exact_same_instruction_constant_as_openai():
    from cutsell_worker import multimodal_besttake_gemini
    assert multimodal_besttake_gemini.INSTRUCTION is INSTRUCTION


def test_gemini_prompt_contains_no_fixture_specific_terms():
    lowered = INSTRUCTION.lower()
    for term in ("pimples", "papillary", "stomach", "gynaecolog", "sonography", "vamos", "diagnosis", "hereditary", "acné", "acne"):
        assert term not in lowered


def test_gemini_prompt_contains_no_video00_or_decision_id_language():
    lowered = INSTRUCTION.lower()
    for term in ("video00", "modal", "runpod", "d-097", "d-123", "d-136", "d-137", "d-138", "d-139"):
        assert term not in lowered


def test_gemini_prompt_contains_no_oracle_leakage():
    lowered = INSTRUCTION.lower()
    for term in ("should_select", "should_equivalent", "should_uncertain", "should_trim", "expected_outcome", "human gold", "cut.ai", "oracle"):
        assert term not in lowered


def test_gemini_prompt_never_claims_to_hear_or_listen():
    lowered = INSTRUCTION.lower()
    assert "you cannot hear tone, cadence" in lowered
    assert "you can hear" not in lowered
    assert "you can listen" not in lowered


# ---------------------------------------------------------------------------
# 19. No live pipeline import
# ---------------------------------------------------------------------------

def test_no_live_pipeline_import_or_instantiation():
    import cutsell_worker.pipeline as pipeline_module
    source = Path(pipeline_module.__file__).read_text()
    assert "multimodal_besttake_gemini" not in source
    assert "GeminiMultimodalBestTakeArbiter" not in source


# ---------------------------------------------------------------------------
# 20. Overlap non-authority
# ---------------------------------------------------------------------------

def test_no_overlap_or_source_asset_fields_referenced():
    source = Path("cutsell_worker/multimodal_besttake_gemini.py").read_text()
    for forbidden in ("dialogue_overlap_enabled", "audio_overlap", "SourceAsset"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Bonus: model-availability verification helpers (never guess from naming
# convention -- always confirmed against Google's own ListModels catalog).
# ---------------------------------------------------------------------------

def test_model_supports_generate_content_exact_match_bare_id():
    models = [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]
    assert model_supports_generate_content(models, "gemini-2.5-flash") is True
    assert model_supports_generate_content(models, "models/gemini-2.5-flash") is True


def test_model_supports_generate_content_rejects_different_generation():
    models = [{"name": "models/gemini-2.5-flash-lite", "supportedGenerationMethods": ["generateContent"]}]
    # "gemini-2.5-flash" must NOT match "gemini-2.5-flash-lite" -- exact
    # bare-id match only, never a substring/prefix match.
    assert model_supports_generate_content(models, "gemini-2.5-flash") is False


def test_model_supports_generate_content_requires_generate_content_method():
    models = [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["embedContent"]}]
    assert model_supports_generate_content(models, "gemini-2.5-flash") is False


def test_list_gemini_models_parses_real_shaped_response():
    class _FakeListResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]}

    session = _FakeGeminiSession(models_response=_FakeListResponse())
    models = list_gemini_models("fake-key", session=session)
    assert len(models) == 1
    assert models[0]["name"] == "models/gemini-2.5-flash"
