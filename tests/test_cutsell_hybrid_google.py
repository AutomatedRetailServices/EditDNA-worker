import json

import pytest

from cutsell_worker.hybrid_google import (
    build_gemini_generate_content_request,
    editorial_response_schema,
    parse_gemini_generate_content_response,
)


def payload():
    return {
        "task": "classify_best_take_within_single_bounded_creator_session",
        "session_id": "hs_test",
        "source_asset_id": "source",
        "local_confidence": 0.62,
        "conflict_score": 0.71,
        "candidates": [
            {"clip_id": "a", "text": "I love these... wait", "local_label": "alternate"},
            {"clip_id": "b", "text": "I love these jeans because they fit perfectly", "local_label": "winner"},
        ],
    }


def test_request_uses_structured_json_and_no_network_or_key_material():
    request = build_gemini_generate_content_request(payload(), max_output_tokens=500)
    config = request["generationConfig"]
    assert config["maxOutputTokens"] == 500
    assert config["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == editorial_response_schema(2)
    assert "responseFormat" not in config
    serialized = json.dumps(request)
    assert "API_KEY" not in serialized
    assert "http" not in serialized.lower()


def test_schema_constrains_labels_confidence_omits_echoed_ids_and_never_bounds_count():
    # An isolation probe proved Gemini's structured-output validator rejects an
    # exact-length minItems==maxItems array bound at scale even with this exact
    # schema and model (works at 5 items, 400s at 90); the decision-count check
    # already lives downstream in hybrid_google_transport.py, so the schema must
    # never re-add this bound regardless of the candidate_count argument passed.
    schema = editorial_response_schema(6)
    decisions = schema["properties"]["decisions"]
    item = decisions["items"]
    assert "winner" in item["properties"]["label"]["enum"]
    assert item["properties"]["confidence"]["minimum"] == 0.0
    assert item["properties"]["confidence"]["maximum"] == 1.0
    assert item["required"] == ["label", "confidence"]
    assert "clip_id" not in item["properties"]
    assert "reason_code" not in item["properties"]
    assert item["additionalProperties"] is False
    assert "minItems" not in decisions
    assert "maxItems" not in decisions


def test_parser_extracts_ordered_compact_decisions_and_usage_and_joins_text_parts():
    encoded = json.dumps({"decisions": [
        {"label": "failed", "confidence": 0.96},
        {"label": "winner", "confidence": 0.98},
    ]})
    split = len(encoded) // 2
    response = {
        "candidates": [{
            "content": {"parts": [{"text": encoded[:split]}, {"text": encoded[split:]}]}
        }],
        "usageMetadata": {"candidatesTokenCount": 51},
    }
    parsed = parse_gemini_generate_content_response(response)
    assert parsed["decisions"][1]["label"] == "winner"
    assert "clip_id" not in parsed["decisions"][1]
    assert parsed["output_tokens"] == 51


def test_ordered_schema_materially_reduces_twelve_candidate_output_shape():
    ordered = {"decisions": [
        {"label": "keep", "confidence": 0.97}
        for _ in range(12)
    ]}
    prior_compact = {"decisions": [
        {"clip_id": f"candidate_with_long_runtime_id_{i}", "label": "keep", "confidence": 0.97}
        for i in range(12)
    ]}
    ordered_chars = len(json.dumps(ordered, separators=(",", ":")))
    prior_chars = len(json.dumps(prior_compact, separators=(",", ":")))
    assert ordered_chars < prior_chars * 0.60
    assert ordered_chars < 500


def test_parser_rejects_missing_or_invalid_json_with_finish_diagnostics():
    with pytest.raises(ValueError, match="missing candidates"):
        parse_gemini_generate_content_response({})
    with pytest.raises(ValueError, match="invalid JSON; finish_reason=MAX_TOKENS; output_tokens=320"):
        parse_gemini_generate_content_response({
            "candidates": [{
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": "not-json"}]},
            }],
            "usageMetadata": {"candidatesTokenCount": 320},
        })


def test_thinking_level_is_explicit_and_bounded():
    request = build_gemini_generate_content_request(payload(), max_output_tokens=400, thinking_level="medium")
    assert request["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "medium"
    with pytest.raises(ValueError, match="unsupported Gemini thinking level"):
        build_gemini_generate_content_request(payload(), max_output_tokens=400, thinking_level="unlimited")
