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
    assert config["responseFormat"]["text"]["mimeType"] == "application/json"
    schema = config["responseFormat"]["text"]["schema"]
    assert schema == editorial_response_schema()
    serialized = json.dumps(request)
    assert "API_KEY" not in serialized
    assert "http" not in serialized.lower()


def test_schema_constrains_labels_and_confidence():
    item = editorial_response_schema()["properties"]["decisions"]["items"]
    assert "winner" in item["properties"]["label"]["enum"]
    assert item["properties"]["confidence"]["minimum"] == 0.0
    assert item["properties"]["confidence"]["maximum"] == 1.0
    assert item["additionalProperties"] is False


def test_parser_extracts_decisions_and_usage():
    response = {
        "candidates": [{
            "content": {"parts": [{"text": json.dumps({"decisions": [
                {"clip_id": "a", "label": "failed", "confidence": 0.96, "reason_code": "restart"},
                {"clip_id": "b", "label": "winner", "confidence": 0.98, "reason_code": "complete_take"},
            ]})}]}
        }],
        "usageMetadata": {"candidatesTokenCount": 73},
    }
    parsed = parse_gemini_generate_content_response(response)
    assert parsed["decisions"][1]["clip_id"] == "b"
    assert parsed["output_tokens"] == 73


def test_parser_rejects_missing_or_invalid_json():
    with pytest.raises(ValueError, match="missing candidates"):
        parse_gemini_generate_content_response({})
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_gemini_generate_content_response({
            "candidates": [{"content": {"parts": [{"text": "not-json"}]}}]
        })


def test_thinking_level_is_explicit_and_bounded():
    request = build_gemini_generate_content_request(payload(), max_output_tokens=400, thinking_level="medium")
    assert request["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "medium"
    with pytest.raises(ValueError, match="unsupported Gemini thinking level"):
        build_gemini_generate_content_request(payload(), max_output_tokens=400, thinking_level="unlimited")
