"""D-041 follow-up: the only logic in runpod_endpoint_inspect.py worth
pinning without live RunPod credentials is its allowlist filter -- the
guarantee that a template's `env` dict (which carries real secrets: AWS
keys, GEMINI_API_KEY) can never reach stdout, and that only fields we have
explicitly reviewed as non-secret are ever printed for either an endpoint or
a template response.
"""
from __future__ import annotations

from runpod_endpoint_inspect import _SAFE_ENDPOINT_KEYS, _SAFE_TEMPLATE_KEYS, filter_safe


def test_filter_safe_drops_template_env_even_though_not_asked_for():
    template = {
        "id": "tmpl-1",
        "name": "cutsell",
        "imageName": "ghcr.io/example/image",
        "env": {"AWS_SECRET_ACCESS_KEY": "super-secret", "GEMINI_API_KEY": "also-secret"},
    }
    result = filter_safe(template, _SAFE_TEMPLATE_KEYS)
    assert "env" not in result
    assert result == {"id": "tmpl-1", "name": "cutsell", "imageName": "ghcr.io/example/image"}


def test_filter_safe_drops_unrecognized_fields_by_default():
    endpoint = {
        "id": "ep-1",
        "workersMax": 1,
        "someFutureFieldWeHaventReviewed": "could-be-anything",
    }
    result = filter_safe(endpoint, _SAFE_ENDPOINT_KEYS)
    assert result == {"id": "ep-1", "workersMax": 1}


def test_filter_safe_keeps_known_safe_endpoint_fields():
    endpoint = {
        "id": "ep-1",
        "templateId": "tmpl-1",
        "workersMin": 0,
        "workersMax": 1,
        "gpuIds": "AMPERE_80,ADA_24",
        "scalerType": "QUEUE_DELAY",
    }
    result = filter_safe(endpoint, _SAFE_ENDPOINT_KEYS)
    assert result == endpoint


def test_filter_safe_handles_non_dict_input():
    assert filter_safe(None, _SAFE_ENDPOINT_KEYS) == {}
    assert filter_safe("not-a-dict", _SAFE_ENDPOINT_KEYS) == {}
