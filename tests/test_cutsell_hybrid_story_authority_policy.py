from dataclasses import dataclass

from cutsell_worker.hybrid_story_guard import _authoritative_applied_delete_ids


@dataclass
class _Result:
    diagnostics: tuple


def _result(*decisions):
    return _Result(diagnostics=({"decisions": list(decisions)},))


def test_semantic_only_high_confidence_delete_is_not_irrevocable():
    result = _result({
        "clip_id": "diagnosis",
        "applied_delete": True,
        "delete_basis": "high_confidence_semantic",
        "label": "failed",
        "confidence": 0.95,
        "local_failure_corroborated": False,
    })
    assert _authoritative_applied_delete_ids(result) == set()


def test_semantic_delete_with_local_performance_remains_irrevocable():
    result = _result({
        "clip_id": "failed-take",
        "applied_delete": True,
        "delete_basis": "semantic_failed_plus_local_performance",
        "label": "failed",
        "confidence": 0.88,
        "local_failure_corroborated": True,
    })
    assert _authoritative_applied_delete_ids(result) == {"failed-take"}
