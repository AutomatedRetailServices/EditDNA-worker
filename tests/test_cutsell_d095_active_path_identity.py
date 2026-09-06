"""D-095 addendum: active-path identity -- proof of which code produced a result."""
from __future__ import annotations

from pathlib import Path

from cutsell_worker.active_path_identity import (
    SCHEMA_VERSION,
    build_active_path_identity,
    component_markers,
    package_fingerprint,
)


def test_package_fingerprint_is_deterministic_and_counts_modules(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "sub").mkdir()
    (pkg / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"ignored")
    first = package_fingerprint(pkg)
    second = package_fingerprint(pkg)
    assert first == second
    assert first["module_count"] == 2 and first["byte_count"] == 12
    (pkg / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert package_fingerprint(pkg)["sha256"] != first["sha256"]


def test_real_package_fingerprint_matches_a_fresh_computation():
    root = Path("cutsell_worker")
    assert package_fingerprint(root) == package_fingerprint(root)
    assert package_fingerprint(root)["module_count"] > 50


def test_component_markers_reflect_evidence_present_in_the_result():
    result = {
        "stage_status": {"attempt_reconstruction": {"status": "complete"}, "hybrid_editorial": "provider_complete"},
        "diagnostics": {
            "take_judge_groups": [{"group_id": "g", "semantic_label_source": "family_window"}],
            "distinct_idea_grouping_safety": {"status": "applied"},
            "selection_boundary_contract": {"plan_id": "p"},
            "authoritative_story_placement": [],
        },
    }
    rows = {r["component"]: r["present"] for r in component_markers(result)}
    assert rows["AttemptReconstructor"] is True
    assert rows["HybridSemanticPass"] is True
    assert rows["BestTakeResolver.family_window_labels"] is True
    assert rows["IdeaClusterer.grouping_safety"] is True
    assert rows["SelectionFreeze"] is True
    assert rows["RealizationResolver.story_placement"] is False  # empty list = no evidence
    assert rows["PostAuthorityValidation"] is False
    assert rows["EditorialSlotResolution"] is False


def test_identity_block_carries_build_sha_env_and_absent_components():
    env = {"CUTSELL_BUILD_GIT_SHA": "abc123", "CUTSELL_UNIFIED_REALIZATION_RESOLVER": "AUTHORITATIVE",
           "CUTSELL_HYBRID_MAX_EDIT_USD": "0.02"}
    ident = build_active_path_identity({"stage_status": {}, "diagnostics": {}}, env=env)
    assert ident["schema_version"] == SCHEMA_VERSION
    assert ident["build_git_sha"] == "abc123"
    assert ident["resolver_mode_env"] == "AUTHORITATIVE"
    assert ident["hybrid_max_edit_usd_env"] == "0.02"
    assert ident["package"]["module_count"] > 50
    assert ident["components_present"] == 0
    assert "SelectionFreeze" in ident["components_absent"]


def test_identity_is_json_native_and_never_editorial():
    import json
    ident = build_active_path_identity({"stage_status": {}, "diagnostics": {}}, env={})
    json.dumps(ident)
    forbidden = ("selected", "discarded", "keep", "winner")
    assert not any(key in ident for key in forbidden)
    src = Path("cutsell_worker/active_path_identity.py").read_text(encoding="utf-8")
    for needle in ("VIDEO-2026", "D40F1D43", "5E01F214"):
        assert needle not in src
