"""D-239R: AUTHORITATIVE P1 TARGET DIAGNOSTICS PARITY FIX -- offline tests.

D-239Q proved `_apply_post_authority_validation_only` (the AUTHORITATIVE
second pass -- the path every real RAW dispatch actually serializes,
since the workflow unconditionally overlays
`CUTSELL_UNIFIED_REALIZATION_RESOLVER=AUTHORITATIVE`) accepted
`exact_p1_target_evidence_by_clip_id` as a parameter but never wrote
`diagnostics["final_story_coherence_validation"]["exact_p1_target_
evidence"]`. D-239R adds the missing 3-line re-projection, mirroring the
already-correct legacy pass exactly -- this file proves parity between
the two paths for identical inputs, and that the fix touches nothing
else (Freeze/materiality/repair/P1 authority, selection membership).

Follows the established D-235X test precedent (`monkeypatch.setattr(fscv,
"_lost_semantic_atoms", fake_lost_semantic_atoms)`) so this suite never
re-derives content-loss detection -- it only proves the NEW re-projection
seam, given controlled, already-known `lost_semantic_atoms` rows.
"""
from __future__ import annotations

import cutsell_worker.final_story_coherence_validation as fscv
from cutsell_worker.canonical_edit_plan import AuthoritativePlanSource
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import (
    apply_final_story_coherence_validation,
    apply_post_authority_story_validation,
)
from cutsell_worker.post_authority_validation import PostAuthorityValidationContext


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------
def _clip(clip_id, start, end, text, *, selected):
    return DraftClip(
        clip_id=clip_id, source_asset_id="s1", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def _draft(*, selected=(), discarded=()):
    # No `take_judge_groups` diagnostics -> `_residual_multi_select_groups`
    # returns [] -> `assess_authoritative_membership` is never consulted --
    # the SAME trivial-context shape every existing D-090 "no residual
    # family" test already relies on.
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded, diagnostics={},
    )


def _lost_atom_row(clip_id, provenance_id, **overrides):
    base = {
        "clip_id": clip_id,
        "lost_atom_provenance_id": provenance_id,
        "text": "a generic lost fragment of real speech",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
    }
    base.update(overrides)
    return base


def _trivial_authoritative_context():
    """A minimal, structurally-valid `PostAuthorityValidationContext` --
    bypasses `build_post_authority_validation_context`'s own real-resolver
    plumbing (covered by D-090's own tests) since this suite tests ONLY
    the diagnostics re-projection seam, not context construction."""
    plan_source = AuthoritativePlanSource(status="SEMANTICALLY_RESOLVED", decisions={})
    return PostAuthorityValidationContext(
        authoritative_status="SEMANTICALLY_RESOLVED",
        plan_source=plan_source,
        source_identity="test_source_identity",
        decision_count=0,
    )


def _both_paths(monkeypatch, rows, *, exact_p1_target_evidence_by_clip_id, with_context=True):
    """Run the SAME draft/inputs through the legacy pass and the
    authoritative pass, returning (legacy_diag, authoritative_diag)."""
    def fake_lost_semantic_atoms(*args, **kwargs):
        return list(rows)

    monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)

    selected = (_clip("winner", 0.0, 1.0, "the kept content", selected=True),)
    discarded = tuple(
        _clip(row["clip_id"], 2.0 + i, 3.0 + i, "discarded text", selected=False)
        for i, row in enumerate(rows)
    )
    draft = _draft(selected=selected, discarded=discarded)

    legacy = apply_final_story_coherence_validation(
        draft, exact_p1_target_evidence_by_clip_id=exact_p1_target_evidence_by_clip_id,
    )
    context = _trivial_authoritative_context() if with_context else None
    authoritative = apply_post_authority_story_validation(
        draft, context=context, exact_p1_target_evidence_by_clip_id=exact_p1_target_evidence_by_clip_id,
    )
    return (
        legacy.diagnostics["final_story_coherence_validation"],
        authoritative.diagnostics["final_story_coherence_validation"],
    )


# ---------------------------------------------------------------------------
# 1-2: both paths write the key.
# ---------------------------------------------------------------------------
def test_1_legacy_path_writes_exact_p1_target_evidence(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy, _ = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    assert "exact_p1_target_evidence" in legacy
    assert len(legacy["exact_p1_target_evidence"]) == 1


def test_2_authoritative_path_writes_exact_p1_target_evidence(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    _, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    assert "exact_p1_target_evidence" in authoritative
    assert len(authoritative["exact_p1_target_evidence"]) == 1


# ---------------------------------------------------------------------------
# 3: same inputs -> same rows (the parity requirement).
# ---------------------------------------------------------------------------
def test_3_same_inputs_produce_structurally_equivalent_rows(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED", "role": "POST_TAKE_RESET"}}
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    assert legacy["exact_p1_target_evidence"] == authoritative["exact_p1_target_evidence"]


# ---------------------------------------------------------------------------
# 4-5: one / multiple lost atoms correlate independently.
# ---------------------------------------------------------------------------
def test_4_one_lost_atom_correlates(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    for diag in (legacy, authoritative):
        rows_out = diag["exact_p1_target_evidence"]
        assert len(rows_out) == 1
        assert rows_out[0]["clip_id"] == "c1"
        assert rows_out[0]["lost_atom_provenance_id"] == "prov_1"


def test_5_multiple_lost_atoms_correlate_independently(monkeypatch):
    row_a = _lost_atom_row("c1", "prov_1")
    row_b = _lost_atom_row("c2", "prov_2")
    target_map = {
        "c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"},
        "c2": {"clip_id": "c2", "p1_target_lookup_status": "NO_UNDERSTANDING_SPAN"},
    }
    legacy, authoritative = _both_paths(monkeypatch, [row_a, row_b], exact_p1_target_evidence_by_clip_id=target_map)
    for diag in (legacy, authoritative):
        rows_out = {r["clip_id"]: r for r in diag["exact_p1_target_evidence"]}
        assert set(rows_out) == {"c1", "c2"}
        assert rows_out["c1"]["identity"]["p1_target_lookup_status"] == "MOMENT_FOUND_RESOLVED"
        assert rows_out["c2"]["identity"]["p1_target_lookup_status"] == "NO_UNDERSTANDING_SPAN"


# ---------------------------------------------------------------------------
# 6-8: no map / empty map / unmatched clip ids -> [].
# ---------------------------------------------------------------------------
def test_6_no_map_yields_empty_list(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=None)
    assert legacy["exact_p1_target_evidence"] == []
    assert authoritative["exact_p1_target_evidence"] == []


def test_7_empty_map_yields_empty_list(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id={})
    assert legacy["exact_p1_target_evidence"] == []
    assert authoritative["exact_p1_target_evidence"] == []


def test_8_unmatched_clip_ids_yield_empty_list(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    # The target map has evidence for a DIFFERENT clip_id -- never the
    # lost atom's own -- so no correlation should occur.
    target_map = {"some_other_clip": {"clip_id": "some_other_clip", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    assert legacy["exact_p1_target_evidence"] == []
    assert authoritative["exact_p1_target_evidence"] == []


# ---------------------------------------------------------------------------
# 9: integrity-failure path -> [].
# ---------------------------------------------------------------------------
def test_9_integrity_failure_path_yields_empty_list(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    _, authoritative = _both_paths(
        monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map, with_context=False,
    )
    assert authoritative["exact_p1_target_evidence"] == []
    assert authoritative["status"] == "integrity_failure"


# ---------------------------------------------------------------------------
# 10: no transcript leakage.
# ---------------------------------------------------------------------------
def test_10_no_transcript_leakage(monkeypatch):
    row = _lost_atom_row("c1", "prov_1", text="a real spoken sentence that must never leak")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    for diag in (legacy, authoritative):
        for r in diag["exact_p1_target_evidence"]:
            assert set(r.keys()) == {"lost_atom_provenance_id", "clip_id", "identity"}
            assert "text" not in r["identity"]
            assert "transcript" not in r["identity"]


# ---------------------------------------------------------------------------
# 11: deterministic output.
# ---------------------------------------------------------------------------
def test_11_deterministic_output(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy1, authoritative1 = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    legacy2, authoritative2 = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    assert legacy1["exact_p1_target_evidence"] == legacy2["exact_p1_target_evidence"]
    assert authoritative1["exact_p1_target_evidence"] == authoritative2["exact_p1_target_evidence"]


# ---------------------------------------------------------------------------
# 12-15: no other authority mutated -- diagnostics-only.
# ---------------------------------------------------------------------------
def test_12_no_p1_policy_mutation(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    p1_role_map = {"c1": "POST_TAKE_RESET"}
    p1_audience_map = {"c1": "AUDIENCE_DELIVERY_SUPPORTED"}

    def fake_lost_semantic_atoms(*args, **kwargs):
        return [row]
    monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)

    selected = (_clip("winner", 0.0, 1.0, "the kept content", selected=True),)
    discarded = (_clip("c1", 2.0, 3.0, "discarded text", selected=False),)
    draft = _draft(selected=selected, discarded=discarded)

    authoritative = apply_post_authority_story_validation(
        draft, context=_trivial_authoritative_context(),
        exact_p1_target_evidence_by_clip_id=target_map,
        p1_moment_role_by_clip_id=p1_role_map,
        p1_audience_delivery_status_by_clip_id=p1_audience_map,
    )
    # The exact same maps handed in are never mutated by this pass.
    assert p1_role_map == {"c1": "POST_TAKE_RESET"}
    assert p1_audience_map == {"c1": "AUDIENCE_DELIVERY_SUPPORTED"}
    # And selection membership is untouched (D-090's own invariant).
    assert [c.clip_id for c in authoritative.selected] == ["winner"]
    assert [c.clip_id for c in authoritative.discarded] == ["c1"]


def test_13_no_materiality_mutation(monkeypatch):
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    # D-235Q materiality orchestration key is present and untouched by
    # the new field -- {} shape (flag off in this offline fixture, same
    # fail-open posture every prior gate proved).
    for diag in (legacy, authoritative):
        assert "lost_atom_materiality_orchestration" in diag


def test_14_no_freeze_mutation(monkeypatch):
    row = _lost_atom_row("c1", "prov_1", blocking=True)
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    # freeze_blocked reflects the row's own `blocking` flag exactly as
    # before D-239R -- the new field never influences it either way.
    assert legacy["freeze_blocked"] is True
    assert authoritative["freeze_blocked"] is True


def test_15_no_repair_loop_mutation(monkeypatch):
    # RepairLoop is not invoked by either validation pass directly (it is
    # a separate universal_clean_cut.py stage) -- this test proves the
    # new field's own diagnostics never introduce a "repair_loop" key.
    row = _lost_atom_row("c1", "prov_1")
    target_map = {"c1": {"clip_id": "c1", "p1_target_lookup_status": "MOMENT_FOUND_RESOLVED"}}
    legacy, authoritative = _both_paths(monkeypatch, [row], exact_p1_target_evidence_by_clip_id=target_map)
    assert "repair_loop" not in legacy
    assert "repair_loop" not in authoritative
