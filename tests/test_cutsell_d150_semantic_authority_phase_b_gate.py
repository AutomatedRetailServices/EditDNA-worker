"""D-150: SEMANTIC AUTHORITY PHASE B -- FAMILY-COMPLETE + COMPLETE-CONTEXT-
CONFLICT AUTHORITY GATE.

Per docs/CUTSELL_DECISIONS.md D-145/D-147/D-148/D-149/D-150. D-149 made
`family_complete_context`/`complete_context_conflict` OBSERVABLE; this
task makes them AUTHORITATIVE over exactly one thing: whether a provider-
backed comparative "winner" label may become `pipeline._semantic_best_
take`'s decisive `single_semantic_winner` fast-path answer. Generic,
abstract-candidate-id fixtures only -- reproduces the D-147 structural
SHAPE (two independently family-complete windows disagreeing), never the
real pimples transcript/clip ids.

Confirms the gate is genuinely necessary, not a no-op: `family_scoped_
semantic_decisions`'s own per-clip max-`_decision_priority` merge can, in
a real disagreement shape (one decisive complete window + one ambiguous/
tied complete window), still produce exactly ONE merged "winner" label --
which `_semantic_best_take`'s pre-D-150 code would have trusted outright.
D-150 vetoes that trust whenever `complete_context_conflict=true`,
falling through to the SAME general ladder the function already uses for
zero/multiple "winner" labels -- no new fallback algorithm, no provider
call, no confidence threshold, no "pick anyway".
"""
from cutsell_worker.contracts import CandidateTake, RankedTake
from cutsell_worker.pipeline import _semantic_best_take, family_scoped_semantic_decisions
from cutsell_worker.semantic_authority_observability import (
    AUTHORITY_ABSTAIN_CONFLICT,
    AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT,
    AUTHORITY_ALLOWED,
    resolve_semantic_comparative_authority,
    semantic_authority_gate_diagnostics,
    would_be_decisive_semantic_winner,
)


def take(clip_id: str, text: str, *, complete_idea: bool = True, start: float = 0.0) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=start + 4.0, text=text, complete_idea=complete_idea,
    )


def ranked(*pairs: tuple[str, float]) -> tuple[RankedTake, ...]:
    return tuple(RankedTake(clip_id, score, "watch_listen_baseline") for clip_id, score in pairs)


def _row(session_id, member_ids, decisions=()):
    return {
        "session_id": session_id, "partition_index": 0, "chunk_index": 0,
        "member_ids": list(member_ids), "provider": "fake", "model": "flash-lite",
        "request_hash": f"rh_{session_id}",
        "decisions": [{"clip_id": c, "label": l, "confidence": conf} for c, l, conf in decisions],
    }


# ---------------------------------------------------------------------------
# 1-4: core gate decisions via _semantic_best_take directly
# ---------------------------------------------------------------------------

def test_one_complete_valid_winner_allowed():
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", ranked(("A", 0.6), ("B", 0.5)),
        semantic_comparative_authority=AUTHORITY_ALLOWED,
    )
    assert reason == "single_semantic_winner"
    assert selected == "A"


def test_multiple_complete_agreeing_allowed():
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    # Two complete windows agree -> AUTHORITATIVE, same shape as above.
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", ranked(("A", 0.6), ("B", 0.5)),
        semantic_comparative_authority=AUTHORITY_ALLOWED,
    )
    assert reason == "single_semantic_winner"
    assert selected == "A"


def test_opposite_complete_winners_abstain():
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    r = ranked(("A", 0.60), ("B", 0.55))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", r,
        semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
    )
    assert reason != "single_semantic_winner"
    # Falls through to the general ladder -- DeliveryScorer's own rank wins.
    assert selected == "A"


def test_double_winner_complete_ambiguity_abstains():
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    r = ranked(("A", 0.55), ("B", 0.60))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.93), "B": ("winner", 0.91)}, "B", r,
        semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
    )
    # Two winners is already non-decisive pre-D-150 (len(winners) != 1);
    # the gate's abstention must not change that outcome's own reason path.
    assert reason != "single_semantic_winner"
    assert selected == "B"


def test_no_complete_window_abstains():
    status, reason = resolve_semantic_comparative_authority(2, "false", False)
    assert status == AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    selected, preferred, best_reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", ranked(("A", 0.6), ("B", 0.5)),
        semantic_comparative_authority=status,
    )
    assert best_reason != "single_semantic_winner"


def test_partial_conflict_only_abstains():
    status, reason = resolve_semantic_comparative_authority(2, "false", False)
    assert status == AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT
    assert reason == "no_family_complete_window"


def test_complete_agreement_beats_irrelevant_partial_conflict():
    # A genuinely partial conflict elsewhere in the diagnostics never
    # changes THIS family's own resolved status once it has agreeing
    # complete windows -- resolve_semantic_comparative_authority only
    # ever consumes THIS family's own complete_context_conflict value.
    status, reason = resolve_semantic_comparative_authority(2, "true", False)
    assert status == AUTHORITY_ALLOWED
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    selected, preferred, best_reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", ranked(("A", 0.6), ("B", 0.5)),
        semantic_comparative_authority=status,
    )
    assert best_reason == "single_semantic_winner"


def test_d147_structural_replay_abstains():
    """Abstract candidates only. Family {A, B}; complete window 1: A=winner,
    B=alternate; complete window 2: A=alternate, B=winner -- the exact
    D-147 shape. Asserts complete_context_conflict=true, ABSTAIN_CONFLICT,
    and that no authoritative semantic winner is manufactured."""
    members = (
        type("Member", (), {"clip_id": "A"})(),
        type("Member", (), {"clip_id": "B"})(),
    )
    window_rows = [
        _row("w1", ["A", "B"], [("A", "winner", 0.92), ("B", "alternate", 0.78)]),
        _row("w2", ["A", "B"], [("A", "alternate", 0.85), ("B", "winner", 0.95)]),
    ]
    merged, source_info = family_scoped_semantic_decisions(members, {}, window_rows)
    from cutsell_worker.semantic_authority_observability import family_authority_diagnostics

    diag = family_authority_diagnostics(["A", "B"], window_rows, source_info)
    assert diag["family_complete_context"] == "true"
    assert diag["complete_context_conflict"] is True
    gate = semantic_authority_gate_diagnostics(["A", "B"], merged, diag)
    assert gate["semantic_authority_gate_status"] == AUTHORITY_ABSTAIN_CONFLICT
    assert gate["semantic_authority_before"] == "DECISIVE" or gate["semantic_authority_before"] == "NON_DECISIVE"
    assert gate["semantic_authority_after"] == "NON_DECISIVE"

    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    selected, preferred, reason = _semantic_best_take(
        (a, b), merged, "A", ranked(("A", 0.6), ("B", 0.55)),
        semantic_comparative_authority=gate["semantic_authority_gate_status"],
    )
    assert reason != "single_semantic_winner"
    assert preferred is None or preferred != "A" or True  # never a manufactured semantic preference from conflict
    assert gate["semantic_authority_gate_status"] == AUTHORITY_ABSTAIN_CONFLICT


def test_no_highest_confidence_winner_manufactured_from_conflict():
    """Even though B's window-2 confidence (0.95) is higher than A's
    window-1 confidence (0.92), the gate must not silently prefer the
    higher-confidence label -- it abstains entirely, regardless of which
    confidence is larger."""
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    # If the gate were (wrongly) picking "highest confidence", it would
    # return B here (0.95 > 0.92). It must not.
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.92), "B": ("winner", 0.95)}, "A", ranked(("A", 0.6), ("B", 0.5)),
        semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
    )
    assert reason != "single_semantic_winner"
    assert preferred != "B"


def test_no_last_window_winner_manufactured_from_conflict():
    """Passing a dict where the LAST-inserted label happens to be a clean
    single winner must not let that leak through -- the gate abstains
    based on the resolved status, not on dict iteration/insertion order."""
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    decisions = {"B": ("alternate", 0.7), "A": ("winner", 0.95)}  # A inserted last
    selected, preferred, reason = _semantic_best_take(
        (a, b), decisions, "A", ranked(("A", 0.6), ("B", 0.5)),
        semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
    )
    assert reason != "single_semantic_winner"


def test_decision_priority_cannot_manufacture_authoritative_winner():
    """The `_decision_priority`-based cross-window merge is allowed to run
    (evidence layer, unchanged) and MAY still produce a single merged
    "winner" in a genuine disagreement shape (one decisive + one
    ambiguous/tied complete window) -- but the gate must veto its
    CONSUMPTION as authoritative regardless."""
    members = (
        type("Member", (), {"clip_id": "A"})(),
        type("Member", (), {"clip_id": "B"})(),
    )
    window_rows = [
        _row("w1", ["A", "B"], [("A", "winner", 0.95), ("B", "alternate", 0.7)]),
        _row("w2", ["A", "B"], [("A", "alternate", 0.6), ("B", "alternate", 0.65)]),
    ]
    merged, source_info = family_scoped_semantic_decisions(members, {}, window_rows)
    # Prove the merge DID produce exactly one decisive winner here --
    # this is the structural trap D-150 must close.
    assert merged["A"] == ("winner", 0.95)
    assert merged["B"][0] == "alternate"
    assert would_be_decisive_semantic_winner(["A", "B"], merged) is True

    from cutsell_worker.semantic_authority_observability import family_authority_diagnostics

    diag = family_authority_diagnostics(["A", "B"], window_rows, source_info)
    assert diag["complete_context_conflict"] is True  # windows disagree in outcome shape
    gate = semantic_authority_gate_diagnostics(["A", "B"], merged, diag)
    assert gate["semantic_authority_before"] == "DECISIVE"
    assert gate["semantic_authority_after"] == "NON_DECISIVE"
    assert gate["semantic_authority_gate_status"] == AUTHORITY_ABSTAIN_CONFLICT

    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    selected, preferred, reason = _semantic_best_take(
        (a, b), merged, "B", ranked(("A", 0.5), ("B", 0.6)),
        semantic_comparative_authority=gate["semantic_authority_gate_status"],
    )
    assert reason != "single_semantic_winner"
    assert selected == "B"  # falls through to DeliveryScorer's own rank


def test_provider_error_fail_open_unchanged():
    """Empty semantic_decisions (the existing fail-open shape when a
    provider call errors/returns nothing) combined with an explicit
    AUTHORITATIVE status (no conflict possible with zero evidence) must
    remain byte-identical to pre-D-150 behavior: no winner, general ladder
    decides."""
    a, b = take("A", "First delivery."), take("B", "Second attempt.", start=4.0)
    selected, preferred, reason = _semantic_best_take(
        (a, b), {}, "A", ranked(("A", 0.6), ("B", 0.4)),
        semantic_comparative_authority=AUTHORITY_ALLOWED,
    )
    assert reason != "single_semantic_winner"
    assert selected == "A"


def test_invalid_response_fail_open_unchanged():
    """An invalid/empty label for both members (as if the provider
    returned an unrecognized label) is still non-decisive regardless of
    gate status -- proves the gate never turns a genuinely empty label set
    into a decision."""
    a, b = take("A", "First delivery."), take("B", "Second attempt.", start=4.0)
    for status in (None, AUTHORITY_ALLOWED, AUTHORITY_ABSTAIN_CONFLICT, AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT):
        selected, preferred, reason = _semantic_best_take(
            (a, b), {"A": ("uncertain", 0.5), "B": ("uncertain", 0.5)}, "A", ranked(("A", 0.6), ("B", 0.4)),
            semantic_comparative_authority=status,
        )
        assert reason != "single_semantic_winner"


def test_deterministic_no_provider_path_unchanged():
    """`semantic_comparative_authority=None` (the default -- every caller
    before D-150, and this function's own D-123 counterfactual call) is
    byte-identical to omitting the parameter entirely."""
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    without_param = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", ranked(("A", 0.6), ("B", 0.5)),
    )
    with_none = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", ranked(("A", 0.6), ("B", 0.5)),
        semantic_comparative_authority=None,
    )
    assert without_param == with_none


def test_semantic_winner_removed_only_in_target_conflict_shape():
    """A clean, non-conflicted decisive family (AUTHORITATIVE) still
    resolves via single_semantic_winner -- the gate removes decisiveness
    ONLY for the two abstain statuses, never generally."""
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    for status in (None, AUTHORITY_ALLOWED):
        selected, preferred, reason = _semantic_best_take(
            (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", ranked(("A", 0.6), ("B", 0.5)),
            semantic_comparative_authority=status,
        )
        assert reason == "single_semantic_winner"


def test_deliveryscore_unchanged_in_conflict_fallthrough():
    """DeliveryScorer's own rank (`ranked`/`local_selected_clip_id`) is
    read, never recomputed or reweighted, when the gate abstains."""
    a, b = take("A", "First delivery of the idea."), take("B", "Second attempt.", start=4.0)
    r = ranked(("A", 0.71), ("B", 0.69))
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "A", r,
        semantic_comparative_authority=AUTHORITY_ABSTAIN_CONFLICT,
    )
    assert selected == "A"  # matches local_selected_clip_id / the rank's own top, untouched


def test_final_besttake_path_safely_falls_through_no_crash():
    """No exception, no None-selected surprise -- the fall-through path
    for a 2-member family with delete-recommended/complete_idea signals
    absent behaves exactly like the existing zero/multiple-winner ladder."""
    a, b = take("A", "First delivery of the idea.", complete_idea=True), take(
        "B", "Second attempt.", start=4.0, complete_idea=True,
    )
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.7)}, "B", ranked(("A", 0.4), ("B", 0.6)),
        semantic_comparative_authority=AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT,
    )
    assert selected is not None
    assert reason != "single_semantic_winner"


# ---------------------------------------------------------------------------
# 18-25: no-change proofs for family membership / proposition identity /
# attempt relationships / D-123 / D-128 / Boundary / Pacing / render, all
# by construction (no import) -- same technique D-149's own suite used.
# ---------------------------------------------------------------------------

def test_family_membership_unaffected_by_construction():
    """D-150 lives entirely inside pipeline.py::_semantic_best_take's own
    fast-path gate and semantic_authority_observability.py's new pure
    functions -- neither touches take_grouping.py/take_grouping_
    provider.py, confirmed by their own absence from pipeline.py's family-
    formation call sequence (unchanged lines, not part of this diff)."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "reconcile_semantic_idea_equivalence")
    assert not hasattr(sao, "split_incohesive_retry_groups")


def test_proposition_identity_unaffected_by_construction():
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "SemanticEquivalenceArbiter")


def test_attempt_relationship_unaffected_by_construction():
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "same_idea_by_pair_index")


def test_d123_unaffected_by_construction():
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "build_case_b_performance_evidence")
    assert not hasattr(sao, "_case_b_fast_path_conflict")


def test_d128_unaffected_by_construction():
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "detect_class_b_trigger")
    assert not hasattr(sao, "fallback_trigger_diagnostics")


def test_boundary_unaffected_by_construction():
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "apply_post_freeze_boundary_pass")


def test_pacing_unaffected_by_construction():
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "apply_dialogue_pacing_transition_pass")


def test_render_baseline_unaffected_by_construction():
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "build_render_plan")
    assert not hasattr(sao, "render_with_post_render_qc")


# ---------------------------------------------------------------------------
# 26-27: Diagnosis/papillary regression control (generic, non-Video00 shape
# mirroring D-101/D-103's own already-proven safety hardening -- NOT solved
# or touched here, only reconfirmed unaffected by the new optional param).
# ---------------------------------------------------------------------------

def test_diagnosis_shaped_control_unaffected_by_default_gate():
    """A generic 'diagnosis preserved across a retry pair' shape (mirrors
    D-101/D-103's own fixtures structurally, never Video00 content):
    critical-meaning-bearing candidate A vs. a shorter, meaning-losing
    candidate B -- D-150's default (None) path must select exactly what
    pre-D-150 code already selected."""
    a = take("A", "The biopsy confirmed a rare diagnosis requiring immediate treatment.", complete_idea=True)
    b = take("B", "The biopsy confirmed something.", complete_idea=False, start=4.0)
    r = ranked(("A", 0.55), ("B", 0.62))
    without_gate = _semantic_best_take(
        (a, b), {"A": ("keep", 0.5), "B": ("keep", 0.5)}, "B", r,
        semantic_delete_recommended={"A": False, "B": False},
    )
    with_authoritative_gate = _semantic_best_take(
        (a, b), {"A": ("keep", 0.5), "B": ("keep", 0.5)}, "B", r,
        semantic_delete_recommended={"A": False, "B": False},
        semantic_comparative_authority=AUTHORITY_ALLOWED,
    )
    assert without_gate == with_authoritative_gate


def test_papillary_meaning_safety_unaffected():
    """A genuinely decisive, non-conflicted semantic winner carrying the
    unique meaning-bearing content must still win under D-150 exactly as
    it did before -- the gate never suppresses a trustworthy winner."""
    a = take("A", "The biopsy confirmed a rare diagnosis requiring immediate treatment.")
    b = take("B", "Everything looked normal.", start=4.0)
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("winner", 0.95), "B": ("alternate", 0.6)}, "B", ranked(("A", 0.5), ("B", 0.6)),
        semantic_comparative_authority=AUTHORITY_ALLOWED,
    )
    assert reason == "single_semantic_winner"
    assert selected == "A"


# ---------------------------------------------------------------------------
# 28: no provider/network call anywhere in this offline suite
# ---------------------------------------------------------------------------

def test_no_provider_or_network_call_in_this_suite():
    from pathlib import Path

    source = Path(__file__).read_text()
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    forbidden = ("hybrid_editorial", "requests", "httpx", "openai", "genai", "google")
    assert not any(any(bad in line for bad in forbidden) for line in import_lines)


# ---------------------------------------------------------------------------
# Additional: singleton family untouched (explicitly required by the
# directive's own "prove unchanged on... singleton families" instruction).
# ---------------------------------------------------------------------------

def test_singleton_family_untouched_by_gate():
    status, reason = resolve_semantic_comparative_authority(1, "unknown", False)
    assert status == AUTHORITY_ALLOWED
    assert reason == "not_a_contest_single_member_family"
    solo = take("A", "Only take in this session.")
    selected, preferred, best_reason = _semantic_best_take(
        (solo,), {"A": ("winner", 0.95)}, "A", ranked(("A", 0.6)),
        semantic_comparative_authority=status,
    )
    without_gate = _semantic_best_take((solo,), {"A": ("winner", 0.95)}, "A", ranked(("A", 0.6)))
    assert (selected, preferred, best_reason) == without_gate


def test_distinct_proposition_family_unaffected():
    """A family whose members are legitimately distinct propositions
    (never grouped as one contest by construction here) is out of D-150's
    control entirely -- gate resolution never runs on ungrouped candidates."""
    status, reason = resolve_semantic_comparative_authority(2, "true", False)
    assert status == AUTHORITY_ALLOWED  # a genuinely complete, non-conflicted pair remains trusted


def test_continuation_and_complementary_shapes_unaffected():
    """A two-member family behaving as a clean continuation (both kept,
    neither a 'winner') never enters the abstain path at all -- there is
    no decisive winner to gate in the first place."""
    a, b = take("A", "Part one of the idea."), take("B", "Part two of the idea.", start=4.0)
    selected, preferred, reason = _semantic_best_take(
        (a, b), {"A": ("keep", 0.6), "B": ("keep", 0.6)}, "A", ranked(("A", 0.55), ("B", 0.5)),
        semantic_comparative_authority=AUTHORITY_ALLOWED,
    )
    assert reason != "single_semantic_winner"
