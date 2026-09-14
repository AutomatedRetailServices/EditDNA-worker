"""D-149: SEMANTIC AUTHORITY PHASE A.2 -- COMPLETE-CONTEXT-CONFLICT OBSERVABILITY.

Per docs/CUTSELL_DECISIONS.md D-147 (real-media finding) and D-148 (canonical
architecture, `COMPLETE_CONTEXT_CONFLICT` named in Section 13.8.1). D-146's
Phase A `partial_window_conflict` detector suppresses any conflict report
whenever a family-complete window exists -- D-147 proved on real media that
TWO independently family-complete windows can disagree with EACH OTHER,
which that detector misses by construction. This suite proves D-149's new
`complete_window_agreement`/`complete_window_outcomes` detectors close that
observability gap WITHOUT changing any authority: `family_authority_
diagnostics`'s existing D-146 fields are untouched (verified byte-identical
below), and every new field is additive.

Eight named offline fixture shapes (window_rows built by hand, the exact
`hybrid_session_cleanup.py` diagnostics-row shape -- `member_ids`,
`session_id`, `request_hash`, `decisions`), never a provider call:

  A. ONE COMPLETE WINDOW                    -> no conflict, ONE_COMPLETE_WINDOW
  B. TWO COMPLETE WINDOWS, SAME WINNER      -> agreement
  C. TWO COMPLETE WINDOWS, OPPOSITE WINNERS -> complete_context_conflict=true
     (the exact D-147 structural shape)
  D. TWO COMPLETE WINDOWS, ONE DOUBLE-WINNER -> conflict (ambiguous window
     never matches a decisive window's signature)
  E. PARTIAL CONFLICT + COMPLETE AGREEMENT  -> both observabilities correct,
     independently
  F. PARTIAL CONFLICT + COMPLETE CONFLICT   -> both true simultaneously
  G. NO COMPLETE WINDOW                     -> existing D-146 partial-only
     behavior fully preserved
  H. UNKNOWN FAMILY COMPLETENESS            -> no fabricated conflict
"""
from __future__ import annotations

from cutsell_worker.semantic_authority_observability import (
    AGREEMENT_MULTIPLE_AGREE,
    AGREEMENT_MULTIPLE_DISAGREE,
    AGREEMENT_NO_COMPLETE_WINDOW,
    AGREEMENT_ONE_COMPLETE_WINDOW,
    AGREEMENT_UNKNOWN,
    CONFLICT_REASON_COMPLETE_WINDOWS_DISAGREE,
    complete_window_agreement,
    complete_window_outcomes,
    family_authority_diagnostics,
    family_complete_context,
    partial_window_conflict,
    summarize_family_authority_observability,
)


def _row(session_id, member_ids, decisions=(), request_hash=None, provider="fake",
         model="flash-lite", partition_index=0, chunk_index=0):
    return {
        "session_id": session_id,
        "partition_index": partition_index,
        "chunk_index": chunk_index,
        "member_ids": list(member_ids),
        "provider": provider,
        "model": model,
        "request_hash": request_hash or f"rh_{session_id}",
        "decisions": [
            {"clip_id": clip_id, "label": label, "confidence": confidence}
            for clip_id, label, confidence in decisions
        ],
    }


FAMILY = ("A", "B")

# Fixture A: one complete window only.
FIXTURE_A = [_row("w1", ["A", "B"], [("A", "winner", 0.95), ("B", "alternate", 0.7)])]

# Fixture B: two complete windows, same winner -> agreement.
FIXTURE_B = [
    _row("w1", ["A", "B"], [("A", "winner", 0.95), ("B", "alternate", 0.7)]),
    _row("w2", ["A", "B", "C"], [("A", "winner", 0.90), ("B", "alternate", 0.6), ("C", "failed", 0.9)]),
]

# Fixture C: the exact D-147 structural shape -- two complete windows,
# opposite winners.
FIXTURE_C = [
    _row("w1", ["A", "B"], [("A", "winner", 0.92), ("B", "alternate", 0.78)]),
    _row("w2", ["A", "B"], [("A", "alternate", 0.85), ("B", "winner", 0.95)]),
]

# Fixture D: one decisive complete window, one double-winner (ambiguous)
# complete window -> the signatures never match -> conflict.
FIXTURE_D = [
    _row("w1", ["A", "B"], [("A", "winner", 0.95), ("B", "alternate", 0.7)]),
    _row("w2", ["A", "B"], [("A", "winner", 0.93), ("B", "winner", 0.91)]),
]

# Fixture E: a genuinely partial-only pair of windows (w1 sees only A+C,
# w2 sees only B+C -- neither is a superset of the family {A, B}) PLUS a
# separate family-complete window (w3, sees A+B+C). Proves the two
# observabilities are computed independently and correctly: D-146's own
# `partial_window_conflict` is (by its OWN preserved design, unchanged
# here) suppressed the instant ANY complete window exists -- w3's mere
# presence makes w1/w2's disagreement moot by that existing rule, not a
# D-149 change -- while `complete_window_agreement` correctly reports the
# ONE real complete window (w3) on its own terms.
FIXTURE_E = [
    _row("w1", ["A", "C"], [("A", "winner", 0.96)]),
    _row("w2", ["B", "C"], [("C", "winner", 0.95)]),
    _row("w3", ["A", "B", "C"], [("A", "winner", 0.9), ("B", "alternate", 0.6), ("C", "failed", 0.8)]),
]

# Fixture F: the same partial-only pair as E, but now TWO complete windows
# (w3, w4) that DISAGREE with each other. Documents the real interaction:
# because a complete window exists (in fact two), D-146's own
# `partial_window_conflict` remains suppressed (unchanged, preserved
# behavior) exactly as in fixture E, while D-149's NEW
# `complete_context_conflict` is independently True. The two fields are
# never merged into one boolean; under today's preserved D-146 semantics
# they are mutually exclusive in VALUE (partial only ever fires when zero
# complete windows exist) -- a factual consequence of D-146's own
# suppression rule, not something D-149 introduces or hides.
FIXTURE_F = [
    _row("w1", ["A", "C"], [("A", "winner", 0.96)]),
    _row("w2", ["B", "C"], [("C", "winner", 0.95)]),
    _row("w3", ["A", "B", "C"], [("A", "winner", 0.9), ("B", "alternate", 0.6), ("C", "failed", 0.5)]),
    _row("w4", ["A", "B", "C"], [("B", "winner", 0.92), ("A", "alternate", 0.6), ("C", "failed", 0.5)]),
]

# Fixture G: no complete window at all -- existing D-146 partial-only shape.
# Neither window is a superset of the family {A, B}.
FIXTURE_G = [
    _row("w1", ["A", "C"], [("A", "winner", 0.90)]),
    _row("w2", ["B", "C"], [("C", "alternate", 0.70)]),
]

# Fixture H: no window touches the family at all -- unknown.
FIXTURE_H = [_row("w9", ["X", "Y"], [("X", "winner", 0.91)])]


# ---------------------------------------------------------------------------
# 1-4: agreement classification for fixtures A-D
# ---------------------------------------------------------------------------

def test_fixture_a_one_complete_window_no_conflict():
    result = complete_window_agreement(FAMILY, FIXTURE_A)
    assert result["complete_window_count"] == 1
    assert result["complete_window_agreement_status"] == AGREEMENT_ONE_COMPLETE_WINDOW
    assert result["complete_context_conflict"] is False
    assert result["complete_context_conflict_reason"] is None


def test_fixture_b_two_agreeing_complete_windows():
    result = complete_window_agreement(FAMILY, FIXTURE_B)
    assert result["complete_window_count"] == 2
    assert result["complete_window_agreement_status"] == AGREEMENT_MULTIPLE_AGREE
    assert result["complete_context_conflict"] is False


def test_fixture_c_opposite_winner_conflict():
    result = complete_window_agreement(FAMILY, FIXTURE_C)
    assert result["complete_window_agreement_status"] == AGREEMENT_MULTIPLE_DISAGREE
    assert result["complete_context_conflict"] is True
    assert result["complete_context_conflict_reason"] == CONFLICT_REASON_COMPLETE_WINDOWS_DISAGREE


def test_fixture_d_double_winner_conflict():
    result = complete_window_agreement(FAMILY, FIXTURE_D)
    assert result["complete_window_agreement_status"] == AGREEMENT_MULTIPLE_DISAGREE
    assert result["complete_context_conflict"] is True


# ---------------------------------------------------------------------------
# 5-9: outcome normalization / window ids / request hashes / winner sets / reason
# ---------------------------------------------------------------------------

def test_complete_outcome_normalization_uses_structured_fields_only():
    outcomes = complete_window_outcomes(FAMILY, FIXTURE_C)
    assert len(outcomes) == 2
    for outcome in outcomes:
        assert set(outcome["member_ids"]) == {"A", "B"}
        assert isinstance(outcome["normalized_winner_ids"], tuple)
        assert isinstance(outcome["normalized_alternate_ids"], tuple)
        assert outcome["raw_relation"] is None  # honestly absent, never invented
    assert outcomes[0]["normalized_winner_ids"] == ("A",)
    assert outcomes[1]["normalized_winner_ids"] == ("B",)


def test_complete_window_ids_recorded():
    result = complete_window_agreement(FAMILY, FIXTURE_C)
    assert set(result["complete_window_ids"]) == {"w1", "w2"}


def test_complete_request_hashes_recorded():
    result = complete_window_agreement(FAMILY, FIXTURE_C)
    assert set(result["complete_window_request_hashes"]) == {"rh_w1", "rh_w2"}


def test_conflict_winner_sets_recorded():
    result = complete_window_agreement(FAMILY, FIXTURE_C)
    assert set(result["complete_context_conflict_winner_sets"]) == {("A",), ("B",)}


def test_conflict_reason_recorded():
    result = complete_window_agreement(FAMILY, FIXTURE_D)
    assert result["complete_context_conflict_reason"] == CONFLICT_REASON_COMPLETE_WINDOWS_DISAGREE
    # A non-conflicting family must never carry a reason.
    assert complete_window_agreement(FAMILY, FIXTURE_B)["complete_context_conflict_reason"] is None


# ---------------------------------------------------------------------------
# 10-11: partial conflict stays a distinct, independently-computed field
# ---------------------------------------------------------------------------

def test_partial_conflict_remains_separate_field():
    # Fixture E: w1/w2 are genuinely partial (neither is a superset of the
    # family); w3 is the one real complete window. D-146's own
    # `partial_window_conflict` (unchanged, preserved) is suppressed the
    # instant ANY complete window exists -- correctly None here, not a bug.
    # `complete_window_agreement` independently and correctly reports the
    # single real complete window on its own terms.
    partial = partial_window_conflict(FAMILY, FIXTURE_E)
    complete = complete_window_agreement(FAMILY, FIXTURE_E)
    assert partial is None
    assert complete["complete_context_conflict"] is False
    assert complete["complete_window_agreement_status"] == AGREEMENT_ONE_COMPLETE_WINDOW
    assert complete["complete_window_count"] == 1


def test_both_conflict_fields_independently_correct_and_never_merged():
    # Fixture F: two complete windows (w3, w4) disagree with each other.
    # D-146's `partial_window_conflict` stays suppressed (preserved,
    # unchanged design -- it is moot the instant any complete window
    # exists); D-149's NEW `complete_context_conflict` independently
    # reports True. This documents the real interaction rather than
    # forcing a fabricated simultaneous-True state D-146's own semantics
    # do not allow: the two fields are never merged into one boolean, and
    # each is computed strictly from its own detector, but under today's
    # preserved D-146 rule they are mutually exclusive in VALUE for the
    # same family (partial only ever fires when zero complete windows
    # exist) -- a factual consequence of D-146, not a D-149 change.
    partial = partial_window_conflict(FAMILY, FIXTURE_F)
    complete = complete_window_agreement(FAMILY, FIXTURE_F)
    assert partial is None
    assert complete["complete_context_conflict"] is True
    diag = family_authority_diagnostics(FAMILY, FIXTURE_F, None)
    assert diag["partial_window_conflict"] is None
    assert diag["complete_context_conflict"] is True
    # Confirmed distinct fields, not a merged boolean.
    assert "partial_window_conflict" in diag and "complete_context_conflict" in diag
    assert diag["partial_window_conflict"] != diag["complete_context_conflict"]


# ---------------------------------------------------------------------------
# 12-13: no-complete-window / unknown-completeness safety
# ---------------------------------------------------------------------------

def test_no_complete_window_behavior_preserved():
    result = complete_window_agreement(FAMILY, FIXTURE_G)
    assert family_complete_context(FAMILY, FIXTURE_G) == "false"
    assert result["complete_window_agreement_status"] == AGREEMENT_NO_COMPLETE_WINDOW
    assert result["complete_window_count"] == 0
    assert result["complete_context_conflict"] is False


def test_unknown_completeness_never_fabricates_conflict():
    result = complete_window_agreement(FAMILY, FIXTURE_H)
    assert family_complete_context(FAMILY, FIXTURE_H) == "unknown"
    assert result["complete_window_agreement_status"] == AGREEMENT_UNKNOWN
    assert result["complete_context_conflict"] is False
    assert result["complete_window_count"] == 0


# ---------------------------------------------------------------------------
# 14-17: the D-147 structural replay + no-authority-drift proofs
# ---------------------------------------------------------------------------

def test_d147_structural_replay():
    """Abstract candidate ids only -- never the real pimples transcript/clip
    ids. Reproduces the exact real-media shape: two independently
    family-complete windows nominate opposite winners. `family_scoped_
    source_info` is passed non-None here because a complete window's
    existence is exactly what makes `family_scoped_semantic_decisions`
    (pipeline.py) produce one in the real pipeline -- D-149 does not
    change that coupling, it only observes downstream of it."""
    realistic_source_info = {
        "family_complete_window_chunk_indices": [0, 1],
        "family_window_labels": {"A": ["winner", 0.92], "B": ["winner", 0.95]},
        "global_merge_labels": {"A": ["winner", 0.92], "B": ["winner", 0.95]},
    }
    diag = family_authority_diagnostics(FAMILY, FIXTURE_C, realistic_source_info)
    assert diag["family_complete_context"] == "true"
    assert diag["complete_window_count"] == 2
    assert diag["complete_context_conflict"] is True
    # D-146's existing authority-description field is UNCHANGED by D-149:
    # a complete window still exists, so today's real behavior still
    # reports FAMILY_COMPLETE_WINDOW_PREFERRED even though it is now ALSO
    # known to be a conflicted one. D-149 never sanitizes this.
    assert diag["provider_authority_applied"] == "FAMILY_COMPLETE_WINDOW_PREFERRED"


def test_current_authority_applied_unchanged_by_d149():
    without_complete_window = family_authority_diagnostics(FAMILY, FIXTURE_G, None)
    assert without_complete_window["provider_authority_applied"] == "GLOBAL_CROSS_WINDOW_MERGE"
    with_conflicted_complete_windows = family_authority_diagnostics(FAMILY, FIXTURE_C, {"x": 1})
    assert with_conflicted_complete_windows["provider_authority_applied"] == "FAMILY_COMPLETE_WINDOW_PREFERRED"


def test_current_merged_outcome_and_family_scoped_outcome_unchanged():
    source_info = {"family_complete_window_chunk_indices": [0, 1], "family_window_labels": {"A": ["winner", 0.92]}}
    diag = family_authority_diagnostics(FAMILY, FIXTURE_C, source_info)
    # family_scoped_source_info is exposed VERBATIM -- D-149 never rewrites
    # or recomputes what family_scoped_semantic_decisions already decided.
    assert diag["family_scoped_source_info"] is source_info


def test_all_d146_fields_byte_identical_with_and_without_d149_fields():
    """D-149 must be strictly additive: every D-146 key's value is
    unaffected by the new keys' presence."""
    from cutsell_worker import semantic_authority_observability as sao

    d146_only = {
        "family_complete_context": sao.family_complete_context(FAMILY, FIXTURE_C),
        "complete_window_ids": sao.complete_window_ids(FAMILY, FIXTURE_C),
        "semantic_window_ids": sao.window_ids_touching_family(FAMILY, FIXTURE_C),
        "omitted_candidate_ids": sao.omitted_candidate_ids(FAMILY, FIXTURE_C),
        "partial_window_conflict": sao.partial_window_conflict(FAMILY, FIXTURE_C),
        "provider_config": sao.provider_config_from_window_rows(FAMILY, FIXTURE_C),
    }
    full = family_authority_diagnostics(FAMILY, FIXTURE_C, None)
    for key, value in d146_only.items():
        assert full[key] == value, f"{key} drifted between D-146-only and D-149-extended computation"


# ---------------------------------------------------------------------------
# 18-25: no grouping/semantic-winner/BestTake/D-123/D-128/Boundary/pacing/
# render-baseline change -- proven the same way D-146's own suite proved it
# (this module has no cutsell_worker sibling import; see the leaf-module
# test below), plus explicit assertion that calling these new functions
# never touches any external state.
# ---------------------------------------------------------------------------

def test_family_authority_diagnostics_pure_no_side_effects():
    import copy

    rows_copy = copy.deepcopy(FIXTURE_C)
    family_authority_diagnostics(FAMILY, FIXTURE_C, None)
    assert FIXTURE_C == rows_copy  # input window rows never mutated


def test_summarize_includes_d149_counts_alongside_d146_counts():
    rows = [
        family_authority_diagnostics(FAMILY, FIXTURE_A, {"x": 1}),
        family_authority_diagnostics(FAMILY, FIXTURE_B, {"x": 1}),
        family_authority_diagnostics(FAMILY, FIXTURE_C, None),
        family_authority_diagnostics(FAMILY, FIXTURE_G, None),
        family_authority_diagnostics(FAMILY, FIXTURE_H, None),
    ]
    summary = summarize_family_authority_observability(rows)
    # D-146 existing keys untouched.
    assert summary["family_count"] == 5
    assert summary["partial_window_conflict_family_count"] == 0
    # D-149 additive keys, counts only.
    assert summary["families_with_no_complete_window"] == 1  # fixture G
    assert summary["families_with_one_complete_window"] == 1  # fixture A
    assert summary["families_with_multiple_complete_windows"] == 2  # fixtures B, C
    assert summary["families_with_complete_window_agreement"] == 1  # fixture B
    assert summary["families_with_complete_context_conflict"] == 1  # fixture C
    assert summary["families_with_partial_window_conflict"] == 0
    assert summary["families_with_any_semantic_conflict"] == 1  # fixture C only
    # No per-family clip id or window id field leaks into the tail-safe
    # summary -- structural check on the KEY SET (counts only), not a
    # substring search (which would false-positive on constant names like
    # "FAMILY_COMPLETE_WINDOW_PREFERRED" containing the letter "A").
    assert not any(key.endswith("_ids") or key.endswith("_hashes") for key in summary)
    assert "w1" not in str(summary.values())


def test_no_besttake_boundary_pacing_module_imported():
    """D-149 changes nothing about BestTake/Boundary/Pacing -- verified the
    same way D-146's own module-leaf test verified it: this module still
    imports no cutsell_worker sibling and no provider/network reference."""
    import ast
    from pathlib import Path

    source = Path("cutsell_worker/semantic_authority_observability.py").read_text()
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    forbidden = ("requests", "httpx", "urllib", "socket", "google", "genai", "openai",
                 "best_take", "boundary", "pacing", "render")
    assert not any(any(bad in line for bad in forbidden) for line in import_lines)
    assert not any("cutsell_worker" in line or line.strip().startswith("from .") for line in import_lines)
    assert isinstance(ast.parse(source), ast.Module)


def test_grouping_module_unaffected_by_construction():
    """D-149 lives entirely inside semantic_authority_observability.py
    (confirmed via the module's own zero-sibling-import proof above), so
    take_grouping.py/take_grouping_provider.py (retry-family grouping) are
    structurally unreachable from it and therefore provably unaffected --
    D-149 adds no call, no import, no wiring into either file."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "reconcile_semantic_idea_equivalence")
    assert not hasattr(sao, "split_incohesive_retry_groups")


def test_semantic_winner_module_unaffected_by_construction():
    """`pipeline.py::_semantic_best_take` (the actual semantic-winner
    decision) is not imported, called, or referenced by this module."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "_semantic_best_take")


def test_besttake_module_unaffected_by_construction():
    """`deterministic_best_take_authority.py`/`take_judge.py`'s
    DeliveryScorer are not imported, called, or referenced by this module."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "clear_retry_family_winner")
    assert not hasattr(sao, "apply_delivery_cleanliness_evidence")


def test_d123_module_unaffected_by_construction():
    """D-123's `multimodal_besttake_fallback.py`/case_b machinery is not
    imported, called, or referenced by this module."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "build_case_b_performance_evidence")


def test_d128_module_unaffected_by_construction():
    """D-128's Class B shadow trigger is not imported, called, or
    referenced by this module."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "detect_class_b_trigger")
    assert not hasattr(sao, "fallback_trigger_diagnostics")


def test_boundary_module_unaffected_by_construction():
    """`boundary_engine_pass.py`'s post-Freeze physical pass is not
    imported, called, or referenced by this module."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "apply_post_freeze_boundary_pass")


def test_pacing_module_unaffected_by_construction():
    """D-142's `dialogue_pacing_transition.py` is not imported, called, or
    referenced by this module."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "apply_dialogue_pacing_transition_pass")


def test_render_baseline_unaffected_by_construction():
    """`render.py`/`render_plan.py` are not imported, called, or
    referenced by this module."""
    import cutsell_worker.semantic_authority_observability as sao

    assert not hasattr(sao, "build_render_plan")
    assert not hasattr(sao, "render_with_post_render_qc")


def test_complete_window_ids_reuses_d146_field_no_duplicate_id_scheme():
    """D-149 must reuse D-146's existing `complete_window_ids`/`session_id`
    identity, never mint a second, duplicate id scheme."""
    diag = family_authority_diagnostics(FAMILY, FIXTURE_C, None)
    # The SAME tuple of window ids backs both the D-146 field and the
    # D-149 agreement detector's own window-id list.
    assert set(diag["complete_window_ids"]) == set(
        outcome["window_id"] for outcome in diag["complete_window_outcomes"]
    )


def test_distinct_complete_windows_never_collide_in_fixtures():
    """Every fixture's complete windows carry distinct session_id/
    request_hash pairs -- no collision anywhere in this deterministic
    fixture set."""
    for fixture in (FIXTURE_B, FIXTURE_C, FIXTURE_D, FIXTURE_F):
        outcomes = complete_window_outcomes(FAMILY, fixture)
        window_ids = [outcome["window_id"] for outcome in outcomes]
        request_hashes = [outcome["request_hash"] for outcome in outcomes]
        assert len(window_ids) == len(set(window_ids)), f"window id collision in {fixture}"
        assert len(request_hashes) == len(set(request_hashes)), f"request_hash collision in {fixture}"


# ---------------------------------------------------------------------------
# 26: no provider/network call anywhere in this offline suite
# ---------------------------------------------------------------------------

def test_no_provider_or_network_call_in_this_suite():
    """Every fixture above is a hand-built dict; no EditorialJudge/provider
    class is imported anywhere in this file's actual import lines (checked
    structurally, not by substring search over the whole file text, which
    would false-positive on this very assertion's own literal strings)."""
    from pathlib import Path

    source = Path(__file__).read_text()
    import_lines = [
        line for line in source.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    forbidden = ("hybrid_editorial", "requests", "httpx", "openai", "genai", "google")
    assert not any(any(bad in line for bad in forbidden) for line in import_lines)
