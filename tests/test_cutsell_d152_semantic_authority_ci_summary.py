"""D-152: SEMANTIC AUTHORITY GATE -- TAIL-SAFE REAL-MEDIA OBSERVABILITY.

Per docs/CUTSELL_DECISIONS.md D-151/D-152. D-151's real-media qualification
of the D-150 semantic authority gate could not recover the gate's own field
values (`semantic_authority_gate_status/_reason/_before/_after`, `family_
complete_context`, `complete_context_conflict`, `complete_window_agreement_
status`) because they print inside the large "Print full canonical
diagnostics" step and, by job end, scroll outside the CI log tool's
retrievable tail. This task adds exactly two pure, additive functions to
`semantic_authority_observability.py` -- `semantic_authority_ci_row`
(one bounded per-family row) and `summarize_semantic_authority_gate_counts`
(the top-level counts object) -- both PURE PROJECTIONS of fields `pipeline.py`
already writes onto each `take_judge_groups` row. Neither function reads a
transcript, calls a provider, or changes family formation, a semantic label,
the gate's own decision, a DeliveryScorer score, or a BestTake winner --
they only make already-computed values printable in a small, bounded,
tail-safe shape. Generic synthetic fixtures only; no pimples-specific code,
no transcript keywords, no special case.
"""
import json

from cutsell_worker.semantic_authority_observability import (
    AUTHORITY_ABSTAIN_CONFLICT,
    AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT,
    AUTHORITY_ALLOWED,
    semantic_authority_ci_row,
    summarize_semantic_authority_gate_counts,
)


def _judge_group_row(
    family_id,
    *,
    member_labels,  # sequence of (clip_id, label, confidence)
    gate_evaluated=True,
    gate_status=AUTHORITY_ALLOWED,
    gate_reason="family_complete_context_true_no_conflict",
    before="DECISIVE",
    after="DECISIVE",
    family_complete_context="true",
    complete_context_conflict=False,
    complete_window_agreement_status="ONE_COMPLETE_WINDOW",
    complete_window_count=1,
    semantic_fast_path_candidate=None,
    deliveryscore_top_candidate="A",
    winner_path_after="SEMANTIC_FAST_PATH",
):
    """Builds a synthetic dict shaped exactly like one `pipeline.py`
    `judge_group_diagnostics` ("take_judge_groups") row -- the real object
    D-152's two new functions consume. Abstract clip ids only (never real
    Video00 transcript text or clip ids)."""
    return {
        "group_id": family_id,
        "selected_clip_id": deliveryscore_top_candidate,
        "semantic_best_take_reason": "delivery_tie_break_among_survivors",
        "semantic_candidates": [
            {"clip_id": cid, "label": label, "confidence": conf}
            for cid, label, conf in member_labels
        ],
        "semantic_authority_observability": {
            "complete_window_count": complete_window_count,
            "complete_window_agreement_status": complete_window_agreement_status,
            "complete_context_conflict": complete_context_conflict,
            "family_complete_context": family_complete_context,
        },
        "semantic_authority_gate_evaluated": gate_evaluated,
        "semantic_authority_gate_status": gate_status,
        "semantic_authority_gate_reason": gate_reason,
        "semantic_authority_before": before,
        "semantic_authority_after": after,
        "complete_context_conflict": complete_context_conflict,
        "family_complete_context": family_complete_context,
        "complete_window_agreement_status": complete_window_agreement_status,
        "semantic_fast_path_candidate": semantic_fast_path_candidate,
        "deliveryscore_top_candidate": deliveryscore_top_candidate,
        "winner_path_after": winner_path_after,
        "winner_path_before": winner_path_after,
        "final_winner": deliveryscore_top_candidate,
    }


# ---------------------------------------------------------------------------
# 1-5: top-level counts
# ---------------------------------------------------------------------------

def test_top_level_counts_present():
    rows = [
        _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)]),
    ]
    counts = summarize_semantic_authority_gate_counts(rows)
    required = {
        "family_count", "semantic_authority_gate_evaluated_count",
        "semantic_authority_allowed_count", "semantic_authority_abstain_incomplete_count",
        "semantic_authority_abstain_conflict_count", "semantic_authority_advisory_count",
        "families_with_one_complete_window", "families_with_multiple_complete_windows",
        "families_with_complete_window_agreement", "families_with_complete_context_conflict",
        "families_with_no_complete_window",
    }
    assert required <= set(counts)


def test_allowed_count_correct():
    rows = [
        _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)], gate_status=AUTHORITY_ALLOWED),
        _judge_group_row("tg2", member_labels=[("C", "winner", 0.95)], gate_status=AUTHORITY_ALLOWED),
        _judge_group_row(
            "tg3", member_labels=[("D", "winner", 0.95), ("E", "winner", 0.95)],
            gate_status=AUTHORITY_ABSTAIN_CONFLICT, complete_context_conflict=True,
            complete_window_agreement_status="MULTIPLE_COMPLETE_WINDOWS_DISAGREE",
        ),
    ]
    counts = summarize_semantic_authority_gate_counts(rows)
    assert counts["semantic_authority_allowed_count"] == 2


def test_abstain_incomplete_count_correct():
    rows = [
        _judge_group_row(
            "tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)],
            gate_status=AUTHORITY_ABSTAIN_INCOMPLETE_CONTEXT,
            gate_reason="no_family_complete_window",
            family_complete_context="false", complete_window_agreement_status="NO_COMPLETE_WINDOW",
            complete_window_count=0,
        ),
        _judge_group_row("tg2", member_labels=[("C", "winner", 0.95)], gate_status=AUTHORITY_ALLOWED),
    ]
    counts = summarize_semantic_authority_gate_counts(rows)
    assert counts["semantic_authority_abstain_incomplete_count"] == 1


def test_abstain_conflict_count_correct():
    rows = [
        _judge_group_row(
            "tg1", member_labels=[("A", "winner", 0.95), ("B", "winner", 0.95)],
            gate_status=AUTHORITY_ABSTAIN_CONFLICT,
            gate_reason="multiple_family_complete_windows_disagree_on_comparative_winner",
            complete_context_conflict=True,
            complete_window_agreement_status="MULTIPLE_COMPLETE_WINDOWS_DISAGREE",
            complete_window_count=2,
        ),
        _judge_group_row("tg2", member_labels=[("C", "winner", 0.95)], gate_status=AUTHORITY_ALLOWED),
        _judge_group_row("tg3", member_labels=[("D", "winner", 0.95)], gate_status=AUTHORITY_ALLOWED),
    ]
    counts = summarize_semantic_authority_gate_counts(rows)
    assert counts["semantic_authority_abstain_conflict_count"] == 1


def test_complete_context_conflict_count_correct():
    rows = [
        _judge_group_row(
            "tg1", member_labels=[("A", "winner", 0.95), ("B", "winner", 0.95)],
            gate_status=AUTHORITY_ABSTAIN_CONFLICT, complete_context_conflict=True,
            complete_window_agreement_status="MULTIPLE_COMPLETE_WINDOWS_DISAGREE",
        ),
        _judge_group_row(
            "tg2", member_labels=[("C", "winner", 0.95), ("D", "winner", 0.95)],
            gate_status=AUTHORITY_ABSTAIN_CONFLICT, complete_context_conflict=True,
            complete_window_agreement_status="MULTIPLE_COMPLETE_WINDOWS_DISAGREE",
        ),
        _judge_group_row("tg3", member_labels=[("E", "winner", 0.95)], gate_status=AUTHORITY_ALLOWED),
    ]
    counts = summarize_semantic_authority_gate_counts(rows)
    assert counts["families_with_complete_context_conflict"] == 2
    assert counts["family_count"] == 3
    assert counts["semantic_authority_gate_evaluated_count"] == 3


# ---------------------------------------------------------------------------
# 6-14: bounded per-family row contract
# ---------------------------------------------------------------------------

def test_per_family_row_emitted():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)])
    ci_row = semantic_authority_ci_row(row)
    assert isinstance(ci_row, dict) and ci_row


def test_family_id_present():
    row = _judge_group_row("tg_generic_family_id", member_labels=[("A", "winner", 0.95)])
    assert semantic_authority_ci_row(row)["family_id"] == "tg_generic_family_id"


def test_gate_status_present():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)], gate_status=AUTHORITY_ABSTAIN_CONFLICT)
    assert semantic_authority_ci_row(row)["semantic_authority_gate_status"] == AUTHORITY_ABSTAIN_CONFLICT


def test_gate_reason_present():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)], gate_reason="not_a_contest_single_member_family")
    assert semantic_authority_ci_row(row)["semantic_authority_gate_reason"] == "not_a_contest_single_member_family"


def test_authority_before_after_present():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "winner", 0.95)], before="DECISIVE", after="NON_DECISIVE")
    ci_row = semantic_authority_ci_row(row)
    assert ci_row["semantic_authority_before"] == "DECISIVE"
    assert ci_row["semantic_authority_after"] == "NON_DECISIVE"


def test_complete_window_agreement_present():
    row = _judge_group_row(
        "tg1", member_labels=[("A", "winner", 0.95), ("B", "winner", 0.95)],
        complete_window_agreement_status="MULTIPLE_COMPLETE_WINDOWS_DISAGREE",
    )
    ci_row = semantic_authority_ci_row(row)
    assert ci_row["complete_window_agreement_status"] == "MULTIPLE_COMPLETE_WINDOWS_DISAGREE"
    assert ci_row["complete_window_count"] == 1  # default in the helper


def test_complete_window_agreement_falls_back_to_nested_when_top_level_absent():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)])
    del row["complete_window_agreement_status"]
    ci_row = semantic_authority_ci_row(row)
    assert ci_row["complete_window_agreement_status"] == "ONE_COMPLETE_WINDOW"


def test_semantic_winner_field_bounded_single_winner():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)])
    assert semantic_authority_ci_row(row)["semantic_winner_id"] == "A"


def test_semantic_winner_field_bounded_ambiguous_is_none():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "winner", 0.95)])
    assert semantic_authority_ci_row(row)["semantic_winner_id"] is None


def test_semantic_winner_field_bounded_no_winner_is_none():
    row = _judge_group_row("tg1", member_labels=[("A", "alternate", 0.6), ("B", "keep", 0.5)])
    assert semantic_authority_ci_row(row)["semantic_winner_id"] is None


def test_deliveryscore_winner_field_bounded():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)], deliveryscore_top_candidate="B")
    assert semantic_authority_ci_row(row)["deliveryscore_winner_id"] == "B"


def test_winner_path_present():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)], winner_path_after="DELIVERYSCORE_PATH")
    assert semantic_authority_ci_row(row)["winner_path_after"] == "DELIVERYSCORE_PATH"


def test_member_count_bounded_to_candidate_list_length():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6), ("C", "keep", 0.5)])
    assert semantic_authority_ci_row(row)["member_count"] == 3


# ---------------------------------------------------------------------------
# 15-17: bounded output / no leakage
# ---------------------------------------------------------------------------

def test_no_transcript_leakage():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)])
    # A realistic pipeline row never carries "text" inside semantic_candidates,
    # but simulate a hypothetical future field to prove the projector is an
    # explicit allowlist, not a passthrough.
    row["semantic_candidates"][0]["text"] = "some real spoken sentence"
    row["some_future_transcript_field"] = "spoken words that must never leak"
    ci_row = semantic_authority_ci_row(row)
    blob = json.dumps(ci_row)
    assert "spoken" not in blob
    assert "text" not in ci_row


def test_no_raw_provider_prose():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)])
    row["case_b_evidence"] = {"A": {"delivery_events": ["raw provider prose here"]}}
    row["ranked"] = [{"clip_id": "A", "score": 1.0, "reason": "provider said this exact thing"}]
    ci_row = semantic_authority_ci_row(row)
    blob = json.dumps(ci_row)
    assert "raw provider prose" not in blob
    assert "provider said" not in blob
    assert "case_b_evidence" not in ci_row
    assert "ranked" not in ci_row


def test_summary_size_remains_bounded_on_multi_family_fixture():
    rows = [
        _judge_group_row(f"tg{i}", member_labels=[(f"A{i}", "winner", 0.95), (f"B{i}", "alternate", 0.6)])
        for i in range(50)
    ]
    summary = {
        "counts": summarize_semantic_authority_gate_counts(rows),
        "families": [semantic_authority_ci_row(r) for r in rows],
    }
    blob = json.dumps(summary)
    # 50 families, ~14 short scalar fields (with verbose key names) each --
    # should stay well under 1KB/family; this is a bounded-shape smoke test
    # (no per-window/per-frame/transcript payload growth), not a tight byte
    # budget.
    assert len(blob) < 50 * 1000
    assert summary["counts"]["family_count"] == 50


# ---------------------------------------------------------------------------
# 18-21: zero editorial effect (semantic authority / family / DeliveryScorer / BestTake)
# ---------------------------------------------------------------------------

def test_no_authority_behavior_change_pure_function():
    """semantic_authority_ci_row / summarize_semantic_authority_gate_counts
    never mutate their input rows -- proves they are read-only projections,
    never a second decision authority."""
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)])
    before = json.loads(json.dumps(row, default=str))
    semantic_authority_ci_row(row)
    summarize_semantic_authority_gate_counts([row])
    after = json.loads(json.dumps(row, default=str))
    assert before == after


def test_no_semantic_winner_change():
    """The projector reads semantic_candidates/semantic_fast_path_candidate
    verbatim -- it never recomputes or overwrites which member is the
    semantic winner."""
    row = _judge_group_row(
        "tg1", member_labels=[("A", "winner", 0.95), ("B", "alternate", 0.6)],
        semantic_fast_path_candidate="A",
    )
    ci_row = semantic_authority_ci_row(row)
    assert ci_row["semantic_fast_path_candidate"] == "A" == row["semantic_fast_path_candidate"]
    assert ci_row["semantic_winner_id"] == "A"


def test_no_deliveryscore_change():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)], deliveryscore_top_candidate="A")
    ci_row = semantic_authority_ci_row(row)
    assert ci_row["deliveryscore_winner_id"] == row["deliveryscore_top_candidate"]


def test_no_besttake_change():
    row = _judge_group_row("tg1", member_labels=[("A", "winner", 0.95)], deliveryscore_top_candidate="A")
    before_final_winner = row["final_winner"]
    semantic_authority_ci_row(row)
    summarize_semantic_authority_gate_counts([row])
    assert row["final_winner"] == before_final_winner


# ---------------------------------------------------------------------------
# 22-25: module-leaf no-import proofs (D-123/D-128/Boundary/Pacing untouched)
# ---------------------------------------------------------------------------

def test_d123_unchanged_no_import():
    import cutsell_worker.semantic_authority_observability as sao
    assert not hasattr(sao, "build_case_b_performance_evidence")
    assert not hasattr(sao, "case_b_performance_evidence_diagnostics")


def test_d128_unchanged_no_import():
    import cutsell_worker.semantic_authority_observability as sao
    assert not hasattr(sao, "detect_class_b_trigger")
    assert not hasattr(sao, "fallback_trigger_diagnostics")


def test_boundary_unchanged_no_import():
    import cutsell_worker.semantic_authority_observability as sao
    assert not hasattr(sao, "boundary_engine_pass")
    assert not hasattr(sao, "apply_post_freeze_boundary_pass")


def test_pacing_unchanged_no_import():
    import cutsell_worker.semantic_authority_observability as sao
    assert not hasattr(sao, "dialogue_pacing_transition")
    assert not hasattr(sao, "apply_dialogue_pacing_transition_pass")


# ---------------------------------------------------------------------------
# 26: no network/provider call
# ---------------------------------------------------------------------------

def test_no_provider_or_network_call_in_this_module():
    import inspect

    import cutsell_worker.semantic_authority_observability as sao
    source = inspect.getsource(sao)
    forbidden_import_lines = (
        "import requests", "import httpx", "import urllib", "import socket",
        "from google", "import openai", "import anthropic", "GeminiClient(",
        "EditorialJudge(",
    )
    for line in source.splitlines():
        stripped = line.strip()
        for forbidden in forbidden_import_lines:
            assert forbidden not in stripped, f"{forbidden!r} found in module source: {stripped!r}"


def test_pipeline_py_unmodified_d150_gate_call_shape():
    """Sanity: D-152 added no new pipeline.py call site -- _semantic_best_
    take's D-150 gate parameter and semantic_authority_gate_diagnostics's
    signature are exactly what D-150 left them as."""
    import inspect

    from cutsell_worker.pipeline import _semantic_best_take
    from cutsell_worker.semantic_authority_observability import semantic_authority_gate_diagnostics

    sig = inspect.signature(_semantic_best_take)
    assert "semantic_comparative_authority" in sig.parameters
    gate_sig = inspect.signature(semantic_authority_gate_diagnostics)
    assert list(gate_sig.parameters)[:3] == ["member_ids", "semantic_decisions", "family_authority_row"]
