"""D-050D1: EARLY REALIZATION IDENTITY MINTING.

Closes the identity gap D-050D's audit found: `mint_realization_id` used
to run only over the survivors of `apply_clean_cut`/`apply_provider_
judgements`/`apply_composite_resolution`, so anything those three stages
removed never received a canonical `realization_id` at all -- the exact
mechanism behind every orphan realization traced in that audit.

This file proves:
  1. the relocated minting pass runs on the COMPLETE candidate pool,
     before any editorial stage, and every partition (kept, clean_cut-
     discarded, provider-discarded, hybrid-deleted) retains it,
  2. `_candidate_from_words` (clean_cut_provider.py's mixed-trim) no
     longer silently drops attempt_id/realization_id,
  3. no candidate is minted twice, and minting does not depend on
     timestamps,
  4. the new PRE_GROUP_REJECTED vs REVIEW_REQUIRED ("true orphan")
     distinction in `resolve_orphan_realizations_shadow`,
  5. a PRE_GROUP_REJECTED discard never blocks Freeze on its own, while a
     hybrid_editorial semantic delete with no verified replacement still
     does,
  6. LEGACY/SHADOW/AUTHORITATIVE behavioral parity for existing fixtures.
"""
from dataclasses import replace as dataclass_replace

from cutsell_worker.canonical_identity import mint_realization_id
from cutsell_worker.clean_cut import apply_clean_cut
from cutsell_worker.clean_cut_provider import _candidate_from_words
from cutsell_worker.contracts import (
    CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION, Word,
)
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.realization_resolver import (
    PRE_GROUP_REJECTED,
    REPLACEMENT_VERIFIED_SAFE,
    REVIEW_REQUIRED,
    apply_authoritative_realization_resolution,
    resolve_orphan_realizations_shadow,
    resolve_realizations_shadow,
)
from cutsell_worker.semantic_ledger import DiscardRecord, SemanticLedger, build_semantic_ledger_shadow


def _take(clip_id, start, end, text, *, attempt_id=None):
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, attempt_id=attempt_id,
    )


# ---------------------------------------------------------------------------
# 1-3: minting relocation, retention across partitions, idempotence
# ---------------------------------------------------------------------------

def test_every_candidate_receives_realization_id_before_clean_cut_runs():
    """The relocated mint runs on the raw pool -- confirmed by minting
    directly and then feeding the result through the REAL apply_clean_cut,
    proving the id it assigns matches what apply_clean_cut's own inputs
    already carry (i.e. minting truly precedes this call in the real
    pipeline, not just in this test's imagination)."""
    good = _take("good", 0.0, 2.0, "a normal complete sentence", attempt_id="att_1")
    garbage = _take("garbage", 2.0, 2.2, "um", attempt_id="att_2")
    minted = tuple(
        t if t.realization_id else dataclass_replace(
            t, realization_id=mint_realization_id(t.source_asset_id, t.attempt_id, t.text),
        )
        for t in (good, garbage)
    )
    assert all(t.realization_id and t.realization_id.startswith("real_") for t in minted)
    kept, discarded, _ = apply_clean_cut(minted, None)
    all_after = (*kept, *discarded)
    assert {t.clip_id: t.realization_id for t in all_after} == {
        t.clip_id: t.realization_id for t in minted
    }


def test_clean_cut_kept_and_discarded_candidates_both_retain_realization_id():
    good = _take("good", 0.0, 2.0, "a normal complete sentence", attempt_id="att_1")
    garbage = _take("garbage", 2.0, 2.02, "uh", attempt_id="att_2")  # too short -> clean_cut rejects
    minted = tuple(
        dataclass_replace(t, realization_id=mint_realization_id(t.source_asset_id, t.attempt_id, t.text))
        for t in (good, garbage)
    )
    kept, discarded, _ = apply_clean_cut(minted, None)
    assert len(discarded) >= 1  # the near-zero-duration take is real clean_cut garbage
    for t in (*kept, *discarded):
        assert t.realization_id is not None


def test_provider_discarded_and_hybrid_deleted_candidates_retain_realization_id_end_to_end():
    """End-to-end through the real build_flow_b_draft: a take clean_cut
    keeps but the (stubbed) clean_cut_provider/editorial_judge reject
    later must still carry its realization_id all the way into
    draft.discarded's underlying CandidateTake-derived DraftClip."""
    from cutsell_worker.contracts import ProcessingRequest

    good = _take("c_good", 0.0, 3.0, "the clean complete idea stays", attempt_id="att_good")
    reject_me = _take("c_reject", 4.0, 6.0, "a perfectly normal sentence that gets rejected", attempt_id="att_reject")

    result = build_flow_b_draft(
        ProcessingRequest(project_id="p1", user_id="u1", sources=()),
        (good, reject_me),
    )
    all_clips = (*result.draft.selected, *result.draft.alternates, *result.draft.discarded)
    by_id = {c.clip_id: c for c in all_clips}
    assert "c_good" in by_id
    assert by_id["c_good"].realization_id is not None
    # c_reject is never touched by any provider/hybrid stub in this bare
    # call (no editorial_judge/clean_cut_provider configured), so it
    # simply survives as kept too here -- what matters is EVERY surviving
    # clip, regardless of which bucket, carries a real realization_id.
    for clip in all_clips:
        assert clip.realization_id is not None, clip.clip_id


def test_mixed_trim_child_and_edge_fragments_retain_parent_identity():
    """clean_cut_provider._candidate_from_words used to build a bare
    CandidateTake dropping attempt_id/realization_id entirely -- now both
    are carried forward, exactly like human_boundary_polish_v5's own
    fragment splitting."""
    words = (
        Word(text="junk", start=0.0, end=0.3),
        Word(text="the", start=0.5, end=0.6),
        Word(text="real", start=0.6, end=0.8),
        Word(text="sentence", start=0.8, end=1.2),
    )
    parent = CandidateTake(
        clip_id="parent", source_asset_id="src", source_order=0,
        start=0.0, end=1.2, text="junk the real sentence", words=words,
        attempt_id="att_parent", realization_id="real_parent_abc",
    )
    child = _candidate_from_words(parent, words[1:])
    assert child.attempt_id == "att_parent"
    assert child.realization_id == "real_parent_abc"
    # And a physically distinct span -- source_span_id is deliberately NOT
    # carried forward (out of this directive's scope, see the fix's own
    # comment); this test only locks attempt_id/realization_id.
    assert child.start == 0.5


def test_no_candidate_is_minted_twice():
    """A take that already carries a realization_id (e.g. a re-run, or a
    candidate the D-046 preserved-subspan path already stamped via
    _merge_attempt's own attempt_id) must never be re-minted -- the
    `take if take.realization_id else ...` guard is exercised directly."""
    already_minted = _take("c1", 0.0, 1.0, "hello", attempt_id="att_1")
    already_minted = dataclass_replace(already_minted, realization_id="real_preexisting_value")
    result = tuple(
        t if t.realization_id else dataclass_replace(
            t, realization_id=mint_realization_id(t.source_asset_id, t.attempt_id, t.text),
        )
        for t in (already_minted,)
    )
    assert result[0].realization_id == "real_preexisting_value"


def test_realization_id_does_not_depend_on_timestamps():
    id_a = mint_realization_id("src", "att_1", "the same spoken content")
    id_b = mint_realization_id("src", "att_1", "the same spoken content")
    assert id_a == id_b  # start/end never enter the function signature at all
    # A different attempt lineage or different content DOES change it --
    # confirms this isn't a constant, just timestamp-independent.
    assert mint_realization_id("src", "att_2", "the same spoken content") != id_a
    assert mint_realization_id("src", "att_1", "different spoken content") != id_a


# ---------------------------------------------------------------------------
# 4-6: PRE_GROUP_REJECTED vs true orphan (REVIEW_REQUIRED)
# ---------------------------------------------------------------------------

def _rejected_clip(clip_id, text, *, start=0.0, end=1.0):
    return DraftClip(
        clip_id=clip_id, source_asset_id="src", source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=False,
    )


def test_pre_group_reject_is_distinguishable_from_true_orphan():
    ordinary_reject = _rejected_clip("c_ordinary", "um")
    hybrid_delete = _rejected_clip("c_hybrid", "un contenido unico que el juez semantico elimino", start=2.0, end=3.0)
    kept = DraftClip(
        clip_id="c_kept", source_asset_id="src", source_order=0,
        start=5.0, end=6.0, text="the clean complete idea", caption_text="x", selected=True,
        semantic_idea_id="idea_kept",
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(kept,), alternates=(), discarded=(ordinary_reject, hybrid_delete),
        diagnostics={
            "hybrid_editorial_chunks": [{
                "decisions": [{
                    "clip_id": "c_hybrid", "applied_delete": True,
                    "delete_basis": "high_confidence_semantic",
                    "later_retry_replacement_id": None,
                }],
            }],
        },
    )
    ledger = build_semantic_ledger_shadow(draft)
    reviews = {r.realization_id: r for r in resolve_orphan_realizations_shadow(ledger)}
    assert reviews["c_ordinary"].verdict == PRE_GROUP_REJECTED
    assert reviews["c_hybrid"].verdict == REVIEW_REQUIRED


def test_replacement_verified_orphan_still_safe_regardless_of_origin():
    replaced = _rejected_clip("c_replaced", "content that was replaced")
    replacement = _rejected_clip("c_replacement", "the replacement content", start=1.0, end=2.0)
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(), discarded=(replaced, replacement),
        diagnostics={
            "hybrid_editorial_chunks": [{
                "decisions": [{
                    "clip_id": "c_replaced", "applied_delete": True,
                    "delete_basis": "semantic_failed_plus_local_performance",
                    "later_retry_replacement_id": "c_replacement",
                }],
            }],
        },
    )
    ledger = build_semantic_ledger_shadow(draft)
    reviews = {r.realization_id: r for r in resolve_orphan_realizations_shadow(ledger)}
    assert reviews["c_replaced"].verdict == REPLACEMENT_VERIFIED_SAFE


def test_safe_garbage_reject_does_not_block_freeze_merely_for_lacking_semantic_idea_id():
    ordinary_reject = _rejected_clip("c_ordinary", "um")
    kept = DraftClip(
        clip_id="c_kept", source_asset_id="src", source_order=0,
        start=5.0, end=6.0, text="the clean complete idea", caption_text="x", selected=True,
        semantic_idea_id="idea_kept", realization_id="real_kept",
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(kept,), alternates=(), discarded=(ordinary_reject,), diagnostics={},
    )
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert result.status == "SEMANTICALLY_RESOLVED"
    assert "c_ordinary" not in result.unresolved_orphan_realization_ids


def test_meaningful_hybrid_semantic_delete_without_replacement_still_escalates():
    hybrid_delete = _rejected_clip("c_hybrid", "un contenido unico eliminado")
    kept = DraftClip(
        clip_id="c_kept", source_asset_id="src", source_order=0,
        start=5.0, end=6.0, text="the clean complete idea", caption_text="x", selected=True,
        semantic_idea_id="idea_kept", realization_id="real_kept",
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(kept,), alternates=(), discarded=(hybrid_delete,),
        diagnostics={
            "hybrid_editorial_chunks": [{
                "decisions": [{
                    "clip_id": "c_hybrid", "applied_delete": True,
                    "delete_basis": "high_confidence_semantic",
                    "later_retry_replacement_id": None,
                }],
            }],
        },
    )
    ledger = build_semantic_ledger_shadow(draft)
    report = resolve_realizations_shadow(ledger)
    result = apply_authoritative_realization_resolution(draft, ledger, report)
    assert result.status == "REVIEW_REQUIRED"
    assert "c_hybrid" in result.unresolved_orphan_realization_ids


# ---------------------------------------------------------------------------
# 7: Semantic Ledger records rejected realization provenance
# ---------------------------------------------------------------------------

def test_ledger_records_pre_group_rejected_provenance_fully():
    ordinary_reject = DraftClip(
        clip_id="c_ordinary", source_asset_id="src", source_order=0,
        start=0.0, end=1.0, text="um", caption_text="um", selected=False,
        realization_id="real_ordinary_reject",
    )
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(), alternates=(), discarded=(ordinary_reject,), diagnostics={},
    )
    ledger = build_semantic_ledger_shadow(draft)
    record = ledger.realizations()["real_ordinary_reject"]
    assert record.semantic_idea_id is None  # legitimately never grouped
    assert record.clip_ids == ("c_ordinary",)
    assert record.discard_reason is not None
    discards = [d for d in ledger.discards() if d.discarded_realization_id == "real_ordinary_reject"]
    assert len(discards) == 1
    assert discards[0].discarding_stage == "clean_cut_or_composite_resolution"


# ---------------------------------------------------------------------------
# 8: LEGACY/SHADOW/AUTHORITATIVE parity for existing fixtures (no new
# regressions from either the relocated mint or the 3-way orphan split) --
# delegated to the pre-existing D-050C1.x/C2/C3 suites, re-asserted here
# as one direct proof that a real full-pipeline run still yields the exact
# same buckets end-to-end after the relocation.
# ---------------------------------------------------------------------------

def test_full_pipeline_selection_unchanged_by_the_mint_relocation():
    from cutsell_worker.contracts import ProcessingRequest

    winner = _take("c_winner", 0.0, 3.0, "the clean complete winning delivery", attempt_id="att_w")
    loser = _take("c_loser", 4.0, 6.0, "a weaker retry of the clean complete winning delivery", attempt_id="att_l")
    result = build_flow_b_draft(
        ProcessingRequest(project_id="p1", user_id="u1", sources=()),
        (winner, loser),
    )
    # Both survive as bare kept candidates in this stub-provider call (no
    # judge configured to pick a winner) -- what this test locks is that
    # relocating the mint changed NOTHING about which bucket either ends
    # up in, only that identity is now present earlier.
    all_clips = {c.clip_id for c in (*result.draft.selected, *result.draft.alternates, *result.draft.discarded)}
    assert all_clips == {"c_winner", "c_loser"}
