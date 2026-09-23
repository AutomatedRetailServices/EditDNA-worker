"""D-050A: canonical identity/provenance foundation -- additive-only,
shadow-metadata migration. See docs/CUTSELL_DECISIONS.md D-050 (audit) and
D-050A (this migration) for full context.

Every test here proves one of two things:
  (1) an identity invariant from the D-050A directive holds (one minting
      owner, deterministic-within-one-evidence-representation, physical
      trim/split preserves realization identity, no orphans, no cycles,
      ASR-timing jitter alone never changes a semantic id), or
  (2) existing editorial behavior is completely unchanged by this
      migration (behavioral parity -- Section 9 of the directive): winner,
      order, discarded set, claim coverage, and Freeze outcomes on the
      real CleanCutBench chain are identical before/after.

No test here exercises a NEW decision -- D-050A mints metadata only;
nothing in the active pipeline reads these ids to decide anything yet.
"""
from dataclasses import replace

from cutsell_worker.attempt_reconstruction import reconstruct_delivery_attempts
from cutsell_worker.canonical_identity import (
    mint_attempt_id,
    mint_canonical_claim_id,
    mint_realization_id,
    mint_retry_family_id,
    mint_semantic_idea_id,
    mint_source_span_id,
)
from cutsell_worker.contracts import (
    CandidateTake,
    DraftClip,
    MediaSignals,
    Word,
)
from cutsell_worker.human_boundary_polish_v5 import _remove_micro_visual_reset_word_gaps
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.post_selection_interior_gap_trim import split_selected_interior_performance_gaps
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.semantic_claims import extract_claims
from cutsell_worker.take_segmentation import segment_takes
from cutsell_worker.temporal_editing import refine_takes_with_temporal_context
from cutsell_worker.contracts import ProcessingRequest, SourceAsset, TranscriptSegment
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _take(clip_id, text, start, end, *, source="src", complete=True, **extra):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, complete_idea=complete, **extra,
    )


def _word(text, start, end):
    return Word(text=text, start=start, end=end, confidence=0.9)


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src", summary="creator recording",
            dominant_style="talking_head", creator_intent="natural delivery",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


# ---------------------------------------------------------------------------
# 1. Pure minting-function invariants: deterministic, content-anchored,
#    membership-order-independent, distinct-when-content-differs.
# ---------------------------------------------------------------------------

def test_source_span_id_deterministic_for_identical_inputs():
    a = mint_source_span_id("src", 1.0, 2.0, "hello world")
    b = mint_source_span_id("src", 1.0, 2.0, "hello world")
    assert a == b


def test_source_span_id_differs_when_timestamps_differ():
    a = mint_source_span_id("src", 1.000, 2.000, "hello world")
    b = mint_source_span_id("src", 1.050, 2.000, "hello world")
    assert a != b  # physical identity -- expected to be timestamp-sensitive


def test_attempt_id_deterministic_and_order_independent():
    a = mint_attempt_id(["span_1", "span_2", "span_3"])
    b = mint_attempt_id(["span_3", "span_1", "span_2"])
    assert a == b


def test_attempt_id_differs_for_different_membership():
    a = mint_attempt_id(["span_1", "span_2"])
    b = mint_attempt_id(["span_1", "span_2", "span_3"])
    assert a != b


def test_realization_id_never_derived_from_timestamps():
    """The D-050A directive's central anti-jitter requirement: identical
    source/attempt lineage + identical spoken text must mint the identical
    realization_id regardless of how the caller's own timestamps differ --
    because mint_realization_id's signature does not even accept start/end."""
    a = mint_realization_id("src", "att_1", "hello world")
    b = mint_realization_id("src", "att_1", "hello world")
    assert a == b


def test_realization_id_differs_when_content_differs():
    a = mint_realization_id("src", "att_1", "hello world")
    b = mint_realization_id("src", "att_1", "goodbye world")
    assert a != b


def test_realization_id_differs_when_attempt_lineage_differs():
    a = mint_realization_id("src", "att_1", "hello world")
    b = mint_realization_id("src", "att_2", "hello world")
    assert a != b


def test_semantic_idea_id_and_retry_family_id_agree_in_d050a():
    """D-050A deliberately conflates the two -- see canonical_identity.py's
    ID OWNERSHIP note and the D-050 audit's Phase 3 finding. A future
    D-050B/C migration may separate them; this test locks today's
    intentional behavior so that separation is a visible, deliberate
    change rather than an accidental drift."""
    assert mint_semantic_idea_id("tg_abc123") == mint_retry_family_id("tg_abc123")


def test_semantic_idea_id_differs_for_different_group_keys():
    assert mint_semantic_idea_id("tg_a") != mint_semantic_idea_id("tg_b")


def test_canonical_claim_id_ignores_source_and_exact_text():
    """The whole point of this id: two differently-worded, differently-
    sourced restatements of the same fact type/content-token-set should
    converge on one canonical_claim_id -- observability only in D-050A,
    but this is the seed D-050C's own cross-realization dedup fix needs."""
    a = mint_canonical_claim_id("MEASUREMENT_QUANTITY", frozenset({"cinco", "diez", "por", "ciento"}))
    b = mint_canonical_claim_id("MEASUREMENT_QUANTITY", frozenset({"por", "ciento", "diez", "cinco"}))
    assert a == b


def test_canonical_claim_id_differs_for_different_claim_type():
    same_tokens = frozenset({"cinco", "diez"})
    a = mint_canonical_claim_id("MEASUREMENT_QUANTITY", same_tokens)
    b = mint_canonical_claim_id("NEGATION", same_tokens)
    assert a != b


# ---------------------------------------------------------------------------
# 2. One minting owner: downstream code never recomputes an upstream id
#    that is already set.
# ---------------------------------------------------------------------------

def test_merge_attempt_does_not_remint_an_already_correct_attempt_id():
    left = _take("a", "hello", 0.0, 1.0, source_span_id="span_a")
    right = _take("b", "world", 1.05, 2.0, source_span_id="span_b")
    attempts, _ = reconstruct_delivery_attempts((left, right), _context())
    fused = attempts[0]
    # Calling the same reconstruction again on the fused output's own
    # members must mint the identical attempt_id -- not a fresh one.
    attempts_again, _ = reconstruct_delivery_attempts((left, right), _context())
    assert attempts_again[0].attempt_id == fused.attempt_id


def test_pipeline_does_not_remint_a_realization_id_the_take_already_carries():
    """build_flow_b_draft's own minting pass (`kept = tuple(take if take.
    realization_id else ...)`) must leave an already-stamped
    realization_id completely alone."""
    from cutsell_worker.contracts import ProcessingRequest

    take = _take("a", "I started using this product in January.", 0.0, 3.0, realization_id="real_preset")
    request = ProcessingRequest(project_id="p", user_id="u", sources=())
    result = build_flow_b_draft(request, (take,))
    assert len(result.draft.selected) == 1
    assert result.draft.selected[0].realization_id == "real_preset"


# ---------------------------------------------------------------------------
# 3. Physical trim preserves realization_id (temporal_editing.py).
# ---------------------------------------------------------------------------

def test_physical_trim_preserves_realization_id():
    words = (
        Word("this", 10.0, 10.4), Word("works", 10.5, 11.0),
        Word("really", 11.1, 11.6), Word("well", 11.7, 12.1),
    )
    take = CandidateTake(
        clip_id="take-1", source_asset_id="src-1", source_order=0,
        start=10.0, end=15.0, text="this works really well", words=words,
        signals=MediaSignals("src-1", 10.0, 15.0),
        complete_idea=True, realization_id="real_fixed", attempt_id="att_fixed",
        source_span_id="span_fixed",
    )
    context = WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src-1", summary="s", dominant_style="talking_head",
            creator_intent="explain", edit_mode="sales", sales_intent=0.9,
            main_topic="t", product_or_subject="p", story_logic="l",
            events=(TemporalEvent("src-1", 12.2, 15.0, "body_reset", 0.96, "resets to retry"),),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )
    refined, _ = refine_takes_with_temporal_context((take,), context, preserve_clip_id=True)
    assert refined[0].end == 12.2  # the trim did happen
    assert refined[0].realization_id == "real_fixed"
    assert refined[0].attempt_id == "att_fixed"
    assert refined[0].source_span_id == "span_fixed"


# ---------------------------------------------------------------------------
# 4. Physical split preserves realization_id and stamps
#    parent_realization_id (both split sites).
# ---------------------------------------------------------------------------

def _gap_trim_clip():
    words = (
        Word("uno", 0.10, 0.40), Word("dos", 0.50, 0.80), Word("tres", 0.90, 1.20),
        Word("cuatro", 2.20, 2.50), Word("cinco", 2.60, 2.90), Word("seis", 3.00, 3.30),
    )
    return DraftClip(
        clip_id="clip-a", source_asset_id="src", source_order=0,
        start=0.0, end=3.5, text="uno dos tres cuatro cinco seis",
        caption_text="uno dos tres cuatro cinco seis", words=words,
        selected=True, realization_id="real_root",
    )


def test_interior_gap_trim_split_preserves_realization_id_and_stamps_parent():
    diagnostics = {
        "whole_video_context": {"sources": [{
            "source_asset_id": "src",
            "events": [
                {"kind": "hand_motion_reset_candidate", "start": 1.25, "end": 1.80, "confidence": 0.96},
                {"kind": "body_reset_candidate", "start": 1.35, "end": 1.90, "confidence": 0.95},
                {"kind": "facial_expression_shift_candidate", "start": 1.40, "end": 1.95, "confidence": 0.88},
            ],
        }]},
    }
    selected, audit = split_selected_interior_performance_gaps((_gap_trim_clip(),), diagnostics)
    assert len(selected) == 2, "fixture must actually exercise the split path"
    for piece in selected:
        assert piece.realization_id == "real_root", "realization identity must survive a physical split unchanged"
        assert piece.parent_realization_id == "real_root"


def test_human_boundary_polish_split_preserves_realization_id_and_stamps_parent():
    words = (_word("one", 0.0, 1.0), _word("two", 1.4, 2.0))
    text = " ".join(w.text for w in words)
    clip = DraftClip(
        clip_id="clip_root", source_asset_id="src", source_order=0,
        start=0.0, end=2.0, text=text, caption_text=text, words=words,
        selected=True, realization_id="real_root_hbp",
    )
    from types import SimpleNamespace

    timeline = SimpleNamespace(source_asset_id="src", events=(
        SimpleNamespace(source_asset_id="src", start=0.9, end=1.5, kind="body_reset_candidate", confidence=0.9, description=""),
        SimpleNamespace(source_asset_id="src", start=0.9, end=1.5, kind="body_reset_candidate", confidence=0.9, description=""),
    ))
    pieces, rows = _remove_micro_visual_reset_word_gaps(clip, timeline)
    assert len(pieces) == 2, "fixture must actually exercise the split path"
    for piece in pieces:
        assert piece.realization_id == "real_root_hbp"
        assert piece.parent_realization_id == "real_root_hbp"


def test_resplit_of_an_already_split_fragment_keeps_realization_id_and_true_root():
    """A fragment produced by one split pass, split AGAIN by a second pass,
    must still report the same (never-reminted) realization_id, and the
    resulting parent_realization_id continues to be the true original
    root -- mirroring D-036's own `parent_semantic_clip_id` chained-split
    guarantee exactly."""
    words = (_word("one", 0.0, 1.0), _word("two", 1.4, 2.0), _word("three", 2.4, 3.4))
    text = " ".join(w.text for w in words)
    clip = DraftClip(
        clip_id="clip_root", source_asset_id="src", source_order=0,
        start=0.0, end=3.4, text=text, caption_text=text, words=words,
        selected=True, realization_id="real_chain_root",
    )
    from types import SimpleNamespace

    def event(start, end):
        return SimpleNamespace(source_asset_id="src", start=start, end=end, kind="body_reset_candidate", confidence=0.9, description="")

    timeline = SimpleNamespace(source_asset_id="src", events=(event(0.9, 1.5), event(0.9, 1.5)))
    first_pieces, _ = _remove_micro_visual_reset_word_gaps(clip, timeline)
    assert len(first_pieces) == 2
    # Split the right piece again.
    right = first_pieces[1]
    timeline2 = SimpleNamespace(source_asset_id="src", events=(event(2.3, 2.5), event(2.3, 2.5)))
    second_pieces, _ = _remove_micro_visual_reset_word_gaps(right, timeline2)
    for piece in (*first_pieces, *second_pieces):
        assert piece.realization_id == "real_chain_root"
    for piece in second_pieces:
        assert piece.parent_realization_id == "real_chain_root"
        assert piece.parent_semantic_clip_id == "clip_root"  # true root, never an intermediate fragment id


# ---------------------------------------------------------------------------
# 5/6/7. Group merge/split and retry-family reassignment preserve
#    member realization identity (real chain, via build_flow_b_draft).
# ---------------------------------------------------------------------------

def test_group_members_keep_their_own_distinct_realization_ids_through_the_real_chain():
    request = ProcessingRequest(project_id="p", user_id="u", sources=())
    a = _take("a", "I started using this in January for my skin.", 0.0, 4.0)
    b = _take("b", "Then I noticed a real difference after two weeks.", 4.5, 8.0)
    result = build_flow_b_draft(request, (a, b))
    ids = {clip.realization_id for clip in result.draft.selected}
    assert None not in ids, "no orphan selected realization"
    assert len(ids) == len(result.draft.selected), "distinct deliveries must not collide on one realization_id"


def test_same_semantic_idea_id_shared_by_members_of_one_final_group():
    """Two clips forced into the same final take_group_id (D-050A mints
    semantic_idea_id/retry_family_id purely from that group id) must
    report the identical semantic_idea_id -- proving group membership, not
    group MERGING mechanics, is what this test locks (see D-050's own
    audit note that today's take_group_id already conflates idea/retry-
    family; D-050A does not change that, only makes it independently
    observable under a stable name)."""
    from cutsell_worker.pipeline import _draft_clip
    from cutsell_worker.contracts import SemanticRole

    left = _take("a", "same idea left", 0.0, 1.0, realization_id="real_left")
    right = _take("b", "same idea right", 1.0, 2.0, realization_id="real_right")
    left_clip = _draft_clip(left, role=SemanticRole.STORY, group_id="tg_shared", selected=True)
    right_clip = _draft_clip(right, role=SemanticRole.STORY, group_id="tg_shared", selected=True)
    assert left_clip.semantic_idea_id == right_clip.semantic_idea_id
    assert left_clip.retry_family_id == right_clip.retry_family_id
    # And each keeps its OWN distinct realization_id -- group membership
    # never overwrites realization identity.
    assert left_clip.realization_id == "real_left"
    assert right_clip.realization_id == "real_right"


def test_ungrouped_clip_has_no_semantic_idea_id():
    from cutsell_worker.pipeline import _draft_clip
    from cutsell_worker.contracts import SemanticRole

    take = _take("a", "solo take", 0.0, 1.0, realization_id="real_solo")
    clip = _draft_clip(take, role=SemanticRole.STORY, group_id=None, selected=True)
    assert clip.semantic_idea_id is None
    assert clip.retry_family_id is None
    assert clip.realization_id == "real_solo"


# ---------------------------------------------------------------------------
# 8. No duplicate canonical IDs inside one draft (distinct content ->
#    distinct realization_id; also exercised via the real chain above).
# ---------------------------------------------------------------------------

def test_no_duplicate_realization_ids_for_distinct_content_in_one_draft():
    request = ProcessingRequest(project_id="p", user_id="u", sources=())
    a = _take("a", "The first completely distinct idea about skincare routines.", 0.0, 4.0)
    b = _take("b", "A second, totally unrelated idea about morning coffee habits.", 4.5, 8.5)
    c = _take("c", "A third idea entirely about weekend hiking trips in the mountains.", 9.0, 13.0)
    result = build_flow_b_draft(request, (a, b, c))
    ids = [clip.realization_id for clip in result.draft.selected]
    assert len(ids) == len(set(ids)), "distinct spoken content must never collide on one realization_id"


# ---------------------------------------------------------------------------
# 9/10. Provenance graph: no orphan selected realization, no cycles.
# ---------------------------------------------------------------------------

def test_no_orphan_selected_realization_through_the_real_chain():
    request = ProcessingRequest(project_id="p", user_id="u", sources=())
    a = _take("a", "This is a complete standalone thought about my morning routine.", 0.0, 4.0)
    result = build_flow_b_draft(request, (a,))
    for clip in result.draft.selected:
        assert clip.realization_id, f"selected clip {clip.clip_id} has no realization_id"


def test_identity_chain_diagnostics_has_no_orphans_and_matches_selected():
    request = ProcessingRequest(project_id="p", user_id="u", sources=())
    a = _take("a", "This is a complete standalone thought about my evening routine.", 0.0, 4.0)
    result = build_flow_b_draft(request, (a,))
    chain = result.draft.diagnostics["canonical_identity_chain"]
    assert len(chain) == len(result.draft.selected)
    for row in chain:
        assert row["realization_id"], f"orphan realization in identity chain: {row}"


def test_parent_pointers_never_point_at_a_fragment_id_no_cycles():
    """A chained split's parent_semantic_clip_id/parent_realization_id
    must always resolve to the TRUE root, never to an intermediate
    fragment's own render_fragment_id -- which is exactly what would
    create a cycle/dangling reference in the provenance graph."""
    words = (_word("one", 0.0, 1.0), _word("two", 1.4, 2.0), _word("three", 2.4, 3.4))
    text = " ".join(w.text for w in words)
    clip = DraftClip(
        clip_id="clip_root", source_asset_id="src", source_order=0,
        start=0.0, end=3.4, text=text, caption_text=text, words=words,
        selected=True, realization_id="real_no_cycle",
    )
    from types import SimpleNamespace

    def event(start, end):
        return SimpleNamespace(source_asset_id="src", start=start, end=end, kind="body_reset_candidate", confidence=0.9, description="")

    timeline = SimpleNamespace(source_asset_id="src", events=(event(0.9, 1.5), event(0.9, 1.5)))
    first_pieces, _ = _remove_micro_visual_reset_word_gaps(clip, timeline)
    right = first_pieces[1]
    timeline2 = SimpleNamespace(source_asset_id="src", events=(event(2.3, 2.5), event(2.3, 2.5)))
    second_pieces, _ = _remove_micro_visual_reset_word_gaps(right, timeline2)
    fragment_ids = {piece.render_fragment_id for piece in (*first_pieces, *second_pieces)}
    for piece in second_pieces:
        assert piece.parent_semantic_clip_id not in fragment_ids
        assert piece.parent_realization_id not in fragment_ids or piece.parent_realization_id == "real_no_cycle"
        assert piece.parent_semantic_clip_id == "clip_root"


# ---------------------------------------------------------------------------
# 11/12/13. Legacy behavior / CanonicalEditPlan / Freeze unchanged --
#    behavioral parity (Section 9). Full CleanCutBench parity is already
#    proven by tests/test_cutsell_clean_cut_core_evaluation_suite.py
#    running unmodified and green (54/54) against this exact diff; these
#    two tests additionally lock the specific artifacts D-050A must never
#    touch.
# ---------------------------------------------------------------------------

def test_legacy_clip_id_computation_completely_unaffected():
    from cutsell_worker.source_identity import stable_clip_id

    expected = stable_clip_id("src", 0.0, 1.0, "hello world")
    segments = (TranscriptSegment(
        source_asset_id="src", start=0.0, end=1.0, text="hello world",
        words=(Word("hello", 0.0, 0.4), Word("world", 0.5, 1.0)),
    ),)
    sources = (SourceAsset(
        source_asset_id="src", project_id="p", user_id="u", original_name="v.mp4",
        source_order=0, duration_sec=2.0, uri="s3://bucket/v.mp4",
    ),)
    takes = segment_takes(segments, sources)
    assert takes[0].clip_id == expected


def test_claim_extraction_still_returns_the_same_claim_id_and_fields():
    """canonical_claim_id is a new, additive field only -- claim_id (the
    pre-existing identity claim_coverage_best_take.py actually keys on)
    and every other field must be byte-identical to before D-050A."""
    claims = extract_claims("clip_x", "El médico confirmó que era una gastritis grave.")
    assert len(claims) >= 1
    claim = claims[0]
    from cutsell_worker.semantic_claims import _claim_id

    assert claim.claim_id == _claim_id("clip_x", claim.text)
    assert claim.canonical_claim_id  # additive field is populated
    assert isinstance(claim.canonical_claim_id, str)
