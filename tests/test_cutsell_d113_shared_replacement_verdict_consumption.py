"""D-113 -- shared retry-replacement verdict consumption, wired into the two
D-112 PROVEN-COLLISION hooks (`hybrid_retry_completion_integrity` and
`hybrid_cross_group_retry_integrity`).

Both hooks independently re-derive "does this winner/peer cover/replace the
candidate" via their own evidence, never consulting
`complete_retry_identity_guard`'s already-computed, stricter directional
verdict for the SAME (candidate, proposed replacement) pair. D-112's
forensic sweep proved this collision shape active on real Video00 media at
both hooks in the SAME run (pimples A->C at hook 2, an unrelated ob-gyn pair
at hook 5). This suite proves the shared `complete_retry_identity_guard.
is_rejected_replacement` consumer, wired into both hooks, closes exactly
that gap without widening any threshold or semantics.

Fixtures are generic (no Video00 clip ids, text, or timestamps) -- shaped
after the two real collisions, not copied from them.
"""
from cutsell_worker.complete_retry_identity_guard import (
    LEXICAL_REPLACEMENT_VERIFIED,
    NOT_APPLICABLE,
    SEQUENCE_IDENTITY_BELOW_THRESHOLD,
)
from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.hybrid_cross_group_retry_integrity import (
    collapse_cross_group_semantic_retries,
)
from cutsell_worker.hybrid_retry_completion_integrity import (
    apply_hybrid_retry_completion_integrity,
)
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult

# --- pimples-shaped fixture (Hook #2, `_safe_failed_retry`) ----------------
A_TEXT = "I also noticed some spots, it felt like a rash, an allergy."
C_TEXT = (
    "I also noticed some spots on this part, it felt like a rash, but it "
    "turned out to be a hormonal issue."
)

# --- ob-gyn-shaped fixture (Hook #5, `_covered_by_authoritative_peers`) ----
X_TEXT = "After my contract ended, I asked my doctor."
Y_TEXT = "After my contract ended, I switched doctors and asked her to run a test for me."


def _take(clip_id, start, end, text, *, source="src", complete=True, visual_fumble=0.0):
    signals = (
        MediaSignals(source_asset_id=source, start=start, end=end, visual_fumble=visual_fumble)
        if visual_fumble
        else None
    )
    return CandidateTake(clip_id, source, 0, start, end, text, signals=signals, complete_idea=complete)


def _session_diagnostics(clip_id, replacement_candidate_id, reason):
    return (
        {
            "decisions": [
                {
                    "clip_id": clip_id,
                    "replacement_candidate_clip_id_before_guard": replacement_candidate_id,
                    "replacement_rejection_reason": reason,
                }
            ]
        },
    )


def _cleanup_result(kept, diagnostics, semantic_decisions, deleted=()):
    return HybridSessionCleanupResult(
        kept=kept,
        deleted=deleted,
        requested_chunk_count=len(kept),
        available_chunk_count=len(kept),
        diagnostics=diagnostics,
        semantic_decisions=semantic_decisions,
    )


def _hook2_entries(out):
    return next(
        row["hybrid_retry_completion_integrity"]
        for row in out.diagnostics
        if isinstance(row, dict) and "hybrid_retry_completion_integrity" in row
    )


def _hook5(kept, semantic, session_diagnostics=()):
    return collapse_cross_group_semantic_retries(
        kept, semantic, session_diagnostics=session_diagnostics,
    )


# =============================================================================
# 1. Hook #2 respects an exact X->Y rejection (pimples-structural positive control)
# =============================================================================

def test_hook2_respects_exact_rejection_pimples_shape():
    a = _take("a", 10.0, 15.0, A_TEXT, visual_fumble=0.85)
    c = _take("c", 16.0, 26.0, C_TEXT)
    semantic = (("a", "failed", 0.8), ("c", "winner", 0.92))
    diagnostics = _session_diagnostics("a", "c", SEQUENCE_IDENTITY_BELOW_THRESHOLD)
    result = _cleanup_result((a, c), diagnostics, semantic)

    out = apply_hybrid_retry_completion_integrity(result, (a, c), context=None)

    kept_ids = {t.clip_id for t in out.kept}
    assert kept_ids == {"a", "c"}
    row = next(r for r in _hook2_entries(out) if r["clip_id"] == "a")
    assert row["reason"] == "prior_replacement_rejection_respected"
    assert row["removal_applied"] is False
    assert row["prior_replacement_rejection_found"] is True
    assert row["prior_replacement_rejection_reason"] == SEQUENCE_IDENTITY_BELOW_THRESHOLD
    assert row["proposed_winner_clip_id"] == "c"


# =============================================================================
# 2. Hook #5 respects an exact X->Y rejection (ob-gyn-structural positive control)
# =============================================================================

def test_hook5_respects_exact_rejection_obgyn_shape():
    x = _take("x", 10.0, 14.0, X_TEXT)
    y = _take("y", 15.0, 25.0, Y_TEXT)
    semantic = (("x", "failed", 0.9), ("y", "winner", 0.95))
    session_diagnostics = _session_diagnostics("x", "y", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    survivors, removed, diagnostics = _hook5((x, y), semantic, session_diagnostics)

    survivor_ids = {t.clip_id for t in survivors}
    assert survivor_ids == {"x", "y"}
    assert removed == ()
    row = next(r for r in diagnostics if r["clip_id"] == "x")
    assert row["reason"] == "prior_replacement_rejection_respected"
    assert row["removal_applied"] is False
    assert row["prior_replacement_rejection_found"] is True
    assert row["prior_replacement_rejection_reason"] == SEQUENCE_IDENTITY_BELOW_THRESHOLD
    assert row["proposed_winner_clip_id"] == "y"


# =============================================================================
# 3. Directional negative control: X->Y rejected does not block X->Z
# =============================================================================

def test_hook2_rejection_for_different_proposed_winner_does_not_block_real_one():
    a = _take("a", 10.0, 15.0, A_TEXT, visual_fumble=0.85)
    c = _take("c", 16.0, 26.0, C_TEXT)
    semantic = (("a", "failed", 0.8), ("c", "winner", 0.92))
    # Rejection recorded for (a, some_other_clip) -- never proposed as a's
    # actual winner this run -- must not block a's real removal in favor of c.
    diagnostics = _session_diagnostics("a", "some_other_clip_id", SEQUENCE_IDENTITY_BELOW_THRESHOLD)
    result = _cleanup_result((a, c), diagnostics, semantic)

    out = apply_hybrid_retry_completion_integrity(result, (a, c), context=None)

    kept_ids = {t.clip_id for t in out.kept}
    assert kept_ids == {"c"}
    row = next(r for r in _hook2_entries(out) if r["clip_id"] == "a")
    assert row["removal_applied"] is True


def test_hook5_rejection_for_x_to_y_does_not_block_unrelated_pair_x_to_z():
    x = _take("x", 10.0, 14.0, X_TEXT)
    y = _take("y", 15.0, 25.0, Y_TEXT)
    # A second, unrelated failed/winner pair on the same source -- x's own
    # rejection (x, y) must not affect this pair at all.
    p = _take("p", 40.0, 44.0, X_TEXT.replace("doctor", "nurse"))
    q = _take("q", 45.0, 55.0, Y_TEXT.replace("doctors", "nurses").replace("doctor", "nurse"))
    semantic = (
        ("x", "failed", 0.9), ("y", "winner", 0.95),
        ("p", "failed", 0.9), ("q", "winner", 0.95),
    )
    session_diagnostics = _session_diagnostics("x", "y", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    survivors, removed, diagnostics = _hook5((x, y, p, q), semantic, session_diagnostics)

    survivor_ids = {t.clip_id for t in survivors}
    assert "x" in survivor_ids and "y" in survivor_ids  # x's rejection respected
    assert "p" not in survivor_ids  # p legitimately superseded by q, unaffected
    assert "q" in survivor_ids
    by_clip = {row["clip_id"]: row for row in diagnostics}
    assert by_clip["x"]["removal_applied"] is False
    assert by_clip["p"]["removal_applied"] is True


# =============================================================================
# 4. Reverse-direction control: X->Y rejected does not imply Y->X rejected
# =============================================================================

def test_hook2_reverse_direction_not_implied():
    # Roles swapped from the positive control: here `c` (now labelled
    # failed) is the candidate being considered for removal in favor of `a`
    # (now labelled winner) -- the (c, a) direction. A rejection was
    # recorded only for the ORIGINAL (a, c) direction and must not apply.
    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT, visual_fumble=0.85)
    semantic = (("a", "winner", 0.92), ("c", "failed", 0.9))
    diagnostics = _session_diagnostics("a", "c", SEQUENCE_IDENTITY_BELOW_THRESHOLD)
    result = _cleanup_result((a, c), diagnostics, semantic)

    out = apply_hybrid_retry_completion_integrity(result, (a, c), context=None)

    kept_ids = {t.clip_id for t in out.kept}
    # c is legitimately removed in favor of a: the recorded rejection names
    # (a, c), not (c, a), so it does not apply here.
    assert kept_ids == {"a"}


def test_hook5_reverse_direction_not_implied():
    # Near-identical text on both sides so coverage clears the threshold in
    # EITHER direction -- isolating the test to directionality of the
    # rejection lookup itself, not to this hook's own (asymmetric) coverage
    # math. Swap roles from the positive control: here `y` (now "failed")
    # is considered for removal in favor of `x` (now "winner"). A rejection
    # was recorded only for the ORIGINAL (x, y) direction and must not
    # apply to this (y, x) direction.
    t1 = "After my contract ended, I asked my doctor for a full checkup and test."
    t2 = "After my contract ended, I asked my doctor for a full checkup and blood test."
    x = _take("x", 10.0, 20.0, t1)
    y = _take("y", 21.0, 31.0, t2)
    semantic_reversed = (("y", "failed", 0.9), ("x", "winner", 0.95))
    session_diagnostics = _session_diagnostics("x", "y", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    survivors, removed, diagnostics = _hook5((x, y), semantic_reversed, session_diagnostics)

    survivor_ids = {t.clip_id for t in survivors}
    # y is legitimately removed in favor of x here -- the recorded rejection
    # names (x, y), not (y, x).
    assert survivor_ids == {"x"}


# =============================================================================
# 5. UNKNOWN/no-verdict controls: neither ACCEPTED nor UNKNOWN reasons behave
#    like a rejection
# =============================================================================

def test_hook2_no_session_diagnostics_unchanged():
    a = _take("a", 10.0, 15.0, A_TEXT, visual_fumble=0.85)
    c = _take("c", 16.0, 26.0, C_TEXT)
    semantic = (("a", "failed", 0.8), ("c", "winner", 0.92))
    result = _cleanup_result((a, c), (), semantic)  # no guard evidence at all

    out = apply_hybrid_retry_completion_integrity(result, (a, c), context=None)

    assert {t.clip_id for t in out.kept} == {"c"}


def test_hook2_accepted_note_does_not_behave_like_a_rejection():
    a = _take("a", 10.0, 15.0, A_TEXT, visual_fumble=0.85)
    c = _take("c", 16.0, 26.0, C_TEXT)
    semantic = (("a", "failed", 0.8), ("c", "winner", 0.92))
    diagnostics = _session_diagnostics("a", "c", LEXICAL_REPLACEMENT_VERIFIED)
    result = _cleanup_result((a, c), diagnostics, semantic)

    out = apply_hybrid_retry_completion_integrity(result, (a, c), context=None)

    assert {t.clip_id for t in out.kept} == {"c"}


def test_hook2_not_applicable_note_does_not_behave_like_a_rejection():
    a = _take("a", 10.0, 15.0, A_TEXT, visual_fumble=0.85)
    c = _take("c", 16.0, 26.0, C_TEXT)
    semantic = (("a", "failed", 0.8), ("c", "winner", 0.92))
    diagnostics = _session_diagnostics("a", "c", NOT_APPLICABLE)
    result = _cleanup_result((a, c), diagnostics, semantic)

    out = apply_hybrid_retry_completion_integrity(result, (a, c), context=None)

    assert {t.clip_id for t in out.kept} == {"c"}


def test_hook5_no_session_diagnostics_unchanged():
    x = _take("x", 10.0, 14.0, X_TEXT)
    y = _take("y", 15.0, 25.0, Y_TEXT)
    semantic = (("x", "failed", 0.9), ("y", "winner", 0.95))

    survivors, removed, diagnostics = _hook5((x, y), semantic, session_diagnostics=())

    assert {t.clip_id for t in survivors} == {"y"}
    assert {t.clip_id for t in removed} == {"x"}


def test_hook5_accepted_note_does_not_behave_like_a_rejection():
    x = _take("x", 10.0, 14.0, X_TEXT)
    y = _take("y", 15.0, 25.0, Y_TEXT)
    semantic = (("x", "failed", 0.9), ("y", "winner", 0.95))
    session_diagnostics = _session_diagnostics("x", "y", LEXICAL_REPLACEMENT_VERIFIED)

    survivors, removed, diagnostics = _hook5((x, y), semantic, session_diagnostics)

    assert {t.clip_id for t in survivors} == {"y"}
    assert {t.clip_id for t in removed} == {"x"}


# =============================================================================
# 6. Legitimate retry cleanup positive control (no rejection exists -- both
#    hooks must keep working exactly as before D-113)
# =============================================================================

def test_hook2_legitimate_cleanup_still_works():
    a = _take("a", 10.0, 15.0, A_TEXT, visual_fumble=0.85)
    c = _take("c", 16.0, 26.0, C_TEXT)
    semantic = (("a", "failed", 0.8), ("c", "winner", 0.92))
    result = _cleanup_result((a, c), (), semantic)

    out = apply_hybrid_retry_completion_integrity(result, (a, c), context=None)

    assert {t.clip_id for t in out.kept} == {"c"}
    row = next(r for r in _hook2_entries(out) if r["clip_id"] == "a")
    assert row["reason"] == "semantic_failed_cross_group_retry_covered"
    assert row["removal_applied"] is True
    assert row["prior_replacement_rejection_found"] is False


def test_hook5_legitimate_cleanup_still_works():
    x = _take("x", 10.0, 14.0, X_TEXT)
    y = _take("y", 15.0, 25.0, Y_TEXT)
    semantic = (("x", "failed", 0.9), ("y", "winner", 0.95))

    survivors, removed, diagnostics = _hook5((x, y), semantic)

    assert {t.clip_id for t in survivors} == {"y"}
    row = diagnostics[0]
    assert row["reason"] == "cross_group_semantic_retry_covered_by_authoritative_delivery"
    assert row["removal_applied"] is True
    assert row["prior_replacement_rejection_found"] is False


# =============================================================================
# 7. B/C complementary-shaped regression: two candidates with no covering
#    relationship (insufficient coverage) are left alone by both hooks --
#    unrelated to hybrid_semantic_complementary_rescue.py (untouched by
#    D-113), this proves neither hook manufactures a removal where none of
#    its own evidence exists, regardless of guard evidence.
# =============================================================================

B_TEXT = "Another symptom I had was spots behind my ear and on my neck. It came in seasons."
D_TEXT = "I also noticed some spots, it felt like a rash, an allergy."


def test_hook2_complementary_pair_untouched_when_no_covering_evidence():
    b = _take("b", 10.0, 20.0, B_TEXT, visual_fumble=0.85)
    d = _take("d", 21.0, 26.0, D_TEXT)
    semantic = (("b", "alternate", 0.8), ("d", "winner", 0.95))
    # No session diagnostics at all -- this test is about _safe_failed_retry
    # never firing for an "alternate"-labelled candidate in the first place,
    # independent of any guard evidence.
    result = _cleanup_result((b, d), (), semantic)

    out = apply_hybrid_retry_completion_integrity(result, (b, d), context=None)

    assert {t.clip_id for t in out.kept} == {"b", "d"}


def test_hook5_complementary_pair_untouched_when_coverage_insufficient():
    b = _take("b", 10.0, 20.0, B_TEXT)
    d = _take("d", 21.0, 26.0, D_TEXT)
    semantic = (("b", "alternate", 0.8), ("d", "winner", 0.95))

    survivors, removed, diagnostics = _hook5((b, d), semantic)

    # b's content is long and only weakly overlaps d's short text -- no
    # single-peer coverage match, so b survives untouched regardless of any
    # guard evidence (none is supplied here either).
    assert {t.clip_id for t in survivors} == {"b", "d"}
    assert removed == ()


# =============================================================================
# 8. D-110 (Hook #8) regression -- covered by the existing dedicated suite;
#    re-run here as part of this task's own targeted set for one-command
#    confirmation it is untouched by the D-113 shared-helper refactor.
# =============================================================================

def test_d110_hook8_shared_helper_refactor_smoke():
    from cutsell_worker.hybrid_retry_winner_authority import enforce_proven_retry_winners
    from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext
    from cutsell_worker.providers import ProviderStatus

    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT)
    context = WholeVideoContext(
        sources=(
            SourceVideoContext(
                source_asset_id="src",
                summary="raw talking head with retries",
                dominant_style="talking_head",
                creator_intent="tell personal story naturally",
                events=(TemporalEvent("src", a.end + 0.1, a.end + 0.6, "retry_setup", 0.86, "creator resets"),),
            ),
        ),
        status=ProviderStatus("test", True, True, "applied"),
    )
    session_diagnostics = _session_diagnostics("a", "c", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        session_diagnostics=session_diagnostics,
    )

    assert kept == (a, c)
    assert removed == ()
    assert diagnostics[0]["final_reason"] == "prior_replacement_rejection_respected"
