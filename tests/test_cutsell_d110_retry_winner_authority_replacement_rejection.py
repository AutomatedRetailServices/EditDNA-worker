"""D-109/D-110 -- authority collision fix: `hybrid_retry_winner_authority`
must respect an already-recorded `complete_retry_identity_guard`
replacement rejection for the EXACT (failed, winner) pair, instead of
independently re-deriving "same retry attempt" via its own looser test
and silently overriding the earlier, stricter authority's answer.

Root defect (docs/CUTSELL_DECISIONS.md D-109): a real pimples-family RAW
showed `complete_retry_identity_guard` reject a candidate C as a valid
replacement for candidate A (`SEQUENCE_IDENTITY_BELOW_THRESHOLD`, 0.415 <
0.52), yet `enforce_proven_retry_winners`'s own `_same_retry_attempt`
(shared-content-token test) would independently say "same retry attempt"
for the identical pair and could remove A anyway.

Fixtures are generic (no Video00 clip ids, text, or timestamps) -- the
texts model the SHAPE of the real defect (a short complete beat vs. a
longer later realization sharing only generic opening/topic vocabulary),
not the literal transcript.
"""
from cutsell_worker.complete_retry_identity_guard import (
    LEXICAL_REPLACEMENT_VERIFIED,
    NOT_APPLICABLE,
    SEQUENCE_IDENTITY_BELOW_THRESHOLD,
)
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_retry_winner_authority import enforce_proven_retry_winners
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext

# A: short, complete, audience-facing beat (the "failed"-labelled candidate).
A_TEXT = "I also noticed some spots, it felt like a rash, an allergy."
# C: a later, longer realization sharing only generic opening/topic tokens
# with A -- enough for the module's own looser `_same_retry_attempt` test
# to say "same retry attempt", but NOT enough for the stricter
# `complete_retry_identity_guard` sequence-identity check (already
# rejected in the recorded evidence below).
C_TEXT = (
    "I also noticed some marks on this part right here, behind my ear, "
    "which I always assumed was an allergy, but it turned out to be "
    "marks from a hormonal issue."
)


def _take(clip_id, start, end, text, *, source="src", complete=True):
    return CandidateTake(clip_id, source, 0, start, end, text, complete_idea=complete)


def _context(*events, source="src"):
    return WholeVideoContext(
        sources=(
            SourceVideoContext(
                source_asset_id=source,
                summary="raw talking head with retries",
                dominant_style="talking_head",
                creator_intent="tell personal story naturally",
                events=tuple(events),
            ),
        ),
        status=ProviderStatus("test", True, True, "applied"),
    )


def _retry_setup_event(take, source="src"):
    return TemporalEvent(source, take.end + 0.1, take.end + 0.6, "retry_setup", 0.86, "creator resets")


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


# --- pimples-structural positive control -----------------------------------

def test_prior_replacement_rejection_is_respected_candidate_survives():
    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT)
    context = _context(_retry_setup_event(a))
    session_diagnostics = _session_diagnostics("a", "c", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        session_diagnostics=session_diagnostics,
    )

    assert kept == (a, c)
    assert removed == ()
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["clip_id"] == "a"
    assert row["proposed_winner_clip_id"] == "c"
    assert row["prior_replacement_rejection_found"] is True
    assert row["prior_replacement_rejection_reason"] == SEQUENCE_IDENTITY_BELOW_THRESHOLD
    assert row["retry_winner_deletion_applied"] is False
    assert row["final_reason"] == "prior_replacement_rejection_respected"


# --- legitimate retry positive control (D-097.1's own original shape) -----

def test_no_prior_rejection_existing_retry_winner_behavior_unchanged():
    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT)
    context = _context(_retry_setup_event(a))

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        # no session_diagnostics passed at all -- must match pre-D-109 behavior
    )

    assert kept == (c,)
    assert removed == (a,)
    assert diagnostics[0]["reason"] == "failed_attempt_yields_to_proven_later_retry_winner"
    assert diagnostics[0]["retry_winner_deletion_applied"] is True
    assert diagnostics[0]["prior_replacement_rejection_found"] is False


def test_no_prior_rejection_explicit_empty_session_diagnostics_unchanged():
    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT)
    context = _context(_retry_setup_event(a))

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        session_diagnostics=(),
    )

    assert kept == (c,)
    assert removed == (a,)


# --- directionality controls ------------------------------------------------

def test_rejection_for_a_to_c_does_not_block_b_to_c():
    """A rejection recorded for (a, c) must not stop an UNRELATED failed
    candidate b from being legitimately superseded by c on its own
    evidence -- the rejection is directional to the exact (a, c) pair."""
    a = _take("a", 10.0, 15.0, A_TEXT)
    b = _take("b", 40.0, 45.0, A_TEXT.replace("spots", "bumps"))
    c_for_a = _take("c", 16.0, 26.0, C_TEXT)
    c_for_b = _take("d", 46.0, 56.0, C_TEXT.replace("marks", "bumps"))
    context = _context(_retry_setup_event(a), _retry_setup_event(b))
    session_diagnostics = _session_diagnostics("a", "c", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c_for_a, b, c_for_b),
        (("a", "failed", 0.85), ("c", "winner", 0.92), ("b", "failed", 0.85), ("d", "winner", 0.92)),
        context,
        session_diagnostics=session_diagnostics,
    )

    assert a in kept  # a survives (rejection respected)
    assert c_for_a in kept
    assert b not in kept  # b is legitimately superseded -- unaffected by a's rejection
    assert c_for_b in kept
    by_clip = {row["clip_id"]: row for row in diagnostics}
    assert by_clip["a"]["retry_winner_deletion_applied"] is False
    assert by_clip["b"]["retry_winner_deletion_applied"] is True


def test_rejection_from_another_source_does_not_leak_across_sources():
    """A rejection recorded for a clip on a DIFFERENT source asset must
    never affect this source's (a, c) pair -- clip ids are unique per
    take already, but this proves the lookup is keyed by the exact
    clip_id, never by source or family, so no accidental cross-source
    collision is possible even if two sources reused an id."""
    a = _take("a", 10.0, 15.0, A_TEXT, source="src1")
    c = _take("c", 16.0, 26.0, C_TEXT, source="src1")
    context = _context(_retry_setup_event(a, source="src1"), source="src1")
    # Rejection recorded for an unrelated clip id on a different source.
    session_diagnostics = _session_diagnostics("other_clip_src2", "other_winner_src2", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        session_diagnostics=session_diagnostics,
    )

    assert kept == (c,)
    assert removed == (a,)


# --- weak / non-authoritative diagnostic note is never treated as a
#     real rejection ----------------------------------------------------

def test_accepted_replacement_note_does_not_behave_like_a_rejection():
    """LEXICAL_REPLACEMENT_VERIFIED means the guard ACCEPTED the
    replacement -- it must never be treated as a rejection."""
    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT)
    context = _context(_retry_setup_event(a))
    session_diagnostics = _session_diagnostics("a", "c", LEXICAL_REPLACEMENT_VERIFIED)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        session_diagnostics=session_diagnostics,
    )

    assert kept == (c,)
    assert removed == (a,)


def test_not_applicable_note_does_not_behave_like_a_rejection():
    """NOT_APPLICABLE means the guard was never invoked for this pair --
    it must never be treated as a rejection either."""
    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT)
    context = _context(_retry_setup_event(a))
    session_diagnostics = _session_diagnostics("a", "c", NOT_APPLICABLE)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        session_diagnostics=session_diagnostics,
    )

    assert kept == (c,)
    assert removed == (a,)


def test_rejection_for_a_different_proposed_winner_does_not_block_this_one():
    """A rejection recorded for (a, some_other_clip) must not block a's
    legitimate removal in favor of c when c was never the rejected
    candidate -- directionality applies to the exact proposed pair."""
    a = _take("a", 10.0, 15.0, A_TEXT)
    c = _take("c", 16.0, 26.0, C_TEXT)
    context = _context(_retry_setup_event(a))
    session_diagnostics = _session_diagnostics("a", "some_other_clip_id", SEQUENCE_IDENTITY_BELOW_THRESHOLD)

    kept, removed, diagnostics = enforce_proven_retry_winners(
        (a, c),
        (("a", "failed", 0.85), ("c", "winner", 0.92)),
        context,
        session_diagnostics=session_diagnostics,
    )

    assert kept == (c,)
    assert removed == (a,)
