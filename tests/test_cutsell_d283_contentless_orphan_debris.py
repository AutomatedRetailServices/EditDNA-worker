"""D-283 (RAW #119 audit): a lone, contentless recording-process fragment
survives Selection as its own standalone clip.

Real RAW #119 evidence (Modal run 35756418346, four-way quality-ladder
attribution AttemptReconstructor/RecordingProcessRemoval): clip_id
`clip_af739df7541f33a4dd5c`, text "No" (0.8s), was a lone candidate both
Cut.ai and Human Gold remove -- `family_members`/`retry_family`/`idea_id`
all null in the engine's own diagnostics (a true orphan). Neither reference
kept it, yet CutSell did.

Root cause: `_safe_short_alternate_debris` (hybrid_retry_completion_
integrity.py) is the authority that removes a short "alternate"-labeled
take when it is fully covered by its neighbors' content -- but its
coverage/shared-content gates all key off `_content(take.text)`, and
`_content()` requires tokens >= 4 characters (excluding stopwords). A bare
"No." (2 characters) always has EMPTY content, so `_coverage()` short-
circuits to 0.0 before the 0.75 floor is ever reached -- the rule cannot
fire on this shape regardless of confidence, duration or adjacency,
because there is structurally no content to measure overlap on. This is
not a threshold problem; lowering `minimum_shared_content`-style floors
would not help, since the intersection is always empty.

Fix: `_safe_contentless_alternate_debris`, a separate rule scoped
specifically to the empty-content shape. It cannot use content-overlap
corroboration (there is none), so it requires a higher-confidence
"alternate" semantic label (0.85 vs the sibling rule's 0.74) plus the same
structural adjacency/duration/single-source evidence, and is deliberately
NOT protected by editorial_guardrails_v2's complete_idea guard (a
grammatically complete "No." is still editorially empty -- completeness
says nothing about content). Fixtures are generic (no Video00 text/clip
ids) and use a topic ("scheduling a delivery") unrelated to the real
evidence.
"""
from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.hybrid_retry_completion_integrity import (
    _safe_contentless_alternate_debris,
    apply_hybrid_retry_completion_integrity,
)
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult


def _words(text, start=0.0, step=0.25):
    output = []
    cursor = float(start)
    for token in text.split():
        output.append(Word(token, cursor, cursor + step, 0.95))
        cursor += step
    return tuple(output)


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=_words(text, start),
        signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def _result(kept, deleted=(), decisions=()):
    return HybridSessionCleanupResult(
        kept=tuple(kept),
        deleted=tuple(deleted),
        requested_chunk_count=1,
        available_chunk_count=1,
        diagnostics=(),
        semantic_decisions=tuple(decisions),
    )


# ---------------------------------------------------------------------------
# Unit tests: _safe_contentless_alternate_debris
# ---------------------------------------------------------------------------

def test_bare_filler_orphan_with_high_confidence_alternate_label_is_removed():
    # D-286 (RAW #120 audit) correction: the positive control here used to
    # be a bare "No." -- real RAW #120 evidence proved a bare negation
    # particle is NOT structural debris (it can be the polarity of the very
    # next clip: "No" + "I want to sound..." = "No, I want to sound...").
    # This rule's real target shape is a genuinely contentless, non-negating
    # filler/false-start particle -- exercised here without any negation or
    # numeric token, which is now an explicit guard (see
    # test_cutsell_d286_negation_preserving_debris.py for that guard).
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = _take("orphan", 4.5, 5.3, "Um.", complete=True)
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {"orphan": ("alternate", 0.90)}

    assert _safe_contentless_alternate_debris(orphan, previous, following, semantic) is True


def test_contentless_orphan_is_removed_even_when_grammatically_complete():
    # The existing complete_idea guard (editorial_guardrails_v2.py) protects
    # SHORT-BUT-MEANINGFUL complete deliveries from the sibling rule; a
    # contentless orphan is a different shape and must not inherit that
    # protection just because "Um." parses as a complete sentence.
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = _take("orphan", 4.5, 5.3, "Um.", complete=True)
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {"orphan": ("alternate", 0.90)}

    assert _safe_contentless_alternate_debris(orphan, previous, following, semantic) is True


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

def test_low_confidence_alternate_label_never_removed():
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = _take("orphan", 4.5, 5.3, "No.")
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {"orphan": ("alternate", 0.80)}  # below the 0.85 floor

    assert _safe_contentless_alternate_debris(orphan, previous, following, semantic) is False


def test_take_with_real_content_is_not_this_rules_shape():
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    candidate = _take("candidate", 4.5, 6.0, "Definitely not.")  # "definitely" is real content
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {"candidate": ("alternate", 0.95)}

    assert _safe_contentless_alternate_debris(candidate, previous, following, semantic) is False


def test_no_alternate_label_never_removed():
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = _take("orphan", 4.5, 5.3, "No.")
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {}  # no semantic label at all -- never removed on structure alone

    assert _safe_contentless_alternate_debris(orphan, previous, following, semantic) is False


def test_missing_neighbor_never_removed():
    orphan = _take("orphan", 4.5, 5.3, "No.")
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {"orphan": ("alternate", 0.95)}

    assert _safe_contentless_alternate_debris(orphan, None, following, semantic) is False


def test_far_from_neighbors_never_removed():
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = _take("orphan", 50.0, 50.8, "No.")  # far gap on both sides
    following = _take("following", 90.0, 94.0, "I want to explain what actually happened with the delivery.")
    semantic = {"orphan": ("alternate", 0.95)}

    assert _safe_contentless_alternate_debris(orphan, previous, following, semantic) is False


def test_long_duration_never_removed():
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = _take("orphan", 4.5, 9.0, "No no no no no no no no.", complete=False)  # > 3.0s
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {"orphan": ("alternate", 0.95)}

    assert _safe_contentless_alternate_debris(orphan, previous, following, semantic) is False


def test_different_source_asset_never_removed():
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = CandidateTake(
        clip_id="orphan", source_asset_id="other_src", source_order=0,
        start=4.5, end=5.3, text="No.", complete_idea=True,
    )
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    semantic = {"orphan": ("alternate", 0.95)}

    assert _safe_contentless_alternate_debris(orphan, previous, following, semantic) is False


# ---------------------------------------------------------------------------
# Integration: apply_hybrid_retry_completion_integrity end to end
# ---------------------------------------------------------------------------

def test_orphan_removed_end_to_end_through_the_integrity_pass():
    previous = _take("previous", 0.0, 4.0, "We looked into the scheduling conflict last week.")
    orphan = _take("orphan", 4.5, 5.3, "Um.")
    following = _take("following", 10.0, 14.0, "I want to explain what actually happened with the delivery.")
    result = _result(
        (previous, orphan, following),
        decisions=(("previous", "keep", 0.90), ("orphan", "alternate", 0.92), ("following", "keep", 0.90)),
    )

    repaired = apply_hybrid_retry_completion_integrity(result, (previous, orphan, following))

    assert "orphan" not in {t.clip_id for t in repaired.kept}
    integrity_diag = repaired.diagnostics[-1]["hybrid_retry_completion_integrity"]
    reasons = {d["clip_id"]: d["reason"] for d in integrity_diag}
    assert reasons.get("orphan") == "semantic_contentless_alternate_orphan"


def test_real_content_short_alternate_still_handled_by_the_sibling_rule_unaffected():
    # Confirms the new rule does not change existing behavior for the
    # content-bearing sibling shape (test_cutsell_editorial_guardrails_v2.py
    # already covers this directly; this is a regression guard at the
    # integration level).
    previous = _take("previous", 0.0, 3.0, "shipment delayed customs paperwork")
    candidate = _take("candidate", 4.0, 8.0, "customs paperwork shipment issue", complete=False)
    following = _take("following", 9.0, 12.0, "issue shipment customs paperwork delayed")
    result = _result(
        (previous, candidate, following),
        decisions=(("previous", "keep", 0.90), ("candidate", "alternate", 0.85), ("following", "keep", 0.90)),
    )

    repaired = apply_hybrid_retry_completion_integrity(result, (previous, candidate, following))

    assert "candidate" not in {t.clip_id for t in repaired.kept}
    integrity_diag = repaired.diagnostics[-1]["hybrid_retry_completion_integrity"]
    reasons = {d["clip_id"]: d["reason"] for d in integrity_diag}
    assert reasons.get("candidate") == "semantic_short_alternate_covered_by_neighbors"
