"""D-200.4B: P1 EDITORIAL MOMENT IDENTITY INTEGRITY FIX -- tests.

Proves: (1) `editorial_moment_id` now includes `source_span_id`, closing
the D-200.4A-proven collision (two different P1 candidate occurrences
that bridge onto the same canonical LanguageAttempt, with matching
timing/role, used to mint the identical id); (2) the SAME candidate
occurrence still mints a deterministic, repeatable id; (3) a DIFFERENT
source_span_id always mints a different id, even when every other input
is identical; (4) no other behavior changed -- moment role, confidence,
conflict flags, proposition/attempt ids, D-197 grouping membership, and
D-200.3's structured relation evidence are all byte-identical to before
this fix; (5) the D-200.4A-identified diagnostic false positive
(duplicate-id causing a workflow script's cross-group misread) is
structurally removed once ids are unique; (6) no authority module reads
this id.

See docs/CUTSELL_DECISIONS.md D-200.4A/D-200.4B.
"""
from __future__ import annotations

import inspect

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.editorial_moment_sequence import (
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    _editorial_moment_id,
    classify_editorial_moment,
)
from cutsell_worker.editorial_moment_sequence_integration import (
    build_editorial_moment_understanding_for_source,
)
from cutsell_worker.language_proposition_relation import ClaimSignature, PropositionCandidate, RelationEvidence
from cutsell_worker.language_spine_live_integration import LiveLanguageSpineEvidence
from cutsell_worker.language_utterance_attempt import ATTEMPT_CLEAN, CONFIDENCE_SUPPORTED, MEANING_COMPLETE, LanguageAttempt
from cutsell_worker.watch_listen_understanding import UnderstandingSpan, WatchListenUnderstanding


# ---------------------------------------------------------------------------
# Fixture builders (generic, no Video00 literal / region name anywhere).
# ---------------------------------------------------------------------------
def _take(clip_id, order, s, e, text="generic statement"):
    return CandidateTake(clip_id=clip_id, source_asset_id="src1", source_order=order, start=s, end=e, text=text)


def _span(sid, s, e):
    return UnderstandingSpan(
        span_id=sid, source_asset_id="src1", source_start=s, source_end=e,
        behavior_state_hypotheses=(), behavior_confidence=CONFIDENCE_SUPPORTED,
        attempt_boundary_hypotheses=(), attempt_relation_hypotheses=(),
        relation_confidence=CONFIDENCE_SUPPORTED, meaning_completion_hypothesis=MEANING_COMPLETE,
        performance_usability_hypothesis="USABLE", entry_usability="USABLE",
        delivery_usability="USABLE", exit_usability="USABLE", conflict_flags=(), evidence_provenance={},
    )


def _attempt(attempt_id, s, e, source_asset_id="src1"):
    return LanguageAttempt(
        source_asset_id=source_asset_id, attempt_id=attempt_id, utterance_ids=(attempt_id,),
        source_start=s, source_end=e, text_raw="x", text_normalized="x", attempt_state=ATTEMPT_CLEAN,
        meaning_completion=MEANING_COMPLETE, restart_evidence=False, correction_evidence=False,
        continuation_evidence=False, recording_process_evidence=False, confidence=CONFIDENCE_SUPPORTED,
        provenance="CANONICAL_LANGUAGE_SPINE",
    )


_DUMMY_SIG = ClaimSignature(
    content_tokens=frozenset(), negation_present=False, numbers=frozenset(),
    claim_type="NONE", negation_role="", signature_hash="dummy",
)


def _prop(prop_id, attempt_id, s, e):
    return PropositionCandidate(
        source_asset_id="src1", proposition_candidate_id=prop_id, attempt_ids=(attempt_id,),
        source_start=s, source_end=e, text_raw="x", text_normalized="x", claim_signature=_DUMMY_SIG,
        meaning_completion=MEANING_COMPLETE, editorial_slot_evidence="OTHER", confidence=CONFIDENCE_SUPPORTED,
        provenance="LANGUAGE_ATTEMPT",
    )


def _wlu(*spans, source_asset_id="src1"):
    return WatchListenUnderstanding(source_asset_id=source_asset_id, understanding_spans=tuple(spans))


def _live_spine(*, attempts=(), proposition_candidates=(), relation_evidence=(), source_asset_id="src1"):
    return LiveLanguageSpineEvidence(
        source_asset_id=source_asset_id, words=(), phrases=(), utterances=(), attempts=attempts,
        proposition_candidates=proposition_candidates, relation_evidence=relation_evidence,
        capability_status="AVAILABLE", missing_evidence=(), conflicts=(), provenance=("CANONICAL_LANGUAGE_SPINE",),
    )


def _collision_understanding():
    """Two DIFFERENT P1 candidate occurrences (different source_span_id)
    whose UnderstandingSpans both substantially overlap the SAME one real
    canonical LanguageAttempt -- the exact D-200.4A-proven collision
    shape (many-to-one D-199 bridge + D-195's unfiltered candidate
    pool), generic, no Video00 content."""
    take_a = _take("spanA", 0, 0.0, 2.0)
    take_b = _take("spanB", 1, 0.1, 1.9)
    wlu = _wlu(_span("spanA", 0.0, 2.0), _span("spanB", 0.1, 1.9))
    attempt = _attempt("a0", 0.0, 2.0)
    prop = _prop("p0", "a0", 0.0, 2.0)
    spine = _live_spine(attempts=(attempt,), proposition_candidates=(prop,))
    return build_editorial_moment_understanding_for_source(
        source_asset_id="src1", takes_for_source=[take_a, take_b], watch_listen_understanding=wlu,
        live_language_spine=spine,
    ), take_a, take_b


# ===========================================================================
# 1-3: id includes source_span_id / deterministic / unique per candidate.
# ===========================================================================
def test_01_id_includes_source_span_id():
    src = inspect.signature(_editorial_moment_id)
    assert "source_span_id" in src.parameters


def test_02_same_occurrence_deterministic():
    id1 = _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    id2 = _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    assert id1 == id2


def test_03_different_source_span_id_produces_different_id():
    id_a = _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    id_b = _editorial_moment_id("src1", 0.0, 2.0, "spanB", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    assert id_a != id_b


def test_04_same_canonical_attempt_different_span_unique():
    # Both moments derived from the SAME LanguageAttempt (same attempt_id,
    # same source_start/end) but different source_span_id.
    id_a = _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    id_b = _editorial_moment_id("src1", 0.0, 2.0, "spanB", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    assert id_a != id_b


def test_05_same_timing_different_span_unique():
    id_a = _editorial_moment_id("src1", 5.0, 6.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    id_b = _editorial_moment_id("src1", 5.0, 6.0, "spanB", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    assert id_a != id_b


def test_06_same_role_different_span_unique():
    id_a = _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    id_b = _editorial_moment_id("src1", 0.0, 2.0, "spanB", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    assert id_a != id_b


def test_07_repeated_build_deterministic():
    ids_run1 = [_editorial_moment_id("src1", 0.0, 2.0, sid, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",)) for sid in ("spanA", "spanB")]
    ids_run2 = [_editorial_moment_id("src1", 0.0, 2.0, sid, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",)) for sid in ("spanA", "spanB")]
    assert ids_run1 == ids_run2


def test_08_attempt_id_order_independence_unaffected_by_fix():
    # Pre-existing guarantee (attempt_ids sorted before hashing) must still
    # hold with the new source_span_id input added alongside it.
    id1 = _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0", "a1"))
    id2 = _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a1", "a0"))
    assert id1 == id2


def test_09_none_source_span_id_still_deterministic():
    id1 = _editorial_moment_id("src1", 0.0, 2.0, None, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    id2 = _editorial_moment_id("src1", 0.0, 2.0, None, MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))
    assert id1 == id2
    assert id1 != _editorial_moment_id("src1", 0.0, 2.0, "spanA", MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY, ("a0",))


# ===========================================================================
# 10-17: D-200.4A collision repro -- integration-level, before/after proof.
# ===========================================================================
def test_10_collision_resolved_at_integration_level():
    understanding, take_a, take_b = _collision_understanding()
    assert understanding.moment_count == 2
    ids = [m.editorial_moment_id for m in understanding.moments]
    spans = [m.source_span_id for m in understanding.moments]
    assert spans == ["spanA", "spanB"]
    assert ids[0] != ids[1]  # THE fix -- was a collision (D-200.4A) before this task


def test_11_no_candidate_deduplication():
    # D-195's own doctrine: the unfiltered candidate pool is NEVER
    # deduplicated -- both candidate occurrences still produce their own
    # EditorialMoment, just with distinct ids now.
    understanding, _, _ = _collision_understanding()
    assert understanding.moment_count == 2


def test_12_moment_role_unchanged_by_fix():
    understanding, _, _ = _collision_understanding()
    for m in understanding.moments:
        assert m.moment_role == MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY


def test_13_confidence_unchanged_by_fix():
    understanding, _, _ = _collision_understanding()
    for m in understanding.moments:
        assert m.confidence == CONFIDENCE_SUPPORTED


def test_14_conflict_flags_unchanged_by_fix():
    understanding, _, _ = _collision_understanding()
    for m in understanding.moments:
        assert m.conflict_flags == ()


def test_15_proposition_identity_unchanged_by_fix():
    understanding, _, _ = _collision_understanding()
    for m in understanding.moments:
        assert m.proposition_candidate_ids == ("p0",)


def test_16_language_attempt_identity_unchanged_by_fix():
    understanding, _, _ = _collision_understanding()
    for m in understanding.moments:
        assert m.attempt_ids == ("a0",)


def test_17_structured_relation_architecture_unaffected():
    understanding, _, _ = _collision_understanding()
    # D-200.3's structured relation for the 2nd moment must still compute
    # cleanly (dimension logic reads span ids / RelationEvidence, never
    # editorial_moment_id) -- present and well-formed, not reopened here.
    assert len(understanding.moment_structured_relation) == 2
    assert understanding.moment_structured_relation[0] is None  # no predecessor
    assert understanding.moment_structured_relation[1] is not None


# ===========================================================================
# 18-22: group/sequence membership + one-group-per-sequence invariant.
# ===========================================================================
def test_18_local_group_membership_uses_distinct_ids_now():
    understanding, _, _ = _collision_understanding()
    all_moment_ids_in_groups = [mid for g in understanding.local_groups for mid in g.moment_ids]
    # No duplicate ids across ALL groups combined -- the D-200.4A collision
    # symptom (same id appearing under >1 group) is structurally gone.
    assert len(all_moment_ids_in_groups) == len(set(all_moment_ids_in_groups))


def test_19_sequence_membership_and_kind_present_and_consistent():
    understanding, _, _ = _collision_understanding()
    moment_ids = {m.editorial_moment_id for m in understanding.moments}
    for s in understanding.sequence_hypotheses:
        for mid in s.moment_ids:
            assert mid in moment_ids


def test_20_one_group_per_sequence_invariant_preserved():
    # Structural proof (not a new claim): build_editorial_sequences_for_moments
    # iterates local_groups one entry at a time -- every sequence's own
    # moment_ids come from exactly one group's own moment_ids set.
    understanding, _, _ = _collision_understanding()
    group_id_sets = [frozenset(g.moment_ids) for g in understanding.local_groups]
    for s in understanding.sequence_hypotheses:
        seq_ids = frozenset(s.moment_ids)
        assert any(seq_ids <= g for g in group_id_sets)


def test_21_cross_group_diagnostic_false_positive_removed():
    # Reproduces the EXACT workflow-diagnostic mechanism named in D-200.4A:
    # moment_to_group = {mid: gid for g in local_groups for mid in g.moment_ids}.
    # With unique ids, this dict can never silently overwrite a real
    # group's ownership of a shared id.
    understanding, _, _ = _collision_understanding()
    moment_to_group: dict[str, str] = {}
    collisions = []
    for g in understanding.local_groups:
        for mid in g.moment_ids:
            if mid in moment_to_group and moment_to_group[mid] != g.group_id:
                collisions.append(mid)
            moment_to_group[mid] = g.group_id
    assert collisions == []


def test_22_group_count_and_shape_unaffected_by_fix():
    understanding, _, _ = _collision_understanding()
    # Two isolated (no relation evidence) candidates -> two singleton
    # groups -- grouping RULES (D-197) are untouched by this identity fix.
    assert len(understanding.local_groups) == 2
    assert all(len(g.moment_ids) == 1 for g in understanding.local_groups)


# ===========================================================================
# 23-29: no authority-module consumer (structural, module-leaf).
# ===========================================================================
def test_23_29_no_authority_consumer_of_editorial_moment_id():
    import pathlib
    forbidden_modules = (
        "take_grouping.py", "take_grouping_provider.py", "deterministic_best_take_authority.py",
        "attempt_relationship_authority.py", "bounded_finalist_authority.py", "boundary_engine_pass.py",
        "dialogue_pacing_transition.py", "canonical_edit_plan.py", "realization_resolver.py",
        "composite_resolver.py",
    )
    root = pathlib.Path(__file__).resolve().parent.parent / "cutsell_worker"
    for name in forbidden_modules:
        path = root / name
        if not path.exists():
            continue
        text = path.read_text()
        assert "editorial_moment_id" not in text, f"editorial_moment_id leaked into {name}"


def test_only_p1_modules_reference_editorial_moment_id():
    # D-202: whole_video_editorial_reasoning.py (P2 Phase A) is now an
    # authorized additional reader -- it references EditorialMoment.
    # editorial_moment_id BY ATTRIBUTE ACCESS ONLY (never re-derives or
    # re-mints it), exactly the "reference existing ids, do not copy or
    # recompute" contract D-201/D-202 require. This widens the audit's own
    # allow-list; it does not touch editorial_moment_sequence.py/
    # editorial_moment_sequence_integration.py's own minting authority.
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "cutsell_worker"
    referencing = [
        p.name for p in root.glob("*.py")
        if "editorial_moment_id" in p.read_text()
    ]
    assert set(referencing) == {
        "editorial_moment_sequence.py", "editorial_moment_sequence_integration.py",
        "whole_video_editorial_reasoning.py",
    }


# ===========================================================================
# 30-36: no provider/ASR/RAW/weights/thresholds/hardcoded-region.
# ===========================================================================
def test_30_36_no_provider_asr_weights_thresholds_or_hardcode():
    src = inspect.getsource(_editorial_moment_id)
    forbidden = (
        "requests.", "openai", "gemini", "http", "asr.", "float(0.0", "weight",
        "threshold", "video00", "gynecolog", "pimple",
    )
    lowered = src.lower()
    for needle in forbidden:
        assert needle not in lowered, f"{needle!r} found in _editorial_moment_id"


# ===========================================================================
# 37: no unrelated identity namespace drift (LanguageAttempt/Proposition
# ids untouched by this change -- confirmed structurally: this module
# never mints those, D-166/D-168/D-169 own that namespace unchanged).
# ===========================================================================
def test_37_no_unrelated_identity_namespace_touched():
    src = inspect.getsource(_editorial_moment_id)
    for forbidden in ("mint_attempt_id", "mint_source_span_id(", "mint_semantic_idea_id", "mint_retry_family_id"):
        assert forbidden not in src
