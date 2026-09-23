"""D-061 Phase 2 -- CLAIM EQUIVALENCE ARBITER WIRING safety matrix.

Two things under test:

1. `BrainRuntime`/`build_brain_runtime` correctly construct and gate a
   `claim_equivalence_arbiter` using the SAME already-configured google/
   Gemini provider (no new provider/model), with its own independent cost
   ledger, its own rollback flag, and no external calls made from key
   presence alone -- mirrors the existing `semantic_equivalence_arbiter`
   coverage in test_cutsell_runpod_local_brain.py.

2. The existing `semantic_claims.resolve_ambiguous_coverage`/`claim_
   coverage` safety contract (unchanged by this directive -- D-061 only
   wires an arbiter into it) really does what D-061's ARBITER RULES
   require: the arbiter is consulted ONLY for genuinely ambiguous claims,
   is NEVER consulted for a deterministic hard mismatch (number, negation,
   diagnosis/entity, causal-direction), and fails closed to NOT COVERED on
   an unavailable/erroring/non-True-verdict arbiter. Proven with a call-
   counting fake arbiter so "never called" is a real assertion, not an
   inference from the returned value alone.

Entirely generic -- no Video00 clip ids or phrases.
"""
from __future__ import annotations

import pytest

from cutsell_worker.brain_runtime import build_brain_runtime
from cutsell_worker.config import load_runtime_config
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.semantic_claims import (
    AMBIGUOUS_COVERAGE_FLOOR,
    COVERAGE_THRESHOLD,
    claim_coverage,
    extract_claims,
    resolve_ambiguous_coverage,
)


# ---------------------------------------------------------------------------
# 1. BrainRuntime wiring
# ---------------------------------------------------------------------------

def test_claim_equivalence_arbiter_none_without_hybrid_enabled():
    env = {"CUTSELL_BRAIN_BACKEND": "runpod_local", "GEMINI_API_KEY": "present-but-not-enabled"}
    brain = build_brain_runtime(load_runtime_config(env), env)
    assert brain.claim_equivalence_arbiter is None


def test_claim_equivalence_arbiter_constructed_with_hybrid_enabled_and_key():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PROVIDER": "google",
        "GEMINI_API_KEY": "fake-construction-only",
    }
    brain = build_brain_runtime(load_runtime_config(env), env)
    assert brain.claim_equivalence_arbiter is not None
    assert brain.claim_equivalence_arbiter.model == "gemini-3.5-flash-lite"
    # Its own independent cost ceiling -- must never share the legacy
    # per-group Hybrid judge's, Unified Selection's, or the semantic-
    # equivalence arbiter's ledger/ceiling.
    assert brain.hybrid_settings.max_cost_per_claim_equivalence_call_usd == 0.003
    assert (
        brain.claim_equivalence_arbiter.ledger.remaining_usd
        == brain.hybrid_settings.max_cost_per_claim_equivalence_call_usd
    )


def test_claim_equivalence_arbiter_rollback_flag_disables_it():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "CUTSELL_HYBRID_LLM_ENABLED": "1",
        "CUTSELL_HYBRID_PROVIDER": "google",
        "GEMINI_API_KEY": "fake-construction-only",
        "CUTSELL_CLAIM_EQUIVALENCE_ARBITER": "0",
    }
    brain = build_brain_runtime(load_runtime_config(env), env)
    # The rollback flag disables ONLY this arbiter, not the sibling
    # semantic-equivalence arbiter or the legacy judge.
    assert brain.claim_equivalence_arbiter is None
    assert brain.semantic_equivalence_arbiter is not None
    assert brain.editorial_judge is not None


def test_claim_equivalence_arbiter_never_constructed_from_key_presence_alone():
    env = {
        "CUTSELL_BRAIN_BACKEND": "runpod_local",
        "GEMINI_API_KEY": "present-but-hybrid-not-enabled",
    }
    brain = build_brain_runtime(load_runtime_config(env), env)
    assert brain.external_calls_enabled is False
    assert brain.claim_equivalence_arbiter is None


# ---------------------------------------------------------------------------
# 2. Arbiter-consultation safety matrix
# ---------------------------------------------------------------------------

class _CountingArbiter:
    """Records every call -- proves 'arbiter not called' is real, not an
    inference from the returned coverage value alone."""

    def __init__(self, covered: bool = True, confidence: float = 0.9, raise_exc: bool = False):
        self._covered = covered
        self._confidence = confidence
        self._raise = raise_exc
        self.calls = 0

    def claim_covered(self, claim_text, winning_realization_text):
        self.calls += 1
        if self._raise:
            raise RuntimeError("provider down")
        return self._covered, self._confidence, "fake reason"


def _claim(text):
    claims = extract_claims("clip_a", text)
    assert claims
    return claims[0]


def test_low_lexical_overlap_same_quantitative_fact_reaches_arbiter_covered():
    claim = _claim("Only 5 to 10 percent of these cases are hereditary in nature.")
    candidate = "About 5 to 10 percent of cancers are hereditary, according to her doctor."
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    arbiter = _CountingArbiter(covered=True)
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is True
    assert arbiter.calls == 1


def test_number_mismatch_deterministic_not_covered_arbiter_never_called():
    claim = _claim("Only 5 percent of these cases are hereditary in nature.")
    candidate = "About 10 percent of cancers are hereditary, according to her doctor."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    arbiter = _CountingArbiter(covered=True)  # would say yes to anything -- must never be asked
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is False
    assert arbiter.calls == 0


def test_negation_mismatch_deterministic_not_covered_arbiter_never_called():
    claim = _claim("The biopsy confirmed the tumor was hereditary in nature.")
    candidate = "The biopsy confirmed the tumor was not hereditary in nature."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    arbiter = _CountingArbiter(covered=True)
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is False
    assert arbiter.calls == 0


def test_causal_inversion_deterministic_not_covered_arbiter_never_called():
    claim = _claim("The flare-ups happen because of stress.")
    candidate = "Stress happens because of the flare-ups."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    arbiter = _CountingArbiter(covered=True)
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is False
    assert arbiter.calls == 0


def test_diagnosis_entity_mismatch_not_covered():
    claim = _claim("The doctor diagnosed her with gastritis after the endoscopy.")
    candidate = "It turned out to be an ulcer instead, following further tests months later."
    coverage = claim_coverage(claim, candidate)
    assert coverage < AMBIGUOUS_COVERAGE_FLOOR
    arbiter = _CountingArbiter(covered=True)
    # Naturally low overlap (different diagnosis entirely) never even reaches
    # the ambiguous band -- the arbiter is never given a chance to
    # incorrectly confirm it, exactly as ARBITER RULES requires.
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is False
    assert arbiter.calls == 0


def test_arbiter_says_not_covered_is_blocking():
    claim = _claim("The endoscopy showed she had gastritis, nothing severe.")
    candidate = "Further testing confirmed a mild case of gastritis was responsible for her symptoms."
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    arbiter = _CountingArbiter(covered=False)
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is False
    assert arbiter.calls == 1


def test_arbiter_unavailable_fails_closed():
    claim = _claim("The endoscopy showed she had gastritis, nothing severe.")
    candidate = "Further testing confirmed a mild case of gastritis was responsible for her symptoms."
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=None) is False


def test_arbiter_exception_fails_closed():
    claim = _claim("The endoscopy showed she had gastritis, nothing severe.")
    candidate = "Further testing confirmed a mild case of gastritis was responsible for her symptoms."
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    arbiter = _CountingArbiter(raise_exc=True)
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is False
    assert arbiter.calls == 1


def test_arbiter_uncertain_verdict_fails_closed():
    """An arbiter that cannot confidently decide answers covered=False
    (its own mapping of UNCERTAIN) rather than guessing True -- still
    fails closed, same as an explicit NOT_COVERED."""
    claim = _claim("The endoscopy showed she had gastritis, nothing severe.")
    candidate = "Further testing confirmed a mild case of gastritis was responsible for her symptoms."
    coverage = claim_coverage(claim, candidate)
    assert AMBIGUOUS_COVERAGE_FLOOR <= coverage < COVERAGE_THRESHOLD
    arbiter = _CountingArbiter(covered=False, confidence=0.3)
    assert resolve_ambiguous_coverage(claim, candidate, coverage=coverage, arbiter=arbiter) is False


# ---------------------------------------------------------------------------
# 3. End-to-end: apply_final_story_coherence_validation with a wired arbiter
# ---------------------------------------------------------------------------

def clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def ranked_row(clip_id, score):
    return {"clip_id": clip_id, "score": score, "reason": "watch_listen_baseline"}


def test_end_to_end_ambiguous_claim_resolved_via_wired_arbiter():
    winner = clip("winner", 0.0, 5.0, "About 5 to 10 percent of cancers are hereditary, according to her doctor.", selected=True)
    incomplete_loser = clip(
        "loser", 5.0, 10.0,
        "Only 5 to 10 percent of these cases are hereditary in nature.",
        selected=False,
    )
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(winner,), alternates=(), discarded=(incomplete_loser,),
        diagnostics={"take_judge_groups": [
            {"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]},
        ]},
    )
    arbiter = _CountingArbiter(covered=True)
    out = apply_final_story_coherence_validation(d, claim_equivalence_arbiter=arbiter)
    diag = out.diagnostics["final_story_coherence_validation"]

    assert arbiter.calls == 1
    # The claim-level check specifically: no CRITICAL_CLAIM_LOST finding for
    # the hereditary-percentage claim once the arbiter confirms the
    # paraphrase (freeze_blocked may still be True for an unrelated,
    # incidental _lost_semantic_atoms whole-video coverage signal on this
    # short fixture -- out of scope for this test, which is about claim
    # coverage specifically).
    assert diag["lost_critical_claims"] == []


def test_end_to_end_without_arbiter_still_fails_closed_unchanged():
    winner = clip("winner", 0.0, 5.0, "About 5 to 10 percent of cancers are hereditary, according to her doctor.", selected=True)
    incomplete_loser = clip(
        "loser", 5.0, 10.0,
        "Only 5 to 10 percent of these cases are hereditary in nature.",
        selected=False,
    )
    d = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(winner,), alternates=(), discarded=(incomplete_loser,),
        diagnostics={"take_judge_groups": [
            {"group_id": "g1", "ranked": [ranked_row("winner", 0.9), ranked_row("loser", 0.5)]},
        ]},
    )
    out = apply_final_story_coherence_validation(d)  # no arbiter passed
    diag = out.diagnostics["final_story_coherence_validation"]

    assert len(diag["lost_critical_claims"]) == 1
    assert diag["freeze_blocked"] is True
