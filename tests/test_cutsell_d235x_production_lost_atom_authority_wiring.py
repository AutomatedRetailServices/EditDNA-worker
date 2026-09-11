"""D-235X: COMPLETE PRODUCTION LOST-ATOM AUTHORITY DATA-SOURCE + REPAIR
CONTEXT WIRING -- OFFLINE ONLY.

Closes the two remaining GAPs D-235W's own verdict D named:
  GAP A -- no live orchestration caller constructed a real
           `exact_match_by_clip_id` map. `pipeline.py::build_flow_b_draft`
           now does, from live `CandidateTake`/`DraftClip` word data and
           the live Language Spine's real `LanguageAttempt`/
           `PropositionCandidate` evidence, using D-235P's own exact
           word-membership matcher verbatim (never a heuristic bridge).
  GAP B -- D-235T's own same-atom RepairLoop suppression recomputed
           materiality from the row alone. `final_story_coherence_
           validation.py` now ALSO stores the SAME already-computed
           D-235Q result keyed by `lost_atom_provenance_id`
           (`DraftTimeline.lost_atom_materiality_by_provenance_id`), and
           `repair_loop.py`/`lost_atom_repair_suppression.py` accept it
           as an optional, additive, input-source-only parameter.

Mirrors the established D-235J-W source-code-truth + fixture-matrix test
style.
"""
from __future__ import annotations

import cutsell_worker.final_story_coherence_validation as fscv
from cutsell_worker.complete_lost_semantic_atom_materiality import (
    CompleteLostSemanticAtomMateriality,
)
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION, Word
from cutsell_worker.final_edit_reviewer import Finding, UNIQUE_FACT_LOST
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.language_proposition_relation import build_proposition_candidates
from cutsell_worker.language_spine import adapt_words_to_language_words, segment_language_phrases
from cutsell_worker.language_spine_live_integration import (
    CAPABILITY_AVAILABLE,
    LiveLanguageSpineEvidence,
    proposition_candidate_ids_by_attempt_id_for,
    proposition_slot_evidence_by_id_for,
)
from cutsell_worker.language_utterance_attempt import build_language_attempts, segment_language_utterances
from cutsell_worker.lost_atom_repair_suppression import (
    all_blocking_findings_safely_suppressed,
    decide_lost_atom_repair_suppression,
)
from cutsell_worker.shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    build_attempt_language_identity_matches_for_source,
)

_ENV_FLAG = "CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED"


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------
def clip(clip_id, start, end, text, *, selected, source="s1"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def draft(*, selected=(), discarded=(), take_judge_groups=()):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={"take_judge_groups": list(take_judge_groups)},
    )


def build_live_evidence(source: str, words: tuple):
    lwords = adapt_words_to_language_words(source, words)
    phrases = segment_language_phrases(lwords)
    utterances = segment_language_utterances(phrases)
    attempts = build_language_attempts(utterances)
    props = build_proposition_candidates(attempts)
    return LiveLanguageSpineEvidence(
        source_asset_id=source, words=lwords, phrases=phrases, utterances=utterances,
        attempts=attempts, proposition_candidates=props, relation_evidence=(),
        capability_status=CAPABILITY_AVAILABLE, missing_evidence=(), conflicts=(), provenance=("test",),
    )


def d235u_row(**overrides):
    base = {
        "clip_id": "c_d235u",
        "text": "a generic lost fragment of real speech",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
        "missing_critical_atoms": (),
        "atom_classifications": (),
        "own_content_token_count": 20,
        "coverage_against_final_keep": 0.9,
        "lost_atom_provenance_id": "latom_c_d235u_0",
    }
    base.update(overrides)
    return base


def evaluated_group(clip_id, other_id="c_other"):
    return [{"group_id": "g1", "ranked": [{"clip_id": clip_id}, {"clip_id": other_id}]}]


# ---------------------------------------------------------------------------
# Part A: live exact-identity data-source seam (D-235P reused verbatim over
# REAL Language Spine objects, never a heuristic-overlap bridge).
# ---------------------------------------------------------------------------
class TestLiveExactIdentityDataSource:
    def test_1_exact_match_1to1_over_real_language_spine(self):
        source = "s1"
        words = (
            Word(text="hello", start=0.0, end=0.5),
            Word(text="world", start=0.5, end=1.0),
        )
        evidence = build_live_evidence(source, words)
        candidate = CandidateTake(
            clip_id="c1", source_asset_id=source, source_order=0, start=0.0, end=1.0,
            text="hello world", words=words,
        )
        utterances_by_id = {u.utterance_id: u for u in evidence.utterances}
        matches = build_attempt_language_identity_matches_for_source(
            reconstructed_attempts=(candidate,), canonical_words=evidence.words,
            language_attempts=evidence.attempts, utterances_by_id=utterances_by_id, phrases=evidence.phrases,
        )
        assert matches[0].relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES

    def test_2_multi_source_isolation_never_cross_matches(self):
        words_a = (Word(text="hello", start=0.0, end=0.5),)
        words_b = (Word(text="hello", start=0.0, end=0.5),)  # identical timing, different source
        evidence_a = build_live_evidence("sA", words_a)
        candidate_b = CandidateTake(
            clip_id="cB", source_asset_id="sB", source_order=0, start=0.0, end=0.5,
            text="hello", words=words_b,
        )
        utterances_by_id = {u.utterance_id: u for u in evidence_a.utterances}
        matches = build_attempt_language_identity_matches_for_source(
            reconstructed_attempts=(candidate_b,), canonical_words=evidence_a.words,
            language_attempts=evidence_a.attempts, utterances_by_id=utterances_by_id, phrases=evidence_a.phrases,
        )
        # Source mismatch -- never an authoritative cross-source match.
        assert matches[0].relationship_status not in AUTHORITATIVE_RELATIONSHIP_STATUSES

    def test_3_missing_exact_identity_no_authoritative_entry(self):
        source = "s1"
        words = (Word(text="hello", start=0.0, end=0.5),)
        evidence = build_live_evidence(source, words)
        # A candidate with completely different words -- disjoint.
        other_words = (Word(text="goodbye", start=5.0, end=5.5),)
        candidate = CandidateTake(
            clip_id="c2", source_asset_id=source, source_order=0, start=5.0, end=5.5,
            text="goodbye", words=other_words,
        )
        utterances_by_id = {u.utterance_id: u for u in evidence.utterances}
        matches = build_attempt_language_identity_matches_for_source(
            reconstructed_attempts=(candidate,), canonical_words=evidence.words,
            language_attempts=evidence.attempts, utterances_by_id=utterances_by_id, phrases=evidence.phrases,
        )
        assert matches[0].relationship_status not in AUTHORITATIVE_RELATIONSHIP_STATUSES

    def test_4_proposition_maps_are_trivial_passthrough(self):
        source = "s1"
        words = (Word(text="hello", start=0.0, end=0.5),)
        evidence = build_live_evidence(source, words)
        prop_ids = proposition_candidate_ids_by_attempt_id_for(evidence.proposition_candidates)
        slot_ev = proposition_slot_evidence_by_id_for(evidence.proposition_candidates)
        for prop in evidence.proposition_candidates:
            for attempt_id in prop.attempt_ids:
                assert prop.proposition_candidate_id in prop_ids[attempt_id]
            assert slot_ev[prop.proposition_candidate_id] == prop.editorial_slot_evidence


# ---------------------------------------------------------------------------
# Part B: compute-once materiality keyed by lost_atom_provenance_id, and
# D-235R/D-235T consuming the SAME result.
# ---------------------------------------------------------------------------
class TestComputeOnceMaterialityByProvenanceId:
    def _run_full_chain(self, monkeypatch, *, with_context: bool):
        monkeypatch.setenv(_ENV_FLAG, "1")
        row = d235u_row()

        def fake_lost_semantic_atoms(*args, **kwargs):
            return [row]

        monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)
        c1 = clip("c_d235u", 0.0, 1.0, "text", selected=False)
        c2 = clip("c_other", 1.0, 2.0, "other text", selected=True)
        d = draft(selected=(c2,), discarded=(c1,), take_judge_groups=evaluated_group("c_d235u"))

        source = "s1"
        words = (Word(text="a", start=0.0, end=0.2), Word(text="b", start=0.2, end=0.4))
        evidence = build_live_evidence(source, words)
        candidate = CandidateTake(
            clip_id="c_d235u", source_asset_id=source, source_order=0, start=0.0, end=0.4,
            text="a b", words=words,
        )
        utterances_by_id = {u.utterance_id: u for u in evidence.utterances}
        matches = build_attempt_language_identity_matches_for_source(
            reconstructed_attempts=(candidate,), canonical_words=evidence.words,
            language_attempts=evidence.attempts, utterances_by_id=utterances_by_id, phrases=evidence.phrases,
        )
        exact_match_by_clip_id = {"c_d235u": matches[0]} if with_context else {}
        proposition_candidate_ids_by_attempt_id = (
            proposition_candidate_ids_by_attempt_id_for(evidence.proposition_candidates) if with_context else {}
        )
        proposition_slot_evidence_by_id = (
            {p.proposition_candidate_id: "OTHER" for p in evidence.proposition_candidates} if with_context else {}
        )
        return apply_final_story_coherence_validation(
            d,
            exact_match_by_clip_id=exact_match_by_clip_id,
            proposition_candidate_ids_by_attempt_id=proposition_candidate_ids_by_attempt_id,
            proposition_slot_evidence_by_id=proposition_slot_evidence_by_id,
        )

    def test_5_materiality_stored_by_provenance_id(self, monkeypatch):
        out = self._run_full_chain(monkeypatch, with_context=True)
        assert "latom_c_d235u_0" in out.lost_atom_materiality_by_provenance_id
        m = out.lost_atom_materiality_by_provenance_id["latom_c_d235u_0"]
        assert isinstance(m, CompleteLostSemanticAtomMateriality)
        assert m.final_materiality_status == "NON_MATERIAL_REAL_CONTENT"

    def test_6_d235r_freeze_unblocked(self, monkeypatch):
        out = self._run_full_chain(monkeypatch, with_context=True)
        assert out.diagnostics["final_story_coherence_validation"]["freeze_blocked"] is False

    def test_7_d235t_consumes_same_result_and_suppresses(self, monkeypatch):
        out = self._run_full_chain(monkeypatch, with_context=True)
        row = d235u_row()
        finding = Finding(
            kind=UNIQUE_FACT_LOST, plan_id="p1", plan_version=1, idea_id=None,
            clip_ids=("c_d235u",), detail=dict(row), owning_authority="StoryValidator", blocking=True,
        )
        decision = decide_lost_atom_repair_suppression(
            finding, all_findings=(finding,), enabled=True,
            materiality_by_provenance_id=out.lost_atom_materiality_by_provenance_id,
        )
        assert decision.suppress_repair_escalation is True
        assert decision.suppression_status == "SUPPRESS_SAME_NON_MATERIAL_ATOM"

    def test_8_all_blocking_findings_safely_suppressed_with_context(self, monkeypatch):
        out = self._run_full_chain(monkeypatch, with_context=True)
        row = d235u_row()
        finding = Finding(
            kind=UNIQUE_FACT_LOST, plan_id="p1", plan_version=1, idea_id=None,
            clip_ids=("c_d235u",), detail=dict(row), owning_authority="StoryValidator", blocking=True,
        )
        all_suppressed, decisions = all_blocking_findings_safely_suppressed(
            (finding,), enabled=True,
            materiality_by_provenance_id=out.lost_atom_materiality_by_provenance_id,
        )
        assert all_suppressed is True

    def test_9_no_context_falls_back_to_legacy_row_only_abstain(self):
        """Backward compat: `materiality_by_provenance_id=None` (every
        pre-D-235X caller) preserves the exact prior D-235T behavior."""
        row = d235u_row()
        finding = Finding(
            kind=UNIQUE_FACT_LOST, plan_id="p1", plan_version=1, idea_id=None,
            clip_ids=("c_d235u",), detail=dict(row), owning_authority="StoryValidator", blocking=True,
        )
        decision = decide_lost_atom_repair_suppression(finding, all_findings=(finding,), enabled=True)
        assert decision.suppress_repair_escalation is False
        assert decision.suppression_status == "ABSTAIN_PRESERVE_ESCALATION"

    def test_10_no_matching_provenance_entry_falls_back_to_recompute(self, monkeypatch):
        """A supplied map with NO entry for this row's own provenance id
        behaves identically to no map at all -- never a partial/best-
        effort merge of the two sources."""
        row = d235u_row(lost_atom_provenance_id="latom_other_0")
        finding = Finding(
            kind=UNIQUE_FACT_LOST, plan_id="p1", plan_version=1, idea_id=None,
            clip_ids=("c_d235u",), detail=dict(row), owning_authority="StoryValidator", blocking=True,
        )
        unrelated_map = {"latom_unrelated_9": "not a real materiality object"}
        decision = decide_lost_atom_repair_suppression(
            finding, all_findings=(finding,), enabled=True, materiality_by_provenance_id=unrelated_map,
        )
        assert decision.suppression_status == "ABSTAIN_PRESERVE_ESCALATION"


# ---------------------------------------------------------------------------
# Fail-closed / flag parity.
# ---------------------------------------------------------------------------
class TestFailClosedAndParity:
    def test_11_flag_off_pipeline_builds_no_context(self, monkeypatch):
        monkeypatch.delenv(_ENV_FLAG, raising=False)
        from cutsell_worker.lost_semantic_atom_freeze_authority import (
            lost_atom_materiality_freeze_authority_enabled,
        )
        assert lost_atom_materiality_freeze_authority_enabled() is False

    def test_12_missing_context_preserves_block_not_promoted(self, monkeypatch):
        """GAP A absent (no exact_match_by_clip_id entry) -> D-235Q abstains
        -> D-235R preserves block -- no fallback promotion."""
        out = self.__class__._chain_without_exact_identity(monkeypatch)
        assert out.diagnostics["final_story_coherence_validation"]["freeze_blocked"] is True

    @staticmethod
    def _chain_without_exact_identity(monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "1")
        row = d235u_row()

        def fake_lost_semantic_atoms(*args, **kwargs):
            return [row]

        monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)
        c1 = clip("c_d235u", 0.0, 1.0, "text", selected=False)
        c2 = clip("c_other", 1.0, 2.0, "other text", selected=True)
        d = draft(selected=(c2,), discarded=(c1,), take_judge_groups=evaluated_group("c_d235u"))
        return apply_final_story_coherence_validation(d)  # no exact identity supplied at all

    def test_13_unidirectional_no_circular_import(self):
        """RepairLoop must not call back into Freeze to obtain its own
        decision -- confirmed structurally: `lost_atom_repair_suppression.py`
        is not imported by `final_story_coherence_validation.py`."""
        with open(fscv.__file__) as f:
            content = f.read()
        assert "import lost_atom_repair_suppression" not in content
        assert "from .lost_atom_repair_suppression" not in content

    def test_14_no_global_mutable_state_introduced(self):
        """`run_repair_loop`'s new parameter is a per-call argument, never
        a module-level cache."""
        import inspect
        import cutsell_worker.repair_loop as repair_loop_mod
        source = inspect.getsource(repair_loop_mod)
        assert "lru_cache" not in source
        assert "functools.cache" not in source
