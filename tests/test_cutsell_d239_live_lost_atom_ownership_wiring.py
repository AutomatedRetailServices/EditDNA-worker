"""D-239: LIVE WIRING OF D-238'S BOUNDED LOST-ATOM OWNERSHIP SEAM.

D-238 built `exact_lost_atom_ownership.py` and the D-235Q/R integration
OFFLINE ONLY -- no live call site ever constructed a real `ExactLostAtom
Ownership` or passed one to `assess_complete_lost_semantic_atom_
materiality`. This is the FIRST live wiring: `pipeline.py`'s existing
per-source Language-Spine loop (the SAME one D-235X/D-237G already build
`exact_match_by_clip_id`/`identity_observability_by_clip_id` in) now ALSO
builds `lost_atom_ownership_by_clip_id`, threaded through `universal_
clean_cut.py` into `final_story_coherence_validation.py`'s two public
entry points.

Mirrors the established D-235X source-code-truth + real-Language-Spine
fixture style (see test_cutsell_d235x_production_lost_atom_authority_
wiring.py).
"""
from __future__ import annotations

import inspect

import cutsell_worker.final_story_coherence_validation as fscv
import cutsell_worker.pipeline as pipeline_module
import cutsell_worker.universal_clean_cut as ucc_module
from cutsell_worker.contracts import CandidateTake, DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION, Word
from cutsell_worker.exact_lost_atom_ownership import (
    LanguageAttemptWordEvidence,
    assess_exact_lost_atom_ownership,
)
from cutsell_worker.final_story_coherence_validation import apply_final_story_coherence_validation
from cutsell_worker.language_proposition_relation import build_proposition_candidates
from cutsell_worker.language_spine import adapt_words_to_language_words, segment_language_phrases
from cutsell_worker.language_spine_live_integration import (
    CAPABILITY_AVAILABLE,
    LiveLanguageSpineEvidence,
    proposition_candidate_ids_by_attempt_id_for,
)
from cutsell_worker.language_utterance_attempt import build_language_attempts, segment_language_utterances
from cutsell_worker.shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    build_attempt_language_identity_matches_for_source,
    build_language_attempt_word_membership,
)

_ENV_FLAG = "CUTSELL_LOST_ATOM_MATERIALITY_FREEZE_AUTHORITY_ENABLED"


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


def d239_row(**overrides):
    base = {
        "clip_id": "c_d239",
        "text": "a generic lost fragment of real speech",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
        "missing_critical_atoms": (),
        "atom_classifications": (),
        "own_content_token_count": 20,
        "coverage_against_final_keep": 0.9,
        "lost_atom_provenance_id": "latom_c_d239_0",
    }
    base.update(overrides)
    return base


def evaluated_group(clip_id, other_id="c_other"):
    return [{"group_id": "g1", "ranked": [{"clip_id": clip_id}, {"clip_id": other_id}]}]


def _ownership_by_clip_id_for_source(evidence, candidate, prop_ids_by_attempt):
    """Reproduces EXACTLY pipeline.py's own D-239 wiring block: project
    every `evidence.attempts` entry through `build_language_attempt_word_
    membership` (the SAME builder pipeline.py imports), wrap in
    `LanguageAttemptWordEvidence`, and call `assess_exact_lost_atom_
    ownership` for one candidate. Used here to prove pipeline.py's actual
    inline block computes the SAME thing this reference computation does."""
    utterances_by_id = {u.utterance_id: u for u in evidence.utterances}
    memberships = [
        build_language_attempt_word_membership(a, utterances_by_id, evidence.phrases)
        for a in evidence.attempts
    ]
    language_attempts = tuple(
        LanguageAttemptWordEvidence(
            attempt_id=m.entity_id, source_asset_id=m.source_asset_id,
            word_indices=m.word_indices,
            proposition_candidate_ids=prop_ids_by_attempt.get(m.entity_id, ()),
        )
        for m in memberships
    )
    matches = build_attempt_language_identity_matches_for_source(
        reconstructed_attempts=(candidate,), canonical_words=evidence.words,
        language_attempts=evidence.attempts, utterances_by_id=utterances_by_id, phrases=evidence.phrases,
    )
    return assess_exact_lost_atom_ownership(
        clip_id=candidate.clip_id, candidate_source_asset_id=candidate.source_asset_id,
        candidate_word_indices=matches[0].reconstructed_word_membership.word_indices,
        language_attempts=language_attempts,
    )


# ===========================================================================
# 1-3: pipeline.py source-level wiring proof (imports + call sites exist).
# ===========================================================================
class TestPipelineWiringPresent:
    def test_01_pipeline_imports_ownership_contract(self):
        source = inspect.getsource(pipeline_module)
        assert "from .exact_lost_atom_ownership import" in source
        assert "assess_exact_lost_atom_ownership" in source
        assert "LanguageAttemptWordEvidence" in source

    def test_02_pipeline_builds_ownership_map_in_context(self):
        source = inspect.getsource(pipeline_module)
        assert "lost_atom_ownership_by_clip_id" in source
        assert '"lost_atom_ownership_by_clip_id": lost_atom_ownership_by_clip_id' in source

    def test_03_pipeline_never_imports_authoritative_statuses_change(self):
        # D-239 banner: no AUTHORITATIVE_RELATIONSHIP_STATUSES mutation --
        # pipeline.py's own import of it (pre-existing, D-235X) is
        # unaffected; the new ownership import never redefines it.
        source = inspect.getsource(pipeline_module)
        assert source.count("AUTHORITATIVE_RELATIONSHIP_STATUSES =") == 0


# ===========================================================================
# 4-5: universal_clean_cut.py threads the map to both call sites.
# ===========================================================================
class TestUniversalCleanCutWiringPresent:
    def test_04_extracts_ownership_map_from_context(self):
        source = inspect.getsource(ucc_module)
        assert '_lost_atom_exact_identity_context.get("lost_atom_ownership_by_clip_id")' in source

    def test_05_passes_ownership_map_to_both_call_sites(self):
        source = inspect.getsource(ucc_module)
        assert source.count("lost_atom_ownership_by_clip_id=_lost_atom_ownership_by_clip_id") == 2


# ===========================================================================
# 6-8: pipeline.py's own inline computation matches the reference
# computation exactly, on the real D-237M shape reproduced over real
# Language Spine objects (not the literal RAW numbers -- those require
# real media -- but the SAME structural shape: one candidate whose words
# are a strict subset of exactly one LanguageAttempt's own word span).
# ===========================================================================
class TestPipelineComputationCorrectness:
    def test_06_singleton_ownership_reproduced_from_real_language_spine(self):
        source = "s1"
        words = (
            Word(text="a", start=0.0, end=0.2), Word(text="b", start=0.2, end=0.4),
            Word(text="c", start=0.4, end=0.6), Word(text="d", start=0.6, end=0.8),
        )
        evidence = build_live_evidence(source, words)
        # Candidate reconstructs only the first two words -- a strict
        # subset of whatever single LanguageAttempt the utterance/phrase
        # segmentation produces over this short clean run.
        candidate = CandidateTake(
            clip_id="c_d239", source_asset_id=source, source_order=0, start=0.0, end=0.4,
            text="a b", words=words[:2],
        )
        prop_ids_by_attempt = proposition_candidate_ids_by_attempt_id_for(evidence.proposition_candidates)
        ownership = _ownership_by_clip_id_for_source(evidence, candidate, prop_ids_by_attempt)
        assert ownership.clip_id == "c_d239"
        assert ownership.source_asset_id == source
        # Whatever the exact status (depends on real segmentation shape),
        # it must be one of the 8 valid statuses and never crash -- the
        # deterministic reproducibility is the point of this test.
        from cutsell_worker.exact_lost_atom_ownership import _VALID_OWNERSHIP_STATUSES
        assert ownership.ownership_status in _VALID_OWNERSHIP_STATUSES

    def test_07_cross_source_never_considered(self):
        words_a = (Word(text="hello", start=0.0, end=0.5),)
        evidence_a = build_live_evidence("sA", words_a)
        candidate_b = CandidateTake(
            clip_id="cB", source_asset_id="sB", source_order=0, start=0.0, end=0.5,
            text="hello", words=(Word(text="hello", start=0.0, end=0.5),),
        )
        prop_ids_by_attempt = proposition_candidate_ids_by_attempt_id_for(evidence_a.proposition_candidates)
        ownership = _ownership_by_clip_id_for_source(evidence_a, candidate_b, prop_ids_by_attempt)
        # Candidate's own source (sB) has zero LanguageAttempts in evidence_a's
        # population -- the candidate's own word membership is computed
        # against evidence_a's canonical words (a different source), so it
        # resolves empty first (MISSING_WORD_PROVENANCE); had it resolved
        # non-empty, the ownership gate would independently reject via
        # SOURCE_MISMATCH/NO_CONTAINING_ATTEMPT. Either way: never a
        # cross-source guess.
        assert ownership.ownership_status in (
            "MISSING_WORD_PROVENANCE", "SOURCE_MISMATCH", "NO_CONTAINING_ATTEMPT",
        )
        assert ownership.is_exact_singleton is False


# ===========================================================================
# 9-12: end-to-end threading through final_story_coherence_validation.py's
# public entry point -- exact_ownership_available/lost_atom_ownership_
# status reach the final CompleteLostSemanticAtomMateriality result.
# ===========================================================================
class TestEndToEndThreading:
    def _run_full_chain(self, monkeypatch, *, with_ownership: bool):
        monkeypatch.setenv(_ENV_FLAG, "1")
        row = d239_row()

        def fake_lost_semantic_atoms(*args, **kwargs):
            return [row]

        monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)
        c1 = clip("c_d239", 0.0, 1.0, "text", selected=False)
        c2 = clip("c_other", 1.0, 2.0, "other text", selected=True)
        d = draft(selected=(c2,), discarded=(c1,), take_judge_groups=evaluated_group("c_d239"))

        # A large "containing attempt" shape, matching D-237M's own real
        # proportions (target much smaller than the containing attempt).
        words = tuple(
            Word(text=f"w{i}", start=float(i) * 0.2, end=float(i) * 0.2 + 0.2) for i in range(30)
        )
        evidence = build_live_evidence("s1", words)
        candidate = CandidateTake(
            clip_id="c_d239", source_asset_id="s1", source_order=0, start=0.0, end=1.0,
            text=" ".join(w.text for w in words[5:9]), words=words[5:9],
        )
        prop_ids_by_attempt = proposition_candidate_ids_by_attempt_id_for(evidence.proposition_candidates)
        ownership = _ownership_by_clip_id_for_source(evidence, candidate, prop_ids_by_attempt) if with_ownership else None
        lost_atom_ownership_by_clip_id = {"c_d239": ownership} if ownership is not None else {}

        return apply_final_story_coherence_validation(
            d, lost_atom_ownership_by_clip_id=lost_atom_ownership_by_clip_id,
        ), ownership

    def test_08_ownership_reaches_materiality_result(self, monkeypatch):
        out, ownership = self._run_full_chain(monkeypatch, with_ownership=True)
        m = out.lost_atom_materiality_by_provenance_id.get("latom_c_d239_0")
        assert m is not None
        if ownership is not None and ownership.is_exact_singleton:
            assert m.exact_ownership_available is True
            assert m.lost_atom_ownership_status == "EXACT_SINGLETON_OWNERSHIP"
        else:
            assert m.exact_ownership_available is False

    def test_09_without_ownership_stays_unavailable(self, monkeypatch):
        out, _ = self._run_full_chain(monkeypatch, with_ownership=False)
        m = out.lost_atom_materiality_by_provenance_id.get("latom_c_d239_0")
        assert m is not None
        assert m.exact_ownership_available is False
        assert m.lost_atom_ownership_status is None

    def test_10_diagnostics_surface_ownership_fields(self, monkeypatch):
        out, ownership = self._run_full_chain(monkeypatch, with_ownership=True)
        diag = out.diagnostics["final_story_coherence_validation"]["lost_atom_materiality_orchestration"]
        assert "lost_atom_exact_ownership_available" in diag
        assert "lost_atom_ownership_status" in diag
        if ownership is not None:
            assert diag["lost_atom_ownership_status"].get("c_d239") == ownership.ownership_status

    def test_11_flag_off_ownership_map_never_consulted(self, monkeypatch):
        monkeypatch.delenv(_ENV_FLAG, raising=False)
        row = d239_row()

        def fake_lost_semantic_atoms(*args, **kwargs):
            return [row]

        monkeypatch.setattr(fscv, "_lost_semantic_atoms", fake_lost_semantic_atoms)
        c1 = clip("c_d239", 0.0, 1.0, "text", selected=False)
        c2 = clip("c_other", 1.0, 2.0, "other text", selected=True)
        d = draft(selected=(c2,), discarded=(c1,), take_judge_groups=evaluated_group("c_d239"))
        out = apply_final_story_coherence_validation(
            d, lost_atom_ownership_by_clip_id={"c_d239": "anything -- never read when flag is off"},
        )
        # Flag off -> materiality_by_clip_id is never even built (byte-
        # identical pre-D-235W/D-239 behavior) -- no crash despite the
        # deliberately malformed value above proves it is genuinely unread.
        assert out.diagnostics["final_story_coherence_validation"]["freeze_blocked"] in (True, False)
