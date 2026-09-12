"""D-237G: EXACT IDENTITY REAL-MEDIA OBSERVABILITY FOUNDATION -- tests.

See docs/CUTSELL_DECISIONS.md D-237F (the forensic finding this task
closes the observability gap for) and D-237G (this implementation).

This module (`exact_identity_observability.py`) is PURE OBSERVABILITY:
it recomputes no set relationship, mints no id, and NEVER promotes a
non-authoritative relationship (e.g. containment) to authoritative
status. Every row it builds is a bounded re-projection of already-
computed `shared_attempt_word_identity.py` (D-235P) objects, reused
verbatim.

Covers: relationship-status serialization for all 5 real shapes
(EXACT_SAME_MEMBERSHIP, LANGUAGE_CONTAINS_RECONSTRUCTED,
RECONSTRUCTED_CONTAINS_LANGUAGE, PARTIAL_OVERLAP, DISJOINT),
cross-source isolation, the authoritative boolean mirroring the
canonical status set unchanged, exact_match_by_clip_id presence
reporting, proposition-id reporting, lost-atom correlation, fail-open
behavior on empty/missing data, determinism, boundedness, and explicit
proofs that no authority/threshold/detector/provider/RAW was touched.
"""
from __future__ import annotations

import inspect

from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.language_proposition_relation import build_proposition_candidates
from cutsell_worker.language_spine import adapt_words_to_language_words, segment_language_phrases
from cutsell_worker.language_utterance_attempt import build_language_attempts, segment_language_utterances
import cutsell_worker.exact_identity_observability as observability_module
from cutsell_worker.exact_identity_observability import (
    candidate_take_identity_diagnostic_row,
    exact_identity_observability_diagnostics,
    identity_match_diagnostic_row,
    identity_observability_rows_for_source,
    language_attempt_identity_diagnostic_row,
    lost_atom_identity_correlation,
)
from cutsell_worker.language_spine_live_integration import proposition_candidate_ids_by_attempt_id_for
from cutsell_worker.shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    RELATIONSHIP_DISJOINT,
    RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED,
    RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP,
    RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE,
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    build_attempt_language_identity_matches_for_source,
    build_language_attempt_word_membership,
    build_reconstructed_attempt_word_membership,
)


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------
def _words(text: str, t0: float, *, per_word: float = 0.25, inner_gap: float = 0.05):
    words = []
    t = t0
    for tok in text.split(" "):
        words.append(Word(text=tok, start=t, end=t + per_word, confidence=0.9))
        t = t + per_word + inner_gap
    return tuple(words), t


def _take(clip_id: str, source_asset_id: str, words):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=words[0].start, end=words[-1].end, text=" ".join(w.text for w in words),
        words=tuple(words),
    )


def _real_spine_for_source(source_asset_id: str, all_words, *, silence_intervals=()):
    """Builds a REAL, full Language Spine (D-166/D-168/D-169's own
    builders, unchanged) for one source -- used to exercise the
    observability module against genuinely-computed objects, never a
    synthetic AttemptLanguageIdentityMatch."""
    lwords = adapt_words_to_language_words(source_asset_id, all_words)
    phrases = segment_language_phrases(lwords, audio_silence_intervals=silence_intervals)
    utterances = segment_language_utterances(phrases)
    attempts = build_language_attempts(utterances)
    props = build_proposition_candidates(attempts)
    utterances_by_id = {u.utterance_id: u for u in utterances}
    return lwords, phrases, utterances_by_id, attempts, props


def _two_clip_fixture(source_asset_id="s1"):
    """Two 7-word deliveries, separated by a real silence interval --
    each becomes its own real LanguageAttempt (EXACT_SAME_MEMBERSHIP
    against its own clip)."""
    w1, t1 = _words("first real chunk of many spoken words", 0.0)
    w2, _ = _words("second real chunk of many spoken words", t1 + 1.0)
    all_words = w1 + w2
    lwords, phrases, utterances_by_id, attempts, props = _real_spine_for_source(
        source_asset_id, all_words, silence_intervals=((t1, t1 + 1.0),),
    )
    take1 = _take("clipA", source_asset_id, w1)
    take2 = _take("clipB", source_asset_id, w2)
    return lwords, phrases, utterances_by_id, attempts, props, take1, take2


def _build_matches(reconstructed, canonical_words, attempts, utterances_by_id, phrases):
    return build_attempt_language_identity_matches_for_source(
        reconstructed_attempts=reconstructed, canonical_words=canonical_words,
        language_attempts=attempts, utterances_by_id=utterances_by_id, phrases=phrases,
    )


# ===========================================================================
# 1-5: relationship-status serialization for all 5 real shapes.
# ===========================================================================
def test_01_exact_same_membership_serialized_correctly():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    matches = _build_matches((take1,), lwords, attempts, utterances_by_id, phrases)
    prop_ids = proposition_candidate_ids_by_attempt_id_for(props)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take1,), attempts_by_id={a.attempt_id: a for a in attempts},
        proposition_candidate_ids_by_attempt_id=prop_ids,
    )
    row = rows["clipA"]
    assert row["relationship_status"] == RELATIONSHIP_EXACT_SAME_MEMBERSHIP
    assert row["relationship_is_authoritative"] is True
    assert row["intersection_count"] == row["candidate_take"]["word_index_count"]
    assert row["reconstructed_only_word_indices"] == []
    assert row["language_only_word_indices"] == []


def test_02_language_contains_reconstructed_serialized_correctly():
    # A small, discarded fragment CandidateTake spanning only PART of a
    # larger real LanguageAttempt (no silence between them -> one big
    # attempt) -- exactly D-237F's own root-cause shape.
    w1, t1 = _words("first real chunk of many spoken words", 0.0)
    w2, _ = _words("second real chunk of many spoken words", t1 + 0.10)
    all_words = w1 + w2
    lwords, phrases, utterances_by_id, attempts, props = _real_spine_for_source("s1", all_words)
    assert len(attempts) == 1  # no silence -> one merged attempt, by construction
    small_take = _take("clipFragment", "s1", w1)  # only the FIRST half's words
    matches = _build_matches((small_take,), lwords, attempts, utterances_by_id, phrases)
    prop_ids = proposition_candidate_ids_by_attempt_id_for(props)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(small_take,),
        attempts_by_id={a.attempt_id: a for a in attempts}, proposition_candidate_ids_by_attempt_id=prop_ids,
    )
    row = rows["clipFragment"]
    assert row["relationship_status"] == RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED
    assert row["relationship_is_authoritative"] is False  # containment never auto-promoted
    assert row["language_only_word_index_count"] > 0
    assert row["reconstructed_only_word_index_count"] == 0
    assert row["exact_match_by_clip_id_present"] is False


def test_03_reconstructed_contains_language_serialized_correctly():
    # The reverse: ONE reconstructed attempt spanning BOTH real,
    # separately-silenced LanguageAttempts partitions exactly -- but a
    # reconstructed attempt covering only PART of the canonical words
    # while a language attempt is a strict subset of it needs a
    # different fixture: reuse D-235P's own contract directly here since
    # this exact shape requires a language attempt built from FEWER
    # words than the reconstructed span covers by construction.
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    combined = _take("clipCombined", "s1", tuple(w for w in (take1.words + take2.words)))
    matches = _build_matches((combined,), lwords, attempts, utterances_by_id, phrases)
    row_match = matches[0]
    # Two real attempts, both fully contained -> partition shape (tested
    # separately below); RECONSTRUCTED_CONTAINS_LANGUAGE needs exactly
    # ONE overlapping attempt whose own words are a strict subset of the
    # reconstructed clip's own words -- reuse the D-235P unit directly.
    reconstructed_membership = build_reconstructed_attempt_word_membership(combined, lwords)
    language_membership = build_language_attempt_word_membership(attempts[0], utterances_by_id, phrases)
    from cutsell_worker.shared_attempt_word_identity import classify_word_membership_relationship
    status = classify_word_membership_relationship(reconstructed_membership, language_membership)
    assert status == RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE
    # Serialize this pairwise shape through the same bounded row builder.
    candidate_row = candidate_take_identity_diagnostic_row(
        clip_id="clipCombined", source_asset_id="s1", source_start=combined.start, source_end=combined.end,
        membership=reconstructed_membership,
    )
    attempt_row = language_attempt_identity_diagnostic_row(
        attempt_id=attempts[0].attempt_id, source_asset_id="s1", source_start=attempts[0].source_start,
        source_end=attempts[0].source_end, membership=language_membership,
    )
    assert candidate_row["word_index_count"] > attempt_row["word_index_count"]


def test_04_partial_overlap_serialized_correctly():
    from cutsell_worker.shared_attempt_word_identity import (
        WordMembership, WORD_IDENTITY_AVAILABLE, classify_word_membership_relationship,
    )
    reconstructed = WordMembership(source_asset_id="s1", entity_id="clipX", word_indices=(0, 1, 2, 3), identity_status=WORD_IDENTITY_AVAILABLE)
    language = WordMembership(source_asset_id="s1", entity_id="latt_x", word_indices=(2, 3, 4, 5), identity_status=WORD_IDENTITY_AVAILABLE)
    status = classify_word_membership_relationship(reconstructed, language)
    assert status == RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP
    assert status not in AUTHORITATIVE_RELATIONSHIP_STATUSES


def test_05_disjoint_serialized_correctly():
    from cutsell_worker.shared_attempt_word_identity import (
        WordMembership, WORD_IDENTITY_AVAILABLE, classify_word_membership_relationship,
    )
    reconstructed = WordMembership(source_asset_id="s1", entity_id="clipX", word_indices=(0, 1), identity_status=WORD_IDENTITY_AVAILABLE)
    language = WordMembership(source_asset_id="s1", entity_id="latt_x", word_indices=(10, 11), identity_status=WORD_IDENTITY_AVAILABLE)
    status = classify_word_membership_relationship(reconstructed, language)
    assert status == RELATIONSHIP_DISJOINT
    assert status not in AUTHORITATIVE_RELATIONSHIP_STATUSES


# ===========================================================================
# 6: cross-source isolation.
# ===========================================================================
def test_06_cross_source_isolation():
    lwords_a, phrases_a, utt_a, attempts_a, props_a, take1, _ = _two_clip_fixture("srcA")
    w3, _ = _words("third real chunk of many spoken words", 0.0)
    lwords_b, phrases_b, utt_b, attempts_b, props_b = _real_spine_for_source("srcB", w3)
    take3 = _take("clipC", "srcB", w3)

    matches_a = _build_matches((take1,), lwords_a, attempts_a, utt_a, phrases_a)
    matches_b = _build_matches((take3,), lwords_b, attempts_b, utt_b, phrases_b)
    for m in matches_a:
        assert m.source_asset_id == "srcA"
    for m in matches_b:
        assert m.source_asset_id == "srcB"
    # Rows built per-source stay scoped -- never mixing srcA attempts
    # into srcB's own row.
    rows_a = identity_observability_rows_for_source(
        matches=matches_a, takes=(take1,), attempts_by_id={a.attempt_id: a for a in attempts_a},
    )
    assert rows_a["clipA"]["source_asset_id"] == "srcA"
    for lang_row in rows_a["clipA"]["language_attempts"]:
        assert lang_row["source_asset_id"] == "srcA"


# ===========================================================================
# 7: authoritative boolean mirrors canonical status set (never redefined).
# ===========================================================================
def test_07_authoritative_boolean_mirrors_canonical_set():
    source = inspect.getsource(observability_module)
    # Must import the frozenset, never redefine its members.
    assert "from .shared_attempt_word_identity import" in source
    assert "AUTHORITATIVE_RELATIONSHIP_STATUSES" in source
    assert '"EXACT_SAME_MEMBERSHIP"' not in source  # no re-literalized status strings
    assert '"ONE_RECONSTRUCTED_TO_MULTIPLE' not in source


# ===========================================================================
# 8: no relationship authority changed (shared_attempt_word_identity.py
# itself has zero diff -- this module only imports from it).
# ===========================================================================
def test_08_no_relationship_authority_changed():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "cutsell_worker/shared_attempt_word_identity.py"],
        capture_output=True, text=True, cwd=__file__.rsplit("/tests/", 1)[0],
    )
    assert result.stdout.strip() == "", f"shared_attempt_word_identity.py has an unexpected diff: {result.stdout}"


# ===========================================================================
# 9-10: exact_match map + proposition ids serialized correctly.
# ===========================================================================
def test_09_exact_match_map_presence_serialized_correctly():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    matches = _build_matches((take1, take2), lwords, attempts, utterances_by_id, phrases)
    prop_ids = proposition_candidate_ids_by_attempt_id_for(props)
    exact_match_by_clip_id = {
        t.clip_id: m for t, m in zip((take1, take2), matches)
        if m.relationship_status in AUTHORITATIVE_RELATIONSHIP_STATUSES
    }
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take1, take2),
        attempts_by_id={a.attempt_id: a for a in attempts},
        proposition_candidate_ids_by_attempt_id=prop_ids, exact_match_by_clip_id=exact_match_by_clip_id,
    )
    assert rows["clipA"]["exact_match_by_clip_id_present"] is True
    assert rows["clipB"]["exact_match_by_clip_id_present"] is True
    assert rows["clipA"]["exact_match_attempt_ids"]


def test_10_proposition_ids_serialized_correctly():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    prop_ids_by_attempt = proposition_candidate_ids_by_attempt_id_for(props)
    for attempt in attempts:
        row = language_attempt_identity_diagnostic_row(
            attempt_id=attempt.attempt_id, source_asset_id=attempt.source_asset_id,
            source_start=attempt.source_start, source_end=attempt.source_end,
            membership=build_language_attempt_word_membership(attempt, utterances_by_id, phrases),
            proposition_candidate_ids=prop_ids_by_attempt.get(attempt.attempt_id, ()),
        )
        assert row["proposition_candidate_ids"] == list(prop_ids_by_attempt.get(attempt.attempt_id, ()))


# ===========================================================================
# 11: lost-atom provenance correlation.
# ===========================================================================
def test_11_lost_atom_provenance_correlation():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    matches = _build_matches((take1,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take1,), attempts_by_id={a.attempt_id: a for a in attempts},
    )
    lost_rows = [{"lost_atom_provenance_id": "prov_1", "clip_id": "clipA", "text": "unused"}]
    correlated = lost_atom_identity_correlation(lost_rows, rows)
    assert len(correlated) == 1
    assert correlated[0]["lost_atom_provenance_id"] == "prov_1"
    assert correlated[0]["clip_id"] == "clipA"
    assert correlated[0]["identity"]["clip_id"] == "clipA"


def test_11b_lost_atom_correlation_omits_unmatched_clip():
    lost_rows = [{"lost_atom_provenance_id": "prov_2", "clip_id": "clip_never_computed"}]
    correlated = lost_atom_identity_correlation(lost_rows, {})
    assert correlated == []


# ===========================================================================
# 12-13: multiple attempts same source / multiple propositions.
# ===========================================================================
def test_12_multiple_attempts_same_source():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    assert len(attempts) == 2
    matches = _build_matches((take1, take2), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take1, take2),
        attempts_by_id={a.attempt_id: a for a in attempts},
    )
    assert len(rows) == 2
    assert rows["clipA"]["language_attempts"][0]["attempt_id"] != rows["clipB"]["language_attempts"][0]["attempt_id"]


def test_13_multiple_propositions_where_supported():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    assert len(props) == 2  # one PropositionCandidate per real LanguageAttempt (D-169's own V1 design)
    prop_ids_by_attempt = proposition_candidate_ids_by_attempt_id_for(props)
    assert len(set().union(*prop_ids_by_attempt.values())) == 2


# ===========================================================================
# 14-15: fail-open on empty/missing data.
# ===========================================================================
def test_14_empty_word_indices_fail_closed():
    from cutsell_worker.shared_attempt_word_identity import WORD_IDENTITY_MISSING, WordMembership
    membership = WordMembership(source_asset_id="s1", entity_id="clipEmpty", word_indices=(), identity_status=WORD_IDENTITY_MISSING)
    row = candidate_take_identity_diagnostic_row(
        clip_id="clipEmpty", source_asset_id="s1", source_start=0.0, source_end=1.0, membership=membership,
    )
    assert row["word_index_count"] == 0
    assert row["word_index_min"] is None
    assert row["word_index_max"] is None
    assert row["identity_status"] == WORD_IDENTITY_MISSING


def test_15_missing_attempt_data_fail_closed():
    lwords, phrases, utterances_by_id, attempts, props, take1, _ = _two_clip_fixture()
    matches = _build_matches((take1,), lwords, attempts, utterances_by_id, phrases)
    # D-237L: `takes` is now REQUIRED to be the real, positionally-aligned
    # sequence (the function no longer does a broken clip_id lookup) --
    # the take itself is always available at the real call site. This
    # test now targets the REMAINING fail-open surface: an EMPTY
    # attempts_by_id map -- the row must still be produced, with the
    # candidate side correctly populated from the real `take1` and only
    # the LANGUAGE-attempt side failing open to None, never raising.
    rows = identity_observability_rows_for_source(matches=matches, takes=(take1,), attempts_by_id={})
    assert "clipA" in rows
    assert rows["clipA"]["candidate_take"]["source_start"] == take1.start
    assert rows["clipA"]["language_attempts"][0]["source_start"] is None


def test_15b_no_row_when_takes_sequence_shorter_than_matches():
    # D-237L: `takes` must be positionally aligned with `matches` -- an
    # empty/short `takes` sequence simply yields FEWER rows (zip's own
    # truncation), never a crash and never a row keyed by anything other
    # than a real `take.clip_id`.
    lwords, phrases, utterances_by_id, attempts, props, take1, _ = _two_clip_fixture()
    matches = _build_matches((take1,), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(matches=matches, takes=(), attempts_by_id={})
    assert rows == {}


# ===========================================================================
# 16: deterministic diagnostics.
# ===========================================================================
def test_16_deterministic_diagnostics():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    matches1 = _build_matches((take1, take2), lwords, attempts, utterances_by_id, phrases)
    matches2 = _build_matches((take1, take2), lwords, attempts, utterances_by_id, phrases)
    rows1 = identity_observability_rows_for_source(
        matches=matches1, takes=(take1, take2),
        attempts_by_id={a.attempt_id: a for a in attempts},
    )
    rows2 = identity_observability_rows_for_source(
        matches=matches2, takes=(take1, take2),
        attempts_by_id={a.attempt_id: a for a in attempts},
    )
    assert rows1 == rows2


# ===========================================================================
# 17-18: bounded output, no transcript dump.
# ===========================================================================
def test_17_bounded_diagnostics_summary():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    matches = _build_matches((take1, take2), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take1, take2),
        attempts_by_id={a.attempt_id: a for a in attempts},
    )
    summary = exact_identity_observability_diagnostics(rows.values())
    assert summary["row_count"] == 2
    assert summary["authoritative_count"] == 2
    assert set(summary["relationship_status_counts"]) == {RELATIONSHIP_EXACT_SAME_MEMBERSHIP}


def test_18_no_transcript_dump():
    lwords, phrases, utterances_by_id, attempts, props, take1, take2 = _two_clip_fixture()
    matches = _build_matches((take1, take2), lwords, attempts, utterances_by_id, phrases)
    rows = identity_observability_rows_for_source(
        matches=matches, takes=(take1, take2),
        attempts_by_id={a.attempt_id: a for a in attempts},
    )
    import json
    blob = json.dumps(rows)
    for banned in ("first", "second", "chunk", "spoken", "words"):  # the fixture's own text tokens
        assert banned not in blob


# ===========================================================================
# 19-25: explicit negative-authorization proofs (import-line-scoped, per
# the established D-235P/D-236 "no live module imports this gate" style
# -- avoids false positives on this module's own docstring prose).
# ===========================================================================
def _import_lines(module) -> str:
    lines = inspect.getsource(module).splitlines()
    return "\n".join(line for line in lines if line.strip().startswith(("import ", "from ")))


def test_19_no_provider_call():
    imports = _import_lines(observability_module)
    for banned in ("openai", "gemini", "anthropic", "requests", "httpx"):
        assert banned not in imports.lower()


def test_20_no_raw_or_workflow_dispatch():
    imports = _import_lines(observability_module)
    for banned in ("modal", "runpod"):
        assert banned not in imports.lower()
    assert "workflow_dispatch" not in inspect.getsource(observability_module)


def test_21_no_new_threshold_constant():
    source = inspect.getsource(observability_module)
    assert "DEFAULT_SPLIT_GAP_SEC" not in source
    assert "_BOUNDARY_MATCH_TOLERANCE_SEC" not in source
    assert "_MIN_SHARED_FOR_SAME_PROPOSITION" not in source


def test_22_no_identity_policy_mutation():
    imports = _import_lines(observability_module)
    # Never imports a WRITE surface for AUTHORITATIVE_RELATIONSHIP_STATUSES
    # -- only ever reads it (a frozenset import, never a redefinition).
    source = inspect.getsource(observability_module)
    assert "AUTHORITATIVE_RELATIONSHIP_STATUSES =" not in source
    assert "AUTHORITATIVE_RELATIONSHIP_STATUSES.add" not in source
    assert "AUTHORITATIVE_RELATIONSHIP_STATUSES |" not in source


def test_23_no_freeze_materiality_repair_mutation():
    imports = _import_lines(observability_module)
    for banned in (
        "lost_semantic_atom_freeze_authority", "lost_atom_repair_suppression", "repair_loop",
    ):
        assert banned not in imports
    # complete_lost_semantic_atom_materiality is read-only referenced in
    # the module docstring's prose (explaining D-235Q's own consumer
    # need) but never imported -- confirm no import line either.
    assert "from .complete_lost_semantic_atom_materiality" not in imports


def test_24_no_p1_p2_mutation():
    imports = _import_lines(observability_module)
    for banned in ("editorial_moment_sequence_integration", "whole_video_editorial_reasoning"):
        assert banned not in imports


def test_25_no_pacing_audio_join_mutation():
    imports = _import_lines(observability_module)
    for banned in ("pacing_v2", "dialogue_pacing_transition", "audio_join_treatment"):
        assert banned not in imports


# ===========================================================================
# Wiring-level tests: pipeline.py / final_story_coherence_validation.py.
# ===========================================================================
def test_26_final_story_coherence_validation_wiring_present():
    import cutsell_worker.final_story_coherence_validation as fscv_module
    source = inspect.getsource(fscv_module)
    assert "identity_observability_by_clip_id" in source
    assert "lost_atom_identity_observability" in source
    assert "lost_atom_identity_correlation" in source


def test_27_pipeline_wiring_present():
    import cutsell_worker.pipeline as pipeline_module
    source = inspect.getsource(pipeline_module)
    assert "all_identity_matches_by_clip_id" in source
    assert "identity_observability_by_clip_id" in source
    assert "identity_observability_rows_for_source" in source


def test_28_universal_clean_cut_wiring_present():
    import cutsell_worker.universal_clean_cut as ucc_module
    source = inspect.getsource(ucc_module)
    assert "_identity_observability_by_clip_id" in source


def test_29_backward_compatible_default_none():
    """Every new kwarg defaults to None/{} -- omitting it entirely must
    reproduce byte-identical pre-D-237G behavior."""
    from cutsell_worker.final_story_coherence_validation import (
        _identity_observability_for_lost_atoms,
    )
    assert _identity_observability_for_lost_atoms([], None) == []
    assert _identity_observability_for_lost_atoms([{"lost_atom_provenance_id": "p1", "clip_id": "c1"}], None) == []


def test_30_identity_observability_helper_fails_open_on_empty_rows():
    from cutsell_worker.final_story_coherence_validation import _identity_observability_for_lost_atoms
    assert _identity_observability_for_lost_atoms([], {"c1": {"clip_id": "c1"}}) == []
