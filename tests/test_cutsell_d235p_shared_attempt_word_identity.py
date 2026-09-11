"""D-235P: SHARED ATTEMPT/PROPOSITION IDENTITY SEAM -- CANONICAL
WORD-MEMBERSHIP IMPLEMENTATION, OFFLINE ONLY.

Covers the task's own required fixture matrix (47 items) plus the
structural no-fuzzy-text/no-timestamp-authority/no-provider/no-live-
wiring proofs, mirroring D-235L/M/N/O's own established test style.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.language_spine import adapt_words_to_language_words, segment_language_phrases
from cutsell_worker.language_utterance_attempt import (
    build_language_attempts,
    segment_language_utterances,
)
import cutsell_worker.shared_attempt_word_identity as said


PROD_PATH = "cutsell_worker/shared_attempt_word_identity.py"
CONTRACTS_PATH = "cutsell_worker/contracts.py"


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _code_without_docstrings(path: str) -> str:
    """Structural "forbidden token" checks must assert about actual CODE,
    never about doc text that EXPLAINS an absence -- the exact self-
    reference pitfall D-235N/O's own test suites hit and fixed. AST-based
    (never a fragile string-prefix strip) so it strips every module/class/
    function docstring, not just the leading module one."""
    text = _read(path)
    tree = ast.parse(text)
    docstring_lines: set[int] = set()

    def _mark(node) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
            if isinstance(first.value.value, str):
                end = first.end_lineno or first.lineno
                docstring_lines.update(range(first.lineno, end + 1))

    _mark(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _mark(node)

    lines = text.splitlines()
    return "\n".join(line for i, line in enumerate(lines, start=1) if i not in docstring_lines)


def _wm(source_asset_id="s1", entity_id="e1", indices=(0, 1, 2), status=said.WORD_IDENTITY_AVAILABLE):
    return said.WordMembership(
        source_asset_id=source_asset_id, entity_id=entity_id,
        word_indices=tuple(indices), identity_status=status,
    )


def _word(text, start, end, confidence=0.9):
    return Word(text, start, end, confidence)


def _candidate(clip_id, source_asset_id, words, attempt_id=None):
    start = min(w.start for w in words)
    end = max(w.end for w in words)
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=start, end=end, text=" ".join(w.text for w in words),
        words=tuple(words), attempt_id=attempt_id,
    )


# ---------------------------------------------------------------------------
# 1. Canonical word ordering deterministic.
# ---------------------------------------------------------------------------
class TestCanonicalWordOrdering:
    def test_01_canonical_word_ordering_deterministic(self):
        words = [_word("b", 1.0, 1.5), _word("a", 0.0, 0.5)]
        lwords1 = adapt_words_to_language_words("s1", words)
        lwords2 = adapt_words_to_language_words("s1", list(reversed(words)))
        assert [w.text_raw for w in lwords1] == ["a", "b"]
        assert [w.text_raw for w in lwords2] == ["a", "b"]
        assert [w.word_index for w in lwords1] == [0, 1]


# ---------------------------------------------------------------------------
# 2/3. Same-source exact membership, 1->1.
# ---------------------------------------------------------------------------
class TestSameSourceExactOneToOne:
    def test_02_03_exact_same_membership_one_to_one(self):
        words = [_word("hello", 0.0, 0.4), _word("world", 0.5, 0.9)]
        lwords = adapt_words_to_language_words("s1", words)
        candidate = _candidate("c1", "s1", words, attempt_id="att1")
        rm = said.build_reconstructed_attempt_word_membership(candidate, lwords)
        lm = _wm(source_asset_id="s1", entity_id="latt1", indices=(0, 1))
        match = said.match_reconstructed_attempt_against_language_attempts(rm, (lm,))
        assert match.relationship_status == said.RELATIONSHIP_EXACT_SAME_MEMBERSHIP
        assert match.language_attempt_ids == ("latt1",)
        assert match.exact_shared_word_count == 2


# ---------------------------------------------------------------------------
# 4/5. Containment (both directions).
# ---------------------------------------------------------------------------
class TestContainment:
    def test_04_reconstructed_contains_language(self):
        r = _wm(indices=(0, 1, 2, 3))
        l = _wm(entity_id="l1", indices=(1, 2))
        assert said.classify_word_membership_relationship(r, l) == said.RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE

    def test_05_language_contains_reconstructed(self):
        r = _wm(indices=(1, 2))
        l = _wm(entity_id="l1", indices=(0, 1, 2, 3))
        assert said.classify_word_membership_relationship(r, l) == said.RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED

    def test_containment_never_promoted_to_authoritative_match(self):
        r = _wm(indices=(0, 1, 2, 3))
        l = _wm(entity_id="l1", indices=(1, 2))
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        assert match.relationship_status == said.RELATIONSHIP_EXACT_RECONSTRUCTED_CONTAINS_LANGUAGE
        assert match.relationship_status not in said.AUTHORITATIVE_RELATIONSHIP_STATUSES
        assert said.exact_proposition_candidate_ids_for_match(match, {"l1": ("prop_x",)}) == ()


# ---------------------------------------------------------------------------
# 6/7. Exact 1->N partition / N->1.
# ---------------------------------------------------------------------------
class TestPartition:
    def test_06_exact_one_to_n_partition(self):
        r = _wm(indices=tuple(range(10, 31)))  # 10..30
        a = _wm(entity_id="A", indices=tuple(range(10, 19)))  # 10..18
        b = _wm(entity_id="B", indices=tuple(range(19, 31)))  # 19..30
        match = said.match_reconstructed_attempt_against_language_attempts(r, (a, b))
        assert match.relationship_status == said.RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION
        assert match.language_attempt_ids == ("A", "B")
        assert match.relationship_status in said.AUTHORITATIVE_RELATIONSHIP_STATUSES

    def test_06b_partition_is_never_called_ambiguous_merely_for_being_two_attempts(self):
        # Same fixture as the directive's own worked example -- the mere
        # presence of two overlapping LanguageAttempts must not itself
        # produce AMBIGUOUS when they cleanly, exactly partition the
        # reconstructed attempt's own word set.
        r = _wm(indices=tuple(range(10, 31)))
        a = _wm(entity_id="A", indices=tuple(range(10, 19)))
        b = _wm(entity_id="B", indices=tuple(range(19, 31)))
        match = said.match_reconstructed_attempt_against_language_attempts(r, (a, b))
        assert match.relationship_status != said.RELATIONSHIP_AMBIGUOUS

    def test_07_n_to_one_structurally_possible_not_forced_to_1to1(self):
        # Two reconstructed attempts, each exactly matching a DIFFERENT
        # half of one LanguageAttempt's own word set -- each individually
        # classifies as EXACT_RECONSTRUCTED_CONTAINS... no: each reconstructed
        # is a proper subset of the language attempt -> EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED
        # for each -- never forced into a false 1:1 EXACT_SAME_MEMBERSHIP.
        language = _wm(entity_id="L", indices=tuple(range(0, 20)))
        recon_1 = _wm(entity_id="r1", indices=tuple(range(0, 10)))
        recon_2 = _wm(entity_id="r2", indices=tuple(range(10, 20)))
        match1 = said.match_reconstructed_attempt_against_language_attempts(recon_1, (language,))
        match2 = said.match_reconstructed_attempt_against_language_attempts(recon_2, (language,))
        assert match1.relationship_status == said.RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED
        assert match2.relationship_status == said.RELATIONSHIP_EXACT_LANGUAGE_CONTAINS_RECONSTRUCTED
        assert match1.language_attempt_ids == match2.language_attempt_ids == ("L",)


# ---------------------------------------------------------------------------
# 8. Partial overlap does NOT become exact.
# ---------------------------------------------------------------------------
class TestPartialOverlap:
    def test_08_partial_overlap_never_promoted_to_exact(self):
        r = _wm(indices=(0, 1, 2, 3))
        l = _wm(entity_id="l1", indices=(2, 3, 4, 5))
        rel = said.classify_word_membership_relationship(r, l)
        assert rel == said.RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        assert match.relationship_status == said.RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP
        assert match.relationship_status not in said.AUTHORITATIVE_RELATIONSHIP_STATUSES
        assert said.exact_proposition_candidate_ids_for_match(match, {"l1": ("prop_y",)}) == ()

    def test_08b_partial_overlap_reports_all_overlapping_ids_never_a_single_winner(self):
        r = _wm(indices=(0, 1, 2, 3, 4, 5))
        a = _wm(entity_id="A", indices=(0, 1))
        b = _wm(entity_id="B", indices=(4, 5, 6, 7))  # overlaps on 4,5 but not a clean partition (leaves 2,3 uncovered)
        match = said.match_reconstructed_attempt_against_language_attempts(r, (a, b))
        assert match.relationship_status == said.RELATIONSHIP_EXACT_PARTIAL_WORD_MEMBERSHIP_OVERLAP
        assert set(match.language_attempt_ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# 9. Disjoint membership.
# ---------------------------------------------------------------------------
class TestDisjoint:
    def test_09_disjoint_membership(self):
        r = _wm(indices=(0, 1, 2))
        l = _wm(entity_id="l1", indices=(10, 11, 12))
        assert said.classify_word_membership_relationship(r, l) == said.RELATIONSHIP_DISJOINT
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        assert match.relationship_status == said.RELATIONSHIP_DISJOINT
        assert match.language_attempt_ids == ()


# ---------------------------------------------------------------------------
# 10. Source mismatch.
# ---------------------------------------------------------------------------
class TestSourceMismatch:
    def test_10_source_mismatch(self):
        r = _wm(source_asset_id="s1", indices=(0, 1, 2))
        l = _wm(source_asset_id="s2", entity_id="l1", indices=(0, 1, 2))
        assert said.classify_word_membership_relationship(r, l) == said.RELATIONSHIP_SOURCE_MISMATCH
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        assert match.relationship_status == said.RELATIONSHIP_SOURCE_MISMATCH


# ---------------------------------------------------------------------------
# 11/12. Missing word identity (reconstructed side / language side).
# ---------------------------------------------------------------------------
class TestMissingWordIdentity:
    def test_11_missing_reconstructed_word_identity(self):
        candidate = _candidate("c1", "s1", [_word("hi", 0.0, 0.2)])
        rm = said.build_reconstructed_attempt_word_membership(candidate, ())  # no canonical evidence at all
        assert rm.identity_status == said.WORD_IDENTITY_MISSING
        lm = _wm(entity_id="l1", indices=(0,))
        match = said.match_reconstructed_attempt_against_language_attempts(rm, (lm,))
        assert match.relationship_status == said.RELATIONSHIP_MISSING_WORD_IDENTITY

    def test_12_missing_language_word_identity(self):
        r = _wm(indices=(0, 1))
        l = _wm(entity_id="l1", indices=(), status=said.WORD_IDENTITY_MISSING)
        assert said.classify_word_membership_relationship(r, l) == said.RELATIONSHIP_MISSING_WORD_IDENTITY


# ---------------------------------------------------------------------------
# 13/14/15/16. Retry/correction/continuation/false-start separation via
# distinct canonical word positions.
# ---------------------------------------------------------------------------
class TestRetryCorrectionContinuationSafety:
    def test_13_repeated_identical_text_different_positions(self):
        words_a = [_word("no", 0.0, 0.2), _word("gracias", 0.2, 0.6)]
        words_b = [_word("no", 5.0, 5.2), _word("gracias", 5.2, 5.6)]
        canonical = adapt_words_to_language_words("s1", words_a + words_b)
        ca = _candidate("ca", "s1", words_a, attempt_id="att_a")
        cb = _candidate("cb", "s1", words_b, attempt_id="att_b")
        rm_a = said.build_reconstructed_attempt_word_membership(ca, canonical)
        rm_b = said.build_reconstructed_attempt_word_membership(cb, canonical)
        assert rm_a.word_indices == (0, 1)
        assert rm_b.word_indices == (2, 3)
        assert rm_a.word_indices != rm_b.word_indices

    def test_14_retry_separation_by_canonical_index(self):
        words_a = [_word("quiero", 0.0, 0.4)]
        words_b = [_word("quiero", 3.0, 3.4)]  # a retry of the same word, later
        canonical = adapt_words_to_language_words("s1", words_a + words_b)
        ca = _candidate("ca", "s1", words_a, attempt_id="att_a")
        cb = _candidate("cb", "s1", words_b, attempt_id="att_b")
        rm_a = said.build_reconstructed_attempt_word_membership(ca, canonical)
        rm_b = said.build_reconstructed_attempt_word_membership(cb, canonical)
        rel = said.classify_word_membership_relationship(rm_a, rm_b)
        assert rel == said.RELATIONSHIP_DISJOINT

    def test_15_correction_separation_overlapping_words_different_positions(self):
        # "Correction attempts with overlapping words but different source
        # positions remain physically distinct" -- same TEXT, disjoint
        # canonical word-index sets.
        words_a = [_word("cien", 1.0, 1.3), _word("dolares", 1.3, 1.7)]
        words_b = [_word("cien", 4.0, 4.3), _word("veinte", 4.3, 4.6), _word("dolares", 4.6, 5.0)]
        canonical = adapt_words_to_language_words("s1", words_a + words_b)
        ca = _candidate("ca", "s1", words_a, attempt_id="att_a")
        cb = _candidate("cb", "s1", words_b, attempt_id="att_b")
        rm_a = said.build_reconstructed_attempt_word_membership(ca, canonical)
        rm_b = said.build_reconstructed_attempt_word_membership(cb, canonical)
        assert frozenset(rm_a.word_indices).isdisjoint(frozenset(rm_b.word_indices))

    def test_16_continuation_not_collapsed_on_shared_text(self):
        # A continuation ("...and" + "then...") never collapses identity
        # merely because it shares no text at all with anything else --
        # membership stays whatever the canonical indices actually are.
        words = [_word("and", 0.0, 0.2), _word("then", 1.0, 1.3)]
        canonical = adapt_words_to_language_words("s1", words)
        candidate = _candidate("c1", "s1", words, attempt_id="att1")
        rm = said.build_reconstructed_attempt_word_membership(candidate, canonical)
        assert rm.word_indices == (0, 1)


# ---------------------------------------------------------------------------
# 17. False start / 18. Recording-process speech retain own physical
# word membership (never merged into clean-delivery identity).
# ---------------------------------------------------------------------------
class TestFalseStartAndRecordingProcess:
    def test_17_18_false_start_and_recording_process_retain_own_membership(self):
        false_start_words = [_word("um", 0.0, 0.2), _word("so", 0.2, 0.4)]
        clean_words = [_word("the", 1.0, 1.2), _word("product", 1.2, 1.6)]
        canonical = adapt_words_to_language_words("s1", false_start_words + clean_words)
        false_start = _candidate("fs", "s1", false_start_words, attempt_id="att_fs")
        clean = _candidate("cl", "s1", clean_words, attempt_id="att_clean")
        rm_fs = said.build_reconstructed_attempt_word_membership(false_start, canonical)
        rm_clean = said.build_reconstructed_attempt_word_membership(clean, canonical)
        assert frozenset(rm_fs.word_indices).isdisjoint(frozenset(rm_clean.word_indices))
        assert rm_fs.word_indices == (0, 1)
        assert rm_clean.word_indices == (2, 3)


# ---------------------------------------------------------------------------
# 19/20/21. Multilingual safety (Spanish/English/Spanglish) -- pure value
# matching, no language branch anywhere.
# ---------------------------------------------------------------------------
class TestMultilingualSafety:
    @pytest.mark.parametrize("text_words", [
        [("hola", 0.0, 0.3), ("mundo", 0.3, 0.6)],           # Spanish
        [("hello", 0.0, 0.3), ("world", 0.3, 0.6)],           # English
        [("hola", 0.0, 0.3), ("world", 0.3, 0.6)],            # Spanglish
    ])
    def test_19_20_21_multilingual_membership_matching(self, text_words):
        words = [_word(t, s, e) for t, s, e in text_words]
        canonical = adapt_words_to_language_words("s1", words)
        candidate = _candidate("c1", "s1", words, attempt_id="att1")
        rm = said.build_reconstructed_attempt_word_membership(candidate, canonical)
        assert rm.identity_status == said.WORD_IDENTITY_AVAILABLE
        assert rm.word_indices == (0, 1)

    def test_no_language_branch_in_module(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ('lang == "es"', 'lang == "en"', 'language == "es"', 'language == "en"'):
            assert needle not in content


# ---------------------------------------------------------------------------
# 22/23/24. Candidate-order / dict-order independence, deterministic repeat.
# ---------------------------------------------------------------------------
class TestOrderIndependence:
    def test_22_candidate_order_independence(self):
        words = [_word("a", 0.0, 0.2), _word("b", 0.2, 0.4), _word("c", 0.4, 0.6)]
        canonical1 = adapt_words_to_language_words("s1", words)
        canonical2 = adapt_words_to_language_words("s1", list(reversed(words)))
        candidate = _candidate("c1", "s1", words, attempt_id="att1")
        rm1 = said.build_reconstructed_attempt_word_membership(candidate, canonical1)
        rm2 = said.build_reconstructed_attempt_word_membership(candidate, canonical2)
        assert rm1.word_indices == rm2.word_indices == (0, 1, 2)

    def test_23_dict_order_independence_for_batch_matches(self):
        r = _wm(indices=(0, 1, 2))
        a = _wm(entity_id="A", indices=(0, 1, 2))
        b = _wm(entity_id="B", indices=(10, 11))
        match1 = said.match_reconstructed_attempt_against_language_attempts(r, (a, b))
        match2 = said.match_reconstructed_attempt_against_language_attempts(r, (b, a))
        assert match1.relationship_status == match2.relationship_status == said.RELATIONSHIP_EXACT_SAME_MEMBERSHIP
        assert match1.language_attempt_ids == match2.language_attempt_ids

    def test_24_deterministic_repeat(self):
        words = [_word("a", 0.0, 0.2), _word("b", 0.2, 0.4)]
        canonical = adapt_words_to_language_words("s1", words)
        candidate = _candidate("c1", "s1", words, attempt_id="att1")
        results = {said.build_reconstructed_attempt_word_membership(candidate, canonical) for _ in range(5)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 25/26/27/28. Existing IDs unchanged.
# ---------------------------------------------------------------------------
class TestExistingIdsUnchanged:
    def test_25_existing_attempt_id_unchanged(self):
        words = [_word("a", 0.0, 0.2)]
        candidate = _candidate("c1", "s1", words, attempt_id="att_original")
        canonical = adapt_words_to_language_words("s1", words)
        said.build_reconstructed_attempt_word_membership(candidate, canonical)
        assert candidate.attempt_id == "att_original"
        assert candidate.clip_id == "c1"

    def test_26_language_attempt_id_computation_untouched(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "mint_attempt_id" not in content
        assert "_attempt_id(" not in content

    def test_27_proposition_candidate_id_computation_untouched(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "_proposition_id(" not in content
        assert "hashlib" not in content

    def test_28_p1_moment_id_computation_untouched(self):
        content = _read(PROD_PATH)
        assert "editorial_moment_id" not in content
        assert "classify_editorial_moment" not in content


# ---------------------------------------------------------------------------
# 29/30. Heuristic fallback stays diagnostic-only; exact wins when present.
# ---------------------------------------------------------------------------
class TestHeuristicPrecedence:
    def test_29_heuristic_labelled_never_silently_exact(self):
        result = said.p1_identity_provenance_for_clip(
            exact_match=None, heuristic_attempt_id="latt_heuristic",
            proposition_candidate_ids_by_attempt_id={},
        )
        assert result["identity_source"] == said.IDENTITY_SOURCE_HEURISTIC_OVERLAP
        assert result["exact_proposition_candidate_ids"] == ()

    def test_30_exact_identity_takes_precedence_over_heuristic(self):
        r = _wm(indices=(0, 1))
        l = _wm(entity_id="latt_exact", indices=(0, 1))
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        result = said.p1_identity_provenance_for_clip(
            exact_match=match, heuristic_attempt_id="latt_heuristic_different",
            proposition_candidate_ids_by_attempt_id={"latt_exact": ("prop_1",)},
        )
        assert result["identity_source"] == said.IDENTITY_SOURCE_EXACT_WORD_MEMBERSHIP
        assert result["language_attempt_ids"] == ("latt_exact",)
        assert result["exact_proposition_candidate_ids"] == ("prop_1",)


# ---------------------------------------------------------------------------
# 31/32/33. Exact proposition-set extraction, multi-proposition retained
# as a set, no atom-level guessing.
# ---------------------------------------------------------------------------
class TestPropositionSetExtraction:
    def test_31_exact_proposition_set_extraction(self):
        r = _wm(indices=(0, 1))
        l = _wm(entity_id="latt1", indices=(0, 1))
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        ids = said.exact_proposition_candidate_ids_for_match(match, {"latt1": ("prop_a",)})
        assert ids == ("prop_a",)

    def test_32_multiple_propositions_retained_as_a_set(self):
        r = _wm(indices=tuple(range(0, 10)))
        a = _wm(entity_id="A", indices=tuple(range(0, 5)))
        b = _wm(entity_id="B", indices=tuple(range(5, 10)))
        match = said.match_reconstructed_attempt_against_language_attempts(r, (a, b))
        assert match.relationship_status == said.RELATIONSHIP_ONE_RECONSTRUCTED_TO_MULTIPLE_LANGUAGE_ATTEMPTS_EXACT_PARTITION
        ids = said.exact_proposition_candidate_ids_for_match(
            match, {"A": ("prop_a",), "B": ("prop_b", "prop_c")},
        )
        assert ids == ("prop_a", "prop_b", "prop_c")

    def test_33_no_atom_level_proposition_guessing(self):
        content = _code_without_docstrings(PROD_PATH)
        # No per-atom assignment logic -- the module returns whole sets only.
        assert "atom" not in content.lower()
        assert "owns" not in content.lower()


# ---------------------------------------------------------------------------
# 34. Old artifact / missing-field fail-open.
# ---------------------------------------------------------------------------
class TestOldArtifactFailOpen:
    def test_34_candidate_take_default_word_indices_is_empty_tuple(self):
        candidate = CandidateTake(
            clip_id="c1", source_asset_id="s1", source_order=0, start=0.0, end=1.0, text="hi",
        )
        assert candidate.word_indices == ()

    def test_34b_old_candidate_with_no_words_at_all_fails_open_to_missing(self):
        candidate = CandidateTake(clip_id="c1", source_asset_id="s1", source_order=0, start=0.0, end=1.0, text="hi")
        rm = said.build_reconstructed_attempt_word_membership(candidate, ())
        assert rm.identity_status == said.WORD_IDENTITY_MISSING
        assert rm.word_indices == ()


# ---------------------------------------------------------------------------
# 35/36. No provider, no RAW.
# ---------------------------------------------------------------------------
class TestNoProviderNoRaw:
    def test_35_no_provider_call(self):
        content = _read(PROD_PATH)
        for needle in ("requests.", "httpx", "openai", "google.generativeai", "genai.", "modal.", "runpod"):
            assert needle not in content

    def test_36_no_raw_no_subprocess_no_network(self):
        content = _read(PROD_PATH)
        for needle in ("subprocess", "socket.", "urllib", "boto3"):
            assert needle not in content


# ---------------------------------------------------------------------------
# 37/38/39. No Freeze/P1-grouping/P2 mutation.
# ---------------------------------------------------------------------------
class TestNoAuthorityMutation:
    def test_37_no_freeze_import_or_mutation(self):
        content = _read(PROD_PATH)
        for needle in ("selection_freeze", "SelectionFreeze", "freeze_blocked"):
            assert needle not in content

    def test_38_no_p1_grouping_import(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in (
            "build_editorial_local_groups", "editorial_moment_sequence_integration",
            "build_editorial_moments_for_source",
        ):
            assert needle not in content

    def test_39_no_p2_or_besttake_family_ordering_boundary_import(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in (
            "take_grouping", "deterministic_best_take_authority", "boundary_engine_pass",
            "dialogue_pacing_transition", "repair_loop", "hybrid_session_cleanup",
        ):
            assert needle not in content


# ---------------------------------------------------------------------------
# 40 (part). No live wiring -- module-leaf isolation tests.
# ---------------------------------------------------------------------------
class TestNoLiveWiring:
    LIVE_MODULES = (
        "cutsell_worker/pipeline.py",
        "cutsell_worker/take_segmentation.py",
        "cutsell_worker/attempt_reconstruction.py",
        "cutsell_worker/editorial_moment_sequence_integration.py",
        "cutsell_worker/language_spine_live_integration.py",
        "cutsell_worker/take_judge.py",
        "cutsell_worker/deterministic_best_take_authority.py",
        "cutsell_worker/boundary_engine_pass.py",
        "cutsell_worker/dialogue_pacing_transition.py",
        "cutsell_worker/repair_loop.py",
        "cutsell_worker/hybrid_session_cleanup.py",
        "cutsell_worker/semantic_idea_equivalence.py",
        "cutsell_worker/universal_clean_cut.py",
    )

    def test_40_no_live_module_imports_this_gate(self):
        """D-235X intentionally wires this module INTO `pipeline.py`
        (docs/CUTSELL_DECISIONS.md D-235X's own Part A -- the ONE
        production data-source owner named by that task) -- the remaining
        LIVE_MODULES stay unwired, exactly as before D-235X."""
        for path in self.LIVE_MODULES:
            if path == "cutsell_worker/pipeline.py":
                continue
            content = _read(path)
            assert "shared_attempt_word_identity" not in content, f"live wiring leaked into {path}"

        # The one intentional D-235X wiring seam.
        pipeline_content = _read("cutsell_worker/pipeline.py")
        assert "shared_attempt_word_identity" in pipeline_content


# ---------------------------------------------------------------------------
# 41. compileall (proxy: import + ast parse).
# ---------------------------------------------------------------------------
class TestCompiles:
    def test_41_module_compiles(self):
        import ast
        ast.parse(_read(PROD_PATH))
        ast.parse(_read("tests/test_cutsell_d235p_shared_attempt_word_identity.py"))


# ---------------------------------------------------------------------------
# 42/43/44/45/46. Regression: D-235M/N/O + Language Spine + P1/P2 modules
# stay unmodified in the ways that matter (import graphs, key functions
# still present, byte-identical docstrings' own claims still hold).
# ---------------------------------------------------------------------------
class TestSiblingRegression:
    def test_42_d235m_module_untouched_by_this_gate(self):
        content = _read("cutsell_worker/lost_atom_editorial_requirement_evidence.py")
        assert "shared_attempt_word_identity" not in content

    def test_43_d235n_module_untouched_by_this_gate(self):
        content = _read("cutsell_worker/lost_atom_proposition_identity_forensic.py")
        assert "shared_attempt_word_identity" not in content

    def test_44_d235o_test_file_untouched(self):
        # D-235O's own test suite (design-only self-checks) still holds:
        # this gate did not touch its two named pre-existing files. D-235S
        # added `lost_atom_reviewer_finding_provenance.py` and D-235T (both
        # SEPARATELY-authorized, later gates) added `lost_atom_repair_
        # suppression.py` -- widened here for the same reason as D-235O's
        # own test_17 (see that file's own comment).
        import os
        new_files = [
            f for f in os.listdir("cutsell_worker")
            if f.startswith("lost_atom_") or f.startswith("shared_attempt_")
        ]
        assert set(new_files) == {
            "lost_atom_editorial_requirement_evidence.py",
            "lost_atom_proposition_identity_forensic.py",
            "lost_atom_reviewer_finding_provenance.py",
            "lost_atom_repair_suppression.py",
            "shared_attempt_word_identity.py",
        }

    def test_45_language_spine_still_builds_word_index_the_same_way(self):
        content = _read("cutsell_worker/language_spine.py")
        assert "sorted(words, key=lambda word: (word.start, word.end))" in content
        assert "word_index=index" in content

    def test_46_language_spine_live_integration_bridge_still_labelled_maximum_overlap(self):
        content = _read("cutsell_worker/language_spine_live_integration.py")
        assert "MAXIMUM-OVERLAP" in content
        assert "language_attempts_by_span_id_for_source" in content


# ---------------------------------------------------------------------------
# Structural: no fuzzy text, no timestamp-threshold authority, no numeric
# threshold, no object-identity dependence.
# ---------------------------------------------------------------------------
class TestStructuralSafety:
    def test_no_fuzzy_text_matching(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("SequenceMatcher", "difflib", "fuzzy", "ratio(", "get_close_matches"):
            assert needle not in content

    def test_no_timestamp_or_iou_threshold_authority(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("overlap_ratio", "IoU", "iou", "0.5 *", "threshold", "tolerance_sec", "0.20", "0.36"):
            assert needle not in content

    def test_no_object_identity_dependence(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "id(word)" not in content
        assert " is word" not in content

    def test_relationship_classifier_uses_only_set_arithmetic(self):
        import inspect
        source = inspect.getsource(said.classify_word_membership_relationship)
        for needle in (".start", ".end", "SequenceMatcher", "%"):
            assert needle not in source


# ---------------------------------------------------------------------------
# End-to-end: real Language Spine pipeline produces a real EXACT match.
# ---------------------------------------------------------------------------
class TestEndToEndRealPipeline:
    def test_end_to_end_exact_match_through_real_language_spine(self):
        words = [_word("hello", 0.0, 0.4), _word("world.", 0.5, 0.9)]
        canonical = adapt_words_to_language_words("s1", words)
        phrases = segment_language_phrases(canonical)
        utterances = segment_language_utterances(phrases)
        attempts = build_language_attempts(utterances)
        utterances_by_id = {u.utterance_id: u for u in utterances}
        assert len(attempts) == 1

        candidate = _candidate("c1", "s1", words, attempt_id="att1")
        matches = said.build_attempt_language_identity_matches_for_source(
            reconstructed_attempts=(candidate,), canonical_words=canonical,
            language_attempts=attempts, utterances_by_id=utterances_by_id, phrases=phrases,
        )
        assert len(matches) == 1
        assert matches[0].relationship_status == said.RELATIONSHIP_EXACT_SAME_MEMBERSHIP

    def test_language_attempt_word_membership_always_contiguous(self):
        words = [_word(f"w{i}", float(i), float(i) + 0.3) for i in range(8)]
        canonical = adapt_words_to_language_words("s1", words)
        phrases = segment_language_phrases(canonical)
        utterances = segment_language_utterances(phrases)
        attempts = build_language_attempts(utterances)
        utterances_by_id = {u.utterance_id: u for u in utterances}
        for attempt in attempts:
            lm = said.build_language_attempt_word_membership(attempt, utterances_by_id, phrases)
            indices = lm.word_indices
            if indices:
                assert tuple(indices) == tuple(range(min(indices), max(indices) + 1)), (
                    "LanguageAttempt word membership must be provably contiguous"
                )


# ---------------------------------------------------------------------------
# Diagnostics shape (no transcript dump).
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_per_match_diagnostics_required_fields_present(self):
        r = _wm(indices=(0, 1))
        l = _wm(entity_id="latt1", indices=(0, 1))
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        diag = said.attempt_language_identity_match_diagnostics(match, {"latt1": ("prop_a",)})
        for key in (
            "exact_identity_available", "identity_source", "reconstructed_attempt_id",
            "language_attempt_ids", "exact_proposition_candidate_ids", "relationship_status",
            "canonical_word_count_reconstructed", "canonical_word_count_language_union",
            "missing_identity_reason",
        ):
            assert key in diag

    def test_diagnostics_no_transcript_dump(self):
        r = _wm(indices=(0, 1))
        l = _wm(entity_id="latt1", indices=(0, 1))
        match = said.match_reconstructed_attempt_against_language_attempts(r, (l,))
        diag = said.attempt_language_identity_match_diagnostics(match)
        payload = str(diag)
        assert "text_raw" not in payload

    def test_batch_diagnostics_counts_only(self):
        r1 = _wm(entity_id="r1", indices=(0, 1))
        r2 = _wm(entity_id="r2", indices=(5, 6))
        l = _wm(entity_id="latt1", indices=(0, 1))
        match1 = said.match_reconstructed_attempt_against_language_attempts(r1, (l,))
        match2 = said.match_reconstructed_attempt_against_language_attempts(r2, (l,))
        summary = said.shared_attempt_word_identity_diagnostics((match1, match2))
        assert summary["match_count"] == 2
        assert summary["exact_identity_available_count"] == 1


# ---------------------------------------------------------------------------
# Contracts.py additive field regression.
# ---------------------------------------------------------------------------
class TestContractsAdditiveField:
    def test_word_indices_field_additive_and_default_empty(self):
        content = _read(CONTRACTS_PATH)
        assert "word_indices: Tuple[int, ...] = ()" in content

    def test_live_call_sites_now_populate_word_indices_d235w(self):
        """D-235W Part B closes this GAP intentionally (canonical, source-
        scoped word-index population via `take_segmentation.py`'s own
        `_canonical_word_index_lookup`/`_word_indices_for`, carried forward
        through `_join_takes`/`_merge_attempt`/`_draft_clip` -- never fuzzy
        text, never timestamp-overlap identity, never a second ordinal
        system; see docs/CUTSELL_DECISIONS.md D-235W). This test now
        asserts the POSITIVE, intentional state that the pre-D-235W
        version of this test recorded as absent."""
        live_sites = (
            "cutsell_worker/take_segmentation.py",
            "cutsell_worker/attempt_reconstruction.py",
            "cutsell_worker/pipeline.py",
        )
        for path in live_sites:
            content = _read(path)
            assert "word_indices=" in content
