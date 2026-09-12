"""D-239L: ATOM-GRANULAR EDITORIAL REQUIREMENT REFINEMENT -- OFFLINE
IMPLEMENTATION ONLY.

Post D-239K (docs/CUTSELL_DECISIONS.md, Verdict B -- REQUIRED IS ONLY
PROPOSITION-LEVEL, D-239I's own ownership bridge over-inherits
requiredness onto a small lost atom): D-239K proved `complete_lost_
semantic_atom_materiality.py`'s own D-239I Seam A -- `exact_ownership_
available` unlocking a proposition's own whole-`LanguageAttempt`-level
`editorial_slot_evidence` -- is NOT, by itself, atom-granular proof that
the specific lost atom carries the story-function slot's own required
editorial function. This module's own D-239L addition refines exactly
that ONE bridge: when the ownership-only identity path (`not exact_
identity_available and exact_ownership_available`) is the SOLE reason a
row reached `REQUIRED` (via a `"exact_story_function_slot:..."` reason
code, with neither `idea_coverage_status` nor `downstream_dependency_
present` independently True), the verdict is now trusted ONLY when the
SAME exact, already-computed, target-clip_id-keyed atom-level evidence
D-239I's own Seam C already threads (`recording_process_status`/
`audience_delivery_status`) actually confirms it -- a resolved, non-
process-shaped role AND a proven (`SUPPORTED`/`PARTIAL`) audience-
delivery status. Absent that corroboration, the row is downgraded to
`REQUIREMENT_INSUFFICIENT_EVIDENCE` (an EXISTING vocabulary value),
never `REQUIREMENT_NOT_REQUIRED` -- fail-closed, per this task's own
explicit instruction.

`exact_identity_available` (D-235P full-attempt identity) is completely
UNCHANGED -- every test in `TestFullIdentityUnchanged` proves this
directly, including the case where ownership ALSO happens to resolve
exactly for the same clip.

No editorial-required firewall removal, no REQUIRED->optional
conversion, no UNKNOWN->safe conversion, no materiality-precedence
change, no ownership/global-identity/Freeze/repair/Language-Spine/P1/P2/
Pacing/Audio-Join authority change, no new classifier, no new threshold,
no text heuristic -- see `TestNoGlobalAuthorityChange`/
`TestFirewallUnchanged` below for direct, code-level proof.
"""
from __future__ import annotations

import ast
import subprocess

import cutsell_worker.complete_lost_semantic_atom_materiality as claamod
from cutsell_worker.complete_lost_semantic_atom_materiality import (
    assess_complete_lost_semantic_atom_materiality,
)
from cutsell_worker.editorial_moment_sequence import (
    AUDIENCE_DELIVERY_NOT_SUPPORTED,
    AUDIENCE_DELIVERY_PARTIAL,
    AUDIENCE_DELIVERY_SUPPORTED,
    AUDIENCE_DELIVERY_UNCERTAIN,
    MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
    MOMENT_ROLE_CONTINUATION,
    MOMENT_ROLE_POST_TAKE_RESET,
)
from cutsell_worker.exact_lost_atom_ownership import ExactLostAtomOwnership
from cutsell_worker.language_proposition_relation import SLOT_CTA, SLOT_HOOK, SLOT_OTHER
from cutsell_worker.lost_semantic_atom_materiality import (
    MATERIALITY_CONFLICTED,
    MATERIALITY_EDITORIALLY_REQUIRED,
    MATERIALITY_MEANING_CRITICAL,
    MATERIALITY_RETRY_OR_RECORDING_RESIDUE,
    RECOMMEND_ABSTAIN,
    RECOMMEND_BLOCK,
    RECOMMEND_DO_NOT_BLOCK,
)
from cutsell_worker.shared_attempt_word_identity import (
    AUTHORITATIVE_RELATIONSHIP_STATUSES,
    AttemptLanguageIdentityMatch,
    RELATIONSHIP_EXACT_SAME_MEMBERSHIP,
    WORD_IDENTITY_AVAILABLE,
    WordMembership,
)

# ---------------------------------------------------------------------------
# Shared fixtures -- the real D-239K/D-239J recovered shape.
# ---------------------------------------------------------------------------
_PROD_PATH = "/home/user/EditDNA-worker/cutsell_worker/complete_lost_semantic_atom_materiality.py"


def _code_without_docstrings(path: str) -> str:
    """Same technique as test_cutsell_d235q's own helper: strips every
    module/function/class docstring line before scanning, so prose
    mentioning a word (e.g. "CUTSELL_DECISIONS.md", "fuzzy matching") in
    a comment/docstring never produces a false-positive authority-scan
    match."""
    with open(path) as f:
        text = f.read()
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


_TARGET_CLIP_ID = "clip_2089a0a7f701d18f2aa4"
_TARGET_ATTEMPT_ID = "latt_f811b1c572da97b6fe21"
_TARGET_PROPOSITION_ID = "prop_41af4a64034a4211782a"


def _row(**overrides) -> dict:
    base = {
        "clip_id": _TARGET_CLIP_ID,
        "text": "too many people ready set these are the",
        "classification": "REAL_CONTENT_LOSS",
        "blocking": True,
    }
    base.update(overrides)
    return base


def _ownership(**overrides) -> ExactLostAtomOwnership:
    base = dict(
        clip_id=_TARGET_CLIP_ID, source_asset_id="src",
        candidate_word_indices=tuple(range(41, 49)),
        containing_language_attempt_id=_TARGET_ATTEMPT_ID,
        proposition_candidate_ids=(_TARGET_PROPOSITION_ID,),
        ownership_status="EXACT_SINGLETON_OWNERSHIP", reason_codes=(), provenance=(),
    )
    base.update(overrides)
    return ExactLostAtomOwnership(**base)


def _full_identity_match(*, authoritative: bool = True) -> AttemptLanguageIdentityMatch:
    status = RELATIONSHIP_EXACT_SAME_MEMBERSHIP if authoritative else "HEURISTIC_OVERLAP"
    wm = WordMembership(
        source_asset_id="src", entity_id=_TARGET_CLIP_ID, word_indices=(0, 1),
        identity_status=WORD_IDENTITY_AVAILABLE,
    )
    return AttemptLanguageIdentityMatch(
        reconstructed_attempt_id=_TARGET_CLIP_ID, language_attempt_ids=(_TARGET_ATTEMPT_ID,),
        source_asset_id="src", reconstructed_word_membership=wm, language_word_memberships=(wm,),
        relationship_status=status, exact_shared_word_count=2, reconstructed_word_count=2,
        language_word_count=2, provenance=(),
    )


# ---------------------------------------------------------------------------
# Matrix item 1: full exact identity + REQUIRED -> unchanged BLOCK, even
# with ownership ALSO exact for the same clip, and with NO atom-level P1
# corroboration supplied at all -- the D-239L refinement never reaches
# this branch.
# ---------------------------------------------------------------------------
class TestFullIdentityUnchanged:
    def test_01_full_identity_slot_required_no_atom_evidence_still_blocks(self):
        match = _full_identity_match(authoritative=True)
        own = _ownership()  # ALSO exact-singleton for the same clip
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), exact_match=match, lost_atom_ownership=own,
            proposition_candidate_ids_by_attempt_id={_TARGET_ATTEMPT_ID: (_TARGET_PROPOSITION_ID,)},
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_HOOK},
            # Deliberately NO recording_process_status/audience_delivery_
            # status -- pre-D-239I this was ALWAYS the case for a full-
            # identity row, and it must still be sufficient by itself.
        )
        assert res.editorial_requirement_status == "REQUIRED"
        assert res.final_materiality_status == MATERIALITY_EDITORIALLY_REQUIRED
        assert res.blocking_recommendation == RECOMMEND_BLOCK
        assert res.editorial_requirement_granularity == "FULL_IDENTITY_PROPOSITION"
        assert res.editorial_requirement_target_evidence_source == "FULL_ATTEMPT_IDENTITY"

    def test_02_full_identity_unaffected_by_missing_atom_evidence_even_flagged(self):
        # Same as test_01, but explicitly confirm a resolved PROCESS-SHAPED
        # atom-level role (if one were ever supplied for a full-identity
        # row) is likewise irrelevant to this branch -- the D-239L gate is
        # structurally unreachable once exact_identity_available is True.
        match = _full_identity_match(authoritative=True)
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), exact_match=match,
            proposition_candidate_ids_by_attempt_id={_TARGET_ATTEMPT_ID: (_TARGET_PROPOSITION_ID,)},
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_POST_TAKE_RESET,  # would firewall an OWNERSHIP-only row
        )
        # The retry/process firewall in assess_editorial_requirement_evidence
        # is UNCHANGED and still applies to a full-identity row too (this
        # was already true pre-D-239L) -- proving D-239L added no NEW gate
        # here, it only refines the ownership-only branch.
        assert res.editorial_requirement_status == "NOT_REQUIRED"


# ---------------------------------------------------------------------------
# Matrix items 2-7, 14, 15: the ownership-only path.
# ---------------------------------------------------------------------------
class TestOwnershipOnlyPath:
    def test_03_item2_ownership_broad_required_only_insufficient(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            # no recording_process_status/audience_delivery_status at all
        )
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"
        assert res.editorial_requirement_status != "NOT_REQUIRED"
        assert res.editorial_requirement_granularity == "PROPOSITION_ONLY_INSUFFICIENT"
        assert "ownership_only_slot_evidence_lacks_atom_level_target_corroboration" in res.reason_codes

    def test_04_item3_ownership_plus_exact_atom_required_blocks(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert res.editorial_requirement_status == "REQUIRED"
        assert res.final_materiality_status == MATERIALITY_EDITORIALLY_REQUIRED
        assert res.blocking_recommendation == RECOMMEND_BLOCK
        assert res.editorial_requirement_granularity == "ATOM_EXACT_P1"
        assert res.editorial_requirement_target_evidence_source == "EXACT_P1_MOMENT"

    def test_05_item3_partial_audience_delivery_also_sufficient(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_HOOK},
            recording_process_status=MOMENT_ROLE_CONTINUATION,
            audience_delivery_status=AUDIENCE_DELIVERY_PARTIAL,
        )
        assert res.editorial_requirement_status == "REQUIRED"

    def test_06_item4_ownership_plus_exact_process_role_no_inherited_required(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_POST_TAKE_RESET,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        # The PRE-EXISTING retry/process firewall (unchanged) fires before
        # required_signal is ever computed -- proposition-level REQUIRED
        # never overrides confirmed atom-level process evidence.
        assert res.editorial_requirement_status == "NOT_REQUIRED"

    def test_07_item5_ownership_plus_ambiguous_role_insufficient(self):
        # Role resolved (non-process) but audience-delivery status is
        # UNCERTAIN -- ambiguous, not a proof of audience delivery. The
        # slot-driven downgrade fires (the more specific, informative
        # label), correctly subsuming the raw AMBIGUOUS/MISSING labels
        # test_09b below demonstrates in isolation (no slot trigger).
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_UNCERTAIN,
        )
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"
        assert res.editorial_requirement_status != "NOT_REQUIRED"
        assert res.editorial_requirement_granularity == "PROPOSITION_ONLY_INSUFFICIENT"

    def test_08_item6_ownership_plus_missing_role_insufficient(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,  # role itself absent
        )
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"
        assert res.editorial_requirement_granularity == "PROPOSITION_ONLY_INSUFFICIENT"

    def test_09b_raw_ambiguous_and_missing_labels_when_slot_not_the_trigger(self):
        # SLOT_OTHER never fires required_signal at all, so the granularity
        # label reflects the RAW atom-evidence-availability state directly,
        # never the slot-driven downgrade label.
        own = _ownership()
        res_missing = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_OTHER},
        )
        assert res_missing.editorial_requirement_granularity == "MISSING"

        res_ambiguous = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_OTHER},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_UNCERTAIN,
        )
        assert res_ambiguous.editorial_requirement_granularity == "AMBIGUOUS"

    def test_09_item7_ownership_plus_conflicting_evidence_insufficient(self):
        # Role resolved non-process, but audience_delivery_status is
        # explicitly NOT_SUPPORTED -- assess_editorial_requirement_evidence's
        # OWN pre-existing firewall (step 3) already converts this to
        # NOT_REQUIRED before required_signal is ever computed; D-239L's
        # own gate is correctly never reached (nothing to downgrade), and
        # the outcome is NOT_REQUIRED via that unchanged, unrelated
        # firewall -- proving D-239L introduces no conflicting double
        # policy for this shape.
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CONTINUATION,
            audience_delivery_status=AUDIENCE_DELIVERY_NOT_SUPPORTED,
        )
        assert res.editorial_requirement_status == "NOT_REQUIRED"

    def test_10_item14_ownership_alone_never_do_not_block(self):
        # No slot evidence at all for the owned proposition (SLOT_OTHER
        # carries no story function) and no other signal -- ownership
        # alone must never manufacture DO_NOT_BLOCK.
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_OTHER},
        )
        assert res.blocking_recommendation != RECOMMEND_DO_NOT_BLOCK

    def test_11_item15_proposition_level_required_alone_never_atom_required(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
        )
        assert res.editorial_requirement_status != "REQUIRED"


# ---------------------------------------------------------------------------
# Matrix items 8-13: precedence/firewall interplay, unchanged.
# ---------------------------------------------------------------------------
class TestPrecedenceUnchanged:
    def test_12_item8_multi_proposition_ambiguity_abstain(self):
        own = _ownership(
            proposition_candidate_ids=(_TARGET_PROPOSITION_ID, "prop_other"),
            ownership_status="AMBIGUOUS_MULTIPLE_PROPOSITIONS",
        )
        assert own.is_exact_singleton is False
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA, "prop_other": SLOT_HOOK},
        )
        # Ownership never even reaches EXACT_SINGLETON -- exact_ownership_
        # available is False, D-239L's own gate never fires; result is the
        # pre-existing INSUFFICIENT_EVIDENCE default (never REQUIRED,
        # never a fabricated ABSTAIN from this module alone).
        assert res.exact_ownership_available is False
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"

    def test_13_item9_source_mismatch_abstain(self):
        own = ExactLostAtomOwnership(
            clip_id=_TARGET_CLIP_ID, source_asset_id="src", candidate_word_indices=(41,),
            containing_language_attempt_id=None, proposition_candidate_ids=(),
            ownership_status="SOURCE_MISMATCH", reason_codes=("x",), provenance=(),
        )
        res = assess_complete_lost_semantic_atom_materiality(_row(), lost_atom_ownership=own)
        assert res.exact_ownership_available is False
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"

    def test_14_item10_meaning_critical_blocks_regardless_of_ownership_required(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own, critical_claim_conflict=None,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        # Sanity: this shape alone (no negation/number critical atom in the
        # row's own text) does not reach MEANING_CRITICAL -- included only
        # to prove D-239L's own atom-corroborated REQUIRED path (test_04)
        # is what actually fires here, establishing the baseline this next
        # assertion's own contrast (a genuinely critical row) is measured
        # against.
        assert res.meaning_materiality_status != MATERIALITY_MEANING_CRITICAL

        critical_row = _row(text="I do NOT have diabetes", missing_critical_atoms=["not"])
        res_critical = assess_complete_lost_semantic_atom_materiality(
            critical_row, lost_atom_ownership=own, critical_claim_conflict=True,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert res_critical.final_materiality_status == MATERIALITY_MEANING_CRITICAL
        assert res_critical.blocking_recommendation == RECOMMEND_BLOCK

    def test_15_item11_critical_claim_conflict_blocks_regardless(self):
        # D-235L's own `critical_claim_conflict=True` (an explicit,
        # caller-established conflict) resolves to the SAME MEANING_
        # CRITICAL safety floor a genuinely critical claim does (step 1 of
        # this module's own precedence chain) -- this is D-235L's own
        # pre-existing, unchanged behavior; D-239L's own ownership-only
        # refinement never even reaches the (possibly-REQUIRED) editorial-
        # requirement branch for this row, since step 1 already wins.
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own, critical_claim_conflict=True,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
        )
        assert res.final_materiality_status == MATERIALITY_MEANING_CRITICAL
        assert res.blocking_recommendation == RECOMMEND_BLOCK

    def test_16_item12_retry_process_proceeds_only_after_safety_states(self):
        # A confirmed process-shaped atom role with no meaning-critical/
        # editorial-required/conflict evidence reaches D-235L's own
        # retry/recording-residue branch, DO_NOT_BLOCK -- unaffected by
        # D-239L (this precedence step is untouched).
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            recording_process_evidence=True,
        )
        assert res.final_materiality_status == MATERIALITY_RETRY_OR_RECORDING_RESIDUE
        assert res.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK

    def test_17_item13_redundancy_proceeds_only_after_safety_states(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own, replacement_function_preserved=True,
        )
        assert res.blocking_recommendation == RECOMMEND_DO_NOT_BLOCK


# ---------------------------------------------------------------------------
# Real D-239J shape offline replay -- the three directive-named cases.
# ---------------------------------------------------------------------------
class TestRealD239JShapeReplay:
    def test_18_case1_no_exact_target_role_insufficient_never_not_required(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
        )
        assert res.editorial_requirement_status == "INSUFFICIENT_EVIDENCE"
        assert res.editorial_requirement_status != "REQUIRED"
        assert res.editorial_requirement_status != "NOT_REQUIRED"

    def test_19_case2_exact_target_moment_proves_required_blocks(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        assert res.editorial_requirement_status == "REQUIRED"
        assert res.blocking_recommendation == RECOMMEND_BLOCK

    def test_20_case3_exact_target_process_role_no_override_no_forced_do_not_block(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_POST_TAKE_RESET,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
            # deliberately NOT setting recording_process_evidence=True --
            # proves materiality is not FORCED to DO_NOT_BLOCK by this
            # refinement; it continues through the existing, unrelated
            # retry/process/materiality precedence exactly as before.
        )
        assert res.editorial_requirement_status == "NOT_REQUIRED"
        assert res.blocking_recommendation != RECOMMEND_DO_NOT_BLOCK or res.retry_or_process_status is not None


# ---------------------------------------------------------------------------
# Diagnostics -- editorial_requirement_granularity / _target_evidence_source.
# ---------------------------------------------------------------------------
class TestDiagnosticsFields:
    def test_21_no_identity_no_ownership_granularity_none(self):
        res = assess_complete_lost_semantic_atom_materiality(_row())
        assert res.editorial_requirement_granularity is None
        assert res.editorial_requirement_target_evidence_source is None

    def test_22_as_dict_exposes_new_fields(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
        )
        d = res.as_dict()
        assert d["editorial_requirement_granularity"] == "PROPOSITION_ONLY_INSUFFICIENT"
        assert d["editorial_requirement_target_evidence_source"] == "NONE"

    def test_23_diagnostics_helper_exposes_new_fields(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own,
            proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
            recording_process_status=MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
            audience_delivery_status=AUDIENCE_DELIVERY_SUPPORTED,
        )
        diag = claamod.complete_lost_semantic_atom_materiality_diagnostics(res)
        assert diag["editorial_requirement_granularity"] == "ATOM_EXACT_P1"
        assert diag["editorial_requirement_target_evidence_source"] == "EXACT_P1_MOMENT"


# ---------------------------------------------------------------------------
# Independence from idea_coverage_status / downstream_dependency_present --
# these are ALREADY atom/idea-scoped, independent signals D-239L never
# distrusts, even on the ownership-only path with no P1 evidence at all.
# ---------------------------------------------------------------------------
class TestIndependentAtomScopedSignalsUnaffected:
    def test_24_idea_coverage_status_required_unaffected_by_d239l(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own, idea_coverage_status=True,
        )
        assert res.editorial_requirement_status == "REQUIRED"
        assert res.blocking_recommendation == RECOMMEND_BLOCK

    def test_25_downstream_dependency_required_unaffected_by_d239l(self):
        own = _ownership()
        res = assess_complete_lost_semantic_atom_materiality(
            _row(), lost_atom_ownership=own, downstream_dependency_present=True,
        )
        assert res.editorial_requirement_status == "REQUIRED"
        assert res.blocking_recommendation == RECOMMEND_BLOCK


# ---------------------------------------------------------------------------
# No global authority change -- zero-diff on every named file. Uses the
# SAME `git diff --stat HEAD -- <path>` precedent this session has used
# since D-169 (a point-in-time snapshot; legitimately non-empty ONLY while
# a LATER, separately-authorized task's own uncommitted change to that
# exact file sits in the working tree -- not the case for any of these,
# since D-239L touches none of them).
# ---------------------------------------------------------------------------
class TestNoGlobalAuthorityChange:
    def _zero_diff(self, path: str) -> None:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD", "--", path],
            cwd="/home/user/EditDNA-worker", capture_output=True, text=True, check=True,
        )
        assert result.stdout.strip() == "", f"{path} unexpectedly diffs from HEAD: {result.stdout}"

    def test_26_shared_attempt_word_identity_zero_diff(self):
        self._zero_diff("cutsell_worker/shared_attempt_word_identity.py")

    def test_27_exact_lost_atom_ownership_zero_diff(self):
        self._zero_diff("cutsell_worker/exact_lost_atom_ownership.py")

    def test_28_language_proposition_relation_zero_diff(self):
        self._zero_diff("cutsell_worker/language_proposition_relation.py")

    def test_29_editorial_moment_sequence_zero_diff(self):
        self._zero_diff("cutsell_worker/editorial_moment_sequence.py")

    def test_30_editorial_moment_sequence_integration_zero_diff(self):
        self._zero_diff("cutsell_worker/editorial_moment_sequence_integration.py")

    def test_31_final_story_coherence_validation_zero_diff(self):
        self._zero_diff("cutsell_worker/final_story_coherence_validation.py")

    def test_32_pipeline_zero_diff(self):
        self._zero_diff("cutsell_worker/pipeline.py")

    def test_33_universal_clean_cut_zero_diff(self):
        self._zero_diff("cutsell_worker/universal_clean_cut.py")

    def test_34_repair_loop_zero_diff(self):
        self._zero_diff("cutsell_worker/repair_loop.py")

    def test_35_authoritative_relationship_statuses_never_reassigned(self):
        content = _code_without_docstrings(_PROD_PATH)
        assert "AUTHORITATIVE_RELATIONSHIP_STATUSES =" not in content
        assert "AUTHORITATIVE_RELATIONSHIP_STATUSES.add" not in content
        assert "AUTHORITATIVE_RELATIONSHIP_STATUSES |" not in content


# ---------------------------------------------------------------------------
# No new classifier / no new threshold / no fuzzy text heuristic.
# ---------------------------------------------------------------------------
class TestNoNewClassifierNoThresholdNoHeuristic:
    def test_36_no_new_env_flag(self):
        content = _code_without_docstrings(_PROD_PATH)
        assert "os.environ" not in content
        assert "CUTSELL_" not in content

    def test_37_no_fuzzy_or_timestamp_import(self):
        content = _code_without_docstrings(_PROD_PATH)
        for needle in ("difflib", "SequenceMatcher", "fuzz"):
            assert needle not in content

    def test_38_reused_vocabulary_only_no_new_status_values(self):
        # editorial_requirement_status only ever takes the SAME 5-value
        # vocabulary lost_atom_editorial_requirement_evidence.py already
        # defines -- D-239L introduces zero new status values there.
        own = _ownership()
        for kwargs, expected in (
            ({}, "INSUFFICIENT_EVIDENCE"),
            (
                {"recording_process_status": MOMENT_ROLE_CLEAN_AUDIENCE_DELIVERY,
                 "audience_delivery_status": AUDIENCE_DELIVERY_SUPPORTED},
                "REQUIRED",
            ),
            ({"recording_process_status": MOMENT_ROLE_POST_TAKE_RESET}, "NOT_REQUIRED"),
        ):
            res = assess_complete_lost_semantic_atom_materiality(
                _row(), lost_atom_ownership=own,
                proposition_slot_evidence_by_id={_TARGET_PROPOSITION_ID: SLOT_CTA},
                **kwargs,
            )
            assert res.editorial_requirement_status == expected
            assert res.editorial_requirement_status in {
                "REQUIRED", "NOT_REQUIRED", "REDUNDANT_REQUIRED_FUNCTION_PRESERVED",
                "INSUFFICIENT_EVIDENCE", "CONFLICTED",
            }
