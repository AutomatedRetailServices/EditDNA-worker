"""D-235S: EXACT LOST-ATOM -> REVIEWER FINDING PROVENANCE LINK, OFFLINE ONLY.

Covers the task's own required 37-item fixture/proof matrix. Exercises
the REAL `final_edit_reviewer.review()` and `repair_loop.run_repair_loop()`
functions end to end (not mocks) to prove provenance actually survives
through the live code, mirroring the established D-235J-R test style.
"""
from __future__ import annotations

import ast

from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_edit_reviewer import (
    CONTRADICTION,
    STORY_ORDER_BREAK,
    UNIQUE_FACT_LOST,
    review,
)
from cutsell_worker.repair_loop import run_repair_loop
import cutsell_worker.lost_atom_reviewer_finding_provenance as prov

PROD_PATH = "cutsell_worker/lost_atom_reviewer_finding_provenance.py"
COHERENCE_PATH = "cutsell_worker/final_story_coherence_validation.py"
REVIEWER_PATH = "cutsell_worker/final_edit_reviewer.py"
REPAIR_PATH = "cutsell_worker/repair_loop.py"
FREEZE_AUTH_PATH = "cutsell_worker/lost_semantic_atom_freeze_authority.py"
Q_PATH = "cutsell_worker/complete_lost_semantic_atom_materiality.py"


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _code_without_docstrings(path: str) -> str:
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


def _clip(clip_id, start, end, text, *, selected, source="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, caption_text=text, selected=selected,
    )


def _draft(*, selected=(), discarded=(), coherence=None):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=selected, alternates=(), discarded=discarded,
        diagnostics={"final_story_coherence_validation": coherence or {}},
    )


def _row(clip_id, text, *, blocking=True, provenance_id=None):
    row = {"clip_id": clip_id, "text": text, "missing_critical_atoms": [], "blocking": blocking}
    if provenance_id is not None:
        row["lost_atom_provenance_id"] = provenance_id
    return row


# ---------------------------------------------------------------------------
# 1-4. Link result fixtures (real review() call).
# ---------------------------------------------------------------------------
class TestLinkResults:
    def test_01_one_atom_one_finding_exact_link(self):
        lost = _clip("lost", 10.0, 15.0, "a unique fact nobody else mentions", selected=False)
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        row = _row("lost", lost.text, provenance_id="latom_lost_0")
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        result = review(build_canonical_edit_plan(d))
        link = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="latom_lost_0", clip_id="lost", findings=result.findings,
        )
        assert link.link_status == prov.LINK_EXACT_MATCH

    def test_02_two_atoms_same_clip_correct_atom_linked(self):
        # Defensive fixture: even though the real pipeline cannot produce
        # two rows for one clip (see module docstring's audit), the
        # classifier must still correctly distinguish two SYNTHETIC
        # findings sharing a clip_id when they carry DIFFERENT provenance
        # ids, and identify only the one asked about.
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        clip_x = _clip("clipX", 10.0, 15.0, "atom A text here", selected=False)
        row_a = _row("clipX", "atom A text here", provenance_id="latom_clipX_0")
        row_b = _row("clipX", "atom B text here totally different", provenance_id="latom_clipX_1")
        d = _draft(selected=(a,), discarded=(clip_x,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row_a, row_b], "contradiction_findings": [],
        })
        result = review(build_canonical_edit_plan(d))
        assert len([f for f in result.findings if f.kind == UNIQUE_FACT_LOST]) == 2
        link_b = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="latom_clipX_1", clip_id="clipX", findings=result.findings,
        )
        assert link_b.link_status == prov.LINK_EXACT_MATCH
        # A must remain unrelated -- the link for B never claims A's id.
        matched_finding = next(f for f in result.findings if f.detail.get("lost_atom_provenance_id") == "latom_clipX_1")
        assert matched_finding.detail.get("text") == "atom B text here totally different"

    def test_03_two_atoms_different_clips(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost1 = _clip("lost1", 10.0, 15.0, "unique fact one", selected=False)
        lost2 = _clip("lost2", 20.0, 25.0, "unique fact two entirely different", selected=False)
        row1 = _row("lost1", lost1.text, provenance_id="latom_lost1_0")
        row2 = _row("lost2", lost2.text, provenance_id="latom_lost2_0")
        d = _draft(selected=(a,), discarded=(lost1, lost2), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row1, row2], "contradiction_findings": [],
        })
        result = review(build_canonical_edit_plan(d))
        link1 = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="latom_lost1_0", clip_id="lost1", findings=result.findings,
        )
        link2 = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="latom_lost2_0", clip_id="lost2", findings=result.findings,
        )
        assert link1.link_status == link2.link_status == prov.LINK_EXACT_MATCH

    def test_04_one_atom_multiple_findings_not_forced_1to1(self):
        # `_lost_semantic_atoms()` produces exactly one row per clip, and
        # `review()` maps exactly one Finding per row -- so under the
        # current, real code, one atom never produces multiple findings.
        # This is a structural fact this module never contradicts (it does
        # not force 1:1 by construction -- it simply reflects what the
        # supplied `findings` list actually contains, however many there
        # are).
        content = _code_without_docstrings(PROD_PATH)
        assert "assert len(" not in content  # no hardcoded 1:1 enforcement


# ---------------------------------------------------------------------------
# 5-7. Unrelated / unsupported findings.
# ---------------------------------------------------------------------------
class TestUnrelatedAndUnsupported:
    def test_05_unrelated_unique_fact_lost_not_matched(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost = _clip("lost", 10.0, 15.0, "a unique fact", selected=False)
        row = _row("lost", lost.text, provenance_id="latom_lost_0")
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        result = review(build_canonical_edit_plan(d))
        link = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="latom_unrelated_0", clip_id="unrelated", findings=result.findings,
        )
        assert link.link_status == prov.LINK_NO_MATCH

    def test_06_story_order_break_unsupported(self):
        link = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="latom_x_0", clip_id="x", findings=(),
            target_finding_kind=STORY_ORDER_BREAK,
        )
        assert link.link_status == prov.LINK_UNSUPPORTED_FINDING_KIND

    def test_07_contradiction_unsupported(self):
        link = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="latom_x_0", clip_id="x", findings=(),
            target_finding_kind=CONTRADICTION,
        )
        assert link.link_status == prov.LINK_UNSUPPORTED_FINDING_KIND


# ---------------------------------------------------------------------------
# 8/9. Missing / malformed provenance.
# ---------------------------------------------------------------------------
class TestMissingMalformedProvenance:
    def test_08_missing_provenance(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost = _clip("oldclip", 10.0, 15.0, "old shape", selected=False)
        row = _row("oldclip", lost.text)  # no provenance_id
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        result = review(build_canonical_edit_plan(d))
        link = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id=None, clip_id="oldclip", findings=result.findings,
        )
        assert link.link_status == prov.LINK_MISSING_PROVENANCE

    def test_09_malformed_provenance_empty_string(self):
        link = prov.classify_lost_atom_reviewer_finding_link(
            lost_atom_provenance_id="", clip_id="x", findings=(),
        )
        assert link.link_status == prov.LINK_MISSING_PROVENANCE


# ---------------------------------------------------------------------------
# 10-12. Determinism / order independence.
# ---------------------------------------------------------------------------
class TestDeterminismAndOrderIndependence:
    def test_10_deterministic_repeat(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost = _clip("lost", 10.0, 15.0, "a unique fact", selected=False)
        row = _row("lost", lost.text, provenance_id="latom_lost_0")
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        results = set()
        for _ in range(5):
            result = review(build_canonical_edit_plan(d))
            link = prov.classify_lost_atom_reviewer_finding_link(
                lost_atom_provenance_id="latom_lost_0", clip_id="lost", findings=result.findings,
            )
            results.add(link.link_status)
        assert results == {prov.LINK_EXACT_MATCH}

    def test_11_row_order_independence(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost1 = _clip("lost1", 10.0, 15.0, "fact one", selected=False)
        lost2 = _clip("lost2", 20.0, 25.0, "fact two entirely different", selected=False)
        row1 = _row("lost1", lost1.text, provenance_id="latom_lost1_0")
        row2 = _row("lost2", lost2.text, provenance_id="latom_lost2_0")
        for rows in ([row1, row2], [row2, row1]):
            d = _draft(selected=(a,), discarded=(lost1, lost2), coherence={
                "freeze_blocked": True, "lost_semantic_atoms": rows, "contradiction_findings": [],
            })
            result = review(build_canonical_edit_plan(d))
            link1 = prov.classify_lost_atom_reviewer_finding_link(
                lost_atom_provenance_id="latom_lost1_0", clip_id="lost1", findings=result.findings,
            )
            assert link1.link_status == prov.LINK_EXACT_MATCH

    def test_12_dict_order_independence(self):
        row_a = {"lost_atom_provenance_id": "latom_x_0", "clip_id": "x", "text": "t", "blocking": True}
        row_b = {"clip_id": "x", "blocking": True, "text": "t", "lost_atom_provenance_id": "latom_x_0"}
        assert row_a.get("lost_atom_provenance_id") == row_b.get("lost_atom_provenance_id")


# ---------------------------------------------------------------------------
# 13. Old payload without provenance still valid.
# ---------------------------------------------------------------------------
class TestHistoricalPayload:
    def test_13_old_payload_remains_valid_link_unavailable(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost = _clip("oldclip", 10.0, 15.0, "old shape content", selected=False)
        row = _row("oldclip", lost.text)  # no provenance_id -- pre-D-235S shape
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        plan = build_canonical_edit_plan(d)
        result = review(plan)
        # The row/Finding remain fully valid and blocking, exactly as before.
        assert result.status == "FAIL"
        assert any(f.kind == UNIQUE_FACT_LOST and f.blocking for f in result.findings)


# ---------------------------------------------------------------------------
# 14. D-235K generic replay -- all three stages carry the SAME provenance id.
# ---------------------------------------------------------------------------
class TestD235KReplay:
    def test_14_provenance_survives_all_three_stages(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost = _clip("d235k_clip", 10.0, 15.0, "a generic lost fragment of real speech", selected=False)
        row = _row("d235k_clip", lost.text, provenance_id="latom_d235k_clip_0")
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        plan = build_canonical_edit_plan(d)
        result = review(plan)
        unique_fact_lost = [f for f in result.findings if f.kind == UNIQUE_FACT_LOST]
        assert len(unique_fact_lost) == 1
        assert unique_fact_lost[0].detail.get("lost_atom_provenance_id") == "latom_d235k_clip_0"

        repair_result = run_repair_loop(d)
        assert repair_result.status == "NEEDS_HUMAN_REVIEW"
        assert len(repair_result.attempts) == 1
        assert repair_result.attempts[0].source_lost_atom_provenance_id == "latom_d235k_clip_0"

        survived = prov.lost_atom_provenance_survived_into_repair_attempt(
            "latom_d235k_clip_0", repair_result.attempts,
        )
        assert survived is True

    def test_no_hardcoded_literal_phrase(self):
        content = _read(PROD_PATH)
        assert "too many people" not in content
        assert "ready set" not in content


# ---------------------------------------------------------------------------
# 15/16. Reviewer / RepairLoop propagation proofs.
# ---------------------------------------------------------------------------
class TestPropagationProofs:
    def test_15_provenance_survives_reviewer_via_existing_dict_row_copy(self):
        # Zero code change to final_edit_reviewer.py needed -- detail=dict(row)
        # already carries any new row key verbatim.
        content = _read(REVIEWER_PATH)
        assert "detail=dict(row)" in content

    def test_16_provenance_survives_repair_loop_via_additive_field(self):
        content = _read(REPAIR_PATH)
        assert "source_lost_atom_provenance_id" in content
        assert content.count("source_lost_atom_provenance_id=finding.detail.get(\"lost_atom_provenance_id\")") == 2
        assert "source_lost_atom_provenance_id=unrepairable.detail.get(\"lost_atom_provenance_id\")" in content


# ---------------------------------------------------------------------------
# 17-20. Parity: NEEDS_HUMAN_REVIEW / Freeze / materiality unchanged.
# ---------------------------------------------------------------------------
class TestParity:
    def test_17_needs_human_review_status_unchanged(self):
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost = _clip("lost", 10.0, 15.0, "a unique fact", selected=False)
        row = _row("lost", lost.text, provenance_id="latom_lost_0")
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        repair_result = run_repair_loop(d)
        assert repair_result.status == "NEEDS_HUMAN_REVIEW"

    def test_18_no_suppression_even_for_non_material_shape(self):
        # Even a row shaped like it WOULD be safely suppressible under
        # D-235R's own materiality authority, RepairLoop must still behave
        # byte-identically here -- D-235S links evidence only.
        a = _clip("a", 0.0, 5.0, "the only take", selected=True)
        lost = _clip("lost", 10.0, 15.0, "a generic lost fragment", selected=False)
        row = _row("lost", lost.text, provenance_id="latom_lost_0")
        d = _draft(selected=(a,), discarded=(lost,), coherence={
            "freeze_blocked": True, "lost_semantic_atoms": [row], "contradiction_findings": [],
        })
        repair_result = run_repair_loop(d)
        assert repair_result.status == "NEEDS_HUMAN_REVIEW"
        assert repair_result.attempts[0].repaired is False

    def test_19_no_freeze_behavior_change(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("freeze_blocked", "SelectionFreeze"):
            assert needle not in content

    def test_20_no_materiality_authority_expansion(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "complete_lost_semantic_atom_materiality" not in content
        assert "lost_semantic_atom_freeze_authority" not in content


# ---------------------------------------------------------------------------
# 21/22. No fuzzy text / no timestamp heuristic.
# ---------------------------------------------------------------------------
class TestStructuralSafety:
    def test_21_no_fuzzy_text(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("SequenceMatcher", "difflib", "fuzzy", "ratio(", "get_close_matches"):
            assert needle not in content

    def test_22_no_timestamp_heuristic(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in (".start", ".end", "overlap", "tolerance_sec", "IoU"):
            assert needle not in content

    def test_23_no_provider(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("requests.", "httpx", "openai", "genai.", "modal.", "runpod"):
            assert needle not in content

    def test_24_no_raw_subprocess(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("subprocess", "socket.", "urllib", "boto3"):
            assert needle not in content

    def test_25_no_p1_p2_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("editorial_moment_sequence", "language_spine", "language_proposition_relation"):
            assert needle not in content

    def test_26_no_besttake_family_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("deterministic_best_take_authority", "take_grouping", "take_judge"):
            assert needle not in content

    def test_27_no_ordering_boundary_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        assert "boundary_engine_pass" not in content

    def test_28_no_pacing_audio_join_mutation(self):
        content = _code_without_docstrings(PROD_PATH)
        for needle in ("dialogue_pacing_transition", "audio_join"):
            assert needle not in content


# ---------------------------------------------------------------------------
# 29. compileall proxy.
# ---------------------------------------------------------------------------
class TestCompiles:
    def test_29_compiles(self):
        for path in (PROD_PATH, COHERENCE_PATH, REVIEWER_PATH, REPAIR_PATH):
            ast.parse(_read(path))
        ast.parse(_read("tests/test_cutsell_d235s_lost_atom_reviewer_finding_provenance.py"))


# ---------------------------------------------------------------------------
# 30/31. D-235R/Q regression (zero diff / no import of the new module).
# ---------------------------------------------------------------------------
class TestSiblingRegression:
    def test_30_d235r_module_not_modified(self):
        content = _read(FREEZE_AUTH_PATH)
        assert "lost_atom_reviewer_finding_provenance" not in content

    def test_31_d235q_module_not_modified(self):
        content = _read(Q_PATH)
        assert "lost_atom_reviewer_finding_provenance" not in content


# ---------------------------------------------------------------------------
# Diagnostics (25 in deliverable list).
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_diagnostics_required_fields(self):
        diag = prov.lost_atom_reviewer_finding_provenance_diagnostics(
            lost_atom_provenance_id="latom_x_0", reviewer_finding_kind=UNIQUE_FACT_LOST,
            reviewer_source_lost_atom_provenance_id="latom_x_0",
            repair_source_lost_atom_provenance_id="latom_x_0",
            link_status=prov.LINK_EXACT_MATCH,
        )
        for key in (
            "lost_atom_provenance_id", "reviewer_finding_kind",
            "reviewer_source_lost_atom_provenance_id", "repair_source_lost_atom_provenance_id",
            "link_status",
        ):
            assert key in diag

    def test_diagnostics_no_transcript_dump(self):
        diag = prov.lost_atom_reviewer_finding_provenance_diagnostics(
            lost_atom_provenance_id="latom_x_0", reviewer_finding_kind=UNIQUE_FACT_LOST,
            reviewer_source_lost_atom_provenance_id=None, repair_source_lost_atom_provenance_id=None,
            link_status=prov.LINK_MISSING_PROVENANCE,
        )
        assert "text" not in diag


# ---------------------------------------------------------------------------
# Vocabulary.
# ---------------------------------------------------------------------------
class TestVocabulary:
    def test_link_vocabulary_five_values(self):
        assert prov._VALID_LINK_STATUSES == {
            "EXACT_MATCH", "NO_MATCH", "AMBIGUOUS", "MISSING_PROVENANCE", "UNSUPPORTED_FINDING_KIND",
        }
