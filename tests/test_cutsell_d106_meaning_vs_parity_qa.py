"""D-106: QA semantics correction -- MEANING PRESERVATION ("did CutSell
preserve the required meaning?") vs PREFERRED-REALIZATION PARITY ("did
CutSell select the same realization the QA reference did?") are distinct
axes. A semantically-equivalent paraphrase may PASS meaning while FAILING
parity; the mismatch is never hidden. See docs/CUTSELL_DECISIONS.md D-106.

Generic (non-Video00) fixtures only -- these prove the GENERAL evaluator
behavior, not the specific papillary transcript (that is covered by the
manifest change itself and verified once against real RAW 34077889576
data in the accompanying report, not committed here as a test).
"""
import json

from benchmarks.validate_video00_regression_qa import validate

_TARGET_TEXT = "the customer received a full refund within 5 business days"


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def manifest_with_meaning_check(*, protected: bool = False):
    return {
        "schema_version": "cutsell.video00.regression_qa.v1",
        "baseline_run_id": 1,
        "checks": [
            {
                "id": "conclusion_meaning",
                "kind": "meaning_preservation",
                "protected": protected,
                "text": _TARGET_TEXT,
            },
            {
                "id": "conclusion_parity",
                "kind": "preferred_realization_parity",
                "text": _TARGET_TEXT,
            },
        ],
    }


def base_result(selected_extra_text, discarded_text, *, merge_confidence=None, merge_ids=("clip_lost", "clip_kept")):
    result = {
        "selected": [
            {"clip_id": "clip_intro", "text": "the order shipped a week later than expected"},
            {"clip_id": "clip_kept", "text": selected_extra_text},
        ],
        "discarded": [
            {"clip_id": "clip_lost", "text": discarded_text},
        ],
    }
    if merge_confidence is not None:
        result["diagnostics"] = {
            "semantic_idea_equivalence": {
                "merges": [
                    {"left_clip_id": merge_ids[0], "right_clip_id": merge_ids[1], "confidence": merge_confidence,
                     "reason": "Both describe the customer being made whole after the shipping delay."}
                ]
            }
        }
    return result


def test_1_exact_expected_realization_meaning_pass_and_parity_pass(tmp_path):
    result = base_result(_TARGET_TEXT, "an unrelated aside about warehouse staffing")
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check()),
    )
    assert report["meaning_preservation"]["passed_checks"] == ["conclusion_meaning"]
    assert report["preferred_realization_parity"]["passed_checks"] == ["conclusion_parity"]
    assert ok is True


def test_2_strong_equivalent_paraphrase_meaning_pass_parity_fail(tmp_path):
    # The exact wording is NOT selected; a genuinely different but
    # high-confidence-equivalent paraphrase is kept instead.
    result = base_result(
        "the customer got every dollar back inside a week",
        _TARGET_TEXT,
        merge_confidence=0.95,
    )
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check()),
    )
    assert report["meaning_preservation"]["passed_checks"] == ["conclusion_meaning"]
    assert report["meaning_preservation"]["failed_checks"] == []
    # Parity correctly fails -- the preferred wording was not selected --
    # and this must NEVER gate qa_pass.
    parity = report["preferred_realization_parity"]
    assert parity["passed_checks"] == []
    assert len(parity["failed_checks"]) == 1
    assert parity["failed_checks"][0]["id"] == "conclusion_parity"
    assert ok is True  # meaning PASS -> overall qa_pass unaffected by the parity mismatch


def test_3_weak_uncertain_similarity_meaning_not_pass(tmp_path):
    result = base_result(
        "the customer got every dollar back inside a week",
        _TARGET_TEXT,
        merge_confidence=0.40,  # below the approved D-061 floor (0.85)
    )
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check()),
    )
    meaning = report["meaning_preservation"]
    assert meaning["passed_checks"] == []
    assert meaning["uncertain_check_count"] == 1
    assert meaning["uncertain_checks"][0]["reason"] == "insufficient_equivalence_evidence"
    assert ok is False  # UNCERTAIN never silently counts as PASS


def test_4_contradiction_meaning_fail(tmp_path):
    # High confidence merge, but the credited candidate literally negates
    # the exact shared proposition the target affirms -- equivalence
    # credit must be refused regardless of the confidence number.
    result = base_result(
        "the customer did not receive a full refund within 5 business days",
        _TARGET_TEXT,
        merge_confidence=0.95,
    )
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check()),
    )
    meaning = report["meaning_preservation"]
    assert meaning["passed_checks"] == []
    assert len(meaning["failed_checks"]) == 1
    assert meaning["failed_checks"][0]["reason"] == "protected_contradiction_detected"
    assert ok is False


def test_5_polarity_change_meaning_fail(tmp_path):
    result = base_result(
        "the customer never received a full refund within 5 business days",
        _TARGET_TEXT,
        merge_confidence=0.90,
    )
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check()),
    )
    meaning = report["meaning_preservation"]
    assert len(meaning["failed_checks"]) == 1
    assert meaning["failed_checks"][0]["reason"] == "protected_contradiction_detected"
    assert ok is False


def test_6_numeric_materially_different_proposition_meaning_fail(tmp_path):
    result = base_result(
        "the customer received a full refund within 10 business days",
        _TARGET_TEXT,
        merge_confidence=0.90,
    )
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check()),
    )
    meaning = report["meaning_preservation"]
    assert len(meaning["failed_checks"]) == 1
    assert meaning["failed_checks"][0]["reason"] == "protected_contradiction_detected"
    assert ok is False


def test_6b_protected_check_never_receives_equivalence_credit(tmp_path):
    # A `protected: true` check (diagnosis identity / correction-class
    # proposition) must never be satisfied via equivalence credit at all --
    # the canonical realization itself must be present.
    result = base_result(
        "the customer got every dollar back inside a week",
        _TARGET_TEXT,
        merge_confidence=0.99,
    )
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check(protected=True)),
    )
    meaning = report["meaning_preservation"]
    assert len(meaning["failed_checks"]) == 1
    assert meaning["failed_checks"][0]["reason"] == "missing_required_segment_protected"
    assert ok is False


def test_7_no_benchmark_reference_information_leaks_into_production_selection():
    """The manifest/QA reference text is consumed ONLY by this benchmark
    harness. Production selection code must never import or embed it."""
    import pathlib
    forbidden_needles = (
        "papillary_symptom_realization",
        "the customer received a full refund",
        "video00_regression_qa.json",
    )
    for path in pathlib.Path("cutsell_worker").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden_needles:
            assert needle not in text, f"{needle!r} leaked into production code: {path}"


def test_real_case_shape_missing_realization_entirely_is_still_a_failure(tmp_path):
    """If neither the canonical text NOR any discarded clip carries it at
    all, this is genuine content loss -- never silently downgraded to
    UNCERTAIN just because the check is unprotected."""
    result = {
        "selected": [{"clip_id": "clip_kept", "text": "an unrelated closing remark"}],
        "discarded": [{"clip_id": "clip_other", "text": "also completely unrelated content"}],
    }
    ok, report = validate(
        write_json(tmp_path, "result.json", result),
        write_json(tmp_path, "manifest.json", manifest_with_meaning_check()),
    )
    meaning = report["meaning_preservation"]
    assert meaning["failed_checks"][0]["reason"] == "missing_required_segment_no_realization_found"
    assert ok is False
