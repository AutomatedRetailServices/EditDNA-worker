"""Independent acceptance checks built from the recorded #115/#118/#122/#124
selection shapes. Source intervals and wording appear only in this benchmark;
none of these targets is fed to the production editor.
"""
import json
from pathlib import Path

from benchmarks.validate_video00_regression_qa import (
    find_repeated_closings,
    realization_present,
    validate,
)


MANIFEST = Path("benchmarks/video00_editorial_acceptance.json")
HISTORICAL_MANIFEST = Path("benchmarks/video00_regression_qa.json")


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _row(text, start, end, clip_id="clip"):
    return {"clip_id": clip_id, "text": text, "start": start, "end": end}


def _run(tmp_path, rows, *, manifest=None):
    result_path = tmp_path / "selection.json"
    manifest_path = tmp_path / "manifest.json"
    result_path.write_text(json.dumps({"selected": rows}), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest or _manifest()), encoding="utf-8")
    return validate(str(result_path), str(manifest_path))


def _failures(report):
    return {finding["id"]: finding["reason"] for finding in report["failed_checks"]}


def _check(check_id):
    return next(check for check in _manifest()["checks"] if check["id"] == check_id)


def test_real_118_punctuation_does_not_hide_rejected_take(tmp_path):
    # RAW #118 chose the wrong monolith with a comma after "aquí"; the
    # historical forbidden_contains check falsely passed on that comma.
    bad = (
        "También me salían espinillas en esta parte de aquí, detrás de la oreja "
        "y todo el cuello que yo pensaba que era alergia pero era como espinillas "
        "de personas con problemas hormonales."
    )
    good = (
        "Otro síntoma era que me salían espinillas como si fuera una alergia de "
        "esta parte aquí detrás de la oreja y en el cuello. Me salía por temporadas."
    )
    bad_check = _check("pimples_bad_monolith_absent")
    # The text-only diagnostic detects the old shared-word false PASS; the
    # acceptance manifest uses the source span, which permits re-chunking.
    good_check = {
        "id": "later_text_diagnostic", "kind": "required_realization",
        "text": "Otro síntoma era que me salían espinillas como si fuera una alergia de "
                "esta parte aquí detrás de la oreja y en el cuello. Me salía por",
    }
    assert realization_present(bad, bad_check["text"])
    assert not realization_present(bad, good_check["text"])
    assert not realization_present(good, bad_check["text"])
    _, report = _run(tmp_path, [_row(bad, 198.88, 211.02)], manifest={
        "checks": [bad_check, good_check],
    })
    assert _failures(report) == {
        "pimples_bad_monolith_absent": "forbidden_realization_selected",
        "later_text_diagnostic": "realization_not_present_only_shared_content",
    }
    _, report = _run(tmp_path, [_row(good, 213.44, 222.38)], manifest={
        "checks": [bad_check, good_check],
    })
    assert report["qa_pass"]


def test_bad_and_good_take_identity_survive_asr_word_drift(tmp_path):
    checks = [_check("pimples_bad_source_absent"), _check("pimples_later_source_selected")]
    _, report = _run(tmp_path, [
        _row("The source take with substantially changed wording", 198.88, 211.02),
        _row("A neighboring short microfragment", 192.12, 198.01),
    ], manifest={"checks": checks})
    assert set(_failures(report)) == {c["id"] for c in checks}
    _, report = _run(tmp_path, [
        _row("The complete later take even if its words changed", 213.44, 222.38),
    ], manifest={"checks": checks})
    assert report["qa_pass"]
    _, report = _run(tmp_path, [
        _row("First half of the later take", 213.44, 217.56, "half1"),
        _row("Other half of the same source take", 217.56, 222.38, "half2"),
    ], manifest={"checks": checks})
    assert report["qa_pass"]  # Benign source-take re-chunking remains present.
    _, report = _run(tmp_path, [
        _row("Overlapping reference A", 213.44, 217.56, "repeat1"),
        _row("Overlapping reference B", 213.44, 217.56, "repeat2"),
    ], manifest={"checks": [checks[1]]})
    assert _failures(report) == {"pimples_later_source_selected": "source_overlap_missing"}
    _, report = _run(tmp_path, [
        _row("First half of the rejected take", 198.86, 202.0, "bad1"),
        _row("Other half of the rejected take", 202.0, 205.0, "bad2"),
    ], manifest={"checks": [checks[0]]})
    assert _failures(report) == {"pimples_bad_source_absent": "forbidden_source_overlap_selected"}


def test_source_windows_catch_failed_attempt_and_do_not_treat_absent_timing_as_pass(tmp_path):
    checks = _manifest()["checks"]
    stomach = next(c for c in checks if c["id"] == "abandoned_stomach_attempt_absent")
    gyn = next(c for c in checks if c["id"] == "full_gynecologist_take_selected")
    _, report = _run(tmp_path, [
        _row("A later complete delivery", 95.58, 104.02, "good_gyn"),
        _row("An unfinished stomach attempt", 251.81, 253.37, "orphan_118"),
    ], manifest={"checks": [stomach, gyn]})
    assert _failures(report) == {"abandoned_stomach_attempt_absent": "forbidden_source_overlap_selected"}
    _, report = _run(tmp_path, [
        _row("An earlier, different realization", 82.82, 89.36, "earlier_gyn"),
        _row("An abandoned attempt", 245.39, 251.61, "attempt_124"),
    ], manifest={"checks": [stomach, gyn]})
    assert set(_failures(report)) == {"abandoned_stomach_attempt_absent", "full_gynecologist_take_selected"}
    _, report = _run(tmp_path, [
        _row("Only a partial gynecologist take", 99.1, 104.1, "partial_gyn"),
    ], manifest={"checks": [gyn]})
    assert _failures(report) == {"full_gynecologist_take_selected": "source_overlap_missing"}
    _, report = _run(tmp_path, [
        _row("First complete half", 95.58, 99.6, "half1"),
        _row("Second complete half", 99.6, 104.02, "half2"),
    ], manifest={"checks": [gyn]})
    assert report["qa_pass"]
    _, report = _run(tmp_path, [{"clip_id": "unknown", "text": "No source timing"}],
                     manifest={"checks": [stomach, gyn]})
    assert _failures(report) == {
        "abandoned_stomach_attempt_absent": "source_interval_unavailable",
        "full_gynecologist_take_selected": "source_interval_unavailable",
    }


def test_negation_must_stay_with_its_sentence_in_one_delivery(tmp_path):
    check = _check("same_take_negation_with_sentence")
    manifest = {"checks": [check]}
    _, report = _run(tmp_path, [
        _row("No", 269.37, 270.17),
        _row("quiero sonar a conspiración, pero pasó algo.", 276.09, 283.67),
    ], manifest=manifest)
    assert _failures(report) == {
        "same_take_negation_with_sentence": "phrase_not_contiguous_within_one_delivery",
    }
    _, report = _run(tmp_path, [
        _row("No quiero sonar a conspiración, pero pasó algo.", 275.83, 283.65),
    ], manifest=manifest)
    assert report["qa_pass"]


def test_repeated_closing_is_found_even_with_four_intervening_clips(tmp_path):
    check = next(c for c in _manifest()["checks"] if c["id"] == "closing_exhortation_not_reopened")
    rows = [
        _row("Mi experiencia terminó. Así que cuídate.", 295.52, 313.52, "conclusion"),
        _row("Una explicación familiar.", 319.38, 327.0, "aside"),
        _row("Estoy convencida del porcentaje.", 327.84, 334.25, "restatement"),
        _row("Continúa la explicación.", 335.88, 342.04, "tail"),
        _row("Una frase adicional.", 342.73, 345.56, "additional"),
        _row("Por eso cuídate, aliméntate bien, hidrátate y haz ejercicio.",
             356.13, 361.55, "cta"),
    ]
    assert find_repeated_closings([(r["clip_id"], r["text"]) for r in rows]) == []
    _, report = _run(tmp_path, rows, manifest={"checks": [check]})
    assert _failures(report) == {
        "closing_exhortation_not_reopened": "repeated_closing_reopens_required_segment",
    }
    rows[0]["text"] = "Mi experiencia terminó y así aprendí a cuidarme."
    _, report = _run(tmp_path, rows, manifest={"checks": [check]})
    assert report["qa_pass"]


def test_closing_keeps_instructions_and_says_cuidate_exactly_once(tmp_path):
    checks = [
        _check("closing_exhortation_not_reopened"),
        _check("closing_instruction_preserved"),
        _check("closing_care_exhortation_exactly_once"),
    ]
    _, report = _run(tmp_path, [
        _row("Una historia termina sin exhortación.", 300.0, 311.0),
        _row("Aliméntate bien, hidrátate y haz ejercicio.", 356.13, 361.55),
    ], manifest={"checks": checks})
    assert _failures(report) == {
        "closing_care_exhortation_exactly_once": "phrase_occurrence_count_mismatch",
    }
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó. Así que cuídate.", 300.0, 311.0),
        _row("Aliméntate bien, hidrátate y haz ejercicio.", 356.13, 361.55),
    ], manifest={"checks": checks})
    assert report["qa_pass"]  # The pre-Freeze CTA trim may remove its duplicate prefix.
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó. Así que cuídate.", 300.0, 311.0),
        _row("Alimentate bien, hídrate y haz ejercicio.", 356.13, 361.55),
    ], manifest={"checks": checks})
    assert report["qa_pass"]  # Common accent and imperative-morphology ASR drift.
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó sin exhortación.", 300.0, 311.0),
        _row("Por eso cuídate. Alimbentate bien, hídratate y haz ejercicio.",
             356.13, 361.55),
    ], manifest={"checks": checks})
    assert report["qa_pass"]  # RAW #118's harmless ASR spelling drift.
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó sin exhortación.", 300.0, 311.0),
        _row("Por eso cuidate. Aliméntate bien, hidrátate y haz ejercicio.",
             356.13, 361.55),
    ], manifest={"checks": checks})
    assert report["qa_pass"]  # Accent omitted on 'cuídate' is not absence.
    _, report = _run(tmp_path, [
        _row("Por eso cuídate. Aliméntate bien, no hidrátate y no haz ejercicio.",
             356.13, 361.55),
    ], manifest={"checks": checks})
    assert _failures(report) == {
        "closing_instruction_preserved": "ordered_actions_polarity_conflict",
    }
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó. Así que cuídate.", 300.0, 311.0),
        _row("Aliméntate bien, haz ejercicio.", 356.13, 361.55),
    ], manifest={"checks": checks})
    assert _failures(report) == {
        "closing_exhortation_not_reopened": "missing_required_segment",
        "closing_instruction_preserved": "ordered_actions_not_present_in_one_delivery",
    }
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó. Así que cuídate. Aliméntate bien.", 300.0, 311.0),
        _row("Hidrátate y haz ejercicio.", 356.13, 361.55),
    ], manifest={"checks": [checks[1]]})
    assert _failures(report) == {
        "closing_instruction_preserved": "ordered_actions_not_present_in_one_delivery",
    }
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó. Así que cuídate.", 300.0, 311.0),
        _row("Por eso cuídate, aliméntate bien, hidrátate y haz ejercicio.", 356.13, 361.55),
    ], manifest={"checks": checks})
    assert set(_failures(report)) == {
        "closing_exhortation_not_reopened", "closing_care_exhortation_exactly_once",
    }
    _, report = _run(tmp_path, [
        _row("Mi experiencia terminó. Así que cuídate.", 300.0, 311.0),
        _row("Por eso cuídate.", 356.13, 361.55),
    ], manifest={"checks": checks})
    assert "closing_instruction_preserved" in _failures(report)
    _, report = _run(tmp_path, [
        _row("Cuídate. Aliméntate bien, hidrátate y haz ejercicio.", 80.0, 84.0),
        _row("Mi experiencia terminó sin ningún consejo final.", 295.0, 311.0),
    ], manifest={"checks": checks})
    assert _failures(report) == {
        "closing_exhortation_not_reopened": "missing_required_segment",
        "closing_instruction_preserved": "ordered_actions_not_present_in_one_delivery",
    }


def test_complete_target_can_pass_all_checks_without_changing_historical_oracle(tmp_path):
    historical = json.loads(HISTORICAL_MANIFEST.read_text(encoding="utf-8"))
    assert next(c for c in historical["checks"] if c["id"] == "pimples_later_winner_present")["kind"] == "required_exact"
    manifest = _manifest()
    assert len(manifest["checks"]) == 14
    good = (
        "Otro síntoma era que me salían espinillas como si fuera una alergia de "
        "esta parte aquí detrás de la oreja y en el cuello. Me salía por temporadas."
    )
    rows = [
        _row(
            "Tenía como costumbre cada vez que terminaba un contrato hacerme un chequeo "
            "de rutina con mi ginecóloga.",
            95.58, 104.02,
        ),
        _row("Ahí fue cuando me mandaron a hacer sonografía.", 120.03, 124.34),
        _row(good, 213.44, 222.38),
        _row("Tuve problemas de digestión y me diagnosticaron gastritis.", 258.85, 268.47),
        _row("No quiero sonar a conspiración, pero pasó algo.", 275.83, 283.65),
        _row("Esta fue mi experiencia. Nuestras decisiones de vida importan.", 295.36, 312.36),
        _row("Por eso cuídate, aliméntate bien, hidrátate y haz ejercicio.", 356.13, 361.55),
    ]
    ok, report = _run(tmp_path, rows)
    assert ok and report["failed_check_count"] == 0
    assert set(report["passed_checks"]) == {c["id"] for c in manifest["checks"]}
