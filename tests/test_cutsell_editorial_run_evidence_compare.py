"""Portable evidence-comparison tests; no production video or cloud access."""

import json
import hashlib

import pytest

from benchmarks.compare_editorial_run_evidence import compare_runs, extract_run, main
from cutsell_worker.universal_clean_cut_validation import _sha256_source_media


def _result(
    project: str, *, source: str = "input/movie.mp4", content: str = "same",
    canonical: str = "same", config: str = "same", selected=None, complete: bool = True,
    scoped: str = "source-scoped", package_hash: str = "not-an-input-checksum",
):
    return {
        "project_id": project,
        "source_key": source,
        "output_sha256": "output-is-not-the-input-sha256",
        "active_path_identity": {"package": {"sha256": package_hash}, "build_git_sha": project},
        "selected": selected if selected is not None else [{"start": 0, "end": 3, "text": "A"}],
        "stage_status": {"canonical_asr_evidence": {
            "status": "complete" if complete else "degraded",
            "content_hash": content,
            "canonical_equivalence_hash": canonical,
            "evidence_hash": scoped,
            "asr_config_fingerprint": config,
            "normalized_word_count": 10,
        }},
    }


def test_same_source_key_and_asr_configuration_do_not_prove_same_transcript_or_media():
    left = extract_run(_result("left", content="content-a", canonical="canonical-a"))
    right = extract_run(_result("right", content="content-b", canonical="canonical-b"))
    compared = compare_runs(left, right)
    assert compared["same_source_key"] is True
    assert compared["source_media_status"] == "same_source_key_bytes_unverified"
    assert compared["asr_status"] == "different_canonical_asr"
    assert compared["code_regression_attribution"] == "not_supported_asr_changed"
    assert compared["code_regression_proven"] is False
    assert left["media_sha256"] is None  # output_sha256 and package fingerprint are not source checksums


def test_different_source_scoped_evidence_hash_alone_does_not_mean_asr_changed():
    left = extract_run(_result("left", scoped="different-asset-A"), media_sha256="a" * 64)
    right = extract_run(_result("right", scoped="different-asset-B"), media_sha256="a" * 64)
    compared = compare_runs(left, right)
    assert compared["asr_status"] == "same_content_and_canonical_hashes"
    assert compared["code_regression_attribution"] == "eligible_for_controlled_replay_not_proof_of_code_regression"
    assert compared["code_regression_proven"] is False


@pytest.mark.parametrize("change, expected", [
    ({"content": "new"}, "not_supported_asr_word_text_changed"),
    ({"canonical": "new"}, "not_supported_asr_changed"),
    ({"config": "new"}, "not_supported_asr_config_changed"),
    ({"complete": False}, "not_supported_missing_asr_evidence"),
    ({"content": ""}, "not_supported_missing_asr_evidence"),
    ({"canonical": ""}, "not_supported_missing_asr_evidence"),
])
def test_changed_or_missing_asr_evidence_refuses_code_attribution_even_with_matching_media_hash(change, expected):
    left = extract_run(_result("left"), media_sha256="a" * 64)
    right = extract_run(_result("right", **change), media_sha256="a" * 64)
    compared = compare_runs(left, right)
    assert compared["code_regression_attribution"] == expected
    assert compared["code_regression_proven"] is False


def test_selection_delta_measures_source_coverage_not_clip_count_or_order():
    left = extract_run(_result("left", selected=[{"start": 0, "end": 4}, {"start": 5, "end": 8}]))
    right = extract_run(_result("right", selected=[{"start": 1, "end": 7}, {"start": 7, "end": 8}]))
    compared = compare_runs(left, right)
    assert left["selected_source_coverage_sec"] == 7
    assert right["selected_source_coverage_sec"] == 7
    assert compared["selected_intervals_left_only"] == [
        {"start": 0.0, "end": 1.0, "duration_sec": 1.0}
    ]
    assert compared["selected_intervals_right_only"] == [
        {"start": 4.0, "end": 5.0, "duration_sec": 1.0}
    ]
    assert compared["selected_interval_comparison_status"] == "coverage_only_order_not_compared"


def test_different_verified_input_bytes_do_not_compare_source_timestamps():
    left = extract_run(_result("left"), media_sha256="a" * 64)
    right = extract_run(_result("right"), media_sha256="b" * 64)
    compared = compare_runs(left, right)
    assert compared["source_media_status"] == "verified_different_bytes"
    assert compared["selected_interval_comparison_available"] is False
    assert compared["selected_intervals_left_only"] is None
    assert compared["code_regression_attribution"] == "not_supported_input_media_bytes_unverified_or_changed"


def test_different_source_keys_need_verified_equal_media_bytes_for_interval_comparison():
    left = extract_run(_result("left", source="a/movie.mp4"))
    right = extract_run(_result("right", source="b/movie.mp4"))
    assert compare_runs(left, right)["selected_interval_comparison_available"] is False
    left = extract_run(_result("left", source="a/movie.mp4"), media_sha256="a" * 64)
    right = extract_run(_result("right", source="b/movie.mp4"), media_sha256="a" * 64)
    compared = compare_runs(left, right)
    assert compared["same_source_key"] is False
    assert compared["source_media_status"] == "verified_same_bytes"
    assert compared["selected_interval_comparison_available"] is True


def test_malformed_selected_span_marks_interval_comparison_incomplete():
    left = extract_run(_result("left", selected=[{"start": 0, "end": 1}, {"start": 4, "end": 3}]))
    right = extract_run(_result("right"))
    compared = compare_runs(left, right)
    assert left["selected_invalid_span_count"] == 1
    assert compared["selected_interval_comparison_status"] == "invalid_selected_spans"
    assert compared["selected_left_only_sec"] is None


def test_missing_selected_is_not_reported_as_a_complete_empty_selection():
    raw = _result("left")
    del raw["selected"]
    left = extract_run(raw)
    right = extract_run(_result("right"))
    assert left["selected_invalid_span_count"] == 1
    assert compare_runs(left, right)["selected_interval_comparison_available"] is False


def test_cli_json_reports_all_three_pairwise_comparisons_and_rejects_unknown_media_hash(tmp_path, capsys):
    paths = []
    for i in range(3):
        path = tmp_path / f"result{i}.json"
        path.write_text(json.dumps(_result(f"project{i}", canonical=f"asr-{i}")), encoding="utf-8")
        paths.append(str(path))
    assert main(["--format", "json", *paths]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["runs"]) == 3
    assert len(report["comparisons"]) == 3
    assert {row["asr_status"] for row in report["comparisons"]} == {"different_canonical_asr"}
    with pytest.raises(SystemExit) as error:
        main(["--media-sha256", f"unknown={'a' * 64}", *paths])
    assert error.value.code == 2


def test_media_sha256_rejects_output_identity_or_bad_checksum():
    with pytest.raises(ValueError, match="64-digit hexadecimal"):
        extract_run(_result("left"), media_sha256="package-checksum")


def test_same_source_key_different_downloaded_bytes_are_different_inputs(tmp_path):
    original = tmp_path / "same-key.mp4"
    original.write_bytes(b"the original source media" * 10000)
    sha_a = _sha256_source_media(original)
    assert sha_a == hashlib.sha256(original.read_bytes()).hexdigest()
    first = _result("first")
    first["source_media_sha256"] = sha_a

    original.write_bytes(b"different bytes under the same S3 key" * 10000)
    sha_b = _sha256_source_media(original)
    second = _result("second")
    second["source_media_sha256"] = sha_b
    compared = compare_runs(extract_run(first), extract_run(second))
    assert compared["same_source_key"] is True
    assert compared["source_media_status"] == "verified_different_bytes"
    assert compared["selected_interval_comparison_available"] is False
    assert compared["code_regression_proven"] is False


def test_downloaded_source_checksum_fails_closed_and_cannot_be_overridden(tmp_path):
    with pytest.raises(FileNotFoundError):
        _sha256_source_media(tmp_path / "download-failed.mp4")
    raw = _result("example")
    raw["source_media_sha256"] = "a" * 64
    assert extract_run(raw)["media_sha256_evidence"] == "downloaded_source_bytes"
    with pytest.raises(ValueError, match="does not match"):
        extract_run(raw, media_sha256="b" * 64)
    raw["source_media_sha256"] = "render-not-source"
    with pytest.raises(ValueError, match="recorded source_media_sha256"):
        extract_run(raw)
