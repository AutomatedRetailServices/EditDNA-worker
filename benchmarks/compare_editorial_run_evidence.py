"""Read-only comparison of selected source intervals across engine results.

The same ``source_key`` identifies a location, not immutable source bytes.
The ASR config fingerprint identifies settings, not an identical transcript.
In particular, ``evidence_hash`` includes a dispatch-scoped source asset ID;
``content_hash`` and ``canonical_equivalence_hash`` are the cross-run ASR
identities. None of these hashes is a checksum of the input video.

Run with two or more result JSON files, e.g.::

    python -m benchmarks.compare_editorial_run_evidence first.json second.json

Optionally provide a verified SHA-256 of the *input video* for each project::

    python -m benchmarks.compare_editorial_run_evidence --format json \
        --media-sha256 project-one=<64-hex-digits> \
        --media-sha256 project-two=<64-hex-digits> first.json second.json

The tool never calls ASR, changes editorial decisions, or attributes a
selection difference to code alone. An identical input and transcript would
only make a controlled replay possible; it would not establish causation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _intervals(selected: object) -> tuple[list[tuple[float, float]], int]:
    intervals: list[tuple[float, float]] = []
    invalid = 0
    if not isinstance(selected, list):
        return intervals, 1  # missing selection is not an empty, valid selection
    for row in selected:
        if not isinstance(row, Mapping):
            invalid += 1
            continue
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        # Reject NaN/Infinity and inverted/zero spans, avoiding misleading
        # aggregate duration reports when a malformed row sneaks into JSON.
        if not (float("-inf") < start < end < float("inf")):
            invalid += 1
            continue
        intervals.append((start, end))
    return intervals, invalid


def _union(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _subtract(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Portions of source timeline selected by left and absent in right."""
    result: list[tuple[float, float]] = []
    rights = _union(right)
    for start, end in _union(left):
        cursor = start
        for other_start, other_end in rights:
            if other_end <= cursor:
                continue
            if other_start >= end:
                break
            if other_start > cursor:
                result.append((cursor, min(other_start, end)))
            cursor = min(end, max(cursor, other_end))
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def _round_intervals(intervals: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    return [
        {"start": round(start, 3), "end": round(end, 3), "duration_sec": round(end - start, 3)}
        for start, end in intervals
    ]


def _duration(intervals: Sequence[tuple[float, float]]) -> float:
    return round(sum(end - start for start, end in intervals), 3)


def extract_run(result: Mapping[str, Any], *, media_sha256: str | None = None) -> dict[str, Any]:
    """Summarize only recorded metadata; old-run media checksum may be supplied.

    The engine's render/output hashes (and package SHA-256) are deliberately
    ignored: none establishes the identity of the source media bytes. New
    runs record the source's real SHA-256 at the download-before-ASR seam.
    """
    if media_sha256 is not None and not _SHA256.fullmatch(media_sha256):
        raise ValueError("media_sha256 must be a verified 64-digit hexadecimal SHA-256")
    recorded_sha = result.get("source_media_sha256")
    if recorded_sha is not None and (not isinstance(recorded_sha, str) or not _SHA256.fullmatch(recorded_sha)):
        raise ValueError("recorded source_media_sha256 must be a 64-digit hexadecimal SHA-256")
    if recorded_sha is not None and media_sha256 is not None and recorded_sha.lower() != media_sha256.lower():
        raise ValueError("supplied source-media SHA-256 does not match the recorded download")
    stage_status = result.get("stage_status")
    canonical = (stage_status.get("canonical_asr_evidence") or {}) if isinstance(stage_status, Mapping) else {}
    if not isinstance(canonical, Mapping) or canonical.get("status") != "complete":
        canonical = {}
    raw_selected = result.get("selected")
    intervals, invalid_count = _intervals(raw_selected)
    identity = result.get("active_path_identity") or {}
    return {
        "project_id": str(result.get("project_id") or ""),
        "build_git_sha": str(identity.get("build_git_sha") or "") if isinstance(identity, Mapping) else "",
        "source_key": str(result.get("source_key") or ""),
        "media_sha256": (recorded_sha or media_sha256 or "").lower() or None,
        "media_sha256_evidence": (
            "downloaded_source_bytes" if recorded_sha is not None
            else "externally_verified" if media_sha256 is not None else "unavailable"
        ),
        "asr_config_fingerprint": canonical.get("asr_config_fingerprint") or None,
        "asr_content_hash": canonical.get("content_hash") or None,
        "asr_canonical_equivalence_hash": canonical.get("canonical_equivalence_hash") or None,
        "asr_evidence_hash_source_scoped": canonical.get("evidence_hash") or None,
        "normalized_word_count": canonical.get("normalized_word_count"),
        "selected_clip_count": len(raw_selected) if isinstance(raw_selected, list) else 0,
        "selected_valid_span_count": len(intervals),
        "selected_invalid_span_count": invalid_count,
        "selected_source_intervals": [[start, end] for start, end in intervals],
        "selected_source_coverage_sec": _duration(_union(intervals)),
    }


def compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare run summaries without inferring which editorial choice is good."""
    left_key, right_key = left.get("source_key"), right.get("source_key")
    left_hash, right_hash = left.get("media_sha256"), right.get("media_sha256")
    same_key = bool(left_key and right_key and left_key == right_key)
    if left_hash and right_hash:
        media_status = "verified_same_bytes" if left_hash == right_hash else "verified_different_bytes"
    elif same_key:
        media_status = "same_source_key_bytes_unverified"
    else:
        media_status = "different_or_missing_source_keys_bytes_unverified"

    asr_keys = ("asr_content_hash", "asr_canonical_equivalence_hash", "asr_config_fingerprint")
    if not all(left.get(key) and right.get(key) for key in asr_keys):
        asr_status = "missing_asr_evidence"
        attribution = "not_supported_missing_asr_evidence"
    elif left["asr_config_fingerprint"] != right["asr_config_fingerprint"]:
        asr_status = "different_asr_config"
        attribution = "not_supported_asr_config_changed"
    elif left["asr_canonical_equivalence_hash"] != right["asr_canonical_equivalence_hash"]:
        asr_status = "different_canonical_asr"
        attribution = "not_supported_asr_changed"
    elif left["asr_content_hash"] != right["asr_content_hash"]:
        asr_status = "canonical_equivalent_but_word_text_differs"
        attribution = "not_supported_asr_word_text_changed"
    else:
        asr_status = "same_content_and_canonical_hashes"
        attribution = (
            "eligible_for_controlled_replay_not_proof_of_code_regression"
            if media_status == "verified_same_bytes"
            else "not_supported_input_media_bytes_unverified_or_changed"
        )

    # Different media hashes make source-time differences meaningless.
    # Equal source keys without a checksum permit a descriptive comparison
    # but never proof that the underlying source bytes were equal.
    source_alignment_available = media_status == "verified_same_bytes" or (
        same_key and media_status == "same_source_key_bytes_unverified"
    )
    comparable_intervals = source_alignment_available and not (
        left.get("selected_invalid_span_count") or right.get("selected_invalid_span_count")
    )
    left_only: list[tuple[float, float]] = []
    right_only: list[tuple[float, float]] = []
    if comparable_intervals:
        left_intervals = [tuple(interval) for interval in left.get("selected_source_intervals", ())]
        right_intervals = [tuple(interval) for interval in right.get("selected_source_intervals", ())]
        left_only = _subtract(left_intervals, right_intervals)
        right_only = _subtract(right_intervals, left_intervals)
    return {
        "left_project_id": left.get("project_id"),
        "right_project_id": right.get("project_id"),
        "same_source_key": same_key,
        "source_media_status": media_status,
        "asr_status": asr_status,
        "code_regression_attribution": attribution,
        "code_regression_proven": False,
        "selected_interval_comparison_available": comparable_intervals,
        "selected_interval_comparison_status": (
            "coverage_only_order_not_compared" if comparable_intervals
            else "invalid_selected_spans" if source_alignment_available
            else "source_media_alignment_unavailable"
        ),
        "selected_intervals_left_only": _round_intervals(left_only) if comparable_intervals else None,
        "selected_intervals_right_only": _round_intervals(right_only) if comparable_intervals else None,
        "selected_left_only_sec": _duration(left_only) if comparable_intervals else None,
        "selected_right_only_sec": _duration(right_only) if comparable_intervals else None,
    }


def _media_hashes(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--media-sha256 needs PROJECT_ID=64_HEX_SOURCE_MEDIA_SHA256")
        project_id, checksum = value.rsplit("=", 1)
        if not project_id or not _SHA256.fullmatch(checksum):
            raise ValueError("--media-sha256 needs PROJECT_ID=64_HEX_SOURCE_MEDIA_SHA256")
        if project_id in result:
            raise ValueError(f"duplicate media SHA-256 for project {project_id}")
        result[project_id] = checksum
    return result


def _table(comparisons: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        "| Corridas | Medio origen | ASR | Solo izquierda | Solo derecha | Atribución al código |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in comparisons:
        rows.append("| " + " | ".join([
            f"{item['left_project_id']} → {item['right_project_id']}",
            str(item["source_media_status"]),
            str(item["asr_status"]),
            str(item["selected_left_only_sec"]) if item["selected_left_only_sec"] is not None else "n/a",
            str(item["selected_right_only_sec"]) if item["selected_right_only_sec"] is not None else "n/a",
            str(item["code_regression_attribution"]),
        ]) + " |")
    rows.append("\nDuraciones = partes del origen seleccionadas solo en esa corrida. No miden calidad ni orden editorial.")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="at least two engine result JSON files")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--media-sha256", action="append", default=[], metavar="PROJECT_ID=SHA256")
    args = parser.parse_args(argv)
    if len(args.results) < 2:
        parser.error("provide at least two result JSON files")
    try:
        supplied_hashes = _media_hashes(args.media_sha256)
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        if not all(isinstance(value, dict) for value in payloads):
            raise ValueError("every result JSON must be an object")
        ids = [str(value.get("project_id") or "") for value in payloads]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("results must have distinct, nonempty project_id values")
        if set(supplied_hashes) - set(ids):
            raise ValueError("media hashes were given for project IDs absent from the results")
        runs = [extract_run(value, media_sha256=supplied_hashes.get(value["project_id"])) for value in payloads]
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    comparisons = [compare_runs(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    if args.format == "json":
        print(json.dumps({"runs": runs, "comparisons": comparisons}, indent=2, ensure_ascii=False))
    else:
        print(_table(comparisons))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
