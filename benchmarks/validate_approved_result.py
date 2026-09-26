"""Validate a benchmark result against an explicitly approved source timeline.

QA-only: the approval JSON never enters production selection.  Clip identifiers
may change; ordered source spans, spoken content, deliverability and render QC
are the contract.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]+(?:[-–][0-9]+)?%?", re.IGNORECASE)


def _tokens(text: str) -> tuple[str, ...]:
    out = []
    for token in _TOKEN_RE.findall(str(text or "").casefold()):
        raw = unicodedata.normalize("NFKD", token)
        out.append("".join(ch for ch in raw if not unicodedata.combining(ch)))
    return tuple(out)


def _coverage(expected: tuple[str, ...], actual: tuple[str, ...]) -> float:
    if not expected:
        return 0.0
    expected_set, actual_set = set(expected), set(actual)
    return len(expected_set & actual_set) / len(expected_set)


def validate(result_path: str, approval_path: str) -> tuple[bool, dict]:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    approval = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    failures: list[dict] = []
    selected = list(result.get("selected") or ())

    expected_sha = str(approval.get("source_sha256") or "")
    observed_sha = str(result.get("source_media_sha256") or "")
    if expected_sha and observed_sha != expected_sha:
        failures.append({"kind": "source_identity", "expected": expected_sha, "actual": observed_sha})

    expected_rows = list(approval.get("selected") or ())
    if len(selected) != len(expected_rows):
        failures.append({"kind": "selected_count", "expected": len(expected_rows), "actual": len(selected)})

    boundary_tolerance = float(approval.get("boundary_tolerance_sec") or 0.0)
    text_floor = float(approval.get("text_coverage_floor") or 1.0)
    for index, (expected, actual) in enumerate(zip(expected_rows, selected)):
        for edge in ("start", "end"):
            delta = abs(float(actual.get(edge, 0.0)) - float(expected[edge]))
            if delta > boundary_tolerance + 1e-9:
                failures.append({
                    "kind": "boundary", "selected_index": index, "edge": edge,
                    "expected": expected[edge], "actual": actual.get(edge), "delta_sec": round(delta, 3),
                })
        expected_tokens = _tokens(expected.get("text") or "")
        actual_tokens = _tokens(actual.get("text") or "")
        forward = _coverage(expected_tokens, actual_tokens)
        reverse = _coverage(actual_tokens, expected_tokens)
        if min(forward, reverse) < text_floor:
            failures.append({
                "kind": "spoken_content", "selected_index": index,
                "expected_coverage": round(forward, 4), "actual_coverage": round(reverse, 4),
            })

    duration_expected = float(approval.get("output_duration_sec") or 0.0)
    duration_tolerance = float(approval.get("output_duration_tolerance_sec") or 0.0)
    duration_actual = float(result.get("output_duration_sec") or 0.0)
    if duration_expected and abs(duration_actual - duration_expected) > duration_tolerance + 1e-9:
        failures.append({
            "kind": "output_duration", "expected": duration_expected, "actual": duration_actual,
            "delta_sec": round(abs(duration_actual - duration_expected), 3),
        })

    if approval.get("require_deliverable", True) and result.get("deliverable") is not True:
        failures.append({"kind": "deliverable", "actual": result.get("deliverable")})
    expected_qc = str(approval.get("live_render_qc_status") or "")
    observed_qc = str((result.get("live_render_qc") or {}).get("status") or "")
    if expected_qc and observed_qc != expected_qc:
        failures.append({"kind": "live_render_qc", "expected": expected_qc, "actual": observed_qc})

    report = {
        "schema_version": "cutsell.approved_result_validation.v1",
        "approval_id": approval.get("approval_id"),
        "pass": not failures,
        "selected_count": len(selected),
        "output_duration_sec": duration_actual,
        "failures": failures,
    }
    return not failures, report


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: validate_approved_result.py RESULT_JSON APPROVAL_JSON", file=sys.stderr)
        return 2
    ok, report = validate(argv[1], argv[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
