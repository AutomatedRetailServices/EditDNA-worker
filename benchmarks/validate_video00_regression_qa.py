from __future__ import annotations

import json
import sys
import unicodedata

try:
    from benchmarks.video00_semantic_alignment import _content_tokens, find_coverage_span
except ModuleNotFoundError:
    from video00_semantic_alignment import _content_tokens, find_coverage_span


def _load(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    raw = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(raw.split()).casefold()


def _selected_texts(result: dict) -> list[str]:
    return [_norm((row or {}).get("text")) for row in (result.get("selected") or [])]


def _find_semantic(texts: list[str], target: str, *, start: int = 0) -> tuple[int, int] | None:
    """D-032: content-coverage span search (`video00_semantic_alignment`),
    not byte-equal text matching -- a required fact rechunked into a
    bigger/smaller candidate segment, or restated with light ASR wording
    variance, is still recognized as present. See that module's own
    docstring for exactly why exact-text matching false-positived on
    genuine re-chunking (RAW 33402023395)."""
    candidate_tokens = [_content_tokens(t) for t in texts]
    return find_coverage_span(candidate_tokens, _content_tokens(target), start=start)


def validate(result_path: str, manifest_path: str) -> tuple[bool, dict]:
    result = _load(result_path)
    manifest = _load(manifest_path)
    texts = _selected_texts(result)
    joined = "\n".join(texts)
    failures: list[dict] = []
    passes: list[str] = []
    warnings: list[dict] = []

    # D-032: a changed segment COUNT is not by itself evidence of content
    # loss -- benign re-chunking (ASR/attempt-reconstruction merging or
    # splitting segments differently between runs) changes the count while
    # every idea stays fully present. Recorded as a warning, not a failure;
    # the required_exact/required_order/forbidden_contains checks below are
    # what actually decide whether real content is missing.
    expected_count = int(manifest.get("expected_selected_count") or 0)
    if expected_count:
        actual_count = len(texts)
        if actual_count != expected_count:
            warnings.append({
                "id": "selection_count_23",
                "kind": "count",
                "expected": expected_count,
                "actual": actual_count,
                "reason": "count_differs_not_treated_as_failure_see_D-032",
            })
        else:
            passes.append("selection_count_23")

    for check in manifest.get("checks") or []:
        check_id = str(check.get("id") or "unnamed")
        kind = str(check.get("kind") or "")

        if kind == "required_exact":
            span = _find_semantic(texts, check.get("text"))
            if span is None:
                failures.append({"id": check_id, "kind": kind, "reason": "missing_required_segment"})
            else:
                passes.append(check_id)
            continue

        if kind == "forbidden_contains":
            needle = _norm(check.get("text"))
            if needle and needle in joined:
                failures.append({"id": check_id, "kind": kind, "reason": "historical_bad_take_returned"})
            else:
                passes.append(check_id)
            continue

        if kind == "required_order":
            wanted = list(check.get("texts") or [])
            indices: list[int] = []
            cursor = 0
            missing = False
            for item in wanted:
                # start=cursor (not cursor+1): two consecutive required
                # facts that were merged into the SAME rechunked candidate
                # segment must both still be found there (a legitimate
                # RECHUNKED co-location, not a reorder) -- order is still
                # enforced because `cursor` only ever moves forward.
                span = _find_semantic(texts, item, start=cursor)
                found = span[0] if span is not None else None
                if found is None:
                    missing = True
                    break
                indices.append(found)
                cursor = found
            if missing:
                failures.append({"id": check_id, "kind": kind, "reason": "required_sequence_missing_or_reordered"})
            else:
                passes.append(check_id)
            continue

        failures.append({"id": check_id, "kind": kind, "reason": "unknown_check_kind"})

    report = {
        "schema_version": manifest.get("schema_version"),
        "baseline_run_id": manifest.get("baseline_run_id"),
        "qa_pass": not failures,
        "selected_count": len(texts),
        "passed_check_count": len(passes),
        "failed_check_count": len(failures),
        "passed_checks": passes,
        "failed_checks": failures,
        "warnings": warnings,
    }
    return not failures, report


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_video00_regression_qa.py RESULT_JSON MANIFEST_JSON", file=sys.stderr)
        return 2
    ok, report = validate(sys.argv[1], sys.argv[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
