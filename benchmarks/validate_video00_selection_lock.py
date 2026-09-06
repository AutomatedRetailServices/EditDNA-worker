from __future__ import annotations

import json
import os
import sys

try:
    from benchmarks.validate_video00_regression_qa import validate as validate_regression_qa
    from benchmarks.video00_semantic_alignment import align
except ModuleNotFoundError:
    from validate_video00_regression_qa import validate as validate_regression_qa
    from video00_semantic_alignment import align


def _load(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def validate(result_path: str, lock_path: str) -> tuple[bool, dict]:
    """D-032: compares the candidate's `selected` list against the Human
    Gold baseline via ORDERED SEMANTIC ALIGNMENT (`video00_semantic_
    alignment.align`), not index-by-index positional text equality. See
    that module's own docstring for exactly why the old positional diff
    cascaded a single benign re-chunking into a wall of false errors (RAW
    33402023395) and how alignment tolerates it without hiding real loss.
    """
    result = _load(result_path)
    lock = _load(lock_path)

    actual_rows = result.get("selected") or []
    expected_rows = lock.get("segments") or []
    actual_texts = [_norm((row or {}).get("text")) for row in actual_rows]
    expected_texts = [_norm((row or {}).get("text")) for row in expected_rows]

    alignment = align(expected_texts, actual_texts)

    errors = []
    for row in alignment.rows:
        if row.relation == "MISSING":
            errors.append({
                "reason": "missing_segment",
                "gold_index_range": list(row.gold_span),
                "expected_text": row.gold_text,
            })
    for index in alignment.duplicate_candidate_indices:
        errors.append({
            "reason": "duplicate_segment",
            "candidate_index": index,
            "actual_text": actual_texts[index],
        })

    # A changed segment count is recorded for observability, never treated
    # as an error by itself -- benign re-chunking changes count while every
    # idea stays fully present (D-032; see also the canonical directive's
    # explicit "do not force the result to have exactly N segments").
    expected_count = int(lock.get("selected_count") or len(expected_rows))
    actual_count = int(result.get("selected_count") or len(actual_rows))

    qa_path = os.path.join(os.path.dirname(lock_path), "video00_regression_qa.json")
    qa_ok = True
    qa_report = None
    if os.path.exists(qa_path):
        qa_ok, qa_report = validate_regression_qa(result_path, qa_path)

    alignment_rows = [
        {
            "relation": row.relation,
            "gold_span": list(row.gold_span),
            "candidate_span": list(row.candidate_span),
            "gold_text": row.gold_text,
            "candidate_text": row.candidate_text,
            "content_coverage": row.content_coverage,
        }
        for row in alignment.rows
    ]

    report = {
        "schema_version": lock.get("schema_version"),
        "baseline_run_id": lock.get("baseline_run_id"),
        "expected_selected_count": expected_count,
        "actual_selected_count": actual_count,
        "selection_locked": alignment.aligned and qa_ok,
        "historical_regression_qa_pass": qa_ok,
        "identity_rule": (
            "ordered semantic alignment (content-token coverage, tolerant of benign "
            "re-chunking and minor ASR wording variance) -- not positional index "
            "equality; clip_id and start/end intentionally ignored"
        ),
        "alignment": alignment_rows,
        "extra_candidate_indices": list(alignment.extra_candidate_indices),
        "duplicate_candidate_indices": list(alignment.duplicate_candidate_indices),
        "error_count": len(errors),
        "errors": errors[:100],
        "historical_regression_qa": qa_report,
    }
    return (alignment.aligned and qa_ok), report


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_video00_selection_lock.py RESULT_JSON LOCK_JSON", file=sys.stderr)
        return 2
    ok, report = validate(sys.argv[1], sys.argv[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
