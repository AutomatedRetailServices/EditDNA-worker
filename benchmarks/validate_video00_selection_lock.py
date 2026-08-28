from __future__ import annotations

import json
import os
import sys

try:
    from benchmarks.validate_video00_regression_qa import validate as validate_regression_qa
except ModuleNotFoundError:
    from validate_video00_regression_qa import validate as validate_regression_qa


def _load(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def validate(result_path: str, lock_path: str) -> tuple[bool, dict]:
    result = _load(result_path)
    lock = _load(lock_path)

    actual = result.get("selected") or []
    expected = lock.get("segments") or []
    errors = []

    expected_count = int(lock.get("selected_count") or len(expected))
    actual_count = int(result.get("selected_count") or len(actual))
    if actual_count != expected_count or len(actual) != len(expected):
        errors.append({
            "reason": "selected_count_changed",
            "expected": expected_count,
            "actual": actual_count,
            "expected_rows": len(expected),
            "actual_rows": len(actual),
        })

    max_len = max(len(actual), len(expected))
    for index in range(max_len):
        if index >= len(expected):
            errors.append({
                "reason": "unexpected_segment",
                "index": index,
                "actual_text": _norm((actual[index] or {}).get("text")),
            })
            continue
        if index >= len(actual):
            errors.append({
                "reason": "missing_segment",
                "index": index,
                "expected_text": _norm((expected[index] or {}).get("text")),
            })
            continue

        got = actual[index] or {}
        want = expected[index] or {}
        got_text = _norm(got.get("text"))
        want_text = _norm(want.get("text"))
        if got_text != want_text:
            errors.append({
                "reason": "text_changed",
                "index": index,
                "expected": want_text,
                "actual": got_text,
            })

    qa_path = os.path.join(os.path.dirname(lock_path), "video00_regression_qa.json")
    qa_ok = True
    qa_report = None
    if os.path.exists(qa_path):
        qa_ok, qa_report = validate_regression_qa(result_path, qa_path)

    report = {
        "schema_version": lock.get("schema_version"),
        "baseline_run_id": lock.get("baseline_run_id"),
        "expected_selected_count": len(expected),
        "actual_selected_count": len(actual),
        "selection_locked": not errors,
        "historical_regression_qa_pass": qa_ok,
        "identity_rule": "ordered normalized selected text + selected count; clip_id and start/end intentionally ignored",
        "error_count": len(errors),
        "errors": errors[:100],
        "historical_regression_qa": qa_report,
    }
    return (not errors) and qa_ok, report


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_video00_selection_lock.py RESULT_JSON LOCK_JSON", file=sys.stderr)
        return 2
    ok, report = validate(sys.argv[1], sys.argv[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
