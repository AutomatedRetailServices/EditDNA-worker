from __future__ import annotations

import json
import sys
from pathlib import Path


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

    if int(result.get("selected_count") or len(actual)) != int(lock.get("selected_count") or len(expected)):
        errors.append({
            "reason": "selected_count_changed",
            "expected": int(lock.get("selected_count") or len(expected)),
            "actual": int(result.get("selected_count") or len(actual)),
        })

    max_len = max(len(actual), len(expected))
    for index in range(max_len):
        if index >= len(expected):
            errors.append({"reason": "unexpected_segment", "index": index, "actual": actual[index]})
            continue
        if index >= len(actual):
            errors.append({"reason": "missing_segment", "index": index, "expected": expected[index]})
            continue
        got = actual[index]
        want = expected[index]
        if str(got.get("clip_id") or "") != str(want.get("clip_id") or ""):
            errors.append({
                "reason": "clip_id_changed",
                "index": index,
                "expected": want.get("clip_id"),
                "actual": got.get("clip_id"),
            })
        if _norm(got.get("text")) != _norm(want.get("text")):
            errors.append({
                "reason": "text_changed",
                "index": index,
                "clip_id": got.get("clip_id"),
                "expected": _norm(want.get("text")),
                "actual": _norm(got.get("text")),
            })

    report = {
        "schema_version": lock.get("schema_version"),
        "baseline_run_id": lock.get("baseline_run_id"),
        "expected_selected_count": len(expected),
        "actual_selected_count": len(actual),
        "selection_locked": not errors,
        "error_count": len(errors),
        "errors": errors[:100],
    }
    return not errors, report


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_video00_selection_lock.py RESULT_JSON LOCK_JSON", file=sys.stderr)
        return 2
    ok, report = validate(sys.argv[1], sys.argv[2])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
