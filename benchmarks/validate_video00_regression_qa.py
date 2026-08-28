from __future__ import annotations

import json
import sys
import unicodedata


def _load(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    raw = unicodedata.normalize("NFKC", str(text or ""))
    return " ".join(raw.split()).casefold()


def _selected_texts(result: dict) -> list[str]:
    return [_norm((row or {}).get("text")) for row in (result.get("selected") or [])]


def _find_exact(texts: list[str], target: str) -> int | None:
    target = _norm(target)
    for index, text in enumerate(texts):
        if text == target:
            return index
    return None


def validate(result_path: str, manifest_path: str) -> tuple[bool, dict]:
    result = _load(result_path)
    manifest = _load(manifest_path)
    texts = _selected_texts(result)
    joined = "\n".join(texts)
    failures: list[dict] = []
    passes: list[str] = []

    expected_count = int(manifest.get("expected_selected_count") or 0)
    if expected_count:
        actual_count = len(texts)
        if actual_count != expected_count:
            failures.append({
                "id": "selection_count_23",
                "kind": "count",
                "expected": expected_count,
                "actual": actual_count,
            })
        else:
            passes.append("selection_count_23")

    for check in manifest.get("checks") or []:
        check_id = str(check.get("id") or "unnamed")
        kind = str(check.get("kind") or "")

        if kind == "required_exact":
            index = _find_exact(texts, check.get("text"))
            if index is None:
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
            cursor = -1
            missing = False
            for item in wanted:
                target = _norm(item)
                found = None
                for index in range(cursor + 1, len(texts)):
                    if texts[index] == target:
                        found = index
                        break
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
