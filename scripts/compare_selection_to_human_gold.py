"""Read-only interval comparison of an engine result against a Gold manifest.

Gold labels are evaluation inputs only; do not import this module in runtime.
Input: result JSON's selected [{start,end}, ...] and a JSON manifest with
keep [[start,end], ...]. Rounded labels imply approximate results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _union(windows):
    ordered = sorted((float(start), float(end)) for start, end in windows)
    if any(not (math.isfinite(start) and math.isfinite(end)) or start < 0 or start >= end
           for start, end in ordered):
        raise ValueError("intervals must be finite with 0 <= start < end")
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals):
    return sum(end - start for start, end in intervals)


def _overlap(left, right):
    return sum(max(0.0, min(b, d) - max(a, c)) for a, b in left for c, d in right)


def compare(result, gold):
    expected_key = gold.get("source_key")
    if expected_key and result.get("source_key") != expected_key:
        raise ValueError("source key differs from Gold manifest")
    keep = _union(gold["keep"])
    selected = _union((row["start"], row["end"]) for row in result["selected"])
    kept = _overlap(keep, selected)
    gold_seconds = _duration(keep)
    selected_seconds = _duration(selected)
    return {
        "source_key": result.get("source_key"),
        "gold_seconds": round(gold_seconds, 3),
        "selected_union_seconds": round(selected_seconds, 3),
        "approved_seconds_retained": round(kept, 3),
        "approved_seconds_lost": round(gold_seconds - kept, 3),
        "unwanted_seconds_retained": round(selected_seconds - kept, 3),
        "recall": round(kept / gold_seconds, 4) if gold_seconds else None,
        "precision": round(kept / selected_seconds, 4) if selected_seconds else None,
        "technical_delivery_status": (result.get("live_render_qc") or {}).get("delivery_status"),
        "metric_note": "Rounded Gold intervals are diagnostic; inspect rendered MP4 for acceptance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("gold", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    print(json.dumps(compare(result, gold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
