"""Recalculate interval overlap for saved selector outputs against historical labels.

This is an offline metric replay only. It does not invoke ASR, AV, selection,
Boundary, or rendering. The manifest deliberately keeps separate historical runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def union(rows):
    ordered = sorted((float(row["start"]), float(row["end"])) for row in rows)
    merged = []
    for start, end in ordered:
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
            raise ValueError(f"invalid interval [{start}, {end}]")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def duration(windows):
    return sum(end - start for start, end in windows)


def overlap(left, right):
    return sum(max(0.0, min(b, d) - max(a, c)) for a, b in left for c, d in right)


def replay(manifest):
    source_keys = [source["source_key"] for source in manifest["sources"]]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("manifest contains duplicate source keys")
    output = []
    for source in manifest["sources"]:
        gold_raw = source.get("gold_keep_intervals")
        gold = union([{"start": a, "end": b} for a, b in gold_raw]) if gold_raw else None
        for observation in source["observations"]:
            selected_raw = observation.get("selected_intervals")
            row = {"source_key": source["source_key"], "kind": observation["kind"],
                   "run_id": observation.get("run_id"), "artifact_id": observation.get("artifact_id"),
                   "selection_status": observation.get("selection_status"),
                   "source_sha256": observation.get("source_sha256"),
                   "label_provenance_status": source.get("label_provenance_status", "unverified")}
            if not gold:
                row["metric_status"] = source.get("gold_status", "unscored_gold_missing")
            elif selected_raw is None:
                row["metric_status"] = observation.get("status", "no_selected_intervals")
            else:
                selected = union(selected_raw)
                retained = overlap(gold, selected)
                gold_seconds, selected_seconds = duration(gold), duration(selected)
                row["metric_status"] = "replayed"
                row["metrics"] = {
                    "gold_keep_seconds": round(gold_seconds, 3),
                    "selected_union_seconds": round(selected_seconds, 3),
                    "keep_retained_seconds": round(retained, 3),
                    "keep_lost_seconds": round(gold_seconds - retained, 3),
                    "delete_retained_seconds": round(selected_seconds - retained, 3),
                    "recall": round(retained / gold_seconds, 4) if gold_seconds else None,
                    "precision": round(retained / selected_seconds, 4) if selected_seconds else None,
                }
                reported = observation.get("reported_metric")
                if reported:
                    row["reported_metric_comparison"] = {
                        key: {"reported": reported.get(src), "replayed": row["metrics"].get(dst)}
                        for key, src, dst in (
                            ("keep_retained", "keep_retained_sec", "keep_retained_seconds"),
                            ("keep_lost", "keep_lost_sec", "keep_lost_seconds"),
                            ("delete_retained", "delete_retained_sec", "delete_retained_seconds"),
                        ) if reported.get(src) is not None
                    }
            output.append(row)
    return {"manifest_schema": manifest["schema"],
            "replay_scope": "saved selected interval vs provisional historical label; not an editorial acceptance score",
            "observations": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, help="write replay JSON here instead of stdout")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    serialized = json.dumps(replay(data), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
