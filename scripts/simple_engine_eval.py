#!/usr/bin/env python3
"""Score simple-engine cuts against the Human Gold (benchmarks/simple_engine_gold/gold_v1.json).

A Gold decision is one segment the Product Owner marked keep or delete. The engine "agrees" when it
kept at least half of a keep segment, or less than half of a delete segment.

  python scripts/simple_engine_eval.py                 # score the frozen reference run (no cost)
  python scripts/simple_engine_eval.py results_dir     # score results_dir/<video id>.json
                                                       # each file: {"splits":[{"start","end","visible"}]}

The Gold is QA-only. It must never be shown to the engine's prompts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GOLD = Path(__file__).resolve().parents[1] / "benchmarks" / "simple_engine_gold" / "gold_v1.json"


def score_video(decisions, visible):
    """-> (total, agree, removed_wrongly, kept_wrongly)"""
    total = agree = removed = kept = 0
    for d in decisions:
        covered = sum(max(0.0, min(d["e"], b) - max(d["s"], a)) for a, b in visible) / (d["e"] - d["s"])
        engine = "keep" if covered >= 0.5 else "delete"
        total += 1
        agree += engine == d["mark"]
        removed += d["mark"] == "keep" and engine == "delete"
        kept += d["mark"] == "delete" and engine == "keep"
    return total, agree, removed, kept


def score(results_dir: str | None = None, gold_path: Path = GOLD) -> dict:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    rows = {}
    for video in gold["videos"]:
        if results_dir is None:
            visible = video.get("reference_visible")
        else:
            path = Path(results_dir) / f"{video['id']}.json"
            visible = ([[s["start"], s["end"]] for s in json.loads(path.read_text())["splits"] if s["visible"]]
                       if path.exists() else None)
        if visible is None:
            continue
        rows[video["id"]] = score_video(video["decisions"], visible)
    total = sum(r[0] for r in rows.values())
    agree = sum(r[1] for r in rows.values())
    return {"videos": rows, "total": total, "agree": agree, "percent": round(100.0 * agree / total, 1) if total else 0.0}


def main(argv: list[str]) -> int:
    result = score(argv[1] if len(argv) > 1 else None)
    print(f"{'video':6} {'decisions':>9} {'agree':>6} {'removed wrongly':>16} {'kept wrongly':>13}")
    for name, (total, agree, removed, kept) in result["videos"].items():
        print(f"{name:6} {total:9d} {agree:6d} {removed:16d} {kept:13d}")
    print(f"TOTAL  {result['total']:9d} {result['agree']:6d}  = {result['percent']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
