"""Resumable historical old-vs-new benchmark orchestration."""

import csv
import io
import json
import re
import tempfile
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import benchmark_s3

RESULT_FILES = ("results_v2.jsonl", "disagreements.jsonl", "unmatched.jsonl", "summary.json", "summary.csv", "errors.jsonl")


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(value or "").stem.casefold())


def resolve_sources(rows_by_session: dict[str, list[dict]], objects: list[dict]) -> tuple[dict, dict]:
    """Resolve only unique normalized/session-name matches; never use row order."""
    resolved, unresolved = {}, {}
    for session, rows in sorted(rows_by_session.items()):
        hints = {normalize(session)} | {normalize(str(row.get("source_name") or row.get("source") or "")) for row in rows}
        hints.discard("")
        exact = [obj for obj in objects if normalize(obj["key"]) in hints]
        candidates = exact or [obj for obj in objects if any(h and (h in normalize(obj["key"]) or normalize(obj["key"]) in h) for h in hints)]
        if len(candidates) == 1:
            resolved[session] = candidates[0]
        else:
            unresolved[session] = {"classification": "ambiguous_source" if candidates else "missing_source", "candidates": [x["key"] for x in candidates]}
    return resolved, unresolved


def match_clips(old: list[dict], new: list[dict]) -> tuple[list[dict], list[dict]]:
    matched, unmatched = [], []
    unused = set(range(len(new)))
    for historical in old:
        scored = []
        for index in unused:
            current = new[index]
            text_score = SequenceMatcher(None, str(historical.get("text", "")).casefold(), str(current.get("text", "")).casefold()).ratio()
            overlap = 0.0
            if all(x in historical for x in ("start", "end")) and all(x in current for x in ("start", "end")):
                union = max(float(historical["end"]), float(current["end"])) - min(float(historical["start"]), float(current["start"]))
                overlap = max(0.0, min(float(historical["end"]), float(current["end"])) - max(float(historical["start"]), float(current["start"]))) / max(union, .001)
            scored.append((max(text_score, overlap), index, "boundary_overlap" if overlap >= text_score else "transcript_similarity"))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < .55:
            unmatched.append({"classification": "unmatched_old", "old_clip": historical}); continue
        if len(scored) > 1 and scored[0][0] - scored[1][0] < .05:
            unmatched.append({"classification": "ambiguous_match", "old_clip": historical}); continue
        confidence, index, method = scored[0]; unused.remove(index)
        matched.append(make_result(historical, new[index], method, confidence))
    unmatched.extend({"classification": "new_unmatched", "new_clip": new[index]} for index in sorted(unused))
    return matched, unmatched


def make_result(old: dict, new: dict, method: str, confidence: float) -> dict:
    old_meta, new_meta = old.get("meta") or {}, new.get("meta") or {}
    old_keep, new_keep = bool(old.get("keep", old_meta.get("keep"))), bool(new.get("keep", new_meta.get("keep")))
    old_slot, new_slot = old.get("slot"), new.get("slot")
    return {"session_id": old.get("session_id"), "old_clip_id": old.get("clip_id", old.get("id")), "new_clip_id": new.get("id"),
            "text": new.get("text", old.get("text", "")), "old_keep": old_keep, "new_keep": new_keep,
            "old_slot": old_slot, "new_slot": new_slot, "old_reason": old.get("llm_reason", old.get("reason", "")),
            "new_reason": new.get("llm_reason", new.get("reason", "")), "semantic_v2": new_meta.get("semantic_v2", {}),
            "take_judge_v2": new_meta.get("take_judge_v2", {}), "match_method": method,
            "match_confidence": round(confidence, 4), "changed": old_keep != new_keep or old_slot != new_slot}


def run_benchmark(job_id: str, request: dict, s3=None, pipeline=None, progress=None) -> dict:
    s3, progress = s3 or benchmark_s3.client(), progress or (lambda state: None)
    started = time.monotonic(); prefix = f"{benchmark_s3.OUTPUT_PREFIX}{job_id}/"
    rows = [json.loads(line) for line in benchmark_s3.read_object(s3, request["dataset_key"]).decode().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows: grouped[str(row.get("session_id") or "")].append(row)
    objects = []
    for source_prefix in request["source_prefixes"]: objects.extend(benchmark_s3.list_objects(s3, source_prefix))
    resolved, unresolved = resolve_sources(grouped, objects)
    sessions = sorted(grouped)[:request.get("limit") or None]
    results, unmatched, errors = [], [], []
    completed = set(request.get("completed_sessions", []))
    for position, session in enumerate(sessions, 1):
        if session in completed: continue
        progress({"total_sessions": len(sessions), "processed_sessions": position - 1, "current_session": session})
        if session in unresolved:
            unmatched.append({"session_id": session, **unresolved[session]}); completed.add(session); continue
        if request["mode"] == "inventory_only":
            completed.add(session); continue
        try:
            if pipeline is None: raise RuntimeError("benchmark pipeline adapter is not configured")
            current = pipeline(session, resolved[session]["key"], bool(request.get("render_outputs")), request)
            matched, missing = match_clips(grouped[session], current.get("clips", [])); results.extend(matched); unmatched.extend(missing)
            completed.add(session)
        except Exception as exc:
            errors.append({"session_id": session, "error_type": type(exc).__name__, "message": str(exc)[:500]})
        progress({"total_sessions": len(sessions), "processed_sessions": position, "current_session": None, "completed_sessions": sorted(completed)})
    summary = build_summary(rows, results, unmatched, errors, time.monotonic() - started)
    disagreements = [row for row in results if row["changed"]] + [row for row in unmatched if row.get("classification") in {"ambiguous_match", "unmatched_old", "new_unmatched"}]
    payloads = {"results_v2.jsonl": jsonl(results), "disagreements.jsonl": jsonl(disagreements), "unmatched.jsonl": jsonl(unmatched),
                "errors.jsonl": jsonl(errors), "summary.json": json.dumps(summary, indent=2).encode(), "summary.csv": summary_csv(summary)}
    for name, body in payloads.items(): benchmark_s3.put_output(s3, prefix + name, body, job_id, "text/csv" if name.endswith(".csv") else "application/json")
    return {"summary": summary, "output_prefix": prefix, "result_keys": [prefix + name for name in RESULT_FILES], "completed_sessions": sorted(completed)}


def jsonl(rows): return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def build_summary(old, results, unmatched, errors, elapsed):
    same_keep = sum(x["old_keep"] == x["new_keep"] for x in results); same_slot = sum(x["old_slot"] == x["new_slot"] for x in results)
    return {"total_historical_clips": len(old), "total_matched_clips": len(results), "total_unmatched_old_clips": sum(x.get("classification") == "unmatched_old" for x in unmatched),
            "total_new_unmatched_clips": sum(x.get("classification") == "new_unmatched" for x in unmatched), "total_ambiguous_matches": sum("ambiguous" in x.get("classification", "") for x in unmatched),
            "same_keep_decision": same_keep, "different_keep_decision": len(results)-same_keep, "old_discard_new_keep": sum(not x["old_keep"] and x["new_keep"] for x in results),
            "old_keep_new_discard": sum(x["old_keep"] and not x["new_keep"] for x in results), "same_slot": same_slot, "different_slot": len(results)-same_slot,
            "per_slot_changes": {}, "per_source_session_changes": {}, "semantic_v2_usage": sum(bool(x["semantic_v2"]) for x in results), "semantic_v2_fallbacks": sum(not x["semantic_v2"] for x in results),
            "take_judge_v2_usage": sum(bool(x["take_judge_v2"]) for x in results), "take_judge_v2_fallbacks": sum(not x["take_judge_v2"] for x in results),
            "processing_time_seconds": round(elapsed, 3), "provider_usage": {}, "estimated_cost": None, "failed_sessions": len(errors)}


def summary_csv(summary):
    output = io.StringIO(); writer = csv.writer(output); writer.writerow(("metric", "value"))
    for key, value in summary.items(): writer.writerow((key, json.dumps(value) if isinstance(value, (dict, list)) else value))
    return output.getvalue().encode()
