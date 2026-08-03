"""Resumable historical old-vs-new benchmark orchestration."""

import csv
import hashlib
import io
import json
import os
import re
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import benchmark_s3

RESULT_FILES = ("results_v2.jsonl", "disagreements.jsonl", "unmatched.jsonl", "inventory.json",
                "summary.json", "summary.csv", "errors.jsonl")
GENERIC_SOURCE_HINTS = {"good", "bloopers", "blooper"}
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024


class RetryableBenchmarkSessionsError(RuntimeError):
    """Signals RQ to resume sessions that remain below their attempt limit."""


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(value or "").stem.casefold())


def resolve_sources(rows_by_session: dict[str, list[dict]], objects: list[dict]) -> tuple[dict, dict]:
    """Resolve unique exact names first; uncertain matches are never guessed."""
    resolved, unresolved = {}, {}
    for session, rows in sorted(rows_by_session.items()):
        session_hint = normalize(session)
        optional = {normalize(str(row.get("source_name") or "")) for row in rows}
        optional |= {normalize(str(row.get("source") or "")) for row in rows
                     if str(row.get("source") or "").casefold() not in GENERIC_SOURCE_HINTS}
        hints = ({session_hint} | optional) - {""}
        exact = [obj for obj in objects if normalize(Path(obj["key"]).name) in hints]
        if len(exact) == 1:
            resolved[session] = exact[0]
        else:
            unresolved[session] = {"classification": "ambiguous_source" if exact else "missing_source",
                                   "candidates": [item["key"] for item in exact]}
    return resolved, unresolved


def match_clips(old: list[dict], new: list[dict]) -> tuple[list[dict], list[dict]]:
    matched, unmatched, unused = [], [], set(range(len(new)))
    for historical in old:
        scored = []
        for index in unused:
            current = new[index]
            text_score = SequenceMatcher(None, str(historical.get("text", "")).casefold(),
                                         str(current.get("text", "")).casefold()).ratio()
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
    take_status = new_meta.get("take_judge_execution_status")
    return {"session_id": old.get("session_id"), "old_clip_id": old.get("clip_id", old.get("id")),
            "new_clip_id": new.get("id"), "text": new.get("text", old.get("text", "")),
            "old_keep": old_keep, "new_keep": new_keep, "old_slot": old_slot, "new_slot": new_slot,
            "old_reason": old.get("llm_reason", old.get("reason", "")),
            "new_reason": new.get("llm_reason", new.get("reason", "")),
            "semantic_v2": new_meta.get("semantic_v2", {}),
            "take_judge_v2": {"score": new_meta.get("take_judge_score"),
                              "verdict": new_meta.get("take_judge_verdict")},
            "take_judge_status": take_status,
            "take_judge_not_evaluated": new_meta.get("take_judge_verdict") is None,
            "take_judge_abstained": take_status == "abstained",
            "take_judge_unavailable": take_status in {"provider_unavailable", "provider_error"},
            "take_judge_low_confidence": take_status == "low_confidence",
            "match_method": method, "match_confidence": round(confidence, 4),
            "changed": old_keep != new_keep or old_slot != new_slot,
            "winner_selection_changed": bool(old.get("take_judge_verdict") is not None
                and new_meta.get("take_judge_verdict") is not None
                and old.get("take_judge_verdict") != new_meta.get("take_judge_verdict")),
            "take_judge_evaluation": "newly_evaluated" if old.get("take_judge_verdict") is None and new_meta.get("take_judge_verdict") is not None
                else "not_evaluated" if new_meta.get("take_judge_verdict") is None else "evaluated"}


def _load_checkpoint(s3, job_id, key):
    raw = benchmark_s3.read_output(s3, key, job_id)
    defaults = {"completed_sessions": [], "failed_sessions": [], "terminal_failed_sessions": [],
                "unresolved_sessions": [], "attempts_by_session": {}, "session_result_keys": {},
                "request_fingerprint": None}
    if raw:
        defaults.update(json.loads(raw))
    return defaults


def _save_checkpoint(s3, job_id, key, state):
    body = json.dumps(state, sort_keys=True).encode()
    if len(body) > MAX_CHECKPOINT_BYTES:
        raise ValueError("benchmark checkpoint exceeds bounded resumability limit")
    benchmark_s3.put_output(s3, key, body, job_id, "application/json")


def request_fingerprint(request: dict) -> str:
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def safe_session_id(session_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id).strip("_-")[:80] or "session"
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:12]
    return f"{slug}-{digest}"


def _session_key(prefix: str, session_id: str) -> str:
    return f"{prefix}sessions/{safe_session_id(session_id)}.json"


def _load_session_outputs(s3, job_id: str, keys: dict[str, str]) -> tuple[list, list, list, list, list]:
    results, unmatched, errors, inventory, usage = [], [], [], [], []
    for session in sorted(keys):
        raw = benchmark_s3.read_output(s3, keys[session], job_id, max_bytes=64 * 1024 * 1024)
        if raw is None:
            raise RuntimeError("completed benchmark session output is missing")
        item = json.loads(raw)
        results.extend(item.get("matched_results", [])); unmatched.extend(item.get("unmatched_results", []))
        errors.extend(item.get("errors", []))
        if item.get("inventory_entry"): inventory.append(item["inventory_entry"])
        usage.append({"provider_instrumentation": item.get("provider_instrumentation", {}),
                      "take_judge_execution": item.get("take_judge_execution", {})})
    return results, unmatched, errors, inventory, usage


def run_benchmark(job_id: str, request: dict, s3=None, pipeline=None, progress=None) -> dict:
    s3, progress = s3 or benchmark_s3.client(), progress or (lambda state: None)
    started, prefix = time.monotonic(), f"{benchmark_s3.OUTPUT_PREFIX}{job_id}/"
    rows = [json.loads(line) for line in benchmark_s3.read_object(s3, request["dataset_key"]).decode().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        session_id = str(row.get("session_id") or "")
        if not session_id or len(session_id) > 200:
            raise ValueError("historical session_id must contain 1 to 200 characters")
        grouped[session_id].append(row)
    objects, filtered = [], 0
    for source_prefix in request["source_prefixes"]:
        eligible, stats = benchmark_s3.list_objects_inventory(s3, source_prefix)
        objects.extend(eligible); filtered += stats["filtered_s3_objects"]
    resolved, unresolved = resolve_sources(grouped, objects)
    dataset_sessions = sorted(grouped)
    sessions = dataset_sessions[:request.get("limit") or None]
    selected_rows = [row for session in sessions for row in grouped[session]]
    checkpoint_key = prefix + "checkpoint.json"
    state = _load_checkpoint(s3, job_id, checkpoint_key)
    fingerprint = request_fingerprint(request)
    if state["request_fingerprint"] not in (None, fingerprint):
        raise ValueError("checkpoint request fingerprint does not match")
    state["request_fingerprint"] = fingerprint
    completed = set(state["completed_sessions"])
    max_attempts = max(1, int(os.getenv("BENCHMARK_SESSION_MAX_ATTEMPTS", "3")))
    _save_checkpoint(s3, job_id, checkpoint_key, state)
    environment = {name: bool(os.getenv(name)) for name in
                   ("S3_BUCKET", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")}
    preflight = {"dataset_jsonl_readable": True, "eligible_videos": len(objects),
        "resolved_sessions": sum(session in resolved for session in sessions),
        "missing_sessions": sum(unresolved.get(s, {}).get("classification") == "missing_source" for s in sessions),
        "ambiguous_sessions": sum(unresolved.get(s, {}).get("classification") == "ambiguous_source" for s in sessions),
        "required_environment_configuration_present": all(environment.values()),
        "required_environment_configuration": environment,
        "s3_read_access_successful": True, "s3_benchmark_prefix_write_access_successful": True}

    def report(current=None):
        processed = len(completed); failed = len(state["failed_sessions"])
        unresolved_count = len(state["unresolved_sessions"])
        progress({"total_sessions": len(sessions), "processed_sessions": processed,
                  "successful_sessions": max(0, processed - failed - unresolved_count),
                  "failed_sessions": failed, "unresolved_sessions": unresolved_count,
                  "current_session": current, "progress_percent": round(processed * 100 / len(sessions), 1) if sessions else 100})

    for session in sessions:
        if session in completed: continue
        report(session)
        resolution = unresolved.get(session)
        prior_key = state["session_result_keys"].get(session)
        prior = benchmark_s3.read_output(s3, prior_key, job_id, max_bytes=64 * 1024 * 1024) if prior_key else None
        session_output = json.loads(prior) if prior else {"matched_results": [], "unmatched_results": [], "errors": [],
                          "inventory_entry": {"session_id": session,
                            "resolution_status": resolution["classification"] if resolution else "resolved",
                            "resolved_s3_key": resolved[session]["key"] if not resolution else None,
                            "candidate_s3_keys": resolution["candidates"] if resolution else [resolved[session]["key"]],
                            "historical_clip_count": len(grouped[session])}, "provider_instrumentation": {},
                          "take_judge_execution": {"requested": False, "sources": []}}
        session_output["matched_results"] = []; session_output["unmatched_results"] = []
        if resolution:
            session_output["unmatched_results"].append({"session_id": session, **resolution})
            state["unresolved_sessions"].append(session)
            completed.add(session)
        elif request["mode"] != "inventory_only":
            state["attempts_by_session"][session] = int(state["attempts_by_session"].get(session, 0)) + 1
            try:
                current = pipeline(session, resolved[session]["key"], bool(request.get("render_outputs")), request)
                matched, missing = match_clips(grouped[session], current.get("clips", []))
                session_output["matched_results"] = matched; session_output["unmatched_results"] = missing
                session_output["provider_instrumentation"] = {"session_id": session,
                    "usage": current.get("provider_usage_instrumentation", {"available": False, "reason": "Pipeline did not report provider instrumentation"}),
                    "cost": current.get("estimated_cost_instrumentation", {"available": False, "reason": "Pipeline did not report cost instrumentation"})}
                session_output["take_judge_execution"] = current.get("take_judge_execution", {"requested": False, "sources": []})
                state["failed_sessions"] = [value for value in state["failed_sessions"] if value != session]
                completed.add(session)
            except Exception as exc:
                session_output["errors"].append({"session_id": session, "error_type": type(exc).__name__, "message": str(exc)[:500]})
                state["failed_sessions"].append(session)
                if state["attempts_by_session"][session] >= max_attempts:
                    state["terminal_failed_sessions"].append(session); completed.add(session)
        else:
            completed.add(session)
        key = _session_key(prefix, session)
        benchmark_s3.put_output(s3, key, json.dumps(session_output, sort_keys=True).encode(), job_id, "application/json")
        state["completed_sessions"] = sorted(completed)
        state["failed_sessions"] = sorted(set(state["failed_sessions"]))
        state["terminal_failed_sessions"] = sorted(set(state["terminal_failed_sessions"]))
        state["unresolved_sessions"] = sorted(set(state["unresolved_sessions"]))
        state["session_result_keys"][session] = key
        _save_checkpoint(s3, job_id, checkpoint_key, state); report()
    retryable = sorted(set(state["failed_sessions"]) - set(state["terminal_failed_sessions"]))
    if retryable:
        raise RetryableBenchmarkSessionsError("benchmark sessions remain retryable")
    results, unmatched, errors, inventory, usage = _load_session_outputs(
        s3, job_id, state["session_result_keys"])
    summary = build_summary(selected_rows, sessions, resolved, unresolved, objects, filtered, results, unmatched, errors, usage, time.monotonic() - started)
    summary.update({"dataset_total_historical_sessions": len(dataset_sessions),
                    "dataset_total_historical_clips": len(rows),
                    "selected_historical_sessions": len(sessions),
                    "selected_historical_clips": len(selected_rows), "preflight": preflight})
    disagreements = [row for row in results if row["changed"] or row["winner_selection_changed"]]
    disagreements += [row for row in unmatched if row.get("classification") in {"ambiguous_match", "unmatched_old", "new_unmatched"}]
    payloads = {"results_v2.jsonl": jsonl(results), "disagreements.jsonl": jsonl(disagreements),
                "unmatched.jsonl": jsonl(unmatched), "inventory.json": json.dumps(inventory, indent=2).encode(),
                "errors.jsonl": jsonl(errors), "summary.json": json.dumps(summary, indent=2).encode(),
                "summary.csv": summary_csv(summary)}
    for name, body in payloads.items(): benchmark_s3.put_output(s3, prefix + name, body, job_id, "text/csv" if name.endswith(".csv") else "application/json")
    report()
    return {"summary": summary, "output_prefix": prefix, "result_keys": [prefix + name for name in RESULT_FILES]}


def jsonl(rows): return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def take_judge_used(result: dict) -> bool:
    metadata = result.get("take_judge_v2") or {}
    return metadata.get("score") is not None or metadata.get("verdict") is not None


def build_summary(old, sessions, resolved, unresolved, objects, filtered, results, unmatched, errors, usage, elapsed):
    slot_changes = Counter(f"{x['old_slot']}->{x['new_slot']}" for x in results if x["old_slot"] != x["new_slot"])
    session_changes = Counter(x["session_id"] for x in results if x["changed"] or x["winner_selection_changed"])
    take_statuses = Counter()
    for item in usage:
        execution = item.get("take_judge_execution", {})
        if not execution.get("requested"):
            take_statuses["not_requested"] += 1
        else:
            take_statuses.update(source.get("status", "not_requested") for source in execution.get("sources", []))
    usage_available = any(item.get("provider_instrumentation", {}).get("usage", {}).get("available") for item in usage)
    cost_available = any(item.get("provider_instrumentation", {}).get("cost", {}).get("available") for item in usage)
    same_keep = sum(x["old_keep"] == x["new_keep"] for x in results); same_slot = sum(x["old_slot"] == x["new_slot"] for x in results)
    successful_result_sessions = {x["session_id"] for x in results} | {
        item.get("provider_instrumentation", {}).get("session_id") for item in usage
        if item.get("provider_instrumentation", {}).get("session_id")}
    terminal_error_sessions = {x["session_id"] for x in errors} - successful_result_sessions
    return {"total_historical_sessions": len(sessions), "resolved_sessions": sum(s in resolved for s in sessions),
            "missing_source_sessions": sum(unresolved.get(s, {}).get("classification") == "missing_source" for s in sessions),
            "ambiguous_source_sessions": sum(unresolved.get(s, {}).get("classification") == "ambiguous_source" for s in sessions),
            "filtered_s3_objects": filtered, "eligible_s3_videos": len(objects), "total_historical_clips": len(old),
            "total_matched_clips": len(results), "total_unmatched_old_clips": sum(x.get("classification") == "unmatched_old" for x in unmatched),
            "total_new_unmatched_clips": sum(x.get("classification") == "new_unmatched" for x in unmatched),
            "total_ambiguous_matches": sum("ambiguous" in x.get("classification", "") for x in unmatched),
            "same_keep_decision": same_keep, "different_keep_decision": len(results)-same_keep,
            "old_discard_new_keep": sum(not x["old_keep"] and x["new_keep"] for x in results),
            "old_keep_new_discard": sum(x["old_keep"] and not x["new_keep"] for x in results),
            "same_slot": same_slot, "different_slot": len(results)-same_slot,
            "per_slot_changes": dict(slot_changes), "per_source_session_changes": dict(session_changes),
            "semantic_v2_usage": sum(bool(x["semantic_v2"]) for x in results), "semantic_v2_fallbacks": sum(not x["semantic_v2"] for x in results),
            "take_judge_v2_usage": sum(take_judge_used(x) for x in results),
            "take_judge_v2_fallbacks": sum(not take_judge_used(x) for x in results),
            "take_judge_execution_status_counts": dict(take_statuses),
            "processing_time_seconds": round(elapsed, 3),
            "provider_usage_instrumentation": {"available": usage_available,
                "reason": None if usage_available else "Provider token usage is not instrumented"},
            "estimated_cost_instrumentation": {"available": cost_available,
                "reason": None if cost_available else "Provider cost is not instrumented"},
            "failed_sessions": len(terminal_error_sessions)}


def summary_csv(summary):
    output = io.StringIO(); writer = csv.writer(output); writer.writerow(("metric", "value"))
    for key, value in summary.items(): writer.writerow((key, json.dumps(value) if isinstance(value, (dict, list)) else value))
    return output.getvalue().encode()
