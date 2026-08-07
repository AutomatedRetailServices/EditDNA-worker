"""Bounded real-footage benchmark for CutSell's clean Flow B brain.

This benchmark is deliberately diagnostic, not a claim of human edit quality. It
runs real S3 blooper footage through the production clean worker, records provider
status and draft structure, and flags mechanical anomalies that should never ship.
Human review of rendered previews remains a separate gate.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
import re
from statistics import median
from typing import Any

from .validation import list_validation_videos, run_single_validation

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")


def _normalized_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value).lower()))


def _provider_status(report: dict[str, Any], stage: str) -> str | None:
    status = report.get("stage_status") or {}
    value = status.get(stage)
    if isinstance(value, dict):
        return str(value.get("status") or value.get("provider_status") or "") or None
    return str(value) if value is not None else None


def evaluate_validation_report(report: dict[str, Any]) -> dict[str, Any]:
    selected = list(report.get("selected") or ())
    alternates = list(report.get("alternates") or ())
    discarded = list(report.get("discarded") or ())
    warnings: list[str] = []
    hard_failures: list[str] = []

    if not selected:
        hard_failures.append("empty_selected_timeline")

    selected_ids = [str(item.get("clip_id") or "") for item in selected]
    alternate_ids = [str(item.get("clip_id") or "") for item in alternates]
    discarded_ids = [str(item.get("clip_id") or "") for item in discarded]
    all_ids = selected_ids + alternate_ids + discarded_ids
    if len(all_ids) != len(set(all_ids)):
        hard_failures.append("clip_id_collision_across_draft_buckets")

    invalid_duration = 0
    tiny_fragments = 0
    adjacent_duplicates = 0
    previous_text = None
    for item in selected:
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or 0.0)
        text = str(item.get("text") or "").strip()
        duration = end - start
        if duration <= 0:
            invalid_duration += 1
        token_count = len(_WORD_RE.findall(text))
        if duration < 0.35 and token_count <= 1:
            tiny_fragments += 1
        normalized = _normalized_text(text)
        if normalized and previous_text == normalized:
            adjacent_duplicates += 1
        if normalized:
            previous_text = normalized

    if invalid_duration:
        hard_failures.append(f"invalid_selected_duration:{invalid_duration}")
    if tiny_fragments:
        warnings.append(f"tiny_selected_fragments:{tiny_fragments}")
    if adjacent_duplicates:
        warnings.append(f"adjacent_duplicate_selected_text:{adjacent_duplicates}")

    groups = Counter(
        str(item.get("take_group_id"))
        for item in selected + alternates
        if item.get("take_group_id")
    )
    retry_group_count = sum(1 for count in groups.values() if count >= 2)
    orphan_alternates = sum(
        1 for item in alternates
        if not item.get("take_group_id") or groups[str(item.get("take_group_id"))] < 2
    )
    if orphan_alternates:
        hard_failures.append(f"orphan_alternates:{orphan_alternates}")

    stage_status = report.get("stage_status") or {}
    diagnostics = report.get("diagnostics") or {}
    semantic_status = _provider_status(report, "semantic")
    visual_status = _provider_status(report, "visual")
    take_judge_status = _provider_status(report, "take_judge")

    # Structural score only: never present this as human edit quality.
    score = 100
    score -= 35 * len(hard_failures)
    score -= 5 * len(warnings)
    if semantic_status in {"degraded", "failed", "unavailable"}:
        score -= 8
    if visual_status in {"degraded", "failed", "unavailable"}:
        score -= 8
    score = max(0, min(100, score))

    return {
        "source_key": report.get("source_key"),
        "elapsed_sec": report.get("elapsed_sec"),
        "strategy": report.get("strategy"),
        "selected_count": len(selected),
        "alternate_count": len(alternates),
        "discarded_count": len(discarded),
        "retry_group_count": retry_group_count,
        "tiny_fragment_count": tiny_fragments,
        "adjacent_duplicate_count": adjacent_duplicates,
        "provider_status": {
            "semantic": semantic_status,
            "visual": visual_status,
            "take_judge": take_judge_status,
        },
        "take_judge_status_counts": diagnostics.get("take_judge_status_counts") or {},
        "warnings": warnings,
        "hard_failures": hard_failures,
        "structural_score": score,
        "structural_pass": not hard_failures,
        "stage_status": stage_status,
    }


def _preview_name(index: int, source_key: str) -> str:
    stem = PurePosixPath(source_key).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")[:80] or "video"
    return f"{index:02d}-{safe}.mp4"


def run_golden_benchmark(
    *,
    video_limit: int = 8,
    window_sec: float = 60.0,
    prefix: str | None = None,
    preview_dir: str | None = None,
) -> dict[str, Any]:
    if not 1 <= int(video_limit) <= 20:
        raise ValueError("video_limit must be between 1 and 20")
    if not 10 <= float(window_sec) <= 180:
        raise ValueError("window_sec must be between 10 and 180")

    sources = list_validation_videos(prefix=prefix, limit=int(video_limit))
    if not sources:
        raise RuntimeError("golden benchmark found no real videos")

    preview_root = None
    if preview_dir:
        preview_root = Path(preview_dir)
        preview_root.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for index, source in enumerate(sources):
        key = str(source["key"])
        try:
            preview_output = None
            if preview_root is not None:
                preview_output = str(preview_root / _preview_name(index, key))
            report = run_single_validation(
                key,
                project_id=f"cutsell-golden-{index:03d}",
                source_start_sec=0.0,
                source_end_sec=float(window_sec),
                preview_output=preview_output,
            )
            results.append({
                "validation": report,
                "evaluation": evaluate_validation_report(report),
            })
        except Exception as exc:
            failures.append({
                "source_key": key,
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:500],
            })

    evaluations = [item["evaluation"] for item in results]
    structural_scores = [int(item["structural_score"]) for item in evaluations]
    provider_counts = {
        stage: dict(Counter(
            str(item["provider_status"].get(stage) or "unknown")
            for item in evaluations
        ))
        for stage in ("semantic", "visual", "take_judge")
    }
    return {
        "schema_version": "cutsell.golden.v1",
        "video_limit": int(video_limit),
        "window_sec": float(window_sec),
        "source_count": len(sources),
        "completed_count": len(results),
        "execution_failure_count": len(failures),
        "structural_pass_count": sum(1 for item in evaluations if item["structural_pass"]),
        "structural_failure_count": sum(1 for item in evaluations if not item["structural_pass"]),
        "median_structural_score": median(structural_scores) if structural_scores else None,
        "total_selected": sum(int(item["selected_count"]) for item in evaluations),
        "total_alternates": sum(int(item["alternate_count"]) for item in evaluations),
        "total_discarded": sum(int(item["discarded_count"]) for item in evaluations),
        "total_retry_groups": sum(int(item["retry_group_count"]) for item in evaluations),
        "total_tiny_fragments": sum(int(item["tiny_fragment_count"]) for item in evaluations),
        "total_adjacent_duplicates": sum(int(item["adjacent_duplicate_count"]) for item in evaluations),
        "provider_status_counts": provider_counts,
        "failures": failures,
        "results": results,
    }
