"""Real-footage golden benchmark for CutSell's complete Flow B brain.

Automatic scores are structural diagnostics only. Human review of rendered previews
remains a separate gate for edit quality. FULL mode processes the complete source;
BOUNDED mode remains available for cheap/smoke validation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
import re
from statistics import median
from typing import Any, Iterable

from .validation import _is_real_video_key, list_validation_videos, run_single_validation

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")
VALID_MODES = {"full", "bounded"}


def _normalized_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value).lower()))


def _provider_status(report: dict[str, Any], stage: str) -> str | None:
    value = (report.get("stage_status") or {}).get(stage)
    if isinstance(value, dict):
        return str(value.get("status") or value.get("provider_status") or "") or None
    return str(value) if value is not None else None


def _diagnostic_status(diagnostics: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = diagnostics.get(key)
        if isinstance(value, dict):
            text = str(value.get("status") or value.get("provider_status") or "")
            if text:
                return text
        elif value is not None:
            text = str(value)
            if text:
                return text
    return None


def evaluate_validation_report(report: dict[str, Any]) -> dict[str, Any]:
    selected = list(report.get("selected") or ())
    alternates = list(report.get("alternates") or ())
    discarded = list(report.get("discarded") or ())
    warnings: list[str] = []
    hard_failures: list[str] = []

    if not selected:
        hard_failures.append("empty_selected_timeline")
    all_ids = [str(item.get("clip_id") or "") for item in selected + alternates + discarded]
    if len(all_ids) != len(set(all_ids)):
        hard_failures.append("clip_id_collision_across_draft_buckets")

    invalid_duration = tiny_fragments = adjacent_duplicates = 0
    previous_text = None
    for item in selected:
        start, end = float(item.get("start") or 0), float(item.get("end") or 0)
        text = str(item.get("text") or "").strip()
        duration = end - start
        if duration <= 0:
            invalid_duration += 1
        if duration < 0.35 and len(_WORD_RE.findall(text)) <= 1:
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

    groups = Counter(str(item.get("take_group_id")) for item in selected + alternates if item.get("take_group_id"))
    retry_group_count = sum(1 for count in groups.values() if count >= 2)
    orphan_alternates = sum(
        1 for item in alternates
        if not item.get("take_group_id") or groups[str(item.get("take_group_id"))] < 2
    )
    if orphan_alternates:
        hard_failures.append(f"orphan_alternates:{orphan_alternates}")

    diagnostics = report.get("diagnostics") or {}
    whole_video_status = _provider_status(report, "whole_video_context")
    semantic_status = _provider_status(report, "semantic")
    visual_status = _provider_status(report, "visual")
    take_judge_status = _provider_status(report, "take_judge")
    clean_cut_status = _provider_status(report, "clean_cut")
    grouping_status = _diagnostic_status(
        diagnostics,
        "take_grouping_provider_status",
        "take_grouping_status",
    ) or _provider_status(report, "take_grouping")
    composer_status = _diagnostic_status(
        diagnostics,
        "composer_provider_status",
        "composer_status",
    ) or _provider_status(report, "composer")
    draft_review_status = _diagnostic_status(diagnostics, "draft_review_status") or _provider_status(report, "draft_review")
    draft_review_postable = bool(diagnostics.get("draft_review_postable", False))
    draft_review_issues = list(diagnostics.get("draft_review_issues") or ())
    clean_cut_judge_status = (diagnostics.get("clean_cut_judge_status") or {}).get("status")
    clean_cut_judge_deleted = int(diagnostics.get("clean_cut_judge_deleted_count") or 0)
    clean_cut_judge_mixed = int(diagnostics.get("clean_cut_judge_mixed_count") or 0)

    degraded = {"degraded", "failed", "unavailable", "provider_error", "provider_unavailable", "provider_error_fallback"}
    score = 100 - 35 * len(hard_failures) - 5 * len(warnings)
    for status in (whole_video_status, semantic_status, visual_status, grouping_status, take_judge_status, composer_status, draft_review_status):
        if status in degraded:
            score -= 8
    score = max(0, min(100, score))

    analyzed = float(report.get("analyzed_duration_sec") or 0.0)
    selected_duration = float(report.get("selected_duration_sec") or 0.0)
    compression_ratio = round(selected_duration / analyzed, 4) if analyzed > 0 else None
    return {
        "source_key": report.get("source_key"),
        "elapsed_sec": report.get("elapsed_sec"),
        "source_duration_sec": report.get("source_duration_sec"),
        "analyzed_duration_sec": report.get("analyzed_duration_sec"),
        "selected_duration_sec": report.get("selected_duration_sec"),
        "selected_to_input_ratio": compression_ratio,
        "strategy": report.get("strategy"),
        "edit_mode": _provider_status(report, "edit_mode"),
        "selected_count": len(selected),
        "alternate_count": len(alternates),
        "discarded_count": len(discarded),
        "retry_group_count": retry_group_count,
        "tiny_fragment_count": tiny_fragments,
        "adjacent_duplicate_count": adjacent_duplicates,
        "draft_review_postable": draft_review_postable,
        "draft_review_issues": draft_review_issues,
        "provider_status": {
            "whole_video": whole_video_status,
            "semantic": semantic_status,
            "visual": visual_status,
            "take_grouping": grouping_status,
            "take_judge": take_judge_status,
            "composer": composer_status,
            "draft_review": draft_review_status,
            "clean_cut": clean_cut_status,
            "clean_cut_judge": str(clean_cut_judge_status or "not_requested"),
        },
        "clean_cut_judge_deleted_count": clean_cut_judge_deleted,
        "clean_cut_judge_mixed_count": clean_cut_judge_mixed,
        "take_judge_status_counts": diagnostics.get("take_judge_status_counts") or {},
        "warnings": warnings,
        "hard_failures": hard_failures,
        "structural_score": score,
        "structural_pass": not hard_failures,
        "stage_status": report.get("stage_status") or {},
    }


def _preview_name(index: int, source_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", PurePosixPath(source_key).stem).strip("-._")[:80] or "video"
    return f"{index:02d}-{safe}.mp4"


def _source_records(*, video_limit: int, prefix: str | None, source_keys: Iterable[str] | None) -> tuple[dict[str, Any], ...]:
    explicit = tuple(str(key).strip() for key in (source_keys or ()) if str(key).strip())
    if explicit:
        if len(explicit) > 20:
            raise ValueError("source_keys supports at most 20 videos")
        for key in explicit:
            if not _is_real_video_key(key):
                raise ValueError(f"invalid benchmark source key: {key}")
        return tuple({"key": key, "size": None} for key in explicit)
    return list_validation_videos(prefix=prefix, limit=int(video_limit))


def run_golden_benchmark(
    *,
    video_limit: int = 8,
    mode: str = "full",
    window_sec: float = 60.0,
    prefix: str | None = None,
    preview_dir: str | None = None,
    source_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    if not 1 <= int(video_limit) <= 20:
        raise ValueError("video_limit must be between 1 and 20")
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in VALID_MODES:
        raise ValueError("mode must be full or bounded")
    if normalized_mode == "bounded" and not 10 <= float(window_sec) <= 180:
        raise ValueError("window_sec must be between 10 and 180 in bounded mode")

    sources = _source_records(video_limit=video_limit, prefix=prefix, source_keys=source_keys)
    if not sources:
        raise RuntimeError("golden benchmark found no real videos")

    preview_root = Path(preview_dir) if preview_dir else None
    if preview_root:
        preview_root.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for index, source in enumerate(sources):
        key = str(source["key"])
        try:
            kwargs: dict[str, Any] = {
                "project_id": f"cutsell-golden-{index:03d}",
                "preview_output": str(preview_root / _preview_name(index, key)) if preview_root else None,
            }
            if normalized_mode == "bounded":
                kwargs.update(source_start_sec=0.0, source_end_sec=float(window_sec))
            report = run_single_validation(key, **kwargs)
            results.append({"validation": report, "evaluation": evaluate_validation_report(report)})
        except Exception as exc:
            failures.append({
                "source_key": key,
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:500],
            })

    evaluations = [item["evaluation"] for item in results]
    scores = [int(item["structural_score"]) for item in evaluations]
    provider_stages = (
        "whole_video", "semantic", "visual", "take_grouping",
        "take_judge", "composer", "draft_review", "clean_cut", "clean_cut_judge",
    )
    provider_counts = {
        stage: dict(Counter(str(item["provider_status"].get(stage) or "unknown") for item in evaluations))
        for stage in provider_stages
    }
    total_input_sec = sum(float(item.get("analyzed_duration_sec") or 0.0) for item in evaluations)
    total_output_sec = sum(float(item.get("selected_duration_sec") or 0.0) for item in evaluations)
    return {
        "schema_version": "cutsell.golden.v2",
        "mode": normalized_mode,
        "video_limit": int(video_limit),
        "window_sec": float(window_sec) if normalized_mode == "bounded" else None,
        "source_count": len(sources),
        "completed_count": len(results),
        "execution_failure_count": len(failures),
        "structural_pass_count": sum(1 for item in evaluations if item["structural_pass"]),
        "structural_failure_count": sum(1 for item in evaluations if not item["structural_pass"]),
        "draft_review_postable_count": sum(1 for item in evaluations if item["draft_review_postable"]),
        "draft_review_needs_attention_count": sum(1 for item in evaluations if not item["draft_review_postable"]),
        "median_structural_score": median(scores) if scores else None,
        "total_input_duration_sec": round(total_input_sec, 3),
        "total_selected_duration_sec": round(total_output_sec, 3),
        "selected_to_input_ratio": round(total_output_sec / total_input_sec, 4) if total_input_sec > 0 else None,
        "total_selected": sum(int(item["selected_count"]) for item in evaluations),
        "total_alternates": sum(int(item["alternate_count"]) for item in evaluations),
        "total_discarded": sum(int(item["discarded_count"]) for item in evaluations),
        "total_retry_groups": sum(int(item["retry_group_count"]) for item in evaluations),
        "total_tiny_fragments": sum(int(item["tiny_fragment_count"]) for item in evaluations),
        "total_adjacent_duplicates": sum(int(item["adjacent_duplicate_count"]) for item in evaluations),
        "total_clean_cut_judge_deleted": sum(int(item["clean_cut_judge_deleted_count"]) for item in evaluations),
        "total_clean_cut_judge_mixed": sum(int(item["clean_cut_judge_mixed_count"]) for item in evaluations),
        "edit_mode_counts": dict(Counter(str(item.get("edit_mode") or "unknown") for item in evaluations)),
        "provider_status_counts": provider_counts,
        "failures": failures,
        "results": results,
    }
