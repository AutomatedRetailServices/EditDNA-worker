"""CLEAN RAW gate -- QA metric layer over one CutSell run (D-097, QA-only).

The D-096 post-audit approval makes CLEAN RAW the first ladder rung after
RAW: before Cut.ai parity is even measured, the rendered candidate must be
free of the failure classes the audit ranked -- failed/retry material kept,
redundant realizations of one idea, meaning inversions, interior dead air,
loose entries/exits, reset debris, and any story marked incomplete. This
module reads the persisted engine result (``result.json``) and, when
available, the four-way ladder report, and prints one gate verdict with the
metrics behind it. Nothing here feeds production; absent evidence is
INCOMPLETE_EVIDENCE, never PASS.

    python benchmarks/clean_raw_gate.py --engine-json result.json \
        [--ladder-json video00-quality-ladder.json] [--out-json gate.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "cutsell.clean_raw_gate.v1"
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_INCOMPLETE = "INCOMPLETE_EVIDENCE"

# Ladder refinements that are, by definition, unclean material in the MP4.
FAILED_MATERIAL_REFINEMENTS = frozenset({
    "failed_or_process_material_retained",
    "redundant_realization_both_kept",
    "ungrouped_retry_of_kept_idea",
    "restored_by_realization_resolver",
})
# Physical-edge refinements tolerated only up to this many total seconds.
LOOSE_EDGE_REFINEMENTS = frozenset({"loose_exit_edge", "loose_entry_edge"})
MAX_LOOSE_EDGE_SECONDS = 1.0


def _walk(node: Any, key: str) -> Iterable[Any]:
    """Yield every value stored under ``key`` anywhere inside ``node``."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item, key)


def _count_rows(values: Iterable[Any]) -> int:
    total = 0
    for value in values:
        if isinstance(value, (list, tuple)):
            total += len(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total += int(value)
    return total


def compute_clean_raw_metrics(result: Mapping[str, Any], ladder: Mapping[str, Any] | None = None) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") or {}
    stage_status = result.get("stage_status") or {}
    live_qc = result.get("live_render_qc") or diagnostics.get("live_render_qc") or {}
    attempts = list(live_qc.get("attempts") or ())
    last_attempt = attempts[-1] if attempts else {}
    last_findings = list(last_attempt.get("findings") or ())
    perceptual = result.get("perceptual_watch_listen") or {}
    boundary_pass = diagnostics.get("boundary_engine_pass") or {}
    segmentation = stage_status.get("take_segmentation") if isinstance(stage_status.get("take_segmentation"), dict) else {}

    metrics: dict[str, Any] = {
        "technical_qc_status": live_qc.get("status"),
        "deliverable": live_qc.get("deliverable"),
        "delivery_status": live_qc.get("delivery_status"),
        "story_completeness": stage_status.get("story_completeness"),
        "no_usable_realization_family_count": int(stage_status.get("no_usable_realization_family_count") or 0),
        "interior_dead_air_findings_last_attempt": sum(1 for f in last_findings if f.get("kind") == "LINGERING_ACCIDENTAL_SILENCE"),
        "dead_air_reconciliation_verdicts": sorted({
            str(row.get("verdict")) for attempt in attempts for row in (attempt.get("dead_air_reconciliation") or ())
        }),
        "renderer_trailing_trim_count": len(list(last_attempt.get("renderer_trailing_trims") or ())),
        "renderer_trailing_trim_seconds": round(sum(float(r.get("trim_sec") or 0.0) for r in (last_attempt.get("renderer_trailing_trims") or ())), 3),
        "boundary_engine_pass_stage": boundary_pass.get("stage"),
        "boundary_interior_split_count": int(boundary_pass.get("interior_split_count") or 0),
        "boundary_audio_entry_trim_count": int(boundary_pass.get("audio_entry_trim_count") or 0),
        "boundary_audio_exit_trim_count": int(boundary_pass.get("audio_exit_trim_count") or 0),
        "polarity_rejoin_count": int(segmentation.get("polarity_rejoin_count") or 0),
        "protected_polarity_fragment_count": _count_rows(_walk(diagnostics, "protected_polarity_fragments")),
        # D-097.2: a semantic-label window refused by the per-edit dollar
        # ledger leaves whole retry families unlabeled -- evidence missing,
        # never a PASS (D-094.F2 starvation, runs 34028202024 / 34029861712).
        "hybrid_editorial_stage": stage_status.get("hybrid_editorial"),
        "hybrid_budget_refused_chunk_count": int(diagnostics.get("hybrid_editorial_budget_exhausted_chunk_count") or 0),
        "hybrid_requested_chunk_count": int(diagnostics.get("hybrid_editorial_requested_chunk_count") or 0),
        "perceptual_status": perceptual.get("status"),
        "perceptual_artifact_kind": perceptual.get("artifact_kind"),
        "perceptual_gate_mode": perceptual.get("gate_mode"),
        "perceptual_capability_status_counts": perceptual.get("capability_status_counts") or {},
        "perceptual_routing": perceptual.get("routing") or {},
    }

    if ladder is not None:
        regions = list(ladder.get("physical_regions") or ladder.get("regions") or ())
        level1 = [r for r in regions if r.get("level") == "LEVEL_1"]
        by_refinement: dict[str, dict[str, float]] = {}
        for region in level1:
            key = str(region.get("refinement") or region.get("kind") or "unclassified")
            row = by_refinement.setdefault(key, {"count": 0, "seconds": 0.0})
            row["count"] += 1
            row["seconds"] = round(row["seconds"] + float(region.get("duration_sec") or 0.0), 3)
        metrics["ladder_source"] = "physical_regions" if ladder.get("physical_regions") else "regions"
        metrics["level1_region_count"] = len(level1)
        metrics["level1_seconds"] = round(sum(float(r.get("duration_sec") or 0.0) for r in level1), 3)
        metrics["level1_by_refinement"] = by_refinement
        metrics["failed_material_seconds"] = round(sum(v["seconds"] for k, v in by_refinement.items() if k in FAILED_MATERIAL_REFINEMENTS), 3)
        metrics["failed_material_regions"] = sum(int(v["count"]) for k, v in by_refinement.items() if k in FAILED_MATERIAL_REFINEMENTS)
        metrics["missing_delivery_seconds"] = round(sum(float(r.get("duration_sec") or 0.0) for r in level1 if r.get("kind") == "missing_delivery"), 3)
        metrics["loose_edge_seconds"] = round(sum(v["seconds"] for k, v in by_refinement.items() if k in LOOSE_EDGE_REFINEMENTS), 3)
        summary = ladder.get("summary") or {}
        parity = summary.get("cutai_parity") or {}
        metrics["cutai_parity"] = parity
    return metrics


def evaluate_clean_raw_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    blocking: list[str] = []
    missing: list[str] = []

    if metrics.get("technical_qc_status") is None:
        missing.append("technical_qc_status")
    elif metrics.get("technical_qc_status") != "PASS":
        blocking.append(f"technical_qc:{metrics.get('technical_qc_status')}")
    if metrics.get("story_completeness") not in (None, "complete"):
        blocking.append(f"story:{metrics.get('story_completeness')}")
    if metrics.get("interior_dead_air_findings_last_attempt"):
        blocking.append(f"interior_dead_air:{metrics['interior_dead_air_findings_last_attempt']}")
    if metrics.get("perceptual_status") is None:
        missing.append("perceptual_watch_listen")
    elif metrics.get("perceptual_status") == "FAIL":
        blocking.append("perceptual:FAIL")
    if int(metrics.get("hybrid_budget_refused_chunk_count") or 0) > 0:
        missing.append(
            f"semantic_labels:{metrics['hybrid_budget_refused_chunk_count']}/{metrics.get('hybrid_requested_chunk_count')} "
            "windows refused by the per-edit dollar ledger"
        )
    if "level1_region_count" not in metrics:
        missing.append("ladder")
    else:
        if metrics.get("failed_material_regions"):
            blocking.append(f"failed_material_regions:{metrics['failed_material_regions']} ({metrics.get('failed_material_seconds')} s)")
        if float(metrics.get("loose_edge_seconds") or 0.0) > MAX_LOOSE_EDGE_SECONDS:
            blocking.append(f"loose_edges:{metrics.get('loose_edge_seconds')} s > {MAX_LOOSE_EDGE_SECONDS} s")
        if float(metrics.get("missing_delivery_seconds") or 0.0) > 0.0:
            blocking.append(f"missing_delivery:{metrics.get('missing_delivery_seconds')} s")

    if blocking:
        status = GATE_FAIL
    elif missing:
        status = GATE_INCOMPLETE
    else:
        status = GATE_PASS
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "blocking": blocking,
        "missing_evidence": missing,
        "note": (
            "CLEAN RAW is the first ladder rung after RAW; PASS requires technical QC PASS, a complete story, "
            "no interior dead air, no perceptual FAIL, and (from the ladder) no failed/redundant/restored material, "
            "no missing delivery and <= 1.0 s loose edges. Perceptual UNCERTAIN is not blocking in advisory_v1 but "
            "keeps HUMAN WATCH+LISTEN required."
        ),
    }


def build_gate_report(result: Mapping[str, Any], ladder: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metrics = compute_clean_raw_metrics(result, ladder)
    return {"metrics": metrics, "gate": evaluate_clean_raw_gate(metrics)}


def format_report(report: Mapping[str, Any]) -> str:
    metrics, gate = report["metrics"], report["gate"]
    lines = [
        f"CLEAN RAW GATE: {gate['status']}",
        "  blocking: " + ("; ".join(gate["blocking"]) or "none"),
        f"  semantic labels: stage {metrics.get('hybrid_editorial_stage')}; budget-refused windows "
        f"{metrics.get('hybrid_budget_refused_chunk_count')}/{metrics.get('hybrid_requested_chunk_count')}",
        "  missing evidence: " + (", ".join(gate["missing_evidence"]) or "none"),
        f"  technical QC: {metrics.get('technical_qc_status')} deliverable={metrics.get('deliverable')} delivery_status={metrics.get('delivery_status')}",
        f"  story: {metrics.get('story_completeness')} (no-usable families: {metrics.get('no_usable_realization_family_count')})",
        f"  dead air (last attempt): {metrics.get('interior_dead_air_findings_last_attempt')} reconciliation={metrics.get('dead_air_reconciliation_verdicts')}",
        f"  boundary pass: stage={metrics.get('boundary_engine_pass_stage')} interior_splits={metrics.get('boundary_interior_split_count')} "
        f"audio_entry={metrics.get('boundary_audio_entry_trim_count')} audio_exit={metrics.get('boundary_audio_exit_trim_count')} "
        f"renderer_trailing_trims={metrics.get('renderer_trailing_trim_count')} ({metrics.get('renderer_trailing_trim_seconds')} s)",
        f"  polarity: rejoins={metrics.get('polarity_rejoin_count')} protected_fragments={metrics.get('protected_polarity_fragment_count')}",
        f"  perceptual: {metrics.get('perceptual_status')} ({metrics.get('perceptual_gate_mode')}) capabilities={metrics.get('perceptual_capability_status_counts')} routing={metrics.get('perceptual_routing')}",
    ]
    if "level1_region_count" in metrics:
        lines.append(
            f"  ladder: LEVEL_1 {metrics['level1_region_count']} regions / {metrics['level1_seconds']} s; "
            f"failed material {metrics['failed_material_regions']} regions / {metrics['failed_material_seconds']} s; "
            f"missing delivery {metrics['missing_delivery_seconds']} s; loose edges {metrics['loose_edge_seconds']} s"
        )
        lines.append("  LEVEL_1 by refinement: " + (", ".join(f"{k}: {v['count']} / {v['seconds']} s" for k, v in sorted(metrics["level1_by_refinement"].items(), key=lambda kv: -kv[1]["seconds"])) or "none"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLEAN RAW gate over one CutSell run (QA-only)")
    parser.add_argument("--engine-json", required=True)
    parser.add_argument("--ladder-json", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = json.loads(Path(args.engine_json).read_text(encoding="utf-8"))
    ladder = json.loads(Path(args.ladder_json).read_text(encoding="utf-8")) if args.ladder_json and Path(args.ladder_json).exists() else None
    report = build_gate_report(result, ladder)
    print(format_report(report))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
