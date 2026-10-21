"""Robust RAW -> Human Gold alignment for CutSell benchmark QA.

QA/oracle tooling only. It never becomes runtime editorial authority.
V2.2 uses high-confidence global candidates plus a sparse monotonic path so
anchors that straddle a human edit can be skipped instead of poisoning the map.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .human_gold_decision_map import (
    AlignmentAnchor,
    GoldSourceChunk,
    _audio_features,
    _ffprobe_duration,
    _normalized_window_correlation,
    build_decision_map,
    coalesce_alignment_anchors,
)


def _top_indices(values, k: int):
    import numpy as np
    if len(values) <= k:
        return np.argsort(values)[::-1]
    idx = np.argpartition(values, -k)[-k:]
    return idx[np.argsort(values[idx])[::-1]]


@dataclass(frozen=True)
class _Candidate:
    anchor_index: int
    gold_start_frame: int
    raw_start_frame: int
    correlation: float


def _sparse_monotonic_path(
    rows: Sequence[Sequence[tuple[int, float]]],
    gold_starts: Sequence[int],
    *,
    hop_sec: float,
    minimum_correlation: float,
    maximum_skipped_gold_sec: float = 3.0,
) -> tuple[_Candidate, ...]:
    """Return a high-confidence monotonic candidate chain.

    Unlike V2.1 this path can skip ambiguous Gold anchors at edit boundaries.
    It rewards confidence and continuity, penalizes skipped Gold time, and
    requires the source cursor to move forward whenever Gold moves forward.
    """
    nodes: list[_Candidate] = []
    by_anchor: list[list[int]] = [[] for _ in rows]
    for i, row in enumerate(rows):
        for raw_start, corr in row:
            if corr < minimum_correlation:
                continue
            idx = len(nodes)
            nodes.append(_Candidate(i, int(gold_starts[i]), int(raw_start), float(corr)))
            by_anchor[i].append(idx)

    if not nodes:
        raise RuntimeError("No trusted Human Gold alignment candidates")

    score = [-1e18] * len(nodes)
    parent: list[int | None] = [None] * len(nodes)
    path_len = [1] * len(nodes)

    for ni, node in enumerate(nodes):
        gold_sec = node.gold_start_frame * hop_sec
        # A chain may begin only near the start of Gold.
        if gold_sec <= 2.5:
            score[ni] = node.correlation - 0.0002 * (node.raw_start_frame * hop_sec)

        # Search predecessor anchors within a bounded Gold gap. This allows the
        # 1-3 ambiguous windows around a cut to disappear without allowing the
        # path to skip whole spoken sections.
        min_anchor = max(0, node.anchor_index - int(maximum_skipped_gold_sec / 0.25) - 8)
        for pi, prev in enumerate(nodes[:ni]):
            if prev.anchor_index >= node.anchor_index or prev.anchor_index < min_anchor:
                continue
            if score[pi] <= -1e17:
                continue
            gold_delta = (node.gold_start_frame - prev.gold_start_frame) * hop_sec
            if gold_delta <= 0:
                continue
            raw_delta = (node.raw_start_frame - prev.raw_start_frame) * hop_sec
            # Windows overlap, so source can advance slightly less than Gold.
            if raw_delta < max(0.02, gold_delta - 0.55):
                continue
            skipped_gold = max(0.0, gold_delta - 0.50)
            extra_raw_skip = max(0.0, raw_delta - gold_delta)
            transition_penalty = 0.05 * skipped_gold + 0.0006 * extra_raw_skip
            candidate_score = score[pi] + node.correlation - transition_penalty
            if candidate_score > score[ni]:
                score[ni] = candidate_score
                parent[ni] = pi
                path_len[ni] = path_len[pi] + 1

    end_choices = [
        i for i, node in enumerate(nodes)
        if node.gold_start_frame * hop_sec >= (gold_starts[-1] * hop_sec - 2.5)
        and score[i] > -1e17
    ]
    if not end_choices:
        raise RuntimeError("Human Gold alignment has no end-to-end sparse monotonic path")

    end = max(end_choices, key=lambda i: (score[i], path_len[i], nodes[i].correlation))
    path: list[_Candidate] = []
    cur: int | None = end
    while cur is not None:
        path.append(nodes[cur])
        cur = parent[cur]
    path.reverse()

    if path[0].gold_start_frame * hop_sec > 2.5:
        raise RuntimeError("Sparse Human Gold path does not begin near edit start")
    if path[-1].gold_start_frame * hop_sec < gold_starts[-1] * hop_sec - 2.5:
        raise RuntimeError("Sparse Human Gold path does not reach edit end")
    return tuple(path)


def align_gold_audio_to_raw_v2(
    raw_path: str | Path,
    gold_path: str | Path,
    *,
    anchor_window_sec: float = 1.25,
    anchor_stride_sec: float = 0.50,
    hop_sec: float = 0.02,
    candidates_per_anchor: int = 16,
    minimum_correlation: float = 0.62,
) -> tuple[tuple[AlignmentAnchor, ...], tuple[GoldSourceChunk, ...]]:
    import numpy as np

    raw_features = _audio_features(raw_path, hop_sec=hop_sec)
    gold_features = _audio_features(gold_path, hop_sec=hop_sec)
    raw_duration = _ffprobe_duration(raw_path)
    gold_duration = _ffprobe_duration(gold_path)
    window_frames = max(30, int(round(anchor_window_sec / hop_sec)))
    stride_frames = max(1, int(round(anchor_stride_sec / hop_sec)))

    gold_starts = list(range(0, max(1, gold_features.shape[0] - window_frames + 1), stride_frames))
    last_start = max(0, gold_features.shape[0] - window_frames)
    if not gold_starts or gold_starts[-1] != last_start:
        gold_starts.append(last_start)

    rows: list[list[tuple[int, float]]] = []
    for gold_start in gold_starts:
        template = gold_features[gold_start:gold_start + window_frames]
        corr = _normalized_window_correlation(raw_features, template)
        top: list[tuple[int, float]] = []
        for idx in _top_indices(corr, candidates_per_anchor * 10):
            raw_start = int(idx)
            value = float(corr[raw_start])
            if any(abs(raw_start - kept) * hop_sec < 0.35 for kept, _ in top):
                continue
            top.append((raw_start, value))
            if len(top) >= candidates_per_anchor:
                break
        rows.append(top)

    path = _sparse_monotonic_path(
        rows,
        gold_starts,
        hop_sec=hop_sec,
        minimum_correlation=minimum_correlation,
        maximum_skipped_gold_sec=3.0,
    )

    anchors = tuple(
        AlignmentAnchor(
            gold_time=(node.gold_start_frame + window_frames / 2.0) * hop_sec,
            raw_time=(node.raw_start_frame + window_frames / 2.0) * hop_sec,
            correlation=node.correlation,
        )
        for node in path
    )
    if len(anchors) < max(20, int(gold_duration / 3.5)):
        raise RuntimeError(f"Sparse Human Gold path too small: {len(anchors)} anchors")
    if anchors[0].gold_time > 2.75 or anchors[-1].gold_time < gold_duration - 2.75:
        raise RuntimeError(
            f"Sparse Human Gold path lacks end coverage: first={anchors[0].gold_time:.3f}, "
            f"last={anchors[-1].gold_time:.3f}, gold={gold_duration:.3f}"
        )

    groups = coalesce_alignment_anchors(
        anchors,
        offset_tolerance_sec=0.20,
        maximum_gold_gap_sec=3.25,
        minimum_correlation=minimum_correlation,
    )
    if len(groups) < 10:
        raise RuntimeError(f"Too few Human Gold source plateaus: {len(groups)}")

    boundaries = [0.0]
    for left, right in zip(groups, groups[1:]):
        boundaries.append((float(left[-1].gold_time) + float(right[0].gold_time)) / 2.0)
    boundaries.append(gold_duration)

    chunks: list[GoldSourceChunk] = []
    previous_raw_start = -1.0
    for index, group in enumerate(groups, 1):
        offsets = sorted(a.offset for a in group)
        correlations = sorted(a.correlation for a in group)
        offset = float(offsets[len(offsets) // 2])
        confidence = float(correlations[len(correlations) // 2])
        gold_start = float(boundaries[index - 1])
        gold_end = float(boundaries[index])
        raw_start = gold_start + offset
        raw_end = gold_end + offset
        if raw_start < -0.40 or raw_end > raw_duration + 0.40 or raw_end <= raw_start:
            raise RuntimeError(
                f"Invalid Gold->RAW chunk {index}: gold={gold_start:.3f}-{gold_end:.3f}, "
                f"raw={raw_start:.3f}-{raw_end:.3f}, raw_duration={raw_duration:.3f}"
            )
        if raw_start + 0.10 < previous_raw_start:
            raise RuntimeError(f"Non-monotonic Human Gold source chunk {index}")
        previous_raw_start = raw_start
        chunks.append(GoldSourceChunk(
            index=index,
            gold_start=gold_start,
            gold_end=gold_end,
            raw_start=max(0.0, raw_start),
            raw_end=min(raw_duration, raw_end),
            source_offset_sec=offset,
            alignment_confidence=confidence,
        ))

    if abs(sum(c.duration for c in chunks) - gold_duration) > 0.10:
        raise RuntimeError("Gold chunk coverage mismatch")
    if max(c.raw_end for c in chunks) > raw_duration + 1e-6:
        raise RuntimeError("Human Gold source projection exceeds RAW")

    return anchors, tuple(chunks)


def _write_csv(report: dict, path: str | Path) -> None:
    rows = []
    for row in report.get("human_kept") or ():
        rows.append({
            "human_action": "KEEP", "index": row.get("gold_chunk_index"),
            "raw_start": row.get("raw_start"), "raw_end": row.get("raw_end"),
            "duration_sec": row.get("duration_sec"),
            "engine_selected_coverage": row.get("engine_selected_coverage"),
            "pass": row.get("selection_pass"), "rule_candidate": row.get("rule_candidate"),
        })
    for row in report.get("human_deleted") or ():
        rows.append({
            "human_action": "DELETE", "index": row.get("delete_region_index"),
            "raw_start": row.get("raw_start"), "raw_end": row.get("raw_end"),
            "duration_sec": row.get("duration_sec"),
            "engine_selected_coverage": row.get("engine_selected_overlap_sec"),
            "pass": not bool(row.get("engine_false_keep")), "rule_candidate": row.get("rule_candidate"),
        })
    fields = ["human_action", "index", "raw_start", "raw_end", "duration_sec", "engine_selected_coverage", "pass", "rule_candidate"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True); parser.add_argument("--gold", required=True)
    parser.add_argument("--engine-json", required=True); parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True); args = parser.parse_args(argv)
    raw_duration = _ffprobe_duration(args.raw); gold_duration = _ffprobe_duration(args.gold)
    anchors, chunks = align_gold_audio_to_raw_v2(args.raw, args.gold)
    engine_result = json.loads(Path(args.engine_json).read_text(encoding="utf-8"))
    report = build_decision_map(raw_duration_sec=raw_duration, gold_duration_sec=gold_duration,
                                gold_chunks=chunks, engine_result=engine_result,
                                alignment_anchors=anchors)
    report["schema_version"] = "cutsell.human_gold_decision_map.v2.2"
    report["alignment"]["trusted_anchor_count"] = len(anchors)
    report["alignment"]["first_anchor_gold_sec"] = round(anchors[0].gold_time, 3)
    report["alignment"]["last_anchor_gold_sec"] = round(anchors[-1].gold_time, 3)
    report["alignment"]["maximum_projected_raw_sec"] = round(max(c.raw_end for c in chunks), 3)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(report, args.out_csv)
    print(json.dumps({"schema_version": report["schema_version"],
                      "human_gold_chunk_count": report["human_gold_chunk_count"],
                      "alignment": report["alignment"],
                      "selection_parity": report["selection_parity"],
                      "boundary_parity": report["boundary_parity"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
