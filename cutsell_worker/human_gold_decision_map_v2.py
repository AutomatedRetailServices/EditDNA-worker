"""Robust RAW -> Human Gold alignment for CutSell benchmark QA.

This is QA/oracle tooling only. It never becomes runtime editorial authority.
V2 replaces the greedy alignment path with global candidate matching followed by
monotonic dynamic programming. It refuses to emit a decision map when the Gold
is not aligned end-to-end inside the RAW duration.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def align_gold_audio_to_raw_v2(
    raw_path: str | Path,
    gold_path: str | Path,
    *,
    anchor_window_sec: float = 1.5,
    anchor_stride_sec: float = 0.50,
    hop_sec: float = 0.02,
    candidates_per_anchor: int = 12,
    minimum_correlation: float = 0.48,
) -> tuple[tuple[AlignmentAnchor, ...], tuple[GoldSourceChunk, ...]]:
    """Align the entire edited Gold monotonically back to its RAW source.

    Each Gold anchor is matched globally against RAW. Dynamic programming chooses
    one complete forward-only path. Human edits may legitimately jump a large
    distance in RAW between adjacent Gold phrases, so V2.1 does not impose a hard
    maximum jump. It still penalizes unnecessarily large jumps and validates every
    resulting source chunk against the real RAW duration before emitting parity.
    """
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

    candidate_rows: list[list[tuple[int, float]]] = []
    for gold_start in gold_starts:
        template = gold_features[gold_start:gold_start + window_frames]
        corr = _normalized_window_correlation(raw_features, template)
        top: list[tuple[int, float]] = []
        # Search more raw peaks than we retain so non-max suppression does not
        # accidentally leave only repeated local maxima from the same phrase.
        for idx in _top_indices(corr, candidates_per_anchor * 8):
            score = float(corr[int(idx)])
            raw_start = int(idx)
            if any(abs(raw_start - kept_start) * hop_sec < 0.40 for kept_start, _ in top):
                continue
            top.append((raw_start, score))
            if len(top) >= candidates_per_anchor:
                break
        if not top:
            raise RuntimeError(f"No alignment candidates for Gold anchor {gold_start * hop_sec:.3f}s")
        candidate_rows.append(top)

    # DP over all Gold anchors. Low-correlation anchors are allowed to bridge cuts,
    # but correlation remains the dominant score. A path must progress in RAW by
    # approximately at least the Gold elapsed time; large forward gaps are allowed.
    dp: list[list[float]] = []
    parent: list[list[int | None]] = []
    for i, row in enumerate(candidate_rows):
        dp.append([-1e18] * len(row))
        parent.append([None] * len(row))
        if i == 0:
            for j, (raw_start, corr) in enumerate(row):
                dp[i][j] = corr - 0.0002 * (raw_start * hop_sec)
            continue
        gold_delta = (gold_starts[i] - gold_starts[i - 1]) * hop_sec
        for j, (raw_start, corr) in enumerate(row):
            best_score = -1e18
            best_parent = None
            for pj, (prev_raw, _prev_corr) in enumerate(candidate_rows[i - 1]):
                prev_score = dp[i - 1][pj]
                if prev_score <= -1e17:
                    continue
                raw_delta = (raw_start - prev_raw) * hop_sec
                # Windows overlap in Gold, so tolerate a small source overlap while
                # still forbidding a backwards editorial path.
                if raw_delta < max(0.02, gold_delta - 0.45):
                    continue
                extra_skip = max(0.0, raw_delta - gold_delta)
                # Only a soft penalty: true human cuts can skip tens of seconds.
                transition_penalty = 0.0008 * extra_skip
                score = prev_score + corr - transition_penalty
                if score > best_score:
                    best_score = score
                    best_parent = pj
            if best_parent is not None:
                dp[i][j] = best_score
                parent[i][j] = best_parent

    if not dp or max(dp[-1]) <= -1e17:
        raise RuntimeError("Human Gold alignment has no complete monotonic path")

    j = max(range(len(dp[-1])), key=lambda x: dp[-1][x])
    chosen: list[tuple[int, int, float]] = []
    for i in range(len(candidate_rows) - 1, -1, -1):
        raw_start, corr = candidate_rows[i][j]
        chosen.append((gold_starts[i], raw_start, corr))
        if i:
            p = parent[i][j]
            if p is None:
                raise RuntimeError("Human Gold alignment path broke before start")
            j = p
    chosen.reverse()

    anchors = tuple(
        AlignmentAnchor(
            gold_time=(g + window_frames / 2.0) * hop_sec,
            raw_time=(r + window_frames / 2.0) * hop_sec,
            correlation=float(c),
        )
        for g, r, c in chosen
    )

    trusted = tuple(a for a in anchors if a.correlation >= minimum_correlation)
    if len(trusted) < max(18, int(gold_duration / 3.0)):
        raise RuntimeError(f"Insufficient trusted Human Gold anchors: {len(trusted)}")
    if trusted[0].gold_time > 2.5 or trusted[-1].gold_time < gold_duration - 2.5:
        raise RuntimeError(
            f"Human Gold alignment does not cover full edit: first={trusted[0].gold_time:.3f}, "
            f"last={trusted[-1].gold_time:.3f}, gold={gold_duration:.3f}"
        )

    groups = coalesce_alignment_anchors(
        trusted,
        offset_tolerance_sec=0.18,
        maximum_gold_gap_sec=max(1.5, anchor_window_sec + anchor_stride_sec),
        minimum_correlation=minimum_correlation,
    )
    if not groups:
        raise RuntimeError("No Human Gold source plateaus found")

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
        if raw_start < -0.35 or raw_end > raw_duration + 0.35 or raw_end <= raw_start:
            raise RuntimeError(
                f"Invalid Gold->RAW chunk {index}: gold={gold_start:.3f}-{gold_end:.3f}, "
                f"raw={raw_start:.3f}-{raw_end:.3f}, raw_duration={raw_duration:.3f}"
            )
        if raw_start + 0.05 < previous_raw_start:
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

    mapped_gold = sum(c.duration for c in chunks)
    if abs(mapped_gold - gold_duration) > 0.10:
        raise RuntimeError(f"Gold chunk coverage mismatch: {mapped_gold:.3f} vs {gold_duration:.3f}")
    if chunks[-1].raw_end > raw_duration + 1e-6:
        raise RuntimeError("Final Gold source projection exceeds RAW")

    return anchors, tuple(chunks)


def _write_csv(report: dict, path: str | Path) -> None:
    rows = []
    for row in report.get("human_kept") or ():
        rows.append({
            "human_action": "KEEP",
            "index": row.get("gold_chunk_index"),
            "raw_start": row.get("raw_start"),
            "raw_end": row.get("raw_end"),
            "duration_sec": row.get("duration_sec"),
            "engine_selected_coverage": row.get("engine_selected_coverage"),
            "pass": row.get("selection_pass"),
            "rule_candidate": row.get("rule_candidate"),
        })
    for row in report.get("human_deleted") or ():
        rows.append({
            "human_action": "DELETE",
            "index": row.get("delete_region_index"),
            "raw_start": row.get("raw_start"),
            "raw_end": row.get("raw_end"),
            "duration_sec": row.get("duration_sec"),
            "engine_selected_coverage": row.get("engine_selected_overlap_sec"),
            "pass": not bool(row.get("engine_false_keep")),
            "rule_candidate": row.get("rule_candidate"),
        })
    fields = ["human_action", "index", "raw_start", "raw_end", "duration_sec", "engine_selected_coverage", "pass", "rule_candidate"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--engine-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args(argv)

    raw_duration = _ffprobe_duration(args.raw)
    gold_duration = _ffprobe_duration(args.gold)
    anchors, chunks = align_gold_audio_to_raw_v2(args.raw, args.gold)
    engine_result = json.loads(Path(args.engine_json).read_text(encoding="utf-8"))
    report = build_decision_map(
        raw_duration_sec=raw_duration,
        gold_duration_sec=gold_duration,
        gold_chunks=chunks,
        engine_result=engine_result,
        alignment_anchors=anchors,
    )
    report["schema_version"] = "cutsell.human_gold_decision_map.v2.1"
    report["alignment"]["trusted_anchor_count"] = sum(1 for a in anchors if a.correlation >= 0.48)
    report["alignment"]["first_anchor_gold_sec"] = round(anchors[0].gold_time, 3)
    report["alignment"]["last_anchor_gold_sec"] = round(anchors[-1].gold_time, 3)
    report["alignment"]["maximum_projected_raw_sec"] = round(max(c.raw_end for c in chunks), 3)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(report, args.out_csv)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "human_gold_chunk_count": report["human_gold_chunk_count"],
        "alignment": report["alignment"],
        "selection_parity": report["selection_parity"],
        "boundary_parity": report["boundary_parity"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
