"""RAW -> Human Gold editorial audit for CutSell Clean Cut.

Benchmark/QA tooling only: this module never becomes runtime selection authority.
It aligns a human-approved edit back to the original raw source and compares the
human source decisions with an engine result. The output deliberately separates
Selection parity from Boundary parity and also maps the RAW complement that the
human removed.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


@dataclass(frozen=True)
class AlignmentAnchor:
    gold_time: float
    raw_time: float
    correlation: float

    @property
    def offset(self) -> float:
        return float(self.raw_time) - float(self.gold_time)


@dataclass(frozen=True)
class GoldSourceChunk:
    index: int
    gold_start: float
    gold_end: float
    raw_start: float
    raw_end: float
    source_offset_sec: float
    alignment_confidence: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.gold_end) - float(self.gold_start))

    @property
    def raw_span(self) -> Span:
        return Span(self.raw_start, self.raw_end)


def _round(value: float, places: int = 3) -> float:
    return round(float(value), places)


def normalize_spans(spans: Iterable[Span], *, join_gap_sec: float = 0.0) -> tuple[Span, ...]:
    ordered = sorted(
        (Span(max(0.0, float(s.start)), max(0.0, float(s.end))) for s in spans if s.end > s.start),
        key=lambda s: (s.start, s.end),
    )
    if not ordered:
        return ()
    out = [ordered[0]]
    for span in ordered[1:]:
        last = out[-1]
        if span.start <= last.end + join_gap_sec:
            out[-1] = Span(last.start, max(last.end, span.end))
        else:
            out.append(span)
    return tuple(out)


def overlap_duration(left: Span, right: Span) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def union_intersection_duration(left: Span, spans: Iterable[Span]) -> float:
    intersections = []
    for span in spans:
        start = max(left.start, span.start)
        end = min(left.end, span.end)
        if end > start:
            intersections.append(Span(start, end))
    return sum(span.duration for span in normalize_spans(intersections))


def complement_spans(duration_sec: float, kept_spans: Iterable[Span], *, minimum_sec: float = 0.015) -> tuple[Span, ...]:
    duration_sec = max(0.0, float(duration_sec))
    kept = normalize_spans(
        (Span(max(0.0, span.start), min(duration_sec, span.end)) for span in kept_spans),
        join_gap_sec=0.0,
    )
    cursor = 0.0
    deleted: list[Span] = []
    for span in kept:
        if span.start - cursor >= minimum_sec:
            deleted.append(Span(cursor, span.start))
        cursor = max(cursor, span.end)
    if duration_sec - cursor >= minimum_sec:
        deleted.append(Span(cursor, duration_sec))
    return tuple(deleted)


def coalesce_alignment_anchors(
    anchors: Iterable[AlignmentAnchor],
    *,
    offset_tolerance_sec: float = 0.075,
    maximum_gold_gap_sec: float = 0.8,
    minimum_correlation: float = 0.55,
) -> tuple[tuple[AlignmentAnchor, ...], ...]:
    valid = tuple(sorted(
        (anchor for anchor in anchors if anchor.correlation >= minimum_correlation),
        key=lambda anchor: anchor.gold_time,
    ))
    if not valid:
        return ()
    groups: list[list[AlignmentAnchor]] = [[valid[0]]]
    for anchor in valid[1:]:
        current = groups[-1]
        previous = current[-1]
        offset_center = statistics.median(item.offset for item in current[-9:])
        same_source_plateau = abs(anchor.offset - offset_center) <= offset_tolerance_sec
        continuous_gold = anchor.gold_time - previous.gold_time <= maximum_gold_gap_sec
        if same_source_plateau and continuous_gold:
            current.append(anchor)
        else:
            groups.append([anchor])
    return tuple(tuple(group) for group in groups if group)


def chunks_from_anchor_groups(
    groups: Sequence[Sequence[AlignmentAnchor]],
    *,
    gold_duration_sec: float,
) -> tuple[GoldSourceChunk, ...]:
    if not groups:
        return ()
    boundaries = [0.0]
    for left, right in zip(groups, groups[1:]):
        boundaries.append((float(left[-1].gold_time) + float(right[0].gold_time)) / 2.0)
    boundaries.append(float(gold_duration_sec))
    chunks = []
    for index, group in enumerate(groups):
        offset = float(statistics.median(anchor.offset for anchor in group))
        confidence = float(statistics.median(anchor.correlation for anchor in group))
        gold_start = boundaries[index]
        gold_end = boundaries[index + 1]
        chunks.append(GoldSourceChunk(
            index=index + 1,
            gold_start=gold_start,
            gold_end=gold_end,
            raw_start=gold_start + offset,
            raw_end=gold_end + offset,
            source_offset_sec=offset,
            alignment_confidence=confidence,
        ))
    return tuple(chunks)


def _candidate_span(item: dict) -> Span | None:
    try:
        start = float(item.get("start"))
        end = float(item.get("end"))
    except (TypeError, ValueError):
        return None
    return Span(start, end) if end > start else None


def _engine_candidates(engine_result: dict) -> tuple[dict, ...]:
    out = []
    for bucket in ("selected", "alternates", "discarded"):
        status = "alternate" if bucket == "alternates" else bucket.rstrip("s")
        for item in engine_result.get(bucket) or ():
            span = _candidate_span(item)
            if span is None:
                continue
            out.append({
                "status": status,
                "clip_id": str(item.get("clip_id") or ""),
                "start": span.start,
                "end": span.end,
                "text": str(item.get("text") or ""),
                "take_group_id": item.get("take_group_id"),
            })
    return tuple(out)


def _overlapping_candidates(span: Span, candidates: Sequence[dict], *, minimum_overlap_sec: float = 0.04) -> list[dict]:
    rows = []
    for item in candidates:
        candidate = Span(float(item["start"]), float(item["end"]))
        overlap = overlap_duration(span, candidate)
        if overlap < minimum_overlap_sec:
            continue
        row = dict(item)
        row["overlap_sec"] = _round(overlap)
        row["overlap_ratio_of_human_span"] = _round(overlap / max(span.duration, 1e-9), 4)
        rows.append(row)
    rows.sort(key=lambda row: (-row["overlap_sec"], row["start"], row["end"]))
    return rows


def _keep_rule(overlaps: Sequence[dict], coverage: float, start_error: float | None, end_error: float | None) -> str:
    statuses = {row["status"] for row in overlaps}
    if coverage < 0.50:
        if "alternate" in statuses:
            return "best_take_or_grouping_mismatch"
        if "discarded" in statuses:
            return "clean_cut_or_hybrid_false_delete"
        return "missing_human_delivery"
    if start_error is not None and end_error is not None and max(abs(start_error), abs(end_error)) > 0.45:
        return "boundary_authority_mismatch"
    return "human_selection_matched"


def build_decision_map(
    *,
    raw_duration_sec: float,
    gold_duration_sec: float,
    gold_chunks: Sequence[GoldSourceChunk],
    engine_result: dict,
    alignment_anchors: Sequence[AlignmentAnchor] = (),
) -> dict:
    human_kept = tuple(chunk.raw_span for chunk in gold_chunks)
    human_deleted = complement_spans(raw_duration_sec, human_kept)
    engine_selected = tuple(
        span for span in (_candidate_span(item) for item in (engine_result.get("selected") or ())) if span is not None
    )
    candidates = _engine_candidates(engine_result)

    human_keep_duration = sum(span.duration for span in human_kept)
    engine_selected_duration = sum(span.duration for span in normalize_spans(engine_selected))
    overlap_total = sum(union_intersection_duration(span, engine_selected) for span in human_kept)
    recall = overlap_total / max(human_keep_duration, 1e-9)
    precision = overlap_total / max(engine_selected_duration, 1e-9)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0

    keep_rows = []
    boundary_errors: list[float] = []
    for chunk in gold_chunks:
        span = chunk.raw_span
        overlaps = _overlapping_candidates(span, candidates)
        selected_overlaps = [row for row in overlaps if row["status"] == "selected"]
        coverage = union_intersection_duration(span, engine_selected) / max(span.duration, 1e-9)
        start_error = end_error = None
        if selected_overlaps:
            best = max(selected_overlaps, key=lambda row: row["overlap_sec"])
            start_error = float(best["start"]) - span.start
            end_error = float(best["end"]) - span.end
            boundary_errors.extend((abs(start_error), abs(end_error)))
        keep_rows.append({
            "human_action": "KEEP",
            "gold_chunk_index": chunk.index,
            "gold_start": _round(chunk.gold_start),
            "gold_end": _round(chunk.gold_end),
            "raw_start": _round(chunk.raw_start),
            "raw_end": _round(chunk.raw_end),
            "duration_sec": _round(chunk.duration),
            "source_offset_sec": _round(chunk.source_offset_sec),
            "alignment_confidence": _round(chunk.alignment_confidence, 4),
            "engine_selected_coverage": _round(coverage, 4),
            "selection_pass": coverage >= 0.80,
            "boundary_start_error_sec": None if start_error is None else _round(start_error),
            "boundary_end_error_sec": None if end_error is None else _round(end_error),
            "overlapping_engine_candidates": overlaps[:8],
            "rule_candidate": _keep_rule(overlaps, coverage, start_error, end_error),
        })

    delete_rows = []
    for index, span in enumerate(human_deleted, 1):
        overlaps = _overlapping_candidates(span, candidates)
        selected_overlap = union_intersection_duration(span, engine_selected)
        threshold = min(0.25, max(0.05, span.duration * 0.20))
        false_keep = selected_overlap >= threshold
        delete_rows.append({
            "human_action": "DELETE",
            "delete_region_index": index,
            "raw_start": _round(span.start),
            "raw_end": _round(span.end),
            "duration_sec": _round(span.duration),
            "engine_selected_overlap_sec": _round(selected_overlap),
            "engine_false_keep": false_keep,
            "overlapping_engine_candidates": overlaps[:8],
            "rule_candidate": "retry_or_slack_false_keep" if false_keep else "human_delete_matched",
        })

    correlations = [anchor.correlation for anchor in alignment_anchors]
    sorted_boundary = sorted(boundary_errors)
    p95 = None
    if sorted_boundary:
        p95 = sorted_boundary[min(len(sorted_boundary) - 1, math.floor(0.95 * (len(sorted_boundary) - 1)))]
    return {
        "schema_version": "cutsell.human_gold_decision_map.v1",
        "oracle": "RAW_ORIGINAL_TO_HUMAN_EDIT",
        "raw_duration_sec": _round(raw_duration_sec, 6),
        "gold_duration_sec": _round(gold_duration_sec, 6),
        "human_gold_chunk_count": len(gold_chunks),
        "human_keep_duration_sec": _round(human_keep_duration),
        "human_delete_duration_sec": _round(sum(span.duration for span in human_deleted)),
        "alignment": {
            "anchor_count": len(alignment_anchors),
            "median_correlation": None if not correlations else _round(statistics.median(correlations), 4),
            "minimum_correlation": None if not correlations else _round(min(correlations), 4),
        },
        "selection_parity": {
            "gold_coverage_by_engine": _round(recall, 4),
            "engine_precision_against_gold": _round(precision, 4),
            "selection_f1": _round(f1, 4),
            "human_chunks_passed": sum(1 for row in keep_rows if row["selection_pass"]),
            "human_chunks_failed": sum(1 for row in keep_rows if not row["selection_pass"]),
        },
        "boundary_parity": {
            "boundary_measurement_count": len(boundary_errors),
            "mean_absolute_error_sec": None if not boundary_errors else _round(sum(boundary_errors) / len(boundary_errors)),
            "median_absolute_error_sec": None if not boundary_errors else _round(statistics.median(boundary_errors)),
            "p95_absolute_error_sec": None if p95 is None else _round(p95),
        },
        "human_kept": keep_rows,
        "human_deleted": delete_rows,
    }


def _ffprobe_duration(path: str | Path, *, ffprobe_bin: str = "ffprobe") -> float:
    result = subprocess.check_output([
        ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1", str(path),
    ], text=True).strip()
    return float(result)


def _audio_features(path: str | Path, *, ffmpeg_bin: str = "ffmpeg", sample_rate: int = 8000, hop_sec: float = 0.01):
    try:
        import numpy as np
    except ImportError as exc:  # benchmark runtime only
        raise RuntimeError("numpy is required for Human Gold alignment") from exc
    pcm = subprocess.check_output([
        ffmpeg_bin, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ])
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    hop = max(1, int(round(sample_rate * hop_sec)))
    usable = (len(audio) // hop) * hop
    if usable < hop * 10:
        raise RuntimeError(f"audio too short for alignment: {path}")
    frames = audio[:usable].reshape(-1, hop)
    rms = np.log1p(np.sqrt(np.mean(frames * frames, axis=1) + 1e-9) * 100.0)
    signs = frames >= 0
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1) if hop > 1 else np.zeros(len(frames), dtype=np.float32)
    features = np.stack([rms, zcr], axis=1).astype(np.float32)
    for column in range(features.shape[1]):
        values = features[:, column]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) or float(np.std(values)) or 1.0
        features[:, column] = (values - median) / max(mad * 1.4826, 1e-6)
    return features


def _normalized_window_correlation(search, template):
    import numpy as np
    length = int(template.shape[0])
    if search.shape[0] < length:
        return np.empty((0,), dtype=np.float32)
    combined = None
    weight_total = 0.0
    for column, weight in ((0, 0.82), (1, 0.18)):
        source = search[:, column].astype(np.float64, copy=False)
        target = template[:, column].astype(np.float64, copy=False)
        target = target - target.mean()
        target_norm = math.sqrt(float(np.dot(target, target)))
        if target_norm <= 1e-8:
            continue
        numerator = np.correlate(source, target, mode="valid")
        prefix = np.concatenate(([0.0], np.cumsum(source)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(source * source)))
        sums = prefix[length:] - prefix[:-length]
        sums_sq = prefix_sq[length:] - prefix_sq[:-length]
        variance = np.maximum(sums_sq - (sums * sums / length), 1e-10)
        corr = numerator / (np.sqrt(variance) * target_norm)
        combined = corr * weight if combined is None else combined + corr * weight
        weight_total += weight
    if combined is None or weight_total <= 0:
        return np.zeros((search.shape[0] - length + 1,), dtype=np.float32)
    return (combined / weight_total).astype(np.float32)


def align_gold_audio_to_raw(
    raw_path: str | Path,
    gold_path: str | Path,
    *,
    anchor_window_sec: float = 1.0,
    anchor_stride_sec: float = 0.25,
    hop_sec: float = 0.01,
    maximum_forward_skip_sec: float = 50.0,
    minimum_correlation: float = 0.55,
    offset_tolerance_sec: float = 0.075,
) -> tuple[tuple[AlignmentAnchor, ...], tuple[GoldSourceChunk, ...]]:
    import numpy as np
    raw_features = _audio_features(raw_path, hop_sec=hop_sec)
    gold_features = _audio_features(gold_path, hop_sec=hop_sec)
    window_frames = max(20, int(round(anchor_window_sec / hop_sec)))
    stride_frames = max(1, int(round(anchor_stride_sec / hop_sec)))
    gold_duration = _ffprobe_duration(gold_path)
    anchors: list[AlignmentAnchor] = []
    previous_raw_start = None
    previous_gold_start = None
    for gold_start in range(0, max(1, gold_features.shape[0] - window_frames + 1), stride_frames):
        if gold_start + window_frames > gold_features.shape[0]:
            break
        template = gold_features[gold_start:gold_start + window_frames]
        if previous_raw_start is None:
            raw_min = 0
            raw_max = min(raw_features.shape[0] - window_frames, int(round(120.0 / hop_sec)))
        else:
            delta = gold_start - int(previous_gold_start)
            expected = int(previous_raw_start) + delta
            raw_min = max(int(previous_raw_start) + 1, expected - int(round(0.30 / hop_sec)))
            raw_max = min(raw_features.shape[0] - window_frames, expected + int(round(maximum_forward_skip_sec / hop_sec)))
        if raw_max < raw_min:
            break
        correlations = _normalized_window_correlation(raw_features[raw_min:raw_max + window_frames], template)
        if not len(correlations):
            continue
        best_index = int(np.argmax(correlations))
        best_corr = float(correlations[best_index])
        best_raw_start = raw_min + best_index
        if best_corr < minimum_correlation and previous_raw_start is not None:
            raw_min = max(int(previous_raw_start) + 1, raw_min)
            raw_max = raw_features.shape[0] - window_frames
            correlations = _normalized_window_correlation(raw_features[raw_min:raw_max + window_frames], template)
            if len(correlations):
                best_index = int(np.argmax(correlations))
                best_corr = float(correlations[best_index])
                best_raw_start = raw_min + best_index
        if best_corr >= minimum_correlation:
            anchors.append(AlignmentAnchor(
                (gold_start + window_frames / 2.0) * hop_sec,
                (best_raw_start + window_frames / 2.0) * hop_sec,
                best_corr,
            ))
            previous_raw_start = best_raw_start
            previous_gold_start = gold_start
    groups = coalesce_alignment_anchors(
        anchors,
        offset_tolerance_sec=offset_tolerance_sec,
        maximum_gold_gap_sec=max(0.8, anchor_window_sec + anchor_stride_sec),
        minimum_correlation=minimum_correlation,
    )
    return tuple(anchors), chunks_from_anchor_groups(groups, gold_duration_sec=gold_duration)


def write_csv(report: dict, path: str | Path) -> None:
    rows = []
    for row in report.get("human_kept") or ():
        rows.append({
            "human_action": "KEEP", "index": row.get("gold_chunk_index"),
            "raw_start": row.get("raw_start"), "raw_end": row.get("raw_end"),
            "duration_sec": row.get("duration_sec"), "engine_metric": row.get("engine_selected_coverage"),
            "pass": row.get("selection_pass"), "rule_candidate": row.get("rule_candidate"),
        })
    for row in report.get("human_deleted") or ():
        rows.append({
            "human_action": "DELETE", "index": row.get("delete_region_index"),
            "raw_start": row.get("raw_start"), "raw_end": row.get("raw_end"),
            "duration_sec": row.get("duration_sec"), "engine_metric": row.get("engine_selected_overlap_sec"),
            "pass": not bool(row.get("engine_false_keep")), "rule_candidate": row.get("rule_candidate"),
        })
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "human_action", "index", "raw_start", "raw_end", "duration_sec", "engine_metric", "pass", "rule_candidate"
        ])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CutSell RAW -> Human Gold editorial decision map")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--engine-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv")
    parser.add_argument("--minimum-correlation", type=float, default=0.55)
    args = parser.parse_args(argv)
    engine_result = json.loads(Path(args.engine_json).read_text(encoding="utf-8"))
    anchors, chunks = align_gold_audio_to_raw(args.raw, args.gold, minimum_correlation=args.minimum_correlation)
    report = build_decision_map(
        raw_duration_sec=_ffprobe_duration(args.raw),
        gold_duration_sec=_ffprobe_duration(args.gold),
        gold_chunks=chunks,
        engine_result=engine_result,
        alignment_anchors=anchors,
    )
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_csv:
        write_csv(report, args.out_csv)
    print(json.dumps({
        "human_gold_chunk_count": report["human_gold_chunk_count"],
        "median_alignment_correlation": report["alignment"]["median_correlation"],
        **report["selection_parity"],
        **report["boundary_parity"],
    }, indent=2, sort_keys=True))
    if report["human_gold_chunk_count"] < 2 or (report["alignment"]["median_correlation"] or 0.0) < args.minimum_correlation:
        raise SystemExit("Human Gold alignment quality gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
