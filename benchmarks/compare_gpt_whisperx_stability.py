"""Read-only comparison of the five full-engine trial artifacts (CPU only)."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def lexical(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def token_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def merged_ranges(rows: list[dict]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted((r["start"], r["end"]) for r in rows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def coverage_iou(left: list[dict], right: list[dict]) -> float:
    a, b = merged_ranges(left), merged_ranges(right)
    intersection = sum(max(0, min(e, y) - max(s, x)) for s, e in a for x, y in b)
    union = sum(e - s for s, e in a + b) - intersection
    return intersection / union if union else 1.0


def read_trial(folder: Path, index: int) -> tuple[dict, dict | None]:
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"trial": index, "status": "missing"}
    result_path = folder / "result.json"
    if not result_path.exists():
        return summary, None
    r = json.loads(result_path.read_text())
    compact = json.loads((folder / "compact.json").read_text())
    local_media = folder / "local-media-validation.json"
    summary["original_harness_status"] = summary.get("status")
    summary["engine_ok"] = bool(compact.get("ok"))
    summary["deliverable"] = compact.get("deliverable")
    summary["delivery_status"] = compact.get("delivery_status")
    summary["preview_kind"] = "candidate" if compact.get("preview_uri") else "diagnostic_invalidated"
    if local_media.exists():
        summary["local_media_validation"] = json.loads(local_media.read_text())
    words = [w for s in r["timed_asr_replay_evidence"]["raw_segments"] for w in s["words"]]
    transcript = " ".join(w["text"] for w in words)
    audit = r.get("asr_provider_audit") or {}
    chunks = audit.get("chunks") or []
    durations = [w["end"] - w["start"] for w in words]
    bad = [w for w in words if not all(math.isfinite(w[k]) for k in ("start", "end"))
           or w["end"] <= w["start"] or w["start"] < 0 or w["end"] > r["source_duration_sec"] + .05]
    editorial = json.loads((folder / "editorial-qa.json").read_text())
    gold = json.loads((folder / "gold-qa.json").read_text())
    freeze = r.get("diagnostics", {}).get("selection_boundary_contract", {})
    selection = [{"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"]}
                 for s in r.get("selected", [])]
    perceptual = r.get("perceptual_watch_listen") or {}
    summary.update({
        "trial": index, "source_sha256": r.get("source_media_sha256"),
        "worker_sha": r.get("active_path_identity", {}).get("build_git_sha"),
        "package_sha256": r.get("active_path_identity", {}).get("package", {}).get("sha256"),
        "asr_config": audit.get("config_fingerprint"),
        "alignment_packages": audit.get("alignment_runtime", {}).get("packages"),
        "transcript_sha256": digest(transcript), "lexical_sha256": digest(lexical(transcript)),
        "timed_words_sha256": digest([{k: w[k] for k in ("text", "start", "end")} for w in words]),
        "selection_text_sha256": digest([s["text"] for s in selection]),
        "selection_ranges_sha256": digest([(s["start"], s["end"]) for s in selection]),
        "selection_sha256": digest(selection), "word_count": len(words),
        "raw_segment_count": len(r["timed_asr_replay_evidence"]["raw_segments"]),
        "zero_duration_words": sum(d == 0 for d in durations), "invalid_word_count": len(bad),
        "short_words_under_25ms": sum(d < .025 for d in durations),
        "long_words_over_1_5s": [{"text": w["text"], "start": w["start"], "end": w["end"],
                                 "duration": round(w["end"] - w["start"], 3)}
                                for w in words if w["end"] - w["start"] > 1.5],
        "maximum_word_duration_sec": round(max(durations, default=0), 3),
        "recorded_gpt_requests": len(chunks), "cache_hit_count": audit.get("cache_hit_count"),
        "request_ids": [c.get("request_id") for c in chunks],
        "hard_boundaries": [c["end"] for c in chunks if c.get("end_boundary") == "hard_window"],
        "asr_elapsed_sec": audit.get("elapsed_sec"),
        "alignment_elapsed_sec": audit.get("alignment_runtime", {}).get("elapsed_sec"),
        "engine_elapsed_sec": r.get("elapsed_sec"), "selected_count": len(selection),
        "output_duration_sec": r.get("output_duration_sec"), "selected_duration_sec": r.get("selected_duration_sec"),
        "technical_qc": r.get("live_render_qc", {}).get("status"),
        "freeze_status": freeze.get("status"), "freeze_matches_reviewed_plan": freeze.get("matches_reviewed_plan"),
        "editorial_passed": editorial["passed_check_count"], "editorial_failed": editorial["failed_checks"],
        "gold_passed": gold["passed_check_count"], "gold_failed": gold["failed_checks"],
        "perceptual_status": perceptual.get("status"), "perceptual_findings": perceptual.get("finding_count"),
        "perceptual_capability_counts": perceptual.get("capability_status_counts"),
        "human_watch_listen_required": perceptual.get("human_watch_listen_required"),
        "post_render_microtrim_count": r.get("auto_microtrim_count"),
        "source_text_matches_chunk_text": transcript == " ".join(c["text"] for c in chunks),
        "raw_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    })
    return summary, r


def compare_pair(i: int, a: dict, j: int, b: dict) -> dict:
    aw = [w for s in a["timed_asr_replay_evidence"]["raw_segments"] for w in s["words"]]
    bw = [w for s in b["timed_asr_replay_evidence"]["raw_segments"] for w in s["words"]]
    at = lexical(" ".join(w["text"] for w in aw))
    bt = lexical(" ".join(w["text"] for w in bw))
    distance = token_distance(at, bt)
    # Timing comparisons use matching word sequences only. These are alignment
    # stability measurements, not phoneme accuracy or a ground-truth WER.
    matcher = difflib.SequenceMatcher(None, [w["text"] for w in aw], [w["text"] for w in bw], autojunk=False)
    timing = []
    changed = []
    for tag, x1, x2, y1, y2 in matcher.get_opcodes():
        if tag == "equal":
            for x, y in zip(aw[x1:x2], bw[y1:y2]):
                timing.extend((abs(x["start"] - y["start"]), abs(x["end"] - y["end"])))
        else:
            changed.append({"change": tag, "trial_a_text": " ".join(w["text"] for w in aw[x1:x2]),
                            "trial_b_text": " ".join(w["text"] for w in bw[y1:y2]),
                            "source_time_a": aw[x1]["start"] if x1 < len(aw) else None,
                            "source_time_b": bw[y1]["start"] if y1 < len(bw) else None})
    timing.sort()
    return {"trial_a": i, "trial_b": j, "lexical_token_edit_distance": distance,
            "lexical_difference_percent": round(100 * distance / max(len(at), len(bt), 1), 4),
            "selected_source_coverage_iou_percent": round(100 * coverage_iou(a["selected"], b["selected"]), 3),
            "matching_timed_word_count": len(timing) // 2,
            "matching_boundary_median_drift_sec": round(statistics.median(timing), 4) if timing else None,
            "matching_boundary_p95_drift_sec": round(timing[math.ceil(.95 * len(timing)) - 1], 4) if timing else None,
            "matching_boundary_max_drift_sec": round(max(timing), 4) if timing else None,
            "text_differences": changed}


def build_comparison(artifacts: Path) -> tuple[dict, dict[int, dict]]:
    summaries, results = [], {}
    for index in range(1, 6):
        summary, result = read_trial(artifacts / f"trial-{index}", index)
        summaries.append(summary)
        if result:
            results[index] = result
    pairs = [compare_pair(i, a, j, b) for (i, a), (j, b) in itertools.combinations(results.items(), 2)]
    available = [s for s in summaries if s.get("transcript_sha256")]
    checks = sorted(set(c["id"] for s in available for c in s["editorial_failed"]))
    aggregate = {
        "requested_trials": 5, "results_available": len(available),
        "engine_runs_completed": sum(s.get("engine_ok") is True for s in available),
        "technical_passes": sum(s.get("technical_qc") == "PASS" for s in available),
        "editorial_passes": sum(not s["editorial_failed"] for s in available),
        "unique_exact_transcripts": len({s["transcript_sha256"] for s in available}),
        "unique_lexical_transcripts": len({s["lexical_sha256"] for s in available}),
        "unique_timed_transcripts": len({s["timed_words_sha256"] for s in available}),
        "unique_selections": len({s["selection_sha256"] for s in available}),
        "unique_selection_texts": len({s["selection_text_sha256"] for s in available}),
        "unique_selection_ranges": len({s["selection_ranges_sha256"] for s in available}),
        "same_source_all_five": len(available) == 5 and len({s["source_sha256"] for s in available}) == 1,
        "same_code_all_five": len(available) == 5 and len({s["worker_sha"] for s in available}) == 1,
        "same_package_all_five": len(available) == 5 and len({s["package_sha256"] for s in available}) == 1,
        "same_asr_config_all_five": len(available) == 5 and len({s["asr_config"] for s in available}) == 1,
        "same_alignment_packages_all_five": len(available) == 5 and len({digest(s["alignment_packages"]) for s in available}) == 1,
        "recorded_gpt_requests": sum(s["recorded_gpt_requests"] for s in available),
        "distinct_request_ids": len({rid for s in available for rid in s["request_ids"] if rid}),
        "total_cache_hits": sum(s.get("cache_hit_count") or 0 for s in available),
        "editorial_failure_frequency": {c: sum(any(f["id"] == c for f in s["editorial_failed"]) for s in available) for c in checks},
    }
    return {"schema": "cutsell.gpt_whisperx_five_comparison.v1", "aggregate": aggregate,
            "trials": summaries, "pairwise": pairs,
            "limitations": ["No human listening was performed.",
                            "Pairwise text differences are not accuracy/WER against ground truth.",
                            "Identical source/code does not remove stochastic ASR or Gemini editorial behavior.",
                            "CTC word confidence is alignment confidence, not GPT lexical confidence.",
                            "All five tests use one source; no cross-video generalization established.",
                            "No billing query; request counts are recorded evidence, not dollars."]}, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data, results = build_comparison(args.artifacts)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "Video00_GPT_WhisperX_5_Resultados.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(data["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
