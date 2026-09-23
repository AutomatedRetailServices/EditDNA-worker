"""Bounded real-footage scanner for finding retry-rich validation sources.

This is validation tooling only. It uses ASR + segmentation + lexical retry grouping,
never the legacy editor and never OpenAI. The goal is to find a real source where
Best Take can be exercised honestly instead of forcing singleton groups.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from .asr import FasterWhisperASR
from .config import load_runtime_config
from .contracts import SourceAsset
from .media_probe import probe_media
from .silence_analysis import word_silence_gaps
from .source_identity import stable_source_id
from .storage import download_source
from .take_grouping import group_takes, retry_similarity
from .take_segmentation import segment_takes
from .validation import list_validation_videos


def _top_similarity_pairs(takes, limit: int = 5) -> list[dict[str, Any]]:
    scored = []
    for left_index, left in enumerate(takes):
        for right in takes[left_index + 1 :]:
            if left.source_asset_id != right.source_asset_id:
                continue
            score = retry_similarity(left.text, right.text)
            if score <= 0:
                continue
            scored.append({
                "left_id": left.clip_id,
                "right_id": right.clip_id,
                "left_text": left.text,
                "right_text": right.text,
                "score": score,
                "gap_sec": round(max(0.0, right.start - left.end), 3),
            })
    scored.sort(key=lambda item: (-item["score"], item["gap_sec"], item["left_id"], item["right_id"]))
    return scored[:limit]


def scan_retry_rich_sources(
    *,
    candidate_limit: int = 8,
    max_source_bytes: int = 120 * 1024 * 1024,
    language_hint: str | None = None,
) -> dict[str, Any]:
    if not 1 <= candidate_limit <= 20:
        raise ValueError("candidate_limit must be between 1 and 20")
    config = load_runtime_config()
    videos = [item for item in list_validation_videos(limit=100) if item["size"] <= max_source_bytes]
    videos.sort(key=lambda item: (item["size"], item["key"].casefold()))
    videos = videos[:candidate_limit]
    if not videos:
        raise RuntimeError("no bounded validation videos available for retry scan")

    asr = FasterWhisperASR(model_name=config.asr_model)
    results = []
    best = None

    with tempfile.TemporaryDirectory(prefix="cutsell-retry-scan-") as directory:
        for source_order, item in enumerate(videos):
            key = item["key"]
            source_id = stable_source_id("cutsell-retry-scan", source_order, PurePosixPath(key).name)
            source = SourceAsset(
                source_asset_id=source_id,
                project_id="cutsell-retry-scan",
                user_id="validation",
                original_name=PurePosixPath(key).name,
                source_order=source_order,
                duration_sec=0.0,
                uri=f"s3://{config.s3_bucket}/{key}",
            )
            destination = str(Path(directory) / f"{source_order:03d}-{source.original_name}")
            local = download_source(source.uri, destination)
            probe = probe_media(local)
            source = replace(
                source,
                duration_sec=probe.duration_sec,
                has_audio=probe.has_audio,
                metadata={"width": probe.width, "height": probe.height, "fps": probe.fps},
            )
            transcripts = ()
            if source.has_audio:
                transcripts = asr.transcribe(local, source_asset_id=source_id, language_hint=language_hint)
            gaps = word_silence_gaps(transcripts)
            takes = segment_takes(transcripts, (source,), gaps)
            groups = group_takes(takes)
            retry_groups = [members for members in groups.values() if len(members) >= 2]
            report = {
                "source_key": key,
                "size": item["size"],
                "duration_sec": round(probe.duration_sec, 3),
                "candidate_count": len(takes),
                "retry_group_count": len(retry_groups),
                "retry_group_sizes": sorted((len(group) for group in retry_groups), reverse=True),
                "retry_groups": [
                    [{"clip_id": take.clip_id, "start": take.start, "end": take.end, "text": take.text} for take in group]
                    for group in retry_groups[:3]
                ],
                "top_similarity_pairs": _top_similarity_pairs(takes),
            }
            results.append(report)
            score = (len(retry_groups), max(report["retry_group_sizes"], default=0), len(takes))
            if best is None or score > best[0]:
                best = (score, report)

    return {
        "schema_version": "cutsell.retry-scan.v1",
        "asr_model": config.asr_model,
        "scanned_count": len(results),
        "best_source": best[1] if best else None,
        "sources": results,
    }
