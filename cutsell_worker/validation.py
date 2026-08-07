"""Small real-video validation harness for the clean CutSell worker.

This intentionally does not import the legacy editing pipeline. It only reuses the
existing S3 bucket credentials and runs the clean Flow B worker against one source.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any

from .asr import FasterWhisperASR
from .config import load_runtime_config
from .contracts import ProcessingRequest, SourceAsset
from .flow_b import process_local_sources
from .providers import NoopSemanticProvider
from .semantic_openai import OpenAISemanticProvider
from .source_identity import stable_source_id
from .storage import download_source

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
DEFAULT_VALIDATION_PREFIX = "Editdna bloopers videos/"


def _safe_prefix(prefix: str) -> str:
    path = PurePosixPath(prefix)
    if not prefix or prefix.startswith("/") or ".." in path.parts or "\\" in prefix or not prefix.endswith("/"):
        raise ValueError("validation prefix must be a safe relative S3 prefix ending in '/'")
    return prefix


def list_validation_videos(*, prefix: str | None = None, limit: int = 20, s3=None) -> tuple[dict[str, Any], ...]:
    """List a bounded set of videos from the dedicated validation prefix."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    prefix = _safe_prefix(prefix or os.getenv("CUTSELL_VALIDATION_PREFIX", DEFAULT_VALIDATION_PREFIX))
    if s3 is None:
        import boto3
        s3 = boto3.client("s3", region_name=config.aws_region or "us-east-1")
    response = s3.list_objects_v2(Bucket=config.s3_bucket, Prefix=prefix, MaxKeys=min(1000, limit * 5))
    videos = []
    for item in response.get("Contents", []):
        key = str(item.get("Key") or "")
        if key.endswith("/") or Path(key).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        size = int(item.get("Size") or 0)
        if size <= 1024:
            continue
        videos.append({"key": key, "size": size})
        if len(videos) >= limit:
            break
    return tuple(videos)


def run_single_validation(
    key: str,
    *,
    project_id: str = "cutsell-validation",
    language_hint: str | None = None,
) -> dict[str, Any]:
    """Run one S3 video through the clean Flow B engine and return a compact report."""
    if Path(key).suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("unsupported validation video")
    config = load_runtime_config()
    source_id = stable_source_id(project_id, 0, PurePosixPath(key).name)
    source = SourceAsset(
        source_asset_id=source_id,
        project_id=project_id,
        user_id="validation",
        original_name=PurePosixPath(key).name,
        source_order=0,
        duration_sec=0.0,
        uri=f"s3://{config.s3_bucket}/{key}",
    )
    request = ProcessingRequest(
        project_id=project_id,
        user_id="validation",
        sources=(source,),
        language_hint=language_hint,
    )
    asr = FasterWhisperASR(model_name=config.asr_model)
    semantic = OpenAISemanticProvider(model=config.semantic_model) if config.semantic_ready else NoopSemanticProvider()

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cutsell-validation-") as directory:
        destination = str(Path(directory) / source.original_name)
        local = download_source(source.uri, destination)
        result = process_local_sources(
            request,
            {source_id: local},
            asr_provider=asr,
            semantic_provider=semantic,
        )
    elapsed = round(time.monotonic() - started, 3)
    return {
        "schema_version": result.schema_version,
        "project_id": result.project_id,
        "source_key": key,
        "elapsed_sec": elapsed,
        "strategy": result.draft.strategy.value,
        "selected_count": len(result.draft.selected),
        "alternate_count": len(result.draft.alternates),
        "discarded_count": len(result.draft.discarded),
        "selected": [
            {
                "clip_id": clip.clip_id,
                "start": clip.start,
                "end": clip.end,
                "text": clip.text,
                "semantic_role": clip.semantic_role.value,
                "take_group_id": clip.take_group_id,
            }
            for clip in result.draft.selected
        ],
        "alternates": [
            {"clip_id": clip.clip_id, "text": clip.text, "take_group_id": clip.take_group_id}
            for clip in result.draft.alternates
        ],
        "discarded": [
            {"clip_id": clip.clip_id, "text": clip.text}
            for clip in result.draft.discarded
        ],
        "stage_status": result.stage_status,
    }


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
