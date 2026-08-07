"""Small real-video validation harness for the clean CutSell worker.

This intentionally does not import the legacy editing pipeline. It only reuses the
existing S3 bucket credentials and runs the clean Flow B worker against one source.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
from typing import Any

from .asr import FasterWhisperASR
from .config import load_runtime_config
from .contracts import ProcessingRequest, SourceAsset
from .flow_b import process_local_sources
from .providers import NoopSemanticProvider
from .render import render_preview
from .render_plan import build_render_plan
from .semantic_openai import OpenAISemanticProvider
from .source_identity import stable_source_id
from .storage import download_source
from .take_judge_openai import OpenAITakeJudgeProvider
from .visual_openai import OpenAIVisualProvider

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
DEFAULT_VALIDATION_PREFIX = "Editdna bloopers videos/"
MAX_VALIDATION_WINDOW_SEC = 180.0


def _safe_prefix(prefix: str) -> str:
    path = PurePosixPath(prefix)
    if not prefix or prefix.startswith("/") or ".." in path.parts or "\\" in prefix or not prefix.endswith("/"):
        raise ValueError("validation prefix must be a safe relative S3 prefix ending in '/'")
    return prefix


def _is_real_video_key(key: str) -> bool:
    """Reject directory markers and common macOS metadata masquerading as media."""
    if not key or key.endswith("/"):
        return False
    name = PurePosixPath(key).name
    if name.startswith("._") or name in {".DS_Store", "Thumbs.db"}:
        return False
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


def list_validation_videos(*, prefix: str | None = None, limit: int = 20, s3=None) -> tuple[dict[str, Any], ...]:
    """List a bounded set of real videos from the dedicated validation prefix.

    S3 listings are paginated because folders copied from macOS can contain many
    metadata objects before the first actual MP4. Discovery remains bounded: at
    most 20 S3 pages are scanned and results stop as soon as ``limit`` videos are
    found.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
    prefix = _safe_prefix(prefix or os.getenv("CUTSELL_VALIDATION_PREFIX", DEFAULT_VALIDATION_PREFIX))
    if s3 is None:
        import boto3
        s3 = boto3.client("s3", region_name=config.aws_region or "us-east-1")

    videos = []
    continuation = None
    for _page in range(20):
        request = {
            "Bucket": config.s3_bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if continuation:
            request["ContinuationToken"] = continuation
        response = s3.list_objects_v2(**request)
        for item in response.get("Contents", []):
            key = str(item.get("Key") or "")
            if not _is_real_video_key(key):
                continue
            size = int(item.get("Size") or 0)
            if size <= 64 * 1024:
                continue
            videos.append({"key": key, "size": size})
            if len(videos) >= limit:
                return tuple(videos)
        if not response.get("IsTruncated"):
            break
        continuation = str(response.get("NextContinuationToken") or "")
        if not continuation:
            break
    return tuple(videos)


def _extract_validation_window(
    source_path: str,
    destination: str,
    *,
    start_sec: float,
    end_sec: float,
    runner=subprocess.run,
) -> str:
    start = float(start_sec)
    end = float(end_sec)
    if start < 0 or end <= start:
        raise ValueError("validation window must have 0 <= start < end")
    duration = end - start
    if duration > MAX_VALIDATION_WINDOW_SEC:
        raise ValueError("validation window exceeds bounded duration")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", source_path,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-c:a", "aac", "-movflags", "+faststart", destination,
    ]
    runner(command, capture_output=True, check=True)
    return destination


def run_single_validation(
    key: str,
    *,
    project_id: str = "cutsell-validation",
    language_hint: str | None = None,
    preview_output: str | None = None,
    source_start_sec: float | None = None,
    source_end_sec: float | None = None,
) -> dict[str, Any]:
    """Run one S3 video (or a bounded real window) through the production clean brain."""
    if not _is_real_video_key(key):
        raise ValueError("unsupported validation video")
    if (source_start_sec is None) != (source_end_sec is None):
        raise ValueError("validation window requires both start and end")
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")
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
    visual = OpenAIVisualProvider(model=config.visual_model) if config.visual_ready else None
    take_judge = OpenAITakeJudgeProvider(model=config.take_judge_model) if config.semantic_ready else None

    started = time.monotonic()
    preview_path = None
    with tempfile.TemporaryDirectory(prefix="cutsell-validation-") as directory:
        destination = str(Path(directory) / source.original_name)
        local = download_source(source.uri, destination)
        active_local = local
        window = None
        if source_start_sec is not None and source_end_sec is not None:
            clipped = str(Path(directory) / "validation-window.mp4")
            active_local = _extract_validation_window(
                local,
                clipped,
                start_sec=source_start_sec,
                end_sec=source_end_sec,
            )
            window = {"start_sec": float(source_start_sec), "end_sec": float(source_end_sec)}
        local_paths = {source_id: active_local}
        result = process_local_sources(
            request,
            local_paths,
            asr_provider=asr,
            semantic_provider=semantic,
            visual_provider=visual,
            take_judge_provider=take_judge,
        )
        if preview_output:
            plan = build_render_plan(result.draft, local_paths)
            preview_path = render_preview(plan, preview_output)
    elapsed = round(time.monotonic() - started, 3)
    return {
        "schema_version": result.schema_version,
        "project_id": result.project_id,
        "source_key": key,
        "source_window": window,
        "elapsed_sec": elapsed,
        "preview_path": preview_path,
        "models": {
            "asr": config.asr_model,
            "semantic": config.semantic_model if config.semantic_ready else None,
            "visual": config.visual_model if config.visual_ready else None,
            "take_judge": config.take_judge_model if config.semantic_ready else None,
        },
        "strategy": result.draft.strategy.value,
        "selected_count": len(result.draft.selected),
        "alternate_count": len(result.draft.alternates),
        "discarded_count": len(result.draft.discarded),
        "diagnostics": result.draft.diagnostics,
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
            {"clip_id": clip.clip_id, "start": clip.start, "end": clip.end, "text": clip.text, "take_group_id": clip.take_group_id}
            for clip in result.draft.alternates
        ],
        "discarded": [
            {"clip_id": clip.clip_id, "start": clip.start, "end": clip.end, "text": clip.text}
            for clip in result.draft.discarded
        ],
        "stage_status": result.stage_status,
    }


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
