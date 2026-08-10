"""Real-video validation harness for Universal Clean Cut only."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any

from .asr import FasterWhisperASR
from .clean_cut_openai import OpenAICleanCutProvider
from .config import load_runtime_config
from .contracts import ProcessingRequest, SourceAsset
from .media_probe import probe_media
from .render import render_preview
from .render_plan import build_render_plan
from .source_identity import stable_source_id
from .storage import download_source
from .take_grouping_openai import OpenAITakeGroupingProvider
from .take_judge_openai import OpenAITakeJudgeProvider
from .universal_clean_cut import process_universal_clean_cut_sources
from .validation import _is_real_video_key
from .visual_openai import OpenAIVisualProvider
from .whole_video_openai import OpenAIWholeVideoProvider


def run_single_universal_clean_cut_validation(
    key: str,
    *,
    project_id: str = "cutsell-universal-clean-cut-validation",
    language_hint: str | None = None,
    preview_output: str | None = None,
    preview_captions: bool = False,
) -> dict[str, Any]:
    """Run one full S3 raw through the isolated Universal Clean Cut brain."""
    if not _is_real_video_key(key):
        raise ValueError("unsupported validation video")

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
    whole_video = OpenAIWholeVideoProvider(model=config.visual_model) if config.visual_ready else None
    visual = OpenAIVisualProvider(model=config.visual_model) if config.visual_ready else None
    take_grouping = OpenAITakeGroupingProvider(model=config.semantic_model) if config.semantic_ready else None
    take_judge = OpenAITakeJudgeProvider(model=config.take_judge_model) if config.semantic_ready else None
    clean_cut_judge = (
        OpenAICleanCutProvider(model=config.clean_cut_judge_model)
        if config.clean_cut_judge_ready
        else None
    )

    started = time.monotonic()
    preview_path = None
    with tempfile.TemporaryDirectory(prefix="cutsell-universal-clean-cut-") as directory:
        destination = str(Path(directory) / source.original_name)
        local = download_source(source.uri, destination)
        source_duration_sec = float(probe_media(local).duration_sec)
        local_paths = {source_id: local}

        result = process_universal_clean_cut_sources(
            request,
            local_paths,
            asr_provider=asr,
            whole_video_provider=whole_video,
            visual_provider=visual,
            take_grouping_provider=take_grouping,
            take_judge_provider=take_judge,
            clean_cut_provider=clean_cut_judge,
        )

        if preview_output:
            plan = build_render_plan(result.draft, local_paths)
            if not preview_captions:
                plan = tuple(replace(segment, caption_text="") for segment in plan)
            preview_path = render_preview(plan, preview_output)

    elapsed = round(time.monotonic() - started, 3)
    selected_duration_sec = round(
        sum(max(0.0, clip.end - clip.start) for clip in result.draft.selected), 3
    )
    diagnostics = result.draft.diagnostics
    clean_decisions = list(diagnostics.get("clean_cut_decisions") or ())
    temporal = list(diagnostics.get("temporal_performance_trims") or ())

    return {
        "schema_version": result.schema_version,
        "benchmark_mode": "universal_clean_cut",
        "project_id": result.project_id,
        "source_key": key,
        "source_duration_sec": round(source_duration_sec, 3),
        "selected_duration_sec": selected_duration_sec,
        "selected_to_input_ratio": round(selected_duration_sec / source_duration_sec, 4) if source_duration_sec else None,
        "elapsed_sec": elapsed,
        "preview_path": preview_path,
        "preview_captions": bool(preview_captions),
        "selected_count": len(result.draft.selected),
        "alternate_count": len(result.draft.alternates),
        "discarded_count": len(result.draft.discarded),
        "clean_cut_removed_count": sum(1 for item in clean_decisions if not bool(item.get("keep", True))),
        "temporal_trimmed_count": sum(1 for item in temporal if bool(item.get("applied"))),
        "models": {
            "asr": config.asr_model,
            "whole_video": config.visual_model if config.visual_ready else None,
            "visual": config.visual_model if config.visual_ready else None,
            "take_grouping": config.semantic_model if config.semantic_ready else None,
            "take_judge": config.take_judge_model if config.semantic_ready else None,
            "clean_cut_judge": config.clean_cut_judge_model if config.clean_cut_judge_ready else None,
            "semantic_sales": None,
            "composer": None,
            "draft_review": None,
        },
        "stage_status": result.stage_status,
        "diagnostics": diagnostics,
        "selected": [
            {
                "clip_id": clip.clip_id,
                "start": clip.start,
                "end": clip.end,
                "text": clip.text,
                "take_group_id": clip.take_group_id,
            }
            for clip in result.draft.selected
        ],
        "alternates": [
            {
                "clip_id": clip.clip_id,
                "start": clip.start,
                "end": clip.end,
                "text": clip.text,
                "take_group_id": clip.take_group_id,
            }
            for clip in result.draft.alternates
        ],
        "discarded": [
            {"clip_id": clip.clip_id, "start": clip.start, "end": clip.end, "text": clip.text}
            for clip in result.draft.discarded
        ],
    }
