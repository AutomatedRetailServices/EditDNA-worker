"""RQ entry point for real CutSell Flow B processing."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .asr import FasterWhisperASR
from .config import load_runtime_config
from .flow_b import process_local_sources
from .providers import NoopSemanticProvider
from .semantic_openai import OpenAISemanticProvider
from .serde import request_from_dict, result_to_dict
from .storage import download_source
from .visual_openai import OpenAIVisualProvider


def run_flow_b_job(payload: dict) -> dict:
    from rq import get_current_job

    job = get_current_job()

    def publish(stage: str, percent: int) -> None:
        if job is None:
            return
        job.meta["stage"] = stage
        job.meta["progress_percent"] = max(0, min(100, int(percent)))
        job.save_meta()

    request = request_from_dict(payload)
    config = load_runtime_config()
    asr = FasterWhisperASR(model_name=config.asr_model)
    semantic = OpenAISemanticProvider(model=config.semantic_model) if config.semantic_ready else NoopSemanticProvider()
    visual = OpenAIVisualProvider(model=config.visual_model) if config.visual_ready else None

    try:
        publish("preparing", 1)
        with tempfile.TemporaryDirectory(prefix="cutsell-flow-b-") as directory:
            local_paths = {}
            for index, source in enumerate(request.sources):
                suffix = Path(source.original_name).suffix or ".mp4"
                destination = str(Path(directory) / f"{source.source_order:03d}-{source.source_asset_id}{suffix}")
                local_paths[source.source_asset_id] = download_source(source.uri, destination)
                publish("preparing", min(10, 2 + int((index + 1) * 8 / len(request.sources))))

            result = process_local_sources(
                request,
                local_paths,
                asr_provider=asr,
                semantic_provider=semantic,
                visual_provider=visual,
                progress=publish,
            )
            publish("draft_ready", 100)
            return result_to_dict(result)
    except Exception as exc:
        if job is not None:
            job.meta["stage"] = "failed"
            job.meta["error_code"] = exc.__class__.__name__
            job.save_meta()
        raise
