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


def run_flow_b_job(payload: dict) -> dict:
    request = request_from_dict(payload)
    config = load_runtime_config()
    asr = FasterWhisperASR(model_name=config.asr_model)
    semantic = OpenAISemanticProvider(model=config.semantic_model) if config.semantic_ready else NoopSemanticProvider()

    with tempfile.TemporaryDirectory(prefix="cutsell-flow-b-") as directory:
        local_paths = {}
        for source in request.sources:
            suffix = Path(source.original_name).suffix or ".mp4"
            destination = str(Path(directory) / f"{source.source_order:03d}-{source.source_asset_id}{suffix}")
            local_paths[source.source_asset_id] = download_source(source.uri, destination)

        result = process_local_sources(
            request,
            local_paths,
            asr_provider=asr,
            semantic_provider=semantic,
        )
        return result_to_dict(result)
