"""RQ entry point for real CutSell Flow B processing."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .asr import FasterWhisperASR
from .brain_runtime import build_brain_runtime
from .config import load_runtime_config
from .draft_store import create_initial_draft
from .flow_b import process_local_sources
from .media_probe import probe_media
from .notifications import publish_notification
from .project_tracking import safe_update_project
from .serde import request_from_dict, result_to_dict
from .storage import download_source
from .timeline_asset_storage import store_timeline_assets
from .timeline_assets import generate_filmstrip, waveform_peaks
from .uploads import validate_product_source_uri
from .usage_limits import record_processing_minutes, release_processing_slot


def _build_timeline_assets(request, local_paths: dict[str, str], directory: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for source in request.sources:
        try:
            source_path = local_paths[source.source_asset_id]
            probe = probe_media(source_path)
            asset_dir = str(Path(directory) / "timeline-assets" / source.source_asset_id)
            filmstrip = generate_filmstrip(
                source_path,
                asset_dir,
                duration_sec=probe.duration_sec,
                max_frames=24,
                width=160,
            )
            waveform = waveform_peaks(source_path, buckets=256)
            output[source.source_asset_id] = store_timeline_assets(
                user_id=request.user_id,
                project_id=request.project_id,
                source_asset_id=source.source_asset_id,
                filmstrip=filmstrip,
                waveform=waveform,
            )
        except Exception as exc:
            output[source.source_asset_id] = {
                "status": "degraded",
                "reason": exc.__class__.__name__,
                "filmstrip": [],
                "waveform_uri": None,
                "waveform_bucket_count": 0,
            }
    return output


def _safe_notify(*, user_id: str, project_id: str, kind: str, payload: dict | None = None) -> dict:
    try:
        event = publish_notification(
            user_id=user_id,
            project_id=project_id,
            kind=kind,
            payload=payload,
        )
        return {"status": "queued", "notification_id": event["notification_id"]}
    except Exception as exc:
        return {"status": "degraded", "reason": exc.__class__.__name__}


def run_flow_b_job(payload: dict) -> dict:
    """Run the CutSell Flow B brain inside the RunPod RQ worker.

    Local ASR/vision/timing remains the backbone. Gemini semantic reasoning can be
    added only through brain_runtime's explicit Hybrid feature flag + approved model +
    per-edit dollar guard; stored keys alone cannot activate paid inference.
    """
    from rq import get_current_job

    job = get_current_job()

    def publish(stage: str, percent: int) -> None:
        if job is None:
            return
        job.meta["stage"] = stage
        job.meta["progress_percent"] = max(0, min(100, int(percent)))
        job.save_meta()

    request = request_from_dict(payload)
    job_id = str(getattr(job, "id", "") or "") or None
    basic_sources = [
        {
            "source_asset_id": source.source_asset_id,
            "original_name": source.original_name,
            "source_order": source.source_order,
            "duration_sec": source.duration_sec,
            "uri": source.uri,
        }
        for source in request.sources
    ]
    tracking_start = safe_update_project(
        user_id=request.user_id,
        project_id=request.project_id,
        state="processing",
        sources=basic_sources,
        latest_job_id=job_id,
    )

    config = load_runtime_config()
    brain = build_brain_runtime(config)
    asr = FasterWhisperASR(model_name=config.asr_model)

    measured_seconds = 0.0
    outcome = "failed"
    try:
        publish("preparing", 1)
        with tempfile.TemporaryDirectory(prefix="cutsell-flow-b-") as directory:
            local_paths = {}
            measured_by_source: dict[str, float] = {}
            for index, source in enumerate(request.sources):
                validate_product_source_uri(
                    source.uri,
                    project_id=request.project_id,
                    user_id=request.user_id,
                )
                suffix = Path(source.original_name).suffix or ".mp4"
                destination = str(Path(directory) / f"{source.source_order:03d}-{source.source_asset_id}{suffix}")
                local_paths[source.source_asset_id] = download_source(source.uri, destination)
                probe = probe_media(local_paths[source.source_asset_id])
                measured_by_source[source.source_asset_id] = float(probe.duration_sec)
                measured_seconds += max(0.0, float(probe.duration_sec))
                publish("preparing", min(10, 2 + int((index + 1) * 8 / len(request.sources))))

            result = process_local_sources(
                request,
                local_paths,
                asr_provider=asr,
                semantic_provider=brain.semantic_provider,
                whole_video_provider=brain.whole_video_provider,
                visual_provider=brain.visual_provider,
                take_grouping_provider=brain.take_grouping_provider,
                take_judge_provider=brain.take_judge_provider,
                clean_cut_provider=brain.clean_cut_provider,
                composer_provider=brain.composer_provider,
                draft_review_provider=brain.draft_review_provider,
                editorial_judge=brain.editorial_judge,
                progress=publish,
            )
            serialized = result_to_dict(result)
            serialized["brain_backend"] = brain.backend
            serialized["external_brain_calls_enabled"] = brain.external_calls_enabled
            serialized["hybrid_provider"] = brain.hybrid_settings.provider
            serialized["hybrid_primary_model"] = brain.hybrid_settings.primary_model

            publish("draft_ready", 94)
            timeline_assets = _build_timeline_assets(request, local_paths, directory)
            source_records = [
                {
                    "source_asset_id": source.source_asset_id,
                    "original_name": source.original_name,
                    "source_order": source.source_order,
                    "duration_sec": measured_by_source.get(source.source_asset_id, source.duration_sec),
                    "uri": source.uri,
                    "timeline_assets": timeline_assets.get(source.source_asset_id, {"status": "not_generated"}),
                }
                for source in request.sources
            ]
            create_initial_draft(
                user_id=request.user_id,
                project_id=request.project_id,
                draft=dict(serialized["draft"]),
                sources=source_records,
            )
            serialized["timeline_assets"] = timeline_assets
            serialized["project_tracking_start"] = tracking_start
            serialized["project_tracking"] = safe_update_project(
                user_id=request.user_id,
                project_id=request.project_id,
                state="draft_ready",
                sources=source_records,
                latest_job_id=job_id,
            )
            serialized["notification"] = _safe_notify(
                user_id=request.user_id,
                project_id=request.project_id,
                kind="draft_ready",
                payload={"job_id": job_id, "selected_count": len(serialized["draft"].get("selected") or [])},
            )
            outcome = "draft_ready"
            publish("draft_ready", 100)
            return serialized
    except Exception as exc:
        safe_update_project(
            user_id=request.user_id,
            project_id=request.project_id,
            state="failed",
            latest_job_id=job_id,
        )
        _safe_notify(
            user_id=request.user_id,
            project_id=request.project_id,
            kind="processing_failed",
            payload={"job_id": job_id, "error": exc.__class__.__name__},
        )
        if job is not None:
            job.meta["stage"] = "failed"
            job.meta["error_code"] = exc.__class__.__name__
            job.save_meta()
        raise
    finally:
        if measured_seconds > 0:
            record_processing_minutes(
                user_id=request.user_id,
                project_id=request.project_id,
                minutes=measured_seconds / 60.0,
                metadata={"job_id": job_id, "outcome": outcome, "source_count": len(request.sources)},
            )
        release_processing_slot(user_id=request.user_id)