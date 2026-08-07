"""RQ export job: edited Draft Timeline -> final MP4 -> scoped S3 URL."""
from __future__ import annotations

from pathlib import Path
import tempfile

from .draft_edits import DraftEditError
from .exports import store_export
from .media_overlay_render import LocalMediaOverlay
from .overlay_uploads import validate_overlay_uri
from .render import render_preview
from .render_plan import build_render_plan
from .serde import draft_from_dict
from .storage import download_source
from .uploads import validate_product_source_uri


def run_export_job(payload: dict) -> dict:
    from rq import get_current_job

    job = get_current_job()

    def publish(stage: str, percent: int) -> None:
        if job is None:
            return
        job.meta["stage"] = stage
        job.meta["progress_percent"] = max(0, min(100, int(percent)))
        job.save_meta()

    project_id = str(payload["project_id"])
    user_id = str(payload["user_id"])
    draft = draft_from_dict(dict(payload["draft"]))
    if draft.project_id != project_id:
        raise DraftEditError("draft project does not match export project")
    sources = list(payload.get("sources") or ())
    if not sources:
        raise ValueError("export requires source metadata")

    publish("rendering", 2)
    with tempfile.TemporaryDirectory(prefix="cutsell-export-") as directory:
        local_paths = {}
        seen_source_ids = set()
        for index, item in enumerate(sources):
            source_id = str(item["source_asset_id"])
            if source_id in seen_source_ids:
                raise ValueError("export source_asset_id values must be unique")
            seen_source_ids.add(source_id)
            uri = str(item["uri"])
            validate_product_source_uri(uri, project_id=project_id, user_id=user_id)
            suffix = Path(str(item.get("original_name") or "source.mp4")).suffix or ".mp4"
            destination = str(Path(directory) / f"source-{index:03d}-{source_id}{suffix}")
            local_paths[source_id] = download_source(uri, destination)
            publish("rendering", min(25, 5 + int((index + 1) * 20 / len(sources))))

        required_ids = {clip.source_asset_id for clip in draft.selected}
        missing = required_ids - set(local_paths)
        if missing:
            raise ValueError("export is missing selected source assets")

        local_overlays = []
        overlay_count = max(1, len(draft.media_overlays))
        for index, overlay in enumerate(draft.media_overlays):
            _bucket, _key, actual_kind = validate_overlay_uri(
                overlay.uri, user_id=user_id, project_id=project_id
            )
            if actual_kind != overlay.kind:
                raise ValueError("media overlay kind does not match its S3 object")
            suffix = Path(overlay.uri).suffix or (".jpg" if overlay.kind == "photo" else ".mp4")
            destination = str(Path(directory) / f"overlay-{index:03d}{suffix}")
            download_source(overlay.uri, destination)
            local_overlays.append(LocalMediaOverlay(overlay=overlay, path=destination))
            publish("rendering", min(32, 26 + int((index + 1) * 6 / overlay_count)))

        plan = build_render_plan(draft, local_paths)
        output = str(Path(directory) / "cutsell-export.mp4")
        publish("rendering", 35)
        render_preview(
            plan,
            output,
            text_overlays=draft.text_overlays,
            media_overlays=tuple(local_overlays),
        )
        publish("rendering", 85)
        stored = store_export(output, project_id=project_id, user_id=user_id)
        publish("finished", 100)
        return {
            "project_id": project_id,
            "state": "finished",
            "selected_count": len(draft.selected),
            "text_overlay_count": len(draft.text_overlays),
            "media_overlay_count": len(draft.media_overlays),
            **stored,
        }
