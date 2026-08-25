"""Deterministic replay of an already-approved source selection.

This is deliberately not an editorial authority. It never chooses takes, rewrites text,
or calls an LLM. It renders only caller-supplied source ranges, in order, using the
normal CutSell renderer so a human-approved draft can be re-used as a stable checkpoint
before additional downstream polish is evaluated.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable

from .config import load_runtime_config
from .render import render_preview
from .render_plan import RenderSegment
from .storage import download_source


def _normalize_selection(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, raw in enumerate(rows):
        start = float(raw.get("start"))
        end = float(raw.get("end"))
        if start < 0.0 or end <= start:
            raise ValueError(f"invalid locked selection range at index {index}")
        if start < previous_end - 1e-6:
            raise ValueError("locked selection must be source-ordered and non-overlapping")
        previous_end = end
        out.append({
            "start": start,
            "end": end,
            "text": str(raw.get("text") or ""),
            "take_group_id": str(raw.get("take_group_id") or ""),
        })
    if not out:
        raise ValueError("locked selection cannot be empty")
    return tuple(out)


def run_locked_selection_replay(
    source_key: str,
    selection: Iterable[dict[str, Any]],
    *,
    project_id: str,
    preview_output: str,
) -> dict[str, Any]:
    """Render exactly the approved source ranges without re-running selection logic."""
    key = str(source_key or "").strip()
    if not key:
        raise ValueError("source_key is required")
    rows = _normalize_selection(selection)
    config = load_runtime_config()
    if not config.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    with tempfile.TemporaryDirectory(prefix="cutsell-locked-selection-") as directory:
        destination = str(Path(directory) / PurePosixPath(key).name)
        local = download_source(f"s3://{config.s3_bucket}/{key}", destination)
        segments = tuple(
            RenderSegment(
                clip_id=f"locked-{index:04d}",
                source_asset_id="locked-source-0",
                source_path=local,
                start=row["start"],
                end=row["end"],
                caption_text="",
            )
            for index, row in enumerate(rows)
        )
        render_preview(segments, preview_output)

    return {
        "schema_version": "cutsell-approved-selection-replay-v1",
        "project_id": project_id,
        "source_key": key,
        "selection_authority": "caller_locked_approved_selection",
        "external_brain_calls_enabled": False,
        "selected_count": len(rows),
        "selected_duration_sec": round(sum(row["end"] - row["start"] for row in rows), 3),
        "selected": list(rows),
    }
