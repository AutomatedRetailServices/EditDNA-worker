"""Deterministic replay of an already-approved source selection.

This is deliberately not an editorial authority. It never chooses takes, rewrites text,
or calls an LLM. It renders only caller-supplied source ranges, in order, using the
normal CutSell renderer so a human-approved draft can be re-used as a stable checkpoint
before additional downstream polish is evaluated.

A caller may also supply explicit human-review output cuts. Those cuts are treated as a
separate downstream review authority: they operate only on the already-rendered locked
Gold timeline and never reopen Best Take, semantic ordering, or source selection.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
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


def _normalize_review_cuts(rows: Iterable[dict[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if rows is None:
        return ()
    out: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, raw in enumerate(rows):
        start = float(raw.get("start"))
        end = float(raw.get("end"))
        reason = str(raw.get("reason") or "human_review_visual_reset").strip()
        if start < 0.0 or end <= start:
            raise ValueError(f"invalid review cut at index {index}")
        duration = end - start
        if duration > 1.50:
            raise ValueError("human review micro-cut exceeds 1.50 seconds")
        if start < previous_end - 1e-6:
            raise ValueError("review cuts must be ordered and non-overlapping")
        previous_end = end
        out.append({"start": start, "end": end, "reason": reason})
    return tuple(out)


def _probe_duration(path: str) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("review_cut_duration_probe_failed")
    return float(completed.stdout.strip())


def _apply_review_cuts(path: str, cuts: tuple[dict[str, Any], ...]) -> None:
    """Remove explicit reviewed output intervals without touching source selection."""
    if not cuts:
        return
    duration = _probe_duration(path)
    if cuts[-1]["end"] >= duration - 0.05:
        raise ValueError("review cut cannot remove the final output edge")

    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for cut in cuts:
        start = float(cut["start"])
        end = float(cut["end"])
        if end > duration + 1e-6:
            raise ValueError("review cut exceeds rendered duration")
        if start > cursor + 1e-6:
            keep.append((cursor, start))
        cursor = end
    if cursor < duration - 1e-6:
        keep.append((cursor, duration))
    if not keep:
        raise ValueError("review cuts removed the entire output")

    target = Path(path)
    with tempfile.TemporaryDirectory(prefix="cutsell-review-cuts-") as directory:
        work = Path(directory)
        parts: list[Path] = []
        for index, (start, end) in enumerate(keep):
            if end - start < 0.04:
                continue
            part = work / f"part-{index:03d}.mp4"
            command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(target),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                "-movflags", "+faststart", str(part),
            ]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if completed.returncode != 0 or not part.exists() or part.stat().st_size <= 0:
                raise RuntimeError("review_cut_segment_render_failed")
            parts.append(part)
        if not parts:
            raise RuntimeError("review_cut_no_output_parts")

        concat_file = work / "concat.txt"
        concat_file.write_text("".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8")
        corrected = work / "corrected.mp4"
        completed = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-c", "copy", "-movflags", "+faststart", str(corrected),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not corrected.exists() or corrected.stat().st_size <= 0:
            raise RuntimeError("review_cut_concat_failed")
        corrected.replace(target)


def run_locked_selection_replay(
    source_key: str,
    selection: Iterable[dict[str, Any]],
    *,
    project_id: str,
    preview_output: str,
    review_cuts: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render exactly the approved source ranges, then apply explicit reviewed micro-cuts."""
    key = str(source_key or "").strip()
    if not key:
        raise ValueError("source_key is required")
    rows = _normalize_selection(selection)
    normalized_review_cuts = _normalize_review_cuts(review_cuts)
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
        baseline_output_duration_sec = round(_probe_duration(preview_output), 3)
        _apply_review_cuts(preview_output, normalized_review_cuts)
        output_duration_sec = round(_probe_duration(preview_output), 3)

    return {
        "schema_version": "cutsell-approved-selection-replay-v2",
        "project_id": project_id,
        "source_key": key,
        "selection_authority": "caller_locked_approved_selection",
        "review_cut_authority": "caller_explicit_human_watch_listen" if normalized_review_cuts else None,
        "external_brain_calls_enabled": False,
        "selected_count": len(rows),
        "selected_duration_sec": round(sum(row["end"] - row["start"] for row in rows), 3),
        "baseline_output_duration_sec": baseline_output_duration_sec,
        "output_duration_sec": output_duration_sec,
        "review_cut_count": len(normalized_review_cuts),
        "review_cut_duration_sec": round(sum(row["end"] - row["start"] for row in normalized_review_cuts), 3),
        "review_cuts": list(normalized_review_cuts),
        "selected": list(rows),
    }
