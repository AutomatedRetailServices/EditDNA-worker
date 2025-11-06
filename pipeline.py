# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import json
import time
import uuid
from typing import Any, Dict, List

# these are your own modules under worker/
# we know you have worker.asr because the last run used it
from worker import asr  # type: ignore

# optional: if you have a small s3 util somewhere
try:
    from worker import s3_utils  # type: ignore
    _HAS_S3 = True
except Exception:
    _HAS_S3 = False

# optional: we try moviepy for real duration
try:
    from moviepy.editor import VideoFileClip  # type: ignore
    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False


def _safe_video_duration(path: str, asr_segments: List[Dict[str, Any]]) -> float:
    """
    Try to get the real video duration.
    If moviepy is missing or video can't be read,
    fall back to the max end time from ASR.
    """
    # 1) try moviepy
    if _HAS_MOVIEPY and os.path.exists(path):
        try:
            with VideoFileClip(path) as clip:
                return float(clip.duration)
        except Exception:
            pass

    # 2) fallback: max ASR end
    max_end = 0.0
    for seg in asr_segments:
        try:
            end = float(seg.get("end", 0.0))
            if end > max_end:
                max_end = end
        except Exception:
            continue
    return max_end


def _normalize_asr_segments(raw: Any) -> List[Dict[str, Any]]:
    """
    Your last error was: "string indices must be integers"
    → that happens when a segment is a plain string.

    This normalizes EVERYTHING into:
    { "text": str, "start": float, "end": float }
    so the rest of the pipeline is safe.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out

    idx = 0
    for item in raw:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
        else:
            # item is string or something else
            text = str(item).strip()
            start = 0.0
            end = 0.0

        # skip totally empty
        if not text:
            idx += 1
            continue

        clip_id = f"ASR{idx:04d}"
        out.append({
            "id": clip_id,
            "slot": "STORY",     # we can re-slot later
            "start": start,
            "end": end,
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [clip_id],
            "text": text,
        })
        idx += 1
    return out


def _auto_slots(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Very simple slotting: first → HOOK, last → CTA, middle → FEATURE.
    This is exactly what your earlier JSON looked like.
    """
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    if not clips:
        return slots

    # HOOK = first
    first = clips[0].copy()
    first["slot"] = "HOOK"
    slots["HOOK"].append(first)

    # CTA = last
    if len(clips) > 1:
        last = clips[-1].copy()
        last["slot"] = "CTA"
        slots["CTA"].append(last)

    # rest = FEATURE
    if len(clips) > 2:
        for c in clips[1:-1]:
            fc = c.copy()
            fc["slot"] = "FEATURE"
            slots["FEATURE"].append(fc)

    return slots


def _maybe_upload_s3(local_path: str, s3_prefix: str, session_id: str) -> Dict[str, str]:
    """
    Try to upload to S3 if we have s3_utils. Otherwise return empty.
    """
    if not _HAS_S3:
        return {"s3_key": None, "s3_url": None, "https_url": None}

    # build key: editdna/outputs/<session_id>_<uuid>.mp4
    base_name = f"{session_id}_{uuid.uuid4().hex}.mp4"
    key = os.path.join(s3_prefix, base_name)

    try:
        url = s3_utils.upload_file(local_path, key)  # your util may differ
        # if your util returns https already:
        return {
            "s3_key": key,
            "s3_url": url,
            "https_url": url,
        }
    except Exception:
        return {"s3_key": None, "s3_url": None, "https_url": None}


def run_pipeline(
    *,
    local_video_path: str,
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    """
    MAIN ENTRY called by tasks.job_render(...)
    """
    t0 = time.time()

    # 1) run ASR
    # your worker log said: module 'worker.asr' has no attribute 'transcribe'
    # but after you added it, it worked. So we call exactly that.
    asr_raw = asr.transcribe(local_video_path)

    # we expect asr_raw to have either ["segments"] or be already a list
    if isinstance(asr_raw, dict) and "segments" in asr_raw:
        segments = asr_raw["segments"]
    else:
        segments = asr_raw

    # 2) normalize to safe clips
    clips = _normalize_asr_segments(segments)

    # 3) auto slots
    slots = _auto_slots(clips)

    # 4) get duration
    duration_sec = _safe_video_duration(local_video_path, clips)

    # 5) upload (optional)
    s3_info = _maybe_upload_s3(local_video_path, s3_prefix, session_id)

    elapsed = time.time() - t0

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration_sec if duration_sec else None,
        "s3_key": s3_info["s3_key"],
        "s3_url": s3_info["s3_url"],
        "https_url": s3_info["https_url"],
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,  # you can flip to False if you don’t want to signal it yet
        "vision": False,
        "elapsed_sec": elapsed,
    }
