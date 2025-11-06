# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List

from worker import asr  # your working ASR

# moviepy only for duration
try:
    from moviepy.editor import VideoFileClip
    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False

# boto3 for upload
try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    _HAS_BOTO3 = False


def _safe_video_duration(path: str, asr_segments: List[Dict[str, Any]]) -> float:
    if _HAS_MOVIEPY and os.path.exists(path):
        try:
            with VideoFileClip(path) as clip:
                return float(clip.duration)
        except Exception:
            pass
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
            text = str(item).strip()
            start = 0.0
            end = 0.0
        if not text:
            idx += 1
            continue
        clip_id = f"ASR{idx:04d}"
        out.append({
            "id": clip_id,
            "slot": "STORY",
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
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    if not clips:
        return slots
    first = clips[0].copy()
    first["slot"] = "HOOK"
    slots["HOOK"].append(first)
    if len(clips) > 1:
        last = clips[-1].copy()
        last["slot"] = "CTA"
        slots["CTA"].append(last)
    if len(clips) > 2:
        for c in clips[1:-1]:
            fc = c.copy()
            fc["slot"] = "FEATURE"
            slots["FEATURE"].append(fc)
    return slots


def _maybe_upload_s3(local_path: str, s3_prefix: str, session_id: str) -> Dict[str, str | None]:
    """
    Real S3 upload with boto3.
    Needs env:
      S3_BUCKET
      AWS_ACCESS_KEY_ID
      AWS_SECRET_ACCESS_KEY
      AWS_DEFAULT_REGION
    """
    if not _HAS_BOTO3:
        return {"s3_key": None, "s3_url": None, "https_url": None}

    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        # no bucket → can't upload
        return {"s3_key": None, "s3_url": None, "https_url": None}

    key = os.path.join(s3_prefix, f"{session_id}_{uuid.uuid4().hex}.mp4")
    s3 = boto3.client("s3")

    try:
        s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
        # build https url (standard S3 public-style; adjust if private)
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        https_url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        return {
            "s3_key": key,
            "s3_url": https_url,
            "https_url": https_url,
        }
    except Exception:
        return {"s3_key": None, "s3_url": None, "https_url": None}


def run_pipeline(
    *,
    local_video_path: str,
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    t0 = time.time()

    asr_raw = asr.transcribe(local_video_path)
    if isinstance(asr_raw, dict) and "segments" in asr_raw:
        segments = asr_raw["segments"]
    else:
        segments = asr_raw

    clips = _normalize_asr_segments(segments)
    slots = _auto_slots(clips)
    duration_sec = _safe_video_duration(local_video_path, clips)
    s3_info = _maybe_upload_s3(local_video_path, s3_prefix, session_id)

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
        "semantic": True,
        "vision": False,
        "elapsed_sec": time.time() - t0,
    }
