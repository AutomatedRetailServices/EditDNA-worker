# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

# 1) moviepy is OPTIONAL — don't let it crash imports
try:
    from moviepy.editor import VideoFileClip  # you can add more if you need
    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False

# 2) our own worker helpers
from worker import video    # your /workspace/EditDNA-worker/worker/video.py
from worker import asr      # your /workspace/EditDNA-worker/worker/asr.py
from worker import s3       # your /workspace/EditDNA-worker/worker/s3.py (you created it)

# -------------------------------------------------------------------
# small, explicit trash list — only the clearly broken ASR we saw
# we do NOT want to delete normal slang that makes ad-sense
# -------------------------------------------------------------------
HARD_TRASH_FRAGMENTS = [
    "kuchigai",   # mis-ASR from your sample
    "utas",       # mis-ASR from your sample
    "u-t-i-s yeast worry no more",  # if you want to nuke that exact bad line
]

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------

def _is_hard_trash(text: str) -> bool:
    """Return True only if text contains one of the explicit bad fragments."""
    t = text.lower()
    for bad in HARD_TRASH_FRAGMENTS:
        if bad in t:
            return True
    return False


def _slot_for_text(text: str) -> str:
    """
    Very small heuristic to fill slots.
    We keep it simple so standard English AND slang both go through.
    """
    t = text.lower()

    # hook-style openings
    if t.startswith("if you don't have") or t.startswith("is your ") or t.startswith("are you "):
        return "HOOK"

    # CTA-ish
    if "click the link" in t or "grab one" in t or "get yours" in t:
        return "CTA"

    # otherwise treat as FEATURE (most ad talk is feature/benefit)
    return "FEATURE"


def _build_clip(idx: int, seg: Dict[str, Any]) -> Dict[str, Any]:
    text = seg.get("text", "").strip()
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", max(start, 0.01)))
    clip_id = f"ASR{idx:04d}"

    return {
        "id": clip_id,
        "slot": "STORY",   # we also put them into slots[] below
        "start": start,
        "end": end,
        "score": 2.5,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
        "chain_ids": [clip_id],
        "text": text,
    }


def _organize_slots(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    for c in clips:
        text = c.get("text", "")
        slot_name = _slot_for_text(text)
        # copy a smaller view for slots
        slots[slot_name].append({
            "id": c["id"],
            "slot": slot_name,
            "start": c["start"],
            "end": c["end"],
            "score": c["score"],
            "face_q": c["face_q"],
            "scene_q": c["scene_q"],
            "vtx_sim": c["vtx_sim"],
            "chain_ids": c["chain_ids"],
            "text": c["text"],
        })

    return slots


def _maybe_render_and_upload(
    session_id: str,
    local_video_path: str,
    clips: List[Dict[str, Any]],
    s3_prefix: str,
) -> Tuple[str | None, str | None, str | None]:
    """
    If moviepy is available, render something and upload to S3.
    If not, just return (None, None, None).
    """
    if not _HAS_MOVIEPY:
        return None, None, None

    # super simple: just re-encode original for now
    out_name = f"{session_id}_{uuid.uuid4().hex}.mp4"
    out_local = f"/tmp/{out_name}"

    # this is intentionally simple — your real pipeline can cut by clips[] later
    clip = VideoFileClip(local_video_path)
    clip.write_videofile(out_local, audio_codec="aac")

    # upload
    s3_key = f"{s3_prefix}{out_name}"
    s3_url, https_url = s3.upload_file(out_local, s3_key)
    return s3_key, s3_url, https_url

# -------------------------------------------------------------------
# main entrypoint used by tasks.job_render(...)
# -------------------------------------------------------------------

def run_pipeline(
    session_id: str,
    local_video_path: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    """
    Main pipeline:
      1. ASR
      2. clean/keep slang
      3. build clips
      4. organize slots
      5. maybe render + upload
    """
    t0 = time.time()

    # 1) run ASR (your worker/asr.py must have transcribe_local)
    asr_result = asr.transcribe_local(local_video_path)
    segments = asr_result.get("segments", [])

    clips: List[Dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            # corrupt entry, skip
            continue

        raw_text = seg.get("text", "")
        if not raw_text:
            continue

        # hard-delete ONLY clearly broken fragments
        if _is_hard_trash(raw_text):
            # skip this one
            continue

        # otherwise keep it — even if it’s slangy / slightly messy
        clip = _build_clip(idx, seg)
        clips.append(clip)

    # 2) organize into marketing-ish slots
    slots = _organize_slots(clips)

    # 3) get duration from video helper
    duration_sec = video.probe_duration(local_video_path)

    # 4) try to render & upload (non-fatal)
    s3_key, s3_url, https_url = _maybe_render_and_upload(
        session_id=session_id,
        local_video_path=local_video_path,
        clips=clips,
        s3_prefix=s3_prefix,
    )

    out: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration_sec,
        "s3_key": s3_key,
        "s3_url": s3_url,
        "https_url": https_url,
        "clips": clips,
        "slots": slots,
        # flags like in your old output
        "asr": True,
        "semantic": True,
        "vision": False,
        "elapsed_sec": time.time() - t0,
    }
    return out
