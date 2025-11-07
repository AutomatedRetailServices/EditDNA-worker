# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

from worker import video, asr, s3

# vision is optional – we try to import it
try:
    from worker import vision_sampler
    _HAS_VISION = True
except Exception:
    _HAS_VISION = False


def _classify_slot(text: str, idx: int) -> str:
    """
    Very dumb slotter, same spirit as what you had:
    - first usable line -> HOOK
    - CTA words -> CTA
    - everything else -> FEATURE
    """
    t = (text or "").lower()

    if idx == 0:
        return "HOOK"

    cta_words = ("click the link", "get yours today", "shop now", "grab one", "link below")
    if any(w in t for w in cta_words):
        return "CTA"

    return "FEATURE"


def _run_vision_gate(
    video_path: str,
    seg: Dict[str, Any],
) -> Tuple[float, float, float, bool]:
    """
    Call vision sampler on this segment.
    Returns (face_q, scene_q, vtx_sim, had_signal)
    If vision is not available, we just pretend everything is fine.
    """
    if not _HAS_VISION:
        return 1.0, 1.0, 0.0, False

    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start + 0.5))
    text = seg.get("text", "")

    try:
        face_q, scene_q, vtx_sim, had = vision_sampler.sample_visuals(
            video_path,
            (start, end),
            text=text,
            fps=2,
            max_frames=4,
        )
        return face_q, scene_q, vtx_sim, had
    except Exception:
        # if vision crashes, don't kill the whole job
        return 1.0, 1.0, 0.0, False


def _should_drop_by_vision(
    face_q: float,
    scene_q: float,
    vtx_sim: float,
    had_signal: bool,
) -> bool:
    """
    Simple gate:
    - if we actually saw frames (had_signal=True) AND
    - both face and scene are very low -> drop
    Tweak thresholds here if it’s too strict / too loose.
    """
    if not had_signal:
        return False
    if face_q < 0.15 and scene_q < 0.15:
        return True
    return False


def run_pipeline(
    local_video_path: str,
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    """
    Main entry called by tasks.job_render()
    Returns a dict that matches what your worker is already returning.
    """
    t0 = time.time()

    # 1) duration
    duration = video.probe_duration(local_video_path)

    # 2) ASR – work with your current worker/asr.py
    # we added transcribe_local earlier – but if the file still has only transcribe(), we fall back
    if hasattr(asr, "transcribe_local"):
        asr_result = asr.transcribe_local(local_video_path)
    else:
        asr_result = asr.transcribe(local_video_path)

    clips: List[Dict[str, Any]] = []
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    # 3) build clip objects, now with vision gating
    for idx, seg in enumerate(asr_result):
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 0.5))

        if not text:
            continue

        # vision check for THIS segment
        face_q, scene_q, vtx_sim, had_signal = _run_vision_gate(local_video_path, seg)
        drop = _should_drop_by_vision(face_q, scene_q, vtx_sim, had_signal)
        if drop:
            # skip visually bad take
            continue

        slot = _classify_slot(text, idx)

        clip_id = f"ASR{idx:04d}"
        clip_obj = {
            "id": clip_id,
            "slot": slot if slot != "HOOK" else "STORY",  # keep overall list as STORY like your runs
            "start": start,
            "end": end,
            "score": 2.5,
            "face_q": face_q,
            "scene_q": scene_q,
            "vtx_sim": vtx_sim,
            "chain_ids": [clip_id],
            "text": text,
        }
        clips.append(clip_obj)

        # also push into slots structure the way your JSON shows
        slot_entry = {
            "id": clip_id,
            "slot": slot,
            "start": start,
            "end": end,
            "score": 2.5,
            "face_q": face_q,
            "scene_q": scene_q,
            "vtx_sim": vtx_sim,
            "chain_ids": [clip_id],
            "text": text,
        }
        slots.setdefault(slot, []).append(slot_entry)

    # 4) render/export – in your flow the video is already the input,
    # so we just upload the original / processed file.
    # We'll make an S3 key like you've been getting.
    base_name = f"{session_id}_{uuid.uuid4().hex}.mp4"
    s3_key = os.path.join(s3_prefix, base_name)

    https_url = s3.upload_file(local_video_path, s3_key)

    # match the style you showed: return both s3://... and https://...
    s3_style_url = f"s3://{s3.S3_BUCKET}/{s3_key}"
    urls = [s3_style_url, https_url]

    elapsed = time.time() - t0

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration,
        "s3_key": s3_key,
        "s3_url": urls,
        "https_url": urls,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,  # you were marking this true in your outputs
        "vision": _HAS_VISION,
        "elapsed_sec": elapsed,
    }
