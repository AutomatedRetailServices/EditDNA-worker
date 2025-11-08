# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
from typing import List, Dict, Any, Optional

from worker import asr, video, s3

# --------------------------
# Helper functions
# --------------------------

def _normalize_for_dupe(txt: str) -> str:
    """Normalize text to catch near-duplicate takes."""
    t = txt.strip().lower()
    for lead in ("so ", "and ", "um ", "uh ", "well "):
        if t.startswith(lead):
            t = t[len(lead):]
    return t.replace(",", " ").replace("  ", " ")


def _is_hard_bad(txt: str) -> bool:
    """Drop lines that are obviously bad takes or retakes."""
    bad_bits = [
        "wait",
        "why can't i remember",
        "am i saying that right",
        "what?",
        "is that good?",
    ]
    lt = txt.lower()
    return any(b in lt for b in bad_bits)


def _slots_from_clips(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize clips into slots (HOOK, FEATURE, CTA)."""
    slots = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    if not clips:
        return slots

    # first segment is HOOK
    first = clips[0]
    slots["HOOK"].append({
        "text": first["text"],
        "start": first["start"],
        "end": first["end"],
        "slot": "HOOK",
    })

    def looks_like_cta(t: str) -> bool:
        t = t.lower()
        return any(p in t for p in [
            "grab one",
            "click the link",
            "down below",
            "get yours",
            "check them out",
        ])

    for c in clips[1:]:
        slot = "CTA" if looks_like_cta(c["text"]) else "FEATURE"
        slots[slot].append({
            "text": c["text"],
            "start": c["start"],
            "end": c["end"],
            "slot": slot,
        })

    return slots


# --------------------------
# Main pipeline
# --------------------------

def run_pipeline(
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
    urls: Optional[List[str]] = None,
    local_video_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compatible with tasks.py that calls:
        pipeline.run_pipeline(local_video_path=..., session_id=..., s3_prefix=...)
    """
    print("[pipeline] using filtered pipeline v3 (local_video_path compatible)", flush=True)

    # determine which file to use
    if local_video_path and os.path.exists(local_video_path):
        src_path = local_video_path
    elif urls and urls[0]:
        src_path = video.download_to_local(urls[0])
    else:
        raise ValueError("No video path provided")

    # 1️⃣ Run ASR
    raw_segments = asr.transcribe_local(src_path)

    # 2️⃣ Clean and filter
    cleaned = []
    seen_norm = set()

    for seg in raw_segments:
        txt = seg.get("text", "").strip()
        if not txt or _is_hard_bad(txt):
            continue

        norm = _normalize_for_dupe(txt)
        if norm in seen_norm:
            continue
        seen_norm.add(norm)

        cleaned.append(seg)

    # 3️⃣ Build clips
    clips = [{
        "id": f"ASR{i:04d}",
        "slot": "STORY",
        "start": seg["start"],
        "end": seg["end"],
        "score": 2.5,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
        "chain_ids": [f"ASR{i:04d}"],
        "text": seg["text"],
    } for i, seg in enumerate(cleaned)]

    # 4️⃣ Build slots
    slots = _slots_from_clips(clips)

    # 5️⃣ Upload to S3
    out_key = f"{s3_prefix}{session_id}_{uuid.uuid4().hex}.mp4"
    https_url = s3.upload_file(src_path, out_key)

    # 6️⃣ Return result payload
    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src_path,
        "duration_sec": video.probe_duration(src_path),
        "s3_key": out_key,
        "s3_url": https_url,
        "https_url": https_url,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
    }
