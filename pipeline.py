# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
from typing import List, Dict, Any, Optional

from worker import asr, video, s3

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def _normalize_for_dupe(txt: str) -> str:
    t = txt.strip().lower()
    for lead in ("so ", "and ", "um ", "uh ", "well "):
        if t.startswith(lead):
            t = t[len(lead):]
    t = t.replace(",", " ").replace("  ", " ")
    return t


def _is_hard_bad(txt: str) -> bool:
    bad_bits = [
        "wait",                      # "wait, am I saying that right?"
        "why can't i remember",      # IMG_03
        "am i saying that right",    # IMG_03
        "what?",                     # IMG_03 mid take
        "is that good?",             # IMG_03 check
    ]
    lt = txt.lower()
    for b in bad_bits:
        if b in lt:
            return True
    return False


def _slots_from_clips(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    if not clips:
        return slots

    first = clips[0]
    slots["HOOK"].append(
        {
            "text": first["text"],
            "start": first["start"],
            "end": first["end"],
            "slot": "HOOK",
        }
    )

    def _looks_like_cta(t: str) -> bool:
        t = t.lower()
        return (
            "grab one" in t
            or "click the link" in t
            or "down below" in t
            or "get yours" in t
            or "check them out" in t
        )

    for c in clips[1:]:
        if _looks_like_cta(c["text"]):
            slots["CTA"].append(
                {
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                    "slot": "CTA",
                }
            )
        else:
            slots["FEATURE"].append(
                {
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                    "slot": "FEATURE",
                }
            )

    return slots


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def run_pipeline(
    session_id: str,
    urls: List[str],
    s3_prefix: str = "editdna/outputs/",
    local_video_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    tasks.py calls us with local_video_path=... so we accept it.
    If it's not provided, we download from the first URL.
    """
    print("[pipeline] using filtered pipeline v2 (accepts local_video_path)", flush=True)

    if local_video_path and os.path.exists(local_video_path):
        src_path = local_video_path
    else:
        if not urls:
            raise ValueError("no urls provided")
        src_path = video.download_to_local(urls[0])

    # 1) ASR
    raw_segments = asr.transcribe_local(src_path)

    # 2) filter
    cleaned: List[Dict[str, Any]] = []
    seen_norm = set()

    for seg in raw_segments:
        txt = seg.get("text", "").strip()
        if not txt:
            continue

        if _is_hard_bad(txt):
            continue

        norm = _normalize_for_dupe(txt)
        if norm in seen_norm:
            # drop duplicate wording / micro-retakes
            continue
        seen_norm.add(norm)

        cleaned.append(seg)

    # 3) build clips
    clips: List[Dict[str, Any]] = []
    for i, seg in enumerate(cleaned):
        clips.append(
            {
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
            }
        )

    # 4) slots
    slots = _slots_from_clips(clips)

    # 5) upload original video to S3 (same pattern as before)
    out_key = f"{s3_prefix}{session_id}_{uuid.uuid4().hex}.mp4"
    https_url = s3.upload_file(src_path, out_key)

    result: Dict[str, Any] = {
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
    return result
