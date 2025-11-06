# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import worker.asr as asr  # <-- your ASR module


# ------------------------------------------------------------
# 1. tiny helpers
# ------------------------------------------------------------
def _make_clip_id(idx: int) -> str:
    return f"ASR{idx:04d}"


def _guess_slot(index: int, total: int) -> str:
    """
    keep it simple: first = HOOK, last = CTA, rest = FEATURE
    """
    if total == 1:
        return "HOOK"
    if index == 0:
        return "HOOK"
    if index == total - 1:
        return "CTA"
    return "FEATURE"


# hard blocklist: stuff that clearly means “bad take / restart”
_HARD_BAD_SUBSTRINGS = [
    "wait",                       # "wait, i'm gonna say that right..."
    "why can't i remember",       # explicit blooper
    "why cant i remember",
    "i'm gonna say that right",
    "im gonna say that right",
]

# if you want to also kill the weird word from the first video, keep it here
# uncomment if you want it hard-removed always:
# _HARD_BAD_SUBSTRINGS.append("kuchigai")
# _HARD_BAD_SUBSTRINGS.append("utas")


def _is_bad_line(text: str) -> bool:
    t = text.lower()
    for bad in _HARD_BAD_SUBSTRINGS:
        if bad in t:
            return True
    return False


# ------------------------------------------------------------
# 2. main entrypoint
# ------------------------------------------------------------
def run_pipeline(
    local_video_path: str,
    session_id: str,
    s3_prefix: str,
) -> Dict[str, Any]:
    """
    1) run ASR
    2) normalize segments
    3) filter out obvious bad/blooper lines
    4) build clips + slots
    """
    t0 = time.time()

    asr_result = asr.transcribe(local_video_path)
    raw_segments = asr_result.get("segments", [])

    cleaned_segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        # must be dict
        if not isinstance(seg, dict):
            continue

        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # drop the obvious “i messed up” lines
        if _is_bad_line(text):
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end < start:
            end = start

        cleaned_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    # if everything got filtered, fall back to original so we don't return empty
    if not cleaned_segments:
        cleaned_segments = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if end < start:
                end = start
            cleaned_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    # still nothing? then ASR was really empty
    if not cleaned_segments:
        return {
            "ok": False,
            "error": "no usable ASR segments (all filtered)",
            "session_id": session_id,
            "input_local": local_video_path,
        }

    # --------------------------------------------------------
    # build clips + slots
    # --------------------------------------------------------
    clips: List[Dict[str, Any]] = []
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    total = len(cleaned_segments)
    for idx, seg in enumerate(cleaned_segments):
        slot_name = _guess_slot(idx, total)
        clip_id = _make_clip_id(idx)

        clip_obj = {
            "id": clip_id,
            "slot": "STORY",  # keep your field
            "start": seg["start"],
            "end": seg["end"],
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [clip_id],
            "text": seg["text"],
        }
        clips.append(clip_obj)

        slots.setdefault(slot_name, [])
        slots[slot_name].append(
            {
                "id": clip_id,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "meta": {
                    "slot": slot_name,
                    "score": 2.0,
                    "chain_ids": [clip_id],
                },
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            }
        )

    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": None,
        "s3_key": None,
        "s3_url": None,
        "https_url": None,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
        "elapsed_sec": time.time() - t0,
    }
    return result
