# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Tuple, Optional

# this is the ASR we just fixed
import worker.asr as asr

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _make_clip_id(prefix: str, idx: int) -> str:
    return f"{prefix}{idx:04d}"


def _guess_slot(index: int, total: int) -> str:
    """
    VERY simple slotting:
    - first -> HOOK
    - last  -> CTA
    - everything else -> FEATURE
    You can swap this for your smarter funnel later.
    """
    if total == 1:
        return "HOOK"
    if index == 0:
        return "HOOK"
    if index == total - 1:
        return "CTA"
    return "FEATURE"


def _build_clip_dict(
    clip_id: str,
    slot: str,
    start: float,
    end: float,
    text: str,
    chain_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": clip_id,
        "slot": "STORY",  # keep original field
        "start": start,
        "end": end,
        "score": 2.5,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
        "chain_ids": chain_ids or [clip_id],
        "text": text,
    }


# ------------------------------------------------------------
# main pipeline
# ------------------------------------------------------------

def run_pipeline(
    local_video_path: str,
    session_id: str,
    s3_prefix: str,
) -> Dict[str, Any]:
    """
    Minimal, robust pipeline:
    1. run ASR
    2. normalize segments (skip bad ones)
    3. build clip list
    4. build slots dict
    5. return JSON similar to what your UI expects
    """
    t0 = time.time()

    # 1) ASR
    asr_result = asr.transcribe(local_video_path)
    raw_segments = asr_result.get("segments", [])

    segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        # IMPORTANT: this is what fixes your "string indices must be integers"
        if not isinstance(seg, dict):
            # ASR gave us something weird (like a plain string) -> skip
            continue
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        # Make sure end >= start
        if end < start:
            end = start
        segments.append(
            {
                "start": start,
                "end": end,
                "text": txt,
            }
        )

    # if ASR somehow returned nothing usable
    if not segments:
        return {
            "ok": False,
            "error": "no usable ASR segments",
            "session_id": session_id,
            "input_local": local_video_path,
        }

    # 2) build clips
    clips: List[Dict[str, Any]] = []
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    total = len(segments)
    for idx, seg in enumerate(segments):
        slot_name = _guess_slot(idx, total)
        clip_id = _make_clip_id("ASR", idx)
        clip = _build_clip_dict(
            clip_id=clip_id,
            slot=slot_name,
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            chain_ids=[clip_id],
        )
        clips.append(clip)

        # push into slots
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

    # 3) assemble response
    dt = time.time() - t0
    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        # we don't know exact duration here; your video module can fill it later
        "duration_sec": None,
        # we are not actually uploading in this minimal version
        "s3_key": None,
        "s3_url": None,
        "https_url": None,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
        "elapsed_sec": dt,
    }

    return result
