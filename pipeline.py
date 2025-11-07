# /workspace/EditDNA-worker/pipeline.py

from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List

from worker import video as videoutils
from worker import asr
from worker import s3 as s3mod

# phrases we saw that were clearly ASR trash when they appear ALONE
_HARD_BAD = [
    "kuchigai",
    "utas",
    "utas yeast",
    "odor utas",
]


def _should_drop(seg_text: str) -> bool:
    """
    Rule:
    - if the whole segment is basically JUST the bad fragment → drop
    - if the bad word is inside a real sentence → KEEP (because that's how they talk)
    """
    t = seg_text.strip().lower()
    # very short + contains bad → drop
    if len(t.split()) <= 3:
        for bad in _HARD_BAD:
            if bad in t:
                return True
    # exact match → drop
    for bad in _HARD_BAD:
        if t == bad:
            return True
    return False


def _slot_for_text(seg_text: str) -> str:
    """
    super simple slotting like you had: first becomes HOOK, last becomes CTA, rest FEATURE
    """
    # we’ll assign outside; keep here if you later want smarter logic
    return "FEATURE"


def run_pipeline(
    *,
    session_id: str,
    local_video_path: str,
    s3_prefix: str = "editdna/outputs/",
    funnel_counts: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """
    Main pipeline: ASR → filter → build clips → upload original → return JSON
    """
    # 1) get duration
    duration_sec = videoutils.probe_duration(local_video_path)

    # 2) ASR
    segments = asr.transcribe_local(local_video_path)

    # 3) build clips
    clips: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        txt = seg["text"].strip()
        if not txt:
            continue
        if _should_drop(txt):
            # skip the clearly busted tiny junk
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 2.0))

        clips.append(
            {
                "id": f"ASR{idx:04d}",
                "slot": "STORY",  # we reshape below
                "start": start,
                "end": end,
                "score": 2.5,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "chain_ids": [f"ASR{idx:04d}"],
                "text": txt,
            }
        )

    # 4) assign slots roughly like your old output
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if clips:
        # first = HOOK
        first = clips[0].copy()
        first["slot"] = "HOOK"
        slots["HOOK"].append(first)

        # last = CTA
        if len(clips) > 1:
            last = clips[-1].copy()
            last["slot"] = "CTA"
            slots["CTA"].append(last)

        # middle = FEATURE
        middle = clips[1:-1] if len(clips) > 2 else []
        for m in middle:
            mm = m.copy()
            mm["slot"] = "FEATURE"
            slots["FEATURE"].append(mm)

    # 5) upload original video to S3 so your web has a URL
    # make sure bucket + creds exist in env
    key = f"{s3_prefix}{session_id}_{uuid.uuid4().hex}.mp4"
    https_url = s3mod.upload_file(local_video_path, key)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration_sec,
        "s3_key": key,
        "s3_url": https_url,
        "https_url": https_url,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
    }
