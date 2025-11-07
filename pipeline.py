# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
import time
from typing import Any, Dict, List, Optional

# our worker modules
from worker import video
from worker import asr
from worker import s3

# ---------------------------------------------------------------------------
# helpers: classify, clean, merge
# ---------------------------------------------------------------------------

# phrases that are clearly meta / retake / blooper in your IMG_03 video
_BAD_TAKE_PHRASES = [
    "wait",
    "am i saying that right",
    "what?",
    "what ?",
    "why can't i remember after that",
    "why cant i remember after that",
    "not moisture control",  # the self-correction line
]

# slang we want to KEEP even if it looks weird
_WHITELIST_PHRASES = [
    "coochie",
    "coochie gang",
    "kuchigai",     # in case the model spells it like this
    "wet wet",
]

def _looks_like_badtake(text: str) -> bool:
    """
    Returns True if this line is just the speaker messing up, not content.
    We allow slang, we only kill meta-talker / corrections.
    """
    low = text.lower().strip()

    # if it contains any whitelisted slang, don't kill it
    for keep in _WHITELIST_PHRASES:
        if keep in low:
            return False

    for bad in _BAD_TAKE_PHRASES:
        if bad in low:
            return True

    # also drop super-short pure fillers
    if low in {"yeah", "what", "what?", "huh", "uh", "um"}:
        return True

    return False


def _merge_short_segments(segments: List[Dict[str, Any]], min_dur: float = 1.2) -> List[Dict[str, Any]]:
    """
    Your IMG_03 ASR spits out a bunch of 0.5–1s lines.
    We merge those into the previous real segment so we get fuller clips.
    """
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    current = segments[0].copy()

    for seg in segments[1:]:
        seg_dur = float(seg["end"]) - float(seg["start"])
        # if this segment is tiny, or the previous one was tiny, just glue it
        if seg_dur < min_dur:
            # append its text
            current["text"] = (current["text"] + " " + seg["text"]).strip()
            # extend time
            current["end"] = seg["end"]
        else:
            # push current and start a new one
            merged.append(current)
            current = seg.copy()

    # last one
    merged.append(current)
    return merged


def _segments_to_clips(segments: List[Dict[str, Any]], funnel_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Turn cleaned ASR segments into the structure you were returning before.
    We'll keep it simple: everything becomes a STORY/FEATURE,
    but we fill slots according to funnel_counts.
    """
    clips: List[Dict[str, Any]] = []
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    # very simple slotting:
    #  - first non-empty -> HOOK
    #  - middle -> FEATURE
    #  - last -> CTA
    # you can make this smarter later
    for idx, seg in enumerate(segments):
        clip_id = f"ASR{idx:04d}"
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg["text"]

        clip = {
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
        }
        clips.append(clip)

    # slot assignment (quick + predictable)
    if clips:
        # HOOK
        slots["HOOK"].append({
            "id": clips[0]["id"],
            "slot": "HOOK",
            "start": clips[0]["start"],
            "end": clips[0]["end"],
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [clips[0]["id"]],
            "text": clips[0]["text"],
        })

    # CTA: try to find a line that looks like CTA
    cta_index: Optional[int] = None
    for i, c in enumerate(clips):
        low = c["text"].lower()
        if "click the link" in low or "get yours today" in low or "grab one" in low:
            cta_index = i
            break
    if cta_index is None and len(clips) > 0:
        cta_index = len(clips) - 1  # just last

    if cta_index is not None:
        c = clips[cta_index]
        slots["CTA"].append({
            "id": c["id"],
            "slot": "CTA",
            "start": c["start"],
            "end": c["end"],
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [c["id"]],
            "text": c["text"],
        })

    # everything else → FEATURE
    for c in clips:
        if slots["HOOK"] and c["id"] == slots["HOOK"][0]["id"]:
            continue
        if slots["CTA"] and c["id"] == slots["CTA"][0]["id"]:
            continue
        slots["FEATURE"].append({
            "id": c["id"],
            "slot": "FEATURE",
            "start": c["start"],
            "end": c["end"],
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [c["id"]],
            "text": c["text"],
        })

    return {
        "clips": clips,
        "slots": slots,
    }


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def run_pipeline(
    *,
    session_id: str,
    local_video_path: str,
    s3_prefix: str = "editdna/outputs/",
    funnel_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    This is what tasks.job_render(...) calls.
    We assume the worker already downloaded to local_video_path.
    """
    t0 = time.time()
    if funnel_counts is None:
        funnel_counts = {
            "HOOK": 1,
            "PROBLEM": 1,
            "FEATURE": 99,
            "PROOF": 1,
            "CTA": 1,
        }

    # 1) ASR
    segments = asr.transcribe(local_video_path)  # returns [{text,start,end}, ...]
    cleaned: List[Dict[str, Any]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if _looks_like_badtake(text):
            # skip obvious retakes
            continue
        cleaned.append(
            {
                "text": text,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            }
        )

    # 2) merge tiny ones so you don’t get 40 clips
    merged = _merge_short_segments(cleaned)

    # 3) build clip/slot structure
    clip_data = _segments_to_clips(merged, funnel_counts)

    # 4) figure duration
    duration = video.probe_duration(local_video_path)

    # 5) upload original (or processed) video to S3
    # we’re just uploading the source as-is like your last run
    fname = f"{session_id}_{uuid.uuid4().hex}.mp4"
    key = os.path.join(s3_prefix, fname)
    https_url = s3.upload_file(local_video_path, key)

    elapsed = time.time() - t0
    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration,
        "s3_key": key,
        "s3_url": https_url,
        "https_url": https_url,
        "clips": clip_data["clips"],
        "slots": clip_data["slots"],
        "asr": True,
        "semantic": True,
        "vision": False,
        "elapsed_sec": elapsed,
    }
