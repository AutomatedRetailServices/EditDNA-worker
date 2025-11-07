# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
import time
from typing import Any, Dict, List, Tuple

from worker import asr
from worker import video as videoutil
from worker import s3 as s3util

# how many clips per slot we try to surface
DEFAULT_FUNNEL_COUNTS = {
    "HOOK": 1,
    "PROBLEM": 1,
    "FEATURE": 12,
    "PROOF": 1,
    "CTA": 1,
}

# IMG_03 style junk / self-corrections / obvious ASR garbage
_RETAPEY_PHRASES = [
    "wait, am i saying that right",
    "wait am i saying that right",
    "why can't i remember",
    "why cant i remember",
    "what?",
    "what ?",
    "is that good? yeah.",
    "is that good ? yeah.",
    "am i saying that right",
    "wait, not moisture control",
]

# really bad ASR tokens
_GARBAGE_TOKENS = [
    "kuchigai",
    "utas",
]

# IMG_02 specific tail-babble we saw (“so if you wanna check them out …”)
_TRAILING_CTA_BLAB = [
    "if you wanna check them out",
    "so if you wanna check them out",
    "grab one of these for yourself",
    "i left it for you down below",
    "grab one of these westland",
]


def _get_funnel_counts_from_env() -> Dict[str, int]:
    raw = os.getenv("FUNNEL_COUNTS")
    counts = dict(DEFAULT_FUNNEL_COUNTS)
    if not raw:
        return counts
    try:
        import json
        data = json.loads(raw)
        for k, v in data.items():
            try:
                counts[k] = int(v)
            except Exception:
                pass
    except Exception:
        pass
    return counts


def _is_retake_or_flub(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return True

    # obvious retakes (IMG_03)
    for bad in _RETAPEY_PHRASES:
        if bad in t:
            return True

    # obvious garbage tokens
    for bad in _GARBAGE_TOKENS:
        if bad in t:
            return True

    # tiny question fragments like "what?"
    if len(t.split()) <= 2 and t.endswith("?"):
        return True

    return False


def _is_trailing_babble(text: str) -> bool:
    """
    This is for IMG_02 type endings where the speaker re-says
    the CTA 2–3 times. We just drop those lines.
    """
    t = text.lower().strip()
    for pat in _TRAILING_CTA_BLAB:
        if pat in t:
            return True
    # also if it starts with "so if you wanna" we kill it
    if t.startswith("so if you wanna"):
        return True
    return False


def _slot_for_text(text: str) -> str:
    lt = text.lower()
    if "click the link" in lt or "get yours today" in lt or "get yours" in lt:
        return "CTA"
    if "if you wanna check them out" in lt:
        return "CTA"
    if "does your" in lt or "is your" in lt or "if you don't have" in lt:
        return "HOOK"
    return "FEATURE"


def _build_clip(seg_id: str, slot: str, start: float, end: float, text: str) -> Dict[str, Any]:
    return {
        "id": seg_id,
        "slot": slot,
        "start": start,
        "end": end,
        "score": 2.5,
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
        "chain_ids": [seg_id],
        "text": text,
    }


def _upload_rendered(local_path: str, s3_prefix: str, session_id: str) -> Tuple[str, str]:
    base = f"{session_id}_{uuid.uuid4().hex}.mp4"
    key = os.path.join(s3_prefix, base)
    url = s3util.upload_file(local_path, key)
    return key, url


def run_pipeline(
    *,
    local_video_path: str,
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    t0 = time.time()
    funnel_counts = _get_funnel_counts_from_env()

    # 1) duration
    duration = videoutil.probe_duration(local_video_path)

    # 2) ASR
    asr_segments = asr.transcribe_local(local_video_path)
    # asr_segments = [{"text": "...", "start": 0.0, "end": 2.0}, ...]

    clips: List[Dict[str, Any]] = []
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    for idx, seg in enumerate(asr_segments):
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or (start + 2.0))

        # 1) kill obvious bad / retake
        if _is_retake_or_flub(text):
            continue

        # 2) kill IMG_02 tail babble
        if _is_trailing_babble(text):
            continue

        slot = _slot_for_text(text)
        seg_id = f"ASR{idx:04d}"

        clip = _build_clip(seg_id, slot, start, end, text)
        clips.append(clip)

        bucket = slots.get(slot)
        if bucket is not None:
            if len(bucket) < funnel_counts.get(slot, 999):
                bucket.append(clip)

    # 4) upload "render" (right now, original)
    try:
        s3_key, s3_url = _upload_rendered(local_video_path, s3_prefix, session_id)
    except Exception:
        s3_key, s3_url = None, None

    elapsed = time.time() - t0

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration,
        "s3_key": s3_key,
        "s3_url": s3_url,
        "https_url": s3_url,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
        "elapsed_sec": elapsed,
    }
