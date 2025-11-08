# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from worker import video, s3, asr


# ------------------------------------------------------------
# small helpers
# ------------------------------------------------------------
def _gen_output_key(session_id: str) -> str:
    fname = f"{session_id}_{uuid.uuid4().hex}.mp4"
    prefix = os.getenv("S3_PREFIX", "editdna/outputs/")
    return os.path.join(prefix, fname)


def _normalize_text(t: str) -> str:
    return " ".join(t.strip().lower().split())


# stuff we want to drop in the sloppy/slang takes
HARD_BAD_PHRASES = [
    "wait, am i saying that right",
    "wait not moisture control",
    "why can't i remember after that",
    "what?",
    "is that good?",
    "yeah.",
]

# repetitive CTA-ish lines we saw in img_02
DUPLICATE_FUZZY_PREFIXES = [
    "if you wanna check them out",
    "so if you wanna check them out",
    "if you want to check them out",
    "so if you want to check them out",
    "and grab one of these",
]


def _is_hard_bad(text: str) -> bool:
    nt = _normalize_text(text)
    for bad in HARD_BAD_PHRASES:
        if bad in nt:
            return True
    return False


def _is_fuzzy_duplicate(text: str, seen_stubs: set) -> Tuple[bool, str]:
    nt = _normalize_text(text)
    for prefix in DUPLICATE_FUZZY_PREFIXES:
        if nt.startswith(prefix):
            if prefix in seen_stubs:
                return True, prefix
            seen_stubs.add(prefix)
            return False, prefix
    return False, ""


def _slots_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slots = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    if not segments:
        return slots

    CTA_WORDS = ["click", "grab", "get yours", "down below", "link"]

    def looks_like_cta(t: str) -> bool:
        t = t.lower()
        return any(w in t for w in CTA_WORDS)

    # first segment → HOOK
    first = segments[0].copy()
    first["slot"] = "HOOK"
    slots["HOOK"].append(first)

    # middle
    for seg in segments[1:-1]:
        seg2 = seg.copy()
        if looks_like_cta(seg["text"]):
            seg2["slot"] = "CTA"
            slots["CTA"].append(seg2)
        else:
            seg2["slot"] = "FEATURE"
            slots["FEATURE"].append(seg2)

    # last
    last = segments[-1].copy()
    if looks_like_cta(last["text"]):
        last["slot"] = "CTA"
        slots["CTA"].append(last)
    else:
        last["slot"] = "FEATURE"
        slots["FEATURE"].append(last)

    return slots


# ------------------------------------------------------------
# main pipeline
# ------------------------------------------------------------
def run_pipeline(
    session_id: str,
    urls: Optional[List[str]] = None,
    s3_prefix: str = "editdna/outputs/",
    local_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Your tasks.py is calling with local_video_path=..., so we accept it here.
    """
    # 1) pick the video path
    if local_video_path and os.path.exists(local_video_path):
        vid_path = local_video_path
    else:
        if not urls:
            raise ValueError("run_pipeline: no local_video_path and no urls provided")
        first_url = urls[0]
        vid_path = video.download_to_local(first_url)

    # 2) probe duration
    duration = video.probe_duration(vid_path)

    # 3) run ASR
    raw_segments = asr.transcribe_local(vid_path)

    # 4) clean segments
    cleaned: List[Dict[str, Any]] = []
    seen_fuzzy = set()

    for seg in raw_segments:
        txt = seg.get("text", "").strip()
        if not txt:
            continue

        # kill obvious bad-take lines
        if _is_hard_bad(txt):
            continue

        # kill duplicate CTA-ish lines
        is_dup, _ = _is_fuzzy_duplicate(txt, seen_fuzzy)
        if is_dup:
            continue

        cleaned.append(seg)

    # if we accidentally deleted everything, just use original ASR
    if not cleaned:
        cleaned = raw_segments

    # 5) build slots
    slots = _slots_from_segments(cleaned)

    # 6) upload original video (like your current pipeline does)
    out_key = _gen_output_key(session_id)
    https_url = s3.upload_file(vid_path, out_key)

    # 7) convert cleaned segments to clips
    clips = []
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

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": vid_path,
        "duration_sec": duration,
        "s3_key": out_key,
        "s3_url": https_url,
        "https_url": https_url,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
    }
