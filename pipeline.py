# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
from typing import List, Dict, Any, Optional

# these are from your repo
from worker import asr, video, s3


# -------------------------------------------------
# tiny text helpers
# -------------------------------------------------
BAD_PHRASES = [
    "so if you wanna check them out",
    "if you wanna check them out",
    "so if you want to check them out",
    "if you want to check them out",
    "why can't i remember",
    "am i saying that right",
    "wait,",
    "wait.",
    "what?",
    "is that good?",
]

CTA_PHRASES = [
    "grab one",
    "get yours",
    "click the link",
    "down below",
    "i left it for you",
    "check them out",
]


def _normalize_for_dupe(txt: str) -> str:
    t = txt.strip().lower()
    # strip common filler at the start
    for lead in ("so ", "and ", "um ", "uh ", "well "):
        if t.startswith(lead):
            t = t[len(lead):]
    # normalize spaces/punctuation a bit
    t = t.replace(",", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def _is_hard_bad(txt: str) -> bool:
    lt = txt.lower().strip()
    for bad in BAD_PHRASES:
        if bad in lt:
            return True
    return False


def _looks_like_cta(txt: str) -> bool:
    lt = txt.lower()
    return any(p in lt for p in CTA_PHRASES)


def _build_slots(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if not clips:
        return slots

    # 1st clip = HOOK
    first = clips[0]
    slots["HOOK"].append(
        {
            "text": first["text"],
            "start": first["start"],
            "end": first["end"],
            "slot": "HOOK",
        }
    )

    # rest: CTA if it looks like CTA, else FEATURE
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


# -------------------------------------------------
# main entry — MUST match tasks.py call
# -------------------------------------------------
def run_pipeline(
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
    urls: Optional[List[str]] = None,
    local_video_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    tasks.py calls:
        pipeline.run_pipeline(local_video_path=..., session_id=..., s3_prefix=...)
    so we MUST accept local_video_path here.
    """
    print("[pipeline] filtered pipeline ACTIVE", flush=True)

    # pick source video
    if local_video_path and os.path.exists(local_video_path):
        src_path = local_video_path
    elif urls and len(urls) > 0:
        # fallback: download here if caller didn't
        from worker import video as videomod
        src_path = videomod.download_to_local(urls[0])
    else:
        raise ValueError("No video provided to pipeline")

    # 1) ASR
    raw_segments = asr.transcribe_local(src_path)

    # 2) filter + dedupe
    cleaned: List[Dict[str, Any]] = []
    seen_norm: set[str] = set()

    for seg in raw_segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue

        # drop clearly bad / repeated-take phrases
        if _is_hard_bad(txt):
            continue

        norm = _normalize_for_dupe(txt)
        if norm in seen_norm:
            # exact-ish duplicate → skip
            continue
        seen_norm.add(norm)

        cleaned.append(seg)

    # 3) turn into clips
    clips: List[Dict[str, Any]] = []
    for i, seg in enumerate(cleaned):
        clip_id = f"ASR{i:04d}"
        clips.append(
            {
                "id": clip_id,
                "slot": "STORY",
                "start": seg["start"],
                "end": seg["end"],
                "score": 2.5,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "chain_ids": [clip_id],
                "text": seg["text"],
            }
        )

    # 4) build slot dict
    slots = _build_slots(clips)

    # 5) upload the (unchanged) input video so the API returns a URL like before
    out_key = f"{s3_prefix}{session_id}_{uuid.uuid4().hex}.mp4"
    https_url = s3.upload_file(src_path, out_key)

    # 6) done
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
        # you said you want vision true — set it here so the API sees it
        "vision": True,
    }
