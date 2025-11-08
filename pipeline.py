# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Tuple

from worker import video, s3, asr  # <-- your current worker package


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _gen_output_key(session_id: str) -> str:
    # this mimics what your jobs have been doing
    fname = f"{session_id}_{uuid.uuid4().hex}.mp4"
    prefix = os.getenv("S3_PREFIX", "editdna/outputs/")
    return os.path.join(prefix, fname)


def _normalize_text(t: str) -> str:
    return " ".join(t.strip().lower().split())


# phrases we really don’t want in the final script (IMG_03 junk etc.)
HARD_BAD_PHRASES = [
    "wait, am i saying that right",
    "wait not moisture control",
    "why can't i remember after that",
    "what?",
    "is that good?",
    "yeah.",
]

# filler that often repeats in CTA-ish products
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
    """
    For IMG_02: “if you wanna check them out …” appears 3x.
    We keep the first, drop later ones.
    """
    nt = _normalize_text(text)
    for prefix in DUPLICATE_FUZZY_PREFIXES:
        if nt.startswith(prefix):
            stub = prefix  # can be more fancy if needed
            if stub in seen_stubs:
                return True, stub
            else:
                seen_stubs.add(stub)
                return False, stub
    return False, ""


def _slots_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Very simple slotter:
      - first non-empty becomes HOOK
      - last CTA-like line becomes CTA
      - rest become FEATURE
    You already had that shape in your JSON, so we keep it.
    """
    slots = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if not segments:
        return slots

    # heuristic CTA detector
    CTA_WORDS = ["click", "grab", "get yours", "down below", "link"]

    def looks_like_cta(t: str) -> bool:
        t = t.lower()
        return any(w in t for w in CTA_WORDS)

    # HOOK = first
    first = segments[0].copy()
    first["slot"] = "HOOK"
    slots["HOOK"].append(first)

    # middle
    for seg in segments[1:-1]:
        t = seg["text"]
        if looks_like_cta(t):
            seg2 = seg.copy()
            seg2["slot"] = "CTA"
            slots["CTA"].append(seg2)
        else:
            seg2 = seg.copy()
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


# ---------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------
def run_pipeline(
    session_id: str,
    source_urls: List[str],
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    """
    What tasks.py calls.
    1) download video
    2) ASR it
    3) filter/clean segments
    4) upload final mp4 to S3 (you already have a render step in your real repo;
       here we just re-upload the input for demo, same as your earlier JSONs)
    """
    # 1) download the first video
    first_url = source_urls[0]
    local_video_path = video.download_to_local(first_url)

    # 2) probe duration (your video.py already has it)
    dur = video.probe_duration(local_video_path)

    # 3) run ASR (we renamed so pipeline calls transcribe_local, but your worker has both)
    raw_segments = asr.transcribe_local(local_video_path)

    # 4) clean up segments
    cleaned: List[Dict[str, Any]] = []
    seen_fuzzy = set()

    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # hard bad lines (IMG_03 “wait…”, “why can’t i remember…”)
        if _is_hard_bad(text):
            continue

        # fuzzy duplicate lines (IMG_02 “if you wanna check them out…”)
        is_dup, _ = _is_fuzzy_duplicate(text, seen_fuzzy)
        if is_dup:
            continue

        cleaned.append(seg)

    # if we cleaned too hard, fall back to raw
    if not cleaned:
        cleaned = raw_segments

    # 5) build slots
    slots = _slots_from_segments(cleaned)

    # 6) "render" – for now, like your earlier runs, just re-upload the same clip
    out_key = _gen_output_key(session_id)
    https_url = s3.upload_file(local_video_path, out_key)

    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": dur,
        "s3_key": out_key,
        "s3_url": https_url,
        "https_url": https_url,
        "clips": [
            {
                "id": f"ASR{i:04d}",
                "slot": "STORY",  # raw view
                "start": seg["start"],
                "end": seg["end"],
                "score": 2.5,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "chain_ids": [f"ASR{i:04d}"],
                "text": seg["text"],
            }
            for i, seg in enumerate(cleaned)
        ],
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
    }

    return result
