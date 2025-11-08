# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import uuid
from typing import List, Dict, Any, Tuple

from worker import asr, video, s3

# ---------------------------------------------------------
# small helpers
# ---------------------------------------------------------

def _normalize_for_dupe(txt: str) -> str:
    """
    Make a text comparable so we can spot near-duplicates like:
    "if you wanna check them out" vs "so if you wanna check them out"
    """
    t = txt.strip().lower()

    # drop leading fillers that show up in retakes
    for lead in ("so ", "and ", "um ", "uh ", "well "):
        if t.startswith(lead):
            t = t[len(lead):]

    # kill commas and extra spaces
    t = t.replace(",", " ").replace("  ", " ")
    return t


def _is_hard_bad(txt: str) -> bool:
    """
    Lines we NEVER want — obvious re-takes / meta / mistakes.
    You can add to this list as you see more.
    """
    bad_bits = [
        "wait",                      # "wait, am i saying that right?"
        "why can't i remember",      # IMG_03
        "am i saying that right",    # IMG_03
        "what?",                     # IMG_03 mid-take
        "is that good?",             # IMG_03 mid-check
    ]
    lt = txt.lower()
    for b in bad_bits:
        if b in lt:
            return True
    return False


def _slots_from_clips(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the same slot structure your current output has.
    We'll keep it simple: first line -> HOOK, last CTA-ish -> CTA, rest -> FEATURE
    """
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if not clips:
        return slots

    # hook = first line
    first = clips[0]
    slots["HOOK"].append({
        "text": first["text"],
        "start": first["start"],
        "end": first["end"],
        "slot": "HOOK",
    })

    # CTA heuristic: if a line mentions "grab one", "click", "down below" → CTA
    def _looks_like_cta(t: str) -> bool:
        t = t.lower()
        return (
            "grab one" in t
            or "click the link" in t
            or "down below" in t
            or "get yours" in t
            or "check them out" in t
        )

    # everything else → FEATURE, but if CTA-ish → CTA
    for c in clips[1:]:
        if _looks_like_cta(c["text"]):
            slots["CTA"].append({
                "text": c["text"],
                "start": c["start"],
                "end": c["end"],
                "slot": "CTA",
            })
        else:
            slots["FEATURE"].append({
                "text": c["text"],
                "start": c["start"],
                "end": c["end"],
                "slot": "FEATURE",
            })

    return slots


# ---------------------------------------------------------
# main pipeline
# ---------------------------------------------------------

def run_pipeline(
    session_id: str,
    urls: List[str],
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    """
    tasks.py is calling us like:
        pipeline.run_pipeline(session_id=..., urls=..., s3_prefix=...)
    and BEFORE it calls us, tasks.py already downloaded the video to /tmp/....mp4
    (you can see that in your logs)

    So here we:
      1) take the first URL (that was downloaded) and re-download/normalize to local
      2) run ASR
      3) filter
      4) upload original video (or the same file) to S3
      5) return JSON in the same shape your UI expects
    """
    print("[pipeline] using filtered pipeline v1", flush=True)

    if not urls:
        raise ValueError("no urls provided")

    # ensure local video path
    # your tasks.py already called video.download_to_local(...) before calling us,
    # but we can safely call it again – if it's local, video.download_to_local just returns the same path.
    src = urls[0]
    local_video_path = video.download_to_local(src)

    # 1) run ASR
    raw_segments: List[Dict[str, Any]] = asr.transcribe_local(local_video_path)

    # 2) filter segments
    cleaned: List[Dict[str, Any]] = []
    seen_norm = set()

    for seg in raw_segments:
        txt = seg.get("text", "").strip()
        if not txt:
            continue

        # hard-bad lines: camera talk, "wait", "why can't i remember..."
        if _is_hard_bad(txt):
            continue

        norm = _normalize_for_dupe(txt)
        if norm in seen_norm:
            # it's a retake / duplicate wording → skip
            continue
        seen_norm.add(norm)

        cleaned.append(seg)

    # 3) build clips (your current shape)
    clips: List[Dict[str, Any]] = []
    for i, seg in enumerate(cleaned):
        clips.append(
            {
                "id": f"ASR{i:04d}",
                "slot": "STORY",   # we'll still give STORY; slots block will refine
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

    # 4) build slots
    slots = _slots_from_clips(clips)

    # 5) upload to S3 (same as your worker/s3.py style)
    # we’ll just re-upload the input video as the output mp4, to match your previous results
    out_key = f"{s3_prefix}{session_id}_{uuid.uuid4().hex}.mp4"
    https_url = s3.upload_file(local_video_path, out_key)

    # 6) final payload
    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        # if you want real duration, use worker.video.probe_duration(...)
        "duration_sec": video.probe_duration(local_video_path),
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
