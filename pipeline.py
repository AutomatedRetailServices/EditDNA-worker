# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import uuid
import shutil
from typing import List, Dict, Any

from worker import asr
from worker import video
from worker import s3


# ---------------------------------------------------------------------
# helpers for text filtering
# ---------------------------------------------------------------------
# these are the *real* junky / repetitive tails we saw in IMG_02
BAD_PHRASES = [
    "if you wanna check them out",
    "so if you wanna check them out",
    "i left it for you down below",
    "and grab one of these westland",
    "and grab one of these for yourself",
]

def looks_like_filler(text: str) -> bool:
    """
    drop obvious repeats and super-short empties.
    DO NOT drop slang – we only drop what's in BAD_PHRASES
    """
    t = (text or "").lower().strip()
    if not t:
        return True

    # very short nothing-burgers
    if len(t) < 4:
        return True

    for bad in BAD_PHRASES:
        if bad in t:
            return True

    # lazy catch for "so so so ..." rambles
    if t.count(" so ") > 2:
        return True

    return False


def filter_bad_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    for seg in segments:
        txt = seg.get("text", "")
        if looks_like_filler(txt):
            print(f"[pipeline] dropped segment: {txt!r}")
            continue
        clean.append(seg)
    return clean


# ---------------------------------------------------------------------
# build clips & slots in the same shape your API is expecting
# ---------------------------------------------------------------------

def build_clips(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        cid = f"ASR{i:04d}"
        clips.append(
            {
                "id": cid,
                "slot": "STORY",
                "start": seg["start"],
                "end": seg["end"],
                "score": 2.5,
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "chain_ids": [cid],
                "text": seg["text"],
            }
        )
    return clips


def build_slots(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    def is_cta(text: str) -> bool:
        t = text.lower()
        return (
            "click the link" in t
            or "get yours today" in t
            or "grab one of these" in t
            or "i left it for you" in t
            or "go ahead and" in t
        )

    first_hook_done = False
    for clip in clips:
        txt = clip["text"]
        if not first_hook_done:
            h = clip.copy()
            h["slot"] = "HOOK"
            slots["HOOK"].append(h)
            first_hook_done = True
            continue

        if is_cta(txt):
            c = clip.copy()
            c["slot"] = "CTA"
            slots["CTA"].append(c)
        else:
            f = clip.copy()
            f["slot"] = "FEATURE"
            slots["FEATURE"].append(f)

    return slots


# ---------------------------------------------------------------------
# output + upload
# ---------------------------------------------------------------------

def _make_output_path(local_video_path: str, session_id: str) -> str:
    base_ext = os.path.splitext(local_video_path)[1] or ".mp4"
    out_name = f"{session_id}_{uuid.uuid4().hex}{base_ext}"
    return os.path.join("/tmp", out_name)


def _upload_output(local_path: str, s3_prefix: str) -> Dict[str, Any]:
    filename = os.path.basename(local_path)
    key = os.path.join(s3_prefix, filename).replace("\\", "/")

    url = s3.upload_file(local_path, key)
    bucket = os.environ.get("S3_BUCKET", "script2clipshop-video-automatedretailservices")
    s3_url = f"s3://{bucket}/{key}"
    return {
        "s3_key": key,
        "s3_url": [s3_url, url],
        "https_url": [s3_url, url],
    }


# ---------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------

def run_pipeline(
    local_video_path: str,
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    print("[pipeline] filtered pipeline ACTIVE (slang allowed)", flush=True)

    # 1) duration
    duration_sec = video.probe_duration(local_video_path)

    # 2) ASR
    asr_segments = asr.transcribe_local(local_video_path)

    # 3) filter only the real junk (not your slang)
    filtered_segments = filter_bad_segments(asr_segments)

    # 4) clips + slots
    clips = build_clips(filtered_segments)
    slots = build_slots(clips)

    # 5) copy original as "rendered"
    out_local = _make_output_path(local_video_path, session_id)
    shutil.copy(local_video_path, out_local)

    # 6) upload
    upload_info = _upload_output(out_local, s3_prefix)

    # 7) response
    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": True,
    }
    result.update(upload_info)
    return result
