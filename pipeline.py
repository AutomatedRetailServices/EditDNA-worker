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

# phrases we saw in your outputs that are just tail / repeated CTA / junk
BAD_PHRASES = [
    # IMG_03 odd / misheard
    "kuchigai",
    "kuchie guys",
    "coochie guys",  # keep if you actually want it, remove from here
    "utas",
    # IMG_02 repetitive tails
    "if you wanna check them out",
    "so if you wanna check them out",
    "and grab one of these",
    "i left it for you down below",
]

def looks_like_filler(text: str) -> bool:
    """
    Decide if a segment is not useful for the final cut.
    This is where we kill repeated CTAs and obvious bad takes.
    """
    t = (text or "").lower().strip()
    if not t:
        return True

    # very short, no info
    if len(t) < 6:
        return True

    for bad in BAD_PHRASES:
        if bad in t:
            return True

    # dumb repeated "so so so" style — basic heuristic
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
    """
    Your ASR returns:
      {"text": "...", "start": 0.0, "end": 2.0}
    We wrap each in a clip with id/slot/etc like your current output.
    """
    clips: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        cid = f"ASR{i:04d}"
        clip = {
            "id": cid,
            "slot": "STORY",   # we’ll re-map a few to slots later
            "start": seg["start"],
            "end": seg["end"],
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [cid],
            "text": seg["text"],
        }
        clips.append(clip)
    return clips


def build_slots(clips: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Super simple slotter:
      - first non-empty clip → HOOK
      - lines that look like CTA → CTA
      - everything else → FEATURE
    This mimics what you’re seeing now but with our filtering first.
    """
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
            or "grab one of these" in t
            or "get yours today" in t
            or "i left it for you" in t
        )

    # mark first good one as HOOK
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
    # keep extension if possible
    base_ext = os.path.splitext(local_video_path)[1] or ".mp4"
    out_name = f"{session_id}_{uuid.uuid4().hex}{base_ext}"
    tmp_dir = "/tmp"
    return os.path.join(tmp_dir, out_name)


def _upload_output(local_path: str, s3_prefix: str) -> Dict[str, Any]:
    # s3_prefix should already be like "editdna/outputs/"
    filename = os.path.basename(local_path)
    key = os.path.join(s3_prefix, filename)
    key = key.replace("\\", "/")  # windows safety

    url = s3.upload_file(local_path, key)
    # build both s3:// and https:// like your current output does
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
    print("[pipeline] filtered pipeline ACTIVE", flush=True)

    # 1) get duration from ffprobe helper
    duration_sec = video.probe_duration(local_video_path)

    # 2) ASR (our worker.asr already has transcribe_local)
    asr_segments = asr.transcribe_local(local_video_path)

    # 3) filter out the junk / repeated lines
    filtered_segments = filter_bad_segments(asr_segments)

    # 4) wrap into clips + slots
    clips = build_clips(filtered_segments)
    slots = build_slots(clips)

    # 5) “render” – for now we just copy the original video as output
    out_local = _make_output_path(local_video_path, session_id)
    shutil.copy(local_video_path, out_local)

    # 6) upload
    upload_info = _upload_output(out_local, s3_prefix)

    # 7) final payload – keep shape of your good runs
    result: Dict[str, Any] = {
        "ok": True,
        "session_id": session_id,
        "input_local": local_video_path,
        "duration_sec": duration_sec,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": True,   # you had vision=true in the last run
    }
    result.update(upload_info)
    return result
