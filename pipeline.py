# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import uuid
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any

from worker import asr
from worker import video
from worker import s3


# ------------------------------------------------------------
# 1) phrases we actually want to drop (from your logs)
# ------------------------------------------------------------
DEFAULT_BAD_PHRASES = [
    "if you wanna check them out",
    "so if you wanna check them out",
    "i left it for you down below",
    "and grab one of these westland",
    "and grab one of these for yourself",
    "go ahead and click the link",
]


def _load_bad_phrases() -> List[str]:
    env_val = os.getenv("BAD_PHRASES", "").strip()
    if not env_val:
        return [p.lower() for p in DEFAULT_BAD_PHRASES]
    return [p.strip().lower() for p in env_val.split("|") if p.strip()]


BAD_PHRASES = _load_bad_phrases()


def is_bad_phrase(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    for bad in BAD_PHRASES:
        if bad in t:
            return True
    return False


def is_near_duplicate(prev_text: str, curr_text: str) -> bool:
    if not prev_text or not curr_text:
        return False
    p = prev_text.lower().strip()
    c = curr_text.lower().strip()
    if p == c:
        return True
    if p in c or c in p:
        return True
    return False


# ------------------------------------------------------------
# 2) turn ASR segments into filtered clips
# ------------------------------------------------------------
def build_filtered_clips(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    last_kept_text = ""

    for i, seg in enumerate(segments):
        txt = seg["text"]

        # drop lines we know we don't want
        if is_bad_phrase(txt):
            continue

        # drop immediate duplicates
        if last_kept_text and is_near_duplicate(last_kept_text, txt):
            continue

        cid = f"ASR{i:04d}"
        clip = {
            "id": cid,
            "slot": "STORY",  # we'll classify later
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [cid],
            "text": txt,
        }
        clips.append(clip)
        last_kept_text = txt

    return clips


# ------------------------------------------------------------
# 3) slot builder (same shape as your previous results)
# ------------------------------------------------------------
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
            c = clip.copy()
            c["slot"] = "HOOK"
            slots["HOOK"].append(c)
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


# ------------------------------------------------------------
# 4) ffmpeg helpers to CUT and CONCAT
# ------------------------------------------------------------
def _run_ffmpeg(cmd: List[str]) -> None:
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cut_clip_ffmpeg(src: str, start: float, end: float, out_path: str) -> None:
    """
    Cut one piece from src → out_path using stream copy (fast)
    """
    duration = max(0.0, end - start)
    # -ss BEFORE -i is faster but less accurate; this is OK for your use-case
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start}",
        "-i",
        src,
        "-t",
        f"{duration}",
        "-c",
        "copy",
        out_path,
    ]
    _run_ffmpeg(cmd)


def concat_clips_ffmpeg(parts: List[str], out_path: str) -> None:
    """
    Use concat demuxer: create a text file with all parts
    """
    if not parts:
        raise ValueError("no parts to concat")

    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for p in parts:
            # paths must be quoted or escaped; simplest is single quotes
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        out_path,
    ]
    _run_ffmpeg(cmd)
    os.remove(list_file)


# ------------------------------------------------------------
# 5) upload helper
# ------------------------------------------------------------
def _upload_output(local_path: str, s3_prefix: str) -> Dict[str, Any]:
    filename = os.path.basename(local_path)
    key = os.path.join(s3_prefix, filename).replace("\\", "/")
    bucket = os.environ.get("S3_BUCKET", "script2clipshop-video-automatedretailservices")
    url = s3.upload_file(local_path, key)
    s3_url = f"s3://{bucket}/{key}"
    return {
        "s3_key": key,
        "s3_url": [s3_url, url],
        "https_url": [s3_url, url],
    }


# ------------------------------------------------------------
# 6) main entry – tasks.py calls this
# ------------------------------------------------------------
def run_pipeline(
    local_video_path: str,
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    print("[pipeline] CUT-BY-SCRIPT pipeline ACTIVE", flush=True)

    # 1) get duration
    duration_sec = video.probe_duration(local_video_path)

    # 2) ASR
    segments = asr.transcribe_local(local_video_path)

    # 3) FILTER into clips
    clips = build_filtered_clips(segments)

    # 4) build slots (for your UI)
    slots = build_slots(clips)

    # 5) actually CUT the video according to those clips
    tmp_dir = tempfile.mkdtemp(prefix="cutparts_")
    part_paths: List[str] = []
    for idx, c in enumerate(clips):
        part_out = os.path.join(tmp_dir, f"part_{idx:04d}.mp4")
        # guard bad times
        start = max(0.0, float(c["start"]))
        end = max(start, float(c["end"]))
        cut_clip_ffmpeg(local_video_path, start, end, part_out)
        part_paths.append(part_out)

    # if nothing survived filtering, just upload original
    if part_paths:
        final_local = os.path.join("/tmp", f"{session_id}_{uuid.uuid4().hex}.mp4")
        concat_clips_ffmpeg(part_paths, final_local)
    else:
        # nothing to concat – upload original
        final_local = os.path.join("/tmp", f"{session_id}_{uuid.uuid4().hex}.mp4")
        shutil.copy(local_video_path, final_local)

    upload_info = _upload_output(final_local, s3_prefix)

    # 6) return JSON similar to your earlier ones
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

    # clean temp parts (best-effort)
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    return result
