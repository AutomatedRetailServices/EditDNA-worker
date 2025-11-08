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
# 1) phrases we actually want to drop (seen in your JSON)
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
        return DEFAULT_BAD_PHRASES
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
    # tiny variations
    if p in c or c in p:
        return True
    return False


# ------------------------------------------------------------
# build filtered clips
# ------------------------------------------------------------
def build_filtered_clips(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    last_kept = ""

    for i, seg in enumerate(segments):
        txt = seg["text"]

        # drop known junk
        if is_bad_phrase(txt):
            continue

        # drop immediate repeats
        if last_kept and is_near_duplicate(last_kept, txt):
            continue

        cid = f"ASR{i:04d}"
        clip = {
            "id": cid,
            "slot": "STORY",  # temp
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
        last_kept = txt

    return clips


# ------------------------------------------------------------
# slots (same structure your UI shows)
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

    first_hook = True
    for clip in clips:
        txt = clip["text"]
        if first_hook:
            h = dict(clip)
            h["slot"] = "HOOK"
            slots["HOOK"].append(h)
            first_hook = False
            continue

        if is_cta(txt):
            c = dict(clip)
            c["slot"] = "CTA"
            slots["CTA"].append(c)
        else:
            f = dict(clip)
            f["slot"] = "FEATURE"
            slots["FEATURE"].append(f)

    return slots


# ------------------------------------------------------------
# ffmpeg helpers
# ------------------------------------------------------------
def _run_ffmpeg(cmd: List[str]) -> None:
    subprocess.check_call(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _make_output_path(session_id: str) -> str:
    name = f"{session_id}_{uuid.uuid4().hex}.mp4"
    return os.path.join("/tmp", name)


def cut_video_from_clips(
    input_video: str,
    clips: List[Dict[str, Any]],
    session_id: str,
) -> str:
    """
    Actually create a new video that is ONLY the kept clips.
    We'll re-encode every piece to avoid keyframe problems.
    """
    if not clips:
        # nothing kept – just return original
        return input_video

    tmpdir = tempfile.mkdtemp(prefix="editdna_")
    part_paths: List[str] = []

    # 1) make small parts
    for idx, clip in enumerate(clips):
        start = clip["start"]
        end = clip["end"]
        duration = max(0.01, end - start)
        part_path = os.path.join(tmpdir, f"part_{idx:03d}.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start}",
            "-t", f"{duration}",
            "-i", input_video,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "faststart",
            part_path,
        ]
        _run_ffmpeg(cmd)
        part_paths.append(part_path)

    # 2) make concat file
    concat_path = os.path.join(tmpdir, "concat.txt")
    with open(concat_path, "w") as f:
        for p in part_paths:
            # concat demuxer wants "file 'path'"
            f.write(f"file '{p}'\n")

    # 3) concat to final
    final_path = _make_output_path(session_id)
    cmd2 = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_path,
        "-c", "copy",
        final_path,
    ]
    # if copy fails on your ffmpeg build, you can re-encode instead:
    # cmd2 = [
    #     "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    #     "-i", concat_path,
    #     "-c:v", "libx264", "-c:a", "aac", "-movflags", "faststart",
    #     final_path,
    # ]
    _run_ffmpeg(cmd2)

    # we can leave tmpdir there or clean — for now just return final_path
    return final_path


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
# main entry
# ------------------------------------------------------------
def run_pipeline(
    local_video_path: str,
    session_id: str,
    s3_prefix: str = "editdna/outputs/",
) -> Dict[str, Any]:
    print("[pipeline] filtered+cut pipeline ACTIVE", flush=True)

    # 1) get duration
    duration_sec = video.probe_duration(local_video_path)

    # 2) transcribe
    segments = asr.transcribe_local(local_video_path)

    # 3) filter to only good talking parts
    clips = build_filtered_clips(segments)

    # 4) build slots for UI
    slots = build_slots(clips)

    # 5) ACTUAL VIDEO CUT HERE
    final_local = cut_video_from_clips(local_video_path, clips, session_id)

    # 6) upload
    upload_info = _upload_output(final_local, s3_prefix)

    # 7) return metadata
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
