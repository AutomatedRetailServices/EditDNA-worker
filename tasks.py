#!/usr/bin/env python3
# tasks.py
#
# RQ worker entrypoint.
# The web service enqueues tasks.job_render(...)
# The worker (on RunPod) imports this file and runs job_render().
#
# This file:
#   - downloads the input video from S3 (or URL)
#   - runs run_pipeline() from pipeline.py (Mode B logic)
#   - uploads stitched mp4 to your S3 bucket
#   - returns JSON like FastAPI expects

import os, io, uuid, time, json, tempfile, shutil
import boto3
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from pipeline import run_pipeline  # <- Mode B brain

S3_BUCKET   = os.getenv("S3_BUCKET", "")
AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
S3_PREFIX   = os.getenv("S3_PREFIX", "editdna/outputs")
S3_ACL      = os.getenv("S3_ACL", "public-read")

PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "86400"))

FFMPEG_BIN   = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN  = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

def _download_to_tmp(url:str)->str:
    """
    Pull remote file (S3 https or whatever) down to /tmp/xxx.mp4
    """
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    suffix = ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="tmp", suffix=suffix)
    with os.fdopen(tmp_fd,"wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    return tmp_path

def _upload_file_to_s3(local_path:str, key:str)->str:
    """
    Upload final mp4 to S3 and return public https_url (or presigned).
    """
    session = boto3.session.Session(region_name=AWS_REGION)
    s3c = session.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID",""),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY",""),
    )
    with open(local_path,"rb") as f:
        s3c.upload_fileobj(
            f,
            S3_BUCKET,
            key,
            ExtraArgs={"ACL": S3_ACL, "ContentType": "video/mp4"}
        )

    # public-style URL (your bucket is public-read in current setup)
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return https_url

def job_render(payload: Dict[str,Any]) -> Dict[str,Any]:
    """
    Called by RQ with a dict like:
    {
      "session_id": "...",
      "files": ["https://...mp4"],
      "portrait": true,
      "max_duration": 120,
      "audio": true
    }
    We only use files[0] for now.
    """

    t0=time.time()
    print("[tasks.job_render] INCOMING PAYLOAD KEYS:", list(payload.keys()), flush=True)

    session_id = payload.get("session_id","session")
    files      = payload.get("files",[])
    if not files:
        return {"ok": False, "error":"no files[] in payload"}

    # download first file
    src_url    = files[0]
    local_path = _download_to_tmp(src_url)

    # run pipeline
    result_core = run_pipeline(local_path=local_path, session_id=session_id)

    # after run_pipeline, we have a stitched file on disk INSIDE result_core? not yet.
    # We need to re-stitch according to what run_pipeline decided.
    # BUT run_pipeline in Mode B ALREADY exported and stitched and returned only JSON.
    # So we slightly adjust: modify run_pipeline to ALSO return 'final_path'.
    # -----
    # We didn't expose final_path above yet, so let's patch that now:
    # small hack: rerun internal stitching here, matching chosen clips.

    # Instead of re-running heavy logic, easiest path:
    # We assume final_path is NOT yet uploaded. We'll just reconstruct it here
    # using the chosen clips = result_core["clips"].
    # BUT run_pipeline already stitched and never told us the file path.
    # So: we add _stitch_again() helper here mirroring pipeline.stitch_video.
    from pipeline import stitch_video, build_funnel_order, pick_best_by_slot, tag_all, merge_chains, drop_retries, asr_segments

    # regenerate minimal final video path so we can upload:
    segs      = asr_segments(local_path)
    segs2     = drop_retries(segs)
    merged    = merge_chains(segs2)
    tagged    = tag_all(merged)
    best_map  = pick_best_by_slot(tagged)
    ordered   = build_funnel_order(best_map)
    if not ordered:
        ordered = merged[:1]

    final_path, final_dur, used_clips = stitch_video(local_path, ordered, float(os.getenv("MAX_DURATION_SEC","220")))

    # Upload that stitched mp4 to S3
    file_uuid = uuid.uuid4().hex
    s3_key = f"{S3_PREFIX}/{file_uuid}_{int(time.time())}.mp4"
    https_url = _upload_file_to_s3(final_path, s3_key)

    out = {
        "ok": True,
        "input_local": local_path,
        "duration_sec": round(final_dur,3),
        "s3_key": s3_key,
        "s3_url": f"s3://{S3_BUCKET}/{s3_key}",
        "https_url": https_url,
        "clips": [
            {
                "id": c["id"],
                "slot": c["slot"],
                "start": round(c["start"],2),
                "end": round(c["end"],2),
                "score": round(c["score"],2),
            } for c in used_clips
        ],
        "slots": result_core.get("slots", {}),
        "semantic": True,
        "vision": False,
        "asr": True,
    }

    print(f"[tasks.job_render] DONE in {time.time()-t0:.2f}s ok={out['ok']}", flush=True)
    return out
