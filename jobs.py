# jobs.py — worker ffmpeg render + S3 upload

import os
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any

from s3_utils import upload_file, presigned_url, parse_s3_url

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

def _write_concat_file(tmpdir: str, files: List[str]) -> str:
    """
    Write an ffconcat-style list file for the concat demuxer.
    Each line: file '...'
    """
    list_path = os.path.join(tmpdir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            safe = str(p).replace("'", "'\\''")
            f.write("file '{}'\n".format(safe))
    return list_path

def _run_ffmpeg(cmd: List[str]) -> None:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode})\nCMD: {' '.join(cmd)}\n---\n{proc.stdout}\n---"
        )

def job_render(session_id: str, files: List[str], output_prefix: str) -> Dict[str, Any]:
    """
    Concatenate the given clips into a 1080x1920 MP4, upload to S3, return presigned URL.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    out_path = os.path.join(tmpdir, f"{session_id}.mp4")

    try:
        concat_txt = _write_concat_file(tmpdir, files)

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
            "-safe", "0",
            "-f", "concat",
            "-i", concat_txt,
            "-ignore_unknown", "1",
            "-vf", "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
                   "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ]
        _run_ffmpeg(cmd)

        # Upload to S3 and presign
        s3_url = upload_file(out_path, key_prefix=output_prefix, content_type="video/mp4")
        bucket, key = parse_s3_url(s3_url)
        if bucket is None:
            # When upload_file returns s3://<DEFAULT_BUCKET>/key, bucket will be None here only
            # if the URL didn’t include bucket — but our helper always does. Guard anyway.
            raise RuntimeError(f"Could not parse bucket/key from {s3_url}")
        url = presigned_url(bucket, key, expires=3600)

        return {
            "ok": True,
            "session_id": session_id,
            "mode": "concat",
            "inputs": files,
            "output_s3": s3_url,
            "output_url": url,
        }
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def job_render_chunked(session_id: str, files: List[str], output_prefix: str) -> Dict[str, Any]:
    # For now same as job_render
    return job_render(session_id, files, output_prefix)
