# jobs.py — rendering pipeline (downloads from S3 → ffmpeg stitch → uploads MP4 to S3)
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

from s3_utils import download_to_tmp, upload_file


def _find_ffmpeg() -> str:
    """
    Find ffmpeg. We prefer system ffmpeg (you have it via apt.txt).
    If you later want to bundle, set env FFMPEG_PATH to a custom binary.
    """
    return os.getenv("FFMPEG_PATH", "ffmpeg")


def _write_concat_file(file_paths: List[str], concat_txt_path: str) -> None:
    """
    Write a concat list compatible with `-f concat -safe 0`.
    Each line must be: file 'absolute/path'
    We must escape any single quotes in the path as: '\''  (ffmpeg-friendly)
    """
    with open(concat_txt_path, "w", encoding="utf-8") as f:
        for p in file_paths:
            # escape single quotes for ffmpeg concat list
            escaped = p.replace("'", "'\\''")
            line = "file '{}'\n".format(escaped)
            f.write(line)


def _sorted_by_name(paths: List[str]) -> List[str]:
    """Deterministic ordering (alphabetical by filename)."""
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def _run(cmd: List[str]) -> None:
    """Run a shell command and raise on error (captures output for Render logs)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n---\n{proc.stdout}\n---")


def render_from_files(
    session_id: str,
    input_s3_urls: List[str],
    output_key_prefix: str,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Dict[str, Any]:
    """
    Minimal V1 renderer:
      1) Download inputs from S3 to a temp folder
      2) Sort by name (deterministic)
      3) Concat with ffmpeg (scales/pads to 9:16)
      4) Upload MP4 back to S3
    Returns: { ok, session_id, inputs, output_s3 }
    """
    if not input_s3_urls:
        return {"ok": False, "session_id": session_id, "error": "No input files provided"}

    ffmpeg = _find_ffmpeg()

    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    clips_dir = os.path.join(workdir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    try:
        # 1) Download
        local_paths: List[str] = []
        for url in input_s3_urls:
            local_paths.append(download_to_tmp(url, clips_dir))

        # 2) Order
        ordered = _sorted_by_name(local_paths)

        # 3) Concat & render
        concat_txt = os.path.join(workdir, "concat.txt")
        _write_concat_file(ordered, concat_txt)

        out_path = os.path.join(workdir, f"{session_id}.mp4")

        # Scale and pad to vertical 9:16 while preserving aspect
        vf = (
            f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

        cmd = [
            ffmpeg, "-y",
            "-safe", "0",
            "-f", "concat", "-i", concat_txt,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ]
        _run(cmd)

        # 4) Upload
        s3_uri = upload_file(out_path, key_prefix=f"{output_key_prefix.strip('/')}/{session_id}", content_type="video/mp4")

        return {
            "ok": True,
            "session_id": session_id,
            "inputs": len(input_s3_urls),
