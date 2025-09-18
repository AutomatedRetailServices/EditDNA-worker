# jobs.py — robust renderers (downloads from S3 → ffmpeg → uploads MP4)
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

from s3_utils import download_to_tmp, upload_file


def _ffmpeg() -> str:
    return os.getenv("FFMPEG_PATH", "ffmpeg")


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\n---\n{}\n---".format(
                proc.returncode, " ".join(cmd), proc.stdout
            )
        )


def _sorted(paths: List[str]) -> List[str]:
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


# -------------------------------
# Simple filter-concat (works; re-encodes all at once) — still heavy
# -------------------------------
def render_from_files(
    session_id: str,
    input_s3_urls: List[str],
    output_key_prefix: str,
    target_width: int = 720,
    target_height: int = 1280,
) -> Dict[str, Any]:
    if not input_s3_urls:
        return {"ok": False, "session_id": session_id, "error": "No input files provided"}

    ffmpeg = _ffmpeg()
    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    clips_dir = os.path.join(workdir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    try:
        locals_: List[str] = [download_to_tmp(u, clips_dir) for u in input_s3_urls]
        ordered = _sorted(locals_)

        # Inputs
        cmd: List[str] = [ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "info",
                          "-analyzeduration", "100M", "-probesize", "100M"]
        for p in ordered:
            cmd += ["-i", p]

        # Filters
        n = len(ordered)
        per, labels = [], []
        for i in range(n):
            vi = f"v{i}"
            per.append(
                f"[{i}:v]scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p,setsar=1[{vi}]"
            )
            labels.append(f"[{vi}]")
        filter_complex = ";".join(per + ["".join(labels) + f"concat=n={n}:v=1:a=0[v]"])

        out_path = os.path.join(workdir, f"{session_id}.mp4")
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart",
            "-max_muxing_queue_size", "9999",
            "-threads", "1",
            out_path,
        ]
        _run(cmd)

        s3_uri = upload_file(out_path, key_prefix=f"{output_key_prefix.strip('/')}/{session_id}",
                             content_type="video/mp4")
        return {"ok": True, "session_id": session_id, "inputs": len(ordered), "output_s3": s3_uri}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# -------------------------------
# CHUNKED pipeline (low RAM/CPU): transcode each clip → identical intermediates → concat fast
# -------------------------------
def render_chunked(
    session_id: str,
    input_s3_urls: List[str],
    output_key_prefix: str,
    target_width: int = 720,
    target_height: int = 1280,
) -> Dict[str, Any]:
    if not input_s3_urls:
        return {"ok": False, "session_id": session_id, "error": "No input files provided"}

    ffmpeg = _ffmpeg()
    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    clips_dir = os.path.join(workdir, "clips")
    interm_dir = os.path.join(workdir, "interm")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(interm_dir, exist_ok=True)

    try:
        # 1) Download
        locals_: List[str] = [download_to_tmp(u, clips_dir) for u in input_s3_urls]
        ordered = _sorted(locals_)

        # 2) Transcode each to uniform MP4 (video only) — one-by-one (low memory)
        interm_paths: List[str] = []
        for idx, src in enumerate(ordered):
            outi = os.path.join(interm_dir, f"part_{idx:04d}.mp4")
            vf = (f"scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,"
                  f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                  "format=yuv420p,setsar=1")
            cmd = [
                ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "info",
                "-analyzeduration", "100M", "-probesize", "100M",
                "-i", src,
                "-vf", vf,
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-movflags", "+faststart",
                "-max_muxing_queue_size", "9999",
                "-threads", "1",
                outi,
            ]
            _run(cmd)
            interm_paths.append(outi)

        # 3) Concat intermediates via demuxer (now identical => safe & very fast)
        concat_txt = os.path.join(workdir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in interm_paths:
                esc = p.replace("'", "'\\''")
                f.write(f"file '{esc}'\n")

        out_path = os.path.join(workdir, f"{session_id}.mp4")
        cmd2 = [
            ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "info",
            "-safe", "0",
            "-f", "concat", "-i", concat_txt,
            "-c", "copy",            # no re-encode on the final stitch
            "-movflags", "+faststart",
            out_path,
        ]
        _run(cmd2)

        # 4) Upload
        s3_uri = upload_file(out_path, key_prefix=f"{output_key_prefix.strip('/')}/{session_id}",
                             content_type="video/mp4")
        return {"ok": True, "session_id": session_id, "inputs": len(ordered), "output_s3": s3_uri}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
