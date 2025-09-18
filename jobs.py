# jobs.py — rendering pipeline (downloads from S3 → ffmpeg stitch → uploads MP4 to S3)
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

from s3_utils import download_to_tmp, upload_file


def _find_ffmpeg() -> str:
    return os.getenv("FFMPEG_PATH", "ffmpeg")


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\n---\n{}\n---".format(
                proc.returncode, " ".join(cmd), proc.stdout
            )
        )


def _sorted_by_name(paths: List[str]) -> List[str]:
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


def render_from_files(
    session_id: str,
    input_s3_urls: List[str],
    output_key_prefix: str,
    # ↓ reduce output size to cut CPU/RAM on starter plan
    target_width: int = 720,
    target_height: int = 1280,
) -> Dict[str, Any]:
    """
    Robust V1 renderer (video-only):
      - Downloads inputs from S3
      - Scales/pads each clip to 9:16
      - Concatenates via filter_complex (re-encodes; mixed codecs OK)
      - Uploads MP4 back to S3
    """
    if not input_s3_urls:
        return {"ok": False, "session_id": session_id, "error": "No input files provided"}

    ffmpeg = _find_ffmpeg()

    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    clips_dir = os.path.join(workdir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    try:
        # 1) Download all clips
        local_paths: List[str] = []
        for url in input_s3_urls:
            local_paths.append(download_to_tmp(url, clips_dir))

        ordered = _sorted_by_name(local_paths)

        # 2) Inputs
        cmd: List[str] = [
            ffmpeg, "-y",
            "-nostdin", "-hide_banner",
            "-loglevel", "info",
            "-analyzeduration", "100M", "-probesize", "100M",
        ]
        for p in ordered:
            cmd += ["-i", p]

        # 3) Filter graph
        n = len(ordered)
        per_inputs, vlabels = [], []
        for i in range(n):
            vi = f"v{i}"
            per_inputs.append(
                f"[{i}:v]scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"format=yuv420p,setsar=1[{vi}]"
            )
            vlabels.append(f"[{vi}]")
        filter_complex = ";".join(per_inputs + [ "".join(vlabels) + f"concat=n={n}:v=1:a=0[v]" ])

        out_path = os.path.join(workdir, f"{session_id}.mp4")

        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart",
            "-max_muxing_queue_size", "9999",
            "-threads", "1",                # ↓ keep memory/CPU low
            out_path,
        ]

        _run(cmd)

        s3_uri = upload_file(
            out_path,
            key_prefix=f"{output_key_prefix.strip('/')}/{session_id}",
            content_type="video/mp4",
        )
        return {"ok": True, "session_id": session_id, "inputs": len(input_s3_urls), "output_s3": s3_uri}
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
