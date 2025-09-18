# jobs.py — rendering pipeline (downloads from S3 → ffmpeg stitch → uploads MP4 to S3)
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

from s3_utils import download_to_tmp, upload_file


def _find_ffmpeg() -> str:
    """Prefer system ffmpeg (you have it via apt.txt)."""
    return os.getenv("FFMPEG_PATH", "ffmpeg")


def _run(cmd: List[str]) -> None:
    """Run a shell command and raise on error (captures output for logs)."""
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
    target_width: int = 1080,
    target_height: int = 1920,
) -> Dict[str, Any]:
    """
    Robust V1 renderer (video-only):
      - Downloads inputs from S3
      - Scales/pads each clip to 9:16
      - Concatenates via filter_complex (re-encodes, so mixed codecs are OK)
      - Uploads MP4 back to S3
    """
    if not input_s3_urls:
        return {"ok": False, "session_id": session_id, "error": "No input files provided"}

    ffmpeg = _find_ffmpeg()

    workdir = tempfile.mkdtemp(prefix="editdna-{}-".format(session_id))
    clips_dir = os.path.join(workdir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    try:
        # 1) Download all clips
        local_paths: List[str] = []
        for url in input_s3_urls:
            local_paths.append(download_to_tmp(url, clips_dir))

        ordered = _sorted_by_name(local_paths)

        # 2) Build ffmpeg inputs
        cmd: List[str] = [
            ffmpeg, "-y",
            "-nostdin",               # never wait for tty input
            "-hide_banner",
            "-loglevel", "info",
            "-analyzeduration", "100M",
            "-probesize", "100M",
        ]
        for p in ordered:
            cmd += ["-i", p]

        # 3) Build filter_complex for N inputs
        n = len(ordered)
        per_inputs = []
        vlabels = []
        for i in range(n):
            vi = f"v{i}"
            per_inputs.append(
                f"[{i}:v]scale=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease,"
                f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"format=yuv420p,setsar=1[{vi}]"
            )
            vlabels.append(f"[{vi}]")

        concat_part = "".join(vlabels) + f"concat=n={n}:v=1:a=0[v]"
        filter_complex = ";".join(per_inputs + [concat_part])

        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-an",                           # video-only output (V1)
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-movflags", "+faststart",
        ]

        out_path = os.path.join(workdir, "{}.mp4".format(session_id))
        cmd += [out_path]

        # 4) Run ffmpeg
        _run(cmd)

        # 5) Upload to S3
        s3_uri = upload_file(
            out_path,
            key_prefix="{}/{}".format(output_key_prefix.strip("/"), session_id),
            content_type="video/mp4",
        )

        return {
            "ok": True,
            "session_id": session_id,
            "inputs": len(input_s3_urls),
            "output_s3": s3_uri,
        }
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
