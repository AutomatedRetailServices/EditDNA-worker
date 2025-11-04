import os, time, uuid, tempfile, subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import boto3

# ---------------- ENV ----------------
def _env_str(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d

def _env_float(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip().split()[:1]
    try:
        return float(v[0]) if v else d
    except:
        return d

FFMPEG_BIN  = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET  = _env_str("S3_BUCKET", "")
S3_PREFIX  = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION = _env_str("AWS_REGION", "us-east-1")
S3_ACL     = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 2.0)


@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start


# ---------------- helpers ----------------
def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()


def _tmpfile(suffix=".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix); os.close(fd); return p


def _download_to_tmp(url: str) -> str:
    local_path = _tmpfile(suffix=".mp4")
    code, out, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local_path


def _ffprobe_duration(path: str) -> float:
    code, out, err = _run([
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path,
    ])
    if code != 0:
        return 0.0
    try:
        return float(out.strip())
    except:
        return 0.0


def _export_concat(src: str, takes: List[Take], burn_srt: bool = False) -> str:
    """
    Stitch selected takes into one final mp4
    """
    if not takes:
        # fallback 5s
        takes = [Take(id="FALLBACK", start=0.0, end=min(5.0, MAX_DURATION_SEC), text="")]

    parts: List[str] = []
    listfile = _tmpfile(suffix=".txt")

    for idx, t in enumerate(takes, start=1):
        part = _tmpfile(suffix=f".part{idx:02d}.mp4")
        parts.append(part)
        _run([
            FFMPEG_BIN, "-y",
            "-ss", f"{t.start:.3f}",
            "-i", src,
            "-t", f"{t.dur:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-g", "48",
            "-c:a", "aac",
            "-b:a", "128k",
            part
        ])

    with open(listfile, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    final = _tmpfile(suffix=".mp4")
    _run([
        FFMPEG_BIN, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", listfile,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        final
    ])
    return final


def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")

    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh,
            S3_BUCKET,
            key,
            ExtraArgs={
                "ACL": S3_ACL,
                "ContentType": "video/mp4",
            },
        )
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
    }


# ---------------- main pipeline ----------------
def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts,
    max_duration: float,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    # 1) get video locally
    src = _download_to_tmp(file_urls[0])

    # 2) measure actual video
    vid_dur = _ffprobe_duration(src)
    # limit to caller max
    cap = min(float(max_duration or MAX_DURATION_SEC), vid_dur if vid_dur > 0 else MAX_DURATION_SEC)

    # 3) create dumb 20s segments from duration
    takes: List[Take] = []
    t = 0.0
    seg_idx = 1
    while t < cap:
        end = min(t + MAX_TAKE_SEC, cap)
        if (end - t) >= MIN_TAKE_SEC:
            takes.append(
                Take(
                    id=f"SEG{seg_idx:04d}",
                    start=t,
                    end=end,
                    text=f"Auto segment {seg_idx} ({t:.1f}s–{end:.1f}s)",
                )
            )
            seg_idx += 1
        t = end

    # 4) for now, story = all takes in order
    story = takes

    # 5) export video from those takes
    final_path = _export_concat(src, story)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    # 6) build JSON blocks
    clips_block = [
        {
            "id": t.id,
            "slot": "STORY",
            "start": t.start,
            "end": t.end,
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [t.id],
            "text": t.text,
        }
        for t in story
    ]

    # super simple funnel: 1st = HOOK, middle = PROBLEM/FEATURE, last = CTA
    slots_block = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    if story:
        slots_block["HOOK"].append(
            {
                "id": story[0].id,
                "start": story[0].start,
                "end": story[0].end,
                "text": story[0].text,
                "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [story[0].id]},
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            }
        )
    if len(story) > 2:
        for mid in story[1:-1]:
            slots_block["FEATURE"].append(
                {
                    "id": mid.id,
                    "start": mid.start,
                    "end": mid.end,
                    "text": mid.text,
                    "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": [mid.id]},
                    "face_q": 1.0,
                    "scene_q": 1.0,
                    "vtx_sim": 0.0,
                    "has_product": False,
                    "ocr_hit": 0,
                }
            )
    if len(story) >= 2:
        last = story[-1]
        slots_block["CTA"].append(
            {
                "id": last.id,
                "start": last.start,
                "end": last.end,
                "text": last.text,
                "meta": {"slot": "CTA", "score": 2.0, "chain_ids": [last.id]},
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            }
        )

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips_block,
        "slots": slots_block,
        "semantic": False,
        "vision": False,
        "asr": False,
    }
