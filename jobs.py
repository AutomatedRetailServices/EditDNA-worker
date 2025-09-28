# jobs.py — ASR-aware “human-edit” picker with V2 flags:
# face+fluency scoring, whole-clip veto, funnel ordering, proxy-fuse, env overrides.

from __future__ import annotations

import os
import re
import math
import uuid
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np

# ------------ S3 / captions helpers from your repo ------------
from s3_utils import upload_file, presigned_url, S3_BUCKET, download_to_tmp
from captions import write_srt, burn_captions
# --------------------------------------------------------------

FFMPEG  = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# ---------- knobs (env) ----------
ASR_ENABLED     = os.getenv("ASR_ENABLED", "true").lower() in ("1","true","yes","on")
ASR_MODEL_SIZE  = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANGUAGE    = os.getenv("ASR_LANG", "en")

BIN_SEC         = float(os.getenv("BIN_SEC", "0.5"))

# base weights (blend score from analysis bins + human checks)
W_AUDIO         = float(os.getenv("W_AUDIO",   "0.30"))
W_SCENE         = float(os.getenv("W_SCENE",   "0.15"))
W_SPEECH        = float(os.getenv("W_SPEECH",  "0.30"))
W_FACE          = float(os.getenv("W_FACE",    "0.20"))
W_FLUENCY       = float(os.getenv("W_FLUENCY", "0.35"))

# scene detect
SCENE_THRESH    = float(os.getenv("SCENE_THRESH", "0.04"))

# face/gaze heuristics
FACE_MIN_SIZE   = float(os.getenv("FACE_MIN_SIZE", "0.08"))   # % of frame min dimension
FACE_CENTER_TOL = float(os.getenv("FACE_CENTER_TOL", "0.35")) # normalized center distance tolerance

# fluency heuristics
FLUENCY_MIN_WPM        = float(os.getenv("FLUENCY_MIN_WPM", "95"))
FLUENCY_FILLER_PENALTY = float(os.getenv("FLUENCY_FILLER_PENALTY", "0.65"))
FILLERS = set(["um","uh","erm","hmm","like","you know","sort of","kinda"]) # simple

# veto
VETO_MIN_SCORE = float(os.getenv("VETO_MIN_SCORE", "0.40"))

# “human” smoothing
GRACE_SEC   = float(os.getenv("GRACE_SEC", "0.6"))    # allow small bad gaps inside a good region
MAX_BAD_SEC = float(os.getenv("MAX_BAD_SEC", "1.2"))  # cap them

# -------- V2 feature flags / overrides (NEW) --------
V2_SLOT_SCORER    = os.getenv("V2_SLOT_SCORER", "0") in ("1","true","yes","on")
V2_PROXY_FUSE     = os.getenv("V2_PROXY_FUSE", "0") in ("1","true","yes","on")
V2_VARIANT_EXPAND = os.getenv("V2_VARIANT_EXPAND", "0") in ("1","true","yes","on")
V2_CAPTIONER      = os.getenv("V2_CAPTIONER", "0") in ("1","true","yes","on")
CAPTIONS_MODE     = os.getenv("CAPTIONS", "off").strip().lower()  # off|soft|hard
MAX_DURATION_SEC_ENV = float(os.getenv("MAX_DURATION_SEC", "0"))  # 0 = unlimited
MIN_TAKE_SEC_ENV  = float(os.getenv("MIN_TAKE_SEC", "0"))  # 0 = use payload/defaults

# funnel targets (Hook, Feature, Proof, CTA)
FUNNEL_COUNTS = os.getenv("FUNNEL_COUNTS", "1,1,1,1")
FN_HOOK, FN_FEATURE, FN_PROOF, FN_CTA = [int(x) for x in FUNNEL_COUNTS.split(",")] if FUNNEL_COUNTS else [1,1,1,1]

# ---------- funnel keyword buckets ----------
HOOK_KWS    = ["stop", "wait", "before you", "attention", "hook", "secret", "did you know", "problem"]
FEATURE_KWS = ["feature", "includes", "comes with", "what it does", "works by", "how it works"]
PROOF_KWS   = ["proof", "testimonial", "reviews", "results", "case study", "worked for", "customers say"]
CTA_KWS     = ["call to action", "link in bio", "shop now", "buy now", "sign up", "get started", "cta"]

# ---------- utils ----------
def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n---\n{p.stdout}\n---")
    return p.stdout

def _duration(path: str) -> float:
    out = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nokey=1:noprint_wrappers=1", path])
    try:
        return float(out.strip())
    except Exception:
        return 0.0

def _safe_concat_list(path: str, files: List[str]) -> str:
    list_path = os.path.join(path, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            safe = str(p).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    return list_path

def _ensure_local(path_or_url: str, tmpdir: str) -> str:
    if path_or_url.startswith("s3://") or (path_or_url.startswith(("http://","https://")) and ".s3." in path_or_url):
        return download_to_tmp(path_or_url, tmpdir)

    if path_or_url.startswith(("http://","https://")):
        local = os.path.join(tmpdir, f"dl-{uuid.uuid4().hex}.mp4")
        cmd = [
            FFMPEG, "-y",
            "-analyzeduration", "200M", "-probesize", "200M",
            "-err_detect", "ignore_err",
            "-i", path_or_url,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            local
        ]
        try:
            _run(cmd)
            return local
        except Exception:
            cmd2 = [
                FFMPEG, "-y",
                "-i", path_or_url,
                "-map", "0:v:0",
                "-an",
                "-c:v", "copy",
                "-movflags", "+faststart",
                local
            ]
            _run(cmd2)
            return local

    return path_or_url

def _normalize_input(local_in: str, tmpdir: str) -> str:
    dur = max(0.0, _duration(local_in))
    base = os.path.join(tmpdir, f"norm-{uuid.uuid4().hex}.mp4")
    cmd1 = [
        FFMPEG, "-y",
        "-analyzeduration", "500M", "-probesize", "500M",
        "-fflags", "+genpts+ignidx",
        "-err_detect", "ignore_err",
        "-i", local_in,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        base
    ]
    try:
        _run(cmd1)
        return base
    except Exception:
        pass

    silent = os.path.join(tmpdir, f"sil-{uuid.uuid4().hex}.wav")
    if dur <= 0:
        dur = 3600.0
    _run([
        FFMPEG, "-y",
        "-f", "lavfi", "-t", f"{dur:.3f}",
        "-i", "anullsrc=r=48000:cl=stereo",
        "-c:a", "pcm_s16le", silent
    ])
    base2 = os.path.join(tmpdir, f"norm-silent-{uuid.uuid4().hex}.mp4")
    cmd2 = [
        FFMPEG, "-y",
        "-i", local_in,
        "-i", silent,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        base2
    ]
    _run(cmd2)
    return base2

@dataclass
class Clip:
    src: str
    start: float
    end: float
    score: float
    label: Optional[str] = None
    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# … (rest of code unchanged until job_render)

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    sess = str(payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}")
    files = list(payload.get("files") or [])
    out_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")
    portrait = bool(payload.get("portrait", True))
    mode = str(payload.get("mode") or "concat").lower().strip()

    min_clip = float(payload.get("min_clip_seconds") or (MIN_TAKE_SEC_ENV if MIN_TAKE_SEC_ENV > 0 else 2.5))
    max_clip = float(payload.get("max_clip_seconds") or 10.0)

    max_duration = payload.get("max_duration")
    take_top_k = payload.get("take_top_k")

    with_captions = bool(payload.get("with_captions", False))
    if V2_CAPTIONER and CAPTIONS_MODE in ("soft","hard"):
        with_captions = True

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{sess}-")
    try:
        if not files:
            return {"ok": False, "error": "No input files provided", "session_id": sess}

        local_norm: List[str] = []
        for f in files:
            loc = _ensure_local(f, tmpdir)
            norm = _normalize_input(loc, tmpdir)
            local_norm.append(norm)

        if mode in ("best","best_funnel","funnel"):
            clips, seg_cache = _pick_best(files, tmpdir, max_duration, take_top_k, min_clip, max_clip)
            if not clips:
                out_local = _render_concat(tmpdir, local_norm, portrait=portrait)
                mode_used = "concat_fallback"
            else:
                if mode in ("best_funnel","funnel") and ASR_ENABLED:
                    clips = _order_funnel(clips, seg_cache, float(max_duration) if max_duration else None)
                out_local = _render_clips(tmpdir, clips, portrait=portrait)
                mode_used = "funnel" if mode in ("best_funnel","funnel") else "best"
        else:
            out_local = _render_concat(tmpdir, local_norm, portrait=portrait)
            mode_used = "concat"

        srt_s3 = None
        srt_url = None
        if with_captions and ASR_ENABLED:
            segs = _asr_segments(out_local)
            if segs:
                srt_path = os.path.join(tmpdir, "subs.srt")
                write_srt(segs, srt_path)
                if CAPTIONS_MODE == "hard":
                    cap_out = os.path.join(tmpdir, "out_captions.mp4")
                    burn_captions(out_local, srt_path, cap_out)
                    out_local = cap_out
                else:
                    # soft: upload sidecar .srt and return its URL
                    srt_s3 = upload_file(srt_path, out_prefix + f"/{sess}", content_type="application/x-subrip")
                    _, srt_key = srt_s3.replace("s3://", "", 1).split("/", 1)
                    srt_url = presigned_url(S3_BUCKET, srt_key, expires=3600)

        s3_uri = upload_file(out_local, out_prefix + f"/{sess}", content_type="video/mp4")
        _, key = s3_uri.replace("s3://", "", 1).split("/", 1)
        url = presigned_url(S3_BUCKET, key, expires=3600)

        return {
            "ok": True,
            "session_id": sess,
            "mode": mode_used,
            "output_s3": s3_uri,
            "output_url": url,
            "inputs": files,
            "captions": bool(with_captions),
            "srt_s3": srt_s3,
            "srt_url": srt_url,
        }
    except Exception as e:
        return {"ok": False, "session_id": sess, "error": str(e), "inputs": files}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    return job_render(payload)
