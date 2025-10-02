# jobs.py — ASR-aware “human-edit” picker with funnel + proxy-fuse

from __future__ import annotations

import os, re, math, uuid, shutil, tempfile, subprocess
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

# ---------- knobs ----------
ASR_ENABLED     = os.getenv("ASR_ENABLED", "true").lower() in ("1","true","yes","on")
ASR_MODEL_SIZE  = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANGUAGE    = os.getenv("ASR_LANG", "en")

BIN_SEC         = float(os.getenv("BIN_SEC", "0.5"))

W_AUDIO   = float(os.getenv("W_AUDIO",   "0.30"))
W_SCENE   = float(os.getenv("W_SCENE",   "0.15"))
W_SPEECH  = float(os.getenv("W_SPEECH",  "0.30"))
W_FACE    = float(os.getenv("W_FACE",    "0.20"))
W_FLUENCY = float(os.getenv("W_FLUENCY", "0.35"))

SCENE_THRESH    = float(os.getenv("SCENE_THRESH", "0.04"))
FACE_MIN_SIZE   = float(os.getenv("FACE_MIN_SIZE", "0.08"))
FACE_CENTER_TOL = float(os.getenv("FACE_CENTER_TOL", "0.35"))
FLUENCY_MIN_WPM = float(os.getenv("FLUENCY_MIN_WPM", "95"))
FLUENCY_FILLER_PENALTY = float(os.getenv("FLUENCY_FILLER_PENALTY", "0.65"))

VETO_MIN_SCORE  = float(os.getenv("VETO_MIN_SCORE", "0.40"))

GRACE_SEC   = float(os.getenv("GRACE_SEC", "0.6"))
MAX_BAD_SEC = float(os.getenv("MAX_BAD_SEC", "1.2"))

V2_CAPTIONER  = os.getenv("V2_CAPTIONER", "0") in ("1","true","yes","on")
CAPTIONS_MODE = os.getenv("CAPTIONS", "off").strip().lower()

# funnel keyword buckets
HOOK_KWS    = ["stop", "wait", "before you", "attention", "secret", "did you know"]
FEATURE_KWS = ["feature", "includes", "comes with", "works by", "how it works"]
PROOF_KWS   = ["proof", "testimonial", "reviews", "results", "customers say"]
CTA_KWS     = ["link in bio", "shop now", "buy now", "sign up", "get started"]

# ---------- utils ----------
def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed {cmd}\n---\n{p.stdout}")
    return p.stdout

def _duration(path: str) -> float:
    try:
        out = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nokey=1:noprint_wrappers=1", path])
        return float(out.strip())
    except: return 0.0

def _ensure_local(url: str, tmpdir: str) -> str:
    if url.startswith("s3://") or (url.startswith(("http://","https://")) and ".s3." in url):
        return download_to_tmp(url, tmpdir)
    return url

def _render_concat(tmpdir: str, files: List[str], portrait: bool=True) -> str:
    list_path = os.path.join(tmpdir, "concat.txt")
    with open(list_path,"w") as f:
        for p in files: f.write(f"file '{p}'\n")
    out = os.path.join(tmpdir, "out_concat.mp4")
    _run([FFMPEG,"-y","-f","concat","-safe","0","-i",list_path,"-c","copy",out])
    return out

def _render_clips(tmpdir: str, clips: List["Clip"], portrait: bool=True) -> str:
    outs = []
    for c in clips:
        seg = os.path.join(tmpdir,f"seg-{uuid.uuid4().hex}.mp4")
        _run([FFMPEG,"-y","-i",c.src,"-ss",f"{c.start:.2f}","-to",f"{c.end:.2f}",
              "-c","copy",seg])
        outs.append(seg)
    return _render_concat(tmpdir, outs, portrait)

# ---------- ASR stub ----------
def _asr_segments(path: str) -> List[Tuple[float,float,str]]:
    # TODO: wire whisper
    return []

@dataclass
class Clip:
    src: str
    start: float
    end: float
    score: float
    label: Optional[str]=None
    @property
    def dur(self): return max(0.0,self.end-self.start)

# ---------- pick best ----------
def _pick_best(files: List[str], tmpdir: str, max_duration, topk, min_clip, max_clip):
    clips=[]; cache={}
    # TODO: wire scoring: audio/scene/face/fluency
    # For now return empty → concat fallback
    return clips, cache

def _order_funnel(clips: List[Clip], cache: Dict, max_duration: Optional[float]):
    buckets = defaultdict(list)
    for c in clips:
        txt = c.label or ""
        if any(k in txt for k in HOOK_KWS): buckets["hook"].append(c)
        elif any(k in txt for k in FEATURE_KWS): buckets["feature"].append(c)
        elif any(k in txt for k in PROOF_KWS): buckets["proof"].append(c)
        elif any(k in txt for k in CTA_KWS): buckets["cta"].append(c)
    return buckets.get("hook",[])+buckets.get("feature",[])+buckets.get("proof",[])+buckets.get("cta",[])

# ---------- main ----------
def job_render(payload: Dict[str,Any]) -> Dict[str,Any]:
    sess = str(payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}")
    files = list(payload.get("files") or [])
    out_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")
    portrait = bool(payload.get("portrait",True))
    mode = str(payload.get("mode") or "concat").lower()

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{sess}-")
    try:
        if not files: return {"ok":False,"error":"No files","session_id":sess}

        local_norm=[_ensure_local(f,tmpdir) for f in files]

        if mode in ("best","best_funnel","funnel"):
            clips, seg_cache = _pick_best(files,tmpdir,None,None,2.5,10.0)
            if not clips:
                out_local=_render_concat(tmpdir,local_norm,portrait)
                mode_used="concat_fallback"
            else:
                if mode in ("best_funnel","funnel") and ASR_ENABLED:
                    clips=_order_funnel(clips,seg_cache,None)
                out_local=_render_clips(tmpdir,clips,portrait)
                mode_used="funnel" if mode in ("best_funnel","funnel") else "best"
        else:
            out_local=_render_concat(tmpdir,local_norm,portrait)
            mode_used="concat"

        s3_uri=upload_file(out_local,f"{out_prefix}/{sess}",content_type="video/mp4")
        _,key=s3_uri.replace("s3://","",1).split("/",1)
        url=presigned_url(S3_BUCKET,key,expires=3600)

        return {"ok":True,"session_id":sess,"mode":mode_used,
                "output_s3":s3_uri,"output_url":url,"inputs":files}
    except Exception as e:
        return {"ok":False,"session_id":sess,"error":str(e),"inputs":files}
    finally:
        shutil.rmtree(tmpdir,ignore_errors=True)

def job_render_chunked(payload: Dict[str,Any]) -> Dict[str,Any]:
    return job_render(payload)
