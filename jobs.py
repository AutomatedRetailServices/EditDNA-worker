# jobs.py — minimal-robust render pipeline (concat or simple-picked clips)
# - Works end-to-end out of the box (always returns an MP4)
# - Defines _pick_best / _render_clips / _render_concat and helpers
# - Uses your existing s3_utils (upload_file, presigned_url, download_to_tmp)
# - Ready for later upgrade to ASR + scoring without API changes

from __future__ import annotations

import os, uuid, shutil, tempfile, subprocess, shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ------------ S3 / captions helpers from your repo ------------
from s3_utils import upload_file, presigned_url, S3_BUCKET, download_to_tmp
# If you want soft/hard captions later, wire these:
# from captions import write_srt, burn_captions
# --------------------------------------------------------------

FFMPEG  = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# ---------- knobs (kept compatible with your older file) ----------
ASR_ENABLED     = os.getenv("ASR_ENABLED", "false").lower() in ("1","true","yes","on")
ASR_MODEL_SIZE  = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANGUAGE    = os.getenv("ASR_LANG", "en")

BIN_SEC         = float(os.getenv("BIN_SEC", "0.5"))

# weights (not used by the simple heuristic, placeholders for future scoring)
W_AUDIO         = float(os.getenv("W_AUDIO",   "0.30"))
W_SCENE         = float(os.getenv("W_SCENE",   "0.15"))
W_SPEECH        = float(os.getenv("W_SPEECH",  "0.30"))
W_FACE          = float(os.getenv("W_FACE",    "0.20"))
W_FLUENCY       = float(os.getenv("W_FLUENCY", "0.35"))

SCENE_THRESH    = float(os.getenv("SCENE_THRESH", "0.04"))
FACE_MIN_SIZE   = float(os.getenv("FACE_MIN_SIZE", "0.08"))
FACE_CENTER_TOL = float(os.getenv("FACE_CENTER_TOL", "0.35"))
FLUENCY_MIN_WPM = float(os.getenv("FLUENCY_MIN_WPM", "95"))
FLUENCY_FILLER_PENALTY = float(os.getenv("FLUENCY_FILLER_PENALTY", "0.65"))

VETO_MIN_SCORE  = float(os.getenv("VETO_MIN_SCORE", "0.40"))

GRACE_SEC   = float(os.getenv("GRACE_SEC", "0.6"))
MAX_BAD_SEC = float(os.getenv("MAX_BAD_SEC", "1.2"))

V2_CAPTIONER  = os.getenv("V2_CAPTIONER", "0") in ("1","true","yes","on")
CAPTIONS_MODE = os.getenv("CAPTIONS", "off").strip().lower()  # off|soft|hard

# Funnel keyword buckets (reserved for future ASR-based ordering)
HOOK_KWS    = ["stop", "wait", "before you", "attention", "secret", "did you know"]
FEATURE_KWS = ["feature", "includes", "comes with", "works by", "how it works"]
PROOF_KWS   = ["proof", "testimonial", "reviews", "results", "customers say"]
CTA_KWS     = ["link in bio", "shop now", "buy now", "sign up", "get started"]

# ---------- utils ----------
def _run(cmd: str) -> str:
    """Run a shell command (string) and raise on failure."""
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n{cmd}\n---\n{p.stdout}")
    return p.stdout

def _duration(path: str) -> float:
    """Read media duration with ffprobe; returns 0.0 on error."""
    try:
        out = _run(f'{FFPROBE} -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "{path}"')
        return float(out.strip())
    except Exception:
        return 0.0

def _ensure_local(path_or_url: str, tmpdir: str) -> str:
    """Ensure input is local. If s3/http(s) to S3, use download_to_tmp; otherwise pass through."""
    if path_or_url.startswith("s3://") or (path_or_url.startswith(("http://", "https://")) and ".s3." in path_or_url):
        return download_to_tmp(path_or_url, tmpdir)
    return path_or_url

def _normalize_input(local_in: str, tmpdir: str) -> str:
    """
    Normalize codecs/containers to avoid concat/copy issues.
    H.264/AAC MP4 with faststart.
    """
    base = os.path.join(tmpdir, f"norm-{uuid.uuid4().hex}.mp4")
    cmd = (
        f'{FFMPEG} -y -hide_banner -loglevel error '
        f'-i "{local_in}" '
        f'-map 0:v:0 -map 0:a:0? '
        f'-c:v libx264 -preset veryfast -crf 20 '
        f'-c:a aac -ar 48000 -ac 2 '
        f'-movflags +faststart "{base}"'
    )
    _run(cmd)
    return base

def _cut_snippet(src: str, start: float, dur: float, out_path: str):
    """Cut a precise snippet, re-encoding to guarantee smooth concat when sources differ."""
    cmd = (
        f'{FFMPEG} -y -hide_banner -loglevel error '
        f'-ss {start:.3f} -i "{src}" -t {dur:.3f} '
        f'-c:v libx264 -preset veryfast -crf 20 '
        f'-c:a aac -ar 48000 -ac 2 '
        f'-movflags +faststart "{out_path}"'
    )
    _run(cmd)

def _concat_mp4(parts: List[str], out_path: str):
    """Concat via intermediate list file; inputs should already be compatible."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p}'\n")
        flist = f.name
    cmd = f'{FFMPEG} -y -hide_banner -loglevel error -f concat -safe 0 -i "{flist}" -c copy "{out_path}"'
    _run(cmd)

def _render_concat(tmpdir: str, files: List[str], portrait: bool=True) -> str:
    """Concat whole normalized files (simple fallback)."""
    out = os.path.join(tmpdir, "out_concat.mp4")
    _concat_mp4(files, out)
    return out

def _render_clips(tmpdir: str, clips: List["Clip"], portrait: bool=True) -> str:
    """Render selected clips (cuts) and concat them."""
    parts = []
    for c in clips:
        seg = os.path.join(tmpdir, f"seg-{uuid.uuid4().hex}.mp4")
        _cut_snippet(c.src, c.start, max(0.1, c.dur), seg)
        parts.append(seg)
    out = os.path.join(tmpdir, "out_clips.mp4")
    _concat_mp4(parts, out)
    return out

# ---------- ASR stub (placeholder for later Whisper wiring) ----------
def _asr_segments(path: str) -> List[Tuple[float, float, str]]:
    # TODO: integrate Whisper and return [(start, end, text), ...]
    return []

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

# ---------- simple heuristic picker (safe default) ----------
def _pick_best(files: List[str], tmpdir: str, max_duration: Optional[float],
               take_top_k: Optional[int], min_clip: float, max_clip: float) -> Tuple[List[Clip], Dict]:
    """
    Safe, dependency-light heuristic:
      - Normalize the first N inputs.
      - Slice a few evenly spaced segments within max_clip bounds.
      - Respect take_top_k if provided.
    This guarantees output while we later upgrade to ASR/face/fluency scoring.
    """
    if not files:
        return [], {}

    # how many total seconds of output we target (if provided)
    target_total = float(max_duration) if (max_duration and float(max_duration) > 0) else 20.0

    # how many clips
    k = int(take_top_k) if (take_top_k and int(take_top_k) > 0) else 6

    # normalize just the first file (common real-world case: one main talking clip)
    first_local = _ensure_local(files[0], tmpdir)
    norm = _normalize_input(first_local, tmpdir)
    dur = max(0.0, _duration(norm))
    if dur <= 0:
        return [], {}

    # decide per-clip duration within [min_clip, max_clip]
    per = max(min_clip, min(max_clip, target_total / max(1, k)))
    # space starts across the timeline
    spacing = max(per + 1.0, dur / (k + 1))

    clips: List[Clip] = []
    t = 0.0
    for i in range(k):
        start = i * spacing
        if start + 0.25 >= dur:  # nothing left
            break
        end = min(dur, start + per)
        clips.append(Clip(src=norm, start=start, end=end, score=1.0, label="auto"))
    return clips, {}

def _order_funnel(clips: List[Clip], cache: Dict, max_duration: Optional[float]) -> List[Clip]:
    """
    Placeholder funnel ordering; currently returns as-is.
    Later: bucket by ASR text (HOOK/FEATURE/PROOF/CTA) and order.
    """
    total = 0.0
    out: List[Clip] = []
    limit = float(max_duration) if (max_duration and float(max_duration) > 0) else 0.0
    for c in clips:
        if limit and total + c.dur > limit:
            break
        out.append(c); total += c.dur
    return out

# ---------- main ----------
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepted payload keys (compatible with your previous API):
      - session_id, files[], portrait, mode ("best"|"best_funnel"|"funnel"|"concat"),
        min_clip_seconds, max_clip_seconds, max_duration, take_top_k,
        with_captions (future), output_prefix
    Returns: { ok, session_id, mode, output_s3, output_url, ... }
    """
    sess = str(payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}")
    files = list(payload.get("files") or [])
    out_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")
    portrait = bool(payload.get("portrait", True))
    mode = str(payload.get("mode") or "concat").lower().strip()

    min_clip = float(payload.get("min_clip_seconds") or 1.5)
    max_clip = float(payload.get("max_clip_seconds") or 4.0)

    max_duration = payload.get("max_duration")  # may be None
    take_top_k = payload.get("take_top_k")      # may be None
    # with_captions = bool(payload.get("with_captions", False))  # reserved for later

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{sess}-")
    try:
        if not files:
            return {"ok": False, "error": "No input files provided", "session_id": sess}

        # Make every input local & normalized so concat never explodes on codec/container
        local_norm: List[str] = []
        for f in files:
            loc = _ensure_local(f, tmpdir)
            norm = _normalize_input(loc, tmpdir)
            local_norm.append(norm)

        # Decide path
        if mode in ("best", "best_funnel", "funnel"):
            clips, seg_cache = _pick_best(files, tmpdir, max_duration, take_top_k, min_clip, max_clip)
            if not clips:
                # robust fallback
                out_local = _render_concat(tmpdir, local_norm, portrait=portrait)
                mode_used = "concat_fallback"
            else:
                if mode in ("best_funnel", "funnel"):
                    clips = _order_funnel(clips, seg_cache, max_duration)
                out_local = _render_clips(tmpdir, clips, portrait=portrait)
                mode_used = "funnel" if mode in ("best_funnel", "funnel") else "best"
        else:
            out_local = _render_concat(tmpdir, local_norm, portrait=portrait)
            mode_used = "concat"

        # Upload result
        s3_uri = upload_file(out_local, f"{out_prefix}/{sess}", content_type="video/mp4")
        _, key = s3_uri.replace("s3://", "", 1).split("/", 1)
        url = presigned_url(S3_BUCKET, key, expires=3600)

        return {
            "ok": True,
            "session_id": sess,
            "mode": mode_used,
            "output_s3": s3_uri,
            "output_url": url,
            "inputs": files,
        }

    except Exception as e:
        return {"ok": False, "session_id": sess, "error": str(e), "inputs": files}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    return job_render(payload)
