try:
    from worker.semantic_visual_pass import continuity_chains
except Exception:
    def continuity_chains(takes):
        # fallback: don’t chain; return each take alone
        return [[t] for t in takes]

# jobs.py — robust render pipeline with optional Whisper ASR (off by default)
from __future__ import annotations
import os, uuid, shutil, tempfile, subprocess, shlex, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from s3_utils import upload_file, presigned_url, S3_BUCKET, download_to_tmp

FFMPEG  = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

ASR_ENABLED     = os.getenv("ASR_ENABLED", "0").lower() in ("1","true","yes","on")
ASR_MODEL_SIZE  = os.getenv("ASR_MODEL_SIZE", "base")
ASR_LANGUAGE    = os.getenv("ASR_LANG", "en")

def _run(cmd: str) -> str:
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n{cmd}\n---\n{p.stdout}")
    return p.stdout

def _duration(path: str) -> float:
    try:
        out = _run(f'{FFPROBE} -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "{path}"')
        return float(out.strip())
    except Exception:
        return 0.0

def _ensure_local(path_or_url: str, tmpdir: str) -> str:
    if path_or_url.startswith("s3://") or (path_or_url.startswith(("http://", "https://")) and ".s3." in path_or_url):
        return download_to_tmp(path_or_url, tmpdir)
    return path_or_url

def _normalize_input(local_in: str, tmpdir: str) -> str:
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
    cmd = (
        f'{FFMPEG} -y -hide_banner -loglevel error '
        f'-ss {start:.3f} -i "{src}" -t {dur:.3f} '
        f'-c:v libx264 -preset veryfast -crf 20 '
        f'-c:a aac -ar 48000 -ac 2 '
        f'-movflags +faststart "{out_path}"'
    )
    _run(cmd)

def _concat_mp4(parts: List[str], out_path: str):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in parts:
            f.write(f"file '{p}'\n")
        flist = f.name
    cmd = f'{FFMPEG} -y -hide_banner -loglevel error -f concat -safe 0 -i "{flist}" -c copy "{out_path}"'
    _run(cmd)

def _render_concat(tmpdir: str, files: List[str], portrait: bool=True) -> str:
    out = os.path.join(tmpdir, "out_concat.mp4")
    _concat_mp4(files, out)
    return out

def _render_clips(tmpdir: str, clips: List["Clip"], portrait: bool=True) -> str:
    parts = []
    for c in clips:
        seg = os.path.join(tmpdir, f"seg-{uuid.uuid4().hex}.mp4")
        _cut_snippet(c.src, c.start, max(0.1, c.dur), seg)
        parts.append(seg)
    out = os.path.join(tmpdir, "out_clips.mp4")
    _concat_mp4(parts, out)
    return out

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

# ---------- Silence→Speech gating (fast, default) ----------
def _silence_regions(path: str, noise_db: float = -35.0, min_silence: float = 0.35) -> List[Tuple[float, float]]:
    noise = f"{noise_db:+.0f}dB"
    cmd = (
        f'{FFMPEG} -hide_banner -nostats -i "{path}" '
        f'-af silencedetect=noise={noise}:d={min_silence:.2f} -f null -'
    )
    out = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    sil_starts, sil_ends = [], []
    for line in out.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start:\s*([0-9.]+)", line);   sil_starts += [float(m.group(1))] if m else []
        if "silence_end" in line:
            m = re.search(r"silence_end:\s*([0-9.]+)", line);     sil_ends   += [float(m.group(1))] if m else []
    pairs, si, ei = [], 0, 0
    while si < len(sil_starts) or ei < len(sil_ends):
        s = sil_starts[si] if si < len(sil_starts) else None
        e = sil_ends[ei]   if ei < len(sil_ends)   else None
        if s is not None and (e is None or s < e):
            if ei < len(sil_ends):
                pairs.append((s, sil_ends[ei])); ei += 1; si += 1
            else:
                break
        elif e is not None:
            pairs.append((0.0, e)); ei += 1
        else:
            break
    return pairs

def _speech_regions(path: str) -> List[Tuple[float, float]]:
    dur = _duration(path)
    if dur <= 0: return []
    sil = _silence_regions(path)
    if not sil:
        return [(0.0, dur)]
    sil = [(max(0.0, s), min(dur, e)) for s, e in sil if e > s]; sil.sort()
    speech, t = [], 0.0
    for (s, e) in sil:
        if s > t: speech.append((t, s))
        t = max(t, e)
    if t < dur: speech.append((t, dur))
    return [(max(0.0, a), b) for a, b in speech if b - a >= 0.15]

def _cap_total_duration(clips: List["Clip"], limit: Optional[float]) -> List["Clip"]:
    if not limit or float(limit) <= 0: return clips
    total, out, L = 0.0, [], float(limit)
    for c in clips:
        if total + c.dur <= L:
            out.append(c); total += c.dur
        else:
            remain = L - total
            if remain > 0.25:
                out.append(Clip(src=c.src, start=c.start, end=min(c.start + remain, c.end), score=c.score, label=c.label))
            break
    return out

def _pick_best_speech(files: List[str], tmpdir: str, max_duration: Optional[float],
                      take_top_k: Optional[int], min_clip: float, max_clip: float) -> Tuple[List[Clip], Dict]:
    if not files:
        return [], {}
    first_local = _ensure_local(files[0], tmpdir)
    norm = _normalize_input(first_local, tmpdir)
    dur = _duration(norm)
    if dur <= 0:
        return [], {}
    speech = _speech_regions(norm)
    target_total = float(max_duration) if (max_duration and float(max_duration) > 0) else 60.0
    k = int(take_top_k) if (take_top_k and int(take_top_k) > 0) else 6
    per = max(min_clip, min(max_clip, target_total / max(1, k)))

    clips: List[Clip] = []
    for (a, b) in speech:
        t = a
        while t + 0.25 < b and len(clips) < k * 2:
            end = min(b, t + per)
            clips.append(Clip(src=norm, start=t, end=end, score=1.0, label="speech"))
            t = end + 0.25
    if take_top_k:
        clips = clips[:k]
    clips = _cap_total_duration(clips, max_duration)
    cache = {"speech_regions": speech, "norm_duration": dur}
    return clips, cache

# ---------- (Optional) Whisper ASR when enabled ----------
def _asr_segments(path: str) -> List[Tuple[float, float, str]]:
    try:
        import whisper
    except Exception:
        return []
    model = whisper.load_model(ASR_MODEL_SIZE)
    result = model.transcribe(path, language=ASR_LANGUAGE)
    return [(float(seg.get("start",0.0)), float(seg.get("end",0.0)), seg.get("text","").strip())
            for seg in result.get("segments", []) if float(seg.get("end",0.0))>float(seg.get("start",0.0))]

def _order_funnel(clips: List[Clip], cache: Dict, max_duration: Optional[float]) -> List[Clip]:
    total = 0.0
    out: List[Clip] = []
    limit = float(max_duration) if (max_duration and float(max_duration) > 0) else 0.0
    for c in clips:
        if limit and total + c.dur > limit:
            break
        out.append(c); total += c.dur
    return out

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    sess = str(payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}")
    files = list(payload.get("files") or [])
    out_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")
    portrait = bool(payload.get("portrait", True))
    mode = str(payload.get("mode") or "funnel").lower().strip()

    min_clip = float(payload.get("min_clip_seconds") or 1.5)
    max_clip = float(payload.get("max_clip_seconds") or 4.0)
    max_duration = payload.get("max_duration")
    take_top_k = payload.get("take_top_k")

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{sess}-")
    try:
        if not files:
            return {"ok": False, "error": "No input files provided", "session_id": sess}

        local_norm: List[str] = []
        for f in files:
            loc = _ensure_local(f, tmpdir)
            norm = _normalize_input(loc, tmpdir)
            local_norm.append(norm)

        if mode in ("best", "best_funnel", "funnel"):
            clips, seg_cache = _pick_best_speech(files, tmpdir, max_duration, take_top_k, min_clip, max_clip)
            if not clips:
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
 
