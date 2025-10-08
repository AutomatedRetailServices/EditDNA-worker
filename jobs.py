# jobs.py — robust render pipeline with optional Whisper ASR + semantic chaining
from __future__ import annotations

# ---------- semantic import (safe + flexible) ----------
def _noop_continuity(takes):
    # fallback: return each take alone
    return [[t] for t in takes]

try:
    # preferred when repo root is PYTHONPATH and worker/ is a package
    from worker.semantic_visual_pass import continuity_chains  # type: ignore
    print("[jobs.py] Semantic pass loaded (package).", flush=True)
except Exception as e1:
    try:
        # fallback if file sits next to jobs.py or PYTHONPATH points at worker/
        from semantic_visual_pass import continuity_chains  # type: ignore
        print("[jobs.py] Semantic pass loaded (flat).", flush=True)
    except Exception as e2:
        print(f"[jobs.py] Semantic pass unavailable ({e1 or e2}); using no-op.", flush=True)
        continuity_chains = _noop_continuity  # type: ignore

# ---------- std imports ----------
import os, uuid, shutil, tempfile, subprocess, shlex, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------- optional ASR ----------
ASR_ENABLED     = os.getenv("ASR_ENABLED", "0").lower() in ("1","true","yes","on")
ASR_MODEL_SIZE  = os.getenv("ASR_MODEL_SIZE", "base")
ASR_LANGUAGE    = os.getenv("ASR_LANG", "en")

try:
    import whisper  # only used if ASR_ENABLED=1
except Exception:
    whisper = None  # type: ignore

# ---------- ffmpeg paths ----------
FFMPEG  = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# ---------- S3 helpers ----------
from s3_utils import upload_file, presigned_url, S3_BUCKET, download_to_tmp

# ======================================================
#                    low-level utils
# ======================================================
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
    # S3 or S3-HTTP → download to tmp; otherwise assume local file path
    if path_or_url.startswith("s3://") or (path_or_url.startswith(("http://", "https://")) and ".s3." in path_or_url):
        return download_to_tmp(path_or_url, tmpdir)
    return path_or_url

def _normalize_input(local_in: str, tmpdir: str) -> str:
    """
    Re-mux/transcode to a sane baseline for stable cutting/concat.
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
    cmd = (
        f'{FFMPEG} -y -hide_banner -loglevel error '
        f'-ss {start:.3f} -i "{src}" -t {max(0.1, dur):.3f} '
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

# ======================================================
#                 render helpers (concat/clips)
# ======================================================
def _render_concat(tmpdir: str, files: List[str]) -> str:
    out = os.path.join(tmpdir, "out_concat.mp4")
    _concat_mp4(files, out)
    return out

def _render_clips(tmpdir: str, clips: List["Clip"]) -> str:
    parts = []
    for c in clips:
        seg = os.path.join(tmpdir, f"seg-{uuid.uuid4().hex}.mp4")
        _cut_snippet(c.src, c.start, c.dur, seg)
        parts.append(seg)
    out = os.path.join(tmpdir, "out_clips.mp4")
    _concat_mp4(parts, out)
    return out

# ======================================================
#                       clip model
# ======================================================
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

# ======================================================
#          speech/silence segmentation (fast path)
# ======================================================
def _silence_regions(path: str, noise_db: float = -35.0, min_silence: float = 0.35) -> List[Tuple[float, float]]:
    """
    Returns [(silence_start, silence_end), ...]
    """
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
    # drop too-short gaps
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
    """
    Simple one-file best-speech picker. For multi-file, normalize/merge first or iterate.
    """
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
    cache = {"speech_regions": speech, "norm_duration": dur, "norm_path": norm}
    return clips, cache

# ======================================================
#                 (Optional) Whisper ASR
# ======================================================
def _asr_segments(path: str) -> List[Tuple[float, float, str]]:
    """
    Returns [(start, end, text), ...] using Whisper if ASR_ENABLED=1 and whisper available.
    """
    if not ASR_ENABLED or whisper is None:
        return []
    model = whisper.load_model(ASR_MODEL_SIZE)
    result = model.transcribe(path, language=ASR_LANGUAGE)
    out = []
    for seg in result.get("segments", []):
        a = float(seg.get("start", 0.0))
        b = float(seg.get("end", 0.0))
        if b > a:
            out.append((a, b, seg.get("text", "").strip()))
    return out

# ======================================================
#                   clip ordering strategy
# ======================================================
def _order_funnel(clips: List[Clip], max_duration: Optional[float]) -> List[Clip]:
    total = 0.0
    out: List[Clip] = []
    limit = float(max_duration) if (max_duration and float(max_duration) > 0) else 0.0
    for c in clips:
        if limit and total + c.dur > limit:
            break
        out.append(c); total += c.dur
    return out

# ======================================================
#                        main job
# ======================================================
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render entrypoint. Supports modes:
      - "concat"      → just concatenates normalized inputs
      - "best"        → pick best speech chunks
      - "funnel"      → best + simple ordering (default)
      - "best_funnel" → alias for funnel
    Optional semantic chaining applied on top when available (groups contiguous takes).
    """
    sess = str(payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}")
    files = list(payload.get("files") or [])
    out_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")
    mode = str(payload.get("mode") or "funnel").lower().strip()

    min_clip = float(payload.get("min_clip_seconds") or 1.5)
    max_clip = float(payload.get("max_clip_seconds") or 4.0)
    max_duration = payload.get("max_duration")
    take_top_k = payload.get("take_top_k")

    print(f"[jobs.py] Start job session={sess} mode={mode} files={len(files)}", flush=True)

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{sess}-")
    try:
        if not files:
            return {"ok": False, "error": "No input files provided", "session_id": sess}

        # Normalize inputs (for concat fallback or explicit concat)
        local_norm: List[str] = []
        for f in files:
            loc = _ensure_local(f, tmpdir)
            norm = _normalize_input(loc, tmpdir)
            local_norm.append(norm)

        # Main path selection
        if mode in ("best", "best_funnel", "funnel"):
            # Fast speech-gating
            clips, seg_cache = _pick_best_speech(files, tmpdir, max_duration, take_top_k, min_clip, max_clip)

            # Optional ASR → refine with text + semantic continuity
            asr_takes: List[Dict[str, Any]] = []
            if ASR_ENABLED and local_norm:
                # run ASR on the first normalized asset
                segs = _asr_segments(local_norm[0])  # [(a,b,text)]
                for (a, b, txt) in segs:
                    asr_takes.append({"text": txt, "start": a, "end": b})

            # If ASR gave us text, let semantic continuity regroup;
            # otherwise, pass silent "takes" derived from clips (no text).
            if asr_takes:
                print(f"[jobs.py] ASR segments: {len(asr_takes)} → semantic continuity", flush=True)
                chains = continuity_chains([type("T", (), t) for t in asr_takes])
                # flatten back into clip slices following chain order (cap with max_duration)
                flattened: List[Clip] = []
                for chain in chains:
                    for t in chain:
                        flattened.append(Clip(src=seg_cache.get("norm_path", local_norm[0]),
                                              start=float(getattr(t, "start", 0.0)),
                                              end=float(getattr(t, "end", 0.0)),
                                              score=1.0, label="asr"))
                clips = _cap_total_duration(flattened, max_duration) or clips  # fallback to speech if empty
            else:
                # no text → at least group consecutive clips (semantic pass will no-op if not present)
                pseudo_takes = [{"text": "", "start": c.start, "end": c.end} for c in clips]
                _ = continuity_chains([type("T", (), t) for t in pseudo_takes])  # returns chains; we don't alter clips here

            if not clips:
                print("[jobs.py] No clips from speech/ASR → concat fallback.", flush=True)
                out_local = _render_concat(tmpdir, local_norm)
                mode_used = "concat_fallback"
            else:
                ordered = _order_funnel(clips, max_duration)
                out_local = _render_clips(tmpdir, ordered)
                mode_used = "funnel" if mode in ("best_funnel", "funnel") else "best"

        elif mode == "concat":
            out_local = _render_concat(tmpdir, local_norm)
            mode_used = "concat"

        else:
            # unknown mode → fallback to concat
            out_local = _render_concat(tmpdir, local_norm)
            mode_used = "concat"

        # Upload and presign
        s3_uri = upload_file(out_local, f"{out_prefix}/{sess}", content_type="video/mp4")
        _, key = s3_uri.replace("s3://", "", 1).split("/", 1)
        url = presigned_url(S3_BUCKET, key, expires=3600)

        result = {
            "ok": True,
            "session_id": sess,
            "mode": mode_used,
            "output_s3": s3_uri,
            "output_url": url,
            "inputs": files,
        }
        print(f"[jobs.py] Done: {result['output_s3']}", flush=True)
        return result

    except Exception as e:
        print(f"[jobs.py] ERROR: {e}", flush=True)
        return {"ok": False, "session_id": sess, "error": str(e), "inputs": files}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# Back-compat alias expected by your web API
def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    return job_render(payload)
