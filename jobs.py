# jobs.py — ASR-aware “best clips” picker + optional captions + renderer (robust audio + funnel ordering)

from __future__ import annotations

import os
import re
import math
import uuid
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

from s3_utils import upload_file, presigned_url, S3_BUCKET, download_to_tmp
from captions import write_srt, burn_captions

FFMPEG  = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# ---------- config toggles ----------
ASR_ENABLED   = os.getenv("ASR_ENABLED", "true").lower() in ("1", "true", "yes", "on")
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny")  # tiny | base | small
ASR_LANGUAGE   = os.getenv("ASR_LANG", "en")

# Weights for scoring components (sum ~1.0)
W_AUDIO  = float(os.getenv("W_AUDIO",  "0.45"))  # loudness
W_SCENE  = float(os.getenv("W_SCENE",  "0.15"))  # motion/cuts
W_SPEECH = float(os.getenv("W_SPEECH", "0.40"))  # speech presence/density

# Scene detection sensitivity & analysis bin size (seconds)
SCENE_THRESH = float(os.getenv("SCENE_THRESH", "0.04"))
BIN_SEC      = float(os.getenv("BIN_SEC", "0.5"))

# Where to create session/temp folders (bind-mounted on the VM)
SESSION_ROOT = os.getenv("SESSION_ROOT", "/sessions")
if not os.path.isdir(SESSION_ROOT):
    SESSION_ROOT = "/tmp"
os.makedirs(SESSION_ROOT, exist_ok=True)

# ---------- funnel keyword config (simple, extend anytime) ----------
HOOK_KWS    = ("stop scrolling", "watch this", "attention", "listen", "hook", "big news", "did you know", "you won’t believe", "you will not believe", "hack", "tip", "secret")
FEATURE_KWS = ("it has", "it does", "feature", "includes", "comes with", "built-in", "spec", "specs", "function", "works by", "how it works")
PROOF_KWS   = ("i tried", "i tested", "before and after", "before/after", "review", "unboxing", "demo", "results", "it worked", "testimonial", "my experience")
CTA_KWS     = ("buy now", "link in bio", "shop now", "add to cart", "use code", "discount", "limited time", "order now", "grab yours")

# ---------- utils ----------
def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n---\n{p.stdout}\n---"
        )
    return p.stdout

def _duration(path: str) -> float:
    out = _run([
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ])
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
    # S3 or S3-hosted https
    if path_or_url.startswith("s3://") or (
        path_or_url.startswith(("http://", "https://")) and ".s3." in path_or_url
    ):
        return download_to_tmp(path_or_url, tmpdir)

    # generic https -> copy locally via ffmpeg (robust against CORS/range)
    if path_or_url.startswith(("http://", "https://")):
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
            # last resort: just remux video, drop audio; we’ll add silent later
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

    # already local
    return path_or_url

def _normalize_input(local_in: str, tmpdir: str) -> str:
    """
    Make every input safe/consistent for concat:
      - H.264 video (keeps original fps/size)
      - Stereo AAC 48k. If source audio is broken, auto-creates SILENT audio.
    """
    dur = max(0.0, _duration(local_in))
    base = os.path.join(tmpdir, f"norm-{uuid.uuid4().hex}.mp4")

    # 1) Try with real audio
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

    # 2) If audio decode failed, add a SILENT track
    silent = os.path.join(tmpdir, f"sil-{uuid.uuid4().hex}.wav")
    if dur <= 0.0:
        dur = 3600.0  # absurdly high cap; we'll -shortest to video anyway
    _run([
        FFMPEG, "-y",
        "-f", "lavfi", "-t", f"{dur:.3f}",
        "-i", "anullsrc=r=48000:cl=stereo",
        "-c:a", "pcm_s16le",
        silent
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
    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ---------- analysis (scene + audio) ----------
def _analyze_scene_markers(path: str) -> List[float]:
    cmd = [
        FFMPEG, "-hide_banner", "-nostats", "-i", path,
        "-vf", f"select='gt(scene,{SCENE_THRESH})',showinfo",
        "-an", "-f", "null", "-"
    ]
    log = _run(cmd)
    times: List[float] = []
    for line in log.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            m = re.search(r"pts_time:([0-9]+\.[0-9]+)", line)
            if m:
                times.append(float(m.group(1)))
    return times

def _analyze_audio_rms_bins(path: str, dur: float, bin_sec: float) -> List[float]:
    a_cmd = [
        FFMPEG, "-hide_banner", "-i", path,
        "-vn", "-af", f"astats=metadata=1:reset={bin_sec}",
        "-f", "null", "-"
    ]
    log = _run(a_cmd)
    rms_db: List[float] = []
    for line in log.splitlines():
        if "RMS_level:" in line:
            m = re.search(r"RMS_level:\s*(-?\d+\.?\d*)", line)
            if m:
                try:
                    rms_db.append(float(m.group(1)))
                except Exception:
                    pass

    num_bins = int(math.ceil(dur / bin_sec))
    if len(rms_db) < num_bins:
        rms_db += [min(rms_db + [-60.0])] * (num_bins - len(rms_db))
    if len(rms_db) > num_bins:
        rms_db = rms_db[:num_bins]

    loud_vals = [max(0.0, 60.0 + v) for v in rms_db]
    max_loud = max(loud_vals) or 1.0
    return [v / max_loud for v in loud_vals]

def _scene_counts_to_bins(scene_times: List[float], dur: float, bin_sec: float) -> List[float]:
    num_bins = int(math.ceil(dur / bin_sec))
    counts = [0] * num_bins
    for t in scene_times:
        idx = min(int(t / bin_sec), num_bins - 1)
        counts[idx] += 1
    max_cnt = max(counts) or 1
    return [c / max_cnt for c in counts]

# ---------- ASR (speech-aware) ----------
def _asr_segments(local_path: str) -> List[Tuple[float, float, str]]:
    if not ASR_ENABLED:
        return []
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return []
    model = WhisperModel(ASR_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        local_path,
        language=ASR_LANGUAGE if ASR_LANGUAGE else None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=False,
    )
    out: List[Tuple[float, float, str]] = []
    for seg in segments:
        try:
            out.append((float(seg.start or 0.0), float(seg.end or 0.0), (seg.text or "").strip()))
        except Exception:
            pass
    return out

def _speech_bins_from_segments(segments: List[Tuple[float, float, str]], dur: float, bin_sec: float) -> List[float]:
    num_bins = int(math.ceil(dur / bin_sec))
    pres = [0.0] * num_bins
    wps  = [0.0] * num_bins
    for (s, e, text) in segments:
        s = max(0.0, s); e = max(s, e)
        wcount = max(0, len(text.split()))
        seg_dur = max(1e-6, e - s)
        seg_wps = wcount / seg_dur
        b0 = int(s // bin_sec)
        b1 = int((e - 1e-9) // bin_sec)
        for b in range(max(0, b0), min(num_bins - 1, b1) + 1):
            pres[b] = 1.0
            wps[b] += seg_wps
    max_wps = max(wps) or 1.0
    wps_norm = [v / max_wps for v in wps]
    return [0.6 * pres[i] + 0.4 * wps_norm[i] for i in range(num_bins)]

# ---------- combine to bins ----------
def _bins_for_file(local_path: str, bin_sec: float) -> List[Tuple[float, float, float]]:
    dur = _duration(local_path)
    if dur <= 0:
        return []
    num_bins = int(math.ceil(dur / bin_sec))
    bins = [(i * bin_sec, min(dur, (i + 1) * bin_sec)) for i in range(num_bins)]
    scene_bins = _scene_counts_to_bins(_analyze_scene_markers(local_path), dur, bin_sec)
    audio_bins = _analyze_audio_rms_bins(local_path, dur, bin_sec)
    if ASR_ENABLED:
        segments = _asr_segments(local_path)
        speech_bins = _speech_bins_from_segments(segments, dur, bin_sec)
    else:
        speech_bins = [0.0] * num_bins
    scores = [
        W_AUDIO * audio_bins[i] + W_SCENE * scene_bins[i] + W_SPEECH * speech_bins[i]
        for i in range(num_bins)
    ]
    return [(bins[i][0], bins[i][1], scores[i]) for i in range(num_bins)]

# ---------- collect candidate clips ----------
@dataclass
class Clip:
    src: str
    start: float
    end: float
    score: float
    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

def _collect_candidate_clips(local_path: str, min_clip: float, max_clip: float) -> List[Clip]:
    bins = _bins_for_file(local_path, BIN_SEC)
    if not bins:
        return []
    scores = [s for _, _, s in bins]
    thresh = sorted(scores)[len(scores)//2]
    out: List[Clip] = []
    cur_start: Optional[float] = None
    cur_score_sum = 0.0
    cur_bins = 0
    def flush(end_time: float):
        nonlocal cur_start, cur_score_sum, cur_bins
        if cur_start is None:
            return
        dur = end_time - cur_start
        if dur >= min_clip:
            chunks = int(math.ceil(dur / max_clip))
            chunk_dur = dur / chunks
            base_score = (cur_score_sum / max(1, cur_bins))
            for k in range(chunks):
                s = cur_start + k * chunk_dur
                e = min(end_time, s + chunk_dur)
                if e - s >= min_clip:
                    out.append(Clip(src=local_path, start=s, end=e, score=base_score))
        cur_start = None
        cur_score_sum = 0.0
        cur_bins = 0
    for (b0, b1, sc) in bins:
        if sc >= thresh:
            if cur_start is None:
                cur_start = b0
            cur_score_sum += sc
            cur_bins += 1
        else:
            flush(b0)
    flush(bins[-1][1])
    return out

# ---------- rendering ----------
def _render_concat(tmpdir: str, inputs: List[str], portrait: bool = True) -> str:
    out_path = os.path.join(tmpdir, "out.mp4")
    concat_txt = _safe_concat_list(tmpdir, inputs)
    vf = (
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
        if portrait else
        "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    # -ignore_unknown is a flag (no value)
    cmd = [
        FFMPEG, "-y",
        "-analyzeduration", "500M", "-probesize", "500M",
        "-f", "concat",
        "-safe", "0",
        "-ignore_unknown",
        "-i", concat_txt,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    _run(cmd)
    return out_path

def _render_clips(tmpdir: str, clips: List[Clip], portrait: bool = True) -> str:
    parts: List[str] = []
    vf = (
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
        if portrait else
        "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    for i, c in enumerate(clips):
        part = os.path.join(tmpdir, f"part_{i:03d}.mp4")
        _run([
            FFMPEG, "-y",
            "-ss", f"{c.start:.3f}", "-to", f"{c.end:.3f}",
            "-i", c.src,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            part
        ])
        parts.append(part)
    return _render_concat(tmpdir, parts, portrait=portrait)

# ---------- funnel helpers ----------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _clip_text_for(src: str, s: float, e: float, segs: List[Tuple[float, float, str]]) -> str:
    """Return ASR text overlapping [s,e] with a small padding window."""
    pad = 0.25
    s -= pad; e += pad
    bits: List[str] = []
    for (ts, te, txt) in segs:
        if te >= s and ts <= e and txt:
            bits.append(txt)
    return _normalize_text(" ".join(bits))

def _tag_from_text(txt: str) -> str:
    t = _normalize_text(txt)
    def has_any(toks: Tuple[str, ...]) -> bool:
        return any(k in t for k in toks)
    if has_any(HOOK_KWS):    return "hook"
    if has_any(FEATURE_KWS): return "feature"
    if has_any(PROOF_KWS):   return "proof"
    if has_any(CTA_KWS):     return "cta"
    return "other"

def _order_funnel(clips: List[Clip],
                  seg_cache: Dict[str, List[Tuple[float, float, str]]],
                  want_total: Optional[float]) -> List[Clip]:
    """Bucket clips by tag (hook -> feature -> proof -> cta), then fill in order, with fallbacks."""
    buckets: DefaultDict[str, List[Clip]] = defaultdict(list)
    # Tag each clip by the text in its window
    for c in clips:
        segs = seg_cache.get(c.src, [])
        txt  = _clip_text_for(c.src, c.start, c.end, segs) if segs else ""
        tag  = _tag_from_text(txt) if txt else "other"
        buckets[tag].append(c)

    # Sort each bucket by score desc
    for k in buckets:
        buckets[k].sort(key=lambda x: x.score, reverse=True)

    order = ["hook", "feature", "proof", "cta"]
    chosen: List[Clip] = []
    total = 0.0

    # First pass: pick best from each bucket in order
    for k in order:
        if buckets[k]:
            c = buckets[k].pop(0)
            chosen.append(c)
            total += c.dur

    # Fill remaining time with remaining clips preferring funnel order, then "other"
    remaining = []
    for k in order:
        remaining.extend(buckets[k])
    remaining.extend(sorted(buckets.get("other", []), key=lambda x: x.score, reverse=True))

    for c in remaining:
        if want_total and total + c.dur > want_total + 1e-3:
            continue
        chosen.append(c)
        total += c.dur
        if want_total and total >= want_total:
            break

    # If nothing got picked (e.g., no ASR), fall back to original scoring order
    if not chosen and clips:
        chosen = sorted(clips, key=lambda x: x.score, reverse=True)

    return chosen

# ---------- public jobs ----------
def _pick_best(files: List[str],
               tmpdir: str,
               max_duration: Optional[int],
               take_top_k: Optional[int],
               min_clip: float,
               max_clip: float) -> List[Clip]:
    all_candidates: List[Clip] = []
    for f in files:
        local_f = _ensure_local(f, tmpdir)
        safe_f  = _normalize_input(local_f, tmpdir)
        all_candidates.extend(_collect_candidate_clips(safe_f, min_clip, max_clip))
    if not all_candidates:
        return []
    all_candidates.sort(key=lambda c: c.score, reverse=True)
    if take_top_k and take_top_k > 0:
        all_candidates = all_candidates[:take_top_k]
    if max_duration and max_duration > 0:
        chosen: List[Clip] = []
        acc = 0.0
        for c in all_candidates:
            if acc + c.dur <= max_duration + 1e-3:
                chosen.append(c)
                acc += c.dur
            if acc >= max_duration:
                break
        all_candidates = chosen
    return all_candidates

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload keys:
      session_id, files, output_prefix, portrait, mode,
      max_duration, take_top_k, min_clip_seconds, max_clip_seconds,
      with_captions
    """
    sess = str(payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}")
    files = list(payload.get("files") or [])
    out_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")
    portrait = bool(payload.get("portrait", True))
    mode = str(payload.get("mode") or "concat").lower().strip()
    with_captions = bool(payload.get("with_captions", False))

    max_duration = payload.get("max_duration")
    take_top_k = payload.get("take_top_k")
    min_clip = float(payload.get("min_clip_seconds") or 2.5)
    max_clip = float(payload.get("max_clip_seconds") or 10.0)

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{sess}-", dir=SESSION_ROOT)
    try:
        if not files:
            return {"ok": False, "error": "No input files provided", "session_id": sess}

        # Ensure-local + normalize all inputs up front for concat/best/funnel
        local_norm: List[str] = []
        for f in files:
            loc = _ensure_local(f, tmpdir)
            norm = _normalize_input(loc, tmpdir)
            local_norm.append(norm)

        if mode in ("best", "best_funnel", "funnel"):
            clips = _pick_best(files, tmpdir, max_duration, take_top_k, min_clip, max_clip)

            # If funnel requested and ASR available, reorder by funnel
            if mode in ("best_funnel", "funnel") and ASR_ENABLED:
                # build ASR cache once per source
                seg_cache: Dict[str, List[Tuple[float, float, str]]] = {}
                for path in sorted({c.src for c in clips}):
                    seg_cache[path] = _asr_segments(path)
                clips = _order_funnel(clips, seg_cache, float(max_duration) if max_duration else None)

            if not clips:
                out_local = _render_concat(tmpdir, local_norm, portrait=portrait)
                mode_used = "concat_fallback"
            else:
                out_local = _render_clips(tmpdir, clips, portrait=portrait)
                mode_used = "funnel" if mode in ("best_funnel", "funnel") else "best"

        else:
            # simple concat
            out_local = _render_concat(tmpdir, local_norm, portrait=portrait)
            mode_used = "concat"

        # Optional captions (after assembly)
        if with_captions and mode_used in ("best", "funnel", "concat", "concat_fallback") and ASR_ENABLED:
            segments = _asr_segments(out_local)
            if segments:
                srt_path = os.path.join(tmpdir, "subs.srt")
                write_srt(segments, srt_path)
                cap_out = os.path.join(tmpdir, "out_captions.mp4")
                burn_captions(out_local, srt_path, cap_out)
                out_local = cap_out

        # Upload to S3 and presign — keep outputs grouped by session
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
            "captions": bool(with_captions),
        }
    except Exception as e:
        return {"ok": False, "session_id": sess, "error": str(e), "inputs": files}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    return job_render(payload)
