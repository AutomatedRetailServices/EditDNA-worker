# jobs.py — ASR-aware “human-edit” picker: face+fluency scoring, whole-clip veto, funnel ordering

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

from s3_utils import upload_file, presigned_url, S3_BUCKET, download_to_tmp
from captions import write_srt, burn_captions

FFMPEG  = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# ---------- knobs (env) ----------
ASR_ENABLED   = os.getenv("ASR_ENABLED", "true").lower() in ("1","true","yes","on")
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANGUAGE   = os.getenv("ASR_LANG", "en")

BIN_SEC   = float(os.getenv("BIN_SEC", "0.5"))

# base weights
W_AUDIO   = float(os.getenv("W_AUDIO",   "0.30"))
W_SCENE   = float(os.getenv("W_SCENE",   "0.15"))
W_SPEECH  = float(os.getenv("W_SPEECH",  "0.30"))
W_FACE    = float(os.getenv("W_FACE",    "0.20"))
W_FLUENCY = float(os.getenv("W_FLUENCY", "0.35"))

# scene detect
SCENE_THRESH = float(os.getenv("SCENE_THRESH", "0.04"))

# face/gaze heuristics
FACE_MIN_SIZE    = float(os.getenv("FACE_MIN_SIZE", "0.08"))   # % of frame min dimension
FACE_CENTER_TOL  = float(os.getenv("FACE_CENTER_TOL", "0.35")) # normalized center distance tolerance

# fluency heuristics
FLUENCY_MIN_WPM         = float(os.getenv("FLUENCY_MIN_WPM", "95"))
FLUENCY_FILLER_PENALTY  = float(os.getenv("FLUENCY_FILLER_PENALTY", "0.65"))
FILLERS = set(["um","uh","erm","hmm","like","you know","sort of","kinda"]) # simple

# veto: if a file never reaches this blended (face+fluency) score, drop all clips from it
VETO_MIN_SCORE = float(os.getenv("VETO_MIN_SCORE", "0.40"))

# “human” smoothing
GRACE_SEC   = float(os.getenv("GRACE_SEC", "0.6"))   # allow small bad gaps inside a good region
MAX_BAD_SEC = float(os.getenv("MAX_BAD_SEC", "1.2")) # but cap them

# funnel bucket targets (Hook, Feature, Proof, CTA)
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
    # S3 direct or S3 http
    if path_or_url.startswith("s3://") or (path_or_url.startswith(("http://","https://")) and ".s3." in path_or_url):
        return download_to_tmp(path_or_url, tmpdir)

    # generic http(s)
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

    # create silent track if needed
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
    label: Optional[str] = None # for funnel bucket
    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ---------- analysis ----------
def _analyze_scene_markers(path: str) -> List[float]:
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", path,
           "-vf", f"select='gt(scene,{SCENE_THRESH})',showinfo",
           "-an", "-f", "null", "-"]
    log = _run(cmd)
    times: List[float] = []
    for line in log.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            m = re.search(r"pts_time:([0-9]+\.[0-9]+)", line)
            if m:
                times.append(float(m.group(1)))
    return times

def _analyze_audio_rms_bins(path: str, dur: float, bin_sec: float) -> List[float]:
    log = _run([FFMPEG, "-hide_banner", "-i", path, "-vn",
                "-af", f"astats=metadata=1:reset={bin_sec}", "-f", "null", "-"])
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

# ---------- ASR ----------
def _asr_segments(local_path: str) -> List[Tuple[float,float,str]]:
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
    out: List[Tuple[float,float,str]] = []
    for seg in segments:
        try:
            out.append((float(seg.start or 0.0), float(seg.end or 0.0), (seg.text or "").strip()))
        except Exception:
            pass
    return out

# ---------- human checks (face + fluency) ----------
def _face_ok(frame: np.ndarray) -> float:
    """Return 0..1 score based on face size and centeredness."""
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        return 0.0
    h, w = gray.shape[:2]
    mind = min(w, h)
    best = 0.0
    for (x,y,fw,fh) in faces:
        cx = x + fw/2
        cy = y + fh/2
        size_ratio = max(fw, fh) / (mind + 1e-6)
        center_dx = abs((cx - w/2)/ (w/2))
        center_dy = abs((cy - h/2)/ (h/2))
        center_dist = math.hypot(center_dx, center_dy)
        size_ok = 1.0 if size_ratio >= FACE_MIN_SIZE else size_ratio / max(FACE_MIN_SIZE,1e-6)
        center_ok = max(0.0, 1.0 - (center_dist / max(FACE_CENTER_TOL, 1e-6)))
        s = 0.5*size_ok + 0.5*center_ok
        best = max(best, s)
    return float(best)

def _fluency_metrics(text: str, dur: float) -> float:
    """Return 0..1 fluency score using WPM + filler penalty."""
    words = [w.strip(".,!?;:()[]\"'").lower() for w in text.split()]
    wcount = len([w for w in words if w])
    if dur <= 0.0:
        return 0.0
    wpm = (wcount / dur) * 60.0
    # filler hits
    filler_hits = sum(1 for w in words if w in FILLERS)
    base = 1.0 if wpm >= FLUENCY_MIN_WPM else max(0.0, wpm / max(FLUENCY_MIN_WPM,1e-6))
    penalty = (FLUENCY_FILLER_PENALTY ** filler_hits) if filler_hits > 0 else 1.0
    return float(max(0.0, min(1.0, base * penalty)))

def _grab_middle_frame(src: str, start: float, end: float, tmpdir: str) -> Optional[np.ndarray]:
    t = (start + end) / 2.0
    out = os.path.join(tmpdir, f"frm-{uuid.uuid4().hex}.jpg")
    try:
        subprocess.run([FFMPEG, "-y", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        img = cv2.imread(out)
        return img
    except Exception:
        return None
    finally:
        try:
            os.remove(out)
        except Exception:
            pass

# ---------- bins + candidates ----------
def _speech_bins_from_segments(segments: List[Tuple[float,float,str]], dur: float, bin_sec: float) -> List[float]:
    num_bins = int(math.ceil(dur / bin_sec))
    pres = [0.0]*num_bins
    wps  = [0.0]*num_bins
    for (s,e,text) in segments:
        s=max(0.0,s); e=max(s,e)
        words = len(text.split())
        seg_d = max(1e-6, e-s)
        seg_wps = words/seg_d
        b0 = int(s // bin_sec); b1 = int((e-1e-9)//bin_sec)
        for b in range(max(0,b0), min(num_bins-1,b1)+1):
            pres[b] = 1.0
            wps[b] += seg_wps
    max_wps = max(wps) or 1.0
    wps_norm = [v/max_wps for v in wps]
    return [0.6*pres[i] + 0.4*wps_norm[i] for i in range(num_bins)]

def _bins_for_file(local_path: str, bin_sec: float) -> List[Tuple[float,float,float]]:
    dur = _duration(local_path)
    if dur <= 0:
        return []
    num_bins = int(math.ceil(dur / bin_sec))
    spans = [(i*bin_sec, min(dur, (i+1)*bin_sec)) for i in range(num_bins)]
    scene_bins = _scene_counts_to_bins(_analyze_scene_markers(local_path), dur, bin_sec)
    audio_bins = _analyze_audio_rms_bins(local_path, dur, bin_sec)
    if ASR_ENABLED:
        segs = _asr_segments(local_path)
        speech_bins = _speech_bins_from_segments(segs, dur, bin_sec)
    else:
        segs = []
        speech_bins = [0.0]*num_bins
    scores = [W_AUDIO*audio_bins[i] + W_SCENE*scene_bins[i] + W_SPEECH*speech_bins[i] for i in range(num_bins)]
    return [(spans[i][0], spans[i][1], scores[i]) for i in range(num_bins)], segs

def _collect_candidate_clips(local_path: str, tmpdir: str, min_clip: float, max_clip: float) -> Tuple[List[Clip], List[Tuple[float,float,str]]]:
    bins, segs = _bins_for_file(local_path, BIN_SEC)
    if not bins:
        return [], segs
    base_scores = [s for _,_,s in bins]
    med = sorted(base_scores)[len(base_scores)//2] if base_scores else 0.0

    out: List[Clip] = []
    cur_start=None; cur_sum=0.0; cur_bins=0; cur_bad=0.0
    def flush(end_t: float):
        nonlocal cur_start, cur_sum, cur_bins, cur_bad
        if cur_start is None:
            return
        dur = end_t - cur_start
        if dur >= min_clip:
            chunks = int(math.ceil(dur / max_clip))
            chunk_d = dur / chunks
            base = (cur_sum / max(1, cur_bins))
            for k in range(chunks):
                s = cur_start + k*chunk_d
                e = min(end_t, s+chunk_d)
                if e - s >= min_clip:
                    out.append(Clip(src=local_path, start=s, end=e, score=base))
        cur_start=None; cur_sum=0.0; cur_bins=0; cur_bad=0.0

    # greedily group >= median score, allow short “bad” holes up to MAX_BAD_SEC
    for (b0,b1,sc) in bins:
        if sc >= med:
            if cur_start is None:
                cur_start = b0
            cur_sum += sc
            cur_bins += 1
            cur_bad = 0.0
        else:
            if cur_start is not None:
                gap = b1 - b0
                cur_bad += gap
                if cur_bad > min(MAX_BAD_SEC, GRACE_SEC):
                    flush(b0)
    if bins:
        flush(bins[-1][1])

    # Human filters: face + fluency re-score each candidate
    rescored: List[Clip] = []
    for c in out:
        frame = _grab_middle_frame(local_path, c.start, c.end, tmpdir)
        face_s = _face_ok(frame)
        # text during that span
        text = " ".join(t for (s,e,t) in segs if not (e <= c.start or s >= c.end))
        flu_s  = _fluency_metrics(text, max(1e-6, c.dur))
        human = (W_FACE*face_s + W_FLUENCY*flu_s)
        c.score = 0.6*c.score + 0.4*human
        rescored.append(c)

    return rescored, segs

# ---------- whole-clip veto ----------
def _veto_whole_clip(clips: List[Clip], segs: List[Tuple[float,float,str]]) -> bool:
    """Return True if this source should be dropped entirely."""
    # If no clips survived or all have tiny scores, veto.
    if not clips:
        return True
    best = max(c.score for c in clips)
    return bool(best < VETO_MIN_SCORE)

# ---------- rendering ----------
def _render_concat(tmpdir: str, inputs: List[str], portrait: bool = True) -> str:
    out_path = os.path.join(tmpdir, "out.mp4")
    concat_txt = _safe_concat_list(tmpdir, inputs)
    vf = ("scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
          if portrait else
          "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black")
    cmd = [
        FFMPEG, "-y",
        "-analyzeduration", "500M", "-probesize", "500M",
        "-safe", "0", "-f", "concat", "-i", concat_txt,
        "-ignore_unknown",
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
    vf = ("scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
          if portrait else
          "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black")
    for i,c in enumerate(clips):
        part = os.path.join(tmpdir, f"part_{i:03d}.mp4")
        _run([
            FFMPEG, "-y",
            "-ss", f"{c.start:.3f}", "-to", f"{c.end:.3f}",
            "-i", c.src,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", part
        ])
        parts.append(part)
    return _render_concat(tmpdir, parts, portrait=portrait)

# ---------- funnel helpers ----------
def _score_text_for_bucket(text: str, kws: List[str]) -> float:
    t = " " .join(re.findall(r"[a-zA-Z]+", text.lower()))
    if not t:
        return 0.0
    s=0
    for k in kws:
        if k in t:
            s += 1
    return float(min(1.0, s / max(1,len(kws))))

def _order_funnel(clips: List[Clip],
                  seg_cache: Dict[str, List[Tuple[float,float,str]]],
                  want_total: Optional[float]) -> List[Clip]:
    # label clips by bucket
    buckets: Dict[str, List[Clip]] = {"hook":[],"feature":[],"proof":[],"cta":[]}
    for c in clips:
        segs = seg_cache.get(c.src, [])
        text = " ".join(t for (s,e,t) in segs if not (e<=c.start or s>=c.end))
        sh = _score_text_for_bucket(text, HOOK_KWS)
        sf = _score_text_for_bucket(text, FEATURE_KWS)
        sp = _score_text_for_bucket(text, PROOF_KWS)
        sc = _score_text_for_bucket(text, CTA_KWS)
        label = max([("hook",sh),("feature",sf),("proof",sp),("cta",sc)], key=lambda x:x[1])[0]
        c.label = label
        buckets[label].append(c)

    for k in buckets.keys():
        buckets[k].sort(key=lambda c: c.score, reverse=True)

    pick: List[Clip] = []
    def take(k, n):
        arr = buckets[k]
        if n<=0 or not arr: return
        take_n = min(n, len(arr))
        pick.extend(arr[:take_n])

    take("hook", FN_HOOK)
    take("feature", FN_FEATURE)
    take("proof", FN_PROOF)
    take("cta", FN_CTA)

    # Optional trim to total runtime
    if want_total and want_total > 0:
        out=[]
        acc=0.0
        for c in pick:
            if acc + c.dur <= want_total + 1e-3:
                out.append(c); acc += c.dur
            else:
                break
        pick = out

    return pick

# ---------- public jobs ----------
def _pick_best(files: List[str], tmpdir: str,
               max_duration: Optional[int], take_top_k: Optional[int],
               min_clip: float, max_clip: float) -> Tuple[List[Clip], Dict[str,List[Tuple[float,float,str]]]]:
    all_candidates: List[Clip] = []
    seg_cache: Dict[str, List[Tuple[float,float,str]]] = {}
    for f in files:
        local_f = _ensure_local(f, tmpdir)
        safe_f  = _normalize_input(local_f, tmpdir)
        clips, segs = _collect_candidate_clips(safe_f, tmpdir, min_clip, max_clip)
        seg_cache[safe_f] = segs
        # veto whole source if never reaches threshold
        if _veto_whole_clip(clips, segs):
            continue
        all_candidates.extend(clips)

    if not all_candidates:
        return [], seg_cache

    all_candidates.sort(key=lambda c: c.score, reverse=True)
    if take_top_k and take_top_k > 0:
        all_candidates = all_candidates[:take_top_k]

    if max_duration and max_duration > 0:
        chosen=[]
        acc=0.0
        for c in all_candidates:
            if acc + c.dur <= max_duration + 1e-3:
                chosen.append(c); acc += c.dur
            if acc >= max_duration:
                break
        all_candidates = chosen

    return all_candidates, seg_cache

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

        if with_captions and mode_used in ("best","funnel","concat","concat_fallback") and ASR_ENABLED:
            # burn captions from final output ASR
            segs = _asr_segments(out_local)
            if segs:
                srt_path = os.path.join(tmpdir, "subs.srt")
                write_srt(segs, srt_path)
                cap_out = os.path.join(tmpdir, "out_captions.mp4")
                burn_captions(out_local, srt_path, cap_out)
                out_local = cap_out

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
        }
    except Exception as e:
        return {"ok": False, "session_id": sess, "error": str(e), "inputs": files}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    return job_render(payload)

