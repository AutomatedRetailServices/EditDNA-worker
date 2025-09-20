# jobs.py — robust "best clips" picker with speech & visual analysis

import os
import re
import math
import uuid
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from s3_utils import upload_file, presigned_url, S3_BUCKET

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# Tunables (env overrides)
BIN_SEC = float(os.getenv("BIN_SEC", "0.5"))
SCENE_THRESH = float(os.getenv("SCENE_THRESH", "0.04"))
SILENCE_DB = float(os.getenv("SILENCE_DB", "-35"))     # silence below this dB
SILENCE_MIN = float(os.getenv("SILENCE_MIN", "0.5"))    # min silence to count
BLACK_D = float(os.getenv("BLACK_D", "0.08"))           # min black duration
BLACK_PIC_TH = float(os.getenv("BLACK_PIC_TH", "0.98"))
BLACK_PIX_TH = float(os.getenv("BLACK_PIX_TH", "0.10"))

# Weights for scoring
W_SPEECH = float(os.getenv("W_SPEECH", "0.55"))
W_LOUD = float(os.getenv("W_LOUD", "0.25"))
W_SCENE = float(os.getenv("W_SCENE", "0.20"))
BLACK_PENALTY = float(os.getenv("BLACK_PENALTY", "0.75"))  # multiply score when bin overlaps black


# ---------- shell helpers ----------

def _run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n---\n{p.stdout}\n---")
    return p.stdout

def _duration(path: str) -> float:
    out = _run([
        FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1", path
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
            f.write("file '{}'\n".format(safe))
    return list_path


# ---------- data ----------

@dataclass
class Clip:
    src: str
    start: float
    end: float
    score: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


# ---------- analysis passes ----------

def _scene_change_times(path: str) -> List[float]:
    cmd = [
        FFMPEG, "-hide_banner", "-nostats", "-i", path,
        "-vf", f"select='gt(scene,{SCENE_THRESH})',showinfo",
        "-an", "-f", "null", "-"
    ]
    log = _run(cmd)
    times: List[float] = []
    for line in log.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            m = re.search(r"pts_time:([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                times.append(float(m.group(1)))
    return times

def _audio_rms_bins(path: str, dur: float) -> List[float]:
    """
    Returns per-BIN_SEC RMS dB list. Uses astats with reset window = BIN_SEC.
    """
    cmd = [
        FFMPEG, "-hide_banner", "-i", path, "-vn",
        "-af", f"astats=metadata=1:reset={BIN_SEC}",
        "-f", "null", "-"
    ]
    log = _run(cmd)
    rms: List[float] = []
    for line in log.splitlines():
        if "RMS_level:" in line:
            m = re.search(r"RMS_level:\s*(-?\d+\.?\d*)", line)
            if m:
                try:
                    rms.append(float(m.group(1)))
                except Exception:
                    pass

    # align to bins
    n_bins = max(1, int(math.ceil(dur / BIN_SEC)))
    if len(rms) < n_bins:
        # pad with quiet values
        pad = [-60.0] * (n_bins - len(rms))
        rms += pad
    elif len(rms) > n_bins:
        rms = rms[:n_bins]
    return rms

def _speech_intervals(path: str) -> List[Tuple[float, float]]:
    """
    Invert silencedetect -> speech intervals.
    """
    cmd = [
        FFMPEG, "-hide_banner", "-i", path, "-vn",
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}",
        "-f", "null", "-"
    ]
    log = _run(cmd)
    # Parse silence_start / silence_end
    silences: List[Tuple[float, float]] = []
    s_start: Optional[float] = None
    for line in log.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                s_start = float(m.group(1))
        elif "silence_end:" in line and "silence_duration" in line:
            m = re.search(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if s_start is not None and m:
                silences.append((s_start, float(m.group(1))))
                s_start = None
    # Convert to speech by subtracting silences from [0, dur]
    dur = _duration(path)
    if dur <= 0:
        return []
    speech: List[Tuple[float, float]] = []
    cur = 0.0
    for (s0, s1) in sorted(silences):
        if s0 > cur:
            speech.append((cur, max(cur, min(s0, dur))))
        cur = max(cur, s1)
    if cur < dur:
        speech.append((cur, dur))
    # Clip negatives / tiny artifacts
    speech = [(max(0.0, a), max(0.0, b)) for (a, b) in speech if b - a > 1e-3]
    return speech

def _black_intervals(path: str) -> List[Tuple[float, float]]:
    cmd = [
        FFMPEG, "-hide_banner", "-nostats", "-i", path,
        "-vf", f"blackdetect=d={BLACK_D}:pic_th={BLACK_PIC_TH}:pix_th={BLACK_PIX_TH}",
        "-an", "-f", "null", "-"
    ]
    log = _run(cmd)
    blacks: List[Tuple[float, float]] = []
    b_start: Optional[float] = None
    for line in log.splitlines():
        if "black_start" in line:
            m = re.search(r"black_start:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                b_start = float(m.group(1))
        elif "black_end" in line:
            m = re.search(r"black_end:\s*([0-9]+(?:\.[0-9]+)?)", line)
            if b_start is not None and m:
                blacks.append((b_start, float(m.group(1))))
                b_start = None
    return blacks


# ---------- scoring ----------

def _build_bins(dur: float) -> List[Tuple[float, float]]:
    bins: List[Tuple[float, float]] = []
    t = 0.0
    while t < dur:
        bins.append((t, min(dur, t + BIN_SEC)))
        t += BIN_SEC
    return bins

def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 <= b0 or b1 <= a0)

def _analyze_file(path: str) -> List[Tuple[float, float, float]]:
    """
    Returns [(bin_start, bin_end, score)] for the file.
    Score is based on speech presence, loudness, scene-change density, and black-penalty.
    """
    dur = _duration(path)
    if dur <= 0:
        return []
    bins = _build_bins(dur)

    # features
    scene_times = _scene_change_times(path)
    rms = _audio_rms_bins(path, dur)
    speech = _speech_intervals(path)
    blacks = _black_intervals(path)

    # normalize loudness from dB -> 0..1
    loud_vals = [max(0.0, 60.0 + v) for v in rms]  # -60dB => 0
    max_loud = max(loud_vals) or 1.0
    loud_norm = [v / max_loud for v in loud_vals]

    # scene changes per bin
    scene_counts = [0] * len(bins)
    for st in scene_times:
        idx = min(int(st / BIN_SEC), len(bins) - 1)
        scene_counts[idx] += 1
    max_scene = max(scene_counts) or 1
    scene_norm = [c / max_scene for c in scene_counts]

    # speech presence per bin (0/1)
    speech_mask = [0.0] * len(bins)
    for i, (b0, b1) in enumerate(bins):
        for (s0, s1) in speech:
            if _overlaps(b0, b1, s0, s1):
                speech_mask[i] = 1.0
                break

    # black penalty per bin
    black_pen = [1.0] * len(bins)
    for i, (b0, b1) in enumerate(bins):
        for (k0, k1) in blacks:
            if _overlaps(b0, b1, k0, k1):
                black_pen[i] = BLACK_PENALTY
                break

    scores: List[float] = []
    for i in range(len(bins)):
        score = (W_SPEECH * speech_mask[i]) + (W_LOUD * loud_norm[i]) + (W_SCENE * scene_norm[i])
        score *= black_pen[i]
        scores.append(score)

    return [(bins[i][0], bins[i][1], scores[i]) for i in range(len(bins))]


# ---------- candidate generation ----------

def _adaptive_threshold(values: List[float]) -> float:
    """
    Adaptive threshold: max(mean*0.9, percentile60). Ensures something passes.
    """
    if not values:
        return 0.0
    vals = sorted(values)
    mean = sum(vals) / len(vals)
    p60 = vals[int(0.60 * (len(vals) - 1))]
    return max(mean * 0.9, p60 * 1.0)

def _collect_candidate_clips(path: str, min_clip: float, max_clip: float) -> List[Clip]:
    bins = _analyze_file(path)
    if not bins:
        print(f"[analysis] no bins for {path}")
        return []

    scores = [s for _, _, s in bins]
    thresh = _adaptive_threshold(scores)

    candidates: List[Clip] = []
    cur_start: Optional[float] = None
    cur_sum = 0.0
    cur_bins = 0

    def flush(end_time: float):
        nonlocal cur_start, cur_sum, cur_bins
        if cur_start is None:
            return
        dur = end_time - cur_start
        if dur >= min_clip:
            # split long regions so none exceed max_clip
            chunks = int(math.ceil(dur / max_clip))
            chunk_dur = dur / chunks
            base = (cur_sum / max(1, cur_bins))
            for k in range(chunks):
                s = cur_start + k * chunk_dur
                e = min(end_time, s + chunk_dur)
                if e - s >= min_clip:
                    candidates.append(Clip(src=path, start=s, end=e, score=base))
        cur_start = None
        cur_sum = 0.0
        cur_bins = 0

    for (b0, b1, sc) in bins:
        if sc >= thresh:
            if cur_start is None:
                cur_start = b0
            cur_sum += sc
            cur_bins += 1
        else:
            flush(b0)
    flush(bins[-1][1])

    print(f"[analysis] {path}: {len(candidates)} candidate clips (thresh={thresh:.3f})")
    return candidates


# ---------- selection & rendering ----------

def _render_concat(tmpdir: str, inputs: List[str], portrait: bool = True) -> str:
    out_path = os.path.join(tmpdir, "out.mp4")
    concat_txt = _safe_concat_list(tmpdir, inputs)
    vf = ("scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
          "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black") if portrait else \
         ("scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black")
    cmd = [
        FFMPEG, "-y",
        "-analyzeduration", "100M", "-probesize", "100M",
        "-safe", "0", "-f", "concat", "-i", concat_txt,
        "-ignore_unknown", "1",
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
    vf = ("scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
          "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black") if portrait else \
         ("scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black")

    for i, c in enumerate(clips):
        part = os.path.join(tmpdir, f"part_{i:03d}.mp4")
        cmd = [
            FFMPEG, "-y",
            "-ss", f"{c.start:.3f}", "-to", f"{c.end:.3f}",
            "-i", c.src,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            part
        ]
        _run(cmd)
        parts.append(part)

    return _render_concat(tmpdir, parts, portrait=portrait)

def _pick_best(files: List[str],
               max_duration: Optional[int],
               take_top_k: Optional[int],
               min_clip: float,
               max_clip: float) -> List[Clip]:
    all_candidates: List[Clip] = []
    for f in files:
        all_candidates.extend(_collect_candidate_clips(f, min_clip, max_clip))

    if not all_candidates:
        # Emergency fallback: slide a min_clip window over the *loudest* file
        print("[selection] no candidates — generating fallback windows")
        best_file = None
        best_rms_sum = -1e9
        for f in files:
            dur = _duration(f)
            if dur <= 0:
                continue
            rms = _audio_rms_bins(f, dur)
            s = sum(sorted(rms, reverse=True)[:max(1, int(min_clip / BIN_SEC))])
            if s > best_rms_sum:
                best_rms_sum = s
                best_file = f
        if best_file:
            dur = _duration(best_file)
            start = 0.0
            end = min(dur, min_clip)
            all_candidates = [Clip(best_file, start, end, 1.0)]

    # sort by score desc
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    # enforce top-k
    if take_top_k and take_top_k > 0:
        all_candidates = all_candidates[:take_top_k]

    # enforce max_duration
    if max_duration and max_duration > 0:
        picked: List[Clip] = []
        acc = 0.0
        for c in all_candidates:
            if acc + c.dur <= max_duration + 1e-3:
                picked.append(c)
                acc += c.dur
            if acc >= max_duration:
                break
        all_candidates = picked or all_candidates[:1]

    print(f"[selection] returning {len(all_candidates)} clips")
    return all_candidates


# ---------- public jobs ----------

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload:
      session_id, files, output_prefix, portrait,
      mode ("concat"|"best"), max_duration, take_top_k,
      min_clip_seconds, max_clip_seconds, drop_* (not used directly)
    """
    sess = payload.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}"
    files = list(payload.get("files") or [])
    out_prefix = (payload.get("output_prefix") or "editdna/outputs").strip("/")
    portrait = bool(payload.get("portrait", True))
    mode = (payload.get("mode") or "concat").lower().strip()

    max_duration = payload.get("max_duration")
    take_top_k = payload.get("take_top_k")
    min_clip = float(payload.get("min_clip_seconds") or 2.5)
    max_clip = float(payload.get("max_clip_seconds") or 10.0)

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{sess}-")
    try:
        if not files:
            return {"ok": False, "error": "No input files provided", "session_id": sess}

        print(f"[job] mode={mode} files={len(files)} sess={sess}")

        if mode == "best":
            clips = _pick_best(files, max_duration, take_top_k, min_clip, max_clip)
            out_local = _render_clips(tmpdir, clips, portrait=portrait)
            mode_used = "best"
        else:
            out_local = _render_concat(tmpdir, files, portrait=portrait)
            mode_used = "concat"

        # Upload + presign
        s3_uri = upload_file(out_local, out_prefix, content_type="video/mp4")
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
