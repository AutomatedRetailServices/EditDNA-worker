# jobs.py — heuristic "best clips" picker + renderer
import os
import re
import math
import uuid
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from s3_utils import upload_file, presigned_url, S3_BUCKET  # your existing helper

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# ---------- utilities ----------

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

@dataclass
class Clip:
    src: str
    start: float
    end: float
    score: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ---------- analysis ----------
# We sample per 0.5s “bins”: audio loudness (via astats) and scene change count (via showinfo).
# Score = weighted sum over the bin; later we merge bins into contiguous windows.

BIN_SEC = 0.5
SCENE_THRESH = float(os.getenv("SCENE_THRESH", "0.04"))  # ffmpeg "scene" threshold

def _analyze_file(path: str) -> List[Tuple[float, float, float]]:
    """
    Returns list of (bin_start, bin_end, score) for `path`.
    """
    # 1) scene change markers
    scene_cmd = [
        FFMPEG, "-hide_banner", "-nostats", "-i", path,
        "-vf", f"select='gt(scene,{SCENE_THRESH})',showinfo",
        "-an", "-f", "null", "-"
    ]
    scene_log = _run(scene_cmd)

    # Extract pts_time from showinfo lines
    scene_times: List[float] = []
    for line in scene_log.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            m = re.search(r"pts_time:([0-9]+\.[0-9]+)", line)
            if m:
                scene_times.append(float(m.group(1)))

    # 2) audio stats (RMS level) sampled densely
    # We use astats to print RMS per window. ‘metadata=1’ prints per frame,
    # we’ll downsample into 0.5s bins.
    a_cmd = [
        FFMPEG, "-hide_banner", "-i", path,
        "-vn",
        "-af", "astats=metadata=1:reset=0.5",
        "-f", "null", "-"
    ]
    a_log = _run(a_cmd)

    # Parse lines like: "RMS_level: -18.3"
    # We’ll accumulate per 0.5s; astats prints a block each reset window.
    rms_levels: List[float] = []
    for line in a_log.splitlines():
        if "RMS_level:" in line:
            m = re.search(r"RMS_level:\s*(-?\d+\.?\d*)", line)
            if m:
                try:
                    rms_levels.append(float(m.group(1)))
                except Exception:
                    pass

    dur = _duration(path)
    if dur <= 0:
        return []

    # Build 0.5s bins across the duration
    bins: List[Tuple[float, float]] = []
    t = 0.0
    while t < dur:
        bins.append((t, min(dur, t + BIN_SEC)))
        t += BIN_SEC

    # Map scene changes to bins (count per bin)
    scene_counts = [0] * len(bins)
    for st in scene_times:
        idx = min(int(st / BIN_SEC), len(bins) - 1)
        scene_counts[idx] += 1

    # Map audio RMS (negative dB; higher (less negative) is louder).
    # Align rms_levels length to bins (pad/trim).
    if len(rms_levels) < len(bins):
        rms_levels += [min(rms_levels + [ -60.0 ])] * (len(bins) - len(rms_levels))
    if len(rms_levels) > len(bins):
        rms_levels = rms_levels[:len(bins)]

    # Normalize components and compute scores
    # Convert dB to positive loudness: loud = max(0, 60 + RMS_dB)
    loud_vals = [max(0.0, 60.0 + v) for v in rms_levels]  # -60 dB -> 0, -10 dB -> 50
    max_loud = max(loud_vals) or 1.0
    max_scene = max(scene_counts) or 1

    scores: List[float] = []
    for i in range(len(bins)):
        loud_norm = loud_vals[i] / max_loud
        scene_norm = scene_counts[i] / max_scene
        # weights: audio 0.65, scene 0.35
        scores.append(0.65 * loud_norm + 0.35 * scene_norm)

    return [(bins[i][0], bins[i][1], scores[i]) for i in range(len(bins))]

def _collect_candidate_clips(path: str,
                             min_clip: float,
                             max_clip: float) -> List[Clip]:
    """
    Merge high-score bins into windows between [min_clip, max_clip].
    """
    bins = _analyze_file(path)
    if not bins:
        return []

    # Simple threshold: keep bins >= median score
    scores = [s for _, _, s in bins]
    thresh = sorted(scores)[len(scores)//2]

    candidates: List[Clip] = []
    cur_start: Optional[float] = None
    cur_score_sum = 0.0
    cur_bins = 0

    def flush(end_time: float):
        nonlocal cur_start, cur_score_sum, cur_bins
        if cur_start is None:
            return
        dur = end_time - cur_start
        if dur >= min_clip:
            # split into <= max_clip chunks
            chunks = int(math.ceil(dur / max_clip))
            chunk_dur = dur / chunks
            base_score = (cur_score_sum / max(1, cur_bins))
            for k in range(chunks):
                s = cur_start + k * chunk_dur
                e = min(end_time, s + chunk_dur)
                if e - s >= min_clip:
                    candidates.append(Clip(src=path, start=s, end=e, score=base_score))
        # reset
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
    # tail
    flush(bins[-1][1])

    return candidates

# ---------- rendering ----------

def _render_concat(tmpdir: str, inputs: List[str], portrait: bool = True) -> str:
    out_path = os.path.join(tmpdir, "out.mp4")
    concat_txt = _safe_concat_list(tmpdir, inputs)
    vf = "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black" if portrait \
        else "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
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
    """
    Trim each clip to an intermediate mp4, then concat them.
    """
    part_files: List[str] = []
    vf = "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black" if portrait \
        else "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"

    for i, c in enumerate(clips):
        part = os.path.join(tmpdir, f"part_{i:03d}.mp4")
        # Re-encode trims for reliability across MOVs/keyframes
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
        part_files.append(part)

    return _render_concat(tmpdir, part_files, portrait=portrait)

# ---------- public jobs ----------

def _pick_best(files: List[str],
               max_duration: Optional[int],
               take_top_k: Optional[int],
               min_clip: float,
               max_clip: float) -> List[Clip]:
    all_candidates: List[Clip] = []
    for f in files:
        all_candidates.extend(_collect_candidate_clips(f, min_clip, max_clip))

    if not all_candidates:
        return []

    # sort by score desc
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    # top-k
    if take_top_k and take_top_k > 0:
        all_candidates = all_candidates[:take_top_k]

    # cut to max_duration
    if max_duration and max_duration > 0:
        picked: List[Clip] = []
        acc = 0.0
        for c in all_candidates:
            if acc + c.dur <= max_duration + 1e-3:
                picked.append(c)
                acc += c.dur
            if acc >= max_duration:
                break
        all_candidates = picked

    return all_candidates

def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected payload (what your web service enqueues):
      {
        "session_id": str,
        "files": [urls...],
        "output_prefix": str,
        "portrait": bool,
        "mode": "concat"|"best",
        "max_duration": int|None,
        "take_top_k": int|None,
        "min_clip_seconds": float|None,
        "max_clip_seconds": float|None,
        "drop_silent": bool|None,   # currently implicit in scoring via audio
        "drop_black": bool|None     # handled indirectly via scene/loudness
      }
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

        if mode == "best":
            clips = _pick_best(files, max_duration, take_top_k, min_clip, max_clip)
            # fallback if analysis yields nothing
            if not clips:
                out_local = _render_concat(tmpdir, files, portrait=portrait)
                mode_used = "concat_fallback"
            else:
                out_local = _render_clips(tmpdir, clips, portrait=portrait)
                mode_used = "best"
        else:
            out_local = _render_concat(tmpdir, files, portrait=portrait)
            mode_used = "concat"

        # Upload to S3 -> return s3:// and presigned URL
        s3_key_prefix = f"{out_prefix}"
        s3_uri = upload_file(out_local, s3_key_prefix, content_type="video/mp4")

        # presign
        # parse "s3://bucket/key"
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
    # For now, same path; keep API compatibility.
    return job_render(payload)
