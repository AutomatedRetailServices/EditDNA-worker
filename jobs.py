# jobs.py — worker side video jobs for EditDNA
import os
import re
import tempfile
import subprocess
import shutil
from typing import List, Dict, Any, Tuple

from s3_utils import upload_file, presigned_url, parse_s3_url

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# ----------------------------
# shell helpers
# ----------------------------
def _run(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout

def _must_run(cmd: List[str]) -> str:
    code, out = _run(cmd)
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}\n---\n{out}\n---")
    return out

# ----------------------------
# ffprobe/ffmpeg scoring utils
# ----------------------------
_VOL_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", re.IGNORECASE)
_BLACK_DUR_RE = re.compile(r"black_start.*?black_end.*?black_duration:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

def probe_duration(path: str) -> float:
    out = _must_run([
        FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ])
    try:
        return max(0.0, float(out.strip()))
    except Exception:
        return 0.0

def mean_volume_db(path: str, analyze_seconds: float = 8.0) -> float:
    """
    Use volumedetect over the first analyze_seconds to estimate perceived loudness.
    Higher (less negative) is 'better'. Defaults to -60 dB if not found.
    """
    # Limit input read to analyze_seconds to keep it fast
    out = _must_run([
        FFMPEG_BIN, "-v", "error",
        "-t", f"{analyze_seconds}",
        "-i", path,
        "-af", "volumedetect",
        "-f", "null", "-"
    ])
    m = _VOL_RE.search(out)
    if not m:
        return -60.0
    return float(m.group(1))

def black_ratio(path: str) -> float:
    """
    Use blackdetect to estimate how much of the clip is black frames.
    Returns fraction (0..1). Falls back to 0 if detection fails.
    """
    dur = max(0.001, probe_duration(path))
    out = _must_run([
        FFMPEG_BIN, "-v", "info", "-i", path,
        "-vf", "blackdetect=d=0:pic_th=0.98",
        "-an", "-f", "null", "-"
    ])
    total_black = 0.0
    for m in _BLACK_DUR_RE.finditer(out):
        try:
            total_black += float(m.group(1))
        except Exception:
            pass
    ratio = min(1.0, max(0.0, total_black / dur))
    return ratio

# ----------------------------
# concat helpers
# ----------------------------
def _write_concat_file(tmpdir: str, files: List[str]) -> str:
    list_path = os.path.join(tmpdir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            safe = str(p).replace("'", "'\\''")
            f.write("file '{}'\n".format(safe))
    return list_path

def _concat_vertical(session_id: str, in_files: List[str], tmpdir: str,
                     portrait: bool = True, max_total_seconds: float | None = None) -> str:
    """
    Concatenate already-trimmed inputs; if max_total_seconds is set, we trim the
    last clip to not exceed the cap.
    """
    # Optional total-cap trimming by creating a temp list possibly with a last
    # partial clip segment via -t.
    files_for_concat = []
    if max_total_seconds is not None:
        remaining = float(max_total_seconds)
        for path in in_files:
            d = probe_duration(path)
            if d <= 0.05:
                continue
            use = min(d, remaining)
            if use < 0.65:       # too tiny to bother adding
                break
            if use < d - 0.01:
                # create a temp trimmed copy
                trimmed = os.path.join(tmpdir, f"cap_{len(files_for_concat):03d}.mp4")
                _must_run([
                    FFMPEG_BIN, "-y", "-t", f"{use}", "-i", path,
                    "-c", "copy", trimmed
                ])
                files_for_concat.append(trimmed)
                break
            else:
                files_for_concat.append(path)
            remaining -= use
            if remaining <= 0.1:
                break
    else:
        files_for_concat = in_files

    concat_txt = _write_concat_file(tmpdir, files_for_concat)
    out_path = os.path.join(tmpdir, f"{session_id}.mp4")

    vf = "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black" \
         if portrait else "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"

    _must_run([
        FFMPEG_BIN,
        "-y",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-safe", "0",
        "-f", "concat",
        "-i", concat_txt,
        "-ignore_unknown", "1",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ])
    return out_path

def _trim_to_bounds(src: str, dst: str, min_s: float | None, max_s: float | None) -> bool:
    dur = probe_duration(src)
    if dur <= 0.05:
        return False
    target = dur
    if max_s is not None:
        target = min(target, float(max_s))
    if min_s is not None:
        if target < float(min_s):
            return False
    _must_run([FFMPEG_BIN, "-y", "-t", f"{target}", "-i", src, "-c", "copy", dst])
    return True

# ----------------------------
# selection strategies
# ----------------------------
def _select_best(files: List[str],
                 take_top_k: int | None,
                 drop_black: bool,
                 min_clip_seconds: float | None,
                 max_clip_seconds: float | None,
                 portrait: bool,
                 tmpdir: str) -> List[str]:
    """
    Score each clip by loudness; optionally drop clips with high black ratio.
    Return list of paths to (possibly trimmed) temporary MP4 files ready to concat.
    """
    scored: List[Tuple[float, str]] = []
    for f in files:
        try:
            if drop_black:
                br = black_ratio(f)
                if br >= 0.60:
                    # looks mostly black; skip
                    continue
            vol = mean_volume_db(f, analyze_seconds=8.0)  # e.g. -7.3 dB is louder than -20 dB
            scored.append((vol, f))
        except Exception:
            # if probe fails, push with very low score so it's unlikely chosen
            scored.append((-90.0, f))

    if not scored:
        return []

    scored.sort(reverse=True, key=lambda x: x[0])  # louder first
    if take_top_k and take_top_k > 0:
        scored = scored[:take_top_k]

    # Trim each to within min/max bounds (copy to tmp mp4), return paths
    out_paths: List[str] = []
    for _, src in scored:
        dst = os.path.join(tmpdir, f"seg_{len(out_paths):03d}.mp4")
        ok = _trim_to_bounds(src, dst, min_clip_seconds, max_clip_seconds)
        if ok:
            out_paths.append(dst)
    return out_paths

# ----------------------------
# public entrypoints
# ----------------------------
def _normalize_payload(*args, **kwargs) -> Dict[str, Any]:
    """
    Accept either a dict payload (new style) or legacy (session_id, files, output_prefix).
    """
    if args and isinstance(args[0], dict):
        payload = dict(args[0])  # shallow copy
    elif len(args) >= 3 and isinstance(args[0], str) and isinstance(args[1], list):
        payload = {
            "session_id": args[0],
            "files": args[1],
            "output_prefix": args[2],
        }
    else:
        payload = dict(kwargs)

    # defaults
    payload.setdefault("session_id", "session")
    payload.setdefault("files", [])
    payload.setdefault("output_prefix", "editdna/outputs")
    payload.setdefault("portrait", True)
    payload.setdefault("mode", "concat")
    payload.setdefault("max_duration", None)
    payload.setdefault("take_top_k", None)
    payload.setdefault("min_clip_seconds", None)
    payload.setdefault("max_clip_seconds", None)
    payload.setdefault("drop_silent", True)  # reserved for future audio gating
    payload.setdefault("drop_black", True)
    return payload

def job_render(*args, **kwargs) -> Dict[str, Any]:
    """
    Main render job. Supports:
      - mode="concat"  → straight concat (as before)
      - mode="best"    → pick loudest non-black clips, trim to bounds, concat
    Uploads final to S3 and returns s3 url + presigned https.
    """
    payload = _normalize_payload(*args, **kwargs)
    session_id: str = payload["session_id"]
    files: List[str] = [str(x) for x in payload["files"]]
    output_prefix: str = str(payload["output_prefix"]).strip("/")
    portrait: bool = bool(payload.get("portrait", True))

    mode: str = str(payload.get("mode", "concat") or "concat").lower()
    max_duration = payload.get("max_duration")
    take_top_k = payload.get("take_top_k")
    min_clip_seconds = payload.get("min_clip_seconds")
    max_clip_seconds = payload.get("max_clip_seconds")
    drop_black = bool(payload.get("drop_black", True))

    tmpdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    try:
        if mode == "best":
            # 1) pick and trim best segments
            chosen = _select_best(
                files=files,
                take_top_k=take_top_k,
                drop_black=drop_black,
                min_clip_seconds=min_clip_seconds,
                max_clip_seconds=max_clip_seconds,
                portrait=portrait,
                tmpdir=tmpdir,
            )
            if not chosen:
                # fallback: take first file within bounds
                fallback = []
                if files:
                    dst = os.path.join(tmpdir, "seg_000.mp4")
                    if _trim_to_bounds(files[0], dst, min_clip_seconds, max_clip_seconds):
                        fallback = [dst]
                chosen = fallback

            out_path = _concat_vertical(session_id, chosen, tmpdir, portrait=portrait,
                                        max_total_seconds=float(max_duration) if max_duration else None)

        else:
            # Straight concat (optionally respecting a total duration cap)
            out_path = _concat_vertical(session_id, files, tmpdir, portrait=portrait,
                                        max_total_seconds=float(max_duration) if max_duration else None)

        # Upload → S3
        s3_uri = upload_file(out_path, output_prefix, content_type="video/mp4")
        bkt, key = parse_s3_url(s3_uri)
        assert bkt is not None
        url = presigned_url(bkt, key, expires=3600)

        return {
            "ok": True,
            "session_id": session_id,
            "mode": mode,
            "output_s3": s3_uri,
            "output_url": url,
            "inputs": files,
        }
    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e), "mode": mode}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# For API compatibility
def job_render_chunked(*args, **kwargs) -> Dict[str, Any]:
    return job_render(*args, **kwargs)
