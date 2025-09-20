# jobs.py — smart rendering (concat / best / split) with S3 upload
from __future__ import annotations

import os
import re
import math
import json
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from s3_utils import upload_file, presigned_url, parse_s3_url, S3_BUCKET

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# ---------- small utils ----------

def _run(cmd: List[str]) -> Tuple[int, str]:
    """Run a command, return (rc, combined_output_str)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout

def _run_or_raise(cmd: List[str]) -> str:
    rc, out = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}\n---\n{out}\n---")
    return out

def _sec(x: float | int | None, default: float = 0.0) -> float:
    return float(x) if x is not None else default

def _safe_basename(url: str) -> str:
    bn = url.rsplit("/", 1)[-1] or "clip"
    return bn.split("?")[0]

@dataclass
class ClipInfo:
    src: str           # local path
    url: str           # original URL (for reporting)
    duration: float    # seconds
    black_ratio: float # 0..1 (approx)
    silence_spans: List[Tuple[float, float]]  # [(start,end), ...]
    voiced_spans: List[Tuple[float, float]]   # derived from silence

@dataclass
class Segment:
    src: str
    url: str
    ss: float     # start
    dur: float    # duration
    score: float  # ranking score (higher is better)

# ---------- probing / analysis ----------

_DURATION_RE = re.compile(r"duration\"\s*:\s*([0-9.]+)", re.I)

def probe_duration(path: str) -> float:
    # fast json probe
    cmd = [
        FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    out = _run_or_raise(cmd)
    try:
        data = json.loads(out)
        dur = float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        # fallback quick parse
        m = _DURATION_RE.search(out)
        dur = float(m.group(1)) if m else 0.0
    return max(0.0, dur)

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END   = re.compile(r"silence_end:\s*([0-9.]+)")

def detect_silence(path: str, noise_db: float = -30.0, min_d: float = 0.35) -> List[Tuple[float, float]]:
    """
    Returns list of (start,end) silence spans using silencedetect.
    """
    cmd = [
        FFMPEG_BIN,
        "-hide_banner", "-nostats", "-vn",
        "-i", path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_d}",
        "-f", "null", "-"
    ]
    rc, out = _run(cmd)
    if rc != 0:
        # treat as "no silence info" instead of failing hard
        return []
    spans: List[Tuple[float, float]] = []
    cur: Optional[float] = None
    for line in out.splitlines():
        ms = _SILENCE_START.search(line)
        if ms:
            cur = float(ms.group(1))
        me = _SILENCE_END.search(line)
        if me and cur is not None:
            spans.append((cur, float(me.group(1))))
            cur = None
    return spans

_BLACK_START = re.compile(r"black_start:(\s*[0-9.]+)")
_BLACK_END   = re.compile(r"black_end:(\s*[0-9.]+)")

def detect_black_ratio(path: str) -> float:
    """
    Approximate black frame ratio using blackdetect.
    """
    cmd = [
        FFMPEG_BIN, "-hide_banner", "-nostats",
        "-i", path,
        "-vf", "blackdetect=d=0.25:pic_th=0.98",
        "-an", "-f", "null", "-"
    ]
    rc, out = _run(cmd)
    if rc != 0:
        return 0.0
    spans: List[Tuple[float, float]] = []
    cur: Optional[float] = None
    for line in out.splitlines():
        ms = _BLACK_START.search(line)
        if ms:
            cur = float(ms.group(1))
        me = _BLACK_END.search(line)
        if me and cur is not None:
            spans.append((cur, float(me.group(1))))
            cur = None
    total_black = sum(max(0.0, e - s) for s, e in spans)
    dur = probe_duration(path) or 1.0
    return max(0.0, min(1.0, total_black / dur))

def invert_spans(total: float, silent: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return voiced spans = complement of silent spans within [0,total]."""
    if total <= 0:
        return []
    s = sorted([(max(0.0, a), min(total, b)) for a, b in silent if b > a], key=lambda x: x[0])
    voiced: List[Tuple[float, float]] = []
    cur = 0.0
    for a, b in s:
        if a > cur:
            voiced.append((cur, a))
        cur = max(cur, b)
    if cur < total:
        voiced.append((cur, total))
    return [(a, b) for a, b in voiced if b - a > 0.0]

def analyze_clip(local: str, url: str) -> ClipInfo:
    dur = probe_duration(local)
    sil = detect_silence(local)
    voiced = invert_spans(dur, sil) if sil else [(0.0, dur)]
    black = detect_black_ratio(local)
    return ClipInfo(src=local, url=url, duration=dur, black_ratio=black,
                    silence_spans=sil, voiced_spans=voiced)

# ---------- selection / ranking ----------

def choose_segments(
    infos: List[ClipInfo],
    *,
    min_clip: float | None,
    max_clip: float | None,
    take_top_k: int | None,
    drop_silent: bool,
    drop_black: bool,
    max_total: float | None,
) -> List[Segment]:
    segs: List[Segment] = []
    min_len = _sec(min_clip, 0.0)
    max_len = _sec(max_clip, 9999)

    for info in infos:
        # candidate spans: voiced or whole
        spans = info.voiced_spans if drop_silent else [(0.0, info.duration)]
        if not spans:
            continue

        # for each span, clamp to min/max and create a scored segment
        for a, b in spans:
            span_len = max(0.0, b - a)
            if span_len < max(0.01, min_len):  # skip too short
                continue
            use_len = min(span_len, max_len)

            # simple score: prefer longer voiced spans, penalize black ratio
            voice_weight = min(1.0, span_len / (max_len if max_len > 0 else 1.0))
            black_penalty = (1.0 - info.black_ratio) if drop_black else 1.0
            score = 0.7 * voice_weight + 0.3 * black_penalty

            segs.append(Segment(
                src=info.src, url=info.url,
                ss=a, dur=use_len,
                score=score
            ))

    # rank best to worst
    segs.sort(key=lambda s: s.score, reverse=True)

    # keep only top_k
    if take_top_k and take_top_k > 0:
        segs = segs[:take_top_k]

    # enforce max_total by trimming the tail
    if max_total and max_total > 0:
        budget = float(max_total)
        trimmed: List[Segment] = []
        for s in segs:
            if budget <= 0:
                break
            d = min(s.dur, budget)
            trimmed.append(Segment(src=s.src, url=s.url, ss=s.ss, dur=d, score=s.score))
            budget -= d
        segs = trimmed

    return segs

# ---------- rendering ----------

def _concat_vf_flags(portrait: bool) -> List[str]:
    if not portrait:
        return []
    # 1080x1920 letterbox
    return [
        "-vf",
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
    ]

def _render_concat_https(urls: Sequence[str], dst: str, portrait: bool) -> None:
    """Concat by url via concat demuxer + https whitelist."""
    with tempfile.TemporaryDirectory(prefix="editdna-") as tmpdir:
        list_path = Path(tmpdir) / "concat.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for u in urls:
                esc = u.replace("'", "'\\''")
                f.write(f"file '{esc}'\n")
        cmd = [
            FFMPEG_BIN, "-y",
            "-analyzeduration", "100M", "-probesize", "100M",
            "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
            "-safe", "0",
            "-f", "concat", "-i", str(list_path),
            "-ignore_unknown",
            *_concat_vf_flags(portrait),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            dst,
        ]
        _run_or_raise(cmd)

def _render_segments_fffilter(segs: Sequence[Segment], dst: str, portrait: bool) -> None:
    """
    Trim with -ss/-t per input, then concat via filter_complex.
    """
    if not segs:
        raise RuntimeError("No segments selected")

    cmd: List[str] = [FFMPEG_BIN, "-y"]
    # inputs
    for s in segs:
        # precise trim: input -ss (good enough), with -t
        cmd += ["-ss", f"{s.ss:.3f}", "-t", f"{s.dur:.3f}", "-i", s.src]

    # build filter graph
    n = len(segs)
    parts: List[str] = []
    for i in range(n):
        # We keep audio & video as-is per trimmed input
        parts.append(f"[{i}:v][{i}:a]")
    filter_graph = "".join(parts) + f"concat=n={n}:v=1:a=1[v][a]"

    cmd += [
        "-filter_complex", filter_graph,
        "-map", "[v]", "-map", "[a]",
        *_concat_vf_flags(portrait),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        dst,
    ]
    _run_or_raise(cmd)

# ---------- public jobs ----------

def job_render(payload_or_session_id: Any = None, files: Any = None, output_prefix: str | None = None, **kwargs) -> Dict[str, Any]:
    """
    Accepts either:
      - a single dict payload (from web): {"session_id","files","output_prefix","portrait","mode",...}
      - legacy signature (session_id, files, output_prefix, **options)
    Returns { ok, session_id, mode, output_s3, output_url, inputs, notes? }
    """
    # --- normalize args ---
    if isinstance(payload_or_session_id, dict):
        p = dict(payload_or_session_id)
    else:
        p = {
            "session_id": payload_or_session_id,
            "files": files,
            "output_prefix": output_prefix,
            **kwargs,
        }

    session_id = str(p.get("session_id") or "session")
    urls: List[str] = [str(u) for u in (p.get("files") or [])]
    if not urls:
        return {"ok": False, "session_id": session_id, "error": "No input files provided"}

    mode = (p.get("mode") or "concat").lower()
    portrait = bool(p.get("portrait", True))
    out_prefix = p.get("output_prefix") or "editdna/outputs"

    # best-mode knobs
    max_duration = p.get("max_duration")
    take_top_k   = p.get("take_top_k")
    min_clip_s   = p.get("min_clip_seconds")
    max_clip_s   = p.get("max_clip_seconds")
    drop_silent  = bool(p.get("drop_silent", True))
    drop_black   = bool(p.get("drop_black", True))

    # --- workspace ---
    work = Path(f"/tmp/editdna-{session_id}")
    work.mkdir(parents=True, exist_ok=True)
    out_path = str(work / f"{session_id}.mp4")

    notes: List[str] = []

    try:
        if mode == "concat":
            # safest path: concat directly from HTTPS with whitelist
            _render_concat_https(urls, out_path, portrait)

        else:
            # download inputs locally for robust trimming/analysis
            local_paths: List[str] = []
            for u in urls:
                # cache friendly local name
                lp = str(work / _safe_basename(u))
                # only download if not present (simple cache between retries)
                if not Path(lp).exists():
                    rc, out = _run(["curl", "-L", "-o", lp, u])
                    if rc != 0 or not Path(lp).exists():
                        raise RuntimeError(f"Failed to download: {u}\n{out}")
                local_paths.append(lp)

            # analyze clips
            infos: List[ClipInfo] = []
            for u, lp in zip(urls, local_paths):
                try:
                    info = analyze_clip(lp, u)
                    infos.append(info)
                except Exception as e:
                    notes.append(f"probe_failed:{_safe_basename(u)}:{e}")

            if not infos:
                raise RuntimeError("No analyzable clips")

            if mode == "best":
                segs = choose_segments(
                    infos,
                    min_clip=min_clip_s,
                    max_clip=max_clip_s,
                    take_top_k=take_top_k,
                    drop_silent=drop_silent,
                    drop_black=drop_black,
                    max_total=max_duration,
                )
                if not segs:
                    raise RuntimeError("No usable segments after selection")
                _render_segments_fffilter(segs, out_path, portrait)

            elif mode == "split":
                # take one best voiced span per clip (or whole clip), then concat
                segs: List[Segment] = []
                for info in infos:
                    spans = info.voiced_spans if drop_silent else [(0.0, info.duration)]
                    if not spans:
                        continue
                    # pick longest span in that clip
                    a, b = max(spans, key=lambda t: (t[1] - t[0]))
                    d = b - a
                    if min_clip_s and d < min_clip_s:
                        continue
                    if max_clip_s:
                        d = min(d, max_clip_s)
                    segs.append(Segment(src=info.src, url=info.url, ss=a, dur=d, score=1.0))
                if max_duration:
                    # same budget trim
                    budget = float(max_duration)
                    trimmed: List[Segment] = []
                    for s in segs:
                        if budget <= 0:
                            break
                        d = min(s.dur, budget)
                        trimmed.append(Segment(src=s.src, url=s.url, ss=s.ss, dur=d, score=s.score))
                        budget -= d
                    segs = trimmed

                if not segs:
                    raise RuntimeError("No segments to render in split mode")
                _render_segments_fffilter(segs, out_path, portrait)

            else:
                raise RuntimeError(f"Unknown mode: {mode}")

        # upload & sign
        s3_uri = upload_file(out_path, out_prefix, content_type="video/mp4")
        bucket, key = parse_s3_url(s3_uri)
        b = bucket or S3_BUCKET
        url = presigned_url(b, key, expires=3600)

        result = {
            "ok": True,
            "session_id": session_id,
            "mode": mode,
            "output_s3": s3_uri,
            "output_url": url,
            "inputs": urls,
        }
        if notes:
            result["notes"] = notes
        return result

    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e), "mode": mode, "notes": notes}

    finally:
        # keep workspace for a bit during debugging? comment next line to persist.
        shutil.rmtree(work, ignore_errors=True)


def job_render_chunked(payload_or_session_id: Any = None, files: Any = None, output_prefix: str | None = None, **kwargs) -> Dict[str, Any]:
    # simple passthrough to job_render for now
    return job_render(payload_or_session_id, files, output_prefix, **kwargs)
