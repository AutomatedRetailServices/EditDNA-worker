# worker.py — concat, split, and "best takes" selection + S3 upload

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Union

from s3_utils import upload_file as s3_upload, presigned_url, parse_s3_url, new_session_id

FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG  = os.getenv("FFMPEG_BIN",  "ffmpeg")


# ----------------- small io helpers -----------------
def _to_str_list(items: Iterable[Any]) -> List[str]:
    return [str(x) for x in items]


def _write_concat_txt(sources: Sequence[str], dst: Path) -> None:
    with dst.open("w", encoding="utf-8") as f:
        for src in sources:
            esc = str(src).replace("'", "'\\''")
            f.write(f"file '{esc}'\n")


def _run_ffmpeg(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        raise RuntimeError(f"ffmpeg failed:\n{err}") from e


def _vf_portrait_args(enabled: bool) -> List[str]:
    if not enabled:
        return []
    return [
        "-vf",
        "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
    ]


# ----------------- ffmpeg actions -----------------
def _ffmpeg_concat_to_mp4(
    files: Sequence[Union[str, os.PathLike]],
    output_path: Union[str, os.PathLike],
    portrait: bool = True,
) -> None:
    files = _to_str_list(files)
    output_path = str(output_path)

    with tempfile.TemporaryDirectory(prefix="editdna-") as tmpdir:
        concat_file = Path(tmpdir) / "concat.txt"
        _write_concat_txt(files, concat_file)

        cmd: List[str] = [
            FFMPEG,
            "-y",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
            "-safe", "0",
            "-f", "concat",
            "-i", str(concat_file),
            "-ignore_unknown",
            *_vf_portrait_args(portrait),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        _run_ffmpeg(cmd)


def _ffmpeg_single_to_mp4(
    src: Union[str, os.PathLike],
    dst: Union[str, os.PathLike],
    portrait: bool = True,
) -> None:
    cmd: List[str] = [
        FFMPEG,
        "-y",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-protocol_whitelist", "file,crypto,data,https,tcp,tls",
        "-i", str(src),
        "-ignore_unknown",
        *_vf_portrait_args(portrait),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    _run_ffmpeg(cmd)


# ----------------- probing & scoring -----------------
def _ffprobe_json(src: str) -> dict:
    cmd = [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", src]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return {}
    try:
        return json.loads(p.stdout.decode("utf-8", "replace"))
    except Exception:
        return {}


def _probe_duration(src: str) -> float:
    meta = _ffprobe_json(src)
    dur = None
    if "format" in meta and meta["format"].get("duration"):
        try:
            dur = float(meta["format"]["duration"])
        except Exception:
            pass
    return max(dur or 0.0, 0.0)


def _probe_volumedetect(src: str) -> dict:
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", src, "-af", "volumedetect", "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mean_db = None
    for line in (p.stderr or "").splitlines():
        if "mean_volume:" in line:
            try:
                mean_db = float(line.split("mean_volume:")[1].split(" dB")[0].strip())
            except Exception:
                pass
    return {"mean_db": mean_db}


def _probe_silence_pct(src: str) -> float:
    dur = _probe_duration(src)
    if dur <= 0:
        return 100.0
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", src, "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    total = 0.0
    last = None
    for ln in (p.stderr or "").splitlines():
        if "silence_start:" in ln:
            try: last = float(ln.split("silence_start:")[1].strip())
            except Exception: last = None
        elif "silence_end:" in ln and "silence_duration:" in ln and last is not None:
            try: total += float(ln.split("silence_duration:")[1].strip())
            except Exception: pass
            last = None
    return max(0.0, min(100.0, 100.0 * total / dur))


def _probe_black_pct(src: str) -> float:
    dur = _probe_duration(src)
    if dur <= 0:
        return 100.0
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-i", src, "-vf", "blackdetect=d=0.10:pic_th=0.98:pix_th=0.10", "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    total = 0.0
    last = None
    for ln in (p.stderr or "").splitlines():
        if "black_start:" in ln:
            try: last = float(ln.split("black_start:")[1].strip())
            except Exception: last = None
        elif "black_end:" in ln and "black_duration:" in ln and last is not None:
            try: total += float(ln.split("black_duration:")[1].strip())
            except Exception: pass
            last = None
    return max(0.0, min(100.0, 100.0 * total / dur))


def _score_clip(src: str) -> dict:
    dur = _probe_duration(src)
    vol = _probe_volumedetect(src).get("mean_db")
    silence = _probe_silence_pct(src)
    black = _probe_black_pct(src)

    sweet_min, sweet_max = 3.0, 12.0
    if dur <= 0:
        dur_bonus = -2.0
    else:
        center = (sweet_min + sweet_max) / 2.0
        sigma = (sweet_max - sweet_min) / 2.0
        dur_bonus = math.exp(-((dur - center) ** 2) / (2 * (sigma ** 2)))

    vol_bonus = 0.0
    if vol is not None:
        vol_bonus = max(0.0, 1.0 - abs((vol + 16.0) / 14.0))  # ~[-30,-2] -> [0..1]

    silence_pen = min(1.0, silence / 100.0)
    black_pen = min(1.0, black / 100.0)

    score = 3.0 * dur_bonus + 1.5 * vol_bonus - 1.5 * silence_pen - 0.7 * black_pen
    return {"src": src, "duration": dur, "mean_db": vol, "silence_pct": silence, "black_pct": black, "score": score}


# ----------------- tasks -----------------
def task_nop() -> dict:
    return {"echo": {"hello": "world"}}


def job_render(*args, **kwargs) -> dict:
    # Accept payload as dict or positional
    if args and isinstance(args[0], dict):
        payload = dict(args[0])
        session_id = payload.get("session_id") or payload.get("sid") or "session"
        files = payload.get("files") or []
        output_prefix = payload.get("output_prefix") or "editdna/outputs"
        portrait = bool(payload.get("portrait", True))
        mode = (payload.get("mode") or "concat").lower()
        max_duration = payload.get("max_duration")
        take_top_k = payload.get("take_top_k")
        min_clip = payload.get("min_clip_seconds")
        max_clip = payload.get("max_clip_seconds")
        drop_silent = payload.get("drop_silent", True)
        drop_black = payload.get("drop_black", True)
    else:
        session_id = kwargs.get("session_id") or new_session_id()
        files = kwargs.get("files") or []
        output_prefix = kwargs.get("output_prefix", "editdna/outputs")
        portrait = bool(kwargs.get("portrait", True))
        mode = (kwargs.get("mode") or "concat").lower()
        max_duration = kwargs.get("max_duration")
        take_top_k = kwargs.get("take_top_k")
        min_clip = kwargs.get("min_clip_seconds")
        max_clip = kwargs.get("max_clip_seconds")
        drop_silent = kwargs.get("drop_silent", True)
        drop_black = kwargs.get("drop_black", True)
        if args:
            if not session_id and len(args) >= 1: session_id = args[0]
            if not files and len(args) >= 2: files = args[1]
            if len(args) >= 3: output_prefix = args[2]

    session_id = str(session_id or "session")
    files = _to_str_list(files or [])

    workdir = Path(f"/tmp/editdna-{session_id}")
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        # ---- split mode: convert & upload each file individually ----
        if mode == "split":
            outputs = []
            for idx, src in enumerate(files, start=1):
                local_out = workdir / f"{session_id}-{idx:03d}.mp4"
                _ffmpeg_single_to_mp4(src, local_out, portrait=portrait)
                url_s3 = s3_upload(str(local_out), key_prefix=output_prefix, content_type="video/mp4")
                bucket, key = parse_s3_url(url_s3)
                out_url = presigned_url(bucket or os.getenv("S3_BUCKET"), key, expires=3600)
                outputs.append({"index": idx, "source": src, "output_s3": url_s3, "output_url": out_url})
            return {"ok": True, "session_id": session_id, "mode": "split", "outputs": outputs, "count": len(outputs)}

        # ---- best mode: probe -> score -> select -> concat ----
        selected_files = files
        if mode == "best":
            metrics = [_score_clip(f) for f in files]
            if drop_silent:
                metrics = [m for m in metrics if m["silence_pct"] <= 70.0]
            if drop_black:
                metrics = [m for m in metrics if m["black_pct"] <= 60.0]
            metrics.sort(key=lambda m: m["score"], reverse=True)
            if take_top_k and take_top_k > 0:
                metrics = metrics[:take_top_k]

            # cap by max_duration (allow partial of last one)
            if max_duration and max_duration > 0:
                acc = 0.0
                kept = []
                for m in metrics:
                    if acc + m["duration"] <= max_duration + 0.01:
                        kept.append(m)
                        acc += m["duration"]
                    else:
                        remaining = max(0.0, max_duration - acc)
                        if remaining > 0 and (not min_clip or remaining >= float(min_clip)):
                            mm = dict(m)
                            mm["_partial_seconds"] = float(min(remaining, float(max_clip or remaining)))
                            kept.append(mm)
                        break
                metrics = kept

            # optional per-clip max trim
            selected_files = []
            for m in metrics:
                src = m["src"]
                dur = m["duration"]
                keep = m.get("_partial_seconds")
                if not keep and max_clip:
                    keep = min(dur, float(max_clip))
                if keep and min_clip and keep < float(min_clip):
                    continue
                if keep and keep < dur:
                    # trim from start; render to tmp mp4
                    tmp = workdir / ("trim-" + os.path.basename(src) + ".mp4")
                    tcmd = [
                        FFMPEG, "-y", "-i", src,
                        "-t", str(max(0.1, keep)),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "128k",
                        str(tmp),
                    ]
                    _run_ffmpeg(tcmd)
                    selected_files.append(str(tmp))
                else:
                    selected_files.append(src)

        # ---- concat (both concat & best paths land here) ----
        final_local = workdir / f"{session_id}.mp4"
        _ffmpeg_concat_to_mp4(selected_files, final_local, portrait=portrait)

        url_s3 = s3_upload(str(final_local), key_prefix=output_prefix, content_type="video/mp4")
        bucket, key = parse_s3_url(url_s3)
        out_url = presigned_url(bucket or os.getenv("S3_BUCKET"), key, expires=3600)

        return {
            "ok": True,
            "session_id": session_id,
            "mode": mode,
            "output_s3": url_s3,     # s3://... key you own
            "output_url": out_url,   # clickable presigned HTTPS
            "inputs": files,
        }

    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}
    finally:
        # keep tmp if you want debugging: comment next line to retain
        shutil.rmtree(workdir, ignore_errors=True)


def job_render_chunked(*args, **kwargs) -> dict:
    return job_render(*args, **kwargs)
