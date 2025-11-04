import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import boto3

# ============================================================
# ENV
# ============================================================

def _env_str(k: str, d: str) -> str:
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d

def _env_float(k: str, d: float) -> float:
    v = (os.getenv(k) or "").split("#")[0].strip().split()[:1]
    try:
        return float(v[0]) if v else d
    except Exception:
        return d

FFMPEG_BIN  = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET   = _env_str("S3_BUCKET", "")
S3_PREFIX   = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION  = _env_str("AWS_REGION", "us-east-1")
S3_ACL      = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 2.0)

# speech junk we want to drop when we DO have ASR
BAD_PHRASES = [
    "wait",
    "hold on",
    "lemme start again",
    "let me start again",
    "start over",
    "no no",
    "redo",
    "i mean",
    "actually",
    "sorry",
]
FILLERS = {"uh", "um", "like", "so", "okay"}


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start


# ============================================================
# SHELL HELPERS
# ============================================================

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _tmpfile(suffix: str = ".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _download_to_tmp(url: str) -> str:
    local_path = _tmpfile(suffix=".mp4")
    code, out, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local_path

def _ffprobe_duration(path: str) -> float:
    code, out, err = _run([
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path,
    ])
    if code != 0:
        return 0.0
    try:
        return float(out.strip())
    except Exception:
        return 0.0


# ============================================================
# S3
# ============================================================

def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh,
            S3_BUCKET,
            key,
            ExtraArgs={"ACL": S3_ACL, "ContentType": "video/mp4"},
        )
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}",
    }


# ============================================================
# EXPORT
# ============================================================

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        takes = [Take(id="FALLBACK", start=0.0, end=min(5.0, MAX_DURATION_SEC), text="")]

    parts: List[str] = []
    listfile = _tmpfile(suffix=".txt")

    for idx, t in enumerate(takes, start=1):
        part = _tmpfile(suffix=f".part{idx:02d}.mp4")
        parts.append(part)
        _run([
            FFMPEG_BIN, "-y",
            "-ss", f"{t.start:.3f}",
            "-i", src,
            "-t", f"{t.dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            part
        ])

    with open(listfile, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    final = _tmpfile(suffix=".mp4")
    _run([
        FFMPEG_BIN, "-y",
        "-f", "concat", "-safe", "0",
        "-i", listfile,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        final
    ])
    return final


# ============================================================
# ASR PATH
# ============================================================

def _load_asr_segments(src: str) -> Optional[List[Dict[str, Any]]]:
    try:
        from worker.asr import transcribe_segments
    except Exception:
        return None
    try:
        segs = transcribe_segments(src)
    except Exception:
        return None
    if not segs:
        return None

    # detect our own placeholder to avoid treating it as real ASR
    txt0 = (segs[0].get("text") or "").lower()
    if "temp placeholder" in txt0:
        return None

    return segs


def _segments_to_takes_asr(segs: List[Dict[str, Any]]) -> List[Take]:
    takes: List[Take] = []
    for i, seg in enumerate(segs, start=1):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))

        # split very long spoken bits into <= MAX_TAKE_SEC
        while (e - s) > MAX_TAKE_SEC:
            takes.append(
                Take(
                    id=f"ASR{i:04d}_{len(takes)+1:02d}",
                    start=s,
                    end=s + MAX_TAKE_SEC,
                    text=txt,
                )
            )
            s = s + MAX_TAKE_SEC

        if (e - s) >= MIN_TAKE_SEC:
            takes.append(
                Take(
                    id=f"ASR{i:04d}",
                    start=s,
                    end=e,
                    text=txt,
                )
            )
    return takes


def _is_bad_speech(txt: str) -> bool:
    low = txt.lower().strip()
    if not low:
        return True
    for p in BAD_PHRASES:
        if p in low:
            return True
    words = [w.strip(",.?!") for w in low.split()]
    if not words:
        return True
    filler_count = sum(1 for w in words if w in FILLERS)
    filler_rate = filler_count / max(1, len(words))
    return filler_rate > 0.4


def _dedupe_takes(takes: List[Take]) -> List[Take]:
    out: List[Take] = []
    seen = set()
    for t in takes:
        norm = "".join(c.lower() for c in t.text if (c.isalnum() or c.isspace())).strip()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(t)
    return out


def _merge_adjacent(takes: List[Take], max_gap: float = 1.0, max_chain: int = 3) -> List[Take]:
    if not takes:
        return []
    takes = sorted(takes, key=lambda x: x.start)
    merged: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (j + 1) < len(takes) and len(chain) < max_chain:
            a = chain[-1]
            b = takes[j + 1]
            if (b.start - a.end) > max_gap:
                break
            chain.append(b)
            j += 1
        first = chain[0]
        last = chain[-1]
        merged.append(
            Take(
                id=f"{first.id}_to_{last.id}",
                start=first.start,
                end=last.end,
                text=" ".join(c.text for c in chain),
            )
        )
        i = j + 1
    return merged


# ============================================================
# TIME-BASED FALLBACK
# ============================================================

def _time_based_takes(vid_dur: float) -> List[Take]:
    takes: List[Take] = []
    t = 0.0
    idx = 1
    while t < vid_dur:
        end = min(t + MAX_TAKE_SEC, vid_dur)
        if (end - t) >= MIN_TAKE_SEC:
            takes.append(
                Take(
                    id=f"SEG{idx:04d}",
                    start=t,
                    end=end,
                    text=f"Auto segment {idx} ({t:.1f}s–{end:.1f}s)",
                )
            )
            idx += 1
        t = end
    return takes


# ============================================================
# PUBLIC ENTRY
# ============================================================

def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts,
    max_duration: float,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    # 1) download video
    src = _download_to_tmp(file_urls[0])

    # 2) measure
    real_dur = _ffprobe_duration(src)
    cap = float(max_duration or MAX_DURATION_SEC)
    if real_dur > 0:
        cap = min(cap, real_dur)

    # 3) try ASR path first
    segs = _load_asr_segments(src)
    if segs is not None:
        takes = _segments_to_takes_asr(segs)
        takes = [t for t in takes if not _is_bad_speech(t.text)]
        takes = _dedupe_takes(takes)
        takes = _merge_adjacent(takes)

        # trim to cap
        story: List[Take] = []
        total = 0.0
        for t in takes:
            if total + t.dur > cap:
                break
            story.append(t)
            total += t.dur
        used_asr = True
    else:
        # fallback: time-based
        story = _time_based_takes(cap)
        used_asr = False

    # 4) export
    final_path = _export_concat(src, story)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    # 5) JSON
    clips_block = [
        {
            "id": t.id,
            "slot": "STORY",
            "start": t.start,
            "end": t.end,
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [t.id],
            "text": t.text,
        }
        for t in story
    ]

    slots_block: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if story:
        first = story[0]
        slots_block["HOOK"].append(
            {
                "id": first.id,
                "start": first.start,
                "end": first.end,
                "text": first.text,
                "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [first.id]},
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            }
        )
    if len(story) > 2:
        for mid in story[1:-1]:
            slots_block["FEATURE"].append(
                {
                    "id": mid.id,
                    "start": mid.start,
                    "end": mid.end,
                    "text": mid.text,
                    "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": [mid.id]},
                    "face_q": 1.0,
                    "scene_q": 1.0,
                    "vtx_sim": 0.0,
                    "has_product": False,
                    "ocr_hit": 0,
                }
            )
    if len(story) >= 2:
        last = story[-1]
        slots_block["CTA"].append(
            {
                "id": last.id,
                "start": last.start,
                "end": last.end,
                "text": last.text,
                "meta": {"slot": "CTA", "score": 2.0, "chain_ids": [last.id]},
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            }
        )

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips_block,
        "slots": slots_block,
        "asr": used_asr,
        "semantic": used_asr,   # we actually cleaned text only if ASR is real
        "vision": False,
    }
