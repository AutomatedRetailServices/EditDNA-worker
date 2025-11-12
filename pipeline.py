# /workspace/EditDNA-worker/pipeline.py

from __future__ import annotations
import os
import uuid
import time
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import boto3

# -------------------------
# ENV + CONSTANTS
# -------------------------

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

# per-snippet bounds
MAX_TAKE_SEC = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC = _env_float("MIN_TAKE_SEC", 1.2)

# always try LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# obvious bad stuff
BAD_PHRASES = [
    "wait",
    "hold on",
    "lemme start again",
    "let me start again",
    "start over",
    "no no",
    "redo",
    "sorry",
    "why can't i remember",
    "why cant i remember",
]

REPEATY_PHRASES = [
    "if you want to check them out",
    "so if you want to check them out",
    "you can't worry no more because i found",
]

FEATURE_HINTS = [
    "pocket", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "it has", "it also has", "it's actually",
    "moisture", "odor", "balance", "gummy", "probiotic",
    "bag", "bum bag", "checkered", "colors available",
]


@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    rule_score: float = 0.0
    llm_score: float = 0.0

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


# -------------------------
# SHELL / IO
# -------------------------

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _tmpfile(suffix: str = ".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _download_to_tmp(url: str) -> str:
    local_path = _tmpfile(".mp4")
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


# -------------------------
# S3
# -------------------------

def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    s3 = boto3.client("s3", region_name=AWS_REGION)
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


# -------------------------
# CONCAT
# -------------------------

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        takes = [Take(id="FALLBACK", start=0.0, end=5.0, text="")]

    parts: List[str] = []
    listfile = _tmpfile(".txt")

    for idx, t in enumerate(takes, start=1):
        part = _tmpfile(f".part{idx:02d}.mp4")
        parts.append(part)
        dur = max(0.05, t.dur)
        _run([
            FFMPEG_BIN, "-y",
            "-ss", f"{t.start:.3f}",
            "-i", src,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            part
        ])

    with open(listfile, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    final = _tmpfile(".mp4")
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


# -------------------------
# ASR
# -------------------------

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
    return segs


# -------------------------
# TEXT → CLAUSES
# -------------------------

def _split_into_clauses(text: str) -> List[str]:
    if not text:
        return []
    text = " ".join(text.split())

    tmp: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".?!":
            tmp.append(buf.strip())
            buf = ""
    if buf.strip():
        tmp.append(buf.strip())

    clauses: List[str] = []
    for piece in tmp:
        piece = piece.replace(" but ", "|SPLIT|").replace(" and ", "|SPLIT|")
        for part in piece.split("|SPLIT|"):
            part = part.strip(" ,.;")
            if part and len(part.split()) >= 3:
                clauses.append(part)
    return clauses


def _assign_times_to_clauses(seg_start: float, seg_end: float, clauses: List[str]) -> List[Tuple[float, float, str]]:
    dur = max(0.05, seg_end - seg_start)
    joined = " ".join(clauses)
    total_len = max(1, len(joined))
    out: List[Tuple[float, float, str]] = []
    cursor = 0
    for c in clauses:
        cl = len(c)
        start_rel = cursor / total_len
        end_rel = (cursor + cl) / total_len
        c_start = seg_start + start_rel * dur
        c_end = seg_start + end_rel * dur
        out.append((c_start, c_end, c.strip()))
        cursor += cl + 1
    return out


# -------------------------
# RULE SCORE
# -------------------------

def _rule_score_clause(c: str) -> float:
    low = c.lower()
    for b in BAD_PHRASES:
        if b in low:
            return 0.0
    for r in REPEATY_PHRASES:
        if r in low:
            return 0.25
    score = 0.4
    for h in FEATURE_HINTS:
        if h in low:
            score += 0.25
    return min(score, 1.0)


# -------------------------
# LLM SCORE (always try)
# -------------------------

def _llm_score_clause(c: str) -> float:
    if not OPENAI_API_KEY:
        return 0.0
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "You judge a single spoken line from a user-generated sales video.\n"
            "Return ONLY a number 0-1 where:\n"
            "0 = bad take / mistake / filler / self-correction / off-topic\n"
            "1 = helpful sales line (benefit, feature, why buy, reassurance, CTA-ish)\n\n"
            f"LINE: {c}\n"
            "NUMBER:"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        txt = resp.choices[0].message.content.strip()
        try:
            val = float(txt)
            if val < 0.0:
                val = 0.0
            if val > 1.0:
                val = 1.0
            return val
        except Exception:
            return 0.0
    except Exception:
        return 0.0


# -------------------------
# PIPELINE
# -------------------------

def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool = False,
    funnel_counts: Any = None,
    max_duration: Optional[float] = None,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:

    if not file_urls:
        return {"ok": False, "error": "no input files"}

    src = _download_to_tmp(file_urls[0])
    real_dur = _ffprobe_duration(src)

    # you said: "no limits" → so by default we allow the whole video
    # BUT if caller passed max_duration we obey that
    if max_duration:
        cap = min(float(max_duration), real_dur if real_dur > 0 else float(max_duration))
    else:
        cap = real_dur  # full video duration

    segs = _load_asr_segments(src)
    takes: List[Take] = []

    if segs:
        for seg in segs:
            seg_text = (seg.get("text") or "").strip()
            if not seg_text:
                continue
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            clauses = _split_into_clauses(seg_text)
            if not clauses:
                continue
            timed = _assign_times_to_clauses(seg_start, seg_end, clauses)
            for idx, (cs, ce, ctext) in enumerate(timed, start=1):
                dur = ce - cs
                if dur < 0.15:
                    continue
                rscore = _rule_score_clause(ctext)
                lscore = _llm_score_clause(ctext)
                takes.append(
                    Take(
                        id=f"ASR{seg.get('id','') or ''}_c{idx}",
                        start=cs,
                        end=ce,
                        text=ctext,
                        rule_score=rscore,
                        llm_score=lscore,
                    )
                )
    else:
        # ASR missing → time-based chunks
        t = 0.0
        idx = 1
        while t < cap:
            end = min(t + 5.0, cap)
            takes.append(
                Take(
                    id=f"AUTO{idx}",
                    start=t,
                    end=end,
                    text=f"auto segment {idx}",
                    rule_score=0.5,
                    llm_score=0.0,
                )
            )
            idx += 1
            t = end

    # pick clips in time order
    chosen: List[Take] = []
    total = 0.0
    MIN_COMBINED = 0.35  # keep it permissive

    for t in sorted(takes, key=lambda x: x.start):
        combined = 0.6 * t.llm_score + 0.4 * t.rule_score
        if combined < MIN_COMBINED:
            continue
        if total + t.dur > cap + 0.5:
            break
        chosen.append(t)
        total += t.dur

    final_path = _export_concat(src, chosen)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    def _trim(txt: str, n: int = 200) -> str:
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    clips_out = [
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
            "text": _trim(t.text),
        }
        for t in chosen
    ]

    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if chosen:
        first = chosen[0]
        slots["HOOK"].append({
            "id": first.id,
            "start": first.start,
            "end": first.end,
            "text": _trim(first.text),
            "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [first.id]},
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "has_product": False,
            "ocr_hit": 0,
        })
        for mid in chosen[1:]:
            slots["FEATURE"].append({
                "id": mid.id,
                "start": mid.start,
                "end": mid.end,
                "text": _trim(mid.text),
                "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": [mid.id]},
                "face_q": 1.0,
                "scene_q": 1.0,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            })
        # no automatic CTA injection

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips_out,
        "slots": slots,
        "asr": bool(segs),
        "semantic": True,
        "vision": True,
    }
