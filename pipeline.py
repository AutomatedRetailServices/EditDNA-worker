# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import boto3

# ---------- ENV HELPERS ----------

def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name, "")
    # allow comments like "value # comment"
    val = raw.split("#", 1)[0].strip()
    return val or default

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    raw = raw.split("#", 1)[0].strip()
    try:
        return float(raw)
    except ValueError:
        return default


FFMPEG_BIN = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET  = _env_str("S3_BUCKET", "")
S3_PREFIX  = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION = _env_str("AWS_REGION", "us-east-1")
S3_ACL     = _env_str("S3_ACL", "public-read")

# maxs are just safety, you can tune
MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 1.2)   # a little shorter is ok now

# ---------- RULE LISTS (english-ish) ----------
# these are *before* LLM, just to kill obvious garbage / repeats
DROP_CONTAINS = [
    "so if you wanna check them out",
    "so if you want to check them out",
    "if you want to check them out",
    "grab one of these westland boat",  # bad ASR
    "why can't i remember after that",
]

# words that are *good* for product talk
FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "moisture", "odor", "balance",
    "you only need to take",  # IMG_03
    "probiotic", "gummy", "flavored", "for women"
]

# ---------- DATA CLASS ----------

@dataclass
class Clause:
    id: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return max(0.01, self.end - self.start)

# ---------- SHELL UTILS ----------

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _tmpfile(suffix: str = ".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _download_to_tmp(url: str) -> str:
    path = _tmpfile(".mp4")
    code, out, err = _run(["curl", "-L", "-o", path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return path

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

# ---------- S3 UPLOAD ----------

def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh, S3_BUCKET, key,
            ExtraArgs={"ACL": S3_ACL, "ContentType": "video/mp4"}
        )
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return {"s3_key": key, "s3_url": f"s3://{S3_BUCKET}/{key}", "https_url": https_url}

# ---------- ASR LOADER ----------

def _load_asr_segments(local_video_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    We rely on your existing worker.asr.transcribe_segments(video_path)
    which you already had in your repo/logs.
    """
    try:
        from worker.asr import transcribe_segments
    except Exception:
        return None

    try:
        segs = transcribe_segments(local_video_path)
    except Exception:
        return None

    if not segs:
        return None
    return segs

# ---------- CLAUSE SPLIT ----------

def _split_text_into_clauses(txt: str) -> List[str]:
    """
    Simple splitter: sentence ends, then split on 'and/but' to avoid long rambles.
    """
    txt = " ".join((txt or "").split())
    if not txt:
        return []
    # first split on sentence-ish ends
    tmp: List[str] = []
    buf = ""
    for ch in txt:
        buf += ch
        if ch in ".?!":
            tmp.append(buf.strip())
            buf = ""
    if buf.strip():
        tmp.append(buf.strip())

    clauses: List[str] = []
    for piece in tmp:
        low = piece.lower()
        if " but " in low or " and " in low:
            piece = piece.replace(" but ", "|SPLIT|").replace(" and ", "|SPLIT|")
            for part in piece.split("|SPLIT|"):
                part = part.strip(" ,.;")
                if part:
                    clauses.append(part)
        else:
            piece = piece.strip(" ,.;")
            if piece:
                clauses.append(piece)

    # drop very short
    clauses = [c for c in clauses if len(c.split()) >= 3]
    return clauses

def _assign_times(seg_start: float, seg_end: float, clauses: List[str]) -> List[Clause]:
    dur = max(0.05, seg_end - seg_start)
    joined = " ".join(clauses)
    total_len = max(1, len(joined))
    out: List[Clause] = []
    cursor = 0
    for idx, c in enumerate(clauses, start=1):
        clen = len(c)
        start_rel = cursor / total_len
        end_rel = (cursor + clen) / total_len
        c_start = seg_start + start_rel * dur
        c_end = seg_start + end_rel * dur
        out.append(Clause(id=f"c{idx}", start=c_start, end=c_end, text=c.strip()))
        cursor += clen + 1
    return out

# ---------- LLM SCORING (ALWAYS-ON) ----------

def _llm_score_clause(text: str) -> Dict[str, Any]:
    """
    Always try to send to OpenAI.
    If it fails, we raise and caller will downgrade to rule-only.
    """
    from openai import OpenAI
    client = OpenAI()

    prompt = f"""
You are helping to auto-edit short UGC sales videos.

Given this spoken line, decide:
1. should we KEEP it in the final sales edit?
2. what ROLE is it? (HOOK, FEATURE, CTA, FILLER)
3. are there obvious mistakes like "wait", "I forgot", "why can't I remember".

Return JSON with keys: keep (true/false), slot (HOOK|FEATURE|CTA|FILLER), bad (true/false), reason.

Spoken line: "{text}"
    """.strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=120,
    )

    raw = resp.choices[0].message.content.strip()
    # try to eval as JSON-ish
    import json
    try:
        data = json.loads(raw)
    except Exception:
        # very defensive: if model didn't return JSON, assume keep as FEATURE
        data = {"keep": True, "slot": "FEATURE", "bad": False, "reason": "fallback"}
    return data

# ---------- RULE FALLBACK ----------

def _rule_should_drop(text: str) -> bool:
    low = text.lower()
    for bad in DROP_CONTAINS:
        if bad in low:
            return True
    return False

def _rule_guess_slot(text: str) -> str:
    low = text.lower()
    if "if you don't have" in low or "this is the perfect" in low:
        return "HOOK"
    for h in FEATURE_HINTS:
        if h in low:
            return "FEATURE"
    if low.startswith("click") or low.startswith("go to") or "i left it for you" in low:
        return "CTA"
    return "FEATURE"

# ---------- EXPORT (CONCAT) ----------

def _export_concat(src: str, clauses: List[Clause]) -> str:
    if not clauses:
        # fallback: just take first 4s
        clauses = [Clause(id="FALLBACK", start=0.0, end=4.0, text="")]
    listfile = _tmpfile(".txt")
    parts: List[str] = []

    for idx, cl in enumerate(clauses, start=1):
        part_path = _tmpfile(f".part{idx:02d}.mp4")
        parts.append(part_path)
        dur = max(0.05, cl.dur)
        _run([
            FFMPEG_BIN, "-y",
            "-ss", f"{cl.start:.3f}",
            "-i", src,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            part_path
        ])

    with open(listfile, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    final_path = _tmpfile(".mp4")
    _run([
        FFMPEG_BIN, "-y",
        "-f", "concat", "-safe", "0",
        "-i", listfile,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        final_path
    ])
    return final_path

# ---------- MAIN ENTRY ----------

def run_pipeline(
    *,
    session_id: str,
    # we support BOTH, so tasks.py can pass local_video_path like before
    local_video_path: Optional[str] = None,
    file_urls: Optional[List[str]] = None,
    s3_prefix: Optional[str] = None,
    max_duration: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:

    # 1) get source path
    if local_video_path and os.path.exists(local_video_path):
        src = local_video_path
    else:
        if not file_urls:
            return {"ok": False, "error": "no video input"}
        src = _download_to_tmp(file_urls[0])

    real_dur = _ffprobe_duration(src)
    cap = float(max_duration or MAX_DURATION_SEC)
    if real_dur > 0:
        cap = min(cap, real_dur)

    # 2) ASR
    segs = _load_asr_segments(src)
    if not segs:
        # no ASR → just export first 6s
        final = _export_concat(src, [Clause(id="FALLBACK", start=0.0, end=min(6.0, cap), text="")])
        up = _upload_to_s3(final, s3_prefix=s3_prefix)
        return {
            "ok": True,
            "session_id": session_id,
            "input_local": src,
            "duration_sec": _ffprobe_duration(final),
            **up,
            "clips": [],
            "slots": {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []},
            "asr": False,
            "semantic": False,
            "vision": False,
        }

    # 3) turn ASR segments into clauses
    all_clauses: List[Clause] = []
    for idx, seg in enumerate(segs, start=1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        if (e - s) > MAX_TAKE_SEC:
            e = s + MAX_TAKE_SEC
        # split into clauses
        clauses_txt = _split_text_into_clauses(text)
        if not clauses_txt:
            continue
        # map to times
        clauses_timed = _assign_times(s, e, clauses_txt)
        # give them stable ids
        for cidx, cl in enumerate(clauses_timed, start=1):
            cl.id = f"ASR{idx:04d}_c{cidx}"
            all_clauses.append(cl)

    # 4) LLM + rules decide which ones to keep
    kept: List[Dict[str, Any]] = []
    total_time = 0.0
    for cl in all_clauses:
        if total_time >= cap:
            break

        # rule pre-drop
        if _rule_should_drop(cl.text):
            continue

        # always try LLM
        keep = False
        slot = "FEATURE"
        bad = False
        try:
            llm_res = _llm_score_clause(cl.text)
            keep = bool(llm_res.get("keep", False))
            slot = (llm_res.get("slot") or "FEATURE").upper()
            bad = bool(llm_res.get("bad", False))
        except Exception:
            # LLM failed → fall back to rules
            slot = _rule_guess_slot(cl.text)
            keep = True
            bad = False

        if not keep or bad:
            continue

        # obey cap
        if total_time + cl.dur > cap:
            break

        kept.append({
            "clause": cl,
            "slot": slot,
        })
        total_time += cl.dur

    # 5) if we somehow didn't keep any, take first 4s
    if not kept:
        final = _export_concat(src, [Clause(id="FALLBACK", start=0.0, end=min(4.0, cap), text="")])
        up = _upload_to_s3(final, s3_prefix=s3_prefix)
        return {
            "ok": True,
            "session_id": session_id,
            "input_local": src,
            "duration_sec": _ffprobe_duration(final),
            **up,
            "clips": [],
            "slots": {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []},
            "asr": True,
            "semantic": True,
            "vision": False,
        }

    # 6) export video
    final_path = _export_concat(src, [k["clause"] for k in kept])
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    # 7) build json response
    def _short(t: str, n: int = 220) -> str:
        return t if len(t) <= n else t[:n].rstrip() + "..."

    clips = []
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    for item in kept:
        cl: Clause = item["clause"]
        slot = item["slot"]
        clip = {
            "id": cl.id,
            "slot": "STORY",
            "start": cl.start,
            "end": cl.end,
            "score": 2.5,
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [cl.id],
            "text": _short(cl.text),
        }
        clips.append(clip)

        slot_dict = {
            "id": cl.id,
            "start": cl.start,
            "end": cl.end,
            "text": _short(cl.text),
            "meta": {"slot": slot, "score": 2.0, "chain_ids": [cl.id]},
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "has_product": False,
            "ocr_hit": 0,
        }
        slots.setdefault(slot, []).append(slot_dict)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src,
        "duration_sec": _ffprobe_duration(final_path),
        **up,
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,
    }
