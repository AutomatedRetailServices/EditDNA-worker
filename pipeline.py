# /workspace/EditDNA-worker/pipeline.py

from __future__ import annotations
import os
import re
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import boto3

# ------------------------------------------------------------
# 0) ENV HELPERS (fixed parsing)
# ------------------------------------------------------------

def _env_raw(k: str) -> str:
    """
    Read env var, strip inline comments and whitespace.
    e.g. MAX_DURATION_SEC="220" # sec  -> "220"
    """
    v = os.getenv(k, "")
    if not v:
        return ""
    v = v.split("#", 1)[0].strip()
    return v

def _clean_num_str(s: str) -> str:
    s = s.strip()
    # remove surrounding quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

def _env_str(k: str, d: str) -> str:
    v = _env_raw(k)
    return v or d

def _env_float(k: str, d: float) -> float:
    v = _env_raw(k)
    if not v:
        return d
    v = _clean_num_str(v)
    try:
        return float(v)
    except Exception:
        # try to extract first number
        m = re.search(r"[-+]?\d+(\.\d+)?", v)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                pass
        return d

def _env_bool(k: str, d: bool = False) -> bool:
    v = _env_raw(k).lower()
    if not v:
        return d
    return v in ("1", "true", "yes", "on")


# ------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------

FFMPEG_BIN  = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET   = _env_str("S3_BUCKET", "")
S3_PREFIX   = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION  = _env_str("AWS_REGION", "us-east-1")
S3_ACL      = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 2.0)

# LLM config
OPENAI_API_KEY   = _env_str("OPENAI_API_KEY", "")
LLM_MODEL        = _env_str("LLM_MODEL", "gpt-4o-mini")  # change if you want
REQUIRE_LLM      = _env_bool("REQUIRE_LLM", False)       # if True and no key -> error
LLM_KEEP_MIN     = _env_float("LLM_KEEP_MIN", 0.35)      # min combined score required


# ------------------------------------------------------------
# 2) TEXT RULES (your scope)
# ------------------------------------------------------------

BAD_PHRASES = [
    "wait",
    "hold on",
    "lemme start again",
    "let me start again",
    "start over",
    "no no",
    "redo",
    "sorry",
]

CTA_FLUFF = [
    "click the link",
    "get yours today",
    "go ahead and click",
    "go ahead and grab",
    "i left it down below",
    "i left it for you down below",
    "grab one of these",
    "if you want to check them out",
]

UGLY_BRANCHES = [
    "but if you don't like the checker print",
    "but if you don't like the checker",
    "but if you do",
    "but if you don't",
    "but if you",
]

FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "comes with", "it has", "it also has",
    "it's actually", "this isn't just", "design",
]


# ------------------------------------------------------------
# 3) DATA MODEL
# ------------------------------------------------------------

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start


# ------------------------------------------------------------
# 4) SHELL / VIDEO HELPERS
# ------------------------------------------------------------

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _tmpfile(suffix: str = ".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

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


# ------------------------------------------------------------
# 5) S3
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 6) ASR LOADING (match your current worker/asr.py)
# ------------------------------------------------------------

def _load_asr_segments_from_local(path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Your asr.py has transcribe() and transcribe_local(). We'll try both.
    """
    try:
        from worker import asr as asr_mod
    except Exception:
        print("[pipeline] could not import worker.asr -> no ASR", flush=True)
        return None

    segs = None
    if hasattr(asr_mod, "transcribe_local"):
        segs = asr_mod.transcribe_local(path)
    else:
        segs = asr_mod.transcribe(path)

    if not segs:
        return None
    return segs


# ------------------------------------------------------------
# 7) CLAUSE UTILITIES (same logic you liked for IMG_02)
# ------------------------------------------------------------

def _split_into_clauses(text: str) -> List[str]:
    if not text:
        return []
    text = " ".join(text.split())
    temp: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".?!":
            temp.append(buf.strip())
            buf = ""
    if buf.strip():
        temp.append(buf.strip())

    clauses: List[str] = []
    for piece in temp:
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

    clauses = [c for c in clauses if len(c.split()) >= 3]
    return clauses

def _rule_score_clause(c: str) -> float:
    """
    1.0 = looks like nice product line
    0.0 = looks like trash / retake / branch we don't want
    """
    low = c.lower().strip()
    if not low:
        return 0.0

    # hard bad
    for p in BAD_PHRASES:
        if p in low:
            return 0.0

    # ugly branches
    for p in UGLY_BRANCHES:
        if p in low:
            return 0.1

    # good featurey
    for h in FEATURE_HINTS:
        if h in low:
            return 0.9

    # CTA lines are okay but not the main meat
    for p in CTA_FLUFF:
        if p in low:
            return 0.6

    # fallback
    return 0.5

# ------------------------------------------------------------
# 8) OPTIONAL LLM SCORING
# ------------------------------------------------------------

_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
        print("[pipeline] LLM scoring ENABLED", flush=True)
    except Exception as e:
        print(f"[pipeline] could not init OpenAI client: {e}", flush=True)
else:
    if REQUIRE_LLM:
        raise RuntimeError("REQUIRE_LLM=1 but OPENAI_API_KEY is not set")
    print("[pipeline] LLM scoring DISABLED (no OPENAI_API_KEY)", flush=True)


def _llm_score_clause(text: str) -> float:
    """
    Ask the model: is this sentence useful for a short product / sales video?
    Return 0.0-1.0. If LLM not available, return 0.0 (we'll use rule score).
    """
    if _client is None:
        return 0.0

    prompt = (
        "You are rating lines from a short social-media sales video. "
        "Score 0 to 1. 1 = keep (on-script, product/features/benefit/CTA), "
        "0 = drop (restart, filler, branch, irrelevant). "
        f"Line: {text!r} "
        "Just return the number."
    )
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip()
        # try to extract float
        m = re.search(r"(\d+(\.\d+)?)", raw)
        if not m:
            return 0.0
        val = float(m.group(1))
        # clamp
        if val < 0.0: val = 0.0
        if val > 1.0: val = 1.0
        return val
    except Exception as e:
        print(f"[pipeline] LLM scoring failed: {e}", flush=True)
        return 0.0


# ------------------------------------------------------------
# 9) TIME DISTRIBUTION INSIDE SEGMENT
# ------------------------------------------------------------

def _assign_times_to_clauses(seg_start: float, seg_end: float, clauses: List[str]) -> List[Tuple[float, float, str]]:
    dur = max(0.05, seg_end - seg_start)
    joined = " ".join(clauses)
    total_len = max(1, len(joined))
    out: List[Tuple[float, float, str]] = []
    cursor = 0
    for c in clauses:
        c_len = len(c)
        start_rel = cursor / total_len
        end_rel = (cursor + c_len) / total_len
        c_start = seg_start + start_rel * dur
        c_end = seg_start + end_rel * dur
        out.append((c_start, c_end, c.strip()))
        cursor += c_len + 1
    return out


# ------------------------------------------------------------
# 10) TEXT CLEANUP
# ------------------------------------------------------------

def _trim_repeated_ngrams(txt: str, n: int = 4) -> str:
    words = txt.split()
    if len(words) <= n * 2:
        return txt
    seen = {}
    for i in range(0, len(words) - n + 1):
        key = " ".join(w.lower() for w in words[i:i+n])
        if key in seen:
            return " ".join(words[:i]).rstrip(" ,.;")
        seen[key] = i
    return txt

def _trim_cta_fluff(txt: str) -> str:
    low = txt.lower()
    for p in CTA_FLUFF:
        idx = low.find(p)
        if idx != -1:
            return txt[:idx].rstrip(" ,.;")
    return txt

def _clean_text(txt: str) -> str:
    txt = _trim_repeated_ngrams(txt, n=4)
    txt = _trim_cta_fluff(txt)
    return txt.strip()


# ------------------------------------------------------------
# 11) FALLBACK TIME-BASED SEGMENTS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 12) EXPORT CONCAT VIDEO
# ------------------------------------------------------------

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        # fallback to first 5s
        takes = [Take(id="FALLBACK", start=0.0, end=min(5.0, MAX_DURATION_SEC), text="")]
    listfile = _tmpfile(".txt")
    parts: List[str] = []
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


# ------------------------------------------------------------
# 13) PUBLIC ENTRY (this is what tasks.py calls)
# ------------------------------------------------------------

def run_pipeline(
    *,
    local_video_path: str,
    session_id: str,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    print("[pipeline] filtered pipeline ACTIVE", flush=True)

    src = local_video_path
    real_dur = _ffprobe_duration(src)
    cap = min(MAX_DURATION_SEC, real_dur) if real_dur > 0 else MAX_DURATION_SEC

    # 1) ASR
    segs = _load_asr_segments_from_local(src)

    if segs:
        # turn segments into fine clause-takes
        clause_takes: List[Take] = []

        for i, seg in enumerate(segs, start=1):
            seg_text = (seg.get("text") or "").strip()
            seg_start = float(seg.get("start", 0.0))
            seg_end   = float(seg.get("end", seg_start))

            if not seg_text:
                continue

            clauses = _split_into_clauses(seg_text)
            if not clauses:
                continue

            # distribute times
            timed_clauses = _assign_times_to_clauses(seg_start, seg_end, clauses)

            for c_idx, (cs, ce, ctext) in enumerate(timed_clauses, start=1):
                ctext_clean = _clean_text(ctext)
                if not ctext_clean:
                    continue

                # rule score
                r_score = _rule_score_clause(ctext_clean)
                # llm score
                l_score = _llm_score_clause(ctext_clean)
                combined = max(r_score, l_score)

                if combined < LLM_KEEP_MIN:
                    # drop this clause
                    continue

                # clamp duration
                dur = ce - cs
                if dur > MAX_TAKE_SEC:
                    ce = cs + MAX_TAKE_SEC
                    dur = ce - cs
                if dur < MIN_TAKE_SEC:
                    # too tiny, skip
                    continue

                clause_takes.append(
                    Take(
                        id=f"ASR{i:04d}_c{c_idx}",
                        start=cs,
                        end=ce,
                        text=ctext_clean,
                    )
                )

        # order and cap total duration
        clause_takes = sorted(clause_takes, key=lambda x: x.start)
        story: List[Take] = []
        total = 0.0
        for t in clause_takes:
            if total + t.dur > cap:
                break
            story.append(t)
            total += t.dur
        used_asr = True

    else:
        # no ASR -> simple time slicing
        story = _time_based_takes(cap)
        used_asr = False

    # 2) export
    final_path = _export_concat(src, story)

    # 3) upload
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    # 4) build response (similar to what your worker expects)
    def _trim(txt: str, n: int = 220) -> str:
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    clips = [
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
        for t in story
    ]

    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if story:
        first = story[0]
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

    if len(story) > 2:
        for mid in story[1:-1]:
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

    if len(story) >= 2:
        last = story[-1]
        slots["CTA"].append({
            "id": last.id,
            "start": last.start,
            "end": last.end,
            "text": _trim(last.text),
            "meta": {"slot": "CTA", "score": 2.0, "chain_ids": [last.id]},
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "has_product": False,
            "ocr_hit": 0,
        })

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips,
        "slots": slots,
        "asr": used_asr,
        "semantic": used_asr,
        "vision": False,
    }
