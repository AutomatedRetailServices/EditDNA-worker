# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import boto3
import cv2

# try to import your ASR helper (you already have worker/asr.py in this repo)
try:
    from worker.asr import transcribe_segments  # must return list of {start, end, text}
except Exception:
    transcribe_segments = None

# ================== ENV / CONSTANTS ==================

def _env_str(k: str, d: str) -> str:
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d

FFMPEG_BIN  = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET   = _env_str("S3_BUCKET", "")
S3_PREFIX   = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION  = _env_str("AWS_REGION", "us-east-1")
S3_ACL      = _env_str("S3_ACL", "public-read")

# max duration is now a soft cap; we can override per call
DEFAULT_MAX_DURATION_SEC = 90.0  # not 40 fixed

# rule lists — IMG_02 and IMG_03 merged
BAD_PHRASES = [
    "wait",
    "hold on",
    "lemme start again",
    "let me start again",
    "start over",
    "no no",
    "redo",
    "sorry",
    "why can't i remember",            # from img_03
    "not moisture control",            # messy correction
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
    "if you wanna check them out",
]

UGLY_BRANCHES = [
    "but if you don't like the checker",
    "but if you don't like the checkered",
    "but if you don't like the plain black strap",  # sometimes we keep, but CTA-ish
    "why can't i remember",
]

FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "moisture", "odor", "odor control", "healthy balance", "probiotic",
    "comes with", "it has", "it also has",
    "it's actually", "this isn't just", "design",
]

# ========== OpenAI (always-on) ==========
# user said: don't make it optional
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required for this pipeline (we run LLM on every clause).")

from openai import OpenAI
_openai_client = OpenAI(api_key=OPENAI_API_KEY)

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ================== DATACLASS ==================

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    face_q: float = 1.0
    llm_score: float = 0.0
    rule_score: float = 0.0

    @property
    def dur(self) -> float:
        return self.end - self.start


# ================== SHELL HELPERS ==================

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


# ================== S3 ==================

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


# ================== EXPORT (FFMPEG CONCAT) ==================

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        # fallback: 3s from start
        takes = [Take(id="FALLBACK", start=0.0, end=3.0, text="")]

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


# ================== VISION (simple face detect) ==================

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)

def _face_score_for_frame(frame_bgr) -> float:
    if frame_bgr is None:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return 1.0 if len(faces) > 0 else 0.0

def _vision_score_for_clip(video_path: str, start: float, end: float) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mid = (start + end) / 2.0
    frame_id = int(mid * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return 0.0
    return _face_score_for_frame(frame)


# ================== ASR → TAKES ==================

def _load_asr_segments(src: str) -> Optional[List[Dict[str, Any]]]:
    if transcribe_segments is None:
        return None
    try:
        segs = transcribe_segments(src)
    except Exception:
        return None
    if not segs:
        return None
    return segs

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


# ================== RULE FILTERS ==================

def _clause_is_bad(c: str) -> bool:
    low = c.lower().strip()
    if not low:
        return True
    for p in BAD_PHRASES:
        if p in low:
            return True
    for p in UGLY_BRANCHES:
        if p in low:
            return True
    if len(low.split()) < 3:
        return True
    return False

def _clause_is_ctaish(c: str) -> bool:
    low = c.lower()
    for p in CTA_FLUFF:
        if p in low:
            return True
    if low.startswith("if you want to") or low.startswith("if you wanna"):
        return True
    return False

def _clause_is_featurey(c: str) -> bool:
    low = c.lower()
    for h in FEATURE_HINTS:
        if h in low:
            return True
    return False

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


# ================== LLM FILTER ==================

def _llm_score_clause(clause: str) -> float:
    """
    Always-on: we call OpenAI here.
    We return 1.0 for keep, 0.0 for drop.
    """
    prompt = (
        "You are an editor for short sales videos. "
        "You are given ONE spoken line from a creator. "
        "Decide if the line is a good, on-script, persuasive line to KEEP in the final video.\n\n"
        f"LINE: \"{clause}\"\n\n"
        "Reply with exactly one word: KEEP or DROP."
    )
    resp = _openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You judge sales lines."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=2,
    )
    ans = resp.choices[0].message.content.strip().upper()
    return 1.0 if "KEEP" in ans else 0.0


# ================== FALLBACK (no ASR) ==================

def _time_based_takes(vid_dur: float) -> List[Take]:
    takes: List[Take] = []
    t = 0.0
    idx = 1
    while t < vid_dur:
        end = min(t + 5.0, vid_dur)
        takes.append(Take(id=f"AUTO{idx:04d}", start=t, end=end, text=f"auto segment {idx}"))
        t = end
        idx += 1
    return takes


# ================== PUBLIC ENTRY ==================

def run_pipeline(
    *,
    session_id: str,
    local_video_path: Optional[str] = None,
    file_urls: Optional[List[str]] = None,
    s3_prefix: Optional[str] = None,
    max_duration: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Entry point used by tasks.py
    - tasks.py is currently calling: run_pipeline(local_video_path=..., session_id=..., s3_prefix=...)
    - API could call: run_pipeline(file_urls=[...], session_id=...)
    We support both.
    """
    if local_video_path:
        src = local_video_path
    elif file_urls:
        src = _download_to_tmp(file_urls[0])
    else:
        return {"ok": False, "error": "no video input"}

    real_dur = _ffprobe_duration(src)
    cap = float(max_duration or DEFAULT_MAX_DURATION_SEC)
    if real_dur > 0:
        cap = min(cap, real_dur)

    segs = _load_asr_segments(src)
    used_asr = segs is not None

    # build candidate clauses
    clause_takes: List[Take] = []

    if segs:
        # last seg id to allow CTA at the end
        for idx, seg in enumerate(segs, start=1):
            seg_txt = (seg.get("text") or "").strip()
            if not seg_txt:
                continue
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            clauses = _split_into_clauses(seg_txt)
            if not clauses:
                continue
            timed = _assign_times_to_clauses(seg_start, seg_end, clauses)
            for c_idx, (cs, ce, ctext) in enumerate(timed, start=1):
                ctext = _clean_text(ctext)
                if not ctext:
                    continue

                # RULE filter (first gate)
                rule_bad = _clause_is_bad(ctext)
                rule_cta = _clause_is_ctaish(ctext)
                rule_feature = _clause_is_featurey(ctext)

                # always run LLM — if OpenAI is down, this will raise → job fails
                llm_keep = _llm_score_clause(ctext)

                # decide: we keep if (not rule_bad) AND llm_keep
                if rule_bad and not rule_feature:
                    continue
                if llm_keep <= 0.5:
                    continue

                take = Take(
                    id=f"ASR{idx:04d}_c{c_idx}",
                    start=cs,
                    end=ce,
                    text=ctext,
                    llm_score=llm_keep,
                    rule_score=0.0 if rule_bad else 1.0,
                )

                # VISION gate
                face_q = _vision_score_for_clip(src, cs, ce)
                if face_q < 0.35:  # tune
                    continue
                take.face_q = face_q

                clause_takes.append(take)
    else:
        # no ASR → dumb split
        clause_takes = _time_based_takes(cap)

    # sort by time and cap total duration
    clause_takes = sorted(clause_takes, key=lambda x: x.start)

    story: List[Take] = []
    total_dur = 0.0
    for t in clause_takes:
        if total_dur + t.dur > cap:
            break
        story.append(t)
        total_dur += t.dur

    # if we somehow got nothing, just take first 3s
    if not story:
        story = [Take(id="FALLBACK", start=0.0, end=min(3.0, cap), text="")]

    # export
    final_path = _export_concat(src, story)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    def _trim(txt: str, n: int = 220) -> str:
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    clips = [
        {
            "id": t.id,
            "slot": "STORY",
            "start": t.start,
            "end": t.end,
            "score": 2.5,
            "face_q": t.face_q,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
            "chain_ids": [t.id],
            "text": _trim(t.text),
        }
        for t in story
    ]

    # slots: first = HOOK, middle = FEATURE, last = CTA
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
            "face_q": first.face_q,
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
                "face_q": mid.face_q,
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
            "face_q": last.face_q,
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
        "semantic": True,
        "vision": True,
    }
