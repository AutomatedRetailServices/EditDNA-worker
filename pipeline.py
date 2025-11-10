# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations

import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# ---------- optional deps ----------
try:
    import cv2  # for vision scoring
except Exception:  # pragma: no cover
    cv2 = None

try:
    # openai 1.x/2.x style
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

# ---------- environment helpers ----------

def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    if not raw:
        return default
    # strip comments, spaces, and quotes
    val = raw.split("#")[0].strip().strip('"').strip("'")
    return val or default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    # remove comment, quotes, etc.
    raw = raw.split("#")[0].strip().strip('"').strip("'")
    try:
        return float(raw)
    except Exception:
        return default


FFMPEG_BIN = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET = _env_str("S3_BUCKET", "")
S3_PREFIX = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION = _env_str("AWS_REGION", "us-east-1")
S3_ACL = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MAX_TAKE_SEC = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC = _env_float("MIN_TAKE_SEC", 2.0)

# ---------- rule lists (from your good IMG_02 logic) ----------

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
    "so if you wanna check them out",
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
    "moisture", "odor", "odor control", "healthy balance",  # img_03 style
]

CTA_HINTS = [
    "grab one",
    "get yours",
    "link below",
    "down below",
    "check them out",
    "take two a day",
    "ladies",
    "made just for women",
]

# ---------- data model ----------

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    rule_score: float = 1.0
    llm_score: float = 0.0
    vision_score: float = 1.0

    @property
    def dur(self) -> float:
        return self.end - self.start


# ---------- shell helpers ----------

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()


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


def _tmpfile(suffix: str = ".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p


# ---------- S3 upload ----------

def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    import boto3  # safe to import here
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


# ---------- ASR loader (uses your worker.asr) ----------

def _load_asr_segments(local_video_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Your asr.py exposes transcribe_local(path) -> list[{text,start,end}]
    """
    try:
        from worker import asr
    except Exception:
        return None

    try:
        segs = asr.transcribe_local(local_video_path)
    except Exception:
        return None

    if not segs:
        return None
    return segs


# ---------- clause utils ----------

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


# ---------- rule scoring ----------

def _rule_score_clause(c: str) -> float:
    low = c.lower().strip()
    if not low:
        return 0.0

    # obvious bad
    for p in BAD_PHRASES:
        if p in low:
            return 0.0

    # super short
    if len(low.split()) < 3:
        return 0.1

    # ugly branches → low but not always 0 (some videos talk like this)
    for p in UGLY_BRANCHES:
        if p in low:
            return 0.3

    # featurey → high
    for h in FEATURE_HINTS:
        if h in low:
            return 0.9

    # CTA-ish → high
    for h in CTA_HINTS:
        if h in low:
            return 0.85

    # no obvious signals → mid
    return 0.6


def _looks_cta(c: str) -> bool:
    low = c.lower()
    for h in CTA_HINTS:
        if h in low:
            return True
    for h in CTA_FLUFF:
        if h in low:
            return True
    return False


# ---------- LLM scoring (optional but we want it ON) ----------

def _get_llm_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if not OpenAI:
        return None
    client = OpenAI(api_key=api_key)
    return client


def _score_clause_llm(client: Any, clause: str) -> float:
    """
    Return 0.0–1.0: how much this sounds like on-script, producty, usable UGC.
    We keep prompt small so it’s cheap.
    """
    try:
        resp = client.chat.completions.create(
            model=os.getenv("EDITDNA_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You score short spoken lines from UGC sales videos. "
                        "Return only a number 0.0 to 1.0. 1.0 = on-script, product, clear. "
                        "0.0 = mistake, restart, meta, forgetting lines."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Line: {clause}",
                },
            ],
            temperature=0.0,
            max_tokens=5,
        )
        text = resp.choices[0].message.content.strip()
        # try to parse float
        try:
            val = float(text)
            return max(0.0, min(1.0, val))
        except Exception:
            # if model replied with words
            if "restart" in text.lower() or "mistake" in text.lower():
                return 0.1
            return 0.5
    except Exception:
        return 0.5


# ---------- vision scoring (optional) ----------

def _score_clause_vision(video_path: str, start: float, end: float) -> float:
    if cv2 is None:
        return 1.0  # no vision installed, don’t block

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 1.0
        mid = (start + end) / 2.0
        # get fps
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = int(mid * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return 0.6
        # simple brightness check
        mean_val = float(frame.mean())
        if mean_val < 5:   # too dark / black
            return 0.3
        return 1.0
    except Exception:
        return 1.0


# ---------- concat export ----------

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        # fallback: first 5 seconds
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


# ---------- main pipeline ----------

def run_pipeline(
    *,
    local_video_path: str,
    session_id: str,
    s3_prefix: Optional[str] = None,
    portrait: Any = None,
    funnel_counts: Any = None,
    max_duration: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Called from tasks.py like:
        pipeline.run_pipeline(local_video_path=..., session_id=..., s3_prefix=...)
    """
    src = local_video_path
    real_dur = _ffprobe_duration(src)
    cap = float(max_duration or MAX_DURATION_SEC)
    if real_dur > 0:
        cap = min(cap, real_dur)

    segs = _load_asr_segments(src)
    used_asr = segs is not None

    llm_client = _get_llm_client()

    clause_takes: List[Take] = []

    if segs:
        # go segment → clauses → score
        for i, seg in enumerate(segs, start=1):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", s))
            if (e - s) > MAX_TAKE_SEC:
                e = s + MAX_TAKE_SEC

            clauses = _split_into_clauses(text)
            if not clauses:
                # treat whole seg as one clause
                clauses = [text]

            timed = _assign_times_to_clauses(s, e, clauses)
            for c_idx, (cs, ce, ctext) in enumerate(timed, start=1):
                rule_score = _rule_score_clause(ctext)
                llm_score = _score_clause_llm(llm_client, ctext) if llm_client else 0.0
                vision_score = _score_clause_vision(src, cs, ce)
                take_id = f"ASR{i:04d}_c{c_idx}"
                clause_takes.append(
                    Take(
                        id=take_id,
                        start=cs,
                        end=ce,
                        text=ctext,
                        rule_score=rule_score,
                        llm_score=llm_score,
                        vision_score=vision_score,
                    )
                )
    else:
        # fallback: chunk by time
        t = 0.0
        idx = 1
        while t < cap:
            end = min(t + MAX_TAKE_SEC, cap)
            if (end - t) >= MIN_TAKE_SEC:
                clause_takes.append(
                    Take(
                        id=f"SEG{idx:04d}",
                        start=t,
                        end=end,
                        text=f"auto segment {idx}",
                        rule_score=0.6,
                        llm_score=0.0,
                        vision_score=_score_clause_vision(src, t, end),
                    )
                )
                idx += 1
            t = end

    # sort by time
    clause_takes.sort(key=lambda x: x.start)

    # pick story in order, using combined score
    story: List[Take] = []
    total_dur = 0.0
    kept_cta = False

    for t in clause_takes:
        # combined score
        combined = (
            0.6 * t.rule_score +
            0.25 * (t.llm_score if t.llm_score is not None else 0.0) +
            0.15 * t.vision_score
        )

        is_cta = _looks_cta(t.text)

        # prefer CTAs later
        if is_cta:
            kept_cta = True

        # main gate
        if combined < 0.45 and not is_cta:
            continue

        if total_dur + t.dur > cap:
            break

        story.append(t)
        total_dur += t.dur

    # ensure we have a CTA at the end
    if not kept_cta:
        # look from the tail of all clauses and pick the best CTA-like
        tail = [c for c in clause_takes[-8:]]  # last 8 clauses
        best_cta: Optional[Take] = None
        best_score = 0.0
        for c in tail:
            if _looks_cta(c.text):
                comb = (
                    0.6 * c.rule_score +
                    0.25 * (c.llm_score if c.llm_score is not None else 0.0) +
                    0.15 * c.vision_score
                )
                if comb > best_score:
                    best_score = comb
                    best_cta = c
        if best_cta and best_cta not in story and total_dur + best_cta.dur <= cap:
            story.append(best_cta)

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
            "face_q": t.vision_score,
            "scene_q": t.vision_score,
            "vtx_sim": 0.0,
            "chain_ids": [t.id],
            "text": _trim(t.text),
        }
        for t in story
    ]

    # slots
    slots: Dict[str, List[Dict[str, Any]]] = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    if story:
        # first → HOOK
        first = story[0]
        slots["HOOK"].append({
            "id": first.id,
            "start": first.start,
            "end": first.end,
            "text": _trim(first.text),
            "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [first.id]},
            "face_q": first.vision_score,
            "scene_q": first.vision_score,
            "vtx_sim": 0.0,
            "has_product": False,
            "ocr_hit": 0,
        })

    # middle → FEATURE
    if len(story) > 2:
        for mid in story[1:-1]:
            slots["FEATURE"].append({
                "id": mid.id,
                "start": mid.start,
                "end": mid.end,
                "text": _trim(mid.text),
                "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": [mid.id]},
                "face_q": mid.vision_score,
                "scene_q": mid.vision_score,
                "vtx_sim": 0.0,
                "has_product": False,
                "ocr_hit": 0,
            })

    # last → CTA
    if story:
        last = story[-1]
        slots["CTA"].append({
            "id": last.id,
            "start": last.start,
            "end": last.end,
            "text": _trim(last.text),
            "meta": {"slot": "CTA", "score": 2.0, "chain_ids": [last.id]},
            "face_q": last.vision_score,
            "scene_q": last.vision_score,
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
        "vision": cv2 is not None,
    }
