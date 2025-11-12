# worker/pipeline.py
import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import boto3

from .asr import transcribe_segments
from .vision import estimate_face_quality

# ================== ENV HELPERS ==================

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

MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 2.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)  # per clip segment max
LLM_MODEL_DEFAULT = _env_str("OPENAI_MODEL", "gpt-4o-mini")

# ================== RULE LISTS ==================
BAD_PHRASES = [
    "wait", "hold on", "lemme start again", "let me start again",
    "start over", "no no", "redo", "sorry",
    "why can't i remember", "i forgot", "uh", "um",
]
CTA_FLUFF = [
    "click the link", "get yours today", "go ahead and click",
    "go ahead and grab", "i left it down below", "grab one of these",
    "if you want to check them out",
]
UGLY_BRANCHES = [
    "but if you don't like the checker print",
    "but if you don't like the checker",
    "but if you do", "but if you don't", "but if you",
]
FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "comes with", "it has", "it also has",
    "it's actually", "this isn't just", "design",
]

# ================== DATA TYPES ==================
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ================== SHELL UTILS ==================
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
    code, _, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local_path

def _ffprobe_duration(path: str) -> float:
    code, out, _ = _run([
        FFPROBE_BIN, "-v", "error",
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
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh, S3_BUCKET, key,
            ExtraArgs={"ACL": S3_ACL, "ContentType": "video/mp4"}
        )
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}",
    }

# ================== TEXT UTILS ==================
def _split_into_clauses(text: str) -> List[str]:
    if not text:
        return []
    text = " ".join(text.split())
    # sentence-ish split
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

def _clause_is_ctaish(c: str) -> bool:
    low = c.lower().strip()
    if low.startswith("if you want to"):
        return True
    for p in CTA_FLUFF:
        if p in low:
            return True
    return False

def _clause_is_featurey(c: str) -> bool:
    low = c.lower()
    for h in FEATURE_HINTS:
        if h in low:
            return True
    return False

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

def _clean_text(txt: str) -> str:
    # trim after repeated 4-gram or at first CTA fluff
    words = txt.split()
    # repeated 4-gram guard
    if len(words) > 8:
        seen = {}
        for i in range(0, len(words) - 3):
            key = " ".join(w.lower() for w in words[i:i+4])
            if key in seen:
                txt = " ".join(words[:i]).rstrip(" ,.;")
                break
            seen[key] = i
    low = txt.lower()
    cut = None
    for p in CTA_FLUFF:
        idx = low.find(p)
        if idx != -1:
            cut = idx
            break
    if cut is not None:
        txt = txt[:cut].rstrip(" ,.;")
    return txt.strip()

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

# ================== LLM SCORER ==================
def _llm_score_clause_openai(clause: str, slot_hint: str, model: str) -> Optional[float]:
    """
    Always-on LLM: returns 0..1 score or None on errors (network/rate limits).
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = (
            "You are a video ad editor assistant. "
            "Rate how on-script and sales-useful this clause is for a short UGC product video. "
            "Output ONLY a JSON object with keys: score (0..1), reason (short). "
            f"Clause: {clause!r}. Slot hint: {slot_hint}."
            "Prefer HOOK/FEATURE/CTA clarity; penalize restarts, self-corrections, filler, confusion."
        )
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        txt = r.choices[0].message.content.strip()
        import json as _json
        data = _json.loads(txt)
        s = float(data.get("score", 0.0))
        # clamp
        return max(0.0, min(1.0, s))
    except Exception:
        return None

def _rule_score(clause: str, slot_hint: str) -> float:
    low = clause.lower()
    if _clause_is_bad(clause):
        return 0.0
    base = 0.55
    if slot_hint == "HOOK" and not _clause_is_ctaish(clause):
        base += 0.10
    if _clause_is_featurey(clause):
        base += 0.20
    if _clause_is_ctaish(clause):
        base += 0.15
    return max(0.0, min(1.0, base))

def _score_clause(clause: str, slot_hint: str, model: str) -> float:
    # Always try LLM first
    s_llm = _llm_score_clause_openai(clause, slot_hint, model)
    if s_llm is not None:
        return s_llm
    # Fallback to rules if LLM hiccups
    return _rule_score(clause, slot_hint)

# ================== EXPORT ==================
def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        # emergency: first 5s
        takes = [Take(id="FALLBACK", start=0.0, end=5.0, text="")]
    # trim for MIN/MAX_TAKE_SEC
    trimmed: List[Take] = []
    for t in takes:
        s, e = float(t.start), float(t.end)
        if (e - s) < MIN_TAKE_SEC:
            continue
        if (e - s) > MAX_TAKE_SEC:
            e = s + MAX_TAKE_SEC
        trimmed.append(Take(id=t.id, start=s, end=e, text=t.text))
    takes = trimmed if trimmed else takes

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

# ================== MAIN PIPE ==================
def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    s3_prefix: Optional[str] = None,
    target_duration: Optional[float] = None,  # None = no target trimming
    max_duration: Optional[float] = None,     # None = no hard cap
    llm_always_on: bool = True,
    force_cta: bool = False,
    openai_model: str = LLM_MODEL_DEFAULT,
    **kwargs,
) -> Dict[str, Any]:

    if not file_urls:
        return {"ok": False, "error": "no input files"}

    src = _download_to_tmp(file_urls[0])
    real_dur = _ffprobe_duration(src)

    # ASR
    segs = transcribe_segments(src)
    if not segs:
        # No ASR? Return first 5 sec to avoid crashing.
        final_path = _export_concat(src, [Take("FALLBACK", 0.0, min(real_dur, 5.0), "")])
        up = _upload_to_s3(final_path, s3_prefix=s3_prefix)
        return {
            "ok": True, "session_id": session_id, "input_local": src,
            "duration_sec": _ffprobe_duration(final_path),
            "s3_key": up["s3_key"], "s3_url": up["s3_url"], "https_url": up["https_url"],
            "clips": [], "slots": {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []},
            "asr": False, "semantic": False, "vision": False
        }

    # Segments → clause-takes
    seg_takes: List[Take] = []
    for i, seg in enumerate(segs, start=1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        if (e - s) <= 0.05:
            continue

        clauses = _split_into_clauses(text)
        if not clauses:
            continue

        timed = _assign_times_to_clauses(s, e, clauses)
        for j, (cs, ce, ctext) in enumerate(timed, start=1):
            ctext = _clean_text(ctext)
            if not ctext:
                continue
            seg_takes.append(Take(id=f"ASR{i:04d}_c{j}", start=cs, end=ce, text=ctext))

    # Score & pick
    # HOOK = prefer the first 6s window’s best clause
    # FEATURE = middle clauses with product detail
    # CTA = final clause that sounds like CTA (or last coherent line)
    hooks: List[Take] = []
    feats: List[Take] = []
    ctas:  List[Take] = []

    for t in seg_takes:
        slot_hint = "FEATURE"
        # early in video → likely hooky
        if t.start <= 6.0:
            slot_hint = "HOOK"
        # last 1/4 of video → encourage CTA if text matches
        if _clause_is_ctaish(t.text) or (real_dur > 0 and t.start >= 0.75 * real_dur):
            slot_hint = "CTA"
        # vision (placeholder for now)
        vq = estimate_face_quality(src, t.start, t.end) or 1.0
        # LLM (always-on, with rule fallback internally)
        score = _score_clause(t.text, slot_hint, openai_model) * vq

        # gate out garbage
        if score < 0.45:
            continue

        if slot_hint == "HOOK":
            hooks.append(t)
        elif slot_hint == "CTA":
            ctas.append(t)
        else:
            feats.append(t)

    # Sort by time to preserve chronology (keeps natural flow)
    hooks = sorted(hooks, key=lambda x: x.start)
    feats = sorted(feats, key=lambda x: x.start)
    ctas  = sorted(ctas,  key=lambda x: x.start)

    # Build story list with optional target trim
    story: List[Take] = []
    def _push(ts: List[Take]):
        nonlocal story
        for t in ts:
            story.append(t)

    if hooks:
        _push([hooks[0]])  # best early hook (chronological)

    if feats:
        _push(feats)

    # ensure a CTA if present
    if ctas:
        _push([ctas[-1]])  # final CTA-ish line

    # If force_cta and none detected but we have any takes, re-use last line as CTA
    if force_cta and not ctas and story:
        last = story[-1]
        ctas = [last]

    # Optional target duration: keep it loose (no cap if None)
    if target_duration is not None and target_duration > 0:
        td = float(target_duration)
        acc = 0.0
        trimmed: List[Take] = []
        for t in story:
            if acc + t.dur <= td or not trimmed:
                trimmed.append(t)
                acc += t.dur
            else:
                break
        story = trimmed if trimmed else story

    # Optional hard cap (rarely used; you can pass None)
    if max_duration is not None and max_duration > 0:
        cap = float(max_duration)
        acc = 0.0
        trimmed: List[Take] = []
        for t in story:
            remaining = cap - acc
            if remaining <= 0.05:
                break
            if t.dur <= remaining:
                trimmed.append(t)
                acc += t.dur
            else:
                # clip this take to fit
                trimmed.append(Take(id=t.id, start=t.start, end=t.start+remaining, text=t.text))
                acc += remaining
                break
        story = trimmed if trimmed else story

    # Export
    final_path = _export_concat(src, story)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    # Build slots for UI
    def _trim(txt: str, n: int = 220) -> str:
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    clips = [{
        "id": t.id, "slot": "STORY", "start": t.start, "end": t.end,
        "score": 2.5, "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0,
        "chain_ids": [t.id], "text": _trim(t.text)
    } for t in story]

    slots: Dict[str, List[Dict[str, Any]]] = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    if story:
        # first → HOOK
        first = story[0]
        slots["HOOK"].append({
            "id": first.id, "start": first.start, "end": first.end, "text": _trim(first.text),
            "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [first.id]},
            "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0, "has_product": False, "ocr_hit": 0
        })
        # middle → FEATURE
        if len(story) > 2:
            for mid in story[1:-1]:
                slots["FEATURE"].append({
                    "id": mid.id, "start": mid.start, "end": mid.end, "text": _trim(mid.text),
                    "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": [mid.id]},
                    "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0, "has_product": False, "ocr_hit": 0
                })
        # last → CTA
        if len(story) >= 2:
            last = story[-1]
            slots["CTA"].append({
                "id": last.id, "start": last.start, "end": last.end, "text": _trim(last.text),
                "meta": {"slot": "CTA", "score": 2.0, "chain_ids": [last.id]},
                "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0, "has_product": False, "ocr_hit": 0
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
        "asr": True,
        "semantic": True,
        "vision": True,  # vision hook placeholder returns 1.0 for now
    }
