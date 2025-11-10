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


# ============================================================
# ENV HELPERS
# ============================================================

def _env_str(k: str, d: str) -> str:
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d


def _env_float(k: str, d: float) -> float:
    raw = (os.getenv(k) or "").split("#")[0].strip()
    if not raw:
        return d
    # allow stuff like 220" or 220s by stripping non-numeric tail
    num = ""
    for ch in raw:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            break
    try:
        return float(num)
    except Exception:
        return d


FFMPEG_BIN = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET = _env_str("S3_BUCKET", os.getenv("AWS_BUCKET", ""))
S3_PREFIX = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION = _env_str("AWS_REGION", "us-east-1")
S3_ACL = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MAX_TAKE_SEC = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC = _env_float("MIN_TAKE_SEC", 2.0)

# if set, we’ll try to LLM-score clauses
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


# ============================================================
# TEXT RULES
# ============================================================

# obvious "bad takes" that we always drop
BAD_PHRASES = [
    "wait",
    "hold on",
    "lemme start again",
    "let me start again",
    "start over",
    "no no",
    "redo",
    "sorry",
    "why can't i remember",   # from your IMG_03 test
]

# repeated CTA-ish junk we saw in your output that must go
REPETITIVE_CTA_PATTERNS = [
    "if you want to check them out",
    "so if you want to check them out",
    "if you wanna check them out",
    "so if you wanna check them out",
]

UGLY_BRANCHES = [
    "but if you don't like the checker print",
    "but if you don't like the checker",
    "but if you do",
    "but if you don't",
    "but if you",
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

FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "comes with", "it has", "it also has",
    "it's actually", "this isn't just", "design",
]


# ============================================================
# DATA CLASS
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
# SHELL / FILE HELPERS
# ============================================================

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, (out or "").strip(), (err or "").strip()


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


def _download_to_tmp(url: str) -> str:
    local_path = _tmpfile(".mp4")
    code, out, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local_path


# ============================================================
# S3 UPLOAD
# ============================================================

def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set in env")

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


# ============================================================
# ASR LOADING
# ============================================================

def _load_asr_segments(local_video_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Try to call your ASR in the safest order:
      1) worker.asr.transcribe_segments
      2) worker.asr.transcribe_local
      3) worker.asr.transcribe
    Return a list of {text, start, end} or None.
    """
    try:
        from worker import asr as asr_mod
    except Exception:
        return None

    # try transcribe_segments
    if hasattr(asr_mod, "transcribe_segments"):
        try:
            segs = asr_mod.transcribe_segments(local_video_path)
            if segs:
                return segs
        except Exception:
            pass

    # try transcribe_local
    if hasattr(asr_mod, "transcribe_local"):
        try:
            segs = asr_mod.transcribe_local(local_video_path)
            if segs:
                return segs
        except Exception:
            pass

    # try transcribe
    if hasattr(asr_mod, "transcribe"):
        try:
            segs = asr_mod.transcribe(local_video_path)
            if segs:
                return segs
        except Exception:
            pass

    return None


# ============================================================
# CLAUSE UTILITIES
# ============================================================

def _split_into_clauses(text: str) -> List[str]:
    """
    Turn 1 ASR segment into several smaller "clauses" so we can drop only the bad ones.
    """
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

    # drop tiny ones
    clauses = [c for c in clauses if len(c.split()) >= 3]
    return clauses


def _is_repetitive_cta(text: str) -> bool:
    low = text.lower().strip()
    for pat in REPETITIVE_CTA_PATTERNS:
        if pat in low:
            return True
    # super-short CTA that doesn't finish
    if low.startswith("if you want to check") and len(low.split()) < 10:
        return True
    if low.startswith("so if you want to check") and len(low.split()) < 10:
        return True
    return False


def _clause_is_featurey(c: str) -> bool:
    low = c.lower()
    return any(h in low for h in FEATURE_HINTS)


def _clause_is_ctaish(c: str) -> bool:
    low = c.lower()
    if any(p in low for p in CTA_FLUFF):
        return True
    if low.startswith("if you want to"):
        return True
    if low.startswith("if you wanna"):
        return True
    return False


def _clause_is_bad(c: str) -> bool:
    """
    Hard rule-based filter: if True -> drop unless LLM rescues
    (except for the hard repetitive CTA which we drop no matter what)
    """
    low = c.lower().strip()
    if not low:
        return True

    # 1) hard repetitive CTA → always drop
    if _is_repetitive_cta(low):
        return True

    # 2) explicit bad phrases
    for p in BAD_PHRASES:
        if p in low:
            return True

    # 3) weird branches
    for p in UGLY_BRANCHES:
        if p in low:
            return True

    # 4) too short
    if len(low.split()) < 3:
        return True

    # 5) obvious unfinished line (no punctuation, very short)
    if len(low.split()) < 6 and not low.endswith((".", "!", "?")):
        return True

    return False


def _assign_times_to_clauses(seg_start: float, seg_end: float, clauses: List[str]) -> List[Tuple[float, float, str]]:
    """
    Evenly map each clause to a sub-interval of the ASR segment.
    """
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


# ============================================================
# TEXT CLEANUP
# ============================================================

def _trim_repeated_ngrams(txt: str, n: int = 4) -> str:
    words = txt.split()
    if len(words) <= n * 2:
        return txt
    seen = {}
    for i in range(0, len(words) - n + 1):
        key = " ".join(w.lower() for w in words[i:i + n])
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


# ============================================================
# OPTIONAL LLM SCORING
# ============================================================

def _llm_score_clause(text: str) -> Optional[float]:
    """
    Return a float 0..1 meaning "how on-script / salesy / keepable this sounds".
    If no key or call fails -> None.
    """
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "You score short spoken clips from UGC product videos.\n"
            "Score 0.0 to 1.0.\n"
            "Give higher scores to: clear product mention, features, benefits, hook, CTA.\n"
            "Give low scores to: restarts, repeats, 'so if you wanna check...', partial sentences, losing track, apologies.\n"
            f"Clip: {text!r}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        raw = resp.choices[0].message.content.strip()
        # try to pull a number
        raw_num = "".join(ch for ch in raw if (ch.isdigit() or ch == ".")) or "0"
        return max(0.0, min(1.0, float(raw_num)))
    except Exception:
        return None


def _rule_score_clause(text: str) -> float:
    low = text.lower()
    score = 0.3  # base
    if _clause_is_ctaish(low):
        score += 0.2
    if _clause_is_featurey(low):
        score += 0.3
    return min(score, 1.0)


def _final_score_clause(text: str, hard_bad: bool) -> float:
    """
    Combine rule + optional LLM. If hard_bad=True and LLM is missing,
    we return 0 so it gets dropped.
    """
    llm_s = _llm_score_clause(text)
    rule_s = _rule_score_clause(text)
    if llm_s is None:
        # no LLM → respect hard_bad
        return 0.0 if hard_bad else rule_s
    # with LLM → take the better
    return max(rule_s, llm_s)


# ============================================================
# EXPORT / CONCAT
# ============================================================

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        # fallback: export first 5 seconds
        takes = [Take(id="FALLBACK", start=0.0, end=min(5.0, MAX_DURATION_SEC), text="")]

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
            part,
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
        final,
    ])
    return final


# ============================================================
# PUBLIC ENTRYPOINT
# ============================================================

def run_pipeline(
    *,
    local_video_path: Optional[str] = None,
    session_id: str,
    s3_prefix: Optional[str] = None,
    file_urls: Optional[List[str]] = None,
    max_duration: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    This signature matches what your tasks.py is calling:
        pipeline.run_pipeline(local_video_path=..., session_id=..., s3_prefix=...)
    But we also allow file_urls for future use.
    """
    # 1) get source video path
    if not local_video_path:
        urls = file_urls or []
        if not urls:
            return {"ok": False, "error": "no input video"}
        local_video_path = _download_to_tmp(urls[0])

    real_dur = _ffprobe_duration(local_video_path)
    cap = float(max_duration or MAX_DURATION_SEC)
    if real_dur > 0:
        cap = min(cap, real_dur)

    # 2) ASR
    segs = _load_asr_segments(local_video_path)
    used_asr = segs is not None

    if not segs:
        # fallback: just export the first cap seconds
        final_path = _export_concat(local_video_path, [
            Take(id="FALLBACK", start=0.0, end=cap, text="")
        ])
        up = _upload_to_s3(final_path, s3_prefix=s3_prefix)
        return {
            "ok": True,
            "session_id": session_id,
            "input_local": local_video_path,
            "duration_sec": _ffprobe_duration(final_path),
            "s3_key": up["s3_key"],
            "s3_url": up["s3_url"],
            "https_url": up["https_url"],
            "clips": [],
            "slots": {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []},
            "asr": False,
            "semantic": False,
            "vision": False,
        }

    # 3) turn ASR into clause-takes
    clause_takes: List[Take] = []
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
            continue

        timed = _assign_times_to_clauses(s, e, clauses)

        for c_idx, (c_s, c_e, c_text) in enumerate(timed, start=1):
            c_text = _clean_text(c_text)
            if not c_text:
                continue

            hard_bad = _clause_is_bad(c_text)
            score = _final_score_clause(c_text, hard_bad)

            # hard repetitive CTA → skip no matter what
            if _is_repetitive_cta(c_text):
                continue

            # if hard bad but LLM didn't rescue → skip
            if hard_bad and score < 0.6:
                continue

            dur = c_e - c_s
            if dur < 0.05:
                continue

            clause_takes.append(
                Take(
                    id=f"ASR{i:04d}_c{c_idx}",
                    start=c_s,
                    end=c_e,
                    text=c_text,
                )
            )

    clause_takes = sorted(clause_takes, key=lambda x: x.start)

    # 4) pick story within cap, but try to keep one CTA at the end
    story: List[Take] = []
    total_dur = 0.0
    last_cta: Optional[Take] = None

    for t in clause_takes:
        is_cta = _clause_is_ctaish(t.text)
        if is_cta:
            last_cta = t
        if total_dur + t.dur <= cap:
            story.append(t)
            total_dur += t.dur

    # if we have a CTA not already included → append it
    if last_cta and all(x.id != last_cta.id for x in story):
        story.append(last_cta)

    # 5) export
    final_path = _export_concat(local_video_path, story)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    # 6) build clips + slots for your API
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
        # first = HOOK
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

    if story:
        last = story[-1]
        if _clause_is_ctaish(last.text):
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
        "input_local": local_video_path,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips,
        "slots": slots,
        "asr": used_asr,
        "semantic": True,
        "vision": False,
    }
