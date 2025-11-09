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

# ------------------------------------------------------------
# 1) ENV HELPERS
# ------------------------------------------------------------

def _env_str(key: str, default: str = "") -> str:
    """
    Read an env var, strip comments and quotes.
    e.g. MAX_DURATION_SEC="220" # test  → we want 220
    """
    raw = os.getenv(key)
    if not raw:
        return default
    # drop comments
    raw = raw.split("#", 1)[0].strip()
    # drop quotes
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()
    return raw or default


def _env_float(key: str, default: float) -> float:
    raw = _env_str(key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


# ------------------------------------------------------------
# 2) ENV CONFIG
# ------------------------------------------------------------

FFMPEG_BIN  = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET   = _env_str("S3_BUCKET", "")
S3_PREFIX   = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION  = _env_str("AWS_REGION", "us-east-1")
S3_ACL      = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)  # total output max
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)       # single segment max
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 2.0)        # single segment min

# phrases / patterns
BAD_PHRASES = [
    "wait",
    "hold on",
    "lemme start again",
    "let me start again",
    "start over",
    "no no",
    "redo",
    "sorry",
    "why can't i remember",  # we saw this in img_03
]

CTA_PHRASES = [
    "click the link",
    "get yours today",
    "go ahead and click",
    "go ahead and grab",
    "i left it down below",
    "i left it for you down below",
    "grab one of these",
    "and grab one of these",
    "if you want to check them out",
    "if you wanna check them out",
]

UGLY_BRANCHES = [
    "but if you don't like the checker print",
    "but if you don't like the checker",
    "but if you don't like the checkered print",
    "but if you do",
    "but if you don't",
    "but if you",
]

FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "comes with", "it has", "it also has",
    "it's actually", "this isn't just", "design",
    "support moisture", "odor", "balance", "probiotic", "gummy",
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # may be None


# ------------------------------------------------------------
# 3) DATA CLASS
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
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
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


# ------------------------------------------------------------
# 5) S3 UPLOAD
# ------------------------------------------------------------

def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set in env")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"

    with open(local_path, "rb") as fh:
        extra = {"ACL": S3_ACL, "ContentType": "video/mp4"} if S3_ACL else {"ContentType": "video/mp4"}
        s3.upload_fileobj(fh, S3_BUCKET, key, ExtraArgs=extra)

    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}",
    }


# ------------------------------------------------------------
# 6) ASR LOADING (FROM YOUR worker.asr)
# ------------------------------------------------------------

def _load_asr_segments(local_video_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Your asr.py has transcribe_local(path) → list of {text, start, end}
    We wrap that.
    """
    try:
        from worker.asr import transcribe_local
    except Exception as e:
        print(f"[pipeline] cannot import worker.asr: {e}", flush=True)
        return None

    try:
        segs = transcribe_local(local_video_path)
    except Exception as e:
        print(f"[pipeline] ASR failed: {e}", flush=True)
        return None

    if not segs:
        return None
    return segs


# ------------------------------------------------------------
# 7) CLAUSE UTILITIES
# ------------------------------------------------------------

def _split_into_clauses(text: str) -> List[str]:
    if not text:
        return []
    text = " ".join(text.split())  # normalize spaces

    # first break on sentence-ish punctuation
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
        low = piece.lower()
        # further split on 'and' / 'but'
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

    # drop super short
    clauses = [c for c in clauses if len(c.split()) >= 3]
    return clauses


def _clause_is_bad(c: str) -> bool:
    low = c.lower()
    for p in BAD_PHRASES:
        if p in low:
            return True
    for p in UGLY_BRANCHES:
        if p in low:
            return True
    if len(low.split()) < 3:
        return True
    return False


def _clause_is_featurey(c: str) -> bool:
    low = c.lower()
    for h in FEATURE_HINTS:
        if h in low:
            return True
    return False


def _clause_is_ctaish(c: str) -> bool:
    low = c.lower()
    for p in CTA_PHRASES:
        if p in low:
            return True
    if low.startswith("if you want to") or low.startswith("if you wanna"):
        return True
    return False


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
# 8) TEXT CLEANUP
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
    for p in CTA_PHRASES:
        idx = low.find(p)
        if idx != -1:
            return txt[:idx].rstrip(" ,.;")
    return txt


def _clean_text(txt: str, *, is_cta: bool = False) -> str:
    txt = _trim_repeated_ngrams(txt, n=4)
    # IMPORTANT: don't strip the actual CTA if this clause is CTA-ish
    if not is_cta:
        txt = _trim_cta_fluff(txt)
    return txt.strip()


# ------------------------------------------------------------
# 9) LLM SCORING (OPTIONAL)
# ------------------------------------------------------------

def _llm_score_clause(text: str) -> Optional[float]:
    """
    Returns a float 0..1 if OPENAI_API_KEY is set, else None.
    We keep it simple: 1 = on-script sales / feature / benefit / CTA
    0 = obvious mistake / retry / confusion
    """
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = (
            "You are scoring lines from a talking-head UGC sales video. "
            "Score 1.0 if the line is smooth, product-related, a benefit, a CTA, or a clear hook. "
            "Score 0.0 if the line is a restart, a forgotten line, a self-correction, or off-topic.\n\n"
            f"LINE: {text!r}\n"
            "Return ONLY a number between 0 and 1."
        )

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=5,
        )
        raw = resp.output_text.strip()
        try:
            val = float(raw)
        except Exception:
            val = 0.5
        val = max(0.0, min(1.0, val))
        return val
    except Exception as e:
        print(f"[pipeline] LLM scoring failed: {e}", flush=True)
        return None


def _rule_score_clause(text: str) -> float:
    low = text.lower()
    # bad
    for p in BAD_PHRASES:
        if p in low:
            return 0.0
    # good-ish
    if _clause_is_featurey(text):
        return 0.9
    if _clause_is_ctaish(text):
        return 0.8
    # default
    return 0.6


# ------------------------------------------------------------
# 10) EXPORT (FFMPEG CONCAT)
# ------------------------------------------------------------

def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        # fallback: first 5s
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
# 11) PUBLIC ENTRYPOINT (WHAT tasks.py CALLS)
# ------------------------------------------------------------

def run_pipeline(
    *,
    local_video_path: str,
    session_id: str,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    print("[pipeline] filtered pipeline ACTIVE", flush=True)

    # 1) how long is the input?
    real_dur = _ffprobe_duration(local_video_path)
    cap = min(real_dur or MAX_DURATION_SEC, MAX_DURATION_SEC)

    # 2) get ASR
    segs = _load_asr_segments(local_video_path)

    used_asr = segs is not None
    clause_takes: List[Take] = []

    if used_asr:
        # turn ASR segments into clause-level takes
        for idx, seg in enumerate(segs, start=1):
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

            for c_idx, (cs, ce, ctext) in enumerate(timed, start=1):
                is_cta = _clause_is_ctaish(ctext)
                ctext_clean = _clean_text(ctext, is_cta=is_cta)
                if not ctext_clean:
                    continue
                dur = ce - cs
                if dur < 0.05:
                    continue

                # score
                base_score = _rule_score_clause(ctext_clean)
                llm_score = _llm_score_clause(ctext_clean)
                if llm_score is not None:
                    final_score = 0.6 * base_score + 0.4 * llm_score
                else:
                    final_score = base_score

                # drop obviously bad
                if final_score < 0.25:
                    continue

                clause_takes.append(
                    Take(
                        id=f"ASR{idx:04d}_c{c_idx}",
                        start=cs,
                        end=ce,
                        text=ctext_clean,
                    )
                )

        # sort by time
        clause_takes = sorted(clause_takes, key=lambda t: t.start)

        # now build the final story until we hit cap
        story: List[Take] = []
        total = 0.0
        for t in clause_takes:
            if total + t.dur > cap:
                break
            story.append(t)
            total += t.dur

        # if we didn't get a CTA, try to add one from the end
        has_cta = any(_clause_is_ctaish(t.text) for t in story)
        if not has_cta:
            for t in reversed(clause_takes):
                if _clause_is_ctaish(t.text):
                    story.append(t)
                    break

    else:
        # fallback: time-based chopping
        story = []
        t = 0.0
        idx = 1
        while t < cap:
            end = min(t + MAX_TAKE_SEC, cap)
            if (end - t) >= MIN_TAKE_SEC:
                story.append(
                    Take(
                        id=f"SEG{idx:04d}",
                        start=t,
                        end=end,
                        text=f"Auto segment {idx} ({t:.1f}s–{end:.1f}s)",
                    )
                )
                idx += 1
            t = end

    # 3) export shortened video
    final_path = _export_concat(local_video_path, story)

    # 4) upload
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    # 5) build slots like your old shape
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
        # first = hook
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

        # middle = features
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

        # last = CTA (prefer CTA-ish)
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
