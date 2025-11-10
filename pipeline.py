# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# try to import openai, but don't die if not present
try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

import boto3

# ================== ENV HELPERS ==================


def _env_str(k: str, d: str) -> str:
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d


def _env_float(k: str, d: float) -> float:
    raw = (os.getenv(k) or "").split("#")[0].strip()
    if not raw:
        return d
    try:
        # handle accidental quotes: "220"
        raw = raw.strip().strip('"').strip("'")
        return float(raw)
    except Exception:
        return d


FFMPEG_BIN = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET = _env_str("S3_BUCKET", "")
S3_PREFIX = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION = _env_str("AWS_REGION", "us-east-1")
S3_ACL = _env_str("S3_ACL", "public-read")

# note: these are safety rails, not hard ad durations
MAX_TAKE_SEC = _env_float("MAX_TAKE_SEC", 20.0)
MIN_TAKE_SEC = _env_float("MIN_TAKE_SEC", 2.0)

# ================== TEXT RULES (IMG_02 STYLE) ==================

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
    "if you wanna check them out",
]

UGLY_BRANCHES = [
    "but if you don't like the checker print",
    "but if you don't like the checker",
    "but if you don't like the checkered",
    "but if you do",
    "but if you don't",
    "but if you",
]

FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "comes with", "it has", "it also has",
    "it's actually", "this isn't just", "design",
    "they support", "moisture", "odor", "balance",  # for IMG_03 vibe
]

CTA_PHRASES_STRONG = [
    "i left it for you down below",
    "i left it down below",
    "grab one of these",
    "grab yours",
    "get yours",
    "link down below",
]

# ================== MODEL TYPES ==================


@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


# ================== SHELL / FILE HELPERS ==================


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
        # fallback: 5s from start
        takes = [Take(id="FALLBACK", start=0.0, end=min(5.0, _ffprobe_duration(src)), text="")]

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


# ================== ASR LOADING ==================


def _load_asr_segments(src: str) -> Optional[List[Dict[str, Any]]]:
    """
    We expect worker.asr.transcribe_segments(video_path) → list of {text,start,end}
    If it's missing, we return None and fall back to time-slicing.
    """
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
    # some demos return "temp placeholder" – ignore that
    if "temp placeholder" in (segs[0].get("text") or "").lower():
        return None
    return segs


def _segments_to_takes_asr(segs: List[Dict[str, Any]]) -> List[Take]:
    out: List[Take] = []
    for i, seg in enumerate(segs, start=1):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        if (e - s) > MAX_TAKE_SEC:
            e = s + MAX_TAKE_SEC
        if (e - s) >= MIN_TAKE_SEC:
            out.append(Take(id=f"ASR{i:04d}", start=s, end=e, text=txt))
    return out


# ================== CLAUSE / TEXT UTILS ==================


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

    # drop super short
    clauses = [c for c in clauses if len(c.split()) >= 3]
    return clauses


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


def _clause_is_featurey(c: str) -> bool:
    low = c.lower()
    for h in FEATURE_HINTS:
        if h in low:
            return True
    return False


def _clause_is_ctaish(c: str) -> bool:
    low = c.lower()
    for p in CTA_FLUFF:
        if p in low:
            return True
    for p in CTA_PHRASES_STRONG:
        if p in low:
            return True
    if low.startswith("if you want to"):
        return True
    if low.startswith("if you wanna"):
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


# ================== OPTIONAL LLM SCORER ==================


def _llm_score_clause(text: str) -> float:
    """
    0.0 → bad/off-script
    1.0 → usable product/salesy line
    Only runs if OPENAI_API_KEY is present.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key or not _HAS_OPENAI:
        return 0.0

    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "You are cleaning UGC sales videos. "
            "Score this line from 0 to 1.0 where 1.0 is on-script, clear, about the product, "
            "and 0 is a mistake, restart, or fluff. "
            "Return only the number.\n\n"
            f"LINE: \"{text}\""
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        val = resp.choices[0].message.content.strip()
        try:
            score = float(val)
            if score < 0:
                score = 0.0
            if score > 1:
                score = 1.0
            return score
        except Exception:
            return 0.0
    except Exception:
        return 0.0


# ================== FALLBACK TIME SPLIT ==================


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


# ================== PUBLIC ENTRY ==================


def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    s3_prefix: Optional[str] = None,
    max_duration: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main entry used by your RQ worker.
    """
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    src = _download_to_tmp(file_urls[0])
    real_dur = _ffprobe_duration(src)

    # decide cap: if caller sent max_duration use it, otherwise use whole video
    if max_duration:
        cap = min(float(max_duration), real_dur) if real_dur > 0 else float(max_duration)
    else:
        cap = real_dur

    segs = _load_asr_segments(src)
    used_asr = segs is not None

    if segs is not None:
        seg_takes = _segments_to_takes_asr(segs)

        clause_takes: List[Take] = []
        for seg_take in seg_takes:
            clauses = _split_into_clauses(seg_take.text)
            if not clauses:
                continue

            # map to sub-times
            timed = _assign_times_to_clauses(seg_take.start, seg_take.end, clauses)
            for idx, (cs, ce, ctext) in enumerate(timed, start=1):
                ctext_clean = _clean_text(ctext)
                if not ctext_clean:
                    continue

                # rule score
                rule_bad = _clause_is_bad(ctext)
                rule_feature = _clause_is_featurey(ctext)
                rule_cta = _clause_is_ctaish(ctext)

                # if rules think it's bad, try LLM rescue
                llm_score = 0.0
                if rule_bad and not rule_feature and not rule_cta:
                    llm_score = _llm_score_clause(ctext_clean)
                    if llm_score < 0.55:
                        # still bad
                        continue
                else:
                    # good by rules, optionally boost with llm
                    llm_score = _llm_score_clause(ctext_clean) if _HAS_OPENAI else 0.0

                dur = ce - cs
                if dur < 0.05:
                    continue

                clause_takes.append(
                    Take(
                        id=f"{seg_take.id}_c{idx}",
                        start=cs,
                        end=ce,
                        text=ctext_clean,
                    )
                )

        # sort by time
        clause_takes = sorted(clause_takes, key=lambda x: x.start)

        # pack up to cap, but remember CTA to append even if over
        story: List[Take] = []
        total_dur = 0.0
        last_cta_candidate: Optional[Take] = None

        for t in clause_takes:
            low = (t.text or "").lower()
            is_ctaish = _clause_is_ctaish(low)
            if is_ctaish:
                last_cta_candidate = t

            if total_dur + t.dur > cap:
                # stop packing, but we can append CTA later
                break

            story.append(t)
            total_dur += t.dur

        # append CTA if we saw one and it's not already in
        if last_cta_candidate and last_cta_candidate not in story:
            story.append(last_cta_candidate)

    else:
        # no ASR → dumb time slicing
        story = _time_based_takes(cap)
        used_asr = False

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
        # first → HOOK
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
