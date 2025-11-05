import os, io, json, time, uuid, tempfile, subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from worker.asr import transcribe_segments
import boto3

# -------------------------------------------------
# ENV HELPERS
# -------------------------------------------------
def _env_float(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip().split()[:1]
    try:
        return float(v[0]) if v else d
    except:
        return d

def _env_str(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d

FFMPEG_BIN  = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET   = _env_str("S3_BUCKET", "")
S3_PREFIX   = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION  = _env_str("AWS_REGION", "us-east-1")
S3_ACL      = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC   = _env_float("MAX_DURATION_SEC", 120.0)
MIN_TAKE_SEC       = _env_float("MIN_TAKE_SEC", 2.0)
MAX_TAKE_SEC       = _env_float("MAX_TAKE_SEC", 20.0)

CAPTIONS_MODE      = _env_str("CAPTIONS", "burn").lower()
BURN_CAPTIONS      = CAPTIONS_MODE in ("on","burn","burned","subtitle","1","true","yes")

# -------------------------------------------------
# DATA MODEL
# -------------------------------------------------
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    chain_ids: Optional[List[str]] = None

    @property
    def dur(self) -> float:
        return float(self.end) - float(self.start)

# -------------------------------------------------
# SHELL / IO
# -------------------------------------------------
def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _tmpfile(suffix=".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _ffprobe_duration(path: str) -> float:
    code, out, err = _run([
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ])
    try:
        return float(out.strip()) if code == 0 else 0.0
    except:
        return 0.0

def _download_to_tmp(url: str) -> str:
    local_path = _tmpfile(suffix=".mp4")
    code, out, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local_path

# -------------------------------------------------
# TEXT CLEAN / FILTER CONFIG
# -------------------------------------------------
# words we saw from bad ASR that are NOT really English / UGC
HARD_BAD = {
    "kuchigai",    # nonsense from probiotic video
    "utas",        # nonsense
}

# phrases that look odd but are actually UGC/tiktok language
SLANG_OK = {
    "for the girls",
    "for the girls only",
    "wet wet",
    "worry no more",
    "don't be shy",
    "go ahead and click",
}

RETRY_MARKERS = (
    "wait", "hold on", "let me start again", "let me start over",
    "start over", "no no", "redo", "take two", "i mean actually"
)

FILLERS = {"uh", "um", "like", "so", "okay", "sorry"}

def _norm(s: str) -> str:
    return "".join(c.lower() for c in s if (c.isalnum() or c.isspace())).strip()

def _looks_like_retry_or_filler(text: str) -> bool:
    low = text.lower()
    if any(m in low for m in RETRY_MARKERS):
        return True
    words = [w.strip(",.?!") for w in low.split()]
    if not words:
        return True
    filler_rate = sum(1 for w in words if w in FILLERS) / max(1, len(words))
    return filler_rate > 0.35  # now a bit less aggressive

def _hits_hard_bad(text: str) -> bool:
    low = text.lower()
    return any(bad in low for bad in HARD_BAD)

def _hits_slang_ok(text: str) -> bool:
    low = text.lower()
    return any(sl in low for sl in SLANG_OK)

# -------------------------------------------------
# CLAUSE SPLITTER
# -------------------------------------------------
def _split_segment_into_clauses(seg: Dict[str, Any]) -> List[Take]:
    """
    We get 1 ASR segment: {start, end, text}
    We split its text on punctuation / 'but' / 'and then' to get sub-clauses
    and distribute the time across them.
    """
    text = (seg.get("text") or "").strip()
    if not text:
        return []

    start = float(seg["start"])
    end   = float(seg["end"])
    dur   = max(0.01, end - start)

    # naive clause split
    import re
    raw_clauses = re.split(r"(?:,|\.|;| and | but )", text)
    clauses = [c.strip() for c in raw_clauses if c.strip()]
    if not clauses:
        return [Take(id="ASRSEG", start=start, end=end, text=text)]

    per_clause = dur / len(clauses)
    takes: List[Take] = []
    cur_start = start
    for idx, c in enumerate(clauses, start=1):
        c_dur = per_clause
        c_end = cur_start + c_dur
        takes.append(Take(
            id=f"{seg.get('id','ASR')}_c{idx}",
            start=cur_start,
            end=c_end,
            text=c,
        ))
        cur_start = c_end
    # clamp last one to real end
    if takes:
        takes[-1].end = end
    return takes

# -------------------------------------------------
# ASR → CLAUSE TAKES → FILTER
# -------------------------------------------------
def _segments_to_clause_takes(segments: List[Dict[str, Any]]) -> List[Take]:
    all_takes: List[Take] = []
    for seg in segments:
        # preserve original id if present, else make one
        if "id" not in seg:
            seg["id"] = f"ASR{len(all_takes)+1:04d}"
        all_takes.extend(_split_segment_into_clauses(seg))
    return all_takes

def _filter_takes(takes: List[Take]) -> List[Take]:
    """
    1) drop empty
    2) drop obvious retries
    3) drop HARD_BAD only if it's not in SLANG_OK
    4) dedupe text
    """
    out: List[Take] = []
    seen = set()
    for t in takes:
        txt = (t.text or "").strip()
        if not txt:
            continue

        # 1) avoid micro-clips < 0.4s
        if (t.end - t.start) < 0.4:
            continue

        # 2) retry/filler
        if _looks_like_retry_or_filler(txt):
            continue

        # 3) hard bad vs slang
        if _hits_hard_bad(txt) and not _hits_slang_ok(txt):
            # real ASR garbage → drop
            continue

        # 4) dedupe on normalized text
        n = _norm(txt)
        if n in seen:
            continue
        seen.add(n)

        out.append(t)
    return out

# -------------------------------------------------
# STORY PICKER (simple)
# -------------------------------------------------
def _pick_story_in_order(takes: List[Take], max_len: float) -> List[Take]:
    story: List[Take] = []
    total = 0.0
    for t in takes:
        if total + t.dur > max_len:
            break
        story.append(t)
        total += t.dur
    if not story and takes:
        story = [max(takes, key=lambda x: x.dur)]
    return story

# -------------------------------------------------
# EXPORTS
# -------------------------------------------------
def _write_srt(story: List[Take]) -> str:
    def ts(sec: float) -> str:
        ms = int(round((sec - int(sec)) * 1000))
        s = int(sec)
        hh, mm, ss = s // 3600, (s % 3600) // 60, s % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    path = _tmpfile(suffix=".srt")
    with open(path, "w", encoding="utf-8") as fh:
        for i, t in enumerate(story, start=1):
            safe_text = (t.text or ".").replace("\n", " ")
            fh.write(f"{i}\n{ts(t.start)} --> {ts(t.end)}\n{safe_text}\n\n")
    return path

def _export_video(src: str, story: List[Take]) -> str:
    if not story:
        story = [Take(id="FALLBACK", start=0.0, end=min(5.0, MAX_DURATION_SEC), text="")]
    parts, listfile = [], _tmpfile(suffix=".txt")
    for idx, t in enumerate(story, start=1):
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
    if BURN_CAPTIONS:
        srt = _write_srt(story)
        burned = _tmpfile(suffix=".mp4")
        _run([
            FFMPEG_BIN, "-y",
            "-i", final,
            "-vf", f"subtitles={srt}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            burned
        ])
        return burned
    return final

# -------------------------------------------------
# S3
# -------------------------------------------------
def _upload_to_s3(local_path: str) -> Dict[str, str]:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    stem = uuid.uuid4().hex
    key = f"{S3_PREFIX.rstrip('/')}/{stem}_{int(time.time())}.mp4"
    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh, S3_BUCKET, key,
            ExtraArgs={"ACL": S3_ACL, "ContentType": "video/mp4"}
        )
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
    }

# -------------------------------------------------
# PUBLIC ENTRY
# -------------------------------------------------
def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts: str,
    max_duration: float,
    **kwargs,
) -> Dict[str, Any]:
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    raw_local = _download_to_tmp(file_urls[0])

    # 1) real ASR segments
    segs = transcribe_segments(raw_local)

    # 2) ASR → clause-level takes
    clause_takes = _segments_to_clause_takes(segs)

    # 3) filter (retries, hard-bad, dedupe)
    clause_takes = _filter_takes(clause_takes)

    # 4) pick story up to requested max
    cap = float(max_duration or MAX_DURATION_SEC)
    story = _pick_story_in_order(clause_takes, cap)

    # 5) export video + s3
    final_path = _export_video(raw_local, story)
    up = _upload_to_s3(final_path)

    # 6) build clips / slots for the frontend
    clips_block = []
    for t in story:
        clips_block.append({
            "id": t.id,
            "slot": "STORY",
            "start": t.start,
            "end": t.end,
            "score": 2.5,
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "chain_ids": t.chain_ids or [t.id],
            "text": t.text,
        })

    # put everything in FEATURE except first (HOOK) and last (CTA) if they look like CTA
    slots_block = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    for i, t in enumerate(story):
        slot_doc = {
            "id": t.id,
            "start": t.start,
            "end": t.end,
            "text": t.text,
            "meta": {
                "slot": "FEATURE",
                "score": 2.0,
                "chain_ids": t.chain_ids or [t.id],
            },
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "has_product": False,
            "ocr_hit": 0,
        }
        if i == 0:
            slot_doc["meta"]["slot"] = "HOOK"
            slots_block["HOOK"].append(slot_doc)
        elif i == len(story) - 1 and "grab" in (t.text or "").lower():
            slot_doc["meta"]["slot"] = "CTA"
            slots_block["CTA"].append(slot_doc)
        else:
            slots_block["FEATURE"].append(slot_doc)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": raw_local,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips_block,
        "slots": slots_block,
        "asr": True,
        "semantic": True,
        "vision": False,
    }
