import os, io, json, time, uuid, tempfile, subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from worker.asr import transcribe_segments
import boto3

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------
def _env_float(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip().split()[:1]
    try:
        return float(v[0]) if v else d
    except:
        return d

def _env_int(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip().split()[:1]
    try:
        return int(v[0]) if v else d
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
CAPTIONS_MODE      = _env_str("CAPTIONS", "burn").lower()
BURN_CAPTIONS      = CAPTIONS_MODE in ("on","burn","burned","subtitle","1","true","yes")

# ------------------------------------------------------------
# MODEL TYPES
# ------------------------------------------------------------
@dataclass
class ClauseTake:
    id: str
    start: float
    end: float
    text: str
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    chain_ids: Optional[List[str]] = None  # we’ll just store [id] usually

    @property
    def dur(self) -> float:
        return float(self.end) - float(self.start)

# ------------------------------------------------------------
# SHELL HELPERS
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# TEXT / CLAUSE LOGIC
# ------------------------------------------------------------

# 1) things we ALWAYS drop if we see them (your bad ASR words)
HARD_BAD = {
    "kuchigai",
    "utas",
}

# 2) slang/phrases we ALLOW even if they look a bit messy
SLANG_OK = {
    "for the girls",
    "this is for the girls",
    "wet wet",
    "the girls only",
}

# simple helpers
def _norm_txt(s: str) -> str:
    return (s or "").strip().lower()

def _hits_hard_bad(txt: str) -> bool:
    low = _norm_txt(txt)
    return any(bad in low for bad in HARD_BAD)

def _hits_slang_ok(txt: str) -> bool:
    low = _norm_txt(txt)
    return any(ok in low for ok in SLANG_OK)

def _is_too_short(txt: str) -> bool:
    # throw away very short, non-content clauses
    words = [w for w in (txt or "").strip().split() if w]
    return len(words) <= 2  # "i found", "i think", etc.

def _is_really_gibberish(txt: str) -> bool:
    low = _norm_txt(txt)
    # anything with lots of "i found" repetitions but nothing else
    if low.count("i found") >= 2 and len(low) < 40:
        return True
    return False

def _split_segment_into_clauses(seg: Dict[str, Any]) -> List[ClauseTake]:
    """
    We have 1 start/end for the whole ASR segment.
    We'll fake intra-segment clause times by slicing the duration by number of clauses.
    This is what your job output is showing now.
    """
    text = (seg.get("text") or "").strip()
    if not text:
        return []

    seg_start = float(seg["start"])
    seg_end   = float(seg["end"])
    seg_dur   = max(0.01, seg_end - seg_start)

    # naive clause split
    raw_clauses = [c.strip() for c in text.replace("!", ".").replace("?", ".").split(".") if c.strip()]
    n = len(raw_clauses)
    if n == 0:
        return []

    per_clause = seg_dur / n
    out: List[ClauseTake] = []
    for idx, clause in enumerate(raw_clauses, start=1):
        c_start = seg_start + per_clause * (idx - 1)
        c_end   = seg_start + per_clause * idx
        out.append(ClauseTake(
            id=f"{seg.get('id','ASR'):s}_c{idx}",
            start=c_start,
            end=c_end,
            text=clause,
            chain_ids=[f"{seg.get('id','ASR'):s}_c{idx}"],
        ))
    return out

def _filter_clauses(clauses: List[ClauseTake]) -> List[ClauseTake]:
    """
    - drop clauses that contain HARD_BAD unless they also contain approved slang
    - drop tiny filler clauses
    - drop obvious gibberish
    """
    cleaned: List[ClauseTake] = []
    for c in clauses:
        txt = c.text or ""
        low = _norm_txt(txt)

        # 1) hard bad
        if _hits_hard_bad(txt) and not _hits_slang_ok(txt):
            # drop it
            continue

        # 2) very short + not slang
        if _is_too_short(txt) and not _hits_slang_ok(txt):
            continue

        # 3) obvious gibberish
        if _is_really_gibberish(txt):
            continue

        cleaned.append(c)

    return cleaned

def _pick_story_in_order(clauses: List[ClauseTake], max_len: float) -> List[ClauseTake]:
    story: List[ClauseTake] = []
    total = 0.0
    for c in clauses:
        if total + c.dur > max_len:
            break
        story.append(c)
        total += c.dur
    if not story and clauses:
        story = [clauses[0]]
    return story

# ------------------------------------------------------------
# SRT / EXPORT
# ------------------------------------------------------------
def _write_srt(story: List[ClauseTake]) -> str:
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

def _export_video(src: str, story: List[ClauseTake]) -> str:
    if not story:
        story = [ClauseTake(id="FALLBACK", start=0.0, end=min(5.0, MAX_DURATION_SEC), text="")]

    parts = []
    listfile = _tmpfile(suffix=".txt")
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

# ------------------------------------------------------------
# S3
# ------------------------------------------------------------
def _upload_to_s3(local_path: str) -> Dict[str, str]:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    stem = uuid.uuid4().hex
    key = f"{S3_PREFIX.rstrip('/')}/{stem}_{int(time.time())}.mp4"
    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh,
            S3_BUCKET,
            key,
            ExtraArgs={"ACL": S3_ACL, "ContentType": "video/mp4"},
        )
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
    }

# ------------------------------------------------------------
# PUBLIC ENTRY
# ------------------------------------------------------------
def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts: str = "1",
    max_duration: float,
    **kwargs,
) -> Dict[str, Any]:
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    # 1) download
    raw_local = _download_to_tmp(file_urls[0])

    # 2) real ASR -> segments
    segs = transcribe_segments(raw_local)

    # 3) segments -> clause-level takes
    all_clauses: List[ClauseTake] = []
    for seg in segs:
        seg_id = seg.get("id")
        # make sure id exists so we can show it
        if not seg_id:
            seg["id"] = f"ASR{len(all_clauses)+1:04d}"
        clauses = _split_segment_into_clauses(seg)
        all_clauses.extend(clauses)

    # 4) filter bad / tiny / duplicate gibberish
    filtered = _filter_clauses(all_clauses)

    # 5) pick story in order
    cap = float(max_duration or MAX_DURATION_SEC)
    story = _pick_story_in_order(filtered, cap)

    # 6) export video from clauses
    final_path = _export_video(raw_local, story)
    up = _upload_to_s3(final_path)

    # 7) build blocks for frontend
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
            "chain_ids": t.chain_ids or [],
            "text": t.text,
        })

    # simple slotting: first clause -> HOOK, last clause -> CTA, middle -> FEATURE
    slots_block = {
        "HOOK": [],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }
    if story:
        slots_block["HOOK"].append({
            "id": story[0].id,
            "start": story[0].start,
            "end": story[0].end,
            "text": story[0].text,
            "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": story[0].chain_ids or []},
            "face_q": story[0].face_q,
            "scene_q": story[0].scene_q,
            "vtx_sim": story[0].vtx_sim,
            "has_product": False,
            "ocr_hit": 0,
        })
    if len(story) >= 2:
        for mid in story[1:-1]:
            slots_block["FEATURE"].append({
                "id": mid.id,
                "start": mid.start,
                "end": mid.end,
                "text": mid.text,
                "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": mid.chain_ids or []},
                "face_q": mid.face_q,
                "scene_q": mid.scene_q,
                "vtx_sim": mid.vtx_sim,
                "has_product": False,
                "ocr_hit": 0,
            })
    if len(story) >= 2:
        last = story[-1]
        slots_block["CTA"].append({
            "id": last.id,
            "start": last.start,
            "end": last.end,
            "text": last.text,
            "meta": {"slot": "CTA", "score": 2.0, "chain_ids": last.chain_ids or []},
            "face_q": last.face_q,
            "scene_q": last.scene_q,
            "vtx_sim": last.vtx_sim,
            "has_product": False,
            "ocr_hit": 0,
        })

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
        "funnel_counts": {
            "HOOK": 1,
            "PROBLEM": 1,
            "FEATURE": 99,  # let’s not cap hard here
            "PROOF": 1,
            "CTA": 1,
        }
    }
