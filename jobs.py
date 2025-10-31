import os
import io
import time
import uuid
import json
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ---------- ENV ----------
FFMPEG_BIN  = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "220").split()[0])

# micro-cut / silence handling (you already had these envs)
MICRO_CUT        = os.getenv("MICRO_CUT", "1").strip() in ("1","true","yes","on")
MICRO_SILENCE_DB = float(os.getenv("MICRO_SILENCE_DB", "-30").split()[0])
MICRO_SILENCE_MIN= float(os.getenv("MICRO_SILENCE_MIN","0.25").split()[0])

# filler cleanup
SEM_FILLER_LIST      = os.getenv("SEM_FILLER_LIST","um,uh,like,so,okay")
SEM_FILLER_MAX_RATE  = float(os.getenv("SEM_FILLER_MAX_RATE","0.08").split()[0])

# merge knobs
SEM_MERGE_SIM   = float(os.getenv("SEM_MERGE_SIM","0.70").split()[0])
VIZ_MERGE_SIM   = float(os.getenv("VIZ_MERGE_SIM","0.70").split()[0])
MERGE_MAX_CHAIN = int(os.getenv("MERGE_MAX_CHAIN","12").split()[0])

SCENE_Q_MIN     = float(os.getenv("SCENE_Q_MIN","0.4").split()[0])

# ---------- data model ----------
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

# ---------- tiny helpers ----------
def _run(cmd: List[str]) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _tmpfile(suffix=".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

def _download_to_tmp(url: str) -> str:
    """
    curl video from S3 presign / public URL to local tmp.
    """
    local_path = _tmpfile(".mp4")
    code, out, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed: {err}")
    return local_path

def _ffprobe_duration(path: str) -> float:
    code, out, err = _run([
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    try:
        return float(out.strip())
    except:
        return 0.0

# ---------- YOUR REAL LOGIC GOES HERE ----------
# You ALREADY have working code that:
#   - does ASR to get segments with {start,end,text,...}
#   - filters retries / garbage
#   - merges related takes
#   - stitches final ffmpeg concat
#
# We keep the same “shape” you had in your good run at 16:53.
# So I’ll stub helpers and you paste your real guts in each one.

def run_asr_segments(local_video_path: str) -> List[Dict[str,Any]]:
    """
    MUST return a list like:
    [
      {"start":0.00,"end":2.50,"text":"Is your ...?", "face_q":1.0,"scene_q":1.0,"vtx_sim":0.0},
      ...
    ]
    This is where you call Whisper / your ASR code.
    """
    raise NotImplementedError("paste your ASR code here")

def clean_and_merge_segments(segments: List[Dict[str,Any]]) -> List[Take]:
    """
    1. drop filler/restarts
    2. dedupe semantic repeats
    3. merge into longer takes
    Returns list[Take] in timeline order.
    """
    raise NotImplementedError("paste your semantic+merge code here")

def pick_storyline(takes: List[Take], max_len: float) -> List[Take]:
    """
    Keep best flow until max_len sec.
    """
    raise NotImplementedError("paste your pick_best_storyline logic here")

def stitch_video(src_video: str, story_takes: List[Take]) -> str:
    """
    ffmpeg trim each take, concat, return final_local path.
    """
    raise NotImplementedError("paste your export_story logic here")

def build_slots_and_clips(story_takes: List[Take]) -> Dict[str,Any]:
    """
    Turn final takes into:
      clips = [ {id,slot,start,end,text,...}, ... ]
      slots = { "HOOK":[...], "PROBLEM":[...], ... }
    You already had this shape in the good responses.
    We’ll keep slot = "HOOK" for now for each chunk.
    """
    clips = []
    slot_hook = []
    for t in story_takes:
        clip_info = {
            "id": t.id,
            "slot": "HOOK",
            "start": t.start,
            "end": t.end,
            "score": 2.5,
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "chain_ids": t.chain_ids or []
        }
        clips.append(clip_info)

        slot_hook.append({
            "id": t.id,
            "start": t.start,
            "end": t.end,
            "text": t.text,
            "meta": {
                "slot": "HOOK",
                "score": 2.5
            },
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "has_product": False,
            "ocr_hit": 0
        })

    slots = {
        "HOOK": slot_hook,
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    return {
        "clips": clips,
        "slots": slots,
    }

# ---------- MAIN ENTRY POINT ----------
def run_pipeline(
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    max_duration: float,
    audio: bool,
) -> Dict[str, Any]:
    """
    This is called by tasks.job_render().
    """

    # pick first file for now
    if not file_urls:
        return {"ok": False, "error": "no file_urls"}

    src_url = file_urls[0]

    # 1. download to /tmp
    local_raw = _download_to_tmp(src_url)

    # 2. ASR → segments
    segments = run_asr_segments(local_raw)

    # 3. clean + merge retries
    merged_takes = clean_and_merge_segments(segments)

    # 4. pick storyline up to max_duration seconds
    story_takes = pick_storyline(merged_takes, max_len=max_duration)

    if not story_takes:
        return {
            "ok": False,
            "error": "no usable takes after cleanup"
        }

    # 5. stitch with ffmpeg
    final_local = stitch_video(local_raw, story_takes)

    # 6. metadata
    duration_sec = _ffprobe_duration(final_local)

    # 7. clips / slots blocks like your good responses
    blocks = build_slots_and_clips(story_takes)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_raw,
        "final_local": final_local,
        "duration_sec": duration_sec,
        "clips": blocks["clips"],
        "slots": blocks["slots"],
    }
