import os, io, json, time, uuid, tempfile, subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import boto3
import numpy as np

# ---------- ENV / CONFIG ----------
FFMPEG_BIN  = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET   = os.getenv("S3_BUCKET")
S3_PREFIX   = os.getenv("S3_PREFIX", "editdna/outputs")
AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")

# how long we allow final ad to be
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "220"))

# microcut settings (keeps speech tight)
MICRO_CUT = int(os.getenv("MICRO_CUT", "1"))
SILENCE_DB = float(os.getenv("MICRO_SILENCE_DB", "-30"))
SILENCE_MIN = float(os.getenv("MICRO_SILENCE_MIN", "0.25"))

# filler detection
FILLER_WORDS = set(
    (os.getenv("SEM_FILLER_LIST", "um,uh,like,so,okay")
        .lower()
        .replace(" ", "")
        .split(","))
)

FILLER_RATE_MAX = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))

# Sentence model / semantic merge knobs
SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM     = float(os.getenv("SEM_MERGE_SIM", "0.70"))
VIZ_MERGE_SIM     = float(os.getenv("VIZ_MERGE_SIM", "0.70"))
MERGE_MAX_CHAIN   = int(os.getenv("MERGE_MAX_CHAIN", "12"))

# hard scene change break guard in case angle fully changes
SCENE_Q_MIN = float(os.getenv("SCENE_Q_MIN", "0.4"))

# captions on/off (you said yes, burn captions)
BURN_CAPTIONS = True

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
    # meta helpers from previous pipeline stages:
    meta: Dict[str, Any] = None

    @property
    def dur(self) -> float:
        return float(self.end) - float(self.start)

# ---------- helpers ----------
def _run(cmd: List[str]) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

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

def _as_tmpfile(suffix=".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _download_to_tmp(url: str) -> str:
    """
    NOTE: assumes caller provided an https:// S3 presigned URL or public URL.
    We use curl because pod has curl.
    """
    local_path = _as_tmpfile(suffix=".mp4")
    code, out, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local_path

# ---------- speech-to-text / segmentation ----------
def run_asr_segments(video_path: str) -> List[Dict[str,Any]]:
    """
    We assume Whisper (or whatever ASR you already wired) is called somewhere in your old pipeline.
    In your logs we saw:
        [asr] segments: 21
        [seg] takes: 21
    So here we pretend we already have segments with {start,end,text}.
    IMPLEMENTATION NOTE:
    - If you already have a function that returns segments, paste it here instead of this stub.
    - If not, this dummy will throw so you notice.
    """
    raise RuntimeError("run_asr_segments() needs to call your existing ASR step from before. Plug it in from your old code.")

def is_filler_or_restart(text: str) -> bool:
    # high filler rate OR classic restart markers like "wait", "let me start again"
    t_low = text.lower()
    words = t_low.split()
    if any(w in t_low for w in ["wait", "hold on", "let me start again", "start over", "sorry no restart"]):
        return True
    if not words:
        return False
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    rate = filler_count / max(1, len(words))
    return rate > FILLER_RATE_MAX

def dedup_semantic(takes: List[Take]) -> List[Take]:
    """
    Keep first time we say basically the same thing, drop repeats.
    Uses SEM_DUP_THRESHOLD.
    To keep this file self-contained for you, we'll do a cheap text-similarity
    instead of requiring sentence-transformers here.
    You ALREADY load sentence-transformers/all-MiniLM-L6-v2 in the pod,
    so if you want stronger dedupe, you can replace this with that.
    For now: exact-ish match check, lowercase no punctuation.
    """
    seen = []
    out = []
    for t in takes:
        norm = "".join([c.lower() for c in t.text if c.isalnum() or c.isspace()]).strip()
        is_dup = any(norm == s for s in seen)
        if not is_dup:
            seen.append(norm)
            out.append(t)
    return out

def merge_chains(takes: List[Take]) -> List[Take]:
    """
    Build longer continuous phrases by merging adjacent takes
    if scene is still stable, and content flows.
    We'll only merge if:
        - next.start is basically current.end or close,
        - scene_q stays decent,
        - we haven't exceeded MERGE_MAX_CHAIN.
    """
    if not takes:
        return []

    takes = sorted(takes, key=lambda t: t.start)
    merged: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (
            j + 1 < len(takes)
            and len(chain) < MERGE_MAX_CHAIN
        ):
            a = chain[-1]
            b = takes[j+1]
            # require no huge gap and not a hard scene break
            gap = b.start - a.end
            if gap > 1.0:
                break
            if min(a.scene_q, b.scene_q) < SCENE_Q_MIN:
                break
            chain.append(b)
            j += 1

        # smoosh chain into one Take
        first = chain[0]
        last  = chain[-1]
        full_text = " ".join([c.text.strip() for c in chain])
        merged_take = Take(
            id = f"{first.id}_to_{last.id}",
            start = first.start,
            end   = last.end,
            text  = full_text,
            face_q = min(c.face_q for c in chain),
            scene_q = min(c.scene_q for c in chain),
            vtx_sim = max(c.vtx_sim for c in chain),
            meta = {
                "chain_ids": [c.id for c in chain],
                "len_chain": len(chain),
            }
        )
        merged.append(merged_take)
        i = j + 1

    return merged

def pick_best_storyline(merged_takes: List[Take], max_len: float) -> List[Take]:
    """
    Strategy:
    - Keep them in order,
    - Skip obvious garbage/no-text,
    - Stop when we hit max_len.
    Later, we can get fancy and try to ensure HOOK→BENEFIT→PROOF→CTA ordering,
    but right now you just asked:
      "give me the long, logical talking flow with bad retries removed".
    """
    out = []
    total = 0.0
    for t in merged_takes:
        # throw away garbage like just "okay" / "yeah" etc
        clean_txt = t.text.strip()
        # allow tiny "wait" clips to be skipped
        if len(clean_txt.split()) <= 2 and ("wait" in clean_txt.lower() or "okay" in clean_txt.lower()):
            continue
        dur = t.dur
        if total + dur > max_len:
            break
        out.append(t)
        total += dur
    return out

# ---------- caption burn ----------
def _make_srt_from_takes(takes: List[Take]) -> str:
    """
    Build a temporary .srt with timestamps from each take chunk.
    We'll show subtitles for each merged take as one block.
    """
    def fmt_ts(sec: float) -> str:
        # SRT format: HH:MM:SS,mmm
        ms = int(round((sec - int(sec)) * 1000))
        s = int(sec)
        hh = s // 3600
        mm = (s % 3600) // 60
        ss = s % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    lines = []
    for idx, t in enumerate(takes, start=1):
        lines.append(str(idx))
        lines.append(f"{fmt_ts(t.start)} --> {fmt_ts(t.end)}")
        # clean caption text a little
        cap_txt = t.text.strip().replace("\n", " ")
        lines.append(cap_txt if cap_txt else ".")
        lines.append("")
    return "\n".join(lines)

def burn_subtitles(video_in: str, takes: List[Take]) -> str:
    """
    1. write SRT
    2. ffmpeg -vf subtitles=...
    """
    srt_path = _as_tmpfile(suffix=".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(_make_srt_from_takes(takes))

    out_path = _as_tmpfile(suffix=".mp4")
    # draw subs
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_in,
        "-vf", f"subtitles={srt_path}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path
    ]
    code, out, err = _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg burn_subtitles failed: {err}")
    return out_path

# ---------- final stitch ----------
def export_story(video_path: str, story: List[Take]) -> str:
    """
    Cut each chosen take from the original raw,
    concat them with ffmpeg,
    return path of final stitched video (without captions yet).
    """
    part_paths = []
    concat_list_path = _as_tmpfile(suffix=".txt")

    for idx, t in enumerate(story, start=1):
        part_path = _as_tmpfile(suffix=f".part{idx:02d}.mp4")
        part_paths.append(part_path)

        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{t.start:.3f}",
            "-i", video_path,
            "-t", f"{t.dur:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-g", "48",
            "-c:a", "aac",
            "-b:a", "128k",
            part_path
        ]
        code, out, err = _run(cmd)
        if code != 0:
            raise RuntimeError(f"ffmpeg segment export failed: {err}")

    # build concat file
    with open(concat_list_path, "w") as f:
        for p in part_paths:
            f.write(f"file '{p}'\n")

    final_path = _as_tmpfile(suffix=".mp4")
    cmd2 = [
        FFMPEG_BIN, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        final_path
    ]
    code2, out2, err2 = _run(cmd2)
    if code2 != 0:
        raise RuntimeError(f"ffmpeg concat failed: {err2}")

    return final_path

# ---------- s3 upload ----------
def upload_to_s3(local_path: str) -> Dict[str, str]:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    # unique name like your previous style hash_timestamp.mp4
    stem = uuid.uuid4().hex
    key = f"{S3_PREFIX}/{stem}_{int(time.time())}.mp4"

    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh,
            S3_BUCKET,
            key,
            ExtraArgs={"ACL": os.getenv("S3_ACL", "public-read"),
                       "ContentType": "video/mp4"}
        )

    # build https-style url (your bucket is public-read)
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
    }

# ---------- MAIN ENTRY POINT ----------
def job_render(session_id: str,
               files: List[str],
               portrait: bool = True,
               max_duration: Optional[float] = None,
               audio: bool = True,
               output_prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    This is what tasks.job_render() in the worker calls.
    It MUST accept these args.
    """

    # pick first file for now
    src_url = files[0]

    # download raw video to pod temp
    local_raw = _download_to_tmp(src_url)

    # 1. ASR -> segments
    # segments = [{start,end,text,...}, ...]
    segments = run_asr_segments(local_raw)

    # 2. turn ASR segments into Take objects
    rough_takes: List[Take] = []
    for i, seg in enumerate(segments, start=1):
        txt = str(seg.get("text","")).strip()
        if not txt:
            continue
        if is_filler_or_restart(txt):
            continue
        rough_takes.append(Take(
            id=f"T{i:04d}",
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=txt,
            face_q=float(seg.get("face_q",1.0)),
            scene_q=float(seg.get("scene_q",1.0)),
            vtx_sim=float(seg.get("vtx_sim",0.0)),
            meta={"raw":seg}
        ))

    # 3. drop semantic duplicates / retries
    clean_takes = dedup_semantic(rough_takes)

    # 4. merge adjacent stable takes into longer thoughts
    merged = merge_chains(clean_takes)

    # 5. choose story in order until max length
    final_story = pick_best_storyline(
        merged,
        max_len = max_duration if max_duration is not None else MAX_DURATION_SEC
    )

    # 6. stitch story into one ad
    stitched_path = export_story(local_raw, final_story)

    # 7. burn captions (you said YES)
    if BURN_CAPTIONS:
        burned_path = burn_subtitles(stitched_path, final_story)
        final_video_path = burned_path
    else:
        final_video_path = stitched_path

    # 8. upload final ad
    upload_info = upload_to_s3(final_video_path)

    # 9. describe the "clips" + "slots" like your API returns
    clips_block = []
    for t in final_story:
        clips_block.append({
            "id": t.id,
            "slot": "STORY",
            "start": t.start,
            "end": t.end,
            "score": 2.5,
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "chain_ids": t.meta.get("chain_ids", [])
        })

    # slots = single bucket "STORY" for now
    slots_block = {
        "STORY": [
            {
                "id": t.id,
                "start": t.start,
                "end": t.end,
                "text": t.text,
                "meta": {
                    "slot": "STORY",
                    "score": 2.5,
                    "chain_ids": t.meta.get("chain_ids", [])
                },
                "face_q": t.face_q,
                "scene_q": t.scene_q,
                "vtx_sim": t.vtx_sim,
            }
            for t in final_story
        ]
    }

    total_duration = _ffprobe_duration(final_video_path)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_raw,
        "duration_sec": total_duration,
        "s3_key": upload_info["s3_key"],
        "s3_url": upload_info["s3_url"],
        "https_url": upload_info["https_url"],
        "clips": clips_block,
        "slots": slots_block,
        "semantic": True,
        "vision": False,
        "asr": True,
    }

# also expose run_pipeline so tasks.py can fallback if it calls that
def run_pipeline(*, local_path: Optional[str] = None,
                 payload: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    """
    Wrapper mainly for backward compatibility.
    We just forward to job_render.
    """
    if payload is None:
        payload = {}
    return job_render(
        session_id = payload.get("session_id","session"),
        files      = payload.get("files",[local_path] if local_path else []),
        portrait   = payload.get("portrait",True),
        max_duration = payload.get("max_duration", MAX_DURATION_SEC),
        audio      = payload.get("audio", True),
        output_prefix = payload.get("output_prefix")
    )
