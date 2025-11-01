import os, io, json, time, uuid, tempfile, subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import boto3

# ---------- tiny util to safely read env numbers even if you added comments ----------

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    cleaned = raw.split("#")[0].strip().split()[0]
    try:
        return float(cleaned)
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    cleaned = raw.split("#")[0].strip().split()[0]
    try:
        return int(cleaned)
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "")
    if not raw:
        return default
    cleaned = raw.split("#")[0].strip()
    if cleaned == "":
        return default
    return cleaned

# ---------- ENV / CONFIG ----------

FFMPEG_BIN   = _env_str("FFMPEG_BIN",   "/usr/bin/ffmpeg")
FFPROBE_BIN  = _env_str("FFPROBE_BIN",  "/usr/bin/ffprobe")

S3_BUCKET    = _env_str("S3_BUCKET",    "")
S3_PREFIX    = _env_str("S3_PREFIX",    "editdna/outputs")
AWS_REGION   = _env_str("AWS_REGION",   "us-east-1")

MAX_DURATION_SEC      = _env_float("MAX_DURATION_SEC", 220.0)

MICRO_CUT              = _env_int("MICRO_CUT", 1)
SILENCE_DB             = _env_float("MICRO_SILENCE_DB", 30.0)
SILENCE_MIN            = _env_float("MICRO_SILENCE_MIN", 0.25)

FILLER_WORDS           = [
    w.strip().lower() for w in
    _env_str("SEM_FILLER_LIST", "um,uh,like,so,okay").split(",")
]
FILLER_RATE_MAX        = _env_float("SEM_FILLER_MAX_RATE", 0.80)

SEM_DUP_THRESHOLD      = _env_float("SEM_DUP_THRESHOLD", 0.88)
SEM_MERGE_SIM          = _env_float("SEM_MERGE_SIM", 0.70)
VIZ_MERGE_SIM          = _env_float("VIZ_MERGE_SIM", 0.70)
MERGE_MAX_CHAIN        = _env_int("MERGE_MAX_CHAIN", 12)

SCENE_Q_MIN            = _env_float("SCENE_Q_MIN", 0.4)

CAPTIONS_MODE          = _env_str("CAPTIONS", "burn").lower()
BURN_CAPTIONS          = CAPTIONS_MODE in ("on","burn","burned","subtitle","1","true","yes")

S3_ACL                 = _env_str("S3_ACL", "public-read")

# ---------- DATA MODEL ----------

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


# ---------- LOW LEVEL HELPERS ----------

def _run(cmd: List[str]) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def _ffprobe_duration(path: str) -> float:
    code, out, err = _run([
        FFPROBE_BIN,
        "-v","error",
        "-show_entries","format=duration",
        "-of","default=nokey=1:noprint_wrappers=1",
        path,
    ])
    if code != 0:
        return 0.0
    try:
        return float(out.strip())
    except:
        return 0.0

def _tmpfile(suffix=".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _download_to_tmp(url: str) -> str:
    """
    Download the S3/public URL into a local temp .mp4
    """
    local_path = _tmpfile(suffix=".mp4")
    code, out, err = _run(["curl","-L","-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed {{code}}: {err}")
    return local_path


# ---------- STEP 1: ASR / SEGMENTS (currently STUBBED) ----------

def _fake_asr_segments(local_raw: str) -> List[Dict[str,Any]]:
    """
    TEMP SAFE PLACEHOLDER.
    We don't call Whisper yet. We just return 1 fake segment.
    We'll swap real Whisper segmentation later.
    """
    return [{
        "start": 0.0,
        "end":   min(MAX_DURATION_SEC, 10.0),
        "text":  "TEMP PLACEHOLDER: real ASR not wired yet.",
        "face_q": 1.0,
        "scene_q": 1.0,
        "vtx_sim": 0.0,
    }]


# ---------- STEP 2: SEGMENTS -> TAKES, DROP TRASH ----------

def _is_throwaway(txt: str) -> bool:
    # kill obvious junk like "wait wait let me start again"
    # or giant filler rate.
    low = txt.lower().strip()
    words = low.split()
    if any(p in low for p in ["wait","hold on","let me start again","start over"]):
        return True
    if not words:
        return True
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    rate = filler_count / max(1, len(words))
    return rate > FILLER_RATE_MAX

def _segments_to_takes(segments: List[Dict[str,Any]]) -> List[Take]:
    out: List[Take] = []
    for i, seg in enumerate(segments, start=1):
        txt = str(seg.get("text","")).strip()
        if not txt:
            continue
        if _is_throwaway(txt):
            continue
        out.append(
            Take(
                id=f"{i:04d}",
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=txt,
                face_q=float(seg.get("face_q",1.0)),
                scene_q=float(seg.get("scene_q",1.0)),
                vtx_sim=float(seg.get("vtx_sim",0.0)),
                chain_ids=[f"{i:04d}"],
            )
        )
    return out


# ---------- STEP 3: MERGE NEARBY TAKES ----------

def _merge_adjacent(takes: List[Take]) -> List[Take]:
    """
    Greedy merge of adjacent takes if gap is small and scene not trash,
    building "chains" up to MERGE_MAX_CHAIN long.
    """
    if not takes:
        return []

    takes = sorted(takes, key=lambda t: t.start)

    merged: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i + 1
        while j < len(takes):
            a = chain[-1]
            b = takes[j]
            gap = b.start - a.end
            if gap > 1.0:
                break
            # scene quality gate
            if min(a.scene_q, b.scene_q) < SCENE_Q_MIN:
                break
            chain.append(b)
            j += 1
            if len(chain) >= MERGE_MAX_CHAIN:
                break

        first = chain[0]
        last  = chain[-1]
        full_text = " ".join([c.text.strip() for c in chain])

        merged_take = Take(
            id=f"{first.id}_to_{last.id}",
            start=first.start,
            end=last.end,
            text=full_text,
            face_q=min(c.face_q for c in chain),
            scene_q=min(c.scene_q for c in chain),
            vtx_sim=max(c.vtx_sim for c in chain),
            chain_ids=[c.id for c in chain],
        )

        merged.append(merged_take)
        i = j

    return merged


# ---------- STEP 4: PICK STORY UNDER max_duration ----------

def _pick_story(merged: List[Take], max_len: float) -> List[Take]:
    """
    Walk in order, keep adding until max_len.
    Fallback: if too tiny, fall back to the single longest take.
    """
    story: List[Take] = []
    total = 0.0
    for t in merged:
        # skip useless "uh ok wait"
        words = t.text.strip().split()
        if len(words) <= 2:
            continue
        dur = t.dur
        if total + dur > max_len:
            break
        story.append(t)
        total += dur

    if total < 10.0 and merged:
        # fallback = longest single take
        longest = max(merged, key=lambda x: x.dur)
        story = [longest]

    return story


# ---------- STEP 5: CUT + CONCAT THE STORY ----------

def _export_story_video(src_path: str, story: List[Take]) -> str:
    """
    For each chosen Take in "story", make a clip with ffmpeg -ss/-to.
    Concat them. Return final_path (temp .mp4).
    """

    if not story:
        # Hard fallback: first 5 seconds of raw
        story = [Take(
            id="FALLBACK",
            start=0.0,
            end=min(5.0, MAX_DURATION_SEC),
            text="(fallback clip)",
            chain_ids=["FALLBACK"],
        )]

    concat_list_path = _tmpfile(suffix=".txt")
    part_paths: List[str] = []

    for idx, t in enumerate(story, start=1):
        part_path = _tmpfile(suffix=f"_{idx:02d}.mp4")
        part_paths.append(part_path)

        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", f"{t.start:.3f}",
            "-to", f"{t.end:.3f}",
            "-i", src_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            part_path,
        ]
        _run(cmd)

    with open(concat_list_path, "w") as fh:
        for p in part_paths:
            fh.write(f"file '{p}'\n")

    final_path = _tmpfile(suffix=".mp4")
    cmd2 = [
        FFMPEG_BIN, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        final_path,
    ]
    _run(cmd2)

    return final_path


# ---------- STEP 6: OPTIONAL CAPTIONS (burn-in) ----------

def _srt_timestamp(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = int(sec % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def _write_srt_for_story(story: List[Take]) -> str:
    srt_path = _tmpfile(suffix=".srt")
    with open(srt_path, "w", encoding="utf-8") as fh:
        for idx, t in enumerate(story, start=1):
            fh.write(f"{idx}\n")
            fh.write(f"{_srt_timestamp(t.start)} --> {_srt_timestamp(t.end)}\n")
            line_txt = t.text.strip().replace("\n"," ")
            fh.write(f"{line_txt}\n\n")
    return srt_path

def _burn_subtitles(in_path: str, story: List[Take]) -> str:
    """
    If captions are enabled, burn .srt as hard subtitles using ffmpeg filter "subtitles=".
    """
    if not BURN_CAPTIONS:
        return in_path

    srt_path = _write_srt_for_story(story)
    out_path = _tmpfile(suffix=".mp4")

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", in_path,
        "-vf", f"subtitles='{srt_path}'",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]
    _run(cmd)

    return out_path


# ---------- STEP 7: UPLOAD TO S3 ----------

def _upload_to_s3(local_path: str,
                  s3_prefix: str) -> Dict[str,str]:
    """
    Upload final clip to S3.
    Returns dict {"s3_key": "...", "s3_url": "...", "https_url": "..."}
    """

    if not S3_BUCKET:
        # dev mode fallback: don't upload, just return temp path
        return {
            "s3_key": "",
            "s3_url": "",
            "https_url": local_path,
        }

    s3 = boto3.client("s3", region_name=AWS_REGION)
    stem = uuid.uuid4().hex
    key  = f"{s3_prefix.strip('/')}/{stem}_{int(time.time())}.mp4"

    with open(local_path, "rb") as fh:
        s3.upload_fileobj(
            fh,
            S3_BUCKET,
            key,
            ExtraArgs={
                "ACL": S3_ACL,
                "ContentType": "video/mp4",
            },
        )

    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
    }


# ---------- PUBLIC ENTRY CALLED BY tasks.job_render ----------

def run_pipeline(
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_count: str,
    max_duration: float,
    s3_prefix: str,
    **kwargs,
) -> Dict[str,Any]:
    """
    MAIN ORCHESTRATION.
    """

    if not file_urls:
        return {"ok": False, "error": "no input files"}

    # 1. download first file
    local_raw = _download_to_tmp(file_urls[0])

    # 2. ASR placeholder -> segments
    segments = _fake_asr_segments(local_raw)

    # 3. segments -> takes (drop trash)
    takes = _segments_to_takes(segments)

    # 4. merge adjacent
    merged = _merge_adjacent(takes)

    # 5. pick story under max_duration
    story = _pick_story(merged, max_duration)

    # 6. stitch story
    stitched_path = _export_story_video(local_raw, story)

    # 7. burn subtitles if needed
    final_path = _burn_subtitles(stitched_path, story)

    # 8. upload final clip
    up = _upload_to_s3(final_path, s3_prefix)

    # 9. build clips_block + slots_block to send back to web
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

    slots_block = {
        "HOOK": [
            {
                "id": t.id,
                "start": t.start,
                "end": t.end,
                "text": t.text,
                "meta": {
                    "slot": "HOOK",
                    "score": 2.5,
                    "chain_ids": t.chain_ids or [],
                },
                "face_q": t.face_q,
                "scene_q": t.scene_q,
                "vtx_sim": t.vtx_sim,
                "has_product": False,
                "ocr_hit": 0,
            }
            for t in story
        ],
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }

    dur_final = _ffprobe_duration(final_path)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_raw,
        "duration_sec": dur_final,
        "s3_key": up.get("s3_key",""),
        "s3_url": up.get("s3_url",""),
        "https_url": up.get("https_url",""),  # <-- web clicks this to preview video
        "clips": clips_block,
        "slots": slots_block,
        "semantic": True,
        "vision": False,
        "asr": True,
    }
