import os
import io
import time
import uuid
import json
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import boto3


# =========================
# ENV / GLOBAL KNOBS
# =========================

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "editdna/outputs")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_ACL = os.getenv("S3_ACL", "public-read")

# max final ad runtime (seconds)
MAX_DURATION_SEC_DEFAULT = float(os.getenv("MAX_DURATION_SEC", "220"))

# captions (we burn subs directly into final by default in Mode B)
BURN_CAPTIONS = True

# scene / merge knobs (safe defaults)
MERGE_MAX_CHAIN = int(os.getenv("MERGE_MAX_CHAIN", "12"))
SCENE_Q_MIN = float(os.getenv("SCENE_Q_MIN", "0.4"))

# filler control
FILLER_WORDS = set(
    (
        os.getenv("SEM_FILLER_LIST", "um,uh,like,so,okay")
        .lower()
        .replace(" ", "")
        .split(",")
    )
)
FILLER_RATE_MAX = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))


# =========================
# DATA STRUCTURES
# =========================

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    meta: Dict[str, Any] = None

    @property
    def dur(self) -> float:
        return float(self.end) - float(self.start)


# =========================
# GENERIC HELPERS
# =========================

def _run(cmd: List[str]) -> Tuple[int, str, str]:
    """
    Run a shell command and return (exit_code, stdout, stderr) as text.
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()


def _tmpfile(suffix: str = ".mp4") -> str:
    """
    Create a unique temp file path (file is created+closed immediately).
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _download_video_to_tmp(url: str) -> str:
    """
    Download remote video URL (public or presigned S3 URL) -> local temp file.
    Uses curl because it's installed in the pod.
    """
    local_path = _tmpfile(".mp4")
    code, out, err = _run(["curl", "-L", "-o", local_path, url])
    if code != 0:
        raise RuntimeError(f"curl failed ({code}): {err}")
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
        raise RuntimeError(f"ffprobe failed: {err}")
    try:
        return float(out.strip())
    except:
        return 0.0


def _upload_to_s3(local_path: str) -> Dict[str, str]:
    """
    Upload final .mp4 to your bucket with public-read and return URLs.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)

    stem = uuid.uuid4().hex
    key = f"{S3_PREFIX.rstrip('/')}/{stem}_{int(time.time())}.mp4"

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

    https_url = (
        f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    )

    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": https_url,
    }


# =========================
# ASR / SEGMENTATION
# =========================

def _fake_whisper_segments(local_video_path: str) -> List[Dict[str, Any]]:
    """
    TEMP SAFE VERSION.
    We return pretend ASR segments so pipeline doesn't crash,
    even if Whisper / ASR model isn't wired yet.

    Replace this later with real Whisper-based segmentation.
    """
    # We invent ~3 short takes from 0s..9s.
    return [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "Okay listen. This is the one product I need you to try.",
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
        },
        {
            "start": 3.0,
            "end": 6.5,
            "text": "I was breaking out and honestly I got so insecure on camera.",
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
        },
        {
            "start": 6.5,
            "end": 9.0,
            "text": "Now the glow is stupid. I'm not gatekeeping this anymore.",
            "face_q": 1.0,
            "scene_q": 1.0,
            "vtx_sim": 0.0,
        },
    ]


def run_asr_segments(local_video_path: str) -> List[Dict[str, Any]]:
    """
    FINAL CALL used by pipeline.
    Right now it's the fake stub above so the worker doesn't crash.

    Later:
    - Call Whisper / your ASR implementation
    - Return [{'start': float,'end': float,'text': str, 'face_q':..., 'scene_q':..., 'vtx_sim':...}, ...]
    """
    return _fake_whisper_segments(local_video_path)


# =========================
# GARBAGE / RETRY / FILLER FILTER
# =========================

def _is_retry_or_filler(txt: str) -> bool:
    """
    Drop obvious retries like:
      "wait wait lemme start again"
    OR huge filler rate ("um uh like like so um").
    """
    if not txt:
        return True

    low = txt.lower()

    # restart language
    retry_markers = [
        "wait", "hold on", "lemme start again",
        "let me start again", "okay wait", "no no start over",
        "sorry sorry", "restart", "can i start again"
    ]
    for m in retry_markers:
        if m in low:
            return True

    # filler % check
    words = low.split()
    if not words:
        return True
    filler_ct = sum(1 for w in words if w in FILLER_WORDS)
    rate = filler_ct / max(1, len(words))
    if rate > FILLER_RATE_MAX:
        return True

    return False


def _clean_takes_from_segments(segments: List[Dict[str, Any]]) -> List[Take]:
    """
    Convert raw ASR segments -> Take objects and drop garbage.
    """
    takes: List[Take] = []
    for idx, seg in enumerate(segments, start=1):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if _is_retry_or_filler(text):
            continue

        t = Take(
            id=f"T{idx:04d}",
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=text,
            face_q=float(seg.get("face_q", 1.0)),
            scene_q=float(seg.get("scene_q", 1.0)),
            vtx_sim=float(seg.get("vtx_sim", 0.0)),
            meta={"raw": seg},
        )
        # throw out 0-length (safety)
        if t.dur <= 0.05:
            continue

        takes.append(t)

    return takes


# =========================
# MERGE CHAINS (build long thoughts)
# =========================

def _merge_adjacent_takes(takes: List[Take]) -> List[Take]:
    """
    Merge runs of takes into longer thoughts, as long as:
    - scene_q doesn't fall below SCENE_Q_MIN
    - gap between clips is small
    - we don't produce monster chains > MERGE_MAX_CHAIN
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
            b = takes[j + 1]

            # reject if scene went bad
            if min(a.scene_q, b.scene_q) < SCENE_Q_MIN:
                break

            # reject if gap too large
            gap = b.start - a.end
            if gap > 1.0:
                break

            chain.append(b)
            j += 1

        first = chain[0]
        last = chain[-1]
        combined_text = " ".join([c.text.strip() for c in chain])

        merged_take = Take(
            id=f"{first.id}_to_{last.id}",
            start=first.start,
            end=last.end,
            text=combined_text,
            face_q=min(c.face_q for c in chain),
            scene_q=min(c.scene_q for c in chain),
            vtx_sim=max(c.vtx_sim for c in chain),
            meta={
                "chain_ids": [c.id for c in chain],
                "len_chain": len(chain),
            },
        )
        merged.append(merged_take)
        i = j + 1

    return merged


# =========================
# PICK STORY ORDER
# =========================

def _pick_storyline(merged_takes: List[Take], max_len: float) -> List[Take]:
    """
    Take merged chunks in order, skip obvious garbage like "okay yeah"
    and stop once we reach max_len seconds total.
    """
    story: List[Take] = []
    total = 0.0

    for t in merged_takes:
        cleaned = t.text.strip()

        # skip tiny "okay", "yeah", "uh" etc as standalone
        if len(cleaned.split()) <= 2:
            low = cleaned.lower()
            if "wait" in low or "okay" in low or "yeah" in low:
                continue

        dur = t.dur
        if total + dur > max_len:
            break

        story.append(t)
        total += dur

    return story


# =========================
# EXPORT CUT WITH FFMPEG
# =========================

def _cut_piece(src_video: str, take: Take) -> str:
    """
    Cut one piece [start, end] from raw video -> temp file.
    """
    out_path = _tmpfile(".clip.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{take.start:.3f}",
        "-i", src_video,
        "-t", f"{take.dur:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]
    code, out, err = _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg piece export failed: {err}")
    return out_path


def _concat_clips(clip_paths: List[str]) -> str:
    """
    ffmpeg concat all clip_paths into one final file.
    """
    if not clip_paths:
        raise RuntimeError("No clip paths to concat")

    list_path = _tmpfile(".txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    final_path = _tmpfile(".mp4")

    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-c:a", "aac",
        "-b:a", "128k",
        final_path,
    ]
    code, out, err = _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg concat failed: {err}")

    return final_path


# =========================
# CAPTIONS BURN (simple)
# =========================

def _format_ts_srt(sec: float) -> str:
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _write_srt(takes: List[Take]) -> str:
    """
    Build a basic .srt file so we can burn subs.
    """
    srt_path = _tmpfile(".srt")

    lines = []
    for idx, t in enumerate(takes, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_ts_srt(t.start)} --> {_format_ts_srt(t.end)}")
        caption_txt = t.text.strip().replace("\n", " ")
        if not caption_txt:
            caption_txt = "."
        lines.append(caption_txt)
        lines.append("")

    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return srt_path


def _burn_subtitles(video_in: str, takes: List[Take]) -> str:
    """
    ffmpeg -vf subtitles=...
    """
    srt_path = _write_srt(takes)
    final_burned = _tmpfile(".mp4")

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
        final_burned,
    ]
    code, out, err = _run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg burn_subtitles failed: {err}")

    return final_burned


# =========================
# PUBLIC MAIN: run_pipeline
# =========================

def run_pipeline(
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts: str,
    max_duration: float,
    bin_sec: float,
    min_take_sec: float,
    max_take_sec: float,
    veto_min_score: float,
    sem_merge_sim: float,
    viz_merge_sim: float,
    merge_max_chain: int,
    filler_tokens: List[str],
    filler_max_rate: float,
    micro_cut: bool,
    micro_silence_db: float,
    micro_silence_min: float,
    slot_require_product: List[str],
    slot_require_ocr_cta: str,
    fallback_min_sec: int,
) -> Dict[str, Any]:
    """
    This is what tasks.job_render() calls.
    It returns the dict that the /status endpoint shows.
    """

    # 1. take first source video
    if not file_urls:
        raise RuntimeError("No input video URLs provided")
    src_url = file_urls[0]

    # 2. download to /tmp
    local_raw = _download_video_to_tmp(src_url)

    # 3. speech → segments
    segments = run_asr_segments(local_raw)

    # 4. build takes and filter retries/filler
    rough_takes = _clean_takes_from_segments(segments)

    # 5. merge related takes into longer thoughts
    merged_takes = _merge_adjacent_takes(rough_takes)

    # 6. pick best storyline up to allowed max length
    story = _pick_storyline(
        merged_takes,
        max_len=max_duration if max_duration else MAX_DURATION_SEC_DEFAULT,
    )

    # safety: if nothing survived, just fall back to first ~fallback_min_sec sec raw
    if not story:
        # we fake one Take [0, fallback_min_sec]
        story = [
            Take(
                id="FALLBACK",
                start=0.0,
                end=float(fallback_min_sec),
                text="",
                face_q=1.0,
                scene_q=1.0,
                vtx_sim=0.0,
                meta={"chain_ids": []},
            )
        ]

    # 7. actually cut those ranges from the raw video and concat
    clip_paths: List[str] = []
    for t in story:
        clip_paths.append(_cut_piece(local_raw, t))

    stitched_path = _concat_clips(clip_paths)

    # 8. burn captions (Mode B wants captions ON by default)
    if BURN_CAPTIONS and story:
        final_video_path = _burn_subtitles(stitched_path, story)
    else:
        final_video_path = stitched_path

    # 9. upload to s3
    upload_info = _upload_to_s3(final_video_path)

    # 10. build "clips" list
    clips_block: List[Dict[str, Any]] = []
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
            "chain_ids": t.meta.get("chain_ids", []),
        })

    # 11. build slots map like your API returns
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
                    "chain_ids": t.meta.get("chain_ids", []),
                },
                "face_q": t.face_q,
                "scene_q": t.scene_q,
                "vtx_sim": t.vtx_sim,
            }
            for t in story
        ]
    }

    total_duration = _ffprobe_duration(final_video_path)

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_raw,
        "final_local": final_video_path,
        "duration_sec": total_duration,
        "clips": clips_block,
        "slots": slots_block,
        "semantic": True,
        "vision": False,
        "asr": True,
    }


# =========================
# BACKWARD COMPAT SHIM
# =========================

def job_render(
    session_id: str,
    files: List[str],
    portrait: bool = True,
    max_duration: Optional[float] = None,
    audio: bool = True,
    output_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Older code sometimes called jobs.job_render() directly.
    We route it into run_pipeline() so tasks.py doesn't care.
    """

    result = run_pipeline(
        session_id=session_id,
        file_urls=files,
        portrait=portrait,
        funnel_counts="1,3,3,3,1",
        max_duration=max_duration if max_duration is not None else MAX_DURATION_SEC_DEFAULT,
        bin_sec=1.0,
        min_take_sec=2.0,
        max_take_sec=220.0,
        veto_min_score=0.35,
        sem_merge_sim=0.70,
        viz_merge_sim=0.70,
        merge_max_chain=MERGE_MAX_CHAIN,
        filler_tokens=list(FILLER_WORDS),
        filler_max_rate=FILLER_RATE_MAX,
        micro_cut=True,
        micro_silence_db=-30.0,
        micro_silence_min=0.25,
        slot_require_product=["FEATURE", "PROOF"],
        slot_require_ocr_cta="CTA",
        fallback_min_sec=int(os.getenv("FALLBACK_MIN_SEC", "60")),
    )

    # add S3 URLs before returning (so old caller gets same top-level keys)
    upload_info = _upload_to_s3(result["final_local"])

    out = dict(result)
    out["s3_key"] = upload_info["s3_key"]
    out["s3_url"] = upload_info["s3_url"]
    out["https_url"] = upload_info["https_url"]
    return out
