import os, json, time, uuid, tempfile, subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

from worker.asr import transcribe_segments
import boto3

# ---------------- ENV ----------------
def _env_float(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip().split()[:1]
    try:
        return float(v[0]) if v else d
    except:
        return d


def _env_str(k, d):
    v = (os.getenv(k) or "").split("#")[0].strip()
    return v or d


FFMPEG_BIN = _env_str("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env_str("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET = _env_str("S3_BUCKET", "")
S3_PREFIX = _env_str("S3_PREFIX", "editdna/outputs")
AWS_REGION = _env_str("AWS_REGION", "us-east-1")
S3_ACL = _env_str("S3_ACL", "public-read")

MAX_DURATION_SEC = _env_float("MAX_DURATION_SEC", 120.0)
MIN_TAKE_SEC = _env_float("MIN_TAKE_SEC", 2.0)
MAX_TAKE_SEC = _env_float("MAX_TAKE_SEC", 20.0)
FILLER_MAX_RATE = _env_float("SEM_FILLER_MAX_RATE", 0.08)
CAPTIONS_MODE = _env_str("CAPTIONS", "burn").lower()
BURN_CAPTIONS = CAPTIONS_MODE in ("on", "burn", "burned", "subtitle", "1", "true", "yes")


# ---------------- MODEL ----------------
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


# ---------------- HELPERS ----------------
def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()


def _tmpfile(suffix=".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p


def _ffprobe_duration(path: str) -> float:
    code, out, err = _run(
        [
            FFPROBE_BIN,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            path,
        ]
    )
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


# ---------------- TEXT FILTERS ----------------
FILLERS = {"uh", "um", "like", "so", "okay", "sorry"}
RETRY_MARKERS = (
    "wait",
    "hold on",
    "let me start again",
    "start over",
    "no no",
    "redo",
    "take two",
    "i mean",
    "actually",
)


def _is_retry_or_filler(txt: str) -> bool:
    low = txt.lower()
    if any(m in low for m in RETRY_MARKERS):
        return True
    words = [w.strip(",.?!") for w in low.split()]
    if not words:
        return True
    rate = sum(1 for w in words if w in FILLERS) / max(1, len(words))
    return rate > FILLER_MAX_RATE


def _clamp_span(s: float, e: float) -> Tuple[float, float]:
    dur = e - s
    if dur <= 0:
        return s, s
    if dur > MAX_TAKE_SEC:
        return s, s + MAX_TAKE_SEC
    return s, e


# ---------------- ASR → TAKES ----------------
def _segments_to_takes(segments: List[Dict[str, Any]]) -> List[Take]:
    takes: List[Take] = []
    for i, seg in enumerate(segments, start=1):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        s, e = float(seg["start"]), float(seg["end"])
        while (e - s) > MAX_TAKE_SEC:
            chunk_s, chunk_e = s, s + MAX_TAKE_SEC
            takes.append(
                Take(
                    id=f"T{i:04d}_{len(takes)+1:02d}",
                    start=chunk_s,
                    end=chunk_e,
                    text=txt,
                )
            )
            s = chunk_e
        s, e = _clamp_span(s, e)
        if (e - s) >= MIN_TAKE_SEC:
            takes.append(Take(id=f"T{i:04d}", start=s, end=e, text=txt))
    return takes


def _drop_retries_and_trash(takes: List[Take]) -> List[Take]:
    out = []
    seen_text_norm = set()
    for t in takes:
        if _is_retry_or_filler(t.text):
            continue
        norm = "".join(c.lower() for c in t.text if (c.isalnum() or c.isspace())).strip()
        if norm in seen_text_norm:
            continue
        seen_text_norm.add(norm)
        out.append(t)
    return out


# ---------------- STORY SELECTOR ----------------
def _merge_adjacent_by_flow(takes: List[Take], max_chain: int = 3) -> List[Take]:
    if not takes:
        return []
    takes = sorted(takes, key=lambda x: x.start)
    merged: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (j + 1) < len(takes) and len(chain) < max_chain:
            a, b = chain[-1], takes[j + 1]
            if (b.start - a.end) > 1.0:
                break
            chain.append(b)
            j += 1
        first, last = chain[0], chain[-1]
        merged.append(
            Take(
                id=f"{first.id}_to_{last.id}",
                start=first.start,
                end=last.end,
                text=" ".join(c.text.strip() for c in chain),
                face_q=min(c.face_q for c in chain),
                scene_q=min(c.scene_q for c in chain),
                vtx_sim=max(c.vtx_sim for c in chain),
                chain_ids=[c.id for c in chain],
            )
        )
        i = j + 1
    return merged


def _pick_story_in_order(merged: List[Take], max_len: float) -> List[Take]:
    story, total = [], 0.0
    for t in merged:
        if total + t.dur > max_len:
            break
        story.append(t)
        total += t.dur
    if not story and merged:
        story = [max(merged, key=lambda x: x.dur)]
    return story


# ---------------- EXPORT ----------------
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
        _run(
            [
                FFMPEG_BIN,
                "-y",
                "-ss",
                f"{t.start:.3f}",
                "-i",
                src,
                "-t",
                f"{t.dur:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "48",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                part,
            ]
        )
    with open(listfile, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    final = _tmpfile(suffix=".mp4")
    _run(
        [
            FFMPEG_BIN,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            listfile,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "48",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            final,
        ]
    )
    if BURN_CAPTIONS:
        srt = _write_srt(story)
        burned = _tmpfile(suffix=".mp4")
        _run(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                final,
                "-vf",
                f"subtitles={srt}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "48",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                burned,
            ]
        )
        return burned
    return final


# ---------------- S3 ----------------
def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    stem = uuid.uuid4().hex
    key = f"{prefix}/{stem}_{int(time.time())}.mp4"
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


# ---------------- PUBLIC ENTRY ----------------
def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts: Union[str, Dict[str, int], None],
    max_duration: float,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    # normalize funnel counts
    if isinstance(funnel_counts, str):
        try:
            funnel_counts = json.loads(funnel_counts)
        except Exception:
            funnel_counts = None
    if funnel_counts is None:
        funnel_counts = {
            "HOOK": 1,
            "PROBLEM": 1,
            "FEATURE": 1,
            "PROOF": 1,
            "CTA": 1,
        }

    raw_local = _download_to_tmp(file_urls[0])

    # 1) ASR
    segs = transcribe_segments(raw_local)
    # 2) segments -> takes
    takes = _segments_to_takes(segs)
    # 3) drop retries/fillers/dups
    takes = _drop_retries_and_trash(takes)
    # 4) merge adjacent
    merged = _merge_adjacent_by_flow(takes, max_chain=3)
    # 5) pick story
    cap = float(max_duration or MAX_DURATION_SEC)
    story = _pick_story_in_order(merged, cap)

    # 6) export
    final_path = _export_video(raw_local, story)
    up = _upload_to_s3(final_path, s3_prefix=s3_prefix)

    clips_block = [
        {
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
        }
        for t in story
    ]

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
        "semantic": True,
        "vision": False,
        "asr": True,
        "funnel_counts": funnel_counts,
    }
