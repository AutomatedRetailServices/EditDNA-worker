# /workspace/EditDNA-worker/pipeline.py
from __future__ import annotations
import os
import time
import uuid
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------
# ENV & OPENAI (mandatory as you asked)
# ---------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set — pipeline requires LLM scoring")

client = OpenAI(api_key=OPENAI_API_KEY)

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "editdna/outputs").rstrip("/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_ACL = os.getenv("S3_ACL", "public-read")

MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "120"))
MAX_TAKE_SEC = float(os.getenv("MAX_TAKE_SEC", "20"))
MIN_TAKE_SEC = float(os.getenv("MIN_TAKE_SEC", "2"))

BAD_PHRASES = [
    "wait",
    "hold on",
    "lemme start again",
    "let me start again",
    "start over",
    "no no",
    "no, no",
    "redo",
    "sorry",
    "that's not it",
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
    "so if you want to check them out",
]

UGLY_BRANCHES = [
    "but if you don't like the checker print",
    "but if you don't like the checker",
    "but if you do",
    "but if you don't",
    "but if you",
]

FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware",
    "comes with", "it has", "it also has",
    "it's actually", "this isn't just", "design",
]


from dataclasses import dataclass

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    rule_score: float = 0.0
    sem_score: float = 0.0
    vis_score: float = 0.0
    llm_score: float = 0.0

    @property
    def dur(self) -> float:
        return self.end - self.start


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()


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


def _load_asr_segments(local_video_path: str) -> List[Dict[str, Any]]:
    from worker import asr as worker_asr
    if hasattr(worker_asr, "transcribe_segments"):
        segs = worker_asr.transcribe_segments(local_video_path)
    else:
        segs = worker_asr.transcribe_local(local_video_path)
    norm = []
    for s in segs:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        norm.append(
            {
                "text": txt,
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
            }
        )
    return norm


def _split_into_clauses(text: str) -> List[str]:
    if not text:
        return []
    text = " ".join(text.split())
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

    clauses = [c for c in clauses if len(c.split()) >= 3]
    return clauses


def _clause_rule_score(c: str) -> float:
    low = c.lower().strip()
    if not low:
        return 0.0
    for p in BAD_PHRASES:
        if p in low:
            return 0.0
    for p in UGLY_BRANCHES:
        if p in low:
            return 0.1
    for p in CTA_FLUFF:
        if p in low:
            return 0.3
    for h in FEATURE_HINTS:
        if h in low:
            return 0.9
    return 0.6


def _assign_times(seg_start: float, seg_end: float, clauses: List[str]) -> List[Tuple[float, float, str]]:
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


def _semantic_score(text: str) -> float:
    try:
        from worker import semantic as sem
        return float(sem.score_text(text))
    except Exception:
        pass
    try:
        from worker import micro_semantic as sem2
        return float(sem2.score_text(text))
    except Exception:
        pass
    return 0.0


def _vision_score(local_video_path: str, start: float, end: float) -> float:
    try:
        from worker import vision as vis
        return float(vis.score_clip(local_video_path, start, end))
    except Exception:
        return 0.0


LLM_SYSTEM = (
    "You check if a spoken clause belongs in a clean product/sales UGC video. "
    "Return ONLY JSON: {\"keep\": true/false, \"reason\": \"...\"}. "
    "Keep product/benefit/feature/hook/closing CTA. Drop restarts, rambles, branches."
)


def _llm_keep_clause(text: str) -> float:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=120,
    )
    content = resp.choices[0].message.content or ""
    lc = content.lower()
    if "\"keep\": true" in lc or "keep\":true" in lc:
        return 1.0
    return 0.0


def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
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


def _upload_to_s3(local_path: str, s3_prefix: Optional[str] = None) -> Dict[str, str]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    import boto3
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


def run_pipeline(
    *,
    local_video_path: str,
    session_id: str,
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    real_dur = _ffprobe_duration(local_video_path)
    cap = min(real_dur or MAX_DURATION_SEC, MAX_DURATION_SEC)

    segs = _load_asr_segments(local_video_path)

    clause_takes: List[Take] = []
    for idx, seg in enumerate(segs, start=1):
        seg_text = seg["text"]
        seg_start = seg["start"]
        seg_end = min(seg["end"], seg_start + MAX_TAKE_SEC)

        clauses = _split_into_clauses(seg_text)
        if not clauses:
            continue

        timed = _assign_times(seg_start, seg_end, clauses)
        for c_idx, (cs, ce, ctext) in enumerate(timed, start=1):
            if (ce - cs) < MIN_TAKE_SEC:
                continue

            rule_s = _clause_rule_score(ctext)
            sem_s = _semantic_score(ctext)
            vis_s = _vision_score(local_video_path, cs, ce)
            llm_s = _llm_keep_clause(ctext)

            clause_takes.append(
                Take(
                    id=f"ASR{idx:04d}_c{c_idx}",
                    start=cs,
                    end=ce,
                    text=ctext,
                    rule_score=rule_s,
                    sem_score=sem_s,
                    vis_score=vis_s,
                    llm_score=llm_s,
                )
            )

    story: List[Take] = []
    total_dur = 0.0
    for t in clause_takes:
        keep = False
        if t.llm_score >= 0.9:
            keep = True
        else:
            combined = (0.4 * t.rule_score) + (0.3 * t.sem_score) + (0.3 * t.vis_score)
            if combined >= 0.55:
                keep = True
        if not keep:
            continue
        if total_dur + t.dur > cap:
            break
        story.append(t)
        total_dur += t.dur

    if not story:
        story = [
            Take(id="FALLBACK", start=0.0, end=min(5.0, cap), text="")
        ]

    final_path = _export_concat(local_video_path, story)
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
            "face_q": t.vis_score or 1.0,
            "scene_q": 1.0,
            "vtx_sim": t.sem_score or 0.0,
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
        first = story[0]
        slots["HOOK"].append(
            {
                "id": first.id,
                "start": first.start,
                "end": first.end,
                "text": _trim(first.text),
                "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [first.id]},
            }
        )
    if len(story) > 2:
        for mid in story[1:-1]:
            slots["FEATURE"].append(
                {
                    "id": mid.id,
                    "start": mid.start,
                    "end": mid.end,
                    "text": _trim(mid.text),
                    "meta": {"slot": "FEATURE", "score": 2.0, "chain_ids": [mid.id]},
                }
            )
    if len(story) >= 2:
        last = story[-1]
        slots["CTA"].append(
            {
                "id": last.id,
                "start": last.start,
                "end": last.end,
                "text": _trim(last.text),
                "meta": {"slot": "CTA", "score": 2.0, "chain_ids": [last.id]},
            }
        )

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
        "asr": True,
        "semantic": True,
        "vision": True,
    }
