# jobs.py — EditDNA worker: ASR → micro-cuts → semantic+visual scoring → funnel compose → render → S3
from __future__ import annotations
import os, json, shlex, subprocess, tempfile, uuid, time, pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# ---------- infra helpers ----------
from s3_utils import upload_file  # must exist in your repo

# ---------- optional deps ----------
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

# semantic / visual pass (we just replaced this in worker/semantic_visual_pass.py)
from worker.semantic_visual_pass import (
    Take, dedup_takes, tag_slot, score_take, stitch_chain
)

# micro-cuts (safe fallback if module missing)
try:
    from worker.sentence_boundary import micro_cut_takes  # expected function name in your file
    _HAS_MICRO = True
except Exception:
    _HAS_MICRO = False
    def micro_cut_takes(video_path: str, takes: List[Take]) -> List[Take]:
        # no-op fallback: keep original takes
        return takes

# ---------- env / config ----------
FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

# Granularity and cap
MAX_TAKE_SEC = float(os.getenv("MAX_TAKE_SEC", "20"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "120"))

# Funnel counts: "H,P,F,PR,CTA" where 0 = unlimited
_FUNNEL_COUNTS_ENV = (os.getenv("FUNNEL_COUNTS", "1,1,1,1,1").strip() or "1,1,1,1,1")
def _parse_counts(val: str) -> Tuple[int,int,int,int,int]:
    parts = (val or "").split(",")
    parts = [p.strip() for p in parts if p.strip() != ""]
    while len(parts) < 5: parts.append("1")
    H,P,F,PR,CTA = [int(x) if x.isdigit() else 1 for x in parts[:5]]
    return H,P,F,PR,CTA
FUNNEL_COUNTS = _parse_counts(_FUNNEL_COUNTS_ENV)

# S3
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

# ---------- tiny utils ----------
def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    print("[ff] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def _tmp(ext=""):
    return os.path.join(tempfile.gettempdir(), f"ed_{uuid.uuid4().hex}{ext}")

def _float(s: str) -> float:
    try: return float(s)
    except: return 0.0

def probe_duration(path: str) -> float:
    try:
        out = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nokey=1:noprint_wrappers=1", path], check=True)
        return _float(out.stdout.decode().strip())
    except Exception as e:
        print("[ffprobe] duration failed:", e)
        return 0.0

# ---------- ASR ----------
def asr_segments(audio_path: str, lang: Optional[str]="en") -> List[Dict]:
    if not _HAS_WHISPER:
        dur = probe_duration(audio_path)
        return [{"start": 0.0, "end": dur, "text": ""}]
    print(f"[asr] whisper tiny @ lang={lang}")
    model = whisper.load_model(os.getenv("ASR_MODEL_SIZE", "tiny"))
    res = model.transcribe(audio_path, language=lang, verbose=False)
    segs = []
    for s in res.get("segments", []):
        segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","").strip()})
    if not segs:
        dur = probe_duration(audio_path)
        segs = [{"start": 0.0, "end": dur, "text": res.get("text","").strip()}]
    print(f"[asr] segments: {len(segs)}")
    return segs

# ---------- segmentation to takes (pre-micro) ----------
def segment_to_takes(video_path: str, segments: List[Dict]) -> List[Take]:
    takes: List[Take] = []
    for seg in segments:
        s, e, txt = float(seg["start"]), float(seg["end"]), seg.get("text","").strip()
        cur = s
        while cur < e:
            nxt = min(e, cur + MAX_TAKE_SEC)
            if (nxt - cur) > 0.35:  # avoid zero-length
                takes.append(Take(
                    id=f"T{len(takes)+1:04d}",
                    start=cur, end=nxt, text=txt,
                    face_q=1.0, scene_q=1.0, vtx_sim=0.0, has_product=False, ocr_hit=0, meta={}
                ))
            cur = nxt
    print(f"[seg] takes: {len(takes)}")
    return takes

# ---------- render helpers ----------
def ffmpeg_subclip(in_path: str, out_path: str, start: float, end: float) -> None:
    dur = max(0.0, end - start)
    if dur <= 0.01:
        raise RuntimeError("subclip length too small")
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{start:.3f}", "-i", in_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    _run(cmd)

def ffmpeg_concat(tsv_paths: List[str], out_path: str) -> None:
    lst = _tmp(".txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in tsv_paths:
            f.write(f"file '{p}'\n")
    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0", "-i", lst,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    _run(cmd)
    try: os.remove(lst)
    except: pass

# ---------- selection helpers ----------
def _pick(lst: List[Take], k: int) -> List[Take]:
    if k == 0:   # 0 means "unlimited"
        return sorted(lst, key=lambda x: x.meta.get("score", 0.0), reverse=True)
    return sorted(lst, key=lambda x: x.meta.get("score", 0.0), reverse=True)[:max(0,k)]

def _cap_to_duration(in_path: str, clips: List[Take], max_sec: float) -> List[Take]:
    out, cur = [], 0.0
    for t in clips:
        dur = max(0.0, t.end - t.start)
        if cur + dur <= max_sec + 1e-3:
            out.append(t); cur += dur
        else:
            # trim last clip to fit exactly
            room = max_sec - cur
            if room > 0.35:
                trimmed = Take(**{**t.__dict__})
                trimmed.end = trimmed.start + room
                out.append(trimmed)
            break
    return out

# ---------- main public job ----------
def job_render(local_path: str, options: Optional[Dict]=None) -> Dict:
    """
    local_path: absolute path to the input video (tasks.py downloads URLs for you)
    options: optional dict from web layer to override env at runtime:
        {
          "FUNNEL_COUNTS": "1,0,0,0,1",
          "MAX_DURATION_SEC": 220,
          "SEM_MERGE_SIM": 0.70, "VIZ_MERGE_SIM": 0.70, "MERGE_MAX_CHAIN": 10
        }
    """
    assert S3_BUCKET, "S3_BUCKET env is required"
    options = options or {}

    # allow runtime override
    counts = _parse_counts(str(options.get("FUNNEL_COUNTS", _FUNNEL_COUNTS_ENV)))
    global MAX_DURATION_SEC
    MAX_DURATION_SEC = float(options.get("MAX_DURATION_SEC", MAX_DURATION_SEC))

    # 1) extract audio for ASR
    wav_path = _tmp(".wav")
    _run([FFMPEG, "-y", "-i", local_path, "-vn", "-ac", "1", "-ar", "16000", wav_path])
    segs = asr_segments(wav_path)

    # 2) coarse takes → 3) micro-cuts (sentence boundary pass)
    coarse = segment_to_takes(local_path, segs)
    micro  = micro_cut_takes(local_path, coarse) if _HAS_MICRO else coarse
    print(f"[micro] input_takes={len(coarse)} → micro_takes={len(micro)}")

    # 4) semantic/visual dedup + slot tagging + scoring
    kept = dedup_takes(micro)
    by_slot: Dict[str, List[Take]] = {"HOOK":[], "PROBLEM":[], "FEATURE":[], "PROOF":[], "CTA":[]}
    for t in kept:
        slot = tag_slot(t)  # benefits map into PROBLEM here
        t.meta["slot"] = slot
        t.meta["score"] = score_take(t, slot)
        if t.meta["score"] >= 0:
            by_slot[slot].append(t)

    # 5) smart stitch for longer demo/feature proof
    by_slot["PROOF"]   = stitch_chain(by_slot["PROOF"])
    by_slot["FEATURE"] = stitch_chain(by_slot["FEATURE"])

    # 6) pick clips according to FUNNEL_COUNTS (0 = unlimited)
    h,p,f,pr,cta = counts
    picks: List[Take] = []
    picks += _pick(by_slot["HOOK"],    h)
    picks += _pick(by_slot["PROBLEM"], p)
    picks += _pick(by_slot["FEATURE"], f)
    picks += _pick(by_slot["PROOF"],   pr)
    picks += _pick(by_slot["CTA"],     cta)

    # safety: if some slots empty, still try to output something (fallback: top hooks/features)
    if not picks:
        fallback = sorted([*by_slot["HOOK"], *by_slot["FEATURE"], *by_slot["PROOF"]],
                          key=lambda x: x.meta.get("score", 0.0), reverse=True)[:3]
        picks = fallback

    # 7) cap to MAX_DURATION_SEC and render
    picks = _cap_to_duration(local_path, picks, MAX_DURATION_SEC)
    parts = []
    clips_meta = []
    for i, t in enumerate(picks, 1):
        outp = _tmp(f".part{i:02d}.mp4")
        ffmpeg_subclip(local_path, outp, t.start, t.end)
        parts.append(outp)
        clips_meta.append({
            "id": t.id, "slot": t.meta.get("slot"),
            "start": t.start, "end": t.end,
            "score": t.meta.get("score"),
            "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
            "chain_ids": t.meta.get("chain_ids", [])
        })

    final_path = _tmp(".mp4")
    if parts:
        ffmpeg_concat(parts, final_path)
    else:
        # 1-sec black fail-safe
        _run([
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            final_path
        ])

    # 8) upload
    ts = int(time.time())
    out_key = f"editdna/outputs/{uuid.uuid4().hex}_{ts}.mp4"
    upload_file(final_path, out_key, bucket=S3_BUCKET, region=AWS_REGION, content_type="video/mp4")

    # build slot debug payload
    slots_debug = {k: [
        {"id": t.id, "start": t.start, "end": t.end, "text": t.text,
         "meta": {**t.meta}, "face_q": t.face_q, "scene_q": t.scene_q,
         "vtx_sim": t.vtx_sim, "has_product": t.has_product, "ocr_hit": t.ocr_hit}
        for t in v
    ] for k,v in by_slot.items()}

    # 9) cleanup temp parts
    for p in parts:
        try: os.remove(p)
        except: pass

    dur = probe_duration(final_path)
    try: os.remove(wav_path)
    except: pass
    try: os.remove(final_path)
    except: pass

    return {
        "ok": True,
        "input_local": local_path,
        "duration_sec": round(dur, 3),
        "s3_key": out_key,
        "s3_url": f"s3://{S3_BUCKET}/{out_key}",
        "https_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{out_key}",
        "clips": clips_meta,
        "slots": slots_debug,
        "semantic": True,   # semantic pass active
        "vision": False,    # set True when you wire vision sampler
        "asr": _HAS_WHISPER,
    }

# local smoke
if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) < 2:
        print("usage: python3 jobs.py /abs/path/input.mp4")
        raise SystemExit(2)
    res = job_render(sys.argv[1], options={"FUNNEL_COUNTS": os.getenv("FUNNEL_COUNTS","1,1,1,1,1"),
                                           "MAX_DURATION_SEC": float(os.getenv("MAX_DURATION_SEC","120"))})
    pprint.pprint(res)
