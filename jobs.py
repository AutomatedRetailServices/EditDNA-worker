# jobs.py — EditDNA pipeline (ASR → semantic/visual → compose → render → S3)
from __future__ import annotations
import os, json, shlex, subprocess, tempfile, uuid, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# --- helpers for S3 ---
from s3_utils import upload_file

# --- semantic/visual pass (fast + local) ---
try:
    from worker.semantic_visual_pass import (
        Take, dedup_takes, tag_slot, score_take, stitch_chain
    )
    _HAS_SEM = True
except Exception as e:
    print(f"[jobs] semantic pass unavailable ({e}); using no-op.")
    _HAS_SEM = False
    @dataclass
    class Take:
        id: str; start: float; end: float; text: str = ""; meta: Dict = None
        face_q: float = 1.0; scene_q: float = 1.0; vtx_sim: float = 0.0
        has_product: bool = False; ocr_hit: int = 0
    def dedup_takes(xs): return xs
    def tag_slot(t): return "FEATURE"
    def score_take(t, slot): return 1.0
    def stitch_chain(xs): return xs

# --- optional: ASR (whisper) ---
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

# -------- env / config --------
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

MAX_TAKE_SEC      = float(os.getenv("MAX_TAKE_SEC", "20"))
MAX_DURATION_SEC  = float(os.getenv("MAX_DURATION_SEC", "180"))

ASR_ENABLED       = int(os.getenv("ASR_ENABLED", "1"))
ASR_MODEL_SIZE    = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANG          = os.getenv("ASR_LANG", "en")

S3_BUCKET         = os.environ.get("S3_BUCKET")
AWS_REGION        = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

# Slot order: HOOK → PROBLEM/benefits → FEATURE → PROOF → CTA
SLOT_ORDER = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]

def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    print("[ff] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def _tmp(ext=""):
    return os.path.join(tempfile.gettempdir(), f"ed_{uuid.uuid4().hex}{ext}")

def _float(s: str) -> float:
    try: return float(s)
    except: return 0.0

# ---------- media probes ----------
def probe_duration(path: str) -> float:
    try:
        out = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nokey=1:noprint_wrappers=1", path], check=True)
        return _float(out.stdout.decode().strip())
    except Exception as e:
        print("[ffprobe] duration failed:", e)
        return 0.0

# ---------- ASR ----------
def asr_transcribe(audio_path: str) -> List[Dict]:
    if not ASR_ENABLED or not _HAS_WHISPER:
        dur = probe_duration(audio_path)
        return [{"start": 0.0, "end": dur, "text": ""}]
    print(f"[asr] loading whisper model: {ASR_MODEL_SIZE}")
    model = whisper.load_model(ASR_MODEL_SIZE)
    res = model.transcribe(audio_path, language=ASR_LANG, verbose=False)
    segs = []
    for s in res.get("segments", []):
        segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","").strip()})
    if not segs:
        dur = probe_duration(audio_path)
        segs = [{"start": 0.0, "end": dur, "text": res.get("text","").strip()}]
    return segs

# ---------- segmentation to takes ----------
def segment_to_takes(video_path: str, segments: List[Dict]) -> List[Take]:
    takes: List[Take] = []
    tid = 0
    for seg in segments:
        s, e, txt = float(seg["start"]), float(seg["end"]), seg.get("text","").strip()
        cur = s
        while cur < e:
            nxt = min(e, cur + MAX_TAKE_SEC)
            if (nxt - cur) >= 1.0:  # avoid micro slivers
                tid += 1
                takes.append(Take(
                    id=f"T{tid:04d}",
                    start=cur, end=nxt, text=txt,
                    face_q=1.0, scene_q=1.0, vtx_sim=0.0,
                    has_product=False, ocr_hit=0, meta={}
                ))
            cur = nxt
    return takes

# ---------- render helpers ----------
def ffmpeg_subclip(in_path: str, out_path: str, start: float, end: float) -> None:
    dur = max(0.0, end - start)
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

def ffmpeg_concat(parts: List[str], out_path: str) -> None:
    lst = _tmp(".txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
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

# ---------- semantic selection ----------
def _parse_counts() -> Dict[str, int]:
    raw = os.getenv("FUNNEL_COUNTS", "0,0,0,0,0").strip()
    try:
        c_hook, c_prob, c_feat, c_proof, c_cta = [int(x) for x in raw.split(",")]
    except Exception:
        c_hook, c_prob, c_feat, c_proof, c_cta = 0, 0, 0, 0, 0
    return {
        "HOOK": c_hook, "PROBLEM": c_prob,
        "FEATURE": c_feat, "PROOF": c_proof, "CTA": c_cta
    }

def select_by_slots(takes: List[Take]) -> Dict[str, List[Take]]:
    kept = dedup_takes(takes) if _HAS_SEM else takes
    by_slot: Dict[str, List[Take]] = {k: [] for k in SLOT_ORDER}

    for t in kept:
        slot = tag_slot(t)
        t.meta["slot"] = slot
        t.meta["score"] = score_take(t, slot) if _HAS_SEM else 1.0
        if t.meta["score"] is None:
            t.meta["score"] = 0.0
        if t.meta["score"] >= 0:
            by_slot[slot].append(t)

    # sort per slot by score desc
    for k in by_slot:
        by_slot[k] = sorted(by_slot[k], key=lambda x: x.meta.get("score", 0.0), reverse=True)

    # stitch longer chains for idea continuity
    by_slot["FEATURE"] = stitch_chain(by_slot["FEATURE"])
    by_slot["PROOF"]   = stitch_chain(by_slot["PROOF"])
    by_slot["HOOK"]    = stitch_chain(by_slot["HOOK"])
    # PROBLEM and CTA typically short; stitch anyway for safety:
    by_slot["PROBLEM"] = stitch_chain(by_slot["PROBLEM"])
    by_slot["CTA"]     = stitch_chain(by_slot["CTA"])

    # apply caps per slot; 0 = unlimited
    want = _parse_counts()
    for k, limit in want.items():
        if limit > 0:
            by_slot[k] = by_slot[k][:limit]
    return by_slot

# ---------- compose final timeline ----------
def build_timeline(by_slot: Dict[str, List[Take]], max_total: float) -> List[Take]:
    """Walk HOOK→PROBLEM→FEATURE→PROOF→CTA and add clips until we hit max_total."""
    tline: List[Take] = []
    cur = 0.0
    for slot in SLOT_ORDER:
        for t in by_slot.get(slot, []):
            dur = max(0.0, t.end - t.start)
            if cur + dur <= max_total:
                tline.append(t); cur += dur
            else:
                # trim last clip to fit remaining budget if any
                remain = max(0.0, max_total - cur)
                if remain >= 1.0:
                    trimmed = Take(
                        id=t.id, start=t.start, end=t.start + remain, text=t.text,
                        face_q=t.face_q, scene_q=t.scene_q, vtx_sim=t.vtx_sim,
                        has_product=t.has_product, ocr_hit=t.ocr_hit,
                        meta=dict(t.meta or {})
                    )
                    tline.append(trimmed); cur += remain
                return tline
    return tline

# ---------- render funnel ----------
def render_funnel(video_path: str, by_slot: Dict[str, List[Take]]) -> Tuple[str, List[Dict]]:
    timeline = build_timeline(by_slot, MAX_DURATION_SEC)
    if not timeline:
        # fallback: export a 3s black clip so API still returns something
        final_path = _tmp(".mp4")
        _run([
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=3",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            final_path
        ])
        return final_path, []
    # export parts then concat
    part_files, clip_meta = [], []
    for i, t in enumerate(timeline, 1):
        outp = _tmp(f".part{i:02d}.mp4")
        ffmpeg_subclip(video_path, outp, t.start, t.end)
        part_files.append(outp)
        clip_meta.append({
            "id": t.id, "slot": t.meta.get("slot"),
            "start": t.start, "end": t.end,
            "score": t.meta.get("score"),
            "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
            "chain_ids": t.meta.get("chain_ids", [])
        })
    final_path = _tmp(".mp4")
    ffmpeg_concat(part_files, final_path)
    # cleanup
    for p in part_files:
        try: os.remove(p)
        except: pass
    return final_path, clip_meta

# ---------- public job ----------
def job_render(local_path: str) -> Dict:
    assert S3_BUCKET, "S3_BUCKET env is required"

    # 1) extract audio for ASR (wav mono 16k)
    wav_path = _tmp(".wav")
    _run([FFMPEG, "-y", "-i", local_path, "-vn", "-ac", "1", "-ar", "16000", wav_path])

    # 2) ASR
    segments = asr_transcribe(wav_path)
    print(f"[asr] segments: {len(segments)}")

    # 3) split to takes
    takes = segment_to_takes(local_path, segments)
    print(f"[seg] takes: {len(takes)}")

    # 4) select by slots (semantic aware; stitches long ideas)
    by_slot = select_by_slots(takes)

    # 5) render
    out_path, clip_meta = render_funnel(local_path, by_slot)

    # 6) upload
    ts = int(time.time())
    out_key = f"editdna/outputs/{uuid.uuid4().hex}_{ts}.mp4"
    upload_file(out_path, out_key, bucket=S3_BUCKET, region=AWS_REGION, content_type="video/mp4")

    data = {
        "ok": True,
        "input_local": local_path,
        "duration_sec": probe_duration(out_path),
        "s3_key": out_key,
        "s3_url": f"s3://{S3_BUCKET}/{out_key}",
        "https_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{out_key}",
        "clips": clip_meta,
        "slots": {
            k: [
                {
                    "id": t.id, "start": t.start, "end": t.end, "text": t.text,
                    "meta": {"chain_ids": t.meta.get("chain_ids", [])},
                    "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
                    "has_product": t.has_product, "ocr_hit": t.ocr_hit
                } for t in by_slot.get(k, [])
            ] for k in SLOT_ORDER
        },
        "semantic": bool(_HAS_SEM),
        "vision": False,  # (wire up later if you enable vtx/ocr)
        "asr": bool(ASR_ENABLED and _HAS_WHISPER),
    }
    # cleanup
    try: os.remove(wav_path)
    except: pass
    try: os.remove(out_path)
    except: pass
    return data

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 jobs.py /abs/path/input.mp4")
        raise SystemExit(2)
    print(json.dumps(job_render(sys.argv[1]), indent=2))
