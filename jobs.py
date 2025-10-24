# jobs.py — EditDNA worker (ASR → visual → semantic → composer → render → S3)
from __future__ import annotations
import os, json, shlex, subprocess, tempfile, uuid, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# --------- infra helpers ----------
from s3_utils import upload_file  # must exist in your repo

# --------- optional deps (graceful fallbacks) ----------
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

# semantic + visual pass (our fused logic)
try:
    from semantic_visual_pass import (
        Take, dedup_takes, tag_slot, score_take, stitch_chain
    )
    _HAS_SEM = True
except Exception as e:
    print(f"[jobs.py] semantic_visual_pass unavailable ({e}); using no-op.")
    _HAS_SEM = False
    @dataclass
    class Take:
        id: str; start: float; end: float; text: str = ""; meta: Dict = None
        face_q: float = 1.0; scene_q: float = 1.0; vtx_sim: float = 0.0
        has_product: bool = False; ocr_hit: int = 0
    def dedup_takes(xs): return xs
    def tag_slot(t): return "PROOF"
    def score_take(t, slot): return 1.0
    def stitch_chain(xs): return xs

# micro-cut (sentence boundary)
try:
    from sentence_boundary import micro_cut_segments
    _HAS_MICROCUT = True
except Exception as e:
    print(f"[jobs.py] micro-cut unavailable ({e}); skipping fine sentence trimming.")
    _HAS_MICROCUT = False

# lightweight visual sampler (face/scene/vtx)
try:
    from vision_sampler import sample_visuals
    _HAS_VISION = True
except Exception as e:
    print(f"[jobs.py] vision_sampler unavailable ({e}); using defaults.")
    _HAS_VISION = False

# --------- env / config ----------
FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")      # using system ffmpeg
FFPROBE = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

MAX_TAKE_SEC      = float(os.getenv("MAX_TAKE_SEC", "20"))
MIN_TAKE_SEC      = float(os.getenv("MIN_TAKE_SEC", "1.5"))
MAX_DURATION_SEC  = float(os.getenv("MAX_DURATION_SEC", "120"))

ASR_ENABLED       = int(os.getenv("ASR_ENABLED", "1"))
ASR_MODEL_SIZE    = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANG          = os.getenv("ASR_LANG", "en")

S3_BUCKET         = os.environ.get("S3_BUCKET")
AWS_REGION        = (
    os.environ.get("AWS_DEFAULT_REGION")
    or os.environ.get("AWS_REGION")
    or "us-east-1"
)

S3_PREFIX         = os.getenv("S3_PREFIX", "editdna/outputs").rstrip("/")
S3_PUBLIC_BASE    = os.getenv("S3_PUBLIC_BASE")  # optional public dir override
S3_ACL            = os.getenv("S3_ACL", "public-read")  # your infra expects public-read

SLOT_ORDER = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]

# ---------- small utilities ----------
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
    print(f"[asr] loading whisper model: {ASR_MODEL_SIZE}", flush=True)
    model = whisper.load_model(ASR_MODEL_SIZE)
    res = model.transcribe(audio_path, language=ASR_LANG, verbose=False)
    segs = []
    for s in res.get("segments", []):
        segs.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s.get("text","").strip()
        })
    if not segs:
        dur = probe_duration(audio_path)
        segs = [{"start": 0.0, "end": dur, "text": res.get("text","").strip()}]
    print(f"[asr] segments: {len(segs)}", flush=True)
    return segs

# ---------- segmentation to takes ----------
def segment_to_takes(video_path: str, segments: List[Dict]) -> List[Take]:
    takes: List[Take] = []
    for seg in segments:
        s, e, txt = float(seg["start"]), float(seg["end"]), seg.get("text","").strip()
        cur = s
        while cur < e:
            nxt = min(e, cur + MAX_TAKE_SEC)
            if (nxt - cur) >= MIN_TAKE_SEC:
                takes.append(Take(
                    id=f"T{len(takes)+1:04d}",
                    start=cur, end=nxt, text=txt, meta={}
                ))
            cur = nxt
    print(f"[seg] takes: {len(takes)}", flush=True)
    return takes

# ---------- visual sampling ----------
def attach_visual_scores(video_path: str, takes: List[Take]) -> None:
    for t in takes:
        if _HAS_VISION:
            face_q, scene_q, vtx_sim, had_signal = sample_visuals(
                video_path, (t.start, t.end), text=t.text
            )
        else:
            face_q, scene_q, vtx_sim, had_signal = 1.0, 1.0, 0.0, False
        t.face_q  = float(face_q)
        t.scene_q = float(scene_q)
        t.vtx_sim = float(vtx_sim)
        t.meta["has_signal"] = bool(had_signal)

# ---------- semantic selection ----------
def select_by_slots(takes: List[Take]) -> Dict[str, List[Take]]:
    kept = dedup_takes(takes)
    by_slot: Dict[str, List[Take]] = {k: [] for k in SLOT_ORDER}
    for t in kept:
        slot = tag_slot(t)
        t.meta["slot"] = slot
        t.meta["score"] = score_take(t, slot)
        if t.meta["score"] >= 0:
            by_slot[slot].append(t)
    for k in by_slot:
        by_slot[k] = sorted(by_slot[k], key=lambda x: x.meta.get("score", 0.0), reverse=True)
    # stitch FEATURE/PROOF (smart long-demo merge)
    by_slot["FEATURE"] = stitch_chain(by_slot["FEATURE"])
    by_slot["PROOF"]   = stitch_chain(by_slot["PROOF"])
    return by_slot

# ---------- rendering (libx264 — safe on RunPod) ----------
def ffmpeg_subclip(in_path: str, out_path: str, start: float, end: float) -> None:
    dur = max(0.0, end - start)
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{start:.3f}", "-i", in_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", "48",
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

def render_funnel(video_path: str, by_slot: Dict[str, List[Take]]) -> Tuple[str, List[Dict]]:
    # order best 1 per slot (you can tune to 2+ if needed)
    def pick_best(lst, k): return sorted(lst, key=lambda x: x.meta.get("score",0.0), reverse=True)[:k]
    order: List[Take] = []
    order += pick_best(by_slot.get("HOOK",    []), 1)
    order += pick_best(by_slot.get("PROBLEM", []), 1)
    order += pick_best(by_slot.get("FEATURE", []), 1)
    order += pick_best(by_slot.get("PROOF",   []), 1)
    order += pick_best(by_slot.get("CTA",     []), 1)

    # trim to MAX_DURATION_SEC by clipping last segment if needed
    cur = 0.0
    clips: List[Tuple[Take, float, float]] = []
    for t in order:
        dur = max(0.0, t.end - t.start)
        if cur + dur > MAX_DURATION_SEC:
            take_end = t.start + max(0.0, MAX_DURATION_SEC - cur)
            if take_end > t.start:
                clips.append((t, t.start, take_end))
                cur = MAX_DURATION_SEC
            break
        else:
            clips.append((t, t.start, t.end))
            cur += dur

    tmp_parts = []
    clip_meta = []
    for i, (t, ss, ee) in enumerate(clips, 1):
        outp = _tmp(f".part{i:02d}.mp4")
        ffmpeg_subclip(video_path, outp, ss, ee)
        tmp_parts.append(outp)
        clip_meta.append({
            "id": t.id, "slot": t.meta.get("slot"),
            "start": ss, "end": ee,
            "score": t.meta.get("score"),
            "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
            "chain_ids": t.meta.get("chain_ids", []),
        })

    # concat or fallback to 1s black
    final_path = _tmp(".mp4")
    if tmp_parts:
        ffmpeg_concat(tmp_parts, final_path)
    else:
        _run([
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            final_path
        ])

    # cleanup part files
    for p in tmp_parts:
        try: os.remove(p)
        except: pass

    return final_path, clip_meta

# ---------- public job ----------
def job_render(local_input_path: str, product_context: Optional[Dict]=None) -> Dict:
    """
    Entry-point called by tasks.job_render (adapter).
    Expects a LOCAL absolute path to the input video file.
    """
    assert S3_BUCKET, "S3_BUCKET env is required"
    assert isinstance(local_input_path, str) and os.path.exists(local_input_path), \
        f"local_input_path does not exist: {local_input_path}"

    local_in = local_input_path

    # 1) extract audio for ASR
    wav_path = _tmp(".wav")
    _run([FFMPEG, "-y", "-i", local_in, "-vn", "-ac", "1", "-ar", "16000", wav_path])

    # 2) ASR → segments
    segments = asr_transcribe(wav_path)

    # 2.1) micro-cut (optional; controlled by env MICRO_CUT=1)
    if _HAS_MICROCUT and int(os.getenv("MICRO_CUT", "1")):
        segments = micro_cut_segments(segments, min_words=3)

    # 3) segmenter → takes
    takes = segment_to_takes(local_in, segments)

    # 4) visual scoring (face/scene/vtx)
    attach_visual_scores(local_in, takes)

    # 5) semantic selection & stitching (slots)
    by_slot = select_by_slots(takes) if _HAS_SEM else {"HOOK":[], "PROBLEM":[], "FEATURE":takes[:1], "PROOF":takes[1:2], "CTA":[]}

    # 6) render
    out_path, clip_meta = render_funnel(local_in, by_slot)

    # 7) upload to S3
    ts = int(time.time())
    out_key = f"{S3_PREFIX}/{uuid.uuid4().hex}_{ts}.mp4"
    upload_file(out_path, out_key, bucket=S3_BUCKET, region=AWS_REGION, content_type="video/mp4", acl=S3_ACL)

    https_url = (
        f"{S3_PUBLIC_BASE.rstrip('/')}/{out_key}"
        if S3_PUBLIC_BASE
        else f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{out_key}"
    )

    data = {
        "ok": True,
        "input_local": local_in,
        "duration_sec": probe_duration(out_path),
        "s3_key": out_key,
        "s3_url": f"s3://{S3_BUCKET}/{out_key}",
        "https_url": https_url,
        "clips": clip_meta,
        "slots": {k: [asdict(t) if hasattr(t, "__dataclass_fields__") else getattr(t, "__dict__", {}) for t in v]
                  for k,v in by_slot.items()},
        "semantic": bool(_HAS_SEM),
        "vision": bool(_HAS_VISION),
        "asr": bool(ASR_ENABLED and _HAS_WHISPER),
    }

    # cleanup
    try: os.remove(wav_path)
    except: pass
    try: os.remove(out_path)
    except: pass

    print(f"[jobs] uploaded final -> {https_url}", flush=True)
    return data


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 jobs.py /abs/path/input.mp4")
        raise SystemExit(2)
    print(json.dumps(job_render(sys.argv[1]), indent=2))
