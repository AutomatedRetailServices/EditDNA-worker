# jobs.py  — EditDNA worker (unified ASR + Visual + Semantic + Render)
from __future__ import annotations
import os, json, math, shlex, subprocess, tempfile, uuid, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# --- infra helpers ---
from s3_utils import upload_file, presigned_url

# --- optional deps (graceful fallbacks) ---
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

# semantic pass (must exist – you added it)
try:
    from worker.semantic_visual_pass import (
        Take, dedup_takes, tag_slot, score_take, stitch_chain, continuity_chains
    )
    _HAS_SEM = True
except Exception as e:
    print(f"[jobs.py] Semantic pass unavailable ({e}); using no-op.")
    _HAS_SEM = False
    @dataclass
    class Take:
        id: str; start: float; end: float; text: str = ""; meta: Dict = None
    def dedup_takes(xs): return xs
    def tag_slot(t): return "PROOF"
    def score_take(t, slot): return 1.0
    def stitch_chain(xs): return xs
    def continuity_chains(xs): return [[t] for t in xs]

try:
    from worker.vision_sampler import sample_visuals
    _HAS_VISION = True
except Exception as e:
    print(f"[jobs.py] vision_sampler unavailable ({e}); vtx/face/scene = defaults.")
    _HAS_VISION = False

# -------- env / config --------
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

MAX_TAKE_SEC = float(os.getenv("MAX_TAKE_SEC", "20"))
MIN_TAKE_SEC = float(os.getenv("MIN_TAKE_SEC", "1.5"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "120"))

ASR_ENABLED = int(os.getenv("ASR_ENABLED", "1"))
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANG = os.getenv("ASR_LANG", "en")

FUNNEL_COUNTS = os.getenv("FUNNEL_COUNTS", "99,99,99,99")
# we’ll just pick 1 per slot for MVP; ENV kept for compatibility

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

SLOT_ORDER = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]


# ---------- small utilities ----------
def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    print("[ff] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def _tmp(ext=""):
    p = os.path.join(tempfile.gettempdir(), f"ed_{uuid.uuid4().hex}{ext}")
    return p

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
    """
    Return list of segments: [{'start': float, 'end': float, 'text': str}, ...]
    """
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
    """
    Split ASR segments into ≤ MAX_TAKE_SEC windows, honoring MIN_TAKE_SEC.
    """
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
    return takes


# ---------- visual sampling ----------
def attach_visual_scores(video_path: str, takes: List[Take]) -> None:
    """
    Mutates each take.meta with: face_q, scene_q, vtx_sim
    """
    for t in takes:
        if _HAS_VISION:
            face_q, scene_q, vtx_sim, had_signal = sample_visuals(video_path, (t.start, t.end), text=t.text)
        else:
            face_q, scene_q, vtx_sim, had_signal = 1.0, 1.0, 0.0, False
        t.meta["face_q"]  = float(face_q)
        t.meta["scene_q"] = float(scene_q)
        t.meta["vtx_sim"] = float(vtx_sim)
        t.meta["has_signal"] = bool(had_signal)


# ---------- semantic selection ----------
def select_by_slots(takes: List[Take]) -> Dict[str, List[Take]]:
    """
    Tag, score, dedup, and return best candidate(s) per slot.
    """
    kept = dedup_takes(takes)
    by_slot: Dict[str, List[Take]] = {k: [] for k in SLOT_ORDER}

    for t in kept:
        slot = tag_slot(t)
        t.meta["slot"] = slot
        # score_take uses the env-weighted fusion inside semantic_visual_pass
        t.meta["score"] = score_take(t, slot)
        if t.meta["score"] >= 0:
            by_slot[slot].append(t)

    # simple pick-1 per slot (you can expand to top-k later)
    for k in by_slot:
        by_slot[k] = sorted(by_slot[k], key=lambda x: x.meta.get("score", 0.0), reverse=True)[:1]

    # allow stitching for FEATURE / PROOF to keep long demos
    if by_slot["FEATURE"]:
        by_slot["FEATURE"] = stitch_chain(by_slot["FEATURE"])
    if by_slot["PROOF"]:
        by_slot["PROOF"] = stitch_chain(by_slot["PROOF"])

    return by_slot


# ---------- rendering ----------
def ffmpeg_subclip(in_path: str, out_path: str, start: float, end: float) -> None:
    dur = max(0.0, end - start)
    cmd = [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", in_path, "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "128k", out_path]
    _run(cmd)

def ffmpeg_concat(tsv_paths: List[str], out_path: str) -> None:
    """
    Concat with demuxer (file list). tsv_paths is a list of video file paths.
    """
    lst = _tmp(".txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in tsv_paths:
            f.write(f"file '{p}'\n")
    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
           "-c:a", "aac", "-b:a", "128k", out_path]
    _run(cmd)
    os.remove(lst)

def render_funnel(video_path: str, by_slot: Dict[str, List[Take]]) -> Tuple[str, List[Dict]]:
    """
    Renders: HOOK → PROBLEM → FEATURE → PROOF → CTA (any missing slot is skipped).
    Returns (final_path, clip_meta[])
    """
    order = []
    for slot in SLOT_ORDER:
        order.extend(by_slot.get(slot, []))

    # hard cap final duration
    cur = 0.0
    clips: List[Tuple[Take, float, float]] = []
    for t in order:
        dur = t.end - t.start
        if cur + dur > MAX_DURATION_SEC:
            # trim last
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
            "face_q": t.meta.get("face_q"), "scene_q": t.meta.get("scene_q"), "vtx_sim": t.meta.get("vtx_sim")
        })

    final_path = _tmp(".mp4")
    if tmp_parts:
        ffmpeg_concat(tmp_parts, final_path)
    else:
        # nothing selected – emit a 1-sec silent black as fallback
        final_path = _tmp(".mp4")
        _run([FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
              "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
              "-shortest",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
              "-c:a", "aac", "-b:a", "128k", final_path])

    for p in tmp_parts:
        try: os.remove(p)
        except: pass

    return final_path, clip_meta


# ---------- public job ----------
def job_render(s3_key: str, product_context: Optional[Dict]=None) -> Dict:
    """
    Main entry called by your RQ worker/web API.
    - downloads S3 object to /tmp
    - ASR → takes
    - visual sampling
    - semantic select
    - render & upload
    """
    assert S3_BUCKET, "S3_BUCKET env is required"

    # 1) download input (RunPod mounts usually read from pre-staged local file; here we assume S3)
    # If your caller puts the file locally, just pass the local path directly.
    local_in = _tmp(".mp4")

    # use AWS CLI-less download via presigned URL if your infra provides it; otherwise let the web tier upload file here.
    # For now we expect the pod has the file mounted or s3 gateway available – keeping simple:
    # -> If you already store the raw on S3, you can fetch with 'aws s3 cp' but AWS CLI isn’t installed in the image.
    # So assume caller placed the raw at /tmp and passed that path as s3_key starting with "/".
    if s3_key.startswith("/"):
        local_in = s3_key
    else:
        # try to stream via ffmpeg (S3 public) – if not public, consider presigned in your web tier
        raise RuntimeError("For this worker variant, pass a local file path for s3_key (e.g. /workspace/in/raw.mp4)")

    # 2) extract audio for ASR (whisper likes wav)
    wav_path = _tmp(".wav")
    _run([FFMPEG, "-y", "-i", local_in, "-vn", "-ac", "1", "-ar", "16000", wav_path])

    # 3) ASR
    segments = asr_transcribe(wav_path)
    print(f"[asr] segments: {len(segments)}")

    # 4) split to takes
    takes = segment_to_takes(local_in, segments)
    print(f"[seg] takes: {len(takes)}")

    # 5) attach visual scores
    attach_visual_scores(local_in, takes)

    # 6) semantic select by funnel slots
    by_slot = select_by_slots(takes)

    # 7) render
    out_path, clip_meta = render_funnel(local_in, by_slot)

    # 8) upload
    ts = int(time.time())
    out_key = f"renders/{uuid.uuid4().hex}_{ts}.mp4"
    upload_file(out_path, out_key, bucket=S3_BUCKET, region=AWS_REGION, content_type="video/mp4")

    # 9) response
    data = {
        "ok": True,
        "input_local": local_in,
        "duration_sec": probe_duration(out_path),
        "s3_key": out_key,
        "s3_url": f"s3://{S3_BUCKET}/{out_key}",
        "https_url": f"https://{S3_BUCKET}.s3.amazonaws.com/{out_key}",
        "clips": clip_meta,
        "slots": {k: [asdict(t) if hasattr(t, "__dataclass_fields__") else t.__dict__ for t in v] for k,v in by_slot.items()},
        "semantic": bool(_HAS_SEM),
        "vision": bool(_HAS_VISION),
        "asr": bool(ASR_ENABLED and _HAS_WHISPER),
    }
    try: os.remove(wav_path)
    except: pass
    try: os.remove(out_path)
    except: pass
    return data


# -------- tiny local test entry (optional) --------
if __name__ == "__main__":
    # Example local test:
    #   python3 jobs.py /path/to/local/input.mp4
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 jobs.py /abs/path/input.mp4")
        raise SystemExit(2)
    print(json.dumps(job_render(sys.argv[1]), indent=2))
