# jobs.py — EditDNA worker (ASR → Micro-cut → Visual/Semantic → Compose → Render → S3)
from __future__ import annotations
import os, json, shlex, subprocess, tempfile, uuid, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# --- infra helpers ---
from s3_utils import upload_file  # your existing helper

# --- optional deps (graceful fallbacks) ---
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

# semantic + visual pass (you added these files already)
try:
    from worker.semantic_visual_pass import (
        Take, dedup_takes, tag_slot, score_take, stitch_chain
    )
    _HAS_SEM = True
except Exception as e:
    print(f"[jobs] semantic_visual_pass unavailable ({e}); using no-op.")
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

try:
    from worker.sentence_boundary import micro_split_and_clean
    _HAS_SB = True
except Exception as e:
    print(f"[jobs] sentence_boundary unavailable ({e}); skipping micro-cuts.")
    _HAS_SB = False

# (Optional) lightweight visual sampler — safe defaults if missing
try:
    from worker.vision_sampler import sample_visuals
    _HAS_VISION = True
except Exception as e:
    print(f"[jobs] vision_sampler unavailable ({e}); vtx/face/scene = defaults.")
    _HAS_VISION = False

# -------- env / config --------
FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

MAX_TAKE_SEC = float(os.getenv("MAX_TAKE_SEC", "20"))
MIN_TAKE_SEC = float(os.getenv("MIN_TAKE_SEC", "1.5"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "120"))

ASR_ENABLED = int(os.getenv("ASR_ENABLED", "1"))
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANG = os.getenv("ASR_LANG", "en")

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

# slot order (BENEFITS is optional alias merged into PROBLEM)
SLOT_ORDER_CANON = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]

# ---------- small utilities ----------
def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    print("[ff] $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def _tmp(ext=""):
    return os.path.join(tempfile.gettempdir(), f"ed_{uuid.uuid4().hex}{ext}")

def _float(s: str) -> float:
    try: return float(s)
    except: return 0.0

def _parse_funnel_counts() -> Dict[str, Optional[int]]:
    """
    Accepts FUNNEL_COUNTS in either 5 or 6 comma form:
    - 5 numbers: HOOK, PROBLEM, FEATURE, PROOF, CTA
    - 6 numbers: HOOK, PROBLEM, BENEFITS, FEATURE, PROOF, CTA  (BENEFITS merged into PROBLEM)
    Use 0 for 'no limit'.
    """
    raw = (os.getenv("FUNNEL_COUNTS", "").strip() or "")
    if not raw:
        # Default: 1 of each slot
        return {"HOOK":1, "PROBLEM":1, "FEATURE":1, "PROOF":1, "CTA":1}
    parts = [p.strip() for p in raw.split(",") if p.strip()!=""]
    nums = []
    for p in parts:
        try:
            v = int(p)
            if v < 0: v = 0
            nums.append(v)
        except:
            nums.append(0)
    if len(nums) == 5:
        h, pr, fe, pf, cta = nums
        return {"HOOK":h or None, "PROBLEM":pr or None, "FEATURE":fe or None, "PROOF":pf or None, "CTA":cta or None}
    if len(nums) == 6:
        h, pr, bn, fe, pf, cta = nums
        # merge BENEFITS into PROBLEM count (sum; 0 means unlimited)
        if pr == 0 or bn == 0:
            pr_merged = None  # unlimited
        else:
            pr_merged = pr + bn
        return {"HOOK":(h or None), "PROBLEM":pr_merged, "FEATURE":(fe or None), "PROOF":(pf or None), "CTA":(cta or None)}
    # Fallback
    return {"HOOK":1, "PROBLEM":1, "FEATURE":1, "PROOF":1, "CTA":1}

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

# ---------- micro-cut (silence+punctuation → cleaner mini-takes) ----------
def apply_micro_cuts(video_path: str, takes: List[Take]) -> List[Take]:
    if not _HAS_SB:
        return takes
    micro: List[Take] = []
    for t in takes:
        spans = micro_split_and_clean(video_path, (t.start, t.end), t.text or "")
        if not spans:
            micro.append(t)
            continue
        for i, (a,b,tx) in enumerate(spans, 1):
            if (b - a) < MIN_TAKE_SEC:
                continue
            micro.append(Take(
                id=f"{t.id}s{i:02d}",
                start=a, end=b, text=tx,
                face_q=t.face_q, scene_q=t.scene_q, vtx_sim=t.vtx_sim,
                has_product=t.has_product, ocr_hit=t.ocr_hit,
                meta=dict(t.meta or {})
            ))
    print(f"[micro] input_takes={len(takes)} → micro_takes={len(micro)}")
    return micro or takes

# ---------- visual sampling (optional) ----------
def attach_visual_scores(video_path: str, takes: List[Take]) -> None:
    for t in takes:
        if _HAS_VISION:
            try:
                face_q, scene_q, vtx_sim, had_signal = sample_visuals(video_path, (t.start, t.end), text=t.text)
            except Exception:
                face_q, scene_q, vtx_sim, had_signal = 1.0, 1.0, 0.0, False
        else:
            face_q, scene_q, vtx_sim, had_signal = 1.0, 1.0, 0.0, False
        t.face_q = float(face_q)
        t.scene_q = float(scene_q)
        t.vtx_sim = float(vtx_sim)
        t.meta = t.meta or {}
        t.meta["has_signal"] = bool(had_signal)

# ---------- rendering ----------
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

def render_funnel(video_path: str, timeline: List[Take]) -> Tuple[str, List[Dict]]:
    # cut & concat
    parts = []
    clip_meta = []
    cur = 0.0
    for i, t in enumerate(timeline, 1):
        seg = min(t.end - t.start, max(0.0, MAX_DURATION_SEC - cur))
        if seg <= 0: break
        outp = _tmp(f".part{i:02d}.mp4")
        ffmpeg_subclip(video_path, outp, t.start, t.start + seg)
        parts.append(outp)
        clip_meta.append({
            "id": t.id, "slot": t.meta.get("slot"),
            "start": t.start, "end": t.start + seg,
            "score": t.meta.get("score"),
            "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
            "chain_ids": t.meta.get("chain_ids", [])
        })
        cur += seg
        if cur >= MAX_DURATION_SEC: break

    final_path = _tmp(".mp4")
    if parts:
        ffmpeg_concat(parts, final_path)
    else:
        # 1s black safety
        _run([
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            final_path
        ])
    for p in parts:
        try: os.remove(p)
        except: pass
    return final_path, clip_meta

# ---------- funnel compose ----------
def compose_funnel(takes: List[Take]) -> Tuple[List[Take], Dict[str, List[Take]]]:
    # dedup + score + per-slot buckets
    kept = dedup_takes(takes)
    by_slot: Dict[str, List[Take]] = {k: [] for k in SLOT_ORDER_CANON}
    for t in kept:
        slot = tag_slot(t)
        t.meta["slot"] = slot
        t.meta["score"] = score_take(t, slot)
        if t.meta["score"] >= 0:
            by_slot[slot].append(t)

    # stitch long demos for PROOF/FEATURE
    by_slot["PROOF"]   = stitch_chain(by_slot["PROOF"])
    by_slot["FEATURE"] = stitch_chain(by_slot["FEATURE"])

    # pick best per counts (0 = unlimited)
    counts = _parse_funnel_counts()  # returns None for unlimited
    def pick_best(lst: List[Take], limit: Optional[int]) -> List[Take]:
        if not lst: return []
        ranked = sorted(lst, key=lambda x: x.meta.get("score",0.0), reverse=True)
        return ranked if limit is None else ranked[:limit]

    selected: List[Take] = []
    for slot in ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]:
        want = counts.get(slot, 1)
        chosen = pick_best(by_slot[slot], want)
        selected.extend(chosen)

    # respect total MAX_DURATION_SEC by trimming at render stage
    # keep timeline order by SLOT_ORDER then by original start
    selected = sorted(selected, key=lambda t: (SLOT_ORDER_CANON.index(t.meta.get("slot","FEATURE")), t.start))
    return selected, by_slot

# ---------- public job ----------
def job_render(local_in: str, product_context: Optional[Dict]=None) -> Dict:
    assert S3_BUCKET, "S3_BUCKET env is required"

    # 1) extract audio for ASR (whisper likes wav)
    wav_path = _tmp(".wav")
    _run([FFMPEG, "-y", "-i", local_in, "-vn", "-ac", "1", "-ar", "16000", wav_path])

    # 2) ASR
    segments = asr_transcribe(wav_path)
    print(f"[asr] segments: {len(segments)}")

    # 3) split to takes (20s windows)
    takes = segment_to_takes(local_in, segments)
    print(f"[seg] takes: {len(takes)}")

    # 4) micro-cut pass (sentence-boundary + silence align)
    takes = apply_micro_cuts(local_in, takes)

    # 5) attach visual scores (face/scene/vtx) — light/optional
    attach_visual_scores(local_in, takes)

    # 6) semantic selection + stitching + compose funnel
    timeline, by_slot = compose_funnel(takes)

    # 7) render
    out_path, clip_meta = render_funnel(local_in, timeline)

    # 8) upload
    ts = int(time.time())
    out_key = f"editdna/outputs/{uuid.uuid4().hex}_{ts}.mp4"
    upload_file(out_path, out_key, bucket=S3_BUCKET, region=AWS_REGION, content_type="video/mp4")

    data = {
        "ok": True,
        "input_local": local_in,
        "duration_sec": probe_duration(out_path),
        "s3_key": out_key,
        "s3_url": f"s3://{S3_BUCKET}/{out_key}",
        "https_url": f"https://{S3_BUCKET}.s3.amazonaws.com/{out_key}",
        "clips": clip_meta,
        "slots": {
            k: [
                {
                    "id": t.id, "start": t.start, "end": t.end, "text": t.text,
                    "meta": t.meta, "face_q": t.face_q, "scene_q": t.scene_q,
                    "vtx_sim": t.vtx_sim, "has_product": t.has_product, "ocr_hit": t.ocr_hit,
                }
                for t in sorted(v, key=lambda x: x.start)
            ]
            for k, v in by_slot.items()
        },
        "semantic": bool(_HAS_SEM),
        "vision": bool(_HAS_VISION),
        "asr": bool(ASR_ENABLED and _HAS_WHISPER),
    }
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
