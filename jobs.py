# jobs.py — EditDNA single-funnel composer + render (FFmpeg + S3)
from __future__ import annotations
import os, json, shlex, subprocess, tempfile, uuid, time, logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

# --- logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("editdna.jobs")

# --- optional deps + helpers ---
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

# semantic + visual fusion
try:
    from worker.semantic_visual_pass import (
        Take as SVTake,
        dedup_takes,
        tag_slot,
        score_take,
        stitch_chain
    )
    _HAS_SEM = True
except Exception as e:
    log.warning(f"[jobs] semantic_visual_pass unavailable: {e}")
    _HAS_SEM = False

# sentence-boundary (micro-cut) — optional
try:
    from worker.sentence_boundary import micro_split_and_clean
    _HAS_SENT = True
except Exception as e:
    log.warning(f"[jobs] sentence_boundary unavailable: {e}")
    _HAS_SENT = False

# S3 helpers
try:
    from s3_utils import upload_file, presigned_url
    _HAS_S3UTIL = True
except Exception as e:
    log.warning(f"[jobs] s3_utils not found ({e}), falling back to boto3")
    _HAS_S3UTIL = False
    import boto3
    AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
    _s3 = boto3.client("s3", region_name=AWS_REGION)

# -------- env / config --------
FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

MAX_TAKE_SEC      = float(os.getenv("MAX_TAKE_SEC", "20"))
MAX_DURATION_SEC  = float(os.getenv("MAX_DURATION_SEC", "120"))
ASR_ENABLED       = int(os.getenv("ASR_ENABLED", "1"))
ASR_MODEL_SIZE    = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANG          = os.getenv("ASR_LANG", "en")

# Funnel counts: HOOK, PROBLEM, BENEFITS, FEATURE, PROOF, CTA
# 0 = unlimited, we merge BENEFITS into PROBLEM internally
_FUNNEL_RAW = os.getenv("FUNNEL_COUNTS", "1,0,0,0,1,1").split(",")
while len(_FUNNEL_RAW) < 6: _FUNNEL_RAW.append("0")
F_HOOK, F_PROBLEM, F_BENEFITS, F_FEATURE, F_PROOF, F_CTA = [int(x or 0) for x in _FUNNEL_RAW]

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
OUT_PREFIX = os.getenv("S3_PREFIX", "editdna/outputs").rstrip("/")

# ---------- small utilities ----------
def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    log.info("[ff] $ " + " ".join(shlex.quote(c) for c in cmd))
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
        log.warning("[ffprobe] duration failed: %s", e)
        return 0.0

# ---------- ASR ----------
def asr_transcribe(audio_path: str) -> List[Dict]:
    if not ASR_ENABLED or not _HAS_WHISPER:
        dur = probe_duration(audio_path)
        return [{"start": 0.0, "end": dur, "text": ""}]
    log.info(f"[asr] loading whisper model: {ASR_MODEL_SIZE}")
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
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0
    meta: Dict = None

def segment_to_takes(video_path: str, segments: List[Dict]) -> List[Take]:
    takes: List[Take] = []
    for seg in segments:
        s, e, txt = float(seg["start"]), float(seg["end"]), seg.get("text","").strip()
        cur = s
        while cur < e:
            nxt = min(e, cur + MAX_TAKE_SEC)
            if (nxt - cur) >= 1.0:
                takes.append(Take(
                    id=f"T{len(takes)+1:04d}",
                    start=cur, end=nxt, text=txt, meta={}
                ))
            cur = nxt
    return takes

# ---------- micro-cut sentence pass (optional) ----------
def apply_micro_sentence_pass(video_path: str, takes: List[Take]) -> List[Take]:
    """Split each take into cleaner sentence spans and drop retries/fillers; then rejoin adjacent good ones."""
    if not _HAS_SENT:
        return takes
    refined: List[Take] = []
    for t in takes:
        spans = micro_split_and_clean(video_path, (t.start, t.end), t.text)  # returns list[(start,end,text)]
        for (ss, ee, tx) in spans:
            refined.append(Take(
                id=f"{t.id}_s{len(refined)+1}", start=ss, end=ee, text=tx,
                face_q=t.face_q, scene_q=t.scene_q, vtx_sim=t.vtx_sim,
                has_product=t.has_product, ocr_hit=t.ocr_hit, meta=t.meta or {}
            ))
    return refined or takes

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

# ---------- composer ----------
def select_by_funnel(sv_takes: List[SVTake]) -> Dict[str, List[SVTake]]:
    """
    Builds slot buckets using semantic tagging/scoring + counts from env.
    BENEFITS are merged into PROBLEM internally.
    """
    if not _HAS_SEM:
        # Fallback: everything becomes FEATURE
        return {"HOOK":[], "PROBLEM":[], "FEATURE":[*sv_takes], "PROOF":[], "CTA":[]}

    # Tag + score
    buckets: Dict[str, List[SVTake]] = {"HOOK":[], "PROBLEM":[], "FEATURE":[], "PROOF":[], "CTA":[]}
    for t in sv_takes:
        slot = tag_slot(t)
        t.meta["slot"] = slot
        t.meta["score"] = score_take(t, slot)
        if t.meta["score"] >= 0:
            # Map BENEFITS -> PROBLEM if someone used that label upstream
            if slot == "BENEFITS": slot = "PROBLEM"
            buckets[slot].append(t)

    # Stitch longer demos where it matters
    buckets["FEATURE"] = stitch_chain(buckets["FEATURE"])
    buckets["PROOF"]   = stitch_chain(buckets["PROOF"])

    # Apply counts (0 = unlimited)
    def pick(lst: List[SVTake], k: int) -> List[SVTake]:
        if k == 0:  # unlimited
            return sorted(lst, key=lambda x: x.meta.get("score",0.0), reverse=True)
        return sorted(lst, key=lambda x: x.meta.get("score",0.0), reverse=True)[:k]

    # counts (BENEFITS merged into PROBLEM)
    pick_hook    = pick(buckets["HOOK"],    F_HOOK)
    pick_problem = pick(buckets["PROBLEM"], F_PROBLEM if F_PROBLEM>0 else 0)  # unlimited if 0
    pick_feature = pick(buckets["FEATURE"], F_FEATURE if F_FEATURE>0 else 0)
    pick_proof   = pick(buckets["PROOF"],   F_PROOF)
    pick_cta     = pick(buckets["CTA"],     F_CTA)

    ordered = [*pick_hook, *pick_problem, *pick_feature, *pick_proof, *pick_cta]
    return {
        "HOOK": pick_hook,
        "PROBLEM": pick_problem,
        "FEATURE": pick_feature,
        "PROOF": pick_proof,
        "CTA": pick_cta,
        "_ordered": ordered
    }

def lay_out_timeline(candidates: List[SVTake], max_secs: float) -> List[Tuple[float,float,SVTake]]:
    """Trim to max runtime, preserving order. Returns list of (start,end,take)."""
    timeline: List[Tuple[float,float,SVTake]] = []
    cur = 0.0
    for t in candidates:
        dur = t.end - t.start
        if cur + dur > max_secs:
            remain = max(0.0, max_secs - cur)
            if remain >= 0.8:
                timeline.append((t.start, t.start+remain, t))
                cur = max_secs
            break
        else:
            timeline.append((t.start, t.end, t))
            cur += dur
        if cur >= max_secs: break
    return timeline

# ---------- public job ----------
def job_render(local_in: str) -> Dict:
    """
    Core job: ASR -> (optional sentence pass) -> semantic+visual select -> compose -> render -> S3
    local_in must be a local path (tasks.py already downloads URLs)
    """
    assert S3_BUCKET, "S3_BUCKET env is required"

    # 1) extract audio for ASR (whisper likes wav)
    wav_path = _tmp(".wav")
    _run([FFMPEG, "-y", "-i", local_in, "-vn", "-ac", "1", "-ar", "16000", wav_path])

    # 2) ASR
    segments = asr_transcribe(wav_path)
    log.info(f"[asr] segments: {len(segments)}")

    # 3) initial takes (bounded by MAX_TAKE_SEC)
    takes = segment_to_takes(local_in, segments)
    log.info(f"[seg] takes: {len(takes)} before micro-sentences")

    # 4) micro-cut sentence pass
    takes = apply_micro_sentence_pass(local_in, takes)
    log.info(f"[seg] takes: {len(takes)} after micro-sentences")

    # 5) adapt to semantic-visual layer
    if _HAS_SEM:
        sv_takes: List[SVTake] = [
            SVTake(
                id=t.id, start=t.start, end=t.end, text=t.text or "",
                face_q=t.face_q, scene_q=t.scene_q, vtx_sim=t.vtx_sim,
                has_product=t.has_product, ocr_hit=t.ocr_hit, meta=t.meta or {}
            )
            for t in takes
        ]
        sv_takes = dedup_takes(sv_takes)
        by_slot = select_by_funnel(sv_takes)
        ordered = by_slot["_ordered"]
    else:
        # fallback: dump first few takes up to cap
        ordered = []
        for t in takes:
            ordered.append(SVTake(
                id=t.id, start=t.start, end=t.end, text=t.text or "",
                face_q=t.face_q, scene_q=t.scene_q, vtx_sim=t.vtx_sim,
                has_product=t.has_product, ocr_hit=t.ocr_hit, meta=t.meta or {}
            ))

    # 6) build final timeline under MAX_DURATION_SEC
    timeline = lay_out_timeline(ordered, MAX_DURATION_SEC)
    log.info(f"[compose] timeline clips: {len(timeline)} (cap={MAX_DURATION_SEC}s)")

    # 7) render parts + concat
    tmp_parts = []
    clip_meta = []
    for i, (ss, ee, t) in enumerate(timeline, 1):
        part = _tmp(f".part{i:02d}.mp4")
        ffmpeg_subclip(local_in, part, ss, ee)
        tmp_parts.append(part)
        clip_meta.append({
            "id": t.id, "slot": t.meta.get("slot"),
            "start": ss, "end": ee,
            "score": t.meta.get("score"),
            "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
            "chain_ids": t.meta.get("chain_ids", [])
        })

    final_path = _tmp(".mp4")
    if tmp_parts:
        ffmpeg_concat(tmp_parts, final_path)
    else:
        # always produce something
        _run([
            FFMPEG, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            final_path
        ])

    # 8) upload
    ts = int(time.time())
    out_key = f"{OUT_PREFIX}/{uuid.uuid4().hex}_{ts}.mp4"

    if _HAS_S3UTIL:
        upload_file(final_path, out_key, bucket=S3_BUCKET, region=AWS_REGION, content_type="video/mp4")
    else:
        with open(final_path, "rb") as f:
            _s3.upload_fileobj(f, S3_BUCKET, out_key, ExtraArgs={"ContentType":"video/mp4"})

    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{out_key}"
    log.info(f"[jobs] uploaded final -> {https_url}")

    # 9) clean
    for p in tmp_parts:
        try: os.remove(p)
        except: pass
    try: os.remove(final_path)
    except: pass
    try: os.remove(wav_path)
    except: pass

    # 10) result
    result = {
        "ok": True,
        "input_local": local_in,
        "duration_sec": probe_duration(final_path) or sum((e-s) for s,e,_ in timeline),
        "s3_key": out_key,
        "s3_url": f"s3://{S3_BUCKET}/{out_key}",
        "https_url": https_url,
        "clips": clip_meta,
        "slots": {
            # best-effort surface (when semantic present)
            "HOOK":  [c for c in clip_meta if c.get("slot")=="HOOK"],
            "PROBLEM":[c for c in clip_meta if c.get("slot")=="PROBLEM"],
            "FEATURE":[c for c in clip_meta if c.get("slot")=="FEATURE"],
            "PROOF": [c for c in clip_meta if c.get("slot")=="PROOF"],
            "CTA":   [c for c in clip_meta if c.get("slot")=="CTA"],
        },
        "semantic": bool(_HAS_SEM),
        "vision": False,   # set True if you later wire vtx/scene scores here
        "asr": bool(ASR_ENABLED and _HAS_WHISPER),
    }
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 jobs.py /abs/path/input.mp4")
        raise SystemExit(2)
    print(json.dumps(job_render(sys.argv[1]), indent=2))
