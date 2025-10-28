# /workspace/editdna/jobs.py
from __future__ import annotations
import os, sys, json, time, math, tempfile, subprocess, shlex
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Ensure our package root is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- Optional helpers from your repo (safe imports) ----
# ASR (Whisper)
_ASR_ENABLED = bool(int(os.getenv("ASR_ENABLED", "1")))
_ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "tiny")
_ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en")
_ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")

# Semantics module (you uploaded it)
from semantic_visual_pass import (
    Take, dedup_takes, tag_slot, score_take, stitch_chain
)

# S3 utils (your helper if present), else fallback boto3
_S3_BUCKET   = os.getenv("S3_BUCKET")
_S3_PREFIX   = os.getenv("S3_PREFIX", "editdna/outputs")
_AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")
_S3_ACL      = os.getenv("S3_ACL", "public-read")
_PRESIGN_SEC = int(os.getenv("PRESIGN_EXPIRES", "0") or "0")

try:
    import boto3
except Exception:
    boto3 = None

try:
    from s3_utils import s3_upload_public  # your helper: (local_path, key) -> (s3_uri, https_url)
except Exception:
    s3_upload_public = None

# ffmpeg/ffprobe
FFMPEG_BIN   = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN  = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

# Time / merging / output controls
BIN_SEC          = float(os.getenv("BIN_SEC", "1.0"))
MIN_TAKE_SEC     = float(os.getenv("MIN_TAKE_SEC", "2.0"))
MAX_TAKE_SEC     = float(os.getenv("MAX_TAKE_SEC", "220"))
MAX_DURATION_SEC = float(os.getenv("MAX_DURATION_SEC", "220"))
MERGE_MAX_CHAIN  = int(os.getenv("MERGE_MAX_CHAIN", "10"))
FALLBACK_MIN_SEC = float(os.getenv("FALLBACK_MIN_SEC", "60"))

# Funnel target counts: "HOOK,PROBLEM,FEATURE,PROOF,CTA"
# NOTE: zeros = unlimited/optional. If all zeros ⇒ concatenates best takes up to caps.
_FUNNEL_COUNTS_RAW = os.getenv("FUNNEL_COUNTS", "0,0,0,0,0").strip()
def parse_counts(s: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    parts = [p.strip() for p in (s or "").split(",")]
    # pad to 5
    while len(parts) < 5: parts.append("0")
    def norm(x: str):
        try:
            v = int(x)
            return None if v <= 0 else v
        except:  # noqa
            return None
    return tuple(norm(p) for p in parts[:5])  # type: ignore
FUNNEL_COUNTS = parse_counts(_FUNNEL_COUNTS_RAW)

# Veto low-quality takes?
VETO_MIN_SCORE = float(os.getenv("VETO_MIN_SCORE", "0.35"))

# Optional “micro cut” using silence hints (we won’t need external deps; simple gap-based)
MICRO_CUT = bool(int(os.getenv("MICRO_CUT", "1")))
MICRO_SILENCE_MIN = float(os.getenv("MICRO_SILENCE_MIN", "0.25"))

# Slot constraints
SLOT_LABELS = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]

# ------------------- Utilities -------------------
def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def _probe_duration(path: str) -> float:
    try:
        out = _run([FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nokey=1:noprint_wrappers=1", path])
        return float(out.stdout.decode().strip())
    except Exception:
        return 0.0

def _ffmpeg_subclip(src: str, dst: str, ss: float, ee: float):
    dur = max(0.0, ee - ss)
    if dur <= 0.01:
        raise RuntimeError("subclip zero/negative duration")
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{ss:.3f}",
        "-i", src,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        dst
    ]
    _run(cmd)

def _ffmpeg_concat(parts: List[str], out_path: str):
    if not parts:
        raise RuntimeError("No parts to concat")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
        list_path = f.name
    cmd = [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-g", "48",
           "-c:a", "aac", "-b:a", "128k", out_path]
    _run(cmd)

def _s3_upload(local_path: str, session_id: str) -> Tuple[str, str, str]:
    """
    Returns (s3_key, s3_uri, https_url)
    """
    key = f"{_S3_PREFIX.rstrip('/')}/{session_id}_{int(time.time())}.mp4"
    if s3_upload_public:
        s3_uri, https_url = s3_upload_public(local_path, key)  # your helper
        return key, s3_uri, https_url
    if not boto3 or not _S3_BUCKET:
        # Local-only fallback
        return key, f"s3://{_S3_BUCKET}/{key}" if _S3_BUCKET else key, ""
    s3 = boto3.client("s3", region_name=_AWS_REGION)
    extra = {"ACL": _S3_ACL} if _S3_ACL else {}
    s3.upload_file(local_path, _S3_BUCKET, key, ExtraArgs=extra)
    https_url = f"https://{_S3_BUCKET}.s3.{_AWS_REGION}.amazonaws.com/{key}"
    if _PRESIGN_SEC and _PRESIGN_SEC > 0:
        try:
            https_url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": _S3_BUCKET, "Key": key},
                ExpiresIn=_PRESIGN_SEC
            )
        except Exception:
            pass
    return key, f"s3://{_S3_BUCKET}/{key}", https_url

# ------------------- ASR -------------------
def _asr_segments(path: str) -> List[Dict[str, Any]]:
    """
    Use OpenAI Whisper (via openai-whisper package) to generate segments.
    Returns list of {start, end, text}.
    """
    if not _ASR_ENABLED:
        # Single "segment" spanning full video if ASR is off
        dur = _probe_duration(path)
        return [{"start": 0.0, "end": dur, "text": ""}]

    import whisper
    model = whisper.load_model(_ASR_MODEL_SIZE, device=_ASR_DEVICE)
    result = model.transcribe(path, language=_ASR_LANGUAGE, verbose=False)
    segs = []
    for s in result.get("segments", []):
        segs.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s.get("text", "").strip()
        })
    return segs

# ------------------- Takes from segments -------------------
def _segments_to_takes(segs: List[Dict[str, Any]]) -> List[Take]:
    takes: List[Take] = []
    for i, s in enumerate(segs, 1):
        st, en = float(s["start"]), float(s["end"])
        if en - st <= 0: 
            continue
        takes.append(Take(
            id=f"T{i:04d}",
            start=st, end=en,
            text=s.get("text",""),
            face_q=1.0, scene_q=1.0, vtx_sim=0.0,
            has_product=False, ocr_hit=0
        ))
    return takes

def _enforce_minmax(takes: List[Take]) -> List[Take]:
    """
    Merge tiny neighboring takes until >= MIN_TAKE_SEC; split very long takes (rare).
    We keep it simple: only merge forward; no splitting for now.
    """
    if not takes: return []
    takes = sorted(takes, key=lambda t: (t.start, t.end))
    merged: List[Take] = []
    cur = takes[0]
    for nxt in takes[1:]:
        # If current is too short, extend into next
        if (cur.end - cur.start) < MIN_TAKE_SEC:
            cur.end = nxt.end
            cur.text = (cur.text + " " + nxt.text).strip()
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    # Hard-cap takes at MAX_TAKE_SEC (optional)
    out: List[Take] = []
    for t in merged:
        if (t.end - t.start) > MAX_TAKE_SEC:
            # keep the first MAX_TAKE_SEC
            t.end = t.start + MAX_TAKE_SEC
        out.append(t)
    return out

# ------------------- Slotting & scoring -------------------
def _slot_and_score(takes: List[Take]) -> Dict[str, List[Take]]:
    slotted: Dict[str, List[Take]] = {k: [] for k in SLOT_LABELS}
    for t in takes:
        slot = tag_slot(t, None)
        t.slot_hint = slot
        t.meta["slot"] = slot
        t.meta["score"] = score_take(t, slot)
        # veto
        if t.meta["score"] < VETO_MIN_SCORE:
            continue
        slotted[slot].append(t)
    # Within each slot, keep semantic-deduped, stitched chains (merge continuity)
    for k in SLOT_LABELS:
        dedup = dedup_takes(slotted[k])
        stitched = stitch_chain(dedup)
        # Trim per-chain length to MAX_TAKE_SEC just in case
        trimmed = []
        for x in stitched:
            if (x.end - x.start) > MAX_TAKE_SEC:
                x.end = x.start + MAX_TAKE_SEC
            trimmed.append(x)
        # Sort best-first by score, then earlier in video
        slotted[k] = sorted(trimmed, key=lambda t: (-float(t.meta.get("score",0)), t.start))
    return slotted

# ------------------- Funnel selection -------------------
def _select_funnel(slotted: Dict[str, List[Take]],
                   counts: Tuple[Optional[int],Optional[int],Optional[int],Optional[int],Optional[int]],
                   max_total_sec: float) -> List[Take]:
    """
    If any count is None => unlimited/optional for that slot.
    If all are None => we’re in “unconstrained” mode: concatenate best takes until cap.
    """
    if all(c is None for c in counts):
        # unconstrained: pour best takes in “narrative” order = HOOK→PROBLEM→FEATURE→PROOF→CTA loop
        order = []
        # flatten by slot priority
        for slot in SLOT_LABELS:
            order += slotted[slot]
        out, total = [], 0.0
        for t in order:
            dur = t.end - t.start
            if total + dur > max_total_sec: break
            out.append(t); total += dur
        return out

    # Constrained funnel: pick up to count for each slot, in order
    out: List[Take] = []
    total = 0.0
    for slot, want in zip(SLOT_LABELS, counts):
        picks = slotted[slot]
        if not picks:
            continue
        if want is None:
            want = len(picks)  # unlimited ⇒ all (until cap)
        for t in picks[:max(0, int(want))]:
            dur = t.end - t.start
            if total + dur > max_total_sec:
                return out
            out.append(t)
            total += dur
    return out

# ------------------- Render -------------------
def _render_parts(src_path: str, chosen: List[Take]) -> Tuple[str, float, List[Dict[str, Any]]]:
    parts = []
    for i, t in enumerate(chosen, 1):
        outp = os.path.join(tempfile.gettempdir(), f"ed_{os.urandom(8).hex()}.part{i:02d}.mp4")
        _ffmpeg_subclip(src_path, outp, t.start, t.end)
        parts.append(outp)
    final_path = os.path.join(tempfile.gettempdir(), f"ed_{os.urandom(8).hex()}.mp4")
    _ffmpeg_concat(parts, final_path)
    dur = _probe_duration(final_path)
    # Return minimal clip meta for API
    clips = [{
        "id": t.id, "slot": t.slot_hint, "start": t.start, "end": t.end,
        "score": float(t.meta.get("score", 0.0)),
        "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
        "chain_ids": t.meta.get("chain_ids", [])
    } for t in chosen]
    return final_path, dur, clips

# ------------------- Public API -------------------
def run_pipeline(local_path: str | None = None, payload: dict | None = None) -> dict:
    """
    Main pipeline:
      1) ASR (optional) → segments
      2) Segment → takes → normalize min/max
      3) Slot + sem dedup + stitch
      4) Select funnel per FUNNEL_COUNTS & MAX_DURATION_SEC
      5) ffmpeg subclips → concat
      6) Upload to S3
    """
    t0 = time.time()
    payload = payload or {}
    session_id = payload.get("session_id", "session")
    max_out = float(payload.get("options", {}).get("MAX_DURATION_SEC", MAX_DURATION_SEC))

    if not local_path:
        # If a remote file was passed in payload.files[0], fetch it
        files = payload.get("files") or []
        if not files:
            return {"ok": False, "error": "No input file"}
        # download to temp
        import urllib.request
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(files[0])[1] or ".mp4")
        with urllib.request.urlopen(files[0]) as r, open(tmp.name, "wb") as w:
            w.write(r.read())
        local_path = tmp.name

    # 1) ASR
    segs = _asr_segments(local_path)

    # 2) Make takes & normalize lengths
    takes = _segments_to_takes(segs)
    takes = _enforce_minmax(takes)

    # 3) Slotting & stitching
    slotted = _slot_and_score(takes)

    # Make slots JSON (for response)
    slots_json: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SLOT_LABELS}
    for k in SLOT_LABELS:
        for t in slotted[k]:
            slots_json[k].append({
                "id": t.id, "start": t.start, "end": t.end, "text": t.text,
                "meta": {"slot": k, "score": float(t.meta.get("score", 0.0))},
                "face_q": t.face_q, "scene_q": t.scene_q, "vtx_sim": t.vtx_sim,
                "has_product": t.has_product, "ocr_hit": t.ocr_hit
            })

    # 4) Select funnel
    chosen = _select_funnel(slotted, FUNNEL_COUNTS, max_out)

    # Fallback: if nothing selected and we want at least some output, pour best takes until >= FALLBACK_MIN_SEC or hit cap
    if not chosen:
        # Flatten everything best-first
        ordered = []
        for slot in SLOT_LABELS:
            ordered += slotted[slot]
        total = 0.0
        for t in ordered:
            dur = t.end - t.start
            if dur <= 0: continue
            if total + dur > max_out: break
            chosen.append(t); total += dur
            if total >= max(FALLBACK_MIN_SEC, 0.0): break

    if not chosen:
        # Still nothing → fail cleanly
        return {
            "ok": False,
            "error": "No suitable segments found after veto/dedup.",
            "slots": slots_json,
            "semantic": True if _ASR_ENABLED else False,
            "asr": _ASR_ENABLED,
            "vision": False
        }

    # 5) Render
    final_path, out_dur, clip_meta = _render_parts(local_path, chosen)

    # 6) Upload
    s3_key, s3_uri, https_url = _s3_upload(final_path, session_id)

    resp = {
        "ok": True,
        "input_local": local_path,
        "duration_sec": round(out_dur, 3),
        "s3_key": s3_key,
        "s3_url": s3_uri,
        "https_url": https_url,
        "clips": clip_meta,
        "slots": slots_json,
        "semantic": True,   # semantic pipeline used
        "vision": False,
        "asr": _ASR_ENABLED
    }
    return resp

def job_render(local_path: str) -> dict:
    # Legacy wrapper
    return run_pipeline(local_path=local_path, payload=None)
