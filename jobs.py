# /workspace/editdna/jobs.py
from __future__ import annotations
import os, json, re, math, subprocess, tempfile, uuid, time, shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------- Env ----------
FFMPEG = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

BIN_SEC            = float(os.getenv("BIN_SEC", "1.0"))
MIN_TAKE_SEC       = float(os.getenv("MIN_TAKE_SEC", "2.0"))
MAX_TAKE_SEC       = float(os.getenv("MAX_TAKE_SEC", "60"))
MAX_DURATION_SEC   = float(os.getenv("MAX_DURATION_SEC", "120"))
MERGE_MAX_CHAIN    = int(os.getenv("MERGE_MAX_CHAIN", "10"))
SEMANTICS_ENABLED  = int(os.getenv("SEMANTICS_ENABLED", "1")) == 1
FALLBACK_MIN_SEC   = float(os.getenv("FALLBACK_MIN_SEC", "15"))

# funnel counts: HOOK,PROBLEM,FEATURE,PROOF,CTA,(optional BENEFITS merged with PROBLEM)
# e.g. "1,0,0,0,0,1" => 1 HOOK, 0 PROBLEM, 0 FEATURE, 0 PROOF, 0 CTA, 1 BENEFITS (merged into PROBLEM)
FUNNEL_COUNTS_ENV  = os.getenv("FUNNEL_COUNTS", "0,0,0,0,0")

S3_BUCKET          = os.getenv("S3_BUCKET")
S3_PREFIX          = os.getenv("S3_PREFIX", "editdna/outputs")
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
S3_ACL             = os.getenv("S3_ACL", "public-read")
PRESIGN_EXPIRES    = int(os.getenv("PRESIGN_EXPIRES", "86400"))

ASR_ENABLED        = int(os.getenv("ASR_ENABLED", "1")) == 1
ASR_MODEL_SIZE     = os.getenv("ASR_MODEL_SIZE", "tiny")
ASR_LANGUAGE       = os.getenv("ASR_LANGUAGE", "en")
ASR_DEVICE         = os.getenv("ASR_DEVICE", "cuda")

# weights for semantic scoring (used by semantic_visual_pass)
W_SEM  = float(os.getenv("W_SEM",  "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE= float(os.getenv("W_SCENE","0.5"))
W_VTX  = float(os.getenv("W_VTX",  "0.8"))
W_PROD = float(os.getenv("W_PROD", "0.0"))
W_OCR  = float(os.getenv("W_OCR",  "0.0"))

# ---------- Simple helpers ----------
def _run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    # Debug-friendly run
    if cmd and os.path.basename(cmd[0]) == "ffmpeg":
        print(f"[ff] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def _ffprobe_duration(path: str) -> float:
    try:
        r = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=nokey=1:noprint_wrappers=1", path], check=True)
        return float(r.stdout.decode().strip())
    except Exception:
        return 0.0

def _safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def _tmpfile(suffix=".mp4") -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _parse_funnel_counts(s: str) -> Tuple[int,int,int,int,int,int]:
    """HOOK,PROBLEM,FEATURE,PROOF,CTA,BENEFITS(optional)"""
    parts = [p.strip() for p in s.split(",")]
    parts = [int(p) if p else 0 for p in parts]
    while len(parts) < 6:
        parts.append(0)
    return tuple(parts[:6])  # type: ignore

# ---------- ASR ----------
def _asr_segments(path: str) -> List[Dict[str, Any]]:
    """
    Returns segments: [{start: float, end: float, text: str}, ...]
    """
    if not ASR_ENABLED:
        # No ASR => single segment whole file
        dur = _ffprobe_duration(path)
        return [{"start": 0.0, "end": dur, "text": ""}]

    try:
        import whisper
        model = whisper.load_model(ASR_MODEL_SIZE, device=ASR_DEVICE)
        print(f"[asr] using whisper model={ASR_MODEL_SIZE} device={ASR_DEVICE}", flush=True)
        res = model.transcribe(path, language=ASR_LANGUAGE)
        segs = []
        for s in res.get("segments", []):
            segs.append({"start": float(s.get("start", 0.0)),
                         "end":   float(s.get("end", 0.0)),
                         "text":  s.get("text","").strip()})
        print(f"[asr] segments: {len(segs)}", flush=True)
        return segs or [{"start": 0.0, "end": _ffprobe_duration(path), "text": ""}]
    except Exception as e:
        print("[asr] error:", repr(e), flush=True)
        dur = _ffprobe_duration(path)
        return [{"start": 0.0, "end": dur, "text": ""}]

# ---------- Sentence & micro cuts ----------
_SENT_END = re.compile(r'([.!?])\s+')

def _split_sentences(text: str) -> List[str]:
    if not text: return []
    parts = []
    last = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        parts.append(text[last:end].strip())
        last = end
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]

def _sent_takes(segs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Cut by sentence boundaries but keep time bins.
    """
    takes: List[Dict[str,Any]] = []
    idx = 1
    for s in segs:
        chunk_sents = _split_sentences(s["text"])
        if not chunk_sents:
            # make a bin-sized take
            t0 = s["start"]
            while t0 < s["end"]:
                t1 = min(s["end"], t0 + BIN_SEC)
                if t1 - t0 >= MIN_TAKE_SEC or (t1 - t0) >= 0.5:
                    takes.append({"id": f"T{idx:04d}", "start": t0, "end": t1, "text": ""})
                    idx += 1
                t0 = t1
            continue

        # naive even split of the time interval across the sentences
        total = max(0.001, s["end"] - s["start"])
        per = total / len(chunk_sents)
        t0 = s["start"]
        for sent in chunk_sents:
            t1 = min(s["end"], t0 + per)
            takes.append({"id": f"T{idx:04d}", "start": t0, "end": t1, "text": sent})
            idx += 1
            t0 = t1
    print(f"[seg] takes: {len(takes)}", flush=True)
    return takes

def _micro_refine(takes: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Optional micro-cuts (silence, breath). Here we keep as-is, but
    enforce min/max durations.
    """
    out = []
    for t in takes:
        d = t["end"] - t["start"]
        if d < 0.25:
            continue
        if d > MAX_TAKE_SEC:
            # split into chunks of MAX_TAKE_SEC
            n = int(math.ceil(d / MAX_TAKE_SEC))
            span = d / n
            cur = t["start"]
            for i in range(n):
                out.append({
                    "id": f'{t["id"]}s{i+1:02d}',
                    "start": cur,
                    "end": min(t["end"], cur + span),
                    "text": t["text"]
                })
                cur += span
        else:
            out.append(t)
    print(f"[micro] input_takes={len(takes)} → micro_takes={len(out)}", flush=True)
    return out

# ---------- Semantic/visual tagging ----------
# We reuse your already-added semantic_visual_pass.py contract if present.
def _semantic_tag_and_score(takes: List[Dict[str,Any]]) -> Dict[str, List[Dict[str,Any]]]:
    """
    Returns slots dict:
    { "HOOK":[...], "PROBLEM":[...], "FEATURE":[...], "PROOF":[...], "CTA":[...] }
    Each item has id/start/end/text/meta/face_q/scene_q/vtx_sim...
    """
    if not SEMANTICS_ENABLED:
        # trivial: everything is HOOK in order
        return {"HOOK": takes[:], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}

    try:
        # Import your implemented module (you already added this earlier)
        from semantic_visual_pass import Take, tag_slot, score_take
        # light fake visual scores for now (1.0)
        enriched = []
        for t in takes:
            tt = Take(
                id=t["id"], start=t["start"], end=t["end"], text=t.get("text",""),
                face_q=1.0, scene_q=1.0, vtx_sim=0.0, has_product=False, ocr_hit=0
            )
            slot = tag_slot(tt, None)
            sc = score_take(tt, slot)
            item = {
                "id": tt.id, "start": tt.start, "end": tt.end, "text": tt.text,
                "meta": {"slot": slot, "score": sc},
                "face_q": tt.face_q, "scene_q": tt.scene_q, "vtx_sim": tt.vtx_sim,
                "has_product": tt.has_product, "ocr_hit": tt.ocr_hit
            }
            enriched.append(item)

        # group by slot
        slots = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
        for it in enriched:
            s = it["meta"]["slot"]
            s = s if s in slots else "HOOK"
            slots[s].append(it)

        # sort each slot by start (preserve narrative) or by score desc if you prefer
        for k in slots:
            slots[k].sort(key=lambda x: (x["start"], -x["meta"].get("score",0)))
        return slots
    except Exception as e:
        print("[semantic] failed:", repr(e), flush=True)
        # graceful fallback
        return {"HOOK": takes[:], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}

# ---------- Funnel selection ----------
def _pick_funnel(slots: Dict[str, List[Dict[str,Any]]], counts: Tuple[int,int,int,int,int,int]) -> List[Dict[str,Any]]:
    want_hook, want_prob, want_feat, want_proof, want_cta, want_benefits = counts
    # merge BENEFITS into PROBLEM pool if provided
    if want_benefits:
        # If you later add a separate BENEFITS bucket, merge it here.
        pass

    order = []
    def take_n(pool, n):
        if n < 0:  # unlimited
            return pool[:]
        if n == 0:
            return []
        return pool[:n]

    order.extend(take_n(slots.get("HOOK", []),    want_hook if want_hook != 0 else 0))
    order.extend(take_n(slots.get("PROBLEM", []), want_prob))
    order.extend(take_n(slots.get("FEATURE", []), want_feat))
    order.extend(take_n(slots.get("PROOF", []),   want_proof))
    order.extend(take_n(slots.get("CTA", []),     want_cta))

    # If strictly nothing selected, create a fallback using the earliest material
    if not order:
        # collect earliest from all buckets
        pool = []
        for k in ("HOOK","PROBLEM","FEATURE","PROOF","CTA"):
            pool.extend(slots.get(k, []))
        pool.sort(key=lambda x: x["start"])
        dur = 0.0
        chosen = []
        for it in pool:
            seg = it["end"] - it["start"]
            if seg <= 0: 
                continue
            chosen.append(it)
            dur += seg
            if dur >= max(FALLBACK_MIN_SEC, 6.0):
                break
        return chosen
    return order

# ---------- FFmpeg ----------
def _ffmpeg_subclip(src: str, dst: str, ss: float, ee: float):
    dur = max(0.001, ee - ss)
    cmd = [
        FFMPEG, "-y",
        "-ss", f"{ss:.3f}", "-i", src,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        dst
    ]
    _run(cmd)

def _ffmpeg_concat(listfile: str, dst: str):
    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", listfile,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-g", "48",
        "-c:a", "aac", "-b:a", "128k",
        dst
    ]
    _run(cmd)

# ---------- S3 ----------
def _s3_upload(path: str) -> Tuple[str,str,str]:
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"{S3_PREFIX}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    extra = {"ACL": S3_ACL, "ContentType": "video/mp4"} if S3_ACL else {"ContentType": "video/mp4"}
    s3.upload_file(path, S3_BUCKET, key, ExtraArgs=extra)
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return key, f"s3://{S3_BUCKET}/{key}", https_url

# ---------- Main pipeline ----------
def run_pipeline(local_path: str, payload: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    payload = payload or {}
    session_id = payload.get("session_id", f"sess-{uuid.uuid4().hex[:8]}")
    options = payload.get("options", {}) or {}

    # allow request-time overrides
    fc = options.get("FUNNEL_COUNTS", FUNNEL_COUNTS_ENV)
    counts = _parse_funnel_counts(str(fc))

    max_out_sec = _safe_float(options.get("MAX_DURATION_SEC", MAX_DURATION_SEC), MAX_DURATION_SEC)

    # 1) ASR → sentence takes → micro refine
    segs = _asr_segments(local_path)
    takes = _sent_takes(segs)
    takes = _micro_refine(takes)

    # 2) Semantic/visual tagging & scoring
    slots = _semantic_tag_and_score(takes)

    # 3) Funnel pick (respect counts & max duration)
    selected = _pick_funnel(slots, counts)

    # enforce max duration
    out = []
    t_acc = 0.0
    for it in selected:
        seg = max(0.0, it["end"] - it["start"])
        if seg <= 0.01: 
            continue
        if t_acc + seg > max_out_sec:
            # trim last piece
            remaining = max(0.0, max_out_sec - t_acc)
            if remaining >= 0.25:
                it = dict(it)
                it["end"] = it["start"] + remaining
                out.append(it)
                t_acc += remaining
            break
        out.append(it)
        t_acc += seg

    # 4) Render with ffmpeg (subclips + concat)
    parts = []
    for i, it in enumerate(out, 1):
        dst = _tmpfile(suffix=f".part{i:02d}.mp4")
        _ffmpeg_subclip(local_path, dst, it["start"], it["end"])
        parts.append(dst)

    if not parts:
        # nothing → fallback first few seconds
        dst = _tmpfile(".mp4")
        _ffmpeg_subclip(local_path, dst, 0.0, min(6.0, _ffprobe_duration(local_path)))
        parts.append(dst)

    concat_list = _tmpfile(".txt")
    with open(concat_list, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    final_out = _tmpfile(".mp4")
    _ffmpeg_concat(concat_list, final_out)
    dur = _ffprobe_duration(final_out)

    # 5) Upload to S3
    s3_key, s3_url, https_url = _s3_upload(final_out)

    result = {
        "ok": True,
        "input_local": local_path,
        "duration_sec": round(dur, 3),
        "s3_key": s3_key,
        "s3_url": s3_url,
        "https_url": https_url,
        "clips": [
            {
                "id": it["id"], "slot": it.get("meta",{}).get("slot"),
                "start": round(it["start"],3), "end": round(it["end"],3),
                "score": it.get("meta",{}).get("score"),
                "face_q": it.get("face_q",1.0),
                "scene_q": it.get("scene_q",1.0),
                "vtx_sim": it.get("vtx_sim",0.0),
                "chain_ids": it.get("meta",{}).get("chain_ids", []),
            } for it in out
        ],
        "slots": {
            k: [
                {
                    "id": it["id"], "start": round(it["start"],3), "end": round(it["end"],3),
                    "text": it.get("text",""),
                    "meta": {**it.get("meta",{}), **({"slot": k} if "slot" not in it.get("meta",{}) else {})},
                    "face_q": it.get("face_q",1.0),
                    "scene_q": it.get("scene_q",1.0),
                    "vtx_sim": it.get("vtx_sim",0.0),
                    "has_product": it.get("has_product", False),
                    "ocr_hit": it.get("ocr_hit", 0),
                } for it in slots.get(k, [])
            ] for k in ("HOOK","PROBLEM","FEATURE","PROOF","CTA")
        },
        "semantic": SEMANTICS_ENABLED,
        "vision": False,
        "asr": ASR_ENABLED,
    }
    return result

# RQ-friendly adapter — you can enqueue "tasks.job_render"
def job_render(payload=None, **kwargs):
    if payload is None:
        payload = {}
    if isinstance(payload, str):
        return run_pipeline(local_path=payload, payload=None)
    local_path = None
    if isinstance(payload, dict):
        local_path = payload.get("local_path")
    return run_pipeline(local_path=local_path, payload=payload or kwargs)
