# jobs.py — EditDNA unified worker pipeline (drop-in)
# Single-file: ASR -> takes -> clean/dedup -> merge -> funnel select -> render -> S3
# No relative imports. Safe fallbacks. Works with RQ calling tasks.job_render(payload).

import os, re, json, uuid, math, tempfile, subprocess, urllib.request
from typing import List, Dict, Tuple, Optional

# ---------- ENV ----------
S3_BUCKET         = os.getenv("S3_BUCKET", "")
S3_REGION         = os.getenv("S3_REGION", os.getenv("AWS_REGION", "us-east-1"))
S3_PREFIX         = os.getenv("S3_PREFIX", "editdna/outputs")
S3_ACL            = os.getenv("S3_ACL", "public-read")

FFMPEG_BIN        = os.getenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN       = os.getenv("FFPROBE_BIN", "/usr/bin/ffprobe")

ASR_MODEL_SIZE    = os.getenv("ASR_MODEL_SIZE", "tiny")   # tiny|base|small|medium|large
ASR_DEVICE        = os.getenv("ASR_DEVICE", "cuda")       # cuda|cpu
ASR_DOWNLOAD_ROOT = os.getenv("ASR_DOWNLOAD_ROOT", "/workspace/.cache/whisper")

BIN_SEC           = float(os.getenv("BIN_SEC", "1.0"))
MIN_TAKE_SEC      = float(os.getenv("MIN_TAKE_SEC", "0.50"))
MAX_TAKE_SEC      = float(os.getenv("MAX_TAKE_SEC", "220"))
MAX_DURATION_SEC  = float(os.getenv("MAX_DURATION_SEC", "220"))

SEMANTICS_ENABLED = os.getenv("SEMANTICS_ENABLED", "1") == "1"
EMBEDDER_NAME     = os.getenv("EMBEDDER", "local")  # not used strictly; we try ST then fallback

SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM     = float(os.getenv("SEM_MERGE_SIM", "0.70"))
VIZ_MERGE_SIM     = float(os.getenv("VIZ_MERGE_SIM", "0.70"))  # placeholder; we don’t have visual yet
MERGE_MAX_CHAIN   = int(os.getenv("MERGE_MAX_CHAIN", "10"))

SEM_FILLER_LIST   = [w.strip() for w in os.getenv("SEM_FILLER_LIST", "um,uh,like,so,okay").split(",") if w.strip()]
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))
RETRY_TOKENS      = re.compile(r"\b(uh|um|wait|hold on|let me start again|start over|sorry|i mean|actually|no no|take two|redo)\b", re.I)

# Product/OCR slot gating — leave empty unless you really want hard constraints
SLOT_REQUIRE_PRODUCT = set([s for s in os.getenv("SLOT_REQUIRE_PRODUCT", "").split(",") if s.strip()])
SLOT_REQUIRE_OCR_CTA = set([s for s in os.getenv("SLOT_REQUIRE_OCR_CTA", "").split(",") if s.strip()])

FALLBACK_MIN_SEC  = float(os.getenv("FALLBACK_MIN_SEC", "0"))

# ---------- Optional deps ----------
def _maybe_import_st():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        return SentenceTransformer, np
    except Exception:
        return None, None

def _maybe_import_boto3():
    try:
        import boto3
        return boto3
    except Exception:
        return None

def _maybe_import_whisper():
    try:
        import whisper
        return whisper
    except Exception:
        return None

def _maybe_import_moviepy():
    try:
        from moviepy.editor import VideoFileClip
        return VideoFileClip
    except Exception:
        return None

# ---------- Helpers ----------
def _run(cmd: List[str]) -> None:
    print("[ff] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def _probe_duration(path: str) -> float:
    try:
        out = subprocess.check_output([
            FFPROBE_BIN, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", path
        ], text=True).strip()
        return float(out)
    except Exception:
        VFC = _maybe_import_moviepy()
        if VFC:
            try:
                return float(VFC(path).duration)
            except Exception:
                pass
    return 0.0

def _upload_to_s3(local_path: str, key: str) -> Tuple[str, str]:
    boto3 = _maybe_import_boto3()
    if not boto3 or not S3_BUCKET:
        print("[s3] boto3 or S3_BUCKET missing — skipping upload, returning local only.")
        return f"s3://{S3_BUCKET}/{key}", f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
    s3 = boto3.client("s3", region_name=S3_REGION)
    extra = {"ACL": S3_ACL} if S3_ACL else {}
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs=extra)
    print(f"[s3] uploaded s3://{S3_BUCKET}/{key}")
    return f"s3://{S3_BUCKET}/{key}", f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"

# ---------- ASR ----------
def _load_whisper():
    whisper = _maybe_import_whisper()
    if not whisper:
        print("[asr] whisper not installed")
        return None, None
    os.makedirs(ASR_DOWNLOAD_ROOT, exist_ok=True)
    print(f"[asr] load model='{ASR_MODEL_SIZE}' device='{ASR_DEVICE}' cache='{ASR_DOWNLOAD_ROOT}'")
    # 1) requested device
    try:
        m = whisper.load_model(ASR_MODEL_SIZE, device=ASR_DEVICE, download_root=ASR_DOWNLOAD_ROOT)
        print(f"[asr] loaded {ASR_MODEL_SIZE} on {ASR_DEVICE}")
        return m, ASR_DEVICE
    except Exception as e:
        print("[asr] primary load failed:", repr(e))
    # 2) CPU fallback
    try:
        m = whisper.load_model(ASR_MODEL_SIZE, device="cpu", download_root=ASR_DOWNLOAD_ROOT)
        print(f"[asr] loaded {ASR_MODEL_SIZE} on cpu")
        return m, "cpu"
    except Exception as e:
        print("[asr] cpu fallback failed:", repr(e))
    # 3) tiny on CPU
    try:
        m = whisper.load_model("tiny", device="cpu", download_root=ASR_DOWNLOAD_ROOT)
        print("[asr] loaded tiny on cpu")
        return m, "cpu"
    except Exception as e:
        print("[asr] tiny cpu fallback failed:", repr(e))
    print("[asr] FATAL: cannot load any whisper model")
    return None, None

def _do_whisper_asr(local_path: str) -> List[Dict]:
    model, device = _load_whisper()
    if not model:
        dur = _probe_duration(local_path)
        print(f"[asr] disabled; returning full-span 0.0–{dur:.2f}s")
        return [{"start": 0.0, "end": dur, "text": ""}]
    try:
        fp16 = (device == "cuda")
        result = model.transcribe(local_path, fp16=fp16)
    except Exception as e:
        print("[asr] transcribe failed:", repr(e))
        dur = _probe_duration(local_path)
        return [{"start": 0.0, "end": dur, "text": ""}]
    segs = []
    for seg in result.get("segments", []):
        st = float(seg.get("start", 0.0))
        en = float(seg.get("end", st))
        if en - st < MIN_TAKE_SEC:
            continue
        segs.append({"start": st, "end": en, "text": (seg.get("text") or "").strip()})
    print(f"[asr] segments: {len(segs)} (min_take={MIN_TAKE_SEC}s)")
    if not segs:
        dur = _probe_duration(local_path)
        return [{"start": 0.0, "end": dur, "text": ""}]
    return segs

# ---------- Text quality / retry detection ----------
def _is_retry_or_noise(text: str) -> bool:
    if not text:
        return False
    words = re.findall(r"\w+", text.lower())
    if not words:
        return False
    fillers = sum(1 for w in words if w in SEM_FILLER_LIST)
    rate = fillers / max(1, len(words))
    return rate > SEM_FILLER_MAX_RATE or bool(RETRY_TOKENS.search(text))

# ---------- Slotting ----------
SLOTS_6 = ["HOOK","PROBLEM","BENEFITS","FEATURE","PROOF","CTA"]
SLOTS_5 = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]  # PROBLEM covers BENEFITS

KEYS = {
    "HOOK":     ["imagine","what if","did you know","stop scrolling","quick tip","why not","secret","listen"],
    "PROBLEM":  ["problem","struggle","hard","pain","issue","annoying","dry","broke","hate","because"],
    "BENEFITS": ["better","faster","saves time","results","benefit","help","you get","you’ll get","improves"],
    "FEATURE":  ["feature","comes with","includes","made of","ingredient","formula","it has","has a"],
    "PROOF":    ["results","testimonials","it works","i use it","demo","watch","evidence","review"],
    "CTA":      ["buy","get","claim","use code","link in bio","shop","today","now","check them out","grab one"],
}

def _guess_slot(text: str, use_six: bool) -> str:
    txt = (text or "").lower()
    labels = SLOTS_6 if use_six else SLOTS_5
    for slot in labels:
        keys = KEYS["PROBLEM"] if (slot=="BENEFITS" and not KEYS.get("BENEFITS")) else KEYS.get(slot, [])
        if any(k in txt for k in keys):
            return slot
    # fallbacks
    if "it has" in txt or "includes" in txt or "made of" in txt:
        return "FEATURE" if not use_six else "FEATURE"
    if "buy" in txt or "check them out" in txt or "grab" in txt:
        return "CTA"
    # long sentences tend to be PROOF/benefit
    if len(txt.split()) > 20:
        return "PROOF" if not use_six else "BENEFITS"
    return "HOOK"

# ---------- Embeddings (optional) ----------
_ST = None
_NP = None
def _get_st():
    global _ST, _NP
    if _ST is not None:
        return _ST, _NP
    ST, NP = _maybe_import_st()
    _ST, _NP = ST, NP
    return _ST, _NP

def _emb_texts(texts: List[str]):
    ST, NP = _get_st()
    if not ST or not NP:
        return None, None
    try:
        mdl = ST("sentence-transformers/all-MiniLM-L6-v2")
        V = mdl.encode(texts, normalize_embeddings=True)
        return V, lambda a, b: float((a*b).sum())
    except Exception as e:
        print("[sem] embed load/encode failed:", repr(e))
        return None, None

# ---------- Build + clean takes ----------
def _build_takes(local_video: str, use_six_slots: bool) -> List[Dict]:
    segs = _do_whisper_asr(local_video)
    takes = []
    for i, s in enumerate(segs):
        st, en = float(s["start"]), float(s["end"])
        if en <= st: 
            continue
        if en - st > MAX_TAKE_SEC:
            en = st + MAX_TAKE_SEC
        txt = s.get("text", "").strip()
        # veto retries/noise
        if _is_retry_or_noise(txt):
            continue
        takes.append({
            "id": f"T{i+1:04d}",
            "start": st,
            "end": en,
            "text": txt,
            "slot": _guess_slot(txt, use_six_slots),
            "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0,
            "has_product": False, "ocr_hit": 0,
        })
    print(f"[seg] takes: {len(takes)}")
    # semantic dedup (optional)
    if SEMANTICS_ENABLED and len(takes) > 1:
        texts = [t["text"] for t in takes]
        V, sim = _emb_texts(texts)
        if V is not None:
            kept = []
            for i, t in enumerate(takes):
                dup = False
                for j, k in enumerate(kept):
                    s = sim(V[i], V[texts.index(k["text"])])
                    if s >= SEM_DUP_THRESHOLD:
                        dup = True
                        break
                if not dup:
                    kept.append(t)
            print(f"[dedup] {len(takes)} → {len(kept)} (thr={SEM_DUP_THRESHOLD})")
            takes = kept
    return takes

# ---------- Merge (semantic continuity) ----------
def _can_merge(a: Dict, b: Dict) -> bool:
    if not SEMANTICS_ENABLED:
        return False
    V, sim = _emb_texts([a["text"], b["text"]])
    if V is None:
        return False
    s_sem = sim(V[0], V[1])
    if s_sem < SEM_MERGE_SIM:
        return False
    # crude visual proxy: scene_q avg threshold
    s_viz = min(float(a.get("scene_q",1.0)), float(b.get("scene_q",1.0)))
    return (s_viz >= VIZ_MERGE_SIM)

def _stitch_chain(takes: List[Dict]) -> List[Dict]:
    if not takes:
        return []
    takes = sorted(takes, key=lambda t: (t["start"], t["end"]))
    out = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (j+1 < len(takes)) and (len(chain) < MERGE_MAX_CHAIN) and _can_merge(takes[j], takes[j+1]):
            chain.append(takes[j+1]); j += 1
        merged = dict(chain[0])
        merged["end"] = chain[-1]["end"]
        merged["chain_ids"] = [c["id"] for c in chain]
        out.append(merged)
        i = j + 1
    print(f"[merge] {len(takes)} → {len(out)} (max_chain={MERGE_MAX_CHAIN})")
    return out

# ---------- Funnel selection ----------
def _parse_counts(s: Optional[str]) -> Tuple[List[str], List[int]]:
    """
    Accepts 5 or 6 CSV ints.
    6 = HOOK,PROBLEM,BENEFITS,FEATURE,PROOF,CTA
    5 = HOOK,PROBLEM,FEATURE,PROOF,CTA   (BENEFITS merges with PROBLEM)
    """
    if not s:
        return SLOTS_5, [0,0,0,0,0]
    parts = [p.strip() for p in s.split(",") if p.strip()!=""]
    vals = [int(x) for x in parts]
    if len(vals) == 6:
        return SLOTS_6, vals
    if len(vals) == 5:
        return SLOTS_5, vals
    # bad input → treat as “no limit”
    return SLOTS_5, [0,0,0,0,0]

def _select_funnel(takes: List[Dict], counts_csv: Optional[str]) -> List[Dict]:
    slots, counts = _parse_counts(counts_csv)
    want = {slot: counts[idx] if idx < len(counts) else 0 for idx, slot in enumerate(slots)}
    # If BENEFITS not present but takes have BENEFITS, map BENEFITS into PROBLEM bucket.
    out: List[Dict] = []
    by_slot: Dict[str, List[Dict]] = {}
    for t in takes:
        s = t["slot"]
        if s == "BENEFITS" and "BENEFITS" not in slots:
            s = "PROBLEM"
            t = dict(t); t["slot"] = "PROBLEM"
        by_slot.setdefault(s, []).append(t)

    # If all requested counts are 0 → no limit, return all ordered by start
    if all(v == 0 for v in want.values()):
        return sorted(takes, key=lambda x: x["start"])

    # otherwise pick up to N per slot in slot order
    for slot in slots:
        need = want.get(slot, 0)
        if need == 0:
            continue
        for t in sorted(by_slot.get(slot, []), key=lambda x: x["start"]):
            # hard slot constraints (product/ocr) if configured
            if slot in SLOT_REQUIRE_PRODUCT and not t.get("has_product"):
                continue
            if slot in SLOT_REQUIRE_OCR_CTA and int(t.get("ocr_hit",0)) < 1:
                continue
            out.append(t)
            if len([x for x in out if x["slot"] == slot]) >= need:
                break
    return out

# ---------- Render (ffmpeg concat) ----------
def _concat_ffmpeg(src_path: str, takes: List[Dict], max_duration: float) -> str:
    # build parts
    parts = []
    acc = 0.0
    for i, t in enumerate(sorted(takes, key=lambda x: x["start"])):
        st, en = t["start"], t["end"]
        dur = en - st
        if max_duration > 0 and acc + dur > max_duration:
            # trim last segment to fit cap
            dur = max(0.0, max_duration - acc)
            if dur < 0.25:  # too tiny, skip
                break
            en = st + dur
        out_part = f"/tmp/ed_part_{i:02d}.mp4"
        _run([
            FFMPEG_BIN, "-y", "-ss", f"{st:.3f}", "-i", src_path,
            "-t", f"{(en-st):.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            out_part
        ])
        parts.append(out_part)
        acc += (en - st)
        if max_duration > 0 and acc >= max_duration:
            break
    if not parts:
        # fallback: return 0–min(3s, video)
        dur = _probe_duration(src_path)
        cut = min(3.0, dur)
        out_part = "/tmp/ed_part_fallback.mp4"
        _run([
            FFMPEG_BIN, "-y", "-ss", "0.000", "-i", src_path,
            "-t", f"{cut:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            out_part
        ])
        parts = [out_part]

    # concat list
    list_path = "/tmp/ed_concat.txt"
    with open(list_path, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    out_path = f"/tmp/ed_{uuid.uuid4().hex}.mp4"
    _run([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
          "-i", list_path,
          "-c:v", "libx264", "-preset", "fast", "-crf", "23",
          "-pix_fmt", "yuv420p", "-g", "48",
          "-c:a", "aac", "-b:a", "128k",
          out_path])
    return out_path

# ---------- Public API ----------
def render_funnel(local_video: str, counts_csv: Optional[str]) -> Tuple[str, List[Dict], Dict[str, List[Dict]]]:
    use_six = (counts_csv and len([x for x in counts_csv.split(",") if x.strip()]) == 6)
    takes = _build_takes(local_video, use_six_slots=use_six)
    takes = _stitch_chain(takes) if SEMANTICS_ENABLED else takes
    selected = _select_funnel(takes, counts_csv)
    if not selected:
        # If selection produced nothing, use all takes (we avoid returning empty)
        selected = sorted(takes, key=lambda x: x["start"])
    # bucketize for JSON
    slots: Dict[str, List[Dict]] = {}
    for t in selected:
        slots.setdefault(t["slot"], []).append({
            k: t[k] for k in ["id","start","end","text","slot"]
            if k in t
        })
    out_path = _concat_ffmpeg(local_video, selected, MAX_DURATION_SEC)
    return out_path, selected, slots

def run_pipeline(local_path: str, payload: Dict) -> Dict:
    counts_csv = None
    try:
        # options overrides
        opts = (payload or {}).get("options", {})
        counts_csv = opts.get("FUNNEL_COUNTS") or os.getenv("FUNNEL_COUNTS")
        # live overrides for thresholds
        for k in ["SEM_MERGE_SIM","VIZ_MERGE_SIM","MERGE_MAX_CHAIN","MAX_DURATION_SEC","MIN_TAKE_SEC","MAX_TAKE_SEC"]:
            if k in opts:
                os.environ[k] = str(opts[k])
        # refresh globals if needed
        global SEM_MERGE_SIM, VIZ_MERGE_SIM, MERGE_MAX_CHAIN, MAX_DURATION_SEC, MIN_TAKE_SEC, MAX_TAKE_SEC
        SEM_MERGE_SIM   = float(os.getenv("SEM_MERGE_SIM", str(SEM_MERGE_SIM)))
        VIZ_MERGE_SIM   = float(os.getenv("VIZ_MERGE_SIM", str(VIZ_MERGE_SIM)))
        MERGE_MAX_CHAIN = int(os.getenv("MERGE_MAX_CHAIN", str(MERGE_MAX_CHAIN)))
        MAX_DURATION_SEC= float(os.getenv("MAX_DURATION_SEC", str(MAX_DURATION_SEC)))
        MIN_TAKE_SEC    = float(os.getenv("MIN_TAKE_SEC", str(MIN_TAKE_SEC)))
        MAX_TAKE_SEC    = float(os.getenv("MAX_TAKE_SEC", str(MAX_TAKE_SEC)))
    except Exception:
        pass

    print(f"[pipeline] FUNNEL_COUNTS='{counts_csv}' max_dur={MAX_DURATION_SEC}s")
    out_path, clips, slots = render_funnel(local_path, counts_csv)

    # Upload
    key = f"{S3_PREFIX}/{uuid.uuid4().hex}_{uuid.uuid4().int % 10**12}.mp4"
    s3_url, https_url = _upload_to_s3(out_path, key)

    duration = _probe_duration(out_path)
    return {
        "ok": True,
        "input_local": local_path,
        "duration_sec": duration,
        "s3_key": key,
        "s3_url": s3_url,
        "https_url": https_url,
        "clips": [
            {k: c.get(k) for k in ["id","slot","start","end","text","face_q","scene_q","vtx_sim","chain_ids"] if k in c}
            for c in clips
        ],
        "slots": slots,
        "semantic": SEMANTICS_ENABLED,
        "vision": False,
        "asr": True
    }

def job_render(payload: Dict) -> Dict:
    """
    RQ entrypoint: tasks.job_render(payload) → dict
    Expected payload:
      {
        "session_id": "...",
        "files": ["https://.../input.mov"],
        "options": {
           "FUNNEL_COUNTS": "1,1,0,1,1,1"  # (6) or "1,1,1,1,1" (5)
           ... overrides ...
        }
      }
    """
    print(f"[job_render] payload keys={list((payload or {}).keys())}")
    if not payload or "files" not in payload or not payload["files"]:
        raise ValueError("payload.files is required")
    url = payload["files"][0]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(url)[1] or ".mp4").name
    print(f"[download] {url} -> {tmp}")
    urllib.request.urlretrieve(url, tmp)
    res = run_pipeline(tmp, payload)
    print("[job_render] done.")
    return res
