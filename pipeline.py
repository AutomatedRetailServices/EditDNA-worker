# worker/pipeline.py
import os, uuid, time, tempfile, subprocess, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------
# Env helpers (safe parsing)
# ---------------------------
def _env(k: str, default: str = "") -> str:
    v = os.getenv(k, "").strip()
    if not v:
        return default
    # strip quotes and inline comments like: VALUE  # comment
    v = v.split("#", 1)[0].strip().strip('"').strip("'")
    return v

def _env_float(k: str, default: float) -> float:
    raw = _env(k, "")
    if not raw:
        return default
    try:
        return float(raw.split()[0])
    except Exception:
        return default

# ---- Binaries ----
FFMPEG_BIN  = _env("FFMPEG_BIN", "/usr/bin/ffmpeg")
FFPROBE_BIN = _env("FFPROBE_BIN", "/usr/bin/ffprobe")

# ---- S3 ----
S3_BUCKET = _env("S3_BUCKET")
S3_PREFIX = _env("S3_PREFIX", "editdna/outputs")
AWS_REGION = _env("AWS_REGION", "us-east-1")
S3_ACL = _env("S3_ACL", "public-read")

# ---- Controls (no hard cap unless caller passes max_duration) ----
MIN_TAKE_SEC     = _env_float("MIN_TAKE_SEC", 2.0)
MAX_TAKE_SEC     = _env_float("MAX_TAKE_SEC", 20.0)
FORCE_CTA_AT_END = _env("FORCE_CTA_AT_END", "1") in ("1", "true", "TRUE")

# ---- LLM (ALWAYS-ON) ----
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL   = _env("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is required (LLM is always-on by design). "
        "Set it in your RunPod env and restart the worker."
    )

# Light client (no streaming)
from openai import OpenAI
_oclient = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Rules / hints
# ---------------------------
BAD_PHRASES = [
    "wait", "hold on", "lemme start again", "let me start again",
    "start over", "no no", "redo", "sorry", "uh", "um",
    "why can't i remember", "i forgot", "not moisture control"  # from IMG_03 noise
]

CTA_FLUFF = [
    "click the link", "get yours today", "go ahead and click",
    "go ahead and grab", "i left it down below", "grab one of these",
    "if you want to check them out"
]

UGLY_BRANCHES = [
    "but if you don't like the checker", "but if you do",
    "but if you don't", "but if you"
]

FEATURE_HINTS = [
    "pocket", "pockets", "zipper", "strap", "opening", "inside",
    "material", "woven", "quality", "hardware", "comes with",
    "it has", "it also has", "it's actually", "design"
]

CTA_STARTERS = [
    "so", "so don't be shy", "ladies", "you only need to take",
    "you can't worry no more", "here's why", "here's how", "get"
]

# ---------------------------
# Data classes
# ---------------------------
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    rscore: float = 0.0  # rule score
    lscore: float = 0.0  # LLM score
    slot_hint: str = ""  # HOOK/FEATURE/PROOF/CTA hint

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ---------------------------
# Shell helpers
# ---------------------------
def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, (out or "").strip(), (err or "").strip()

def _tmpfile(suffix: str) -> str:
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p

def _download_to_tmp(url: str) -> str:
    local = _tmpfile(".mp4")
    code, out, err = _run(["curl", "-L", "-o", local, url])
    if code != 0:
        raise RuntimeError(f"curl failed {code}: {err}")
    return local

def _ffprobe_duration(path: str) -> float:
    code, out, err = _run([
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ])
    if code != 0:
        return 0.0
    try:
        return float(out.strip())
    except Exception:
        return 0.0

# ---------------------------
# ASR
# ---------------------------
def _try_import_transcribe_segments():
    try:
        from worker.asr import transcribe_segments  # your hook
        return transcribe_segments
    except Exception:
        return None

def _asr_segments_via_faster_whisper(src: str) -> List[Dict[str, Any]]:
    # very compact local ASR using faster-whisper (medium or small)
    from faster_whisper import WhisperModel
    model_size = os.getenv("FWHISPER_MODEL", "small")
    compute_type = os.getenv("FWHISPER_COMPUTE", "float16")
    model = WhisperModel(model_size, device="cuda" if _has_cuda() else "cpu",
                         compute_type=compute_type)
    segs = []
    for i, (text, start, end) in enumerate(_iterate_fw_segments(model, src), start=1):
        segs.append({"id": f"ASR{i:04d}", "text": text.strip(), "start": float(start), "end": float(end)})
    return segs

def _iterate_fw_segments(model, src: str):
    segments, _ = model.transcribe(src, vad_filter=True)
    for seg in segments:
        yield seg.text or "", seg.start, seg.end

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _load_asr_segments(src: str) -> Optional[List[Dict[str, Any]]]:
    hook = _try_import_transcribe_segments()
    if hook:
        try:
            segs = hook(src)
            if segs:
                return segs
        except Exception:
            pass
    # fallback to faster-whisper on this box
    try:
        segs = _asr_segments_via_faster_whisper(src)
        return segs if segs else None
    except Exception:
        return None

# ---------------------------
# Clause utilities
# ---------------------------
def _split_into_clauses(text: str) -> List[str]:
    if not text:
        return []
    t = " ".join(text.split())
    # split on punctuation and major conjunctions
    tmp = []
    buf = ""
    for ch in t:
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
                p = part.strip(" ,.;")
                if p:
                    clauses.append(p)
        else:
            p = piece.strip(" ,.;")
            if p:
                clauses.append(p)

    return [c for c in clauses if len(c.split()) >= 3]

def _assign_times(seg_start: float, seg_end: float, clauses: List[str]) -> List[Tuple[float, float, str]]:
    dur = max(0.05, seg_end - seg_start)
    joined = " ".join(clauses)
    total_len = max(1, len(joined))
    out = []
    cursor = 0
    for c in clauses:
        clen = len(c)
        s_rel = cursor / total_len
        e_rel = (cursor + clen) / total_len
        out.append((seg_start + s_rel * dur, seg_start + e_rel * dur, c.strip()))
        cursor += clen + 1
    return out

# ---------------------------
# Scoring: rules + LLM (always-on)
# ---------------------------
def _rule_score(c: str) -> Tuple[float, str]:
    """Return (score, slot_hint). Higher is better."""
    low = c.lower().strip()

    # drop really bad
    for p in BAD_PHRASES:
        if p in low:
            return (-1.0, "")

    # punish "ugly branches" that derail the pitch
    for p in UGLY_BRANCHES:
        if p in low:
            return (0.1, "FEATURE")

    # CTA fluff (we'll keep short CTA at end but not mid-story)
    ctaish = any(p in low for p in CTA_FLUFF) or low.startswith("if you want to")
    if ctaish:
        return (0.6, "CTA")  # useful if at end

    # feature hints
    if any(h in low for h in FEATURE_HINTS):
        return (0.8, "FEATURE")

    # hook-ish (provocative/opinionated/opening)
    if any(low.startswith(s) for s in ("if you don't", "listen", "real talk", "this is")):
        return (0.9, "HOOK")

    # neutral product statement
    return (0.7, "FEATURE")

def _llm_score_clause(text: str) -> Tuple[float, str]:
    """
    Ask GPT to (a) rate usefulness for sales funnel, (b) tag slot.
    Returns (score_0to1, slot_hint).
    """
    prompt = f"""
You are scoring a single spoken clause for use in a short sales video.

Clause: "{text}"

Score 0..1 for:
- HOOK: strong opener that grabs attention
- FEATURE: concrete product features/benefits
- PROOF: social proof, credibility
- CTA: call to action or persuasive close

Return a short JSON with keys: score (0..1 float), slot ("HOOK"|"FEATURE"|"PROOF"|"CTA").
If the line is a mistake, self-correction, filler or ramble, use score 0.0.
"""
    try:
        rsp = _oclient.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
        )
        content = (rsp.choices[0].message.content or "").strip()
        # naive parse
        import json, re
        m = re.search(r"\{.*\}", content, re.S)
        obj = json.loads(m.group(0)) if m else {}
        score = float(obj.get("score", 0.0))
        slot  = str(obj.get("slot", "FEATURE")).upper()
        if slot not in ("HOOK","FEATURE","PROOF","CTA"):
            slot = "FEATURE"
        # clamp
        score = max(0.0, min(1.0, score))
        return score, slot
    except Exception:
        # if LLM call fails (transient), do NOT crash; degrade to safe low score
        return 0.2, "FEATURE"

def _blend(rule: float, llm: float) -> float:
    # privilege LLM, but keep rules as floor/boost
    return 0.25 * rule + 0.75 * llm

# ---------------------------
# Cut export
# ---------------------------
def _export_concat(src: str, takes: List[Take]) -> str:
    if not takes:
        # 3s safety
        takes = [Take("FALLBACK", 0.0, min(3.0, _ffprobe_duration(src)), "")]
    listfile = _tmpfile(".txt")
    parts = []
    for i, t in enumerate(takes, start=1):
        pth = _tmpfile(f".part{i:02d}.mp4")
        parts.append(pth)
        dur = max(0.05, t.dur)
        _run([
            FFMPEG_BIN, "-y",
            "-ss", f"{t.start:.3f}",
            "-i", src,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", "48",
            "-c:a", "aac", "-b:a", "128k",
            pth
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

# ---------------------------
# Public entry
# ---------------------------
def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts,
    max_duration: Optional[float] = None,  # None = no cap
    s3_prefix: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:

    if not file_urls:
        return {"ok": False, "error": "no input files"}

    # 1) load
    src = _download_to_tmp(file_urls[0])
    vid_dur = _ffprobe_duration(src)
    cap = float(max_duration) if (max_duration not in (None, "", 0, "0")) else None

    # 2) ASR → segments
    segs = _load_asr_segments(src)
    if not segs:
        return {"ok": False, "error": "ASR failed or returned empty"}

    # 3) segments → clause-takes
    takes: List[Take] = []
    for i, seg in enumerate(segs, start=1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        if e - s > MAX_TAKE_SEC:
            e = s + MAX_TAKE_SEC
        if e - s < MIN_TAKE_SEC:
            continue

        clauses = _split_into_clauses(text)
        if not clauses:
            continue
        for j, (cs, ce, ctext) in enumerate(_assign_times(s, e, clauses), start=1):
            if ce - cs < 0.15:
                continue
            rscore, rhint = _rule_score(ctext)
            lscore, lhint = _llm_score_clause(ctext)
            hint = rhint if lhint == "FEATURE" and rhint in ("HOOK","PROOF","CTA") else lhint
            takes.append(
                Take(
                    id=f"ASR{i:04d}_c{j}",
                    start=cs, end=ce, text=ctext,
                    rscore=rscore, lscore=lscore, slot_hint=hint
                )
            )

    # 4) choose story: greedy by blended score, keep temporal order
    #    We keep clips sorted by start but filter by a moving threshold.
    takes_sorted = sorted(takes, key=lambda t: t.start)
    # blended threshold: keep > 0.45
    chosen: List[Take] = [t for t in takes_sorted if _blend(t.rscore, t.lscore) >= 0.45 and t.dur >= 0.25]

    # 5) ensure a CTA near the end
    if FORCE_CTA_AT_END:
        # find best CTA by blended score among last 40% of timeline or last 10s
        end_window_start = max(0.0, vid_dur * 0.6, vid_dur - 10.0)
        cta_candidates = [t for t in takes_sorted if t.slot_hint == "CTA" and t.start >= end_window_start]
        if not cta_candidates:
            # if none, accept the globally best CTA
            cta_candidates = [t for t in takes_sorted if t.slot_hint == "CTA"]
        if cta_candidates:
            best_cta = max(cta_candidates, key=lambda t: _blend(t.rscore, t.lscore))
            # if CTA not already in chosen, append it at the end
            if all(c.id != best_cta.id for c in chosen):
                chosen.append(best_cta)
        # filter mid-story CTA fluff unless it's the last clip
        for i in range(0, max(0, len(chosen) - 1)):
            if chosen[i].slot_hint == "CTA":
                # drop mid CTA if not very strong
                if _blend(chosen[i].rscore, chosen[i].lscore) < 0.65:
                    chosen[i] = None  # mark for deletion
        chosen = [c for c in chosen if c is not None]

    # 6) duration trimming (only if cap provided)
    if cap is not None and cap > 0:
        acc = 0.0
        trimmed = []
        for t in chosen:
            if acc + t.dur <= cap + 1e-3:  # allow tiny epsilon
                trimmed.append(t)
                acc += t.dur
            else:
                # if we can fit partial of this take to reach cap +/- 0.25s, do it
                remain = cap - acc
                if remain >= 0.5:
                    trimmed.append(Take(t.id, t.start, t.start + remain, t.text, t.rscore, t.lscore, t.slot_hint))
                    acc += remain
                break
        chosen = trimmed

    # 7) export
    final_path = _export_concat(src, chosen)
    s3info = _upload_to_s3(final_path, s3_prefix)

    # 8) response
    def _trim(txt: str, n: int = 240) -> str:
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    clips = [{
        "id": t.id,
        "slot": "STORY",
        "start": t.start,
        "end": t.end,
        "score": round(_blend(t.rscore, t.lscore), 3),
        "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0,
        "chain_ids": [t.id],
        "text": _trim(t.text),
    } for t in chosen]

    slots: Dict[str, List[Dict[str, Any]]] = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}
    if chosen:
        # HOOK = first strong non-CTA
        for t in chosen:
            if t.slot_hint != "CTA":
                slots["HOOK"].append({
                    "id": t.id, "start": t.start, "end": t.end, "text": _trim(t.text),
                    "meta": {"slot":"HOOK","score":_blend(t.rscore,t.lscore),"chain_ids":[t.id]},
                    "face_q":1.0,"scene_q":1.0,"vtx_sim":0.0,"has_product":False,"ocr_hit":0
                })
                break
        # FEATURES = middles
        middles = [t for t in chosen[1:-1]] if len(chosen) > 2 else chosen[1:]
        for t in middles:
            slots["FEATURE"].append({
                "id": t.id, "start": t.start, "end": t.end, "text": _trim(t.text),
                "meta": {"slot":"FEATURE","score":_blend(t.rscore,t.lscore),"chain_ids":[t.id]},
                "face_q":1.0,"scene_q":1.0,"vtx_sim":0.0,"has_product":False,"ocr_hit":0
            })
        # CTA = last if looks CTAish, otherwise empty
        last = chosen[-1]
        if last.slot_hint == "CTA" or any(k in last.text.lower() for k in ("click", "get yours", "link", "grab one", "shop", "buy", "check them out")):
            slots["CTA"].append({
                "id": last.id, "start": last.start, "end": last.end, "text": _trim(last.text),
                "meta": {"slot":"CTA","score":_blend(last.rscore,last.lscore),"chain_ids":[last.id]},
                "face_q":1.0,"scene_q":1.0,"vtx_sim":0.0,"has_product":False,"ocr_hit":0
            })

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": src,
        "duration_sec": _ffprobe_duration(final_path),
        "s3_key": s3info["s3_key"],
        "s3_url": s3info["s3_url"],
        "https_url": s3info["https_url"],
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": False,  # future: face/product gates
    }

# ---------------------------
# S3 upload
# ---------------------------
import boto3
def _upload_to_s3(local_path: str, s3_prefix: Optional[str]) -> Dict[str,str]:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    client = boto3.client("s3", region_name=AWS_REGION)
    prefix = (s3_prefix or S3_PREFIX).rstrip("/")
    key = f"{prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"
    with open(local_path, "rb") as fh:
        client.upload_fileobj(
            fh, S3_BUCKET, key,
            ExtraArgs={"ACL": S3_ACL, "ContentType": "video/mp4"}
        )
    return {
        "s3_key": key,
        "s3_url": f"s3://{S3_BUCKET}/{key}",
        "https_url": f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}",
    }
