from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os, tempfile, subprocess, json, re, uuid, shutil
from typing import List, Dict, Any, Tuple
from openai import OpenAI

app = FastAPI()

# OpenAI client (requires OPENAI_API_KEY)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -----------------------
# Config / Paths
# -----------------------
BASE_DIR = os.environ.get("S2CS_BASE_DIR", "/tmp/s2cs")
OUTPUT_DIR = os.environ.get("S2CS_OUTPUT_DIR", "/tmp/s2cs/out")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory session index (dev). In prod, move to Redis/DB.
SESSIONS: Dict[str, Dict[str, Any]] = {}

# -----------------------
# Utilities
# -----------------------

def run(cmd: List[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")

def extract_duration_seconds(media_path: str) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return float((probe.stdout or "").strip())
    except:
        return 0.0

def ensure_portrait_filter():
    return "scale=-2:1920,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"

# -----------------------
# Segmentation + Transcription (single file)
# -----------------------

def segment_by_silence_and_transcribe_from_path(src_path: str) -> Tuple[List[Dict[str, Any]], str]:
    segments = []
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        run(["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav])

        proc = subprocess.run(
            ["ffmpeg", "-i", wav, "-af", "silencedetect=noise=-30dB:d=0.35", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        log = proc.stderr or ""

        starts = [0.0]
        ends = []
        for m in re.finditer(r"silence_start:\s*([0-9.]+)", log):
            try: ends.append(float(m.group(1)))
            except: pass
        for m in re.finditer(r"silence_end:\s*([0-9.]+)", log):
            try: starts.append(float(m.group(1)))
            except: pass

        total = extract_duration_seconds(wav)
        if not ends or (ends and ends[-1] < total):
            ends.append(total)

        for s, e in zip(starts, ends):
            if (e - s) < 0.25:
                continue
            chunk = os.path.join(td, f"seg_{int(s*1000)}.wav")
            run([
                "ffmpeg", "-y",
                "-ss", f"{s:.3f}",
                "-t", f"{(e - s):.3f}",
                "-i", wav,
                "-ac", "1", "-ar", "16000",
                chunk
            ])
            with open(chunk, "rb") as f:
                tr = client.audio.transcriptions.create(model="whisper-1", file=f)
            text = (getattr(tr, "text", "") or "").strip()
            if text:
                segments.append({"start": round(s, 3), "end": round(e, 3), "text": text})

        transcript = " ".join(seg["text"] for seg in segments)
        return segments, transcript

# -----------------------
# Classification (text → buckets)
# -----------------------

def classify_text_buckets(transcript: str, segments_flat: List[Dict[str, Any]],
                          features_csv: str, tone: str) -> Dict[str, Any]:
    seg_listing = []
    for seg in segments_flat[:80]:
        seg_listing.append(f"[{seg['file_id']} {seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")
    seg_text = "\n".join(seg_listing)

    system = (
        "You are an ad pre-editor for TikTok Shop/UGC. "
        "From the given transcript and timestamped segments, extract ONLY lines that already exist "
        "(no rewriting). Return STRICT JSON with keys: "
        "hook_lines (array of strings), feature_lines (array of strings), "
        "proof_lines (array of strings), cta_lines (array of strings). "
        "Keep each list to at most 6 short items (verbatim)."
    )
    user = (
        f"Tone: {tone}\n"
        f"Key features to prioritize: {features_csv}\n\n"
        f"Transcript (full):\n{transcript}\n\n"
        f"Segments (timestamped, earliest first):\n{seg_text}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=900,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except:
        data = {}

    def arr(key):
        v = data.get(key, []) if isinstance(data, dict) else []
        return [x for x in v if isinstance(x, str) and x.strip()][:6]

    return {
        "hook_lines": arr("hook_lines"),
        "feature_lines": arr("feature_lines")[:3],
        "proof_lines": arr("proof_lines")[:3],
        "cta_lines": arr("cta_lines")[:3],
    }

# -----------------------
# Mapping + scoring
# -----------------------

EARLY_SEC = 15.0

def find_segment_for_line(line: str, segments: List[Dict[str, Any]]):
    needle = re.sub(r"\s+", " ", line.strip().lower())
    for seg in segments:
        hay = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if needle and needle in hay:
            return seg
    n_tokens = set(needle.split())
    best = None
    best_overlap = 0
    for seg in segments:
        h_tokens = set(re.sub(r"\s+", " ", seg["text"].lower()).split())
        overlap = len(n_tokens & h_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg
    return best

def score_hook(seg: Dict[str, Any]) -> Tuple[float, str, bool]:
    start = float(seg["start"]); end = float(seg["end"])
    dur = max(0.01, end - start); text = seg["text"]
    score = 0.5; reasons = []; is_early = start <= EARLY_SEC
    if is_early: score += 0.25; reasons.append("early (≤15s)")
    if 2.0 <= dur <= 7.0: score += 0.2; reasons.append("snackable length")
    cues = 0
    cues += 1 if re.search(r"\b(stop|wait|hold|don’t|don't)\b", text, re.I) else 0
    cues += 1 if "?" in text else 0
    cues += 1 if re.search(r"\b\d+(\.\d+)?\b", text) else 0
    if cues >= 2: score += 0.15; reasons.append("strong pattern cues")
    elif cues == 1: score += 0.07; reasons.append("hook cue")
    why = "; ".join(reasons) if reasons else "solid line"
    return round(score, 3), why, is_early

def build_hook_sets(hook_lines: List[str], segments: List[Dict[str, Any]]):
    seen = set(); intro, in_body = [], []
    for line in hook_lines:
        seg = find_segment_for_line(line, segments)
        if not seg: continue
        norm = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if norm in seen: continue
        seen.add(norm)
        score, why, is_early = score_hook(seg)
        item = {
            "file_id": seg["file_id"], "filename": seg["filename"],
            "text": seg["text"], "start": seg["start"], "end": seg["end"],
            "score": score, "is_early": is_early, "why": why
        }
        (intro if is_early else in_body).append(item)
    intro.sort(key=lambda x: x["score"], reverse=True)
    in_body.sort(key=lambda x: x["score"], reverse=True)
    return intro[:3], in_body[:3]

def map_lines(lines: List[str], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for line in lines:
        seg = find_segment_for_line(line, segments)
        if not seg: continue
        norm = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if norm in seen: continue
        seen.add(norm)
        out.append({
            "file_id": seg["file_id"], "filename": seg["filename"],
            "text": seg["text"], "start": seg["start"], "end": seg["end"]
        })
    return out[:3]

# -----------------------
# Routes
# -----------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": "0.6.0-step6"}

def _new_session_dir() -> Tuple[str, str]:
    sid = uuid.uuid4().hex
    sdir = os.path.join(BASE_DIR, sid)
    os.makedirs(sdir, exist_ok=True)
    return sid, sdir

@app.post("/process")
async def process_videos(
    videos: List[UploadFile] = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    # 0) Create session; persist originals; collect per-file meta
    sid, sdir = _new_session_dir()
    file_meta: Dict[str, Dict[str, str]] = {}
    all_segments_flat: List[Dict[str, Any]] = []
    full_transcript_parts: List[str] = []

    try:
        for i, uf in enumerate(videos):
            file_id = uuid.uuid4().hex[:8]
            ext = os.path.splitext(uf.filename or f"upload_{i}.mp4")[1] or ".mp4"
            src_path = os.path.join(sdir, f"{file_id}{ext}")
            data = await uf.read()
            with open(src_path, "wb") as f:
                f.write(data)

            # Per-file segmentation/transcription
            segs, trans = segment_by_silence_and_transcribe_from_path(src_path)
            # annotate with file_id/filename
            for s in segs:
                s["file_id"] = file_id
                s["filename"] = uf.filename or f"upload_{i}{ext}"
            all_segments_flat.extend(segs)
            full_transcript_parts.append(trans)

            file_meta[file_id] = {"filename": uf.filename or f"upload_{i}{ext}", "src_path": src_path}

        transcript = " ".join([p for p in full_transcript_parts if p])

        # 1) Classify
        buckets_raw = classify_text_buckets(transcript, all_segments_flat, features_csv, tone)

        # 2) Build swap-ready structures
        intro_hooks, in_body_hooks = build_hook_sets(buckets_raw["hook_lines"], all_segments_flat)
        features_mapped = map_lines(buckets_raw["feature_lines"], all_segments_flat)
        proof_mapped    = map_lines(buckets_raw["proof_lines"], all_segments_flat)
        cta_mapped      = map_lines(buckets_raw["cta_lines"], all_segments_flat)

        # 3) Stash session
        SESSIONS[sid] = {
            "files": file_meta,                     # file_id -> {filename, src_path}
            "segments": all_segments_flat,          # annotated with file_id
            "transcript": transcript,
            "buckets": {
                "hooks": {"intro_hooks": intro_hooks, "in_body_hooks": in_body_hooks},
                "features": features_mapped,
                "proof": proof_mapped,
                "cta": cta_mapped
            },
            "tone": tone,
            "features_csv": features_csv,
            "product_link": product_link
        }

        # 4) Default draft indices
        default_hook_src = "intro_hooks" if intro_hooks else ("in_body_hooks" if in_body_hooks else None)
        default_draft = {
            "hook_source": default_hook_src,
            "hook_index": 0 if default_hook_src else None,
            "feature_index": 0 if features_mapped else None,
            "proof_index": 0 if proof_mapped else None,
            "cta_index": 0 if cta_mapped else None
        }

        # 5) Cut plan (no rendering yet): Hook → Feature → Proof → CTA
        cut_plan: List[Dict[str, Any]] = []
        def add(role, bucket, idx):
            if bucket and idx is not None and idx < len(bucket):
                seg = bucket[idx]
                cut_plan.append({
                    "role": role,
                    "file_id": seg["file_id"],
                    "filename": seg["filename"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                })
        if default_hook_src:
            add("hook", SESSIONS[sid]["buckets"]["hooks"][default_hook_src], 0)
        add("feature", features_mapped, 0)
        add("proof", proof_mapped, 0)
        add("cta", cta_mapped, 0)

        return JSONResponse({
            "ok": True,
            "session_id": sid,
            "files": [{"file_id": fid, "filename": meta["filename"]} for fid, meta in file_meta.items()],
            "transcript_chars": len(transcript),
            "segments": all_segments_flat,     # [{file_id, filename, start, end, text}, ...]
            "buckets": SESSIONS[sid]["buckets"],
            "default_draft": default_draft,
            "cut_plan": cut_plan               # swap-ready plan you can edit in UI
        })
    except Exception as e:
        try: shutil.rmtree(sdir, ignore_errors=True)
        except: pass
        raise HTTPException(status_code=502, detail=f"Processing error: {e}")

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    meta = SESSIONS.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    files = [{"file_id": fid, "filename": m["filename"]} for fid, m in meta["files"].items()]
    return {"ok": True, "session_id": session_id, "files": files, "buckets": meta["buckets"]}

# ---------- Optional export (already Step 7, but works with multi-file) ----------

def _extract_clip(src_path: str, start: float, end: float, out_path: str):
    dur = max(0.01, float(end) - float(start))
    vf = ensure_portrait_filter()
    run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",
        "-i", src_path,
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ])

def _concat_clips(clip_paths: List[str], out_path: str):
    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-r", "30",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ])
    try: os.remove(list_file)
    except: pass

@app.post("/export")
def export_video(
    session_id: str = Form(...),
    hook_source: str = Form(None),  # "intro_hooks" or "in_body_hooks"
    hook_index: int = Form(None),
    feature_index: int = Form(None),
    proof_index: int = Form(None),
    cta_index: int = Form(None),
    filename: str = Form("draft.mp4")
):
    meta = SESSIONS.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    files = meta["files"]; buckets = meta["buckets"]

    if hook_source is None:
        hook_source = "intro_hooks" if buckets["hooks"]["intro_hooks"] else ("in_body_hooks" if buckets["hooks"]["in_body_hooks"] else None)
    if hook_index is None and hook_source: hook_index = 0
    if feature_index is None and buckets["features"]: feature_index = 0
    if proof_index is None and buckets["proof"]: proof_index = 0
    if cta_index is None and buckets["cta"]: cta_index = 0

    chosen: List[Dict[str, Any]] = []
    def choose(bucket, idx):
        if bucket and idx is not None and 0 <= idx < len(bucket):
            chosen.append(bucket[idx])

    if hook_source: choose(buckets["hooks"][hook_source], hook_index)
    choose(buckets["features"], feature_index)
    choose(buckets["proof"], proof_index)
    choose(buckets["cta"], cta_index)

    if not chosen:
        raise HTTPException(status_code=400, detail="No valid segments to export")

    sdir = os.path.join(BASE_DIR, session_id)
    tmp_dir = os.path.join(sdir, "clips")
    os.makedirs(tmp_dir, exist_ok=True)
    clip_paths = []
    for i, seg in enumerate(chosen):
        src_path = files[seg["file_id"]]["src_path"]
        clip_path = os.path.join(tmp_dir, f"part_{i:02d}.mp4")
        _extract_clip(src_path, float(seg["start"]), float(seg["end"]), clip_path)
        clip_paths.append(clip_path)

    out_path = os.path.join(OUTPUT_DIR, f"{session_id}_{filename}")
    _concat_clips(clip_paths, out_path)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "segments_used": chosen,
        "output_path": out_path
    })
