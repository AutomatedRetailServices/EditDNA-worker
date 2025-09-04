from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os, tempfile, subprocess, json, re, uuid
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI

# =========================
# App & Config
# =========================
app = FastAPI()

# OpenAI client (requires OPENAI_API_KEY in env)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# In-memory session store (dev). For prod, move to Redis/DB.
SESSIONS: Dict[str, Dict[str, Any]] = {}

BASE_DIR = os.environ.get("S2CS_BASE_DIR", "/tmp/s2cs")
OUTPUT_DIR = os.environ.get("S2CS_OUTPUT_DIR", "/tmp/s2cs/out")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Utils
# =========================
def run(cmd: List[str]):
    """Run a shell command and raise with stderr on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")

def extract_duration_seconds(media_path: str) -> float:
    """Read total duration (seconds) using ffprobe."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return float((probe.stdout or "").strip())
    except:
        return 0.0

def ensure_portrait_filter() -> str:
    """
    Scale to max height 1920 (keep aspect), then pad to 1080x1920 centered.
    Good default for TikTok-style vertical.
    """
    return "scale=-2:1920,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"

# =========================
# Segmentation + Transcription
# =========================
def segment_by_silence_and_transcribe_from_path(src_path: str, file_id: str, filename: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    1) Extract mono 16k WAV
    2) Detect silences with ffmpeg silencedetect
    3) Cut voiced segments, transcribe each with Whisper
    4) Return [{start,end,text,file_id,filename}], full transcript
    """
    segments: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        run(["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav])

        # Detect silences
        proc = subprocess.run(
            ["ffmpeg", "-i", wav, "-af", "silencedetect=noise=-30dB:d=0.35", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        log = proc.stderr or ""

        starts: List[float] = [0.0]
        ends: List[float] = []
        for m in re.finditer(r"silence_start:\s*([0-9.]+)", log):
            try: ends.append(float(m.group(1)))
            except: pass
        for m in re.finditer(r"silence_end:\s*([0-9.]+)", log):
            try: starts.append(float(m.group(1)))
            except: pass

        total = extract_duration_seconds(wav)
        if not ends or (ends and ends[-1] < total):
            ends.append(total)

        # Cut & transcribe
        for s, e in zip(starts, ends):
            if (e - s) < 0.25:   # skip ultra-short blips
                continue
            chunk = os.path.join(td, f"seg_{int(s*1000)}.wav")
            run([
                "ffmpeg", "-y",
                "-ss", f"{s:.3f}",
                "-t", f"{(e - s):.3f}",
                "-i", wav, "-ac", "1", "-ar", "16000",
                chunk
            ])
            with open(chunk, "rb") as f:
                tr = client.audio.transcriptions.create(model="whisper-1", file=f)
            text = (getattr(tr, "text", "") or "").strip()
            if text:
                segments.append({
                    "file_id": file_id,
                    "filename": filename,
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "text": text
                })

    transcript = " ".join(seg["text"] for seg in segments)
    return segments, transcript

# =========================
# Classification (text → buckets)
# =========================
def classify_text_buckets(transcript: str, segments_flat: List[Dict[str, Any]],
                          features_csv: str, tone: str) -> Dict[str, Any]:
    # Compact segment listing for context
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
        f"Transcript:\n{transcript}\n\n"
        f"Segments:\n{seg_text}"
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

# =========================
# Mapping helpers
# =========================
def find_segment_for_line(line: str, segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Case-insensitive substring match to map a candidate line back to a segment."""
    needle = re.sub(r"\s+", " ", (line or "").strip().lower())
    for seg in segments:
        hay = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if needle and needle in hay:
            return seg
    # fallback: token overlap best match
    n_tokens = set(needle.split())
    best, best_overlap = None, 0
    for seg in segments:
        h_tokens = set(re.sub(r"\s+", " ", seg["text"].lower()).split())
        overlap = len(n_tokens & h_tokens)
        if overlap > best_overlap:
            best_overlap, best = overlap, seg
    return best

def map_lines(lines: List[str], segments: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    out = []
    for line in lines:
        seg = find_segment_for_line(line, segments)
        if seg:
            out.append({
                "file_id": seg["file_id"],
                "filename": seg["filename"],
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"]
            })
        if len(out) >= limit:
            break
    return out

def build_hook_sets(hook_lines: List[str], segments: List[Dict[str, Any]]):
    """Score hooks & split into intro vs in-body (simple heuristic)."""
    EARLY_SEC = 15.0
    intro, in_body = [], []
    for line in hook_lines:
        seg = find_segment_for_line(line, segments)
        if not seg: continue
        start, end, text = float(seg["start"]), float(seg["end"]), seg["text"]
        dur = max(0.01, end - start)
        score = 0.5
        reasons = []
        is_early = start <= EARLY_SEC
        if is_early: score += 0.25; reasons.append("early (≤15s)")
        if 2.0 <= dur <= 7.0: score += 0.2; reasons.append("snackable")
        if "?" in text: score += 0.1; reasons.append("question")
        item = {
            "file_id": seg["file_id"],
            "filename": seg["filename"],
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"],
            "score": round(score, 3),
            "is_early": is_early,
            "why": "; ".join(reasons) if reasons else "solid"
        }
        (intro if is_early else in_body).append(item)
    intro.sort(key=lambda x: x["score"], reverse=True)
    in_body.sort(key=lambda x: x["score"], reverse=True)
    return intro[:3], in_body[:3]

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": "0.9.5-export-never-error"}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    meta = SESSIONS.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    # return a trimmed view (no disk paths)
    return {
        "ok": True,
        "files": [{"file_id": k, "filename": v.get("filename")} for k, v in meta.get("files", {}).items()],
        "segments_count": len(meta.get("segments", [])),
        "buckets": meta.get("buckets", {})
    }

@app.post("/process")
async def process_videos(
    videos: List[UploadFile] = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(BASE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    files_meta: Dict[str, Dict[str, Any]] = {}
    all_segments: List[Dict[str, Any]] = []
    transcript_parts: List[str] = []

    # Save & process each upload
    for i, uf in enumerate(videos):
        file_id = uuid.uuid4().hex[:8]
        filename = uf.filename or f"upload_{i}.mp4"
        src_path = os.path.join(session_dir, filename)
        data = await uf.read()
        with open(src_path, "wb") as f:
            f.write(data)

        segs, trans = segment_by_silence_and_transcribe_from_path(src_path, file_id, filename)
        all_segments.extend(segs)
        if trans: transcript_parts.append(trans)

        files_meta[file_id] = {"filename": filename, "src_path": src_path}

    full_transcript = " ".join([t for t in transcript_parts if t])

    # Classify → map to segments
    buckets_raw = classify_text_buckets(full_transcript, all_segments, features_csv, tone)
    intro_hooks, in_body_hooks = build_hook_sets(buckets_raw["hook_lines"], all_segments)
    features_mapped = map_lines(buckets_raw["feature_lines"], all_segments)
    proof_mapped    = map_lines(buckets_raw["proof_lines"], all_segments)
    cta_mapped      = map_lines(buckets_raw["cta_lines"], all_segments)

    # Persist session
    SESSIONS[session_id] = {
        "files": files_meta,
        "segments": all_segments,
        "transcript": full_transcript,
        "buckets": {
            "hooks": {"intro_hooks": intro_hooks, "in_body_hooks": in_body_hooks},
            "features": features_mapped,
            "proof": proof_mapped,
            "cta": cta_mapped
        }
    }

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "files": [{"file_id": fid, "filename": meta["filename"]} for fid, meta in files_meta.items()],
        "transcript_chars": len(full_transcript),
        "segments": all_segments,
        "buckets": SESSIONS[session_id]["buckets"]
    })

# =========================
# Forgiving Export (never 5xx)
# =========================
@app.post("/export")
def export_video_forgiving(
    session_id: str = Form(...),
    hook_source: str = Form(None),
    hook_index: int = Form(None),
    feature_index: int = Form(None),
    proof_index: int = Form(None),
    cta_index: int = Form(None),
    filename: str = Form("draft.mp4")
):
    """
    Never-5xx exporter:
    - Blank/None/"" fields are fine.
    - Missing buckets are skipped automatically.
    - If all buckets are empty, falls back to earliest 3–4 voiced segments.
    - If any step fails, returns 200 with ok:false + reason (no hard crashes).
    """
    try:
        meta = SESSIONS.get(session_id)
        if not meta:
            return JSONResponse({"ok": False, "reason": "session not found"}, status_code=200)

        files   = meta.get("files", {})
        buckets = meta.get("buckets", {})
        segs    = meta.get("segments", [])

        def to_idx(v):
            try:
                if v is None or v == "": return None
                v = int(v)
                return v if v >= 0 else None
            except: return None

        hook_index    = to_idx(hook_index)
        feature_index = to_idx(feature_index)
        proof_index   = to_idx(proof_index)
        cta_index     = to_idx(cta_index)

        if not hook_source or hook_source.strip() == "":
            hook_source = "intro_hooks" if buckets.get("hooks", {}).get("intro_hooks") else (
                "in_body_hooks" if buckets.get("hooks", {}).get("in_body_hooks") else None
            )

        picked: List[Dict[str, Any]] = []

        def pick(arr, idx):
            if not arr: return
            i = 0 if idx is None else idx
            if 0 <= i < len(arr):
                picked.append(arr[i])

        if hook_source:
            pick(buckets.get("hooks", {}).get(hook_source, []), hook_index)
        pick(buckets.get("features", []), feature_index)
        pick(buckets.get("proof", []),    proof_index)
        pick(buckets.get("cta", []),      cta_index)

        # Fallback: earliest natural segments if nothing picked
        if not picked:
            earliest = []
            for s in sorted(segs, key=lambda x: float(x.get("start", 0.0))):
                try:
                    st, en = float(s["start"]), float(s["end"])
                    if en - st >= 0.10:
                        earliest.append({
                            "file_id": s.get("file_id"),
                            "filename": s.get("filename", ""),
                            "text": s.get("text", ""),
                            "start": st, "end": en
                        })
                    if len(earliest) >= 4:
                        break
                except:
                    continue
            picked = earliest

        # Validate picks; dedupe; map to source paths
        valid, seen = [], set()
        for s in picked:
            try:
                st, en = float(s["start"]), float(s["end"])
                if en - st < 0.10:   # lowered threshold
                    continue
                fid = s.get("file_id")
                fmeta = files.get(fid)
                if not fmeta:
                    continue
                src = fmeta.get("src_path")
                if not src or not os.path.isfile(src):
                    continue
                key = (fid, round(st, 2), round(en, 2))
                if key in seen:
                    continue
                seen.add(key)
                s["_src"], s["_st"], s["_en"] = src, st, en
                valid.append(s)
            except:
                continue

        if not valid:
            return JSONResponse({"ok": False, "reason": "no valid segments to cut"}, status_code=200)

        # Prepare cut dir & portrait filter
        sdir = os.path.join(BASE_DIR, session_id)
        os.makedirs(sdir, exist_ok=True)
        clipdir = os.path.join(sdir, "clips")
        os.makedirs(clipdir, exist_ok=True)
        vf = ensure_portrait_filter()

        # Cut each part
        parts = []
        for i, s in enumerate(valid):
            outp = os.path.join(clipdir, f"part_{i:02d}.mp4")
            dur = max(0.01, s["_en"] - s["_st"])
            try:
                run([
                    "ffmpeg", "-y",
                    "-ss", f"{s['_st']:.3f}",
                    "-t",  f"{dur:.3f}",
                    "-i",  s["_src"],
                    "-vf", vf,
                    "-r",  "30",
                    "-c:v","libx264","-preset","veryfast","-crf","23",
                    "-c:a","aac","-b:a","128k",
                    outp
                ])
                parts.append(outp)
            except:
                # skip failed cuts
                continue

        if not parts:
            return JSONResponse({"ok": False, "reason": "ffmpeg cuts failed"}, status_code=200)

        # Concat
        out_path = os.path.join(OUTPUT_DIR, f"{session_id}_{filename}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        lst = out_path + ".txt"
        with open(lst, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")

        try:
            run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", lst,
                "-c:v", "libx264", "-r", "30",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                out_path
            ])
        except Exception as e:
            return JSONResponse({"ok": False, "reason": f"concat failed: {e}"}, status_code=200)
        finally:
            try: os.remove(lst)
            except: pass

        return JSONResponse({
            "ok": True,
            "session_id": session_id,
            "segments_used": [
                {"file_id": s.get("file_id"), "start": s.get("start"), "end": s.get("end"), "text": s.get("text")}
                for s in valid
            ],
            "output_path": out_path
        }, status_code=200)

    except Exception as e:
        # Final safety net – never 5xx
        return JSONResponse({"ok": False, "reason": f"unexpected: {str(e)}"}, status_code=200)

# =========================
# Download
# =========================
@app.get("/download/{session_id}/{filename}")
def download_export(session_id: str, filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, f"{session_id}_{safe_name}")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found. Make sure /export succeeded.")
    return FileResponse(path, media_type="video/mp4", filename=safe_name)
