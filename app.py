from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from typing import List, Dict, Any, Optional, Tuple
import os, io, re, json, shutil, uuid, secrets, tempfile, subprocess
from datetime import datetime
from openai import OpenAI

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SERVICE_NAME = "script2clipshop-worker"
SERVICE_VERSION = "1.1.0-disk-sessions"
SESS_ROOT = "/tmp/s2cs_sessions"            # sessions persist here
MAX_SEGMENTS_PER_FILE = 80                   # safety bound for classifier prompt
EARLY_SEC = 15.0                             # early hook heuristic
FFMPEG_VF = "scale='min(720,iw)':-2"         # cap width at 720 for stability on Render
FFMPEG_PRESET = "ultrafast"
FFMPEG_CRF = "26"                            # lighter encode to avoid memory spikes

os.makedirs(SESS_ROOT, exist_ok=True)

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = FastAPI(title="Script2ClipShop Worker", version=SERVICE_VERSION)


# -----------------------------------------------------------------------------
# Helpers (disk sessions, shell, ffmpeg)
# -----------------------------------------------------------------------------
def run(cmd: List[str]):
    """Run a shell command and raise with stderr on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def new_session_id() -> str:
    return secrets.token_hex(16)

def new_file_id() -> str:
    return secrets.token_hex(4)

def sess_dir(session_id: str) -> str:
    return ensure_dir(os.path.join(SESS_ROOT, session_id))

def sess_files_dir(session_id: str) -> str:
    return ensure_dir(os.path.join(sess_dir(session_id), "files"))

def sess_tmp_dir(session_id: str) -> str:
    return ensure_dir(os.path.join(sess_dir(session_id), "tmp"))

def sess_exports_dir(session_id: str) -> str:
    return ensure_dir(os.path.join(sess_dir(session_id), "exports"))

def sess_json_path(session_id: str) -> str:
    return os.path.join(sess_dir(session_id), "session.json")

def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_duration_seconds(audio_or_video_path: str) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_or_video_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return float((probe.stdout or "").strip())
    except:
        return 0.0


# -----------------------------------------------------------------------------
# Segmentation + Whisper transcription
# -----------------------------------------------------------------------------
def segment_by_silence_and_transcribe(video_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    1) Convert to 16k mono WAV
    2) Detect silences (ffmpeg silencedetect)
    3) Cut spoken segments (2–60s), transcribe each with Whisper
    4) Return segments=[{start,end,text}], transcript
    """
    segments: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        run(["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", wav])

        # Tune silence gate; 0.35s gaps tends to separate sentences well
        proc = subprocess.run(
            ["ffmpeg", "-i", wav, "-af", "silencedetect=noise=-30dB:d=0.35", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        log = proc.stderr or ""
        starts = [0.0]
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

        # Trim and whisper
        for s, e in zip(starts, ends):
            dur = max(0.0, e - s)
            if dur < 0.25:
                continue
            # Keep cuts reasonably sized for Whisper stability
            if dur > 60.0:
                e = s + 60.0
                dur = 60.0

            chunk = os.path.join(td, f"seg_{int(s*1000)}.wav")
            run([
                "ffmpeg", "-y",
                "-ss", f"{s:.3f}",
                "-t", f"{dur:.3f}",
                "-i", wav,
                "-ac", "1", "-ar", "16000",
                chunk
            ])
            with open(chunk, "rb") as f:
                try:
                    tr = client.audio.transcriptions.create(model="whisper-1", file=f)
                    text = (getattr(tr, "text", "") or "").strip()
                except Exception as ex:
                    text = ""
            if text:
                segments.append({"start": round(s, 3), "end": round(e, 3), "text": text})

    transcript = " ".join(seg["text"] for seg in segments)
    return segments, transcript


# -----------------------------------------------------------------------------
# Classification (text → buckets)
# -----------------------------------------------------------------------------
def classify_text_buckets(transcript: str, segments: List[Dict[str, Any]],
                          features_csv: str, tone: str) -> Dict[str, Any]:
    """
    Extract ONLY verbatim lines into:
    - hook_lines, feature_lines, proof_lines, cta_lines (≤6 each)
    """
    seg_listing = []
    for seg in segments[:MAX_SEGMENTS_PER_FILE]:
        seg_listing.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")
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

    data: Dict[str, Any] = {}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=900,
        )
        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
    except Exception:
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


# -----------------------------------------------------------------------------
# Hook scoring/mapping (swap-ready)
# -----------------------------------------------------------------------------
def find_segment_for_line(line: str, segments: List[Dict[str, Any]]):
    needle = re.sub(r"\s+", " ", line.strip().lower())
    best = None
    best_overlap = 0
    for seg in segments:
        hay = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if needle and needle in hay:
            return seg
        # loose overlap fallback
        n_tokens = set(needle.split())
        h_tokens = set(hay.split())
        overlap = len(n_tokens & h_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg
    return best

def score_hook(seg: Dict[str, Any]) -> Tuple[float, str, bool]:
    start = float(seg["start"]); end = float(seg["end"])
    dur = max(0.01, end - start)
    text = seg["text"]
    score = 0.5; reasons = []
    is_early = start <= EARLY_SEC
    if is_early:
        score += 0.25; reasons.append("early (≤15s)")
    if 2.0 <= dur <= 7.0:
        score += 0.2; reasons.append("snackable length")
    cues = 0
    cues += 1 if re.search(r"\b(stop|wait|hold|don’t|don't)\b", text, re.I) else 0
    cues += 1 if "?" in text else 0
    cues += 1 if re.search(r"\b\d+(\.\d+)?\b", text) else 0
    if cues >= 2:
        score += 0.15; reasons.append("strong pattern cues")
    elif cues == 1:
        score += 0.07; reasons.append("hook cue")
    why = "; ".join(reasons) if reasons else "solid line"
    return round(score, 3), why, is_early

def build_hook_sets(hook_lines: List[str], segments: List[Dict[str, Any]]):
    seen = set(); intro = []; in_body = []
    for line in hook_lines:
        seg = find_segment_for_line(line, segments)
        if not seg: continue
        norm = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if norm in seen: continue
        seen.add(norm)
        score, why, is_early = score_hook(seg)
        item = {
            "text": seg["text"], "start": seg["start"], "end": seg["end"],
            "score": score, "is_early": is_early
        }
        (intro if is_early else in_body).append(item)
    intro.sort(key=lambda x: x["score"], reverse=True)
    in_body.sort(key=lambda x: x["score"], reverse=True)
    return intro[:3], in_body[:3]


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.get("/", response_class=PlainTextResponse)
def root():
    return f"{SERVICE_NAME} {SERVICE_VERSION}"

@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE_NAME, "version": SERVICE_VERSION}

@app.post("/process")
async def process_videos(
    videos: List[UploadFile] = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    """
    Accept one or more videos, produce transcript + segments + buckets.
    Persist everything to disk under /tmp/s2cs_sessions/<session_id>.
    Return a session_id for export/download later.
    """
    session_id = new_session_id()
    sdir = sess_dir(session_id)
    fdir = sess_files_dir(session_id)
    tdir = sess_tmp_dir(session_id)

    saved_files: List[Dict[str, Any]] = []
    all_segments: List[Dict[str, Any]] = []
    full_transcript_parts: List[str] = []

    # Save uploads to disk
    for up in videos:
        file_id = new_file_id()
        # Sanitize filename
        base = os.path.basename(up.filename or f"video_{file_id}.mp4")
        if not re.search(r"\.(mp4|mov|m4v|mpg|mpeg|webm|mkv)$", base, re.I):
            base = f"{base}.mp4"
        dst = os.path.join(fdir, f"{file_id}_{base}")
        ensure_dir(os.path.dirname(dst))
        with open(dst, "wb") as out:
            shutil.copyfileobj(up.file, out)
        # Segment + transcribe
        try:
            segs, transcript = segment_by_silence_and_transcribe(dst)
        except Exception as ex:
            raise HTTPException(status_code=502, detail=f"Segmentation/Transcription error: {ex}")

        # attach file_id, filename to each segment
        for seg in segs:
            seg["file_id"] = file_id
            seg["filename"] = base
        all_segments.extend(segs)
        full_transcript_parts.append(transcript)
        saved_files.append({"file_id": file_id, "filename": base, "path": dst})

    full_transcript = " ".join(x for x in full_transcript_parts if x).strip()

    # Classify → buckets
    try:
        buckets_raw = classify_text_buckets(full_transcript, all_segments, features_csv, tone)
    except Exception as ex:
        buckets_raw = {"hook_lines": [], "feature_lines": [], "proof_lines": [], "cta_lines": []}

    # Hook mapping + scoring
    intro_hooks, in_body_hooks = build_hook_sets(buckets_raw.get("hook_lines", []), all_segments)

    # Convert feature/proof/cta lines into timestamped segments by best match
    def map_lines(lines: List[str]) -> List[Dict[str, Any]]:
        mapped = []
        for line in (lines or []):
            seg = find_segment_for_line(line, all_segments)
            if seg:
                mapped.append({
                    "file_id": seg["file_id"], "filename": seg["filename"],
                    "text": seg["text"], "start": seg["start"], "end": seg["end"]
                })
        return mapped

    features_list = map_lines(buckets_raw.get("feature_lines", []))
    proof_list = map_lines(buckets_raw.get("proof_lines", []))
    cta_list = map_lines(buckets_raw.get("cta_lines", []))

    final_buckets = {
        "hooks": {
            "intro_hooks": intro_hooks,
            "in_body_hooks": in_body_hooks
        },
        "features": features_list,
        "proof": proof_list,
        "cta": cta_list
    }

    default_draft = {
        "hook_source": "intro_hooks" if intro_hooks else ("in_body_hooks" if in_body_hooks else None),
        "hook_index": 0 if (intro_hooks or in_body_hooks) else None,
        "feature_index": 0 if features_list else None,
        "proof_index": 0 if proof_list else None,
        "cta_index": 0 if cta_list else None
    }

    # Persist to disk
    session_obj = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
        "files": saved_files,            # includes disk paths
        "segments": all_segments,
        "transcript": full_transcript,
        "buckets": final_buckets,
        "default_draft": default_draft
    }
    save_json(sess_json_path(session_id), session_obj)

    # Response (no disk paths)
    resp_files = [{"file_id": f["file_id"], "filename": f["filename"]} for f in saved_files]
    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "files": resp_files,
        "transcript_chars": len(full_transcript),
        "segments": all_segments,
        "buckets": final_buckets
    })


@app.post("/export")
async def export_video(
    session_id: str = Form(...),
    filename: str = Form("draft.mp4"),
    hook_source: Optional[str] = Form(None),
    hook_index: Optional[int] = Form(None),
    feature_index: Optional[int] = Form(None),
    proof_index: Optional[int] = Form(None),
    cta_index: Optional[int] = Form(None),
):
    """
    Build MP4 from selected segments. If any bucket/index is missing, we gracefully skip it.
    Output saved to /tmp/s2cs_sessions/<session_id>/exports/<filename>
    """
    try:
        sjson = load_json(sess_json_path(session_id))
    except Exception:
        return {"ok": False, "reason": "session not found"}

    buckets = sjson.get("buckets", {}) or {}
    files_meta = {f["file_id"]: f for f in sjson.get("files", [])}

    # Helper: pick segment safely
    def pick_hook() -> Optional[Dict[str, Any]]:
        nonlocal hook_source, hook_index
        hooks = buckets.get("hooks", {})
        if not hook_source:
            hook_source = "intro_hooks" if hooks.get("intro_hooks") else "in_body_hooks"
        lst = hooks.get(hook_source or "", []) or []
        idx = hook_index if isinstance(hook_index, int) else 0
        return lst[idx] if idx is not None and 0 <= idx < len(lst) else (lst[0] if lst else None)

    def pick_from_list(key: str, idx_opt: Optional[int]) -> Optional[Dict[str, Any]]:
        lst = buckets.get(key, []) or []
        idx = idx_opt if isinstance(idx_opt, int) else 0
        return lst[idx] if 0 <= idx < len(lst) else (lst[0] if lst else None)

    picks = []
    h = pick_hook()
    if h: picks.append(h)
    f = pick_from_list("features", feature_index)
    if f: picks.append(f)
    p = pick_from_list("proof", proof_index)
    if p: picks.append(p)
    c = pick_from_list("cta", cta_index)
    if c: picks.append(c)

    if not picks:
        # Fallback: pick earliest segment across all if available
        segs = sjson.get("segments", [])
        if segs:
            earliest = sorted(segs, key=lambda x: x.get("start", 0.0))[0]
            picks = [{
                "file_id": earliest.get("file_id"),
                "filename": earliest.get("filename"),
                "text": earliest.get("text"),
                "start": earliest.get("start"),
                "end": earliest.get("end"),
            }]
        else:
            return {"ok": False, "reason": "no valid segments to export"}

    # Cut each pick to a temporary mp4
    tdir = sess_tmp_dir(session_id)
    clips = []
    try:
        for i, seg in enumerate(picks):
            fid = seg.get("file_id")
            meta = files_meta.get(fid)
            if not meta:
                continue
            src = meta["path"]
            start = float(seg["start"]); end = float(seg["end"])
            dur = max(0.01, end - start)

            clip_path = os.path.join(tdir, f"clip_{i:02d}.mp4")
            # Re-encode each cut to ensure concat-compatible streams
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-t", f"{dur:.3f}",
                "-i", src,
                "-vf", FFMPEG_VF,
                "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF,
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                clip_path
            ]
            run(cmd)
            clips.append(clip_path)

        if not clips:
            return {"ok": False, "reason": "no cuttable segments"}

        # Concat
        list_txt = os.path.join(tdir, "concat.txt")
        with open(list_txt, "w", encoding="utf-8") as f:
            for cpath in clips:
                f.write(f"file '{cpath}'\n")

        out_dir = sess_exports_dir(session_id)
        out_path = os.path.join(out_dir, filename)

        # Re-encode final for guaranteed compatibility
        cmd2 = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_txt,
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", "23",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            out_path
        ]
        run(cmd2)

        # Save light export manifest for debugging
        manifest = {
            "session_id": session_id,
            "output": out_path,
            "segments_used": picks
        }
        save_json(os.path.join(out_dir, f"{filename}.json"), manifest)

        return {
            "ok": True,
            "message": "export complete",
            "session_id": session_id,
            "filename": filename,
            "download": f"/download/{session_id}/{filename}",
            "segments_used": picks
        }
    except Exception as ex:
        return {"ok": False, "error": f"{ex}"}


@app.get("/download/{session_id}/{filename}")
def download_export(session_id: str, filename: str):
    out_path = os.path.join(sess_exports_dir(session_id), filename)
    if not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(out_path, media_type="video/mp4", filename=filename)


# Optional: inspect saved session JSON quickly
@app.get("/session/{session_id}")
def get_session(session_id: str):
    path = sess_json_path(session_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="session not found")
    return load_json(path)
