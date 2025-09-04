from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os, tempfile, subprocess, json, re
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI()

# OpenAI client (requires OPENAI_API_KEY)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

BASE_DIR = "/tmp/script2clipshop"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------
# Utilities
# -----------------------

def run(cmd: List[str]):
    """Run a shell command and raise with stderr on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")


def extract_duration_seconds(audio_path: str) -> float:
    """Read total duration (seconds) using ffprobe."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return float((probe.stdout or "").strip())
    except:
        return 0.0


def ensure_portrait_filter():
    """Ensure portrait 1080x1920 scaling filter."""
    return "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"


# -----------------------
# Segmentation + Transcription
# -----------------------

def segment_by_silence_and_transcribe(video_bytes: bytes, filename: str, file_id: str):
    """
    1) Save upload to temp
    2) Extract mono 16k WAV
    3) Detect silences with ffmpeg silencedetect
    4) Cut spoken segments, transcribe each with Whisper
    5) Return segments=[{start,end,text}], transcript
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_in:
        tmp_in.write(video_bytes)
        in_path = tmp_in.name

    segments = []
    try:
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "audio.wav")
            run(["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", wav])

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
                    segments.append({
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "text": text,
                        "file_id": file_id,
                        "filename": filename
                    })

            transcript = " ".join(seg["text"] for seg in segments)
            return segments, transcript
    finally:
        try: os.remove(in_path)
        except: pass


# -----------------------
# Classification (text → buckets)
# -----------------------

def classify_text_buckets(transcript: str, segments: List[Dict[str, Any]],
                          features_csv: str, tone: str) -> Dict[str, Any]:
    seg_listing = []
    for seg in segments[:80]:
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
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
# Routes
# -----------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": "0.9.0-export-tolerant"}


@app.post("/process")
async def process_videos(
    videos: List[UploadFile] = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    import uuid
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(BASE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    all_segments, transcript = [], ""
    files_meta = {}

    for up in videos:
        file_id = uuid.uuid4().hex[:8]
        data = await up.read()
        src_path = os.path.join(session_dir, up.filename)
        with open(src_path, "wb") as f:
            f.write(data)
        segs, tr = segment_by_silence_and_transcribe(data, up.filename, file_id)
        for seg in segs:
            seg["file_id"] = file_id
            seg["filename"] = up.filename
        all_segments.extend(segs)
        transcript += " " + tr
        files_meta[file_id] = {"src_path": src_path, "filename": up.filename}

    buckets_raw = classify_text_buckets(transcript, all_segments, features_csv, tone)

    SESSIONS[session_id] = {
        "files": files_meta,
        "segments": all_segments,
        "buckets": {
            "hooks": {"intro_hooks": [], "in_body_hooks": []},
            "features": [{"text": t, "file_id": all_segments[0]["file_id"], "filename": all_segments[0]["filename"], "start": all_segments[0]["start"], "end": all_segments[0]["end"]} for t in buckets_raw["feature_lines"]],
            "proof":    [{"text": t, "file_id": all_segments[0]["file_id"], "filename": all_segments[0]["filename"], "start": all_segments[0]["start"], "end": all_segments[0]["end"]} for t in buckets_raw["proof_lines"]],
            "cta":      [{"text": t, "file_id": all_segments[0]["file_id"], "filename": all_segments[0]["filename"], "start": all_segments[0]["start"], "end": all_segments[0]["end"]} for t in buckets_raw["cta_lines"]],
        }
    }

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "files": [{"file_id": k, "filename": v["filename"]} for k, v in files_meta.items()],
        "transcript_chars": len(transcript),
        "segments": all_segments,
        "buckets": SESSIONS[session_id]["buckets"]
    })


# -----------------------
# Tolerant Export
# -----------------------

@app.post("/export")
def export_video(
    session_id: str = Form(...),
    hook_source: str = Form(None),
    hook_index: int = Form(None),
    feature_index: int = Form(None),
    proof_index: int = Form(None),
    cta_index: int = Form(None),
    filename: str = Form("draft.mp4")
):
    meta = SESSIONS.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    files = meta["files"]
    buckets = meta["buckets"]

    def norm_idx(v):
        try:
            if v is None: return None
            v = int(v)
            return v if v >= 0 else None
        except Exception:
            return None

    hook_index   = norm_idx(hook_index)
    feature_index= norm_idx(feature_index)
    proof_index  = norm_idx(proof_index)
    cta_index    = norm_idx(cta_index)

    if not hook_source:
        hook_source = "intro_hooks" if buckets["hooks"]["intro_hooks"] else (
            "in_body_hooks" if buckets["hooks"]["in_body_hooks"] else None
        )

    chosen: List[Dict[str, Any]] = []

    def choose(arr, idx):
        if not arr: return
        if idx is None:
            idx = 0
        if 0 <= idx < len(arr):
            chosen.append(arr[idx])

    if hook_source:
        choose(buckets["hooks"].get(hook_source, []), hook_index)
    choose(buckets.get("features", []), feature_index)
    choose(buckets.get("proof", []), proof_index)
    choose(buckets.get("cta", []), cta_index)

    if not chosen:
        earliest = []
        for seg in sorted(meta["segments"], key=lambda s: float(s.get("start", 0.0))):
            try:
                st, en = float(seg["start"]), float(seg["end"])
                if en - st >= 0.10:
                    earliest.append({
                        "file_id": seg["file_id"],
                        "filename": seg.get("filename", ""),
                        "text": seg["text"],
                        "start": st, "end": en
                    })
                if len(earliest) >= 4:
                    break
            except Exception:
                continue
        chosen = earliest

    valid: List[Dict[str, Any]] = []
    seen = set()
    for seg in chosen:
        try:
            st, en = float(seg["start"]), float(seg["end"])
            if en - st < 0.10:
                continue
            fid = seg.get("file_id")
            if not fid or fid not in files:
                continue
            src_path = files[fid]["src_path"]
            if not os.path.isfile(src_path):
                continue
            key = (fid, round(st, 2), round(en, 2))
            if key in seen:
                continue
            seen.add(key)
            seg["_src_path"], seg["_start"], seg["_end"] = src_path, st, en
            valid.append(seg)
        except Exception:
            continue

    if not valid:
        raise HTTPException(status_code=400, detail="No valid segments to export (all were too short/invalid).")

    sdir = os.path.join(BASE_DIR, session_id)
    tmp_dir = os.path.join(sdir, "clips")
    os.makedirs(tmp_dir, exist_ok=True)

    vf = ensure_portrait_filter()
    clip_paths: List[str] = []

    for i, seg in enumerate(valid):
        out = os.path.join(tmp_dir, f"part_{i:02d}.mp4")
        dur = max(0.01, seg["_end"] - seg["_start"])
        try:
            run([
                "ffmpeg", "-y",
                "-ss", f"{seg['_start']:.3f}",
                "-t", f"{dur:.3f}",
                "-i", seg["_src_path"],
                "-vf", vf,
                "-r", "30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                out
            ])
            clip_paths.append(out)
        except Exception:
            continue

    if not clip_paths:
        raise HTTPException(status_code=400, detail="All segment cuts failed.")

    out_path = os.path.join(OUTPUT_DIR, f"{session_id}_{filename}")
    lst = out_path + ".txt"
    with open(lst, "w") as f:
        for p in clip_paths:
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
        raise HTTPException(status_code=500, detail=f"Concat failed: {e}")
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
    })


@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    out_path = os.path.join(OUTPUT_DIR, f"{session_id}_{filename}")
    if not os.path.isfile(out_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(out_path, media_type="video/mp4", filename=filename)
