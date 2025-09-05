import os
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI  # v1 SDK

app = FastAPI()

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set")

# Models (adjust if you prefer others)
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1")        # transcription
CLS_MODEL = os.getenv("CLS_MODEL", "gpt-4o-mini")      # classification

client = OpenAI(api_key=OPENAI_API_KEY)

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "1.2.0-stitch+analyze+classify"

# ---- Helpers ----
def sess_dir(session_id: str) -> Path:
    p = SESS_ROOT / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())

def run(cmd: list) -> str:
    """Run a shell command; raise on non-zero exit and return stdout."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def safe_filename(name: str, default: str) -> str:
    s = "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_", ".",)).strip()
    return s or default

# ---- Routes ----
@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": VERSION}

@app.post("/process")
async def process_video(
    videos: List[UploadFile] = File(...),
    tone: str = Form("casual"),
    features_csv: str = Form(""),
    product_link: str = Form("")
):
    # Save uploaded files into a unique session folder and record metadata
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)

    files_meta = []
    file_paths = {}

    for up in videos:
        fid = uuid.uuid4().hex[:8]
        orig_name = up.filename or f"{fid}.mp4"
        dst = sd / orig_name
        with dst.open("wb") as w:
            shutil.copyfileobj(up.file, w)
        files_meta.append({"file_id": fid, "filename": orig_name})
        file_paths[fid] = str(dst)

    session_json = {
        "session_id": session_id,
        "files": files_meta,
        "file_paths": file_paths,
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
    }
    save_json(sd / "session.json", session_json)

    return JSONResponse({"ok": True, "session_id": session_id, "files": files_meta})

# -------- Export (low-RAM) --------
@app.post("/export")
def export_video(
    session_id: str = Form(...),
    filename: str = Form("draft.mp4"),
    start_seconds: float = Form(0.0),
    duration_seconds: float = Form(10.0)
):
    # Trim the first uploaded video to a playable MP4 using a low-RAM strategy
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")

    session = load_json(meta_path)
    files = session.get("files") or []
    file_paths = session.get("file_paths") or {}
    if not files:
        raise HTTPException(status_code=400, detail="no files in session")

    first = files[0]
    fid = first["file_id"]
    src_path = file_paths.get(fid)
    if not src_path or not Path(src_path).exists():
        raise HTTPException(status_code=404, detail="source video not found")

    safe_name = safe_filename(filename, "draft.mp4")
    out_path = sd / safe_name
    start_s = float(start_seconds)
    dur_s = max(0.1, float(duration_seconds))

    # 1) Stream-copy (no re-encode) → minimal memory
    try:
        run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-t", f"{dur_s:.3f}",
            "-i", src_path,
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path)
        ])
    except Exception as e1:
        print("FFMPEG stream-copy failed:", e1)
        # 2) Very light re-encode (720p, 1 thread)
        try:
            run([
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_s:.3f}",
                "-t", f"{dur_s:.3f}",
                "-i", src_path,
                "-vf", "scale=720:-2:flags=lanczos",
                "-r", "30",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
                "-threads", "1",
                "-movflags", "+faststart",
                str(out_path)
            ])
        except Exception as e2:
            print("FFMPEG re-encode failed:", e2)
            # 3) Last resort: copy full source file so there is always a download
            shutil.copy(src_path, out_path)

    return JSONResponse({
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": safe_name,
        "download": f"/download/{session_id}/{safe_name}"
    })

# -------- Stitch (concat multiple segments) --------
@app.post("/stitch")
def stitch_video(
    session_id: str = Form(...),
    manifest: str = Form(...),  # JSON string with {"segments":[...], "fps":30, "scale":720}
    filename: str = Form("final.mp4")
):
    # Build a single video by trimming multiple segments and concatenating
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")

    session = load_json(meta_path)
    file_paths = session.get("file_paths") or {}

    try:
        mani = json.loads(manifest)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid manifest JSON: {e}")

    segs = mani.get("segments") or []
    if not segs:
        raise HTTPException(status_code=400, detail="manifest has no segments")

    fps = int(mani.get("fps", 30))
    scale = int(mani.get("scale", 720))  # 720 default to reduce memory

    safe_name = safe_filename(filename, "final.mp4")
    out_path = sd / safe_name

    work = sd / f"stitch_{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)

    list_file = work / "list.txt"
    used_segments = []
    list_lines = []

    # Generate uniformly-encoded intermediate clips (low RAM, single thread)
    for idx, s in enumerate(segs):
        fid = s.get("file_id")
        if not fid or fid not in file_paths:
            print("stitch: skipping unknown file_id", fid)
            continue

        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        duration = max(0.05, end - start)
        src = file_paths[fid]
        seg_out = work / f"seg_{idx:03d}.mp4"

        try:
            run([
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-t", f"{duration:.3f}",
                "-i", src,
                "-vf", f"scale={scale}:-2:flags=lanczos",
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k",
                "-threads", "1",
                "-movflags", "+faststart",
                str(seg_out)
            ])
            list_lines.append(f"file '{seg_out.as_posix()}'\n")
            used_segments.append({
                "file_id": fid,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
            })
        except Exception as e:
            print("stitch: segment failed, skipping:", e)

    if not list_lines:
        raise HTTPException(status_code=400, detail="no valid segments to stitch")

    list_file.write_text("".join(list_lines))

    # Try concat with stream copy first
    try:
        run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path)
        ])
    except Exception as e1:
        print("stitch: concat copy failed, re-encoding:", e1)
        # Fallback: re-encode the concatenation
        run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-vf", f"scale={scale}:-2:flags=lanczos",
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k",
            "-threads", "1",
            "-movflags", "+faststart",
            str(out_path)
        ])

    return JSONResponse({
        "ok": True,
        "message": "stitch complete",
        "session_id": session_id,
        "filename": safe_name,
        "download": f"/download/{session_id}/{safe_name}",
        "segments_used": used_segments
    })

# -------- Step 9: Analyze (silence segmentation + Whisper transcription) --------
@app.post("/analyze")
def analyze_session(
    session_id: str = Form(...),
    max_segments_per_file: int = Form(8),
    min_segment_sec: float = Form(0.6),
    sil_db: float = Form(-30.0),
    sil_min_sec: float = Form(0.35)
):
    """
    For each uploaded file:
      1) Detect silence points with ffmpeg silencedetect.
      2) Create short segments between silences (>= min_segment_sec).
      3) Transcribe each segment with Whisper (ASR_MODEL).
    Saves results to analysis.json.
    """
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    file_paths: Dict[str, str] = session.get("file_paths") or {}
    files: List[Dict[str, Any]] = session.get("files") or []

    analysis: Dict[str, Any] = {"session_id": session_id, "files": []}

    for f in files:
        fid = f["file_id"]
        src = file_paths.get(fid)
        if not src or not Path(src).exists():
            continue

        # 1) Find silence markers
        # Parse lines with "silence_start" and "silence_end"
        log = run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", src,
            "-af", f"silencedetect=noise={sil_db}dB:duration={sil_min_sec}",
            "-f", "null", "-"
        ])
        # Some ffmpeg builds print to stderr; capture both:
        # (Already captured above; 'run' would have thrown on error.)

        starts = []
        ends = []
        for line in log.splitlines():
            line = line.strip()
            if "silence_start:" in line:
                try:
                    starts.append(float(line.split("silence_start:")[1].strip()))
                except:  # noqa: E722
                    pass
            if "silence_end:" in line and "silence_duration:" in line:
                try:
                    ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))
                except:  # noqa: E722
                    pass

        # Build rough segments from silence boundaries
        # Start at 0.0, then carve by [end_of_prev_silence, start_of_next_silence]
        bounds = [0.0] + sorted(starts + ends)
        segs = []
        for i in range(len(bounds) - 1):
            s = bounds[i]
            e = bounds[i + 1]
            if e - s >= float(min_segment_sec):
                segs.append((s, e))

        # If we got nothing (quiet audio), fall back to first 6 seconds
        if not segs:
            segs = [(0.0, 6.0)]

        # Limit number of segments to control API time/cost
        segs = segs[: int(max_segments_per_file)]

        # 2) Export each segment to wav and 3) transcribe with Whisper
        seg_items = []
        for idx, (start_s, end_s) in enumerate(segs):
            dur = max(0.1, end_s - start_s)
            wav = Path(sd) / f"{fid}_seg_{idx:02d}.wav"
            try:
                run([
                    "ffmpeg", "-y",
                    "-hide_banner", "-loglevel", "error",
                    "-ss", f"{start_s:.3f}",
                    "-t", f"{dur:.3f}",
                    "-i", src,
                    "-ac", "1", "-ar", "16000",
                    str(wav)
                ])
                with wav.open("rb") as fwav:
                    tr = client.audio.transcriptions.create(
                        model=ASR_MODEL,
                        file=fwav,
                        response_format="json",
                        temperature=0
                    )
                text = getattr(tr, "text", "").strip()
            except Exception as e:
                print("analyze: segment transcription failed:", e)
                text = ""

            seg_items.append({
                "file_id": fid,
                "segment_id": f"{fid}_{idx:02d}",
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "duration": round(dur, 3),
                "text": text
            })
            try:
                wav.unlink(missing_ok=True)
            except Exception:
                pass

        analysis["files"].append({
            "file_id": fid,
            "segments": seg_items
        })

    save_json(sd / "analysis.json", analysis)
    return JSONResponse({"ok": True, "session_id": session_id, "analysis": analysis})

# -------- Step 9: Classify (Hook / Feature / Proof / CTA) --------
@app.post("/classify")
def classify_segments(
    session_id: str = Form(...),
    tone: str = Form("casual")
):
    """
    Reads analysis.json and calls an LLM to label each segment as Hook/Feature/Proof/CTA,
    with a score in [0,1]. Saves buckets.json with ranked segments per label.
    """
    sd = sess_dir(session_id)
    meta = load_json(sd / "session.json")
    analysis_path = sd / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(status_code=400, detail="run /analyze first")

    analysis = load_json(analysis_path)
    product_link = meta.get("product_link", "")
    features_csv = meta.get("features_csv", "")

    # Build a compact prompt payload
    items = []
    for f in analysis.get("files", []):
        for s in f.get("segments", []):
            items.append({
                "id": s["segment_id"],
                "text": s.get("text", "")[:500],  # clamp
                "duration": s.get("duration", 0),
            })

    # If no text, we still try (LLM can use position/duration)
    system_msg = (
        "You are labeling UGC ad segments. "
        "Labels: Hook, Feature, Proof, CTA. "
        "Return strict JSON with a list of objects: "
        "[{\"id\":\"segment_id\",\"label\":\"Hook|Feature|Proof|CTA\",\"score\":0..1,\"reason\":\"...\"}] "
        "Score is confidence that this segment fits the label for TikTok-short ads. "
        "Prefer short, punchy hooks; features describe benefits; proof shows results or testimonials; "
        "CTA asks to buy/click/try/follow."
    )
    user_msg = {
        "tone": tone,
        "product_link": product_link,
        "features_csv": features_csv,
        "segments": items
    }

    try:
        comp = client.chat.completions.create(
            model=CLS_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"}
        )
        raw = comp.choices[0].message.content
        result = json.loads(raw)
    except Exception as e:
        print("classify: LLM error:", e)
        # Fallback: naive rule — first is hook, last is CTA, middle features/proof
        fallback = []
        for i, it in enumerate(items):
            if i == 0:
                lab = "Hook"
            elif i == len(items) - 1:
                lab = "CTA"
            else:
                lab = "Feature"
            fallback.append({"id": it["id"], "label": lab, "score": 0.5, "reason": "fallback"})
        result = {"labels": fallback}

    labels = result.get("labels", result if isinstance(result, list) else [])
    if isinstance(labels, dict):
        labels = labels.get("labels", [])

    # Map segment_id -> label info
    by_id = {s["segment_id"]: s for f in analysis.get("files", []) for s in f.get("segments", [])}

    buckets = {"Hook": [], "Feature": [], "Proof": [], "CTA": []}
    for ent in labels:
        sid = ent.get("id")
        lab = ent.get("label", "Feature")
        score = float(ent.get("score", 0.0))
        if sid in by_id and lab in buckets:
            item = by_id[sid].copy()
            item["score"] = round(score, 3)
            item["reason"] = ent.get("reason", "")
            buckets[lab].append(item)

    # Sort each bucket by score desc, then duration asc (snappy first)
    for k in buckets:
        buckets[k].sort(key=lambda x: (-x.get("score", 0), x.get("duration", 999)))

    out = {"session_id": session_id, "buckets": buckets}
    save_json(sd / "buckets.json", out)
    return JSONResponse({"ok": True, "session_id": session_id, "buckets": out["buckets"]})

# -------- Step 9: Auto-manifest from buckets --------
@app.post("/automanifest")
def automanifest(
    session_id: str = Form(...),
    filename: str = Form("final.mp4"),
    fps: int = Form(30),
    scale: int = Form(720),
    max_total_sec: float = Form(12.0)
):
    """
    Reads buckets.json and builds a simple Hook→Feature→Proof→CTA manifest.
    """
    sd = sess_dir(session_id)
    bpath = sd / "buckets.json"
    if not bpath.exists():
        raise HTTPException(status_code=400, detail="run /classify first")

    buckets = load_json(bpath).get("buckets", {})
    order = ["Hook", "Feature", "Proof", "CTA"]
    segments = []
    total = 0.0

    for label in order:
        cand = (buckets.get(label) or [])[:3]  # consider top-3 for each
        picked = None
        for c in cand:
            # enforce minimum 0.6s and max 5s per piece
            dur = max(0.6, min(5.0, float(c.get("duration", 0))))
            if total + dur <= float(max_total_sec):
                picked = {
                    "file_id": c["file_id"],
                    "start": float(c["start"]),
                    "end": float(c["start"]) + dur,
                    "label": label
                }
                total += dur
                break
        if picked:
            segments.append(picked)

    manifest = {"segments": segments, "fps": int(fps), "scale": int(scale)}
    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "filename": filename,
        "manifest": manifest
    })

# -------- Download --------
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
