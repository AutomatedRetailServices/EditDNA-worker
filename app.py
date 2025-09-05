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

ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1")       # transcription
CLS_MODEL = os.getenv("CLS_MODEL", "gpt-4o-mini")     # classification

client = OpenAI(api_key=OPENAI_API_KEY)

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "1.3.0-autoassemble"

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
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout or ""

def safe_filename(name: str, default: str) -> str:
    s = "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_", ".",)).strip()
    return s or default

# --------------------- CORE IMPLEMENTATIONS (re-used) --------------------- #
def impl_analyze(session_id: str, max_segments_per_file: int = 8,
                 min_segment_sec: float = 0.6, sil_db: float = -30.0,
                 sil_min_sec: float = 0.35) -> Dict[str, Any]:
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    file_paths: Dict[str, str] = session.get("file_paths") or {}
    files: List[Dict[str, Any]] = session.get("files") or []
    analysis: Dict[str, Any] = {"session_id": session_id, "files": []}

    for f in files:
        fid = f["file_id"]
        src = file_paths.get(fid)
        if not src or not Path(src).exists():
            continue

        # Silence detection (ffmpeg often prints to stderr; run() captures)
        log = run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", src,
            "-af", f"silencedetect=noise={sil_db}dB:duration={sil_min_sec}",
            "-f", "null", "-"
        ])
        # Parse both stdout & stderr text:
        # In many builds, silencedetect prints to stderr; combine to be safe:
        # (run() already raises on non-zero)

        starts, ends = [], []
        for line in log.splitlines():
            line = line.strip()
            if "silence_start:" in line:
                try:
                    starts.append(float(line.split("silence_start:")[1].strip()))
                except:
                    pass
            if "silence_end:" in line and "silence_duration:" in line:
                try:
                    ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))
                except:
                    pass

        bounds = [0.0] + sorted(starts + ends)
        segs = []
        for i in range(len(bounds) - 1):
            s = bounds[i]
            e = bounds[i + 1]
            if e - s >= float(min_segment_sec):
                segs.append((s, e))
        if not segs:
            segs = [(0.0, 6.0)]

        segs = segs[: int(max_segments_per_file)]

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
            finally:
                try:
                    wav.unlink(missing_ok=True)
                except:
                    pass

            seg_items.append({
                "file_id": fid,
                "segment_id": f"{fid}_{idx:02d}",
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "duration": round(dur, 3),
                "text": text
            })

        analysis["files"].append({"file_id": fid, "segments": seg_items})

    save_json(sd / "analysis.json", analysis)
    return analysis

def impl_classify(session_id: str, tone: str = "casual") -> Dict[str, Any]:
    sd = sess_dir(session_id)
    meta = load_json(sd / "session.json")
    analysis_path = sd / "analysis.json"
    if not analysis_path.exists():
        raise HTTPException(status_code=400, detail="run /analyze first")
    analysis = load_json(analysis_path)

    product_link = meta.get("product_link", "")
    features_csv = meta.get("features_csv", "")

    items = []
    for f in analysis.get("files", []):
        for s in f.get("segments", []):
            items.append({
                "id": s["segment_id"],
                "text": (s.get("text", "") or "")[:500],
                "duration": s.get("duration", 0),
            })

    system_msg = (
        "You are labeling UGC ad segments. "
        "Labels: Hook, Feature, Proof, CTA. "
        "Return strict JSON with key 'labels' → "
        "[{\"id\":\"segment_id\",\"label\":\"Hook|Feature|Proof|CTA\",\"score\":0..1,\"reason\":\"...\"}]. "
        "Prefer short punchy hooks; features = benefits; proof = results/testimonials; "
        "CTA = clear ask."
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
        fallback = []
        for i, it in enumerate(items):
            lab = "Hook" if i == 0 else ("CTA" if i == len(items) - 1 else "Feature")
            fallback.append({"id": it["id"], "label": lab, "score": 0.5, "reason": "fallback"})
        result = {"labels": fallback}

    labels = result.get("labels", [])
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

    for k in buckets:
        buckets[k].sort(key=lambda x: (-x.get("score", 0), x.get("duration", 999)))

    out = {"session_id": session_id, "buckets": buckets}
    save_json(sd / "buckets.json", out)
    return out

def impl_automanifest(session_id: str, fps: int = 30, scale: int = 720,
                      max_total_sec: float = 12.0) -> Dict[str, Any]:
    sd = sess_dir(session_id)
    bpath = sd / "buckets.json"
    if not bpath.exists():
        raise HTTPException(status_code=400, detail="run /classify first")
    buckets = load_json(bpath).get("buckets", {})
    order = ["Hook", "Feature", "Proof", "CTA"]
    segments = []
    total = 0.0
    for label in order:
        cand = (buckets.get(label) or [])[:3]
        picked = None
        for c in cand:
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
    return {"segments": segments, "fps": int(fps), "scale": int(scale)}

def impl_stitch(session_id: str, manifest: Dict[str, Any], out_name: str) -> Dict[str, Any]:
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    file_paths = session.get("file_paths") or {}
    segs = manifest.get("segments") or []
    if not segs:
        raise HTTPException(status_code=400, detail="manifest has no segments")

    fps = int(manifest.get("fps", 30))
    scale = int(manifest.get("scale", 720))
    safe_name = safe_filename(out_name, "final.mp4")
    out_path = sd / safe_name

    work = sd / f"stitch_{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)
    list_file = work / "list.txt"
    used_segments, list_lines = [], []

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

    return {
        "filename": safe_name,
        "download": f"/download/{session_id}/{safe_name}",
        "segments_used": used_segments
    }

# -------------------------- ROUTES (existing) -------------------------- #
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
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)
    files_meta, file_paths = [], {}

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

@app.post("/export")
def export_video(
    session_id: str = Form(...),
    filename: str = Form("draft.mp4"),
    start_seconds: float = Form(0.0),
    duration_seconds: float = Form(10.0)
):
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    session = load_json(meta_path)
    files = session.get("files") or []
    file_paths = session.get("file_paths") or {}
    if not files:
        raise HTTPException(status_code=400, detail="no files in session")
    fid = files[0]["file_id"]
    src_path = file_paths.get(fid)
    if not src_path or not Path(src_path).exists():
        raise HTTPException(status_code=404, detail="source video not found")

    safe_name = safe_filename(filename, "draft.mp4")
    out_path = sd / safe_name
    start_s = float(start_seconds)
    dur_s = max(0.1, float(duration_seconds))
    try:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", src_path,
            "-c", "copy", "-movflags", "+faststart", str(out_path)
        ])
    except Exception:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", src_path,
            "-vf", "scale=720:-2:flags=lanczos", "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
            "-threads", "1", "-movflags", "+faststart", str(out_path)
        ])

    return JSONResponse({
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": safe_name,
        "download": f"/download/{session_id}/{safe_name}"
    })

@app.post("/analyze")
def analyze_session(
    session_id: str = Form(...),
    max_segments_per_file: int = Form(8),
    min_segment_sec: float = Form(0.6),
    sil_db: float = Form(-30.0),
    sil_min_sec: float = Form(0.35)
):
    analysis = impl_analyze(session_id, max_segments_per_file, min_segment_sec, sil_db, sil_min_sec)
    return JSONResponse({"ok": True, "session_id": session_id, "analysis": analysis})

@app.post("/classify")
def classify_segments(
    session_id: str = Form(...),
    tone: str = Form("casual")
):
    buckets = impl_classify(session_id, tone)
    return JSONResponse({"ok": True, "session_id": session_id, "buckets": buckets["buckets"]})

@app.post("/automanifest")
def automanifest(
    session_id: str = Form(...),
    filename: str = Form("final.mp4"),
    fps: int = Form(30),
    scale: int = Form(720),
    max_total_sec: float = Form(12.0)
):
    manifest = impl_automanifest(session_id, fps, scale, max_total_sec)
    return JSONResponse({"ok": True, "session_id": session_id, "filename": filename, "manifest": manifest})

@app.post("/stitch")
def stitch_video(
    session_id: str = Form(...),
    manifest: str = Form(...),
    filename: str = Form("final.mp4")
):
    try:
        mani = json.loads(manifest)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid manifest JSON: {e}")
    out = impl_stitch(session_id, mani, filename)
    return JSONResponse({"ok": True, "message": "stitch complete",
                         "session_id": session_id, **out})

# -------------------------- NEW: ONE-CLICK FLOW -------------------------- #
@app.post("/autoassemble")
def autoassemble(
    session_id: str = Form(...),
    filename: str = Form("final.mp4"),
    tone: str = Form("casual"),
    fps: int = Form(30),
    scale: int = Form(720),
    max_total_sec: float = Form(12.0),
    max_segments_per_file: int = Form(8)
):
    """
    Runs analyze → classify → build manifest → stitch.
    Returns final download link.
    """
    # 1) analyze
    impl_analyze(session_id, max_segments_per_file=max_segments_per_file)
    # 2) classify
    impl_classify(session_id, tone=tone)
    # 3) manifest
    manifest = impl_automanifest(session_id, fps=fps, scale=scale, max_total_sec=max_total_sec)
    # 4) stitch
    out = impl_stitch(session_id, manifest, filename)
    return JSONResponse({
        "ok": True,
        "message": "autoassemble complete",
        "session_id": session_id,
        "filename": out["filename"],
        "download": out["download"],
        "manifest": manifest,
        "segments_used": out["segments_used"]
    })

# -------------------------- Download -------------------------- #
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(f, filename=filename, media_type="video/mp4")
