import os
import re
import uuid
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI  # OpenAI v1 SDK

# -----------------------------------------------------------------------------
# App & Config
# -----------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to Bubble domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set")
client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SESS_ROOT = Path("/tmp/s2c_sessions")
SESS_ROOT.mkdir(parents=True, exist_ok=True)

VERSION = "1.1.0-genscript"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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

def run(cmd: list):
    """Run a shell command and raise on error, capturing stderr for debugging."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def safe_filename(name: str, default: str) -> str:
    s = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".",)).strip()
    return s or default

def extract_json_block(text: str) -> dict:
    """
    Try to extract the first JSON object from a string (even if the model wraps it
    in markdown code fences). Fall back to parsing the whole string.
    """
    # Look for fenced code block with json
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Look for any JSON object
    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass
    # Fall back to direct parse
    return json.loads(text)

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "editdna-worker", "version": VERSION}

# -----------------------------------------------------------------------------
# Step 1 — Upload & Session
# -----------------------------------------------------------------------------
@app.post("/process")
async def process_video(
    videos: List[UploadFile] = File(...),
    tone: str = Form("casual"),
    features_csv: str = Form(""),
    product_link: str = Form(...),           # REQUIRED per updated scope
    # Optional per-file slot metadata could be added later (e.g., "slot: hook/feature/proof/cta")
):
    """
    Saves uploaded videos to /tmp and stores session.json with file paths + context.
    Returns session_id + file list. Product link is required (A & B flows).
    """
    session_id = uuid.uuid4().hex
    sd = sess_dir(session_id)

    files_meta = []
    file_paths: Dict[str, str] = {}

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
        "file_paths": file_paths,        # internal only, not returned to client
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
    }
    save_json(sd / "session.json", session_json)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "files": files_meta
    })

# -----------------------------------------------------------------------------
# Step 2 — Basic Export (playable trim of the first file)
# -----------------------------------------------------------------------------
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

    first = files[0]
    fid = first["file_id"]
    src_path = file_paths.get(fid)
    if not src_path or not Path(src_path).exists():
        raise HTTPException(status_code=404, detail="source video not found")

    safe_name = safe_filename(filename, "draft.mp4")
    out_path = sd / safe_name

    try:
        run([
            "ffmpeg", "-y",
            "-ss", f"{float(start_seconds):.3f}",
            "-t", f"{max(0.1, float(duration_seconds)):.3f}",
            "-i", src_path,
            "-vf", "scale=1080:-2:flags=lanczos",
            "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            str(out_path)
        ])
    except Exception:
        shutil.copy(src_path, out_path)

    return JSONResponse({
        "ok": True,
        "message": "export complete",
        "session_id": session_id,
        "filename": safe_name,
        "download": f"/download/{session_id}/{safe_name}"
    })

# -----------------------------------------------------------------------------
# Step 3 — Stitch (manifest → MP4)
# -----------------------------------------------------------------------------
def impl_stitch(session_id: str, manifest: dict, filename: str) -> dict:
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")

    file_paths: Dict[str, str] = session.get("file_paths") or {}
    segments = manifest.get("segments") or []
    fps = int(manifest.get("fps") or 30)
    scale = int(manifest.get("scale") or 720)

    # Create temporary concat list
    concat_txt = sd / "concat.txt"
    temp_segments: List[Path] = []

    # Render each segment to an intermediate clip
    for i, seg in enumerate(segments):
        fid = seg.get("file_id")
        start = float(seg.get("start", 0))
        end = float(seg.get("end", max(start + 0.2, 0.2)))
        dur = max(0.05, end - start)

        src = file_paths.get(fid)
        if not src or not Path(src).exists():
            print(f"stitch: skipping unknown file_id {fid}")
            continue

        seg_out = sd / f"seg_{i:03d}.mp4"
        temp_segments.append(seg_out)

        # Basic (can be upgraded with SmartCut later)
        run([
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{dur:.3f}",
            "-i", src,
            "-vf", f"scale={scale}:-2:flags=lanczos",
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            str(seg_out)
        ])

    # Build concat file
    with concat_txt.open("w") as w:
        for p in temp_segments:
            w.write(f"file '{p.as_posix()}'\n")

    out_name = safe_filename(filename, "final.mp4")
    out_path = sd / out_name

    if temp_segments:
        run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(out_path)
        ])
    else:
        # Fallback: if no segments, copy first file
        files = session.get("files") or []
        if not files:
            raise HTTPException(status_code=400, detail="no segments and no files")
        first = files[0]
        src = file_paths.get(first["file_id"])
        if not src or not Path(src).exists():
            raise HTTPException(status_code=404, detail="source video not found")
        shutil.copy(src, out_path)

    return {
        "ok": True,
        "message": "stitch complete",
        "session_id": session_id,
        "filename": out_name,
        "download": f"/download/{session_id}/{out_name}"
    }

@app.post("/stitch")
def stitch_video(
    session_id: str = Form(...),
    manifest_json: str = Form(...),
    filename: str = Form("final.mp4")
):
    try:
        manifest = json.loads(manifest_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid manifest_json: {e}")

    out = impl_stitch(session_id, manifest, filename)
    return JSONResponse(out)

# -----------------------------------------------------------------------------
# Step 4 — Auto-Manifest (very simple rules; can be extended later)
# -----------------------------------------------------------------------------
@app.post("/automanifest")
def automanifest(
    session_id: str = Form(...),
    filename: str = Form("final.mp4"),
    fps: int = Form(30),
    scale: int = Form(720),
    max_total_sec: float = Form(20.0)
):
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    files = session.get("files") or []
    file_paths = session.get("file_paths") or {}

    if not files:
        raise HTTPException(status_code=400, detail="no files")

    # Naive: take the first ~max_total_sec seconds across files (2-clip cap)
    segments = []
    remaining = float(max_total_sec)
    for f in files[:4]:
        # Probe duration
        src = file_paths.get(f["file_id"])
        if not src or not Path(src).exists():
            continue
        # Use a fixed chunk ~min(remaining, 6s) per file for this simple pass
        use = min(remaining, 6.0)
        if use <= 0.05:
            break
        segments.append({
            "file_id": f["file_id"],
            "start": 0.0,
            "end": round(use, 3)
        })
        remaining -= use
        if remaining <= 0.05:
            break

    manifest = {
        "fps": int(fps),
        "scale": int(scale),
        "segments": segments
    }

    # Save manifest into session for reference
    session["manifest"] = manifest
    save_json(sd / "session.json", session)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "filename": filename,
        "manifest": manifest
    })

# -----------------------------------------------------------------------------
# Step 5 — Auto-Assemble (pipeline wrapper)
# -----------------------------------------------------------------------------
def impl_analyze(session_id: str, max_segments_per_file: int = 8) -> dict:
    """
    Placeholder analysis (upgrade later with Whisper): just stub one segment per file.
    """
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    files = session.get("files") or []
    analysis = {"files": []}
    for f in files[:max_segments_per_file]:
        analysis["files"].append({
            "file_id": f["file_id"],
            "segments": [
                {"start": 0.0, "end": 6.0}  # simple placeholder
            ],
            "transcript": ""  # fill later with Whisper
        })
    session["analysis"] = analysis
    save_json(sd / "session.json", session)
    return analysis

def impl_classify(session_id: str) -> dict:
    """
    Placeholder classification (upgrade later): map first clip to Hook, next Feature, etc.
    """
    sd = sess_dir(session_id)
    session = load_json(sd / "session.json")
    analysis = session.get("analysis") or {}
    buckets = {"Hook": [], "Feature": [], "Proof": [], "CTA": []}
    order = ["Hook", "Feature", "Proof", "CTA"]
    idx = 0
    for af in (analysis.get("files") or []):
        for seg in af.get("segments") or []:
            label = order[idx % 4]
            buckets[label].append({
                "file_id": af["file_id"],
                "start": seg["start"],
                "end": seg["end"],
                "score": 0.5  # placeholder score
            })
            idx += 1
    session["classification"] = {"buckets": buckets}
    save_json(sd / "session.json", session)
    return session["classification"]

@app.post("/autoassemble")
def autoassemble(
    session_id: str = Form(...),
    filename: str = Form("final.mp4"),
    tone: str = Form("casual"),
    fps: int = Form(30),
    scale: int = Form(720),
    max_total_sec: float = Form(20.0),
    max_segments_per_file: int = Form(8)
):
    # Analyze → Classify → AutoManifest → Stitch
    impl_analyze(session_id, max_segments_per_file=max_segments_per_file)
    impl_classify(session_id)
    mani_resp = automanifest(session_id, filename, fps, scale, max_total_sec)
    manifest = mani_resp.body
    try:
        manifest_data = json.loads(manifest.decode("utf-8"))
    except Exception:
        # Fallback if middleware encoded differently
        manifest_data = mani_resp.media or {}
    manifest_json = json.dumps(manifest_data.get("manifest") or {})
    out = stitch_video(session_id=session_id, manifest_json=manifest_json, filename=filename)
    return out

# -----------------------------------------------------------------------------
# Step 9 — Flow A: /genscript (product_link required; script_text optional)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise script generator for short-form sales ads.
Always output STRICT JSON with this schema:
{
  "style_detected": "TalkingHead|Skit|Testimonial|Voiceover|best_fit",
  "slots": [
    {"slot":"hook","text":"..."},
    {"slot":"feature","text":"..."},
    {"slot":"proof","text":"..."},
    {"slot":"cta","text":"..."}
  ],
  "notes": "short rationale or guidance for voice/visuals (<=280 chars)"
}
Rules:
- Keep TOTAL script length suitable for 20–45s ad (TikTok).
- "hook": 1–2 punchy lines; "feature": 2–4 value points; "proof": social proof, demo, before/after; "cta": 1 line, imperative.
- Write in the requested language and tone; keep it slang-friendly if asked.
- If a user provided a script, detect style; if unclear set "style_detected":"best_fit" and still fill slots.
- NO code blocks, no markdown outside the JSON.
"""

STYLE_HINTS = {
    "TalkingHead": "Direct-to-camera, natural pacing, concise.",
    "Skit": "Two-person or scene-based, Problem→Solution→Demo mapping to Hook→Feature→Proof; quick cuts.",
    "Testimonial": "First-person credibility, before/after, outcome, short CTA.",
    "Voiceover": "Narration over visuals (b-roll), descriptive and vivid, short lines for captions."
}

def call_openai_for_script(model: str, product_link: str, style: str, tone: str,
                           language: str, target_length_sec: float,
                           script_text: Optional[str]) -> dict:
    # Build user prompt
    style_hint = STYLE_HINTS.get(style, "Use best-fit funnel style for short ads.")
    user_prompt = {
        "product_link": product_link,
        "style_requested": style,
        "tone": tone,
        "language": language,
        "target_length_sec": target_length_sec,
        "style_hint": style_hint,
        "script_text_if_any": script_text or ""
    }
    # Ask for strict JSON. We'll parse robustly.
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
        ],
        temperature=0.5,
        max_tokens=600
    )
    text = resp.choices[0].message.content.strip()
    try:
        data = extract_json_block(text)
        # Basic validation
        slots = data.get("slots") or []
        # Normalize slot ids
        normalized = []
        for s in slots:
            slot_key = (s.get("slot") or "").strip().lower()
            if slot_key not in {"hook", "feature", "proof", "cta"}:
                # Try to map from alternates
                mapping = {
                    "problem": "hook",
                    "solution": "feature",
                    "demo": "proof",
                    "credibility": "hook",   # map testimonial openers to hook
                    "before": "feature",
                    "after": "proof",
                    "vo hook": "hook",
                    "vo features": "feature",
                    "vo proof": "proof",
                    "vo cta": "cta",
                }
                slot_key = mapping.get(slot_key, slot_key)
            if slot_key in {"hook", "feature", "proof", "cta"}:
                normalized.append({"slot": slot_key, "text": s.get("text", "").strip()})
        if not normalized:
            raise ValueError("no valid slots")
        data["slots"] = normalized
        # Ensure style_detected
        if not data.get("style_detected"):
            data["style_detected"] = "best_fit"
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to parse script json: {e}")

@app.post("/genscript")
def genscript(
    session_id: str = Form(...),
    product_link: str = Form(...),                   # REQUIRED (all flows)
    style: str = Form(...),                          # TalkingHead|Skit|Testimonial|Voiceover|auto
    tone: str = Form("casual"),
    language: str = Form("en"),
    target_length_sec: float = Form(25.0),
    script_text: str = Form("")                      # optional user script
):
    """
    Generate (or map) a funnel script into slots for Flow A, save storyboard to session.
    - If script_text provided and style='auto' → auto-detect style (model).
    - If no script_text → generate from product_link + style preset.
    """
    sd = sess_dir(session_id)
    meta_path = sd / "session.json"
    if not meta_path.exists():
        # create a minimal session if not uploaded yet
        save_json(meta_path, {
            "session_id": session_id,
            "files": [],
            "file_paths": {},
            "tone": tone,
            "features_csv": "",
            "product_link": product_link
        })

    style_requested = style.strip()
    if style_requested not in {"TalkingHead", "Skit", "Testimonial", "Voiceover", "auto"}:
        raise HTTPException(status_code=400, detail="style must be one of TalkingHead|Skit|Testimonial|Voiceover|auto")

    # If user provided a script and asked for auto, let the model decide style
    chosen_style = style_requested if style_requested != "auto" else "TalkingHead"

    data = call_openai_for_script(
        model=OPENAI_MODEL,
        product_link=product_link,
        style=chosen_style,
        tone=tone,
        language=language,
        target_length_sec=target_length_sec,
        script_text=script_text or None
    )

    # If user asked for auto, we trust model's style_detected for UI display
    storyboard = {
        "style": data.get("style_detected") or style_requested,
        "slots": data.get("slots") or [],
        "notes": data.get("notes", "")
    }

    # Save to session
    session = load_json(meta_path)
    session["storyboard"] = storyboard
    save_json(meta_path, session)

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "product_link": product_link,
        "style_requested": style_requested,
        "storyboard": storyboard
    })

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    sd = sess_dir(session_id)
    f = sd / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="file not found")
    # For now assume mp4; later detect based on suffix
    return FileResponse(f, filename=filename, media_type="video/mp4")
