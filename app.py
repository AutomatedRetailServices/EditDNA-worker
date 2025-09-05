from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import os, tempfile, subprocess, json, re, uuid, time, shutil, hashlib
import requests
from bs4 import BeautifulSoup

# -----------------------
# App & OpenAI client
# -----------------------
app = FastAPI(title="script2clipshop-worker", version="1.5.0")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TMPDIR = "/tmp/script2clipshop"
os.makedirs(TMPDIR, exist_ok=True)

# -----------------------
# Utils
# -----------------------
def run(cmd: List[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def extract_duration_seconds(path: str) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return float((probe.stdout or "").strip())
    except:
        return 0.0

def safe_json(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return obj

# -----------------------
# Product link summarizer
# -----------------------
def summarize_product_links(urls: List[str], timeout_s: int = 7) -> List[Dict[str, Any]]:
    cards = []
    seen = set()
    for u in urls:
        u = (u or "").strip()
        if not u: 
            continue
        h = hashlib.md5(u.encode("utf-8")).hexdigest()[:10]
        cache_path = os.path.join(TMPDIR, f"product_{h}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cards.append(json.load(f))
                continue
            except:
                pass
        try:
            r = requests.get(u, timeout=timeout_s, headers={"User-Agent":"Mozilla/5.0"})
            html = r.text
            soup = BeautifulSoup(html, "html.parser")
            title = (soup.find("title").get_text() if soup.find("title") else "").strip()
            og_title = soup.find("meta", {"property":"og:title"})
            og_desc = soup.find("meta", {"property":"og:description"})
            meta_desc = soup.find("meta", {"name":"description"})
            desc = ""
            for m in [og_desc, meta_desc]:
                if m and m.get("content"): 
                    desc = m["content"].strip(); break
            # Attempt to gather bullet-ish items
            bullets = []
            for li in soup.select("ul li"):
                t = (li.get_text() or "").strip()
                if 4 <= len(t) <= 140:
                    bullets.append(t)
                if len(bullets) >= 12: break
            brand = ""
            for meta_key in ["og:site_name", "twitter:site", "application-name"]:
                tag = soup.find("meta", {"property": meta_key}) or soup.find("meta", {"name": meta_key})
                if tag and tag.get("content"):
                    brand = tag["content"].strip()
                    break
            name = og_title["content"].strip() if og_title and og_title.get("content") else (title or "")
            # extract numbers like "3000 lb", "2-pack"
            numbers = re.findall(r"\b\d[\d,\.]*\s?(lb|lbs|kg|oz|pack|pcs|x|\"|in|cm|mm)\b", " ".join([title, desc, " ".join(bullets)]), re.I)
            numbers = list(set(numbers))[:6]
            cta_terms = ["Shop now", "Add to cart", "Free shipping", "Limited stock", "In stock", "Buy now"]
            card = {
                "url": u, "brand": brand, "name": name,
                "bullets": bullets[:12],
                "description": desc,
                "numbers": numbers,
                "cta_terms": cta_terms
            }
            with open(cache_path, "w") as f:
                json.dump(card, f)
            if u not in seen:
                cards.append(card); seen.add(u)
        except Exception:
            # fail soft; skip this link
            continue
    return cards

# -----------------------
# Segmentation + Transcription
# -----------------------
def silences_for_wav(wav: str, noise_db: str = "-30dB", min_sil: float = 0.35) -> Tuple[List[float], List[float], float]:
    proc = subprocess.run(
        ["ffmpeg", "-i", wav, "-af", f"silencedetect=noise={noise_db}:d={min_sil}", "-f", "null", "-"],
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
    return starts, ends, total

def segment_and_transcribe_one(src_path: str, record_dir: str, language: Optional[str]=None) -> Tuple[List[Dict[str,Any]], str]:
    """Returns (segments_with_text, full_transcript)."""
    segments = []
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        # 16k mono WAV for stable VAD/STT
        run(["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav])
        starts, ends, total = silences_for_wav(wav)
        for s, e in zip(starts, ends):
            if (e - s) < 0.25:
                continue
            chunk = os.path.join(td, f"seg_{int(s*1000)}.wav")
            run(["ffmpeg", "-y", "-ss", f"{s:.3f}", "-t", f"{(e-s):.3f}", "-i", wav, "-ac", "1", "-ar", "16000", chunk])
            with open(chunk, "rb") as f:
                tr = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    **({"language": language} if language else {})
                )
            text = (getattr(tr, "text", "") or "").strip()
            if text:
                segments.append({"start": round(s,3), "end": round(e,3), "text": text})
    transcript = " ".join(seg["text"] for seg in segments)
    return segments, transcript

# -----------------------
# Classification (verbatim extraction)
# -----------------------
EARLY_SEC = 15.0

def classify_text_buckets(transcript: str,
                          segments: List[Dict[str, Any]],
                          features_csv: str,
                          tone: str,
                          product_cards: List[Dict[str,Any]]) -> Dict[str, Any]:
    # Compact segments for context
    seg_listing = "\n".join(
        f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in segments[:100]
    )

    # Build product context text
    pc_txt = []
    for c in (product_cards or []):
        bullets = "; ".join(c.get("bullets", [])[:8])
        nums = ", ".join(c.get("numbers", [])[:6])
        cta = ", ".join(c.get("cta_terms", [])[:6])
        pc_txt.append(
            f"- Brand/Name: {c.get('brand','')}/{c.get('name','')}\n"
            f"  Bullets: {bullets}\n"
            f"  Numbers: {nums}\n"
            f"  CTA terms: {cta}\n"
        )
    product_context = "\n".join(pc_txt) if pc_txt else "(none)"

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
        f"Key features to prioritize (csv): {features_csv}\n\n"
        f"Product context (summarized from links):\n{product_context}\n\n"
        f"Transcript (full):\n{transcript}\n\n"
        f"Segments (timestamped):\n{seg_listing}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
        max_tokens=900
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

def find_segment_for_line(line: str, segments: List[Dict[str, Any]]) -> Optional[Dict[str,Any]]:
    needle = re.sub(r"\s+", " ", line.strip().lower())
    best = None
    best_overlap = 0
    for seg in segments:
        hay = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if needle and needle in hay:
            return seg
        # fallback: token overlap
        n_tokens = set(needle.split())
        h_tokens = set(hay.split())
        overlap = len(n_tokens & h_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = seg
    return best

def score_hook(seg: Dict[str, Any]) -> Tuple[float, str, bool]:
    start, end = float(seg["start"]), float(seg["end"])
    dur = max(0.01, end - start)
    text = seg["text"]
    score, reasons = 0.5, []
    is_early = start <= EARLY_SEC
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

def build_hook_sets(hook_lines: List[str], segments: List[Dict[str, Any]], enrich: Dict[str,Any]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    intro, in_body, seen = [], [], set()
    for line in hook_lines:
        seg = find_segment_for_line(line, segments)
        if not seg: 
            continue
        # dedupe by normalized text
        norm = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if norm in seen: 
            continue
        seen.add(norm)
        score, why, is_early = score_hook(seg)
        hook_item = {
            "file_id": seg["file_id"], "filename": seg["filename"],
            "text": seg["text"], "start": seg["start"], "end": seg["end"],
            "score": score, "is_early": is_early, "why": why
        }
        (intro if is_early else in_body).append(hook_item)
    intro.sort(key=lambda x: x["score"], reverse=True)
    in_body.sort(key=lambda x: x["score"], reverse=True)
    return intro[:3], in_body[:3]

# -----------------------
# Session storage helpers
# -----------------------
def session_path(session_id: str) -> str:
    return os.path.join(TMPDIR, f"session_{session_id}.json")

def save_session(sess: Dict[str,Any]):
    path = session_path(sess["session_id"])
    with open(path, "w") as f:
        json.dump(safe_json(sess), f)

def load_session(session_id: str) -> Dict[str,Any]:
    path = session_path(session_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="session not found")
    with open(path, "r") as f:
        return json.load(f)

# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": app.version}

@app.post("/process")
async def process_videos(
    videos: List[UploadFile] = File(..., description="Attach 1+ raw video files of the SAME product"),
    tone: str = Form("casual"),
    features_csv: str = Form(""),
    product_link: str = Form("", description="Legacy single link; optional"),
    product_links: str = Form("", description="Comma-separated links; optional"),
    product_language: str = Form("", description="Whisper language code (e.g., 'es'); optional")
):
    """
    Accepts multiple files for one product/session.
    - Saves each raw file to /tmp with unique name
    - Segments + transcribes
    - Classifies into buckets (verbatim)
    - Returns session_id + buckets + segments (with file_id/filename)
    - Persists full session JSON to /tmp so /export survives restarts
    """
    if not videos:
        raise HTTPException(status_code=400, detail="No video files provided")

    # Make a working dir for this session
    session_id = uuid.uuid4().hex
    workdir = os.path.join(TMPDIR, session_id)
    os.makedirs(workdir, exist_ok=True)

    # Save files to disk so /export can read them later
    saved_files = []  # [{file_id, filename, path}]
    for up in videos:
        ext = os.path.splitext(up.filename or "upload.mp4")[1] or ".mp4"
        fid = uuid.uuid4().hex[:8]
        dst = os.path.join(workdir, f"{fid}{ext}")
        data = await up.read()
        with open(dst, "wb") as f:
            f.write(data)
        saved_files.append({"file_id": fid, "filename": up.filename or f"upload{ext}", "path": dst})

    # Summarize product links
    links = []
    if product_link.strip():
        links.append(product_link.strip())
    if product_links.strip():
        parts = [x.strip() for x in product_links.split(",") if x.strip()]
        links.extend(parts)
    product_cards = summarize_product_links(links) if links else []

    # For each saved file: segment + transcribe
    all_segments, full_text = [], []
    for rec in saved_files:
        segs, transcript = segment_and_transcribe_one(rec["path"], workdir, language=(product_language or None))
        # attach file metadata
        for s in segs:
            s["file_id"] = rec["file_id"]
            s["filename"] = rec["filename"]
        all_segments.extend(segs)
        if transcript.strip():
            full_text.append(transcript.strip())

    transcript = " ".join(full_text)

    # Classify (verbatim lines)
    try:
        buckets_raw = classify_text_buckets(transcript, all_segments, features_csv, tone, product_cards)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Classification error: {e}")

    # Hook mapping + scoring
    intro_hooks, in_body_hooks = build_hook_sets(buckets_raw["hook_lines"], all_segments, {})

    # Build final buckets; include file metadata on features/proof/cta by mapping lines back to segments
    def map_lines(lines: List[str]) -> List[Dict[str,Any]]:
        mapped = []
        for line in lines:
            seg = find_segment_for_line(line, all_segments)
            if seg:
                mapped.append({
                    "file_id": seg["file_id"], "filename": seg["filename"],
                    "text": seg["text"], "start": seg["start"], "end": seg["end"]
                })
        # dedupe identical texts
        seen = set(); out=[]
        for m in mapped:
            key = (m["file_id"], m["start"], m["end"], re.sub(r"\s+"," ",m["text"].lower()))
            if key in seen: continue
            seen.add(key); out.append(m)
        return out[:3]

    features = map_lines(buckets_raw["feature_lines"])
    proof    = map_lines(buckets_raw["proof_lines"])
    cta      = map_lines(buckets_raw["cta_lines"])

    final_buckets = {
        "hooks": {
            "intro_hooks": intro_hooks,
            "in_body_hooks": in_body_hooks
        },
        "features": features,
        "proof": proof,
        "cta": cta
    }

    default_draft = {
        "hook_source": "intro_hooks" if intro_hooks else ("in_body_hooks" if in_body_hooks else None),
        "hook_index": 0 if (intro_hooks or in_body_hooks) else None,
        "feature_index": 0 if features else None,
        "proof_index": 0 if proof else None,
        "cta_index": 0 if cta else None
    }

    # Persist session
    session_doc = {
        "ok": True,
        "session_id": session_id,
        "files": [{"file_id": f["file_id"], "filename": f["filename"]} for f in saved_files],
        "file_paths": {f["file_id"]: f["path"] for f in saved_files},
        "product_cards": product_cards,
        "tone": tone,
        "features_csv": features_csv,
        "product_links": links,
        "transcript": transcript,
        "transcript_chars": len(transcript),
        "segments": all_segments,
        "buckets": final_buckets,
        "default_draft": default_draft
    }
    save_session(session_doc)

    # Lightweight response (no internal paths)
    resp = dict(session_doc)
    resp.pop("file_paths", None)
    return JSONResponse(resp)

@app.get("/session/{session_id}")
def get_session(session_id: str):
    doc = load_session(session_id)
    # hide internal paths from API response
    doc = dict(doc)
    doc.pop("file_paths", None)
    return JSONResponse(doc)

# -----------------------
# Export helper
# -----------------------
def cut_and_concat(plan: List[Dict[str,Any]], file_paths: Dict[str,str], out_path: str):
    """
    For each item in plan: {"file_id","start","end"}
    Produces a concat MP4 at out_path (1080x1920, 30fps).
    """
    work = tempfile.mkdtemp(prefix="cut_", dir=TMPDIR)
    parts = []
    try:
        for i, item in enumerate(plan):
            fid = item["file_id"]
            start, end = float(item["start"]), float(item["end"])
            dur = max(0.01, end - start)
            src = file_paths[fid]
            part = os.path.join(work, f"p{i:02d}.mp4")
            # Trim + scale/pad to 1080x1920
            # Using scale to fit width, then pad to vertical (simple safe approach)
            vf = "scale=1080:-2:flags=lanczos, pad=1080:1920:(1080-iw*min(1080/iw\\,1920/ih))/2:(1920-ih*min(1080/iw\\,1920/ih))/2"
            run([
                "ffmpeg","-y","-ss",f"{start:.3f}","-t",f"{dur:.3f}","-i",src,
                "-vf", vf, "-r","30","-an","-c:v","libx264","-preset","veryfast","-crf","20", part
            ])
            # extract audio too (then merge later)
            audio = os.path.join(work, f"p{i:02d}_a.m4a")
            run([
                "ffmpeg","-y","-ss",f"{start:.3f}","-t",f"{dur:.3f}","-i",src,
                "-vn","-c:a","aac","-b:a","128k", audio
            ])
            parts.append((part, audio))
        # Concat video
        vlist = os.path.join(work, "list_v.txt")
        with open(vlist,"w") as f:
            for p,_ in parts:
                f.write(f"file '{p}'\n")
        vcat = os.path.join(work, "cat_v.mp4")
        run(["ffmpeg","-y","-f","concat","-safe","0","-i",vlist,"-c","copy", vcat])

        # Concat audio
        alist = os.path.join(work, "list_a.txt")
        with open(alist,"w") as f:
            for _,a in parts:
                f.write(f"file '{a}'\n")
        acat = os.path.join(work, "cat_a.m4a")
        run(["ffmpeg","-y","-f","concat","-safe","0","-i",alist,"-c","copy", acat])

        # Merge A/V
        run([
            "ffmpeg","-y","-i",vcat,"-i",acat,"-c:v","copy","-c:a","aac","-shortest", out_path
        ])
    finally:
        try: shutil.rmtree(work)
        except: pass

@app.post("/export")
async def export_video(
    session_id: str = Form(...),
    filename: str = Form("draft.mp4"),
    hook_index: Optional[int] = Form(None),
    feature_index: Optional[int] = Form(None),
    proof_index: Optional[int] = Form(None),
    cta_index: Optional[int] = Form(None),
    allowed_file_ids: Optional[str] = Form(None)  # comma-separated whitelist
):
    """
    Builds a cut plan from the session buckets and renders MP4 to /tmp.
    Use allowed_file_ids to restrict picks to specific source files (avoid mixing different products).
    """
    doc = load_session(session_id)
    file_paths: Dict[str,str] = doc.get("file_paths", {})
    if not file_paths:
        raise HTTPException(status_code=404, detail="session has no file paths (was it created here?)")

    allowed: Optional[set] = None
    if allowed_file_ids:
        allowed = set(x.strip() for x in allowed_file_ids.split(",") if x.strip())

    def filter_allowed(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        if not allowed: return items
        return [x for x in (items or []) if x.get("file_id") in allowed]

    buckets = doc.get("buckets", {})
    hooks = buckets.get("hooks", {})
    intro_hooks = filter_allowed(hooks.get("intro_hooks", []) or [])
    in_body_hooks = filter_allowed(hooks.get("in_body_hooks", []) or [])
    features = filter_allowed(buckets.get("features", []) or [])
    proof    = filter_allowed(buckets.get("proof", []) or [])
    cta      = filter_allowed(buckets.get("cta", []) or [])

    # pick helper
    def pick(lst: List[Dict[str,Any]], idx: Optional[int]) -> Optional[Dict[str,Any]]:
        if not lst: return None
        if isinstance(idx, int) and 0 <= idx < len(lst): return lst[idx]
        return lst[0]

    # decide hook source
    hook_src = intro_hooks if intro_hooks else in_body_hooks
    sel = []
    hk = pick(hook_src, hook_index)
    if hk: sel.append(hk)
    ft = pick(features, feature_index)
    if ft: sel.append(ft)
    pf = pick(proof, proof_index)
    if pf: sel.append(pf)
    ct = pick(cta, cta_index)
    if ct: sel.append(ct)

    if not sel:
        raise HTTPException(status_code=400, detail="Nothing to export (no buckets available with current filters).")

    # Build plan → trim & concat
    plan = [{"file_id": x["file_id"], "start": x["start"], "end": x["end"]} for x in sel]
    outname = re.sub(r"[^A-Za-z0-9._-]","_", filename or "draft.mp4")
    outdir = os.path.join(TMPDIR, session_id)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, outname)

    try:
        cut_and_concat(plan, file_paths, out_path)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"FFmpeg export error: {e}")

    return JSONResponse({
        "ok": True,
        "session_id": session_id,
        "filename": outname,
        "plan": plan,
        "download_url": f"/download/{session_id}/{outname}"
    })

@app.get("/download/{session_id}/{filename}")
def download(session_id: str, filename: str):
    path = os.path.join(TMPDIR, session_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)
