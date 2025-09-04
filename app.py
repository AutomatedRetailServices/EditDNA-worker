from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os, tempfile, subprocess, json, re
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI()

# OpenAI client (requires OPENAI_API_KEY)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

# -----------------------
# Segmentation + Transcription
# -----------------------

def segment_by_silence_and_transcribe(video_bytes: bytes, filename: str):
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
            # Convert to 16k mono WAV for stable VAD/STT
            run(["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", wav])

            # Detect silences (tune noise/duration if needed)
            proc = subprocess.run(
                ["ffmpeg", "-i", wav, "-af", "silencedetect=noise=-30dB:d=0.35", "-f", "null", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            log = proc.stderr or ""

            # Build start/end lists from silencedetect output
            starts = [0.0]
            ends = []
            for m in re.finditer(r"silence_start:\s*([0-9.]+)", log):
                try: ends.append(float(m.group(1)))
                except: pass
            for m in re.finditer(r"silence_end:\s*([0-9.]+)", log):
                try: starts.append(float(m.group(1)))
                except: pass

            # Ensure last end == total duration
            total = extract_duration_seconds(wav)
            if not ends or (ends and ends[-1] < total):
                ends.append(total)

            # Cut & transcribe each segment between silence regions
            for s, e in zip(starts, ends):
                # skip ultra-short blips
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
                        "text": text
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
    """
    Ask GPT to extract ONLY verbatim lines from transcript into:
    - hook_lines (list of strings — raw candidates)
    - feature_lines (list)
    - proof_lines (list)
    - cta_lines (list)
    We’ll map hook_lines back to segments to get start/end and score them.
    """
    # Compact segment listing for context (kept short to save tokens)
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

    # Normalize
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
# Hook mapping + scoring (A/B testing ready)
# -----------------------

EARLY_SEC = 15.0

def find_segment_for_line(line: str, segments: List[Dict[str, Any]]):
    """
    Find the first segment whose text contains the line (case-insensitive substring match).
    Returns (segment_dict or None).
    """
    needle = re.sub(r"\s+", " ", line.strip().lower())
    for seg in segments:
        hay = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if needle and needle in hay:
            return seg
    # fallback: loose match by token overlap
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

def score_hook(seg: Dict[str, Any]) -> (float, str, bool):
    """
    Simple scoring: prefer early, brevity (2–7s), and pattern cues (numbers, 'stop', 'wait', '?').
    Returns (score, why, is_early)
    """
    start = float(seg["start"])
    end = float(seg["end"])
    dur = max(0.01, end - start)
    text = seg["text"]

    score = 0.5
    reasons = []

    # Early bonus
    is_early = start <= EARLY_SEC
    if is_early:
        score += 0.25
        reasons.append("early (≤15s)")

    # Duration bonus (ideal 2–7s)
    if 2.0 <= dur <= 7.0:
        score += 0.2
        reasons.append("snackable length")

    # Pattern cues
    cues = 0
    cues += 1 if re.search(r"\b(stop|wait|hold|don’t|don't)\b", text, re.I) else 0
    cues += 1 if "?" in text else 0
    cues += 1 if re.search(r"\b\d+(\.\d+)?\b", text) else 0  # numbers
    if cues >= 2:
        score += 0.15
        reasons.append("strong pattern cues")
    elif cues == 1:
        score += 0.07
        reasons.append("hook cue")

    why = "; ".join(reasons) if reasons else "solid line"
    return round(score, 3), why, is_early

def build_hook_sets(hook_lines: List[str], segments: List[Dict[str, Any]]):
    """
    Map hook_lines → segments, score them, and split into
    intro_hooks (start ≤ 15s) and in_body_hooks (> 15s).
    Dedup near-duplicates; keep top 3 each.
    """
    seen_texts = set()
    intro, in_body = [], []

    for line in hook_lines:
        seg = find_segment_for_line(line, segments)
        if not seg:
            continue
        # dedupe by normalized text
        norm = re.sub(r"\s+", " ", seg["text"].strip().lower())
        if norm in seen_texts:
            continue
        seen_texts.add(norm)

        score, why, is_early = score_hook(seg)
        hook_item = {
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"],
            "score": score,
            "is_early": is_early,
            "why": why
        }
        (intro if is_early else in_body).append(hook_item)

    # sort by score desc and cap
    intro.sort(key=lambda x: x["score"], reverse=True)
    in_body.sort(key=lambda x: x["score"], reverse=True)

    return intro[:3], in_body[:3]

# -----------------------
# Routes
# -----------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "script2clipshop-worker", "version": "0.3.0"}

@app.post("/process")
async def process_video(
    video: UploadFile = File(...),
    tone: str = Form(...),
    features_csv: str = Form(...),
    product_link: str = Form(...)
):
    # 1) Read uploaded bytes
    data = await video.read()
    await video.seek(0)

    # 2) Segment + transcribe (timestamped)
    try:
        segments, transcript = segment_by_silence_and_transcribe(
            data, video.filename or "upload.mp4"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Segmentation/Transcription error: {e}")

    # 3) Classify (get raw candidate lines)
    try:
        buckets_raw = classify_text_buckets(transcript, segments, features_csv, tone)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Classification error: {e}")

    # 4) Build hook sets (intro vs in-body) with scores/why
    intro_hooks, in_body_hooks = build_hook_sets(buckets_raw["hook_lines"], segments)

    # 5) Final buckets (features/proof/cta from classifier)
    final_buckets = {
        "hooks": {
            "intro_hooks": intro_hooks,      # for default + swaps
            "in_body_hooks": in_body_hooks   # for A/B testing ideas
        },
        "features": [{"text": t} for t in buckets_raw["feature_lines"]],
        "proof":    [{"text": t} for t in buckets_raw["proof_lines"]],
        "cta":      [{"text": t} for t in buckets_raw["cta_lines"]],
    }

    # 6) Default draft picks (top of each list if exists)
    default_draft = {
        "hook_source": "intro_hooks" if intro_hooks else ("in_body_hooks" if in_body_hooks else None),
        "hook_index": 0 if (intro_hooks or in_body_hooks) else None,
        "feature_index": 0 if final_buckets["features"] else None,
        "proof_index": 0 if final_buckets["proof"] else None,
        "cta_index": 0 if final_buckets["cta"] else None
    }

    # 7) Return everything
    return JSONResponse({
        "ok": True,
        "bytes": len(data),
        "tone": tone,
        "features_csv": features_csv,
        "product_link": product_link,
        "transcript": transcript,                 # full text
        "transcript_chars": len(transcript),
        "segments": segments,                     # [{start,end,text}, ...]
        "buckets": final_buckets,                 # hooks split + feature/proof/cta lists
        "default_draft": default_draft            # initial picks (swap-ready concept)
    })
