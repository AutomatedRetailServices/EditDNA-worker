# worker.py — RQ task definitions (render + analyze + diagnostics)
import os
import time
import socket
import json
import subprocess
import tempfile
from typing import Any, Dict, List
import requests

# -------- OpenAI (v1.x) --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    from openai import OpenAI
    _openai_import_ok = True
except Exception:
    OpenAI = None  # type: ignore
    _openai_import_ok = False

_client = None
if _openai_import_ok and OPENAI_API_KEY:
    try:
        _client = OpenAI(api_key=OPENAI_API_KEY)
        _openai_client_ok = True
    except Exception:
        _openai_client_ok = False
else:
    _openai_client_ok = False

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def _safe_run(cmd: List[str]) -> Dict[str, Any]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return {"ok": False, "stdout": p.stdout, "stderr": p.stderr, "code": p.returncode}
        return {"ok": True, "stdout": p.stdout, "stderr": p.stderr, "code": 0}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "code": -1}

def _write_concat_txt(paths: List[str], txt_path: str) -> None:
    # allow HTTPS inputs for concat demuxer: use -safe 0 AND prepend "file " with quoted URL
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in paths:
            # escape single quotes for ffmpeg concat format
            esc = p.replace("'", "'\\''")
            f.write(f"file '{esc}'\n")

def _ffmpeg_concat_to_mp4(sources: List[str], out_path: str, portrait: bool = True) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="editdna-") as td:
        concat_txt = os.path.join(td, "concat.txt")
        _write_concat_txt(sources, concat_txt)

        vf = "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black" if portrait \
            else "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"

        cmd = [
            "ffmpeg",
            "-y",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-safe", "0",
            "-f", "concat",
            "-i", concat_txt,
            "-ignore_unknown", "1",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ]
        return _safe_run(cmd)

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# =========================================================
# 0) Tiny test job
# =========================================================
def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}

# =========================================================
# 1) Check S3/public URLs (HEAD)
# =========================================================
def _head(url: str, timeout: float = 20.0) -> Dict[str, Any]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        size = int(r.headers.get("Content-Length", "0") or 0)
        return {
            "url": url,
            "status": "OK",
            "http": r.status_code,
            "size": size,
            "method": "HEAD",
        }
    except Exception as e:
        return {
            "url": url,
            "status": "ERROR",
            "http": 0,
            "size": 0,
            "method": "HEAD",
            "error": str(e),
        }

def check_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    urls = payload.get("urls") or []
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    checked = [_head(u) for u in urls]
    return {"session_id": sess, "checked": checked}

# =========================================================
# 2) Analyze session (OpenAI optional)
# =========================================================
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _make_prompt(session_id: str, tone: str, product_link: str, features: List[str]) -> str:
    feat_str = ", ".join(features) if features else "key features"
    return (
        f"You are a marketing writer. Create a short {tone or 'neutral'} promo script "
        f"for product {product_link or '(no link)'} based on these features: {feat_str}. "
        f"Keep it 3–5 sentences, engaging, and suitable for voiceover."
    )

def _parse_features(payload: Dict[str, Any]) -> List[str]:
    feats: List[str] = []
    csv = (payload.get("features_csv") or "").strip()
    if csv:
        feats = [p.strip() for p in csv.split(",") if p.strip()]
    if not feats and isinstance(payload.get("features"), list):
        feats = [str(x).strip() for x in payload["features"] if str(x).strip()]
    return feats

def analyze_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    sess = payload.get("session_id") or f"sess-{int(time.time())}"
    tone = str(payload.get("tone") or "neutral")
    product_link = str(payload.get("product_link") or "")
    features = _parse_features(payload)

    stub = {
        "session_id": sess,
        "engine": "stub",
        "script": f"[DEV STUB] {tone} promo highlighting {', '.join(features) or 'features'}. "
                  f"Product: {product_link or '(no link)'}.",
        "product_link": product_link or None,
        "features": features,
    }

    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        return stub

    diag = {"has_key": bool(OPENAI_API_KEY), "import_ok": _openai_import_ok,
            "client_ok": _openai_client_ok, "attempts": [], "last_error": None}

    prompt = _make_prompt(sess, tone, product_link, features)
    try:
        diag["attempts"].append({"api": "chat.completions", "ok": None, "error": None})
        resp = _client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You write concise, friendly promo scripts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        diag["attempts"][-1]["ok"] = True
        return {
            "session_id": sess,
            "engine": "openai",
            "openai_diag": diag,
            "script": text,
            "product_link": product_link or None,
            "features": features,
        }
    except Exception as e:
        diag["attempts"][-1]["ok"] = False
        diag["attempts"][-1]["error"] = str(e)
        diag["last_error"] = str(e)
        out = dict(stub)
        out["engine"] = "stub"
        out["openai_diag"] = diag
        return out

# =========================================================
# 3) Render (S3/public URLs) — SINGLE PAYLOAD ARG
# payload = {
#   "session_id": "...",
#   "files": [https/s3 urls],
#   "output_prefix": "editdna/outputs"  # optional passthrough
# }
# returns { ok, session_id, output | error }
# =========================================================
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    files: List[str] = payload.get("files") or []
    out_prefix = payload.get("output_prefix")  # not used here, just passthrough if you later upload

    if not files:
        return {"ok": False, "session_id": session_id, "error": "No input files provided."}

    # output file in /tmp
    td = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    out_mp4 = os.path.join(td, f"{session_id}.mp4")

    # do concat (portrait safe)
    r = _ffmpeg_concat_to_mp4(files, out_mp4, portrait=True)
    if not r.get("ok"):
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"ffmpeg failed:\n{r.get('stderr','')}"
        }

    return {"ok": True, "session_id": session_id, "output": out_mp4}

# =========================================================
# 4) Render (chunked) — SINGLE PAYLOAD ARG
# payload = { session_id, files, chunk_size, output_prefix }
# =========================================================
def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    files: List[str] = payload.get("files") or []
    chunk_size = int(payload.get("chunk_size") or 8)
    if chunk_size < 2:
        chunk_size = 2

    if not files:
        return {"ok": False, "session_id": session_id, "error": "No input files provided."}

    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    partials: List[str] = []
    # 1) Render chunks
    for idx, group in enumerate(_chunk(files, chunk_size), start=1):
        part_out = os.path.join(workdir, f"part_{idx:03d}.mp4")
        r = _ffmpeg_concat_to_mp4(group, part_out, portrait=True)
        if not r.get("ok"):
            return {"ok": False, "session_id": session_id, "error": f"ffmpeg chunk {idx} failed:\n{r.get('stderr','')}"}
        partials.append(part_out)

    # 2) Stitch partials
    final_out = os.path.join(workdir, f"{session_id}.mp4")
    r2 = _ffmpeg_concat_to_mp4(partials, final_out, portrait=True)
    if not r2.get("ok"):
        return {"ok": False, "session_id": session_id, "error": f"ffmpeg final merge failed:\n{r2.get('stderr','')}"}

    return {"ok": True, "session_id": session_id, "output": final_out}

# =========================================================
# 5) Diagnostics
# =========================================================
def diag_openai() -> Dict[str, Any]:
    result = {
        "has_key": bool(OPENAI_API_KEY),
        "import_ok": _openai_import_ok,
        "client_ok": _openai_client_ok,
        "attempts": [],
        "last_error": None,
        "reply": None,
    }
    if not (_openai_import_ok and _openai_client_ok and _client and OPENAI_API_KEY):
        result["last_error"] = "SDK/client not initialized."
        return result
    try:
        result["attempts"].append({"api": "chat.completions", "ok": None, "error": None})
        r = _client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a health-check bot."},
                {"role": "user", "content": "Reply with the single word: ok"},
            ],
            max_tokens=3,
        )
        msg = (r.choices[0].message.content or "").strip()
        result["reply"] = msg
        result["attempts"][-1]["ok"] = True
        return result
    except Exception as e:
        result["attempts"][-1]["ok"] = False
        result["attempts"][-1]["error"] = str(e)
        result["last_error"] = str(e)
        return result

def net_probe() -> Dict[str, Any]:
    out = {"dns": None, "tls": None}
    try:
        socket.gethostbyname("api.openai.com")
        out["dns"] = "ok"
    except Exception as e:
        out["dns"] = f"fail: {e}"
    try:
        r = requests.get("https://api.openai.com/v1/models", timeout=10)
        out["tls"] = f"ok: TLSv{getattr(r.raw, 'version', '')}" if hasattr(r.raw, "version") else "ok"
    except Exception as e:
        out["tls"] = f"fail: {e}"
    return out
