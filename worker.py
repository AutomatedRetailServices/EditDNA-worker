# worker.py — RQ task definitions (full file)
import os
import time
import socket
import json
import tempfile
import shutil
import subprocess
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

import requests
import boto3

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

# =========================================================
# Utility: S3 / HTTP helpers
# =========================================================

def _parse_s3_from_url(u: str) -> Optional[Tuple[str, str]]:
    """
    Accepts:
      - https://<bucket>.s3.<region>.amazonaws.com/<key>
      - https://s3.<region>.amazonaws.com/<bucket>/<key>
      - s3://<bucket>/<key>
    Returns (bucket, key) or None if not S3.
    """
    if u.startswith("s3://"):
        p = urlparse(u)
        bucket = p.netloc
        key = p.path.lstrip("/")
        return bucket, key

    if u.startswith("http://") or u.startswith("https://"):
        p = urlparse(u)
        host = p.netloc
        path = p.path.lstrip("/")
        # virtual-hosted-style
        if ".s3." in host and host.endswith("amazonaws.com"):
            bucket = host.split(".s3.")[0]
            key = path
            return bucket, key
        # path-style
        if host.startswith("s3.") and host.endswith("amazonaws.com"):
            # /<bucket>/<key>
            parts = path.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
    return None

def _download_to_tmp(u: str, tmpdir: str, idx: int) -> str:
    """
    Downloads a file (S3 or HTTP) to local tmp dir and returns local path.
    Requires AWS creds for S3 download.
    """
    local = os.path.join(tmpdir, f"input_{idx}.mp4")
    s3_info = _parse_s3_from_url(u)

    if s3_info:
        bucket, key = s3_info
        s3 = boto3.client("s3",
                          region_name=os.getenv("AWS_REGION") or None)
        s3.download_file(bucket, key, local)
        return local

    # Fallback: plain HTTP/HTTPS file
    with requests.get(u, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return local

def _write_concat_list(paths: List[str], list_file: str) -> None:
    # ffmpeg concat demuxer list file
    with open(list_file, "w") as f:
        for p in paths:
            # escape single quotes for ffmpeg list
            esc = p.replace("'", r"'\''")
            f.write(f"file '{esc}'\n")

def _run_ffmpeg_concat(list_file: str, out_path: str) -> Tuple[bool, str]:
    """
    Re-encode to H.264/AAC, pad to 1080x1920 (vertical) preserving aspect ratio.
    """
    vf = "scale=w=1080:h=1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
    cmd = [
        "ffmpeg", "-y",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-safe", "0",
        "-f", "concat",
        "-i", list_file,
        "-ignore_unknown", "1",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ]
    try:
        cp = subprocess.run(
            cmd, check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True, cp.stderr.decode(errors="ignore")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode(errors="ignore")
        return False, err

def _upload_to_s3(local_path: str, key: str) -> str:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("Missing S3_BUCKET env var")
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

# =========================================================
# 0) Tiny test job
# =========================================================
def task_nop() -> Dict[str, Any]:
    return {"echo": {"hello": "world"}}

# =========================================================
# 1) Check S3/public URLs (HEAD request)
# payload: { "urls": [...], "session_id": "sess-..." }
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
# 2) Analyze session → make promo script (OpenAI)
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
# 3) Diagnostics
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

# =========================================================
# 4) Video rendering jobs
# =========================================================
def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Body expected by /render:
      {
        "session_id": "sess-...",
        "files": ["https://.../a.mov", "s3://bucket/key.mp4", ...],
        "output_prefix": "editdna/outputs"
      }
    """
    session_id = payload.get("session_id") or f"sess-{int(time.time())}"
    files = payload.get("files") or []
    output_prefix = payload.get("output_prefix") or "editdna/outputs"

    if not files:
        return {"ok": False, "session_id": session_id, "error": "No files provided"}

    workdir = tempfile.mkdtemp(prefix=f"editdna-{session_id}-")
    list_path = os.path.join(workdir, "concat.txt")
    out_path = os.path.join(workdir, f"{session_id}.mp4")

    try:
        local_paths: List[str] = []
        for i, u in enumerate(files):
            local = _download_to_tmp(u, workdir, i)
            local_paths.append(local)

        _write_concat_list(local_paths, list_path)

        ok, fferr = _run_ffmpeg_concat(list_path, out_path)
        if not ok:
            return {"ok": False, "session_id": session_id, "error": f"ffmpeg failed:\n{fferr}"}

        # Upload to your S3 bucket (env S3_BUCKET must be set)
        dest_key = f"{output_prefix}/{session_id}.mp4"
        s3_url = _upload_to_s3(out_path, dest_key)

        return {"ok": True, "session_id": session_id, "output": s3_url}

    except Exception as e:
        return {"ok": False, "session_id": session_id, "error": str(e)}
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

def job_render_chunked(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder: identical to job_render for now.
    """
    return job_render(payload)
