# jobs.py — main render logic for EditDNA worker
from __future__ import annotations

# --- robust import for the semantic pass ---
def _noop_continuity(takes):
    # fallback: return each take alone
    return [[t] for t in takes]

try:
    # preferred: package style (when repo root is PYTHONPATH and worker/ is a package)
    from worker.semantic_visual_pass import continuity_chains  # type: ignore
    print("[jobs.py] Imported semantic pass (package).", flush=True)
except Exception as e1:
    try:
        # fallback: flat style (if the file sits next to jobs.py or PYTHONPATH points at worker/)
        from semantic_visual_pass import continuity_chains  # type: ignore
        print("[jobs.py] Imported semantic pass (flat).", flush=True)
    except Exception as e2:
        print(f"[jobs.py] Semantic module unavailable ({e1 or e2}); falling back to no-op.", flush=True)
        continuity_chains = _noop_continuity

import os, json, time, uuid, boto3
from typing import Dict, Any

# --- AWS setup ---
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=AWS_REGION)

def s3_put_text(bucket: str, key: str, text: str):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/plain")

# (optional) ASR import placeholder
try:
    import whisper  # noqa: F401
except Exception:
    whisper = None  # type: ignore

def job_render(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified entry for EditDNA worker job (semantic-aware placeholder).
    """
    started = time.time()
    session_id = params.get("session_id", f"job-{uuid.uuid4().hex[:6]}")
    mode = params.get("mode", "funnel")
    files = params.get("files", [])
    output_prefix = params.get("output_prefix", "editdna/outputs")

    print(f"🚀 [jobs.py] Job start | session={session_id} | mode={mode} | file_count={len(files)}", flush=True)

    # --- placeholder segmentation step (replace with real ASR/FFmpeg pipeline as needed) ---
    takes = []
    for f in files:
        takes.append({"text": f"Placeholder segment for {f}", "start": 0.0, "end": 3.0})

    # --- semantic chaining ---
    print(f"🧩 [jobs.py] Invoking semantic continuity on {len(takes)} takes...", flush=True)
    try:
        chains = continuity_chains([type("T", (), t) for t in takes])
        print(f"✅ [jobs.py] Semantic continuity returned {len(chains)} chain(s).", flush=True)
    except Exception as e:
        print(f"❌ [jobs.py] Semantic continuity failed: {e}", flush=True)
        chains = [[type("T", (), t) for t in takes]]

    # --- fake output manifest (so the web API shows a result) ---
    out_name = f"{output_prefix}/{session_id}/out_clips.mp4"
    manifest = {
        "ok": True,
        "session_id": session_id,
        "mode": mode,
        "output_s3": f"s3://{S3_BUCKET}/{out_name}",
        "output_url": f"https://{S3_BUCKET}.s3.amazonaws.com/{out_name}",
        "inputs": files,
        "chains_count": len(chains),
    }

    key = f"{output_prefix}/{session_id}/manifest.json"
    s3_put_text(S3_BUCKET, key, json.dumps(manifest, indent=2))
    elapsed = time.time() - started
    print(f"🏁 [jobs.py] Job complete in {elapsed:.2f}s | output={out_name}", flush=True)
    return manifest
