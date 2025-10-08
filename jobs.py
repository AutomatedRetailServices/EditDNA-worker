# jobs.py — main render logic for EditDNA worker
from __future__ import annotations
import os, json, time, uuid, boto3
from typing import Dict, Any

# --- AWS setup ---
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

def s3_put_text(bucket: str, key: str, text: str):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/plain")

# --- semantic import ---
try:
    from worker.semantic_visual_pass import chain_continuity
    print("🧠 [jobs.py] Semantic module loaded successfully.", flush=True)
except Exception as e:
    print(f"⚠️ [jobs.py] Semantic module unavailable ({e}); falling back to no-op.", flush=True)
    def chain_continuity(takes):
        return [[t] for t in takes]

# --- ASR / base imports ---
try:
    import whisper
except Exception:
    whisper = None

def job_render(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified entry for EditDNA worker job (semantic-aware).
    """
    started = time.time()
    session_id = params.get("session_id", f"job-{uuid.uuid4().hex[:6]}")
    mode = params.get("mode", "funnel")
    files = params.get("files", [])
    output_prefix = params.get("output_prefix", "editdna/outputs")

    print(f"🚀 [jobs.py] Job start | session={session_id} | mode={mode} | file_count={len(files)}", flush=True)

    # --- placeholder ASR segmentation step ---
    takes = []
    for f in files:
        takes.append({"text": f"Placeholder segment for {f}", "start": 0.0, "end": 3.0})

    # --- semantic chaining ---
    print(f"🧩 [jobs.py] Invoking semantic continuity on {len(takes)} takes...", flush=True)
    try:
        chains = chain_continuity([
            type("T", (), t) for t in takes
        ])
        print(f"✅ [jobs.py] Semantic continuity returned {len(chains)} chain(s).", flush=True)
    except Exception as e:
        print(f"❌ [jobs.py] Semantic continuity failed: {e}", flush=True)
        chains = [[type("T", (), t) for t in takes]]

    # --- fake render output (placeholder) ---
    out_name = f"{output_prefix}/{session_id}/out_clips.mp4"
    manifest = {
        "ok": True,
        "session_id": session_id,
        "mode": mode,
        "output_s3": f"s3://{S3_BUCKET}/{out_name}",
        "output_url": f"https://{S3_BUCKET}.s3.amazonaws.com/{out_name}",
        "inputs": files,
    }

    key = f"{output_prefix}/{session_id}/manifest.json"
    s3_put_text(S3_BUCKET, key, json.dumps(manifest, indent=2))
    elapsed = time.time() - started
    print(f"🏁 [jobs.py] Job complete in {elapsed:.2f}s | output={out_name}", flush=True)
    return manifest
