import os
import json
import time
import uuid
import tempfile
import traceback
from typing import Any, Dict

from .s3_utils import upload_file_presign
from .captions import burn_captions_onto_video


def _import_jobs_module():
    """
    Try to import your job implementation.
    We expect editdna/jobs.py in the container.
    """
    try:
        import jobs  # PYTHONPATH is /workspace/editdna in the pod
        return jobs
    except Exception as e_local:
        raise ImportError(
            "Could not import job implementation from jobs.py. "
            "Make sure editdna/jobs.py exists and has run_pipeline() or job_render()."
        ) from e_local


def _coerce_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        # allow "220           # comment"
        return float(raw.strip().split()[0])
    except Exception:
        return default


def _coerce_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip().split()[0])
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is what RQ worker runs.
    The FastAPI enqueues this with your JSON body.
    """
    t0 = time.time()

    # ----- pull request fields -----
    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    files = payload.get("files", [])
    portrait = bool(payload.get("portrait", True))
    max_duration = payload.get("max_duration", None)
    output_prefix = payload.get("output_prefix", "editdna/outputs/")
    # audio flag got removed from FastAPI model, so default True here:
    audio = True

    # if caller gave no files -> fail fast so API can tell frontend
    if not isinstance(files, list) or len(files) == 0:
        return {
            "ok": False,
            "error": "No input files provided",
            "trace": None,
        }

    # ----- read knobs from env -----
    # how long final ad is allowed to be if caller didn't override
    MAX_DURATION_SEC = _coerce_float("MAX_DURATION_SEC", 220.0)
    if max_duration is None:
        max_duration = MAX_DURATION_SEC

    # caption burn mode
    captions_mode = os.getenv("CAPTIONS", "off").strip().lower()
    burn_captions = captions_mode in (
        "on", "burn", "burned", "burn_captions", "subtitle"
    )

    # S3 stuff
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")
    s3_acl = os.getenv("S3_ACL", "public-read")
    presign_expires = _coerce_int("PRESIGN_EXPIRES", 1000000)
    s3_prefix_env = os.getenv("S3_PREFIX", "editdna/outputs").rstrip("/")
    # output_prefix from payload overrides env prefix if provided
    final_prefix = (output_prefix or s3_prefix_env).rstrip("/")

    # ----- import pipeline implementation -----
    try:
        jobs = _import_jobs_module()
    except Exception as import_err:
        return {
            "ok": False,
            "error": (
                "IMPORT ERROR: could not import jobs.py in worker. "
                f"{repr(import_err)}"
            ),
            "trace": traceback.format_exc(),
        }

    # ----- run pipeline -----
    # jobs.run_pipeline MUST:
    #  - download the file_urls[0]
    #  - ASR (if ASR_ENABLED=1)
    #  - semantic cleanup / merge / pick clips
    #  - stitch with ffmpeg
    #  - return dict:
    #    {
    #      "ok": True,
    #      "input_local": "/tmp/tmpabc.mp4",
    #      "final_local": "/tmp/out_final.mp4",
    #      "duration_sec": 42.0,
    #      "clips": [...],
    #      "slots": {...}
    #    }
    try:
        pipe_result = jobs.run_pipeline(
            session_id=session_id,
            file_urls=files,
            portrait=portrait,
            max_duration=max_duration,
            audio=audio,
        )
    except Exception as pipeline_err:
        return {
            "ok": False,
            "error": f"Pipeline crashed: {repr(pipeline_err)}",
            "trace": traceback.format_exc(),
        }

    if not pipe_result or not pipe_result.get("ok"):
        return {
            "ok": False,
            "error": "Pipeline returned not-ok or empty result",
            "trace": json.dumps(pipe_result, indent=2, default=str),
        }

    final_local = pipe_result.get("final_local")
    if not final_local or not os.path.exists(final_local):
        return {
            "ok": False,
            "error": "Pipeline did not output final_local video file",
            "trace": json.dumps(pipe_result, indent=2, default=str),
        }

    # ----- optional subtitle burn -----
    if burn_captions:
        try:
            tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_out.close()
            # pipe_result["slots"] should include text chunks
            burn_slots = pipe_result.get("slots", {})
            # captions.burn_captions_onto_video(in,out,slots)
            burn_captions_onto_video(
                input_path=final_local,
                output_path=tmp_out.name,
                slots=burn_slots,
            )
            final_local = tmp_out.name
        except Exception:
            # soft fail on captions; keep going with non-burned video
            pass

    # ----- upload + presign -----
    try:
        final_key = f"{final_prefix}/{uuid.uuid4().hex}_{int(time.time())}.mp4"

        upload_info = upload_file_presign(
            file_path=final_local,
            bucket=bucket,
            region=region,
            key=final_key,
            acl=s3_acl,
            expires_in=presign_expires,
        )
    except Exception as s3_err:
        return {
            "ok": False,
            "error": f"Upload failed: {repr(s3_err)}",
            "trace": traceback.format_exc(),
        }

    t1 = time.time()

    return {
        "ok": True,
        "input_local": pipe_result.get("input_local"),
        "duration_sec": pipe_result.get("duration_sec"),
        "s3_key": upload_info.get("s3_key"),
        "s3_url": upload_info.get("s3_url"),
        "https_url": upload_info.get("https_url"),
        "clips": pipe_result.get("clips", []),
        "slots": pipe_result.get("slots", {}),
        "semantic": True,
        "vision": bool(int(os.getenv("SEMANTICS_ENABLED", "1"))),
        "asr": bool(int(os.getenv("ASR_ENABLED", "1"))),
        "runtime_sec": round(t1 - t0, 3),
    }
