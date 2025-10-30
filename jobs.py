import os
import json
import time
import uuid
import tempfile
import traceback
from typing import Any, Dict, Optional

from .s3_utils import upload_file_presign
from .captions import burn_captions_onto_video
from .semantic_visual_pass import semantic_visual_pass

#
# small helper:
# this tries to import your pipeline logic from jobs.py (or worker/jobs.py if you ever split later)
#
def _import_jobs_module():
    # 1. try local editdna/jobs.py
    try:
        import jobs as local_jobs  # because PYTHONPATH=/workspace/editdna
        return local_jobs
    except Exception as e_local:
        local_err = e_local

    # 2. try worker.jobs (in case you ever move code under /workspace/editdna/worker/jobs.py)
    try:
        from worker import jobs as worker_jobs  # type: ignore
        return worker_jobs
    except Exception as e_worker:
        worker_err = e_worker

    # 3. nothing worked
    raise ImportError(
        "Could not import job implementation from jobs.py or worker/jobs.py.\n"
        f"jobs.py error: {repr(local_err)}\n"
        f"worker/jobs.py error: {repr(worker_err)}"
    )


def _coerce_env_float(name: str, default: float) -> float:
    val = os.getenv(name, None)
    if val is None:
        return default
    try:
        return float(val.strip().split()[0])
    except Exception:
        return default


def _coerce_env_int(name: str, default: int) -> int:
    val = os.getenv(name, None)
    if val is None:
        return default
    try:
        return int(val.strip().split()[0])
    except Exception:
        return default


def _choose_funnel_counts(payload_opts: Dict[str, Any]) -> str:
    """
    Order: payload.options.FUNNEL_COUNTS > ENV FUNNEL_COUNTS > default "1,3,3,3,1"
    """
    # 1. request override
    if "options" in payload_opts and isinstance(payload_opts["options"], dict):
        fc = payload_opts["options"].get("FUNNEL_COUNTS")
        if isinstance(fc, str) and fc.strip():
            return fc.strip()

    # 2. env
    env_fc = os.getenv("FUNNEL_COUNTS")
    if env_fc and env_fc.strip():
        return env_fc.strip()

    # 3. fallback
    return "1,3,3,3,1"


def _choose_max_duration_sec(payload_opts: Dict[str, Any]) -> int:
    """
    Order: payload.options.MAX_DURATION_SEC > ENV MAX_DURATION_SEC > default 220
    """
    # request override
    if "options" in payload_opts and isinstance(payload_opts["options"], dict):
        md = payload_opts["options"].get("MAX_DURATION_SEC")
        if md is not None:
            try:
                return int(str(md).strip().split()[0])
            except Exception:
                pass

    # env
    env_md = os.getenv("MAX_DURATION_SEC")
    if env_md:
        try:
            return int(env_md.strip().split()[0])
        except Exception:
            pass

    # default
    return 220


def _safe_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    raw = raw.strip().lower()
    return raw in ("1", "true", "yes", "on")


def job_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is what RQ calls.
    payload is what your FastAPI sent to Redis.
    We will:
    1. download / load the source video(s)
    2. run semantic / asr / merging logic (your pipeline)
    3. stitch final edit with ffmpeg
    4. optionally burn captions
    5. upload to s3 and presign
    6. return JSON
    """

    t0 = time.time()

    # --- basic request fields ---
    session_id = payload.get("session_id", f"session-{uuid.uuid4().hex[:8]}")
    files = payload.get("files", [])
    if not isinstance(files, list) or len(files) == 0:
        return {
            "ok": False,
            "error": "No input files provided",
            "trace": None,
        }

    # portrait flag (vertical), default True
    portrait = bool(payload.get("portrait", True))

    # choose funnel counts
    funnel_counts = _choose_funnel_counts(payload)

    # choose max clip length
    max_duration_sec = _choose_max_duration_sec(payload)

    # env-driven knobs
    bin_sec = _coerce_env_float("BIN_SEC", 1.0)                 # how we bucket timestamps
    min_take_sec = _coerce_env_float("MIN_TAKE_SEC", 2.0)       # min chunk
    max_take_sec = _coerce_env_float("MAX_TAKE_SEC", 220.0)     # max chunk
    veto_min_score = _coerce_env_float("VETO_MIN_SCORE", 0.35)  # if below, drop

    # merge logic tuning
    sem_merge_sim = _coerce_env_float("SEM_MERGE_SIM", 0.70)
    viz_merge_sim = _coerce_env_float("VIZ_MERGE_SIM", 0.70)
    merge_max_chain = _coerce_env_int("MERGE_MAX_CHAIN", 12)

    # filler cleanup
    sem_filler_list = os.getenv("SEM_FILLER_LIST", "um,uh,like,so,okay")
    filler_tokens = [w.strip() for w in sem_filler_list.split(",") if w.strip()]
    sem_filler_max_rate = _coerce_env_float("SEM_FILLER_MAX_RATE", 0.08)

    # micro-cut logic
    micro_cut_enabled = _safe_bool_env("MICRO_CUT", True)
    micro_silence_db = _coerce_env_float("MICRO_SILENCE_DB", -30.0)
    micro_silence_min = _coerce_env_float("MICRO_SILENCE_MIN", 0.25)

    # slot requirements for FEATURE/PROOF/CTA
    slot_require_product = os.getenv("SLOT_REQUIRE_PRODUCT", "FEATURE,PROOF")
    slot_require_product = [s.strip().upper() for s in slot_require_product.split(",") if s.strip()]
    slot_require_ocr_cta = os.getenv("SLOT_REQUIRE_OCR_CTA", "CTA").strip().upper()

    # S3 config
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")
    s3_prefix = os.getenv("S3_PREFIX", "editdna/outputs")
    s3_acl = os.getenv("S3_ACL", "public-read")
    presign_expires = int(os.getenv("PRESIGN_EXPIRES", "1000000").strip().split()[0])

    # captions on/off (env)
    captions_mode = os.getenv("CAPTIONS", "off").strip().lower()
    burn_captions = captions_mode in ("on", "burn", "burned", "burn_captions", "subtitle")

    # final fallback runtime minimum if pipeline can't build funnel
    fallback_min_sec = _coerce_env_int("FALLBACK_MIN_SEC", 60)

    # ---- IMPORT USER PIPELINE MODULE ----
    try:
        jobs = _import_jobs_module()
    except Exception as import_err:
        return {
            "ok": False,
            "error": (
                "Could not import job implementation.\n"
                f"{repr(import_err)}"
            ),
            "trace": traceback.format_exc(),
        }

    # ---- CALL YOUR PIPELINE LOGIC ----
    # We expect your jobs.run_pipeline() to do:
    # - ASR (if ASR_ENABLED=1)
    # - semantic scoring
    # - filler cleanup
    # - segment merge
    # - select funnel slots (HOOK/PROBLEM/FEATURE/PROOF/CTA)
    # - stitch final ffmpeg cut
    #
    # It MUST return a dict like:
    # {
    #   "ok": True,
    #   "input_local": "/tmp/tmpabc.mp4",
    #   "final_local": "/tmp/out_final.mp4",
    #   "duration_sec": 42.0,
    #   "clips": [...],
    #   "slots": {...}
    # }
    #
    try:
        result = jobs.run_pipeline(
            session_id=session_id,
            file_urls=files,
            portrait=portrait,
            funnel_counts=funnel_counts,
            max_duration=max_duration_sec,
            bin_sec=bin_sec,
            min_take_sec=min_take_sec,
            max_take_sec=max_take_sec,
            veto_min_score=veto_min_score,
            sem_merge_sim=sem_merge_sim,
            viz_merge_sim=viz_merge_sim,
            merge_max_chain=merge_max_chain,
            filler_tokens=filler_tokens,
            filler_max_rate=sem_filler_max_rate,
            micro_cut=micro_cut_enabled,
            micro_silence_db=micro_silence_db,
            micro_silence_min=micro_silence_min,
            slot_require_product=slot_require_product,
            slot_require_ocr_cta=slot_require_ocr_cta,
            fallback_min_sec=fallback_min_sec,
        )
    except Exception as pipeline_err:
        return {
            "ok": False,
            "error": f"Pipeline crashed: {repr(pipeline_err)}",
            "trace": traceback.format_exc(),
        }

    if not result or not result.get("ok"):
        return {
            "ok": False,
            "error": "Pipeline returned not-ok or empty result",
            "trace": json.dumps(result, indent=2, default=str),
        }

    final_local = result.get("final_local")
    if not final_local or not os.path.exists(final_local):
        return {
            "ok": False,
            "error": "Pipeline did not output final_local video file",
            "trace": json.dumps(result, indent=2, default=str),
        }

    # ---- OPTIONAL CAPTIONS BURN ----
    if burn_captions:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
                burn_captions_onto_video(
                    input_path=final_local,
                    output_path=tmp_out.name,
                    slots=result.get("slots", {}),
                )
                final_local = tmp_out.name
        except Exception as cap_err:
            # we won't fail the job if caption burn fails
            pass

    # ---- UPLOAD TO S3 ----
    try:
        base_key = f"{uuid.uuid4().hex}_{int(time.time())}.mp4"
        s3_key = f"{s3_prefix.rstrip('/')}/{base_key}"

        upload_info = upload_file_presign(
            file_path=final_local,
            bucket=bucket,
            region=region,
            key=s3_key,
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
        "input_local": result.get("input_local"),
        "duration_sec": result.get("duration_sec"),
        "s3_key": upload_info.get("s3_key"),
        "s3_url": upload_info.get("s3_url"),
        "https_url": upload_info.get("https_url"),
        "clips": result.get("clips", []),
        "slots": result.get("slots", {}),
        "semantic": True,
        "vision": bool(int(os.getenv("SEMANTICS_ENABLED", "1"))),
        "asr": bool(int(os.getenv("ASR_ENABLED", "1"))),
        "runtime_sec": round(t1 - t0, 3),
    }
