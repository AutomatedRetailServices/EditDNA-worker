import logging
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from pipeline_errors import JobCanceledError
from job_progress import RQProgressReporter
from rq.exceptions import StopRequested

log = logging.getLogger("editdna.tasks")
log.setLevel(logging.INFO)


def benchmark_video_key(job_id: str, session_id: str) -> str:
    """Return a deterministic, collision-resistant key confined to one benchmark job."""
    import benchmark
    import benchmark_s3

    if not re.fullmatch(r"benchmark-[0-9a-f]{32}", job_id):
        raise ValueError("invalid benchmark job id")
    key = f"editdna/benchmarks/{job_id}/videos/{benchmark.safe_session_id(session_id)}.mp4"
    return benchmark_s3.validate_output_key(key, job_id)


def run_pipeline(**kwargs):
    """Load the heavyweight media pipeline only when a render actually runs."""
    from worker.pipeline import run_pipeline as pipeline_run

    return pipeline_run(**kwargs)


def job_render(
    session_id: str,
    files: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    mode: str = "human",
) -> Dict[str, Any]:
    """
    Punto de entrada que el worker RQ ejecuta como `tasks.job_render`.
    """

    # Direct callers retain the historical signature, but invalid modes are never
    # silently changed into a different editing mode.
    mode_norm = (mode or "human").lower()
    if mode_norm not in ("human", "clean", "blooper"):
        raise ValueError(f"Unsupported render mode: {mode}")

    log.info("[job_render] START session_id=%s mode=%s", session_id, mode_norm)

    reporter = RQProgressReporter()
    reporter.update("downloading", 5, "Preparing source downloads")
    try:
        result = run_pipeline(
            session_id=session_id,
            files=files,
            file_urls=file_urls,
            mode=mode_norm,
            progress=reporter.update,
            check_canceled=reporter.check_canceled,
        )
    except JobCanceledError:
        reporter.update("canceled", reporter.progress, "Render canceled")
        if reporter.job is not None:
            raise StopRequested() from None
        raise
    except Exception:
        reporter.update("failed", reporter.progress, "Render failed")
        raise

    reporter.update("finished", 100, "Render finished")

    log.info(
        f"[job_render] DONE session_id={session_id} "
        f"mode={result.get('composer_mode')} "
        f"duration={result.get('duration_sec')}"
    )

    return result


def job_benchmark(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """RQ entry point; failures are isolated by the benchmark orchestrator."""
    import benchmark
    import benchmark_s3
    from benchmark_clip_sanitizer import sanitize_benchmark_result
    from rq import get_current_job

    s3 = benchmark_s3.client()
    job = get_current_job()

    def save(state):
        if job is not None:
            job.meta.update(state)
            if "stage" not in state:
                job.meta["stage"] = "running"
            job.save_meta()

    def process(session_id, key, render_outputs, request):
        from worker.pipeline import probe_duration, run_pipeline
        with tempfile.TemporaryDirectory(prefix="editdna-benchmark-") as directory:
            path = Path(directory) / (Path(key).name or "source.mp4")
            benchmark_s3.download_video(s3, key, path)
            duration = probe_duration(str(path))
            if duration > float(os.getenv("BENCHMARK_MAX_VIDEO_SECONDS", "7200")):
                raise ValueError("video exceeds benchmark duration limit")
            safe_session = benchmark.safe_session_id(session_id)
            use_semantic_v2 = bool(request.get("use_semantic_v2", False))
            result = run_pipeline(
                session_id=f"benchmark-{job_id}-{safe_session}", local_files=[str(path)], mode="human",
                output_key=(benchmark_video_key(job_id, session_id) if render_outputs else None),
                render_output=render_outputs,
                persist_result_json=False,
                retain_local_files=False,
                use_semantic_v2=use_semantic_v2,
                use_take_judge_v2=bool(request.get("use_take_judge_v2", False)),
            )
            return sanitize_benchmark_result(result, use_semantic_v2=use_semantic_v2)

    result = benchmark.run_benchmark(job_id, payload, s3=s3, pipeline=process, progress=save)
    if not result["preflight_passed"]:
        terminal_stage = "preflight_failed"
    elif result["failed_sessions"] or result["unresolved_sessions"]:
        terminal_stage = "finished_with_failures"
    else:
        terminal_stage = "finished"
    progress_percent = 0 if terminal_stage == "preflight_failed" else 100
    terminal = {"stage": terminal_stage, "preflight_passed": result["preflight_passed"],
        "total_sessions": result["total_sessions"], "processed_sessions": result["processed_sessions"],
        "successful_sessions": result["successful_sessions"], "failed_sessions": result["failed_sessions"],
        "unresolved_sessions": result["unresolved_sessions"], "progress_percent": progress_percent,
        "errors_count": result["errors_count"], "current_session": None,
        "output_prefix": result["output_prefix"]}
    save(terminal)
    result["stage"] = terminal_stage
    result["summary"]["stage"] = terminal_stage
    return result
