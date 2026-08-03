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
            safe_session = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id)[:160] or "session"
            result = run_pipeline(
                session_id=f"benchmark-{job_id}-{safe_session}", local_files=[str(path)], mode="human",
                output_key=(f"editdna/benchmarks/{job_id}/videos/{safe_session}.mp4" if render_outputs else None),
                render_output=render_outputs,
                persist_result_json=False,
                use_semantic_v2=bool(request.get("use_semantic_v2", False)),
                use_take_judge_v2=bool(request.get("use_take_judge_v2", False)),
            )
            return result

    result = benchmark.run_benchmark(job_id, payload, s3=s3, pipeline=process, progress=save)
    save({"stage": "finished", "current_session": None, "progress_percent": 100,
          "failed_sessions": result["summary"]["failed_sessions"],
          "errors_count": result["summary"]["failed_sessions"], "output_prefix": result["output_prefix"]})
    return result
