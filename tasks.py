import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
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


def _attach_semantic_execution(result: Dict[str, Any], *, requested: bool) -> Dict[str, Any]:
    """Decorate a real pipeline result without changing legacy test/stub shapes."""
    if not any(key in result for key in ("clips", "processed_source_indices", "input_file_count")):
        return result
    from worker.execution_observability import attach_semantic_execution

    return attach_semantic_execution(result, requested=requested)


def run_pipeline(**kwargs):
    """Load the heavyweight media pipeline only when a render actually runs."""
    import worker.pipeline as pipeline_module

    result = pipeline_module.run_pipeline(**kwargs)
    semantic_requested = bool(
        kwargs.get("use_semantic_v2", False) or pipeline_module.EDITDNA_USE_LLM
    )
    return _attach_semantic_execution(result, requested=semantic_requested)


@contextmanager
def benchmark_semantic_v2_setting(enabled: bool):
    """Make the benchmark request authoritative for Semantic V2 execution.

    The production pipeline normally honors its global EDITDNA_USE_LLM setting.
    Benchmarks need an explicit per-request off switch so an old-vs-new run with
    use_semantic_v2=false cannot be influenced by Semantic V2 at all. RQ executes
    one job at a time per worker process, and the original setting is restored in
    a finally block before the next job can run.
    """
    import worker.pipeline as pipeline_module

    original = pipeline_module.EDITDNA_USE_LLM
    if not enabled:
        pipeline_module.EDITDNA_USE_LLM = False
    try:
        yield pipeline_module
    finally:
        pipeline_module.EDITDNA_USE_LLM = original


def run_benchmark_pipeline(*, use_semantic_v2: bool, **kwargs):
    """Run one benchmark analysis with an authoritative Semantic V2 flag."""
    with benchmark_semantic_v2_setting(use_semantic_v2) as pipeline_module:
        result = pipeline_module.run_pipeline(
            use_semantic_v2=use_semantic_v2,
            **kwargs,
        )
    return _attach_semantic_execution(result, requested=use_semantic_v2)


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
    from worker.benchmark_observability import (
        collect_semantic_execution_summary,
        embed_semantic_execution,
    )

    s3 = benchmark_s3.client()
    job = get_current_job()

    def save(state):
        if job is not None:
            job.meta.update(state)
            if "stage" not in state:
                job.meta["stage"] = "running"
            job.save_meta()

    def process(session_id, key, render_outputs, request):
        from worker.pipeline import probe_duration
        with tempfile.TemporaryDirectory(prefix="editdna-benchmark-") as directory:
            path = Path(directory) / (Path(key).name or "source.mp4")
            benchmark_s3.download_video(s3, key, path)
            duration = probe_duration(str(path))
            if duration > float(os.getenv("BENCHMARK_MAX_VIDEO_SECONDS", "7200")):
                raise ValueError("video exceeds benchmark duration limit")
            safe_session = benchmark.safe_session_id(session_id)
            use_semantic_v2 = bool(request.get("use_semantic_v2", False))
            result = run_benchmark_pipeline(
                use_semantic_v2=use_semantic_v2,
                session_id=f"benchmark-{job_id}-{safe_session}", local_files=[str(path)], mode="human",
                output_key=(benchmark_video_key(job_id, session_id) if render_outputs else None),
                render_output=render_outputs,
                persist_result_json=False,
                retain_local_files=False,
                use_take_judge_v2=bool(request.get("use_take_judge_v2", False)),
            )
            embed_semantic_execution(result)
            return sanitize_benchmark_result(result, use_semantic_v2=use_semantic_v2)

    result = benchmark.run_benchmark(job_id, payload, s3=s3, pipeline=process, progress=save)
    if not result["preflight_passed"]:
        terminal_stage = "preflight_failed"
    elif result["failed_sessions"] or result["unresolved_sessions"]:
        terminal_stage = "finished_with_failures"
    else:
        terminal_stage = "finished"
    progress_percent = 0 if terminal_stage == "preflight_failed" else 100
    result["stage"] = terminal_stage
    result["summary"]["stage"] = terminal_stage

    if result["preflight_passed"] and payload.get("mode") != "inventory_only":
        semantic_summary = collect_semantic_execution_summary(
            s3,
            job_id,
            result["output_prefix"],
            benchmark_s3.read_output,
        )
        result["summary"].update(semantic_summary)
        benchmark_s3.put_output(
            s3,
            result["output_prefix"] + "summary.json",
            json.dumps(result["summary"], indent=2).encode(),
            job_id,
            "application/json",
        )
        benchmark_s3.put_output(
            s3,
            result["output_prefix"] + "summary.csv",
            benchmark.summary_csv(result["summary"]),
            job_id,
            "text/csv",
        )

    terminal = {"stage": terminal_stage, "preflight_passed": result["preflight_passed"],
        "total_sessions": result["total_sessions"], "processed_sessions": result["processed_sessions"],
        "successful_sessions": result["successful_sessions"], "failed_sessions": result["failed_sessions"],
        "unresolved_sessions": result["unresolved_sessions"], "progress_percent": progress_percent,
        "errors_count": result["errors_count"], "current_session": None,
        "output_prefix": result["output_prefix"]}
    save(terminal)
    return result
