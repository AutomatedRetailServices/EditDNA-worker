"""One-off diagnosis (PO 5 Nov: "No pudimos terminar este corte").

Read-only: reads ONE project's record (by storage hash, never its id) and its latest job from the queue, and prints
the job status, stage and the error trace with links and numbers-like secrets stripped. No transcript, no titles.
Nothing is written. Run: modal run ops/diag_job_error.py (ephemeral app).
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
app = modal.App("cutsell-diag-job-error")
KEY = "cutsell:v1:project:ee497966354368c2f57f:452782ed0d89107f4f31"


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=300, cpu=0.5)
def diagnose():
    import json, os, re
    from redis import Redis
    from rq.job import Job

    r = Redis.from_url(os.environ["REDIS_URL"])
    raw = r.get(KEY)
    if not raw:
        print("project not found"); return
    p = json.loads(raw)
    srcs = p.get("sources") or []
    print("project state=%s sources=%d durations=%s types=%s updated=%s" % (
        p.get("state"), len(srcs), [round(float(s.get("duration_sec") or 0), 1) for s in srcs if isinstance(s, dict)],
        [str(s.get("content_type") or s.get("mime") or "")[:20] for s in srcs if isinstance(s, dict)], str(p.get("updated_at"))[:19]))
    for k in ("failure", "error", "last_error", "failure_reason", "error_code"):
        if p.get(k):
            print(f"project.{k}:", re.sub(r"https?://\S+", "<link>", str(p[k]))[:400])
    jid = p.get("latest_job_id")
    print("has latest job:", bool(jid))
    if not jid:
        return
    try:
        job = Job.fetch(jid, connection=r)
    except Exception as exc:
        print("job not in queue:", type(exc).__name__); return
    meta = job.meta or {}
    print("job status=%s func=%s enqueued=%s started=%s ended=%s" % (job.get_status(), (job.func_name or "")[-60:],
          str(job.enqueued_at)[:19], str(job.started_at)[:19], str(job.ended_at)[:19]))
    print("meta keys:", sorted(meta.keys())[:30])
    for k in ("stage", "progress", "error", "error_code", "failure", "message", "engine", "diagnostics_error"):
        if k in meta:
            print(f"meta.{k}:", re.sub(r"https?://\S+", "<link>", str(meta[k]))[:400])
    exc = job.latest_result().exc_string if hasattr(job, "latest_result") and job.latest_result() else (job.exc_info or "")
    if exc:
        lines = [re.sub(r"https?://\S+", "<link>", l) for l in str(exc).splitlines()]
        print("--- error trace (last 40 lines) ---")
        for l in lines[-40:]:
            print(l[:300])


@app.local_entrypoint()
def main():
    diagnose.remote()
