"""Read-only repro: the last failed export job (its own payload: draft + sources) rendered again here with the
live code (a0addc8) and with feat/keyframes-smooth, the post-render QC findings of each. Nothing is uploaded or
stored. Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-qcrepro-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
         .add_local_dir("old/cutsell_worker", remote_path="/root/old/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))


def _render(pkg, payload, directory):
    import importlib, sys, dataclasses
    from pathlib import Path
    ej = importlib.import_module(f"{pkg}.export_job")
    serde = importlib.import_module(f"{pkg}.serde")
    rp = importlib.import_module(f"{pkg}.render_plan")
    qc = importlib.import_module(f"{pkg}.live_render_qc")
    draft = serde.draft_from_dict(dict(payload["draft"]))
    local = {}
    for i, item in enumerate(payload.get("sources") or ()):
        dest = str(Path(directory) / f"src-{pkg}-{i}.mp4")
        local[str(item["source_asset_id"])] = ej.download_source(str(item["uri"]), dest)
    draft = ej.bound_draft_to_sources(draft, local)
    plan = rp.build_render_plan(draft, local)
    res = qc.render_with_post_render_qc(draft, plan, str(Path(directory) / f"out-{pkg}.mp4"),
                                        text_overlays=draft.text_overlays, media_overlays=())
    return {"status": res.status,
            "segments": [{"start": s.start, "end": s.end, "keys": [list(p) for p in s.transform_keyframes]} for s in plan],
            "attempts": [{"status": a.status, "findings": list(a.findings)} for a in res.attempts]}


@app.function(image=image, secrets=[secret], timeout=900, cpu=4)
def run():
    import sys, tempfile
    sys.path.insert(0, "/root/old")
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq.registry import FailedJobRegistry
    r = Redis.from_url(os.environ["REDIS_URL"])
    reg = FailedJobRegistry(queue=Queue("cutsell", connection=r))
    jobs = []
    for jid in reg.get_job_ids()[-40:]:
        try:
            j = Job.fetch(jid, connection=r)
        except Exception:
            continue
        if j.func_name.endswith("run_export_job"):
            jobs.append(j)
    jobs.sort(key=lambda j: str(j.ended_at))
    payload = dict(jobs[-1].args[0])
    report = {"keyframes": payload["draft"].get("keyframes"), "selected": [
        {k: c.get(k) for k in ("clip_id", "start", "end", "source_asset_id")} for c in payload["draft"]["selected"]]}
    for pkg in ("old.cutsell_worker", "cutsell_worker"):
        with tempfile.TemporaryDirectory() as d:
            try:
                report[pkg] = _render(pkg, payload, d)
            except Exception as exc:
                report[pkg] = {"error": repr(exc)[:400]}
    return report


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump(run.remote(), f, ensure_ascii=False, default=str)
