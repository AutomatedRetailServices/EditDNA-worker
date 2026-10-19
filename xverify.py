"""Read-only: render the creator's own failing draft (19 Oct) with the fixed code, to a temporary file only."""
import json
import os
import time

import modal

app = modal.App("cutsell-export-verify")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .env({"CUTSELL_ENGINE": "simple", "CUTSELL_EXPORT_DELIVERY_GATE": "technical_qc"})
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
PROJECT = "prj_8b0a6e1cee4b4887ba5bb89a94de7905"


@app.function(image=image, secrets=[secret], timeout=900, cpu=2.0, memory=4096)
def verify():
    import tempfile
    from pathlib import Path
    from redis import Redis
    from cutsell_worker.storage import download_source
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.serde import draft_from_dict
    from cutsell_worker.render_plan import build_render_plan
    from cutsell_worker.live_render_qc import render_with_post_render_qc
    r = Redis.from_url(os.environ["REDIS_URL"])
    snap = None
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            s = json.loads(r.get(key))
        except Exception:
            continue
        if s.get("project_id") == PROJECT:
            snap = s
            break
    if not snap:
        return {"error": "draft not found"}
    draft = draft_from_dict(dict(snap["draft"]))
    out = {"selected": len(draft.selected), "last_clip_end": draft.selected[-1].end,
           "overlap": bool(getattr(draft, "audio_overlap_enabled", False))}
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for i, src in enumerate(snap["sources"]):
            paths[src["source_asset_id"]] = download_source(src["uri"], str(Path(d) / f"s{i}.mp4"))
            out[f"source_{i}_duration"] = probe_media(paths[src["source_asset_id"]]).duration_sec
        from cutsell_worker.render_plan import bound_draft_to_sources
        draft = bound_draft_to_sources(draft, paths)
        plan = build_render_plan(draft, paths)
        out["plan_last_end"] = plan[-1].end
        t = time.time()
        qc = render_with_post_render_qc(draft, plan, str(Path(d) / "e.mp4"), text_overlays=draft.text_overlays)
        out["render_s"] = round(time.time() - t, 1)
        out["qc"] = qc.status
        out["export_duration"] = probe_media(str(Path(d) / "e.mp4")).duration_sec
    return out


@app.local_entrypoint()
def main():
    out = verify.remote()
    with open("engine_check.json", "w") as f:
        json.dump(out, f, default=str)
    print("::notice title=verify::" + json.dumps({k: out.get(k) for k in ("qc", "render_s", "error")}))
