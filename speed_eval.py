"""Speed check (read-only): (A) real job timings from the RQ queue -- queued -> started (wake + cold start),
started -> ended; (B) the live engine's stages timed on 2 recent real videos of the owner's account
(download, probe, format gate, audio, Deepgram, each Claude call, refine/snap, scenes). Nothing written
to Redis, storage or the projects. Ephemeral `modal run`."""
import json
import os
import time

import modal

app = modal.App("cutsell-speed-check")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .env({"CUTSELL_ENGINE": "simple"})
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
OWNER = "usr_27bef1bdf1f340d29ad85e63fb6b5e08"


def _ts(value):
    return value.timestamp() if value else None


@app.function(image=image, secrets=[secret], timeout=900, cpu=0.5)
def queue_timings():
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq.registry import FailedJobRegistry, FinishedJobRegistry
    conn = Redis.from_url(os.environ["REDIS_URL"])
    q = Queue("cutsell", connection=conn)
    ids = list(FinishedJobRegistry(queue=q).get_job_ids()) + list(FailedJobRegistry(queue=q).get_job_ids())
    rows = []
    for jid in ids:
        try:
            job = Job.fetch(jid, connection=conn)
        except Exception:
            continue
        payload = (job.args or [{}])[0] if job.args else {}
        if not isinstance(payload, dict):
            payload = {}
        rows.append({
            "kind": str(job.func_name).rsplit(".", 1)[-1],
            "owner": (job.meta or {}).get("user_id") == OWNER,
            "status": str(job.get_status()),
            "enqueued": _ts(job.enqueued_at), "started": _ts(job.started_at), "ended": _ts(job.ended_at),
            "stage": (job.meta or {}).get("stage"), "error_code": (job.meta or {}).get("error_code"),
            "sources": [{"order": s.get("source_order"), "uri": s.get("uri"), "dur": s.get("duration_sec")}
                        for s in payload.get("sources", [])] if job.func_name.endswith("run_flow_b_job") else [],
            "exc": (job.exc_info or "")[-300:] if str(job.get_status()) == "failed" else "",
        })
    return rows


@app.function(image=image, secrets=[secret], timeout=1800, cpu=2.0, memory=4096)
def timed_run(uri: str):
    t0 = time.time()
    marks = {}
    def mark(name, start):
        marks[name] = round(time.time() - start, 2)
    s = time.time()
    from cutsell_worker.storage import download_source
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.worker_job import resolve_sources_for_editorial_entry
    from cutsell_worker.simple_engine import asr, engine
    mark("import_worker_modules", s)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        s = time.time(); path = download_source(uri, str(Path(d) / "src.mp4")); mark("download_from_r2", s)
        size_mb = round(os.path.getsize(path) / 1048576, 1)
        s = time.time(); pr = probe_media(path); mark("probe", s)
        s = time.time(); resolved, fmt, norm, blocked = resolve_sources_for_editorial_entry({"a": path}, output_directory=d); mark("format_gate", s)
        s = time.time(); audio = asr.extract_audio(path, str(Path(d) / "a.mp3")); mark("extract_audio", s)
        s = time.time(); heard = asr.transcribe_words(audio); mark("deepgram", s)
        words = [w for w in heard["words"] if float(w["e"]) > float(w["s"])]
        calls = []
        from cutsell_worker.simple_engine.llm import call_anthropic
        def call(prompt):
            c = time.time(); text, u = call_anthropic(prompt)
            calls.append({"s": round(time.time() - c, 2), "in": u.get("input_tokens"), "out": u.get("output_tokens"),
                          "kind": "pass" if "A PASS is one continuous run" in prompt else "decision" if "STRICT DEFINITION OF A RETAKE" in prompt
                          else "haul" if "FIRST decide the TYPE" in prompt else "sale" if "checking a FIRST CUT" in prompt
                          else "fixwords" if "index:word" in prompt else "other"})
            return text, u
        # same order as process(): decide (+reviews) -> refine -> scenes -> snap -> breathe -> fixwords
        s = time.time(); splits, info = engine.decide_reviewed(words, float(pr.duration_sec), call); mark("claude_decision_and_reviews", s)
        s = time.time(); refined = engine.keep_spoken_words(engine.refine(splits, path, float(pr.duration_sec), words), words); mark("refine_silence", s)
        s = time.time(); cuts = engine.scene_cuts(path); mark("scene_cuts_video", s)
        s = time.time(); engine.snap(refined, words, cuts); out = engine.breathe(refined, words, float(pr.duration_sec)); mark("snap_breathe", s)
        s = time.time(); engine.fix_words_and_moments(words, out, call); mark("claude_fixwords_zoom", s)
        s = time.time(); a2 = asr.extract_audio(path, str(Path(d) / "b.m4a").replace(".m4a", ".mp3")); mark("extract_audio_again", s)
        return {"uri_tail": uri[-12:], "mb": size_mb, "duration": round(float(pr.duration_sec), 1), "words": len(words),
                "format_blocked": bool(blocked), "normalized": bool(norm), "stages": marks, "claude_calls": calls,
                "total": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    rows = queue_timings.remote()
    flow = sorted([r for r in rows if r["kind"] == "run_flow_b_job" and r["started"]], key=lambda r: r["enqueued"] or 0)
    picks = []
    for r in reversed(flow):
        if r["owner"] and r["status"] == "finished" and len(r["sources"]) == 1 and r["sources"][0]["uri"]:
            d = float(r["sources"][0]["dur"] or 0)
            want = "1min" if 30 <= d <= 100 else "3min" if 130 <= d <= 260 else None
            if want and want not in [p[0] for p in picks]:
                picks.append((want, r["sources"][0]["uri"]))
        if len(picks) == 2:
            break
    runs = list(timed_run.map([u for _, u in picks], return_exceptions=True))
    runs = [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in runs]
    for r in rows:
        for s in r["sources"]:
            s["uri"] = (s.get("uri") or "")[-12:]
    with open("dedup_eval.json", "w") as f:
        json.dump({"queue": rows, "picks": [p[0] for p in picks], "runs": runs}, f, default=str)
    waits = [r["started"] - r["enqueued"] for r in flow if r["enqueued"]]
    print(f"::notice title=speed::jobs {len(flow)} picks {[p[0] for p in picks]} wait median {sorted(waits)[len(waits)//2] if waits else -1:.1f}s")
