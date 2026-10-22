"""Prompt-cache test (22 Oct): cut QA videos with the engine code of feat/engine-cache-all in an ephemeral Modal run
(nothing deployed; nothing written to the app's storage or Redis). Per video: tokens, cost with the cache and the same
tokens without it. cache_plan.json says which videos and which cache lifetime. Report encrypted."""
import json
import os
import time

import modal

app = modal.App("cutsell-fail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
PLAN = json.load(open("cache_plan.json"))


@app.function(image=image, secrets=[secret], timeout=1500, cpu=2.0, memory=4096)
def cut(key: str, ttl: str, cache_on: bool):
    import tempfile
    from pathlib import Path
    from cutsell_worker import storage_routing as sr
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.simple_engine import asr, llm, process
    fb = sr.fallback_storage()
    client = sr._make_fallback_client(fb)
    started = time.time()
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "v.mp4")
        client.download_file(fb.bucket, key, path)
        duration = float(probe_media(path).duration_sec)
        heard = asr.transcribe_words(asr.extract_audio(path, str(Path(d) / "a.mp3")))
        words = [w for w in heard["words"] if float(w["e"]) > float(w["s"])]
        t0 = time.time()
        with llm.cache_scope(cache_on, ttl=ttl):
            out = process(words, duration, path, video_path=path)
            report = llm.cache_report(out["usage"], ttl=ttl)
        calls = [{k: u.get(k) for k in ("step", "input_tokens", "output_tokens", "cache_read_input_tokens",
                                        "cache_creation_input_tokens", "cache_creation", "cache_ttl")} for u in out["usage"]]
    return {"key": key, "ttl": ttl, "cache_on": cache_on, "seconds_long": round(duration, 1), "words": len(words),
            "started_utc": time.strftime("%H:%M:%S", time.gmtime(started)), "engine_s": round(time.time() - t0, 1),
            "cut_s": out.get("cut_duration"), "report": report, "calls": calls}


@app.local_entrypoint()
def main():
    rows = []
    for key in PLAN["videos"]:
        try:
            rows.append(cut.remote(key, PLAN.get("ttl", "1h"), PLAN.get("cache_on", True)))
        except Exception as e:
            rows.append({"key": key, "error": f"{e.__class__.__name__}: {str(e)[:300]}"})
    with open("engine_check.json", "w") as f:
        json.dump({"label": PLAN.get("label"), "rows": rows}, f, indent=1, default=str)
