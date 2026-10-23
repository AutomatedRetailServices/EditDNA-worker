"""Diagnosis (PO 23 Oct): why the same video gets very different cuts run to run. Each video is heard ONCE (same words);
then the engine runs N times with the normal API. Every step's answer is kept to see where the runs part ways.
Report encrypted. Nothing written to the app's storage or Redis."""
import json
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


@app.function(image=image, secrets=[secret], timeout=600, cpu=2.0, memory=4096)
def hear(key: str):
    import tempfile
    from pathlib import Path
    from cutsell_worker import storage_routing as sr
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.simple_engine import asr
    fb = sr.fallback_storage()
    client = sr._make_fallback_client(fb)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "v.mp4")
        client.download_file(fb.bucket, key, path)
        duration = float(probe_media(path).duration_sec)
        heard = asr.transcribe_words(asr.extract_audio(path, str(Path(d) / "a.mp3")))
    return {"key": key, "duration": duration, "words": [w for w in heard["words"] if float(w["e"]) > float(w["s"])]}


@app.function(image=image, secrets=[secret], timeout=600, cpu=1.0, memory=2048)
def run_once(heard: dict, run: int):
    from cutsell_worker.simple_engine import engine, llm
    answers = []

    def call(prompt, **kw):
        text, usage = llm.call_anthropic(prompt, **kw)
        answers.append({"step": usage.get("step"), "text": text[-3000:], "out_tokens": usage.get("output_tokens"),
                        "thinking_blocks": usage.get("thinking_blocks")})
        return text, usage

    with llm.cache_scope(True, ttl="1h"):
        out = engine.process(heard["words"], heard["duration"], "/dev/null", video_path=None, scene_cut_list=[], llm=call)
    report = llm.cache_report(out["usage"])
    return {"key": heard["key"], "run": run, "cut_s": out["cut_duration"], "passes": out.get("passes"),
            "kept": [(round(x["start"], 2), round(x["end"], 2)) for x in out["splits"] if x["visible"]],
            "usd": report["usd"], "answers": answers, "log": str(out.get("log"))[:3000]}


@app.local_entrypoint()
def main():
    plan = json.load(open("cache_plan.json"))
    heard = list(hear.map(plan["videos"]))
    args = [(h, r) for h in heard for r in range(plan.get("runs", 5))]
    rows = list(run_once.starmap(args, return_exceptions=True))
    rows = [r if isinstance(r, dict) else {"error": str(r)[:300]} for r in rows]
    with open("engine_check.json", "w") as f:
        json.dump({"label": plan.get("label"), "words": {h["key"]: len(h["words"]) for h in heard}, "rows": rows}, f, default=str)
