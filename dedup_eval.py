"""A/B: simple engine 1.2 (live) vs 1.3 (repetido). Ephemeral `modal run`; nothing deployed, nothing written to
Redis/storage. Gold words are the engine's input only; the Gold decisions are scored afterwards, never prompted."""
import json
import os

import modal

app = modal.App("cutsell-engine-dedup-eval")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .env({"CUTSELL_ENGINE": "simple"})
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", "/root/gold_v1.json")
)
REAL = ("d87a5b66", "9ea8d6f2")   # the 106 s Spanish and the 171 s English cut from the live review


def _engine(version):
    if version == "1.2":
        from cutsell_worker.simple_engine_v12 import engine
    else:
        from cutsell_worker.simple_engine import engine
    return engine


@app.function(image=image, secrets=[secret], timeout=1200, max_containers=10, retries=1)
def gold_one(item):
    vid, version = item
    gold = json.load(open("/root/gold_v1.json"))
    video = next(v for v in gold["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    eng = _engine(version)
    from cutsell_worker.simple_engine.llm import call_anthropic
    usage = []
    def call(prompt):
        text, u = call_anthropic(prompt); usage.append(u); return text, u
    try:
        splits, info = eng.decide(words, float(video["duration"]), call, 0)
        eng.orphans(splits, words)
        return {"id": vid, "version": version, "splits": splits, "usage": usage, "review": info.get("review")}
    except Exception as exc:
        return {"id": vid, "version": version, "error": repr(exc)[:300], "usage": usage}


def _h(v):
    import hashlib
    return hashlib.sha256(str(v).encode()).hexdigest()[:8]


@app.function(image=image, secrets=[secret], timeout=1800, cpu=2.0, memory=4096)
def real_videos():
    import tempfile
    from pathlib import Path
    from redis import Redis
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.simple_engine import asr
    from cutsell_worker.storage import download_source
    r = Redis.from_url(os.environ["REDIS_URL"])
    found = {}
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            snap = json.loads(r.get(key))
        except Exception:
            continue
        if _h(snap.get("project_id")) in REAL:
            found[_h(snap.get("project_id"))] = snap
    out = []
    for ph in REAL:
        snap = found.get(ph)
        if not snap or not snap.get("sources"):
            out.append({"project": ph, "error": "not found"}); continue
        src = snap["sources"][0]
        with tempfile.TemporaryDirectory() as d:
            path = download_source(src["uri"], str(Path(d) / "v.mp4"))
            duration = float(probe_media(path).duration_sec)
            audio = asr.extract_audio(path, str(Path(d) / "a.mp3"))
            heard = asr.transcribe_words(audio)
            words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
            row = {"project": ph, "duration": duration, "language": heard.get("language"), "words": words, "runs": {}}
            for version in ("1.2", "1.3"):
                eng = _engine(version)
                from cutsell_worker.simple_engine.llm import call_anthropic
                usage = []
                def call(prompt):
                    text, u = call_anthropic(prompt); usage.append(u); return text, u
                res = eng.process(words, duration, audio, video_path=path, llm=call)
                row["runs"][version] = {"splits": res["splits"], "cut_duration": res["cut_duration"], "usage": usage,
                                        "review": (res.get("log") or {}).get("revision")}
            out.append(row)
    return out


@app.local_entrypoint()
def main():
    gold = json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))
    items = [(v["id"], ver) for v in gold["videos"] for ver in ("1.2", "1.3")]
    real = real_videos.spawn()
    results = list(gold_one.map(items, return_exceptions=True))
    report = {"gold": [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in results], "real": real.get()}
    with open("dedup_eval.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, default=str)
    tin = sum(int(u.get("input_tokens") or 0) for g in report["gold"] for u in g.get("usage", []))
    tout = sum(int(u.get("output_tokens") or 0) for g in report["gold"] for u in g.get("usage", []))
    print(f"::notice title=gold tokens::in {tin} out {tout} errors {sum(1 for g in report['gold'] if 'error' in g)}")
