"""Read-only: Deepgram on the no-speech cuts' audio with variants (louder audio, Spanish, multi). Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-fail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
JOBS = ["ae9bc9dc", "e64e2472", "e5bdd500"]
USER = "usr_25bb7b9fc4a54767a475fda9a9ccb6c9"


def _dg(mp3, extra):
    import requests
    from cutsell_worker.simple_engine import asr
    key = os.environ["DEEPGRAM_API_KEY"].strip()
    params = [("model", "nova-3"), ("punctuate", "true"), ("filler_words", "true"), ("smart_format", "false")] + extra
    params += [("keyterm", t) for t in asr.KEYTERMS] if ("language", "multi") not in extra else []
    with open(mp3, "rb") as h:
        r = requests.post(asr.API_URL, params=params, data=h, timeout=(20, 600),
                          headers={"Authorization": "Token " + key, "Content-Type": "audio/mpeg"})
    if r.status_code != 200:
        return f"HTTP {r.status_code} {r.text[:200]}"
    ch = r.json()["results"]["channels"][0]
    alt = ch["alternatives"][0]
    ws = alt.get("words", [])
    return {"lang": ch.get("detected_language"), "conf": ch.get("language_confidence"), "n": len(ws),
            "text": (alt.get("transcript") or "")[:300]}


@app.function(image=image, secrets=[secret], timeout=1500, cpu=2.0, memory=4096)
def check(prefix):
    import subprocess
    import tempfile
    from pathlib import Path
    from redis import Redis
    from rq.job import Job
    from cutsell_worker.storage import download_source
    r = Redis.from_url(os.environ["REDIS_URL"])
    jid = None
    for key in r.scan_iter(f"rq:job:{prefix}*", count=2000):
        k = key.decode()
        if k.count(":") == 2:
            jid = k.split(":")[-1]
    job = Job.fetch(jid, connection=r)
    s = job.args[0]["sources"][0]
    out = {"job": jid}
    with tempfile.TemporaryDirectory() as d:
        p = download_source(s["metadata"]["audio_uri"], str(Path(d) / "voice.m4a"))
        raw = str(Path(d) / "raw.mp3")
        loud = str(Path(d) / "loud.mp3")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", raw], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p, "-vn", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                        "-ac", "1", "-ar", "16000", "-b:a", "64k", loud], check=True)
        sd = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", p, "-af", "silencedetect=n=-40dB:d=0.5", "-f", "null", "-"],
                            capture_output=True, text=True).stderr
        sil = sum(float(l.split("silence_duration: ")[1]) for l in sd.splitlines() if "silence_duration" in l)
        out["silence_sec_below_-40dB"] = round(sil, 1)
        vols = []
        for seg in range(0, 60, 10):
            v = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(seg), "-t", "10", "-i", p, "-af", "volumedetect",
                                "-f", "null", "-"], capture_output=True, text=True).stderr
            vols.append([l.split("] ")[-1] for l in v.splitlines() if "mean_volume" in l])
        out["first_minute_mean_db_per_10s"] = vols
        for name, path, extra in (("raw_multi", raw, [("language", "multi")]),
                                  ("loud_detect", loud, [("detect_language", "true")]),
                                  ("loud_detect_es_en", loud, [("detect_language", "es"), ("detect_language", "en")]),
                                  ("loud_multi", loud, [("language", "multi")]),
                                  ("loud_es", loud, [("language", "es")])):
            try:
                out[name] = _dg(path, extra)
            except Exception as e:
                out[name] = f"{e.__class__.__name__}: {str(e)[:200]}"
    return out


@app.function(image=image, secrets=[secret], timeout=300)
def recent():
    from redis import Redis
    from rq.job import Job
    r = Redis.from_url(os.environ["REDIS_URL"])
    rows = []
    for key in r.scan_iter("rq:job:*", count=5000):
        k = key.decode()
        if k.count(":") != 2:
            continue
        try:
            j = Job.fetch(k.split(":")[-1], connection=r)
        except Exception:
            continue
        if (j.meta or {}).get("user_id") != USER or not j.enqueued_at or str(j.enqueued_at) < "2026-10-22 20:30":
            continue
        rows.append({"id": j.id[:8], "func": j.func_name.split(".")[-1], "status": str(j.get_status()), "enq": str(j.enqueued_at)[:19],
                     "names": [x.get("original_name") for x in (j.args[0].get("sources", []) if j.args and isinstance(j.args[0], dict) else [])],
                     "project": (j.meta or {}).get("project_id"), "stage": (j.meta or {}).get("stage"),
                     "err": (j.exc_info or "").strip().splitlines()[-1][:200] if j.exc_info else ""})
    return sorted(rows, key=lambda x: x["enq"])


@app.local_entrypoint()
def main():
    res = list(check.map(JOBS, return_exceptions=True))
    out = {"checks": [x if isinstance(x, dict) else {"error": str(x)[:400]} for x in res], "recent": recent.remote()}
    with open("engine_check.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
