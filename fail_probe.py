"""Read-only: the two "no speech found" cuts -- what audio the engine got (audio-first file and the video's own
track): duration, loudness. Downloads to a temp dir only. Report encrypted."""
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
JOBS = ["ae9bc9dc", "e64e2472"]


def _stats(path):
    import subprocess
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,duration,sample_rate,channels",
                         "-of", "json", path], capture_output=True, text=True)
    vd = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"],
                        capture_output=True, text=True)
    vol = [l.split("] ")[-1] for l in vd.stderr.splitlines() if "mean_volume" in l or "max_volume" in l]
    return {"probe": json.loads(pr.stdout or "{}"), "volume": vol, "bytes": os.path.getsize(path)}


@app.function(image=image, secrets=[secret], timeout=1500, cpu=2.0, memory=4096)
def check(prefix):
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
    if not jid:
        return {"prefix": prefix, "error": "not found"}
    job = Job.fetch(jid, connection=r)
    payload = job.args[0] if job.args else {}
    out = {"job": jid, "language_hint": payload.get("language_hint"), "sources": []}
    with tempfile.TemporaryDirectory() as d:
        for s in payload.get("sources", []):
            row = {"name": s.get("original_name"), "metadata": {k: (v if k != "audio_uri" else "present")
                                                                 for k, v in (s.get("metadata") or {}).items()}}
            audio = (s.get("metadata") or {}).get("audio_uri")
            if audio:
                try:
                    p = download_source(audio, str(Path(d) / "voice.m4a"))
                    row["audio_first"] = _stats(p)
                    from cutsell_worker.simple_engine import asr
                    mp3 = asr.extract_audio(p, str(Path(d) / "voice.mp3"))
                    for hint in (None, "es", "en"):
                        try:
                            h = asr.transcribe_words(mp3, language_hint=hint)
                            row[f"asr_{hint}"] = {"language": h.get("language"), "duration": h.get("duration"),
                                                  "n_words": len(h["words"]),
                                                  "n_valid": len([w for w in h["words"] if float(w["e"]) > float(w["s"])]),
                                                  "first": [w["w"] for w in h["words"][:12]]}
                        except Exception as e:
                            row[f"asr_{hint}"] = f"{e.__class__.__name__}: {str(e)[:200]}"
                except Exception as e:
                    row["audio_first_error"] = f"{e.__class__.__name__}: {str(e)[:200]}"
            row["uri_kind"] = (s.get("uri") or "")[:5]
            try:
                p = download_source(s["uri"], str(Path(d) / ("video" + (Path(s.get("original_name") or "v.mov").suffix or ".mov"))))
                row["video"] = _stats(p)
            except Exception as e:
                row["video_error"] = f"{e.__class__.__name__}: {str(e)[:200]}"
            out["sources"].append(row)
    return out


@app.local_entrypoint()
def main():
    res = list(check.map(JOBS, return_exceptions=True))
    with open("engine_check.json", "w") as f:
        json.dump([x if isinstance(x, dict) else {"error": str(x)[:400]} for x in res], f, indent=1, default=str)
