"""Read-only (22 Oct): (1) IMG_9661: 6 frames, loudness over time, Deepgram auto/es/en/multi on the video's own sound and
the voice file. (2) Old Amazon bucket by category (test inputs / her uploads / old run outputs) and whether each upload
is also in R2 (same key). (3) R2: what the 3-day rule would remove today. Lists and reads only; deletes nothing."""
import base64
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

TEST_INPUTS = ("Editdna longform validation/", "Editdna good videos/", "Editdna bloopers videos/", "Yaskira/", "Bloop",
               "IMG_0", "editdna/training/", "editdna/models/", "editdna/benchmarks/", "editdna/sentence", "editdna/semantic")
USER = ("cutsell/uploads/", "upload", "voice", "overlay", "broll", "render", "cutsell/exports/")
OUTPUTS = ("cutsell/benchmarks/", "cutsell/serverless/", "editdna/outputs/", "cutsell/benchmark-results/",
           "cutsell/august-comparison/", "cutsell/legacy-comparison/", "cutsell/smokes/", "pruebas/", "cutsell/bootstrap")


def _cat(key):
    if key.startswith(TEST_INPUTS):
        return "a_test_inputs"
    if key.startswith(OUTPUTS):
        return "c_old_run_outputs"
    if key.startswith(USER) or any(k in key for k in ("/uploads/", "voice", "overlay", "broll")):
        return "b_user_files"
    return "other"


@app.function(image=image, secrets=[secret], timeout=900, cpu=2.0, memory=4096)
def img9661():
    import subprocess
    import tempfile
    from pathlib import Path
    from redis import Redis
    from rq.job import Job
    from cutsell_worker.storage import download_source
    from cutsell_worker.simple_engine import asr
    r = Redis.from_url(os.environ["REDIS_URL"])
    jid = next(k.decode().split(":")[-1] for k in r.scan_iter("rq:job:ae9bc9dc*", count=2000) if k.decode().count(":") == 2)
    src = Job.fetch(jid, connection=r).args[0]["sources"][0]
    out = {"frames": [], "loudness": [], "asr": {}}
    with tempfile.TemporaryDirectory() as d:
        video = download_source(src["uri"], str(Path(d) / "v.mov"))
        voice = download_source(src["metadata"]["audio_uri"], str(Path(d) / "voice.m4a"))
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                                   capture_output=True, text=True).stdout.strip() or 3)
        out["duration"] = dur
        for i in range(6):
            t = dur * (i + 0.5) / 6
            jpg = str(Path(d) / f"f{i}.jpg")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "5", jpg])
            if os.path.exists(jpg):
                out["frames"].append({"t": round(t, 2), "jpg": base64.b64encode(open(jpg, "rb").read()).decode()})
        for i in range(int(dur / 0.25) + 1):
            st = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{i * 0.25:.2f}", "-t", "0.25", "-i", video,
                                 "-af", "volumedetect", "-vn", "-f", "null", "-"], capture_output=True, text=True).stderr
            mean = [l.split("mean_volume: ")[1] for l in st.splitlines() if "mean_volume" in l]
            out["loudness"].append((round(i * 0.25, 2), mean[0] if mean else None))
        spec = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-af", "highpass=f=300,lowpass=f=3400,volumedetect",
                               "-vn", "-f", "null", "-"], capture_output=True, text=True).stderr
        out["speech_band_volume"] = [l.split("] ")[-1] for l in spec.splitlines() if "_volume" in l]
        for name, f in (("video_sound", video), ("voice_file", voice)):
            mp3 = asr.extract_audio(f, str(Path(d) / f"{name}.mp3"))
            for hint in (None, "es", "en"):
                try:
                    h = asr.transcribe_words(mp3, language_hint=hint)
                    out["asr"][f"{name}_{hint or 'auto'}"] = {"lang": h["language"], "text": " ".join(w["w"] for w in h["words"])}
                except Exception as e:
                    out["asr"][f"{name}_{hint or 'auto'}"] = str(e)[:150]
    return out


@app.function(image=image, secrets=[secret], timeout=1500)
def buckets():
    import datetime as dt
    from collections import defaultdict
    from cutsell_worker import storage_routing as sr
    now = dt.datetime.now(dt.timezone.utc)
    fb = sr.fallback_storage()
    old = sr._make_fallback_client(fb)
    r2 = sr.default_client()
    r2_bucket = sr.current_bucket()
    r2_keys = {}
    for page in r2.get_paginator("list_objects_v2").paginate(Bucket=r2_bucket):
        for o in page.get("Contents", []):
            r2_keys[o["Key"]] = (int(o["Size"]), o["LastModified"])
    cats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    user_missing = [0, 0]
    user_present = [0, 0]
    video_ext = (".mp4", ".mov", ".m4v", ".wav", ".m4a", ".mp3", ".webm")
    for page in old.get_paginator("list_objects_v2").paginate(Bucket=fb.bucket):
        for o in page.get("Contents", []):
            key, size = o["Key"], int(o["Size"])
            c = _cat(key)
            top = "/".join(key.split("/")[:2]) if "/" in key else key.split(".")[0][:8]
            cats[c][top][0] += 1
            cats[c][top][1] += size
            kind = "media" if key.lower().endswith(video_ext) else "other"
            cats[c]["__" + kind][0] += 1
            cats[c]["__" + kind][1] += size
            if c == "b_user_files":
                if key in r2_keys and r2_keys[key][0] == size:
                    user_present[0] += 1; user_present[1] += size
                else:
                    user_missing[0] += 1; user_missing[1] += size
    r2_three_days = [0, 0]
    r2_uploads = [0, 0]
    for key, (size, modified) in r2_keys.items():
        if key.startswith(("cutsell/uploads/",)):
            r2_uploads[0] += 1; r2_uploads[1] += size
            if (now - modified).total_seconds() > 3 * 86400:
                r2_three_days[0] += 1; r2_three_days[1] += size
    gb = lambda v: [v[0], round(v[1] / 1e9, 3)]
    return {"old_bucket": {c: {k: gb(v) for k, v in sorted(d.items(), key=lambda x: -x[1][1])} for c, d in cats.items()},
            "user_files_already_in_r2": gb(user_present), "user_files_not_in_r2": gb(user_missing),
            "r2_uploads": gb(r2_uploads), "r2_uploads_older_than_3_days_by_upload_date": gb(r2_three_days)}


@app.local_entrypoint()
def main():
    out = {}
    for name, fn in (("img9661", img9661), ("buckets", buckets)):
        try:
            out[name] = fn.remote()
        except Exception as e:
            out[name] = f"{e.__class__.__name__}: {str(e)[:300]}"
    with open("engine_check.json", "w") as f:
        json.dump(out, f, default=str)
