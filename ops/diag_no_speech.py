"""One-off diagnosis (PO 29 Oct, approved): why did these cuts fail with "no speech"?

Read-only: fetches the failed jobs from the queue, downloads ONLY their voice file (and the uploaded video's
sound, if it arrived), measures loudness and asks Deepgram once, the same way the engine does. Prints NUMBERS
ONLY (the repository and its Actions logs are public): word counts, words per minute, language, dB, format.
No transcript text, no file names, no keys. Files live in the container's temp dir and vanish with it.
Run: modal run ops/diag_no_speech.py  (ephemeral app; nothing is deployed)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
app = modal.App("cutsell-diag-no-speech")
JOBS = ["f1ba4ad1-f42c-45e0-9d70-7ce3ab5de779", "40cd667a-c0c4-4ebb-aeff-660c60c453b1", "63562881-ef52-4274-970f-925017adeabf"]


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=600, cpu=1.0)
def diagnose():
    import json, os, re, subprocess, tempfile
    from redis import Redis
    from rq.job import Job
    from cutsell_worker.storage import download_source
    from cutsell_worker.simple_engine import asr

    def volume(path):
        err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                             capture_output=True, text=True).stderr
        mean = re.search(r"mean_volume: (-?[\d.]+) dB", err); peak = re.search(r"max_volume: (-?[\d.]+) dB", err)
        return (float(mean.group(1)) if mean else None, float(peak.group(1)) if peak else None)

    def probe(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type,codec_name,channels,sample_rate:stream_disposition=default:format=duration",
                              "-of", "json", path], capture_output=True, text=True).stdout
        d = json.loads(out or "{}")
        streams = [{k: s.get(k) for k in ("index", "codec_type", "codec_name", "channels", "sample_rate")} | {"default": (s.get("disposition") or {}).get("default")}
                   for s in d.get("streams", []) if s.get("codec_type") == "audio"]
        return float((d.get("format") or {}).get("duration") or 0), streams

    def hear(path, tag, tmp):
        mp3 = os.path.join(tmp, tag + ".mp3")
        asr.extract_audio(path, mp3)
        mean, peak = volume(mp3)
        seconds, _ = probe(mp3)
        rows = []
        for hint in (None, "es", "en"):
            h = asr.transcribe_words(mp3, language_hint=hint)
            words = [w for w in h.get("words", []) if float(w["e"]) > float(w["s"])]
            rows.append(f"hint={hint} lang={h.get('language')} words={len(words)} wpm={len(words) / max(seconds, 1) * 60:.0f}")
        print(f"  {tag}: {seconds:.1f}s mean={mean}dB peak={peak}dB | " + " | ".join(rows))

    redis = Redis.from_url(os.environ["REDIS_URL"])
    for job_id in JOBS:
        print(f"JOB {job_id[:8]}")
        try:
            job = Job.fetch(job_id, connection=redis)
        except Exception as exc:
            print("  not in the queue any more:", type(exc).__name__)
            continue
        payload = (job.args or [None])[0] or job.kwargs.get("payload") or {}
        print(f"  status={job.get_status()} language_hint={payload.get('language_hint')} sources={len(payload.get('sources') or [])}")
        with tempfile.TemporaryDirectory() as tmp:
            for i, src in enumerate(payload.get("sources") or []):
                meta = src.get("metadata") or {}
                for tag, uri in (("voice", meta.get("audio_uri")), ("video", src.get("uri"))):
                    if not uri:
                        print(f"  {tag}{i}: none"); continue
                    path = os.path.join(tmp, f"{tag}{i}")
                    try:
                        download_source(uri, path)
                    except Exception as exc:
                        print(f"  {tag}{i}: not downloadable ({type(exc).__name__})"); continue
                    seconds, streams = probe(path)
                    print(f"  {tag}{i} file: {os.path.getsize(path) // 1024} KB, {seconds:.1f}s, audio streams {streams}")
                    if streams:
                        hear(path, f"{tag}{i}", tmp)


@app.local_entrypoint()
def main():
    diagnose.remote()
