"""Read-only: why did the 23 Oct 17:51 UTC cuts fail (RQ job error + the last failed jobs). No writes."""
import json

import modal

app = modal.App("cutsell-fail-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, secrets=[secret], timeout=600)
def audit():
    import os, tempfile, subprocess, json
    from pathlib import Path
    from redis import Redis
    from rq.job import Job
    from cutsell_worker import storage_routing as sr
    from cutsell_worker.simple_engine import asr
    from cutsell_worker import simple_engine_adapter as ad
    r = Redis.from_url(os.environ["REDIS_URL"], socket_timeout=15)
    out = {}
    for jid in ("ca083ad4-03cc-49f3-afe1-4df1d580e4f5",):
        try:
            if len(jid) < 36:
                jid = next(k.decode().split(":")[-1] for k in r.scan_iter(f"rq:job:{jid}*", count=1000) if k.decode().count(":") == 2)
            j = Job.fetch(jid, connection=r)
            src = (j.args[0].get("sources") or [{}])[0]
            meta = src.get("metadata") or {}
            uri = src.get("uri")
            row = {"voice_only": bool(meta.get("audio_uri")), "duration_sec_sent": src.get("duration_sec"), "language_hint": j.args[0].get("language_hint")}
            with tempfile.TemporaryDirectory() as d:
                local = str(Path(d) / ("a" + Path(uri).suffix))
                bucket, key = uri[5:].split("/", 1)
                client = sr.default_client()
                try:
                    client.download_file(sr.current_bucket(), key, local)
                except Exception:
                    sr._make_fallback_client(sr.fallback_storage()).download_file(sr.fallback_storage().bucket, key, local)
                row["file_ext"] = Path(uri).suffix
                p = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", local], capture_output=True, text=True)
                info = json.loads(p.stdout or "{}")
                keep = {"format": {k: v for k, v in (info.get("format") or {}).items() if k in ("size", "bit_rate", "format_name")},
                        "format_tags": {k: v for k, v in ((info.get("format") or {}).get("tags") or {}).items() if "location" not in k.lower() and "iso6709" not in k.lower()},
                        "streams": [{k: v for k, v in st.items() if k in ("codec_type", "codec_name", "bit_rate", "width", "height", "r_frame_rate", "sample_rate", "channels", "duration")}
                                    | {"tags": {k: v for k, v in (st.get("tags") or {}).items() if k in ("handler_name", "encoder", "vendor_id")}} for st in info.get("streams") or []]}
                row["probe"] = keep
                mp3 = asr.extract_audio(local, str(Path(d) / "x.mp3"))
                row["mean_db"] = ad._mean_volume_db(mp3)
                heard = asr.transcribe_words(mp3)
                words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
                row["detected_language"] = heard.get("language")
                row["words_first_pass"] = len(words)
                again = ad._heard_in_her_language(heard, mp3, lambda path, language_hint=None: asr.transcribe_words(path, language_hint=language_hint), None)
                w2 = [w for w in again.get("words", []) if float(w["e"]) > float(w["s"])]
                row["words_after_es_en"] = len(w2)
                row["text_sample_words"] = len(" ".join(w["w"] for w in w2[:12]))
                dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", local],
                                           capture_output=True, text=True).stdout.strip() or 0)
                row["duration_measured"] = dur
                row["rule_no_words"] = not w2
                row["rule_nobody_talking"] = ad._nobody_talking(w2, dur)
                row["rule_background_voice"] = ad._only_background_voice(w2, mp3)
            out[jid[:8]] = row
        except Exception as e:
            out[jid[:8]] = {"error": e.__class__.__name__ + ": " + str(e)[:300]}
    return out


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump({"label": "no_speech 23 Oct: the original video", "rows": [audit.remote()]}, f, indent=1, default=str)
