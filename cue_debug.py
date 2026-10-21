"""Read-only: the words Deepgram gives for the test Mix's clips and which sentences the camera-cue floor hides."""
import json
import os

import modal

app = modal.App("cutsell-cue-debug")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))
TEST_USER = "cutsell-prod-verify-20261021"


@app.function(image=image, secrets=[secret], timeout=600)
def run():
    import tempfile
    from pathlib import Path
    from redis import Redis
    from cutsell_worker import project_store, mix_store, product_catalog as catalog, mix_engine as M
    from cutsell_worker.storage import download_source
    from cutsell_worker.simple_engine import asr, engine as E
    r = Redis.from_url(os.environ["REDIS_URL"])
    work = Path(tempfile.mkdtemp())
    rec = None
    for p in project_store.list_projects(user_id=TEST_USER, limit=100, client=r):
        if p.get("state") == catalog.MIX_CONTAINER_STATE:
            rec = mix_store.load(TEST_USER, p["project_id"], client=r)
    out = {}
    for s in rec["sections"]:
        for c in s["clips"]:
            local = download_source(c["audio_uri"] or c["uri"], str(work / (c["clip_id"] + ".mp4")))
            words = [w for w in asr.transcribe_words(asr.extract_audio(local, str(work / (c["clip_id"] + ".mp3"))),
                                                     language_hint="en")["words"] if w["e"] > w["s"]]
            units = E.lines_of(words)
            out[c["clip_id"]] = [{"text": " ".join(words[k]["w"] for k in u), "s": words[u[0]]["s"], "e": words[u[-1]]["e"],
                                  "cue": M.is_camera_cue([E._norm(words[k]["w"]) for k in u])} for u in units]
    return out


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump(run.remote(), f, ensure_ascii=False)
