"""Side-by-side test: the cut the app has today (stored draft, engine 1.2) and engine 1.3, rendered as MP4.
Nothing in the user's projects/drafts changes; nothing is deleted. The MP4s go to a separate test folder in the
bucket (engine-ab-tests/2026-10-17/) and come back as 7-day links, encrypted before leaving the runner."""
import json
import os

import modal

app = modal.App("cutsell-engine-ab-render")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .env({"CUTSELL_ENGINE": "simple"})
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
PROJECTS = ("d87a5b66",)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PREFIX = "engine-ab-tests/2026-10-17-comprension8"


def _h(v):
    import hashlib
    return hashlib.sha256(str(v).encode()).hexdigest()[:8]


def _render(src, splits, label, out):
    import subprocess
    segs = [(float(s["start"]), float(s["end"])) for s in splits if s.get("visible") and float(s["end"]) - float(s["start"]) > 0.05]
    parts, labels = [], []
    for i, (a, b) in enumerate(segs):
        d = b - a
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}];"
                     f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.02,afade=t=out:st={max(0, d - 0.03):.3f}:d=0.03[a{i}];")
        labels.append(f"[v{i}][a{i}]")
    graph = "".join(parts) + "".join(labels) + f"concat=n={len(segs)}:v=1:a=1[cv][ca];" \
        f"[cv]scale=720:-2,drawtext=fontfile={FONT}:text='{label}':x=(w-text_w)/2:y=60:fontsize=40:" \
        f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=14[ov]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-filter_complex", graph, "-map", "[ov]", "-map", "[ca]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", out], check=True)
    return out


@app.function(image=image, secrets=[secret], timeout=1800, cpu=4.0, memory=8192, max_containers=4)
def one(ph):
    import tempfile
    from pathlib import Path

    import boto3
    from botocore.config import Config
    from redis import Redis
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.simple_engine import asr, engine
    from cutsell_worker.simple_engine.llm import call_anthropic
    from cutsell_worker.storage import download_source

    r = Redis.from_url(os.environ["REDIS_URL"])
    snap = None
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            s = json.loads(r.get(key))
        except Exception:
            continue
        if _h(s.get("project_id")) == ph:
            snap = s
            break
    if not snap or not snap.get("sources"):
        return {"project": ph, "error": "not found"}
    original = (snap.get("undo") or [snap["draft"]])[0]
    stored = (((original.get("diagnostics") or {}).get("simple_engine") or {}).get("sources") or [{}])[0].get("splits") or []
    title = ""
    bucket = os.environ["S3_BUCKET"].strip()
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "auto").strip(), config=Config(signature_version="s3v4"))
    with tempfile.TemporaryDirectory() as d:
        src = download_source(snap["sources"][0]["uri"], str(Path(d) / "v.mp4"))
        duration = float(probe_media(src).duration_sec)
        audio = asr.extract_audio(src, str(Path(d) / "a.mp3"))
        heard = asr.transcribe_words(audio)
        words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
        usage = []
        def call(prompt):
            text, u = call_anthropic(prompt); usage.append(u); return text, u
        new = engine.process(words, duration, audio, video_path=src, llm=call, mode="understand")
        links = {}
        for name, splits, label in (("actual", stored, "ACTUAL (1.2)"), ("nueva", new["splits"], "NUEVA (COMPRENSION)")):
            out = _render(src, splits, label, str(Path(d) / f"{name}.mp4"))
            key = f"{PREFIX}/{ph}-{name}.mp4"
            s3.upload_file(out, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
            links[name] = {"url": s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=7 * 24 * 3600),
                           "seconds": round(sum(float(s["end"]) - float(s["start"]) for s in splits if s.get("visible")), 1),
                           "mb": round(os.path.getsize(out) / 1e6, 1)}
    def show(splits):
        return [{"start": s["start"], "end": s["end"], "visible": s["visible"], "reason": s.get("reason"),
                 "text": " ".join(w["w"] for w in words if float(s["start"]) <= (w["s"] + w["e"]) / 2 <= float(s["end"]))}
                for s in splits]
    return {"project": ph, "duration": round(duration, 1), "language": heard.get("language"), "links": links,
            "actual": show(stored), "nueva": show(new["splits"]), "usage": usage,
            "words": words, "raw_splits": new["splits"], "log": new.get("log"),
            "comprension": (new.get("log") or {}).get("comprension"), "revision": (new.get("log") or {}).get("revision")}


@app.local_entrypoint()
def main():
    out = list(one.map(PROJECTS, return_exceptions=True))
    report = [x if isinstance(x, dict) else {"error": repr(x)[:400]} for x in out]
    with open("render.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, default=str)
    print("::notice title=render::" + ", ".join(f"{x.get('project')}:{'ok' if 'links' in x else x.get('error','?')[:80]}" for x in report))
