"""Does temperature 0 make the simple engine's cut repeatable without losing Human Gold agreement?

For each config (model default sampling vs temperature 0), runs the engine's decision step 3 times on:
- the 25 Human Gold videos (words from benchmarks/simple_engine_gold/gold_v1.json, scored like
  scripts/simple_engine_eval.py), and
- Swanny's sales test video (transcribed once with the engine's own ASR; repeatability only).
Results (cut timestamps and scores only, no transcript) go to CutSell's private storage; the public log
shows only the summary numbers. Nothing is deployed."""
import json, os, sys, time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
# Locally this file is experiments/engine_temperature/measure.py (repo two levels up). Inside the Modal
# container it is /root/measure.py, where parents[2] does not exist: that IndexError crashed every
# container at import, so the earlier runs never reached the engine and stored nothing.
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 and (_HERE.parents[2] / "cutsell_worker").is_dir() else Path("/root")
app = modal.App("cutsell-engine-temperature")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(str(REPO / "requirements.cutsell.worker.cpu.txt"))
    .pip_install("boto3")
    .add_local_dir(str(REPO / "cutsell_worker"), remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(REPO / "scripts"), remote_path="/root/scripts", ignore=["**/__pycache__/**"])
)
secret = modal.Secret.from_name("cutsell-worker")
CONFIGS = ["default", "0"]
RUNS = 3
SALES_KEY = "cutsell/uploads/0aff93c7b85d40da/39b6089655c79080/76bb18b684744ce4b3a0861909b4382e-VIDEO-2026-10-04-16-19-17.mp4"
PREFIX = "pruebas/engine-temperature/"


@app.function(image=image, secrets=[secret], timeout=900)
def sales_words():
    import boto3
    sys.path.insert(0, "/root")
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")
    bucket, cache = os.environ["S3_BUCKET"], PREFIX + "sales_words.json"
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=cache)["Body"].read())
    except Exception:
        pass
    from cutsell_worker.simple_engine.asr import extract_audio, transcribe_words
    s3.download_file(bucket, SALES_KEY, "/tmp/v.mp4")
    out = transcribe_words(extract_audio("/tmp/v.mp4", "/tmp/a.wav"))
    video = {"id": "SALES", "duration": out["duration"], "words": out["words"], "decisions": None}
    s3.put_object(Bucket=bucket, Key=cache, Body=json.dumps(video).encode(), ContentType="application/json",
                  ServerSideEncryption="AES256")
    return video


def _s3():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")


def _row_key(vid, config, run):
    return f"{PREFIX}rows/{vid}-{config}-{run}.json"


@app.function(image=image, secrets=[secret], timeout=1200, max_containers=8)
def one(job):
    sys.path.insert(0, "/root")
    video, config, run = job
    s3, bucket, key = _s3(), os.environ["S3_BUCKET"], _row_key(video["id"], config, run)
    try:                                   # already measured in an earlier (cut-off) run: reuse it
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        pass
    row = _measure(video, config, run)
    if "error" not in row:
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(row).encode(), ContentType="application/json",
                      ServerSideEncryption="AES256")
    return row


def _measure(video, config, run):
    if config == "default":
        os.environ["CUTSELL_SIMPLE_ENGINE_TEMPERATURE"] = "default"
    else:
        os.environ["CUTSELL_SIMPLE_ENGINE_TEMPERATURE"] = config
    from cutsell_worker.simple_engine.engine import decide
    from cutsell_worker.simple_engine.llm import call_anthropic
    t0 = time.time()
    try:
        splits, info = decide(video["words"], video["duration"], call_anthropic, run=0)
    except Exception as exc:
        return {"id": video["id"], "config": config, "run": run, "error": str(exc)[:200]}
    visible = [[s["start"], s["end"]] for s in splits if s["visible"]]
    row = {"id": video["id"], "config": config, "run": run, "secs": round(time.time() - t0, 1),
           "visible": visible, "kept_sec": round(sum(b - a for a, b in visible), 2),
           "tokens_in": sum(u.get("input_tokens", 0) for u in info["usage"]),
           "tokens_out": sum(u.get("output_tokens", 0) for u in info["usage"])}
    if video.get("decisions"):
        from scripts.simple_engine_eval import score_video
        total, agree, removed, kept = score_video(video["decisions"], visible)
        row.update({"total": total, "agree": agree,
                    "labels": [1 if sum(max(0.0, min(d["e"], b) - max(d["s"], a)) for a, b in visible) / (d["e"] - d["s"]) >= 0.5 else 0
                               for d in video["decisions"]]})
    return row


@app.local_entrypoint()
def main():
    gold = json.loads((REPO / "benchmarks/simple_engine_gold/gold_v1.json").read_text())["videos"]
    videos = [{"id": v["id"], "duration": v["duration"], "words": v["words"], "decisions": v["decisions"]} for v in gold]
    videos.append(sales_words.remote())
    jobs = [(v, c, r) for v in videos for c in CONFIGS for r in range(RUNS)]
    rows = list(one.map(jobs, return_exceptions=True))
    rows = [r if isinstance(r, dict) else {"error": str(r)[:200]} for r in rows]

    summary = {}
    for c in CONFIGS:
        mine = [r for r in rows if r.get("config") == c]
        errors = sum(1 for r in mine if "error" in r)
        per_run = []
        for run in range(RUNS):
            g = [r for r in mine if r.get("run") == run and "total" in r]
            per_run.append(round(100.0 * sum(r["agree"] for r in g) / max(1, sum(r["total"] for r in g)), 1))
        # repeatability: share of Gold decisions where all runs gave the same keep/delete
        same = tot = 0
        identical_videos = 0
        for v in videos:
            rs = [r for r in mine if r.get("id") == v["id"] and "visible" in r]
            if len(rs) < RUNS:
                continue
            if all(r["visible"] == rs[0]["visible"] for r in rs):
                identical_videos += 1
            if v["decisions"]:
                for k in range(len(v["decisions"])):
                    vals = {r["labels"][k] for r in rs}
                    tot += 1; same += len(vals) == 1
        sales = sorted(r["kept_sec"] for r in mine if r.get("id") == "SALES" and "kept_sec" in r)
        summary[c] = {"gold_percent_per_run": per_run, "errors": errors,
                      "gold_decisions_identical_all_runs_pct": round(100.0 * same / max(1, tot), 1),
                      "videos_with_identical_cut_all_runs": f"{identical_videos}/{len(videos)}",
                      "sales_video_kept_seconds": sales,
                      "tokens_in": sum(r.get("tokens_in", 0) for r in mine), "tokens_out": sum(r.get("tokens_out", 0) for r in mine)}

    print("RESUMEN", json.dumps(summary, ensure_ascii=False))
    stored = {"summary": summary, "rows": [{k: v for k, v in r.items() if k != "labels"} for r in rows]}
    _store.remote(json.dumps(stored))


@app.function(image=image, secrets=[secret], timeout=120)
def _store(body: str):
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")
    s3.put_object(Bucket=os.environ["S3_BUCKET"], Key=PREFIX + "resultado.json", Body=body.encode(),
                  ContentType="application/json", ServerSideEncryption="AES256")
