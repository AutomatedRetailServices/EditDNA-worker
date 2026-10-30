"""Gold candidate (PO 30 Oct: "add this to the Gold"): the ASR words the engine used for job 92d11b5b.

Read-only: reads that finished job's stored result and writes its words (text + timing, like every Gold video)
to a JSON file that the workflow saves; nothing is printed in the public log except counts.
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("redis>=5.0", "rq>=2.6")
app = modal.App("cutsell-gold-words")
JOB_PREFIX = "92d11b5b"


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=300, cpu=0.5)
def words() -> str:
    import json, os
    from redis import Redis
    from rq.job import Job
    from rq.registry import FinishedJobRegistry

    redis = Redis.from_url(os.environ["REDIS_URL"])
    for job_id in FinishedJobRegistry("cutsell", connection=redis).get_job_ids():
        if not job_id.startswith(JOB_PREFIX):
            continue
        job = Job.fetch(job_id, connection=redis)
        result = job.return_value() if hasattr(job, "return_value") else job.result
        draft = result["draft"]
        rows = {}
        for clip in (draft.get("selected") or []) + (draft.get("alternates") or []):
            for w in clip.get("words") or []:
                s, e = float(w.get("start", w.get("s"))), float(w.get("end", w.get("e")))
                rows[(round(s, 3), round(e, 3))] = str(w.get("text", w.get("w", "")))
        se = (draft.get("diagnostics") or {}).get("simple_engine") or {}
        src = (se.get("sources") or [{}])[0]
        out = {"job": job_id[:8], "language": src.get("language"), "duration": src.get("duration_sec"),
               "engine_word_count": src.get("word_count"),
               "words": [{"w": t, "s": s, "e": e} for (s, e), t in sorted(rows.items())],
               "engine_splits": [{"start": x["start"], "end": x["end"], "visible": x["visible"]} for x in src.get("splits") or []]}
        print(f"words {len(out['words'])} of {out['engine_word_count']}")
        return json.dumps(out, ensure_ascii=False)
    print("job not found")
    return ""


@app.local_entrypoint()
def main():
    text = words.remote()
    if text:
        open("gold_words.json", "w", encoding="utf-8").write(text)
