"""One-off diagnosis (PO 30 Oct): a ~50 s cut "came out cut short, the message is not finished".

Read-only: reads the newest finished cut jobs from the queue (job results already stored there) and, for the
cut whose source is ~50.6 s long, prints NUMBERS AND ENGINE LABELS ONLY (the repository and its Actions logs
are public): source length, each part's start / end / kept-or-hidden / engine reason / word count, and where
the last spoken word is. No transcript text, no names, no keys. Nothing is changed, nothing is deployed.
Run: modal run ops/diag_cut_ending.py  (ephemeral app)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("redis>=5.0", "rq>=2.6")
)
app = modal.App("cutsell-diag-cut-ending")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=300, cpu=0.5)
def diagnose():
    import os
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    from rq.registry import FinishedJobRegistry

    redis = Redis.from_url(os.environ["REDIS_URL"])
    shown = 0
    for name in ("cutsell", "cutsell-bulk", "default"):
        registry = FinishedJobRegistry(name, connection=redis)
        ids = registry.get_job_ids()[-60:]
        print(f"queue {name}: {len(registry.get_job_ids())} finished, reading the newest {len(ids)}")
        jobs = []
        for job_id in ids:
            try:
                jobs.append(Job.fetch(job_id, connection=redis))
            except Exception:
                continue
        jobs.sort(key=lambda j: j.ended_at or j.created_at, reverse=True)
        for job in jobs:
            result = job.return_value() if hasattr(job, "return_value") else job.result
            if not isinstance(result, dict) or not isinstance(result.get("draft"), dict):
                continue
            draft = result["draft"]
            se = (draft.get("diagnostics") or {}).get("simple_engine") or {}
            for src in se.get("sources") or []:
                dur = float(src.get("duration_sec") or 0)
                if not 48.0 <= dur <= 53.5:
                    continue
                shown += 1
                print(f"\nJOB {job.id[:8]} ended {job.ended_at} source {dur:.2f}s cut {src.get('cut_duration_sec')}s "
                      f"words {src.get('word_count')} lang {src.get('language')} passes {src.get('passes')}")
                for i, s in enumerate(src.get("splits") or []):
                    print(f"  part {i:02d} {float(s['start']):6.2f}-{float(s['end']):6.2f} "
                          f"{'KEEP' if s.get('visible') else 'hide'} reason={s.get('reason')}")
                caps = se.get("captions") or []
                if caps:
                    print(f"  captions: {len(caps)} words, last caption ends at {float(caps[-1]['e']):.2f}s of the cut")
                sel = draft.get("selected") or []
                print(f"  kept clips: {len(sel)}; last kept clip ends at source {max((float(c.get('end') or 0) for c in sel), default=0):.2f}s")
                # Every word the engine had in 17-35 s: timing, kept/hidden, length and end punctuation only.
                rows = []
                for kind, clips in (("KEEP", draft.get("selected") or []), ("hide", draft.get("alternates") or [])):
                    for c in clips:
                        for w in c.get("words") or []:
                            st, en = float(w.get("start", w.get("s", 0))), float(w.get("end", w.get("e", 0)))
                            t = str(w.get("text", w.get("w", "")))
                            rows.append((st, en, kind, len(t), t[-1:] if t[-1:] in ".,?!" else "-"))
                for st, en, kind, n, p in sorted(rows):
                    if 17.0 <= st <= 35.0:
                        print(f"  word {st:6.2f}-{en:6.2f} {kind} len={n} end={p}")
                for line in (src.get("log") or [])[-12:]:
                    if isinstance(line, dict):
                        print("  log:", {k: v for k, v in line.items() if isinstance(v, (int, float, bool)) or k in ("kind", "step", "why", "reason")})
            if shown >= 1:
                return


@app.local_entrypoint()
def main():
    diagnose.remote()
