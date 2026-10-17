"""Read-only check of the live engine (no deploy, no writes to Redis/storage).
Runs as an ephemeral `modal run`; results are encrypted to a public key before leaving Modal."""
import json
import os

import modal

app = modal.App("cutsell-engine-check")
secret = modal.Secret.from_name("cutsell-worker")
plain = modal.Image.debian_slim(python_version="3.11")
# The same image definition as modal_cutsell_worker.py (the deployed worker).
worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .env({"CUTSELL_ENGINE": "simple", "CUTSELL_EXPORT_DELIVERY_GATE": "technical_qc"})
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
SHOWN = ("CUTSELL_ENGINE", "CUTSELL_SIMPLE_ENGINE_MODEL")


@app.function(image=plain, timeout=60)
def env_without_secret():
    return sorted(os.environ)


@app.function(image=plain, secrets=[secret], timeout=60)
def env_with_secret():
    return {"names": sorted(os.environ), "values": {k: os.environ.get(k) for k in SHOWN}}


@app.function(image=worker_image, secrets=[secret], timeout=60)
def env_as_worker():
    from cutsell_worker.config import selected_engine
    from cutsell_worker.simple_engine import VERSION
    from cutsell_worker.simple_engine.llm import model_name
    return {"values": {k: os.environ.get(k) for k in SHOWN}, "selected_engine": selected_engine(),
            "engine_version_in_code": VERSION, "model_in_code": model_name()}


def _h(v):
    import hashlib
    return hashlib.sha256(str(v).encode()).hexdigest()[:8]


def _summary(diag, draft):
    if not isinstance(diag, dict) or diag.get("engine") != "simple":
        return {"engine": (diag or {}).get("engine") or "legacy (no simple-engine diagnostics)",
                "diagnostic_keys": sorted((diag or {}).keys())[:20],
                "selected": len(draft.get("selected") or []), "alternates": len(draft.get("alternates") or [])}
    se = diag.get("simple_engine") or {}
    out = {"engine": diag.get("engine"), "engine_version": diag.get("engine_version"), "model": se.get("model"),
           "asr": se.get("asr"), "cut_duration_sec": se.get("cut_duration_sec"), "sources": []}
    for s in se.get("sources") or []:
        usage = s.get("usage") or []
        rev = (s.get("log") or {}).get("revision") or {}
        splits = s.get("splits") or []
        hidden = [x for x in splits if not x.get("visible")]
        reasons = {}
        for x in hidden:
            reasons[x.get("reason") or "?"] = reasons.get(x.get("reason") or "?", 0) + 1
        out["sources"].append({
            "duration_sec": s.get("duration_sec"), "cut_duration_sec": s.get("cut_duration_sec"),
            "language": s.get("language"), "word_count": s.get("word_count"), "passes": s.get("passes"),
            "llm_calls_total": len(usage), "review_calls": rev.get("calls"), "review_errors": rev.get("errors"),
            "review_changes": {"haul": rev.get("haul"), "hide": rev.get("hide"), "unhide": rev.get("unhide")},
            "caption_word_fixes": len((s.get("log") or {}).get("palabras") or []),
            "tokens_in": sum(int(u.get("input_tokens") or 0) for u in usage),
            "tokens_out": sum(int(u.get("output_tokens") or 0) for u in usage),
            "hidden_splits": len(hidden), "hidden_reasons": reasons, "visible_splits": len(splits) - len(hidden),
        })
    out["hidden_with_speech"] = len(draft.get("alternates") or [])
    return out


@app.function(image=worker_image, secrets=[secret], timeout=1800, cpu=2.0, memory=4096)
def inspect(run_test: bool = True, project: str = ""):
    os.environ["ENGINE_CHECK_PROJECT"] = project
    from redis import Redis
    r = Redis.from_url(os.environ["REDIS_URL"])
    report = {"cuts": []}
    # Recent flow-B jobs (results kept 24 h) for context.
    try:
        from rq import Queue
        from rq.job import Job
        from rq.registry import FailedJobRegistry, FinishedJobRegistry
        q = Queue("cutsell", connection=r)
        jobs = []
        for reg, label in ((FinishedJobRegistry(queue=q), "finished"), (FailedJobRegistry(queue=q), "failed")):
            for jid in reg.get_job_ids():
                try:
                    j = Job.fetch(jid, connection=r)
                except Exception:
                    continue
                if not str(j.func_name).endswith("run_flow_b_job"):
                    continue
                jobs.append({"state": label, "ended_at": str(j.ended_at), "project": _h(j.meta.get("project_id")),
                             "error": (str(j.exc_info or "").strip().splitlines() or [""])[-1][:200] if label == "failed" else ""})
        report["flow_b_jobs_last_24h"] = sorted(jobs, key=lambda x: x["ended_at"], reverse=True)[:15]
    except Exception as exc:
        report["flow_b_jobs_error"] = repr(exc)[:200]

    from cutsell_worker.project_store import get_project
    rows = []
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            snap = json.loads(r.get(key))
        except Exception:
            continue
        uid, pid = snap.get("user_id"), snap.get("project_id")
        try:
            proj = get_project(user_id=uid, project_id=pid, client=r)
        except Exception:
            proj = {}
        original = (snap.get("undo") or [snap.get("draft")])[0] or {}
        apple = any(r.exists(k) for k in (f"cutsell:v1:user-apple:{_scope_user(uid)}",)) if uid else False
        rows.append({"snap": snap, "original": original, "project": proj,
                     "when": proj.get("updated_at") or proj.get("created_at") or snap.get("saved_at") or "",
                     "created": proj.get("created_at") or "", "apple": apple})
    rows.sort(key=lambda x: x["created"] or x["when"], reverse=True)
    report["drafts_total"] = len(rows)
    for row in rows[:8]:
        orig, snap = row["original"], row["snap"]
        report["cuts"].append({
            "project": _h(snap.get("project_id")), "user": _h(snap.get("user_id")), "apple_user": row["apple"],
            "title": row["project"].get("title"), "created_at": row["created"], "state": row["project"].get("state"),
            "revision": snap.get("revision"), "original": _summary(orig.get("diagnostics"), orig),
            "current_engine": ((snap.get("draft") or {}).get("diagnostics") or {}).get("engine"),
        })

    if run_test:
        report["test"] = _fresh_run(rows)
    # Stored decisions with their words, for the cuts that hid speech (no new processing).
    report["stored_detail"] = []
    for row in rows[:8]:
        orig = row["original"]
        if not (orig.get("alternates")):
            continue
        clips = sorted([(c.get("start"), c.get("end"), c.get("selected"), c.get("text")) for c in
                        list(orig.get("selected") or []) + list(orig.get("alternates") or [])], key=lambda c: (c[0] or 0))
        se = (orig.get("diagnostics") or {}).get("simple_engine") or {}
        splits = ((se.get("sources") or [{}])[0]).get("splits") or []
        def reason_at(a, b):
            best = None
            for sp in splits:
                o = min(b, sp["end"]) - max(a, sp["start"])
                if o > 0 and (best is None or o > best[0]):
                    best = (o, sp.get("reason"), sp.get("visible"))
            return best[1] if best else None
        report["stored_detail"].append({"project": _h(row["snap"].get("project_id")), "title": row["project"].get("title"),
                                        "clips": [{"start": a, "end": b, "kept": k, "reason": reason_at(a or 0, b or 0), "text": t}
                                                  for a, b, k, t in clips]})
    return report


def _scope_user(uid):
    from cutsell_worker.auth import _user_scope
    return _user_scope(uid)


def _fresh_run(rows):
    """Re-run the live engine code on one REAL video (30-60 s) and return transcript + every cut with its reason.
    Nothing is written anywhere."""
    import tempfile
    from pathlib import Path
    from cutsell_worker.storage import download_source
    from cutsell_worker.media_probe import probe_media
    from cutsell_worker.simple_engine import asr, process
    from cutsell_worker.simple_engine.llm import model_name

    def candidates(apple_only):
        for row in rows:
            if apple_only and not row["apple"]:
                continue
            title = str(row["project"].get("title") or "")
            if title.startswith("copy_B") or "probe" in title.lower():
                continue
            se = (row["original"].get("diagnostics") or {}).get("simple_engine") or {}
            srcs = se.get("sources") or []
            for src in row["snap"].get("sources") or []:
                dur = next((s.get("duration_sec") for s in srcs if s.get("source_asset_id") == src.get("source_asset_id")), None)
                dur = dur or src.get("duration_sec")
                if dur and 30 <= float(dur) <= 60 and src.get("uri"):
                    hidden_speech = len(row["original"].get("alternates") or [])
                    yield (hidden_speech, row, src, float(dur))
    pool = list(candidates(True)) or list(candidates(False))
    if not pool:
        return {"error": "no real 30-60 s video found in stored drafts"}
    want = os.environ.get("ENGINE_CHECK_PROJECT", "")
    if want:
        pool = [x for x in pool if _h(x[1]["snap"].get("project_id")) == want] or pool
    pool.sort(key=lambda x: (x[0], x[1]["created"]), reverse=True)
    hidden_before, row, src, dur = pool[0]
    with tempfile.TemporaryDirectory() as d:
        path = download_source(src["uri"], str(Path(d) / "v.mp4"))
        duration = float(probe_media(path).duration_sec)
        audio = asr.extract_audio(path, str(Path(d) / "a.mp3"))
        heard = asr.transcribe_words(audio)
        words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
        out = process(words, duration, audio, video_path=path)
    def text_in(a, b):
        return " ".join(w["w"] for w in words if a <= (w["s"] + w["e"]) / 2 <= b)
    return {
        "chosen": {"title": row["project"].get("title"), "created_at": row["created"], "apple_user": row["apple"],
                   "duration_sec": round(duration, 2), "hidden_with_speech_in_stored_cut": hidden_before},
        "model": model_name(), "engine_version": out["engine"], "language": heard.get("language"),
        "transcript": " ".join(w["w"] for w in words),
        "splits": [{"start": s["start"], "end": s["end"], "visible": s["visible"], "reason": s.get("reason"),
                    "doubt": s.get("doubt"), "text": text_in(s["start"], s["end"])} for s in out["splits"]],
        "llm_calls": len(out["usage"]), "passes": out["passes"], "review": (out.get("log") or {}).get("revision"),
        "caption_word_fixes": (out.get("log") or {}).get("palabras"),
        "cut_duration_sec": out["cut_duration"],
        "stored_cut": _summary(row["original"].get("diagnostics"), row["original"]),
        "stored_splits": [
            {"start": s.get("start"), "end": s.get("end"), "visible": s.get("visible"), "reason": s.get("reason")}
            for s0 in (((row["original"].get("diagnostics") or {}).get("simple_engine") or {}).get("sources") or [])[:1]
            for s in (s0.get("splits") or [])],
    }


@app.local_entrypoint()
def main():
    plain_names = set(env_without_secret.remote())
    w = env_with_secret.remote()
    secret_names = sorted(set(w["names"]) - plain_names)
    worker = env_as_worker.remote()
    report = inspect.remote(True, "ec57457c")
    report["secret_names"] = secret_names
    report["secret_values_shown"] = w["values"]
    report["as_worker"] = worker
    with open("engine_check.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print("::notice title=secret names::" + ", ".join(secret_names))
    print("::notice title=values::" + json.dumps(w["values"]) + " worker: " + json.dumps(worker))
