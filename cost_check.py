"""Read-only: the cuts made after SINCE (the post-deploy test), with every Claude call's tokens and the cut itself.
Nothing is written. The report is encrypted before it leaves the runner."""
import json
import os

import modal

app = modal.App("cutsell-cost-check")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
)
SINCE = os.environ.get("SINCE", "")


@app.function(image=image, secrets=[secret], timeout=600)
def read(since: str):
    from redis import Redis
    from cutsell_worker.project_store import get_project
    r = Redis.from_url(os.environ["REDIS_URL"])
    out = []
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            snap = json.loads(r.get(key))
        except Exception:
            continue
        try:
            proj = get_project(user_id=snap.get("user_id"), project_id=snap.get("project_id"), client=r)
        except Exception:
            proj = {}
        created = str(proj.get("created_at") or "")
        if not created or created.replace("Z", "") < since:
            continue
        original = (snap.get("undo") or [snap.get("draft")])[0] or {}
        diag = original.get("diagnostics") or {}
        se = diag.get("simple_engine") or {}
        srcs = []
        for s in se.get("sources") or []:
            srcs.append({
                "duration_sec": s.get("duration_sec"), "cut_duration_sec": s.get("cut_duration_sec"),
                "word_count": s.get("word_count"), "prompt_cache": s.get("prompt_cache"),
                "calls": [{k: u.get(k) for k in ("step", "input_tokens", "output_tokens", "cache_creation_input_tokens",
                                                   "cache_read_input_tokens", "max_tokens", "effort", "json_format",
                                                   "json_refused", "cache", "stop_reason", "answer_chars")}
                          for u in s.get("usage") or []],
                "splits": [{"start": x.get("start"), "end": x.get("end"), "visible": x.get("visible"),
                            "reason": x.get("reason")} for x in s.get("splits") or []],
                "log_words_fixed": len((s.get("log") or {}).get("palabras") or []),
            })
        out.append({"title": proj.get("title"), "created_at": created, "state": proj.get("state"),
                    "engine_version": diag.get("engine_version"), "model": se.get("model"),
                    "selected": len(original.get("selected") or []), "alternates": len(original.get("alternates") or []),
                    "captions": len(original.get("captions") or se.get("captions") or []),
                    "kept_text": " ".join(w.get("text", "") for c in original.get("selected") or [] for w in c.get("words") or [])[:600],
                    "sources": srcs})
    return sorted(out, key=lambda x: x["created_at"])


@app.local_entrypoint()
def main():
    rows = read.remote(SINCE)
    with open("engine_check.json", "w") as f:
        json.dump({"cuts": rows}, f, ensure_ascii=False, indent=1, default=str)
    print(f"::notice title=cuts::{len(rows)}")
