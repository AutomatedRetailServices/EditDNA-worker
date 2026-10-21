"""Read-only: Auto Zoom AI on her real cuts. (1) what each stored draft carries (auto zoom on/off, AI zooms,
keyframes, the words where they land) -- why the editor said 0; (2) the NEW word-fix call (feat/keyframes-smooth)
on her latest cuts: which moments it marks and how many zooms the new placement makes. Nothing is stored.
Report encrypted."""
import json
import os

import modal

app = modal.App("cutsell-zoom-probe")
secret = modal.Secret.from_name("cutsell-worker")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
         .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"]))
REAL_USER = "1625a6e9"
RERUN = 8


@app.function(image=image, secrets=[secret], timeout=900)
def run():
    import hashlib
    from types import SimpleNamespace
    from redis import Redis
    from cutsell_worker.simple_engine import engine, llm as _llm
    from cutsell_worker.simple_engine_adapter import _zoom_keyframes
    r = Redis.from_url(os.environ["REDIS_URL"])
    snaps = []
    for key in r.scan_iter("cutsell:v1:draft:*", count=500):
        try:
            snap = json.loads(r.get(key))
        except Exception:
            continue
        if hashlib.sha256(str(snap.get("user_id")).encode()).hexdigest()[:8] == REAL_USER:
            snaps.append(snap)
    snaps.sort(key=lambda s: str(s.get("saved_at") or ""), reverse=True)
    rows = []
    for snap in snaps:
        d = snap.get("draft") or {}
        diag = d.get("diagnostics") or {}
        zooms = d.get("zoom_keyframes") or []
        keys = d.get("keyframes")
        words = [(w.get("text") or w.get("w"), float(w.get("start", w.get("s", 0)))) for c in d.get("selected") or []
                 for w in c.get("words") or []]
        def word_at(t):
            near = [w for w in words if abs(w[1] - t) < 0.6]
            return near[0][0] if near else None
        moments = [m for s in (diag.get("simple_engine") or {}).get("sources") or [] for m in s.get("zoom_moments") or []]
        rows.append({"project": hashlib.sha256(str(snap.get("project_id")).encode()).hexdigest()[:6],
                     "saved_at": snap.get("saved_at"), "engine": diag.get("engine"), "version": diag.get("engine_version"),
                     "mix": bool(diag.get("mix") or d.get("mix_code")), "auto_zoom": d.get("auto_zoom_enabled"),
                     "ai_zooms": len([z for z in zooms if (z.get("origin") or "ai") == "ai"]),
                     "zoom_words": [(z.get("kind"), word_at(float(z.get("source_time", 0)) + 0.1)) for z in zooms],
                     "keyframes_key": keys is not None,
                     "keyframes_ai": len([k for k in keys or [] if k.get("origin") == "ai"]),
                     "keyframes_user": len([k for k in keys or [] if k.get("origin") != "ai"]),
                     "zoom_removed": len(d.get("zoom_removed") or []), "revision": snap.get("revision"),
                     "moments_logged": len(moments)})
    # (2) the new call on her latest normal cuts
    rerun = []
    for snap in [s for s in snaps if (s.get("draft") or {}).get("selected")][:RERUN * 2]:
        if len(rerun) >= RERUN:
            break
        d = snap["draft"]
        by_src = {}
        for c in d["selected"]:
            by_src.setdefault(c.get("source_asset_id"), []).append(c)
        src, clips = max(by_src.items(), key=lambda kv: sum(len(c.get("words") or []) for c in kv[1]))
        words = [{"w": w.get("text") or w.get("w"), "s": float(w.get("start", w.get("s", 0))),
                  "e": float(w.get("end", w.get("e", 0)))} for c in clips for w in c.get("words") or []]
        words = [w for w in words if w["w"] and w["e"] > w["s"]]
        if len(words) < 8:
            continue
        splits = [{"start": float(c["start"]), "end": float(c["end"]), "visible": True} for c in clips]
        try:
            _shown, _fixes, moments, usage = engine.fix_words_and_moments(words, splits, _llm.call_anthropic)
        except Exception as exc:
            rerun.append({"error": repr(exc)[:200]})
            continue
        for m in moments:
            m["s"] = words[m["i"]]["s"]; m["w"] = words[m["i"]]["w"]
        log = []
        made = _zoom_keyframes(SimpleNamespace(source_order=1, source_asset_id=src), moments, splits, log)
        cost = sum(float((u or {}).get("cost_usd") or 0) for u in usage)
        rerun.append({"project": hashlib.sha256(str(snap.get("project_id")).encode()).hexdigest()[:6],
                      "seconds": round(sum(float(c["end"]) - float(c["start"]) for c in clips), 1),
                      "moments": [(m["kind"], " ".join(w["w"] for w in words[m["i"]:m["i"] + 4])) for m in moments],
                      "zooms": [(z.kind, z.scale, z.duration, round(z.source_time, 2)) for z in made], "log": log,
                      "usage": usage, "cost_usd": cost})
    return {"drafts": rows, "rerun": rerun, "model": _llm.model_name()}


@app.local_entrypoint()
def main():
    with open("engine_check.json", "w") as f:
        json.dump(run.remote(), f, ensure_ascii=False, default=str)
