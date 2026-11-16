"""Engine through Anthropic vs. the separate kie.ai copy, side by side (PO 16 Nov 2026). PAID, on request only.

The same Gold videos (benchmarks/simple_engine_gold/gold_v1.json: the words + the PO's keep/delete marks) go through
the engine's decision + reviews (engine.decide_reviewed) and the word fix twice: once with the ORIGINAL call
(simple_engine.llm.call_anthropic, untouched) and once with the kie.ai copy (simple_engine_kie.llm_kie.call_kie).
Reported per video and in total: agreement with the Gold, decisions that differ, time, cost. Remembered answers are
off (both really ask). Nothing of production is changed; prints only numbers.
Run (CI only): `modal run modal_engine_kie_compare.py --videos Y03,Y04,B04,B06,G02`
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", remote_path="/root/gold_v1.json")
    .add_local_file("scripts/simple_engine_eval.py", remote_path="/root/simple_engine_eval.py")
)
app = modal.App("cutsell-engine-kie-compare")
MAX_VIDEOS = 27
PRICES = {"in": 3.00, "write": 3.75, "read": 0.30, "out": 15.00}        # Sonnet 5.5 standard, US$ per million
OPUS_PRICES = {"in": 4.00, "write": 5.00, "read": 0.40, "out": 20.00}   # Opus 5.5 (22 Sep 2026 price)


def anthropic_cost(usages, prices=PRICES) -> float:
    total = 0.0
    for u in usages:
        total += (int(u.get("input_tokens") or 0) * prices["in"] + int(u.get("cache_creation_input_tokens") or 0) * prices["write"]
                  + int(u.get("cache_read_input_tokens") or 0) * prices["read"] + int(u.get("output_tokens") or 0) * prices["out"]) / 1e6
    return round(total, 5)


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=5400, cpu=1.0, memory=1024)
def compare(ids: list, model: str = "", plan: str = "") -> dict:
    import json
    import os
    import time

    os.environ["CUTSELL_SIMPLE_ENGINE_REMEMBER"] = "0"
    if model:
        os.environ["CUTSELL_SIMPLE_ENGINE_MODEL"] = model       # this container only (a separate test run)
    prices = OPUS_PRICES if "opus" in model else PRICES
    from cutsell_worker.simple_engine import engine
    from cutsell_worker.simple_engine import llm as base
    from cutsell_worker.simple_engine_kie.llm_kie import call_kie
    from simple_engine_eval import score_video

    gold = {v["id"]: v for v in json.load(open("/root/gold_v1.json"))["videos"]}
    routes = {"anthropic": base.call_anthropic, "kie": call_kie}
    models = {}
    if plan == "sonnet-vs-opus":            # PO 16 Nov: both on Anthropic direct, all Gold videos
        routes = {"sonnet": base.call_anthropic, "opus": base.call_anthropic}
        models = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
    elif plan == "opus-kie":                # PO 16 Nov: Opus through kie.ai only, all Gold videos (compared offline)
        routes = {"opus_kie": call_kie}
        models = {"opus_kie": "claude-opus-5-5"}
    rows, kept = [], {}
    for vid in ids[:MAX_VIDEOS]:
        video = gold[vid]
        words, duration = video["words"], float(video["duration"])
        for route, call in routes.items():
            if route in models:
                os.environ["CUTSELL_SIMPLE_ENGINE_MODEL"] = models[route]
                prices = OPUS_PRICES if "opus" in models[route] else PRICES
            row = {"video": vid, "route": route}
            started = time.monotonic()
            try:
                splits, info = engine.decide_reviewed(words, duration, call)
                shown, fixes, moments, fix_usage = engine.fix_words_and_moments(words, splits, call)
                usages = list(info["usage"]) + list(fix_usage if isinstance(fix_usage, list) else [fix_usage])
                visible = [[s["start"], s["end"]] for s in splits if s["visible"]]
                total, agree, removed, wrongly_kept = score_video(video["decisions"], visible)
                cost = (round(sum(float(u.get("cost_usd") or 0) for u in usages), 5) if route == "kie"
                        else anthropic_cost(usages, prices))
                row.update(ok=True, seconds=round(time.monotonic() - started, 1), gold=f"{agree}/{total}",
                           removed_wrongly=removed, kept_wrongly=wrongly_kept, cost_usd=cost,
                           fixes=len(fixes), moments=len(moments), calls=len(usages),
                           reasks=sum(int(u.get("reasks") or 0) for u in usages))
                kept[(vid, route)] = visible
            except Exception as exc:                  # noqa: BLE001
                row.update(ok=False, error=f"{exc.__class__.__name__}: {str(exc)[:160]}",
                           seconds=round(time.monotonic() - started, 1))
            rows.append(row)
        names = list(routes)
        if len(names) < 2:
            continue
        first, second = names[:2]
        a, k = kept.get((vid, first)), kept.get((vid, second))
        if a is not None and k is not None:
            same = sum(1 for d in video["decisions"]
                       if (sum(max(0, min(d["e"], b) - max(d["s"], x)) for x, b in a) >= 0.5 * (d["e"] - d["s"]))
                       == (sum(max(0, min(d["e"], b) - max(d["s"], x)) for x, b in k) >= 0.5 * (d["e"] - d["s"])))
            rows.append({"video": vid, "route": "both", "same_decisions": f"{same}/{len(video['decisions'])}"})
    return {"rows": rows}


@app.local_entrypoint()
def main(videos: str = "Y03,Y04,B04,B06,G02", model: str = "", plan: str = ""):
    ids = [v.strip() for v in videos.split(",") if v.strip()][:MAX_VIDEOS]
    print(f"MODEL {model or 'default (claude-sonnet-5-5)'}")
    if videos == "all":
        import json
        ids = [v["id"] for v in json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))["videos"]]
    print(f"PLAN {plan or 'anthropic-vs-kie'} · {len(ids)} videos")
    out = compare.remote(ids, model, plan)
    for r in out["rows"]:
        print("RESULT", r)
    for route in sorted({r.get("route") for r in out["rows"]} - {"both", None}):
        rs = [r for r in out["rows"] if r.get("route") == route and r.get("ok")]
        if rs:
            agree = sum(int(r["gold"].split("/")[0]) for r in rs); total = sum(int(r["gold"].split("/")[1]) for r in rs)
            print(f"TOTAL {route}: gold {agree}/{total} ({100*agree/max(1,total):.0f}%), "
                  f"US${sum(r['cost_usd'] for r in rs):.3f}, {sum(r['seconds'] for r in rs):.0f}s, "
                  f"re-asks {sum(r.get('reasks', 0) for r in rs)}, failed {sum(1 for r in out['rows'] if r.get('route') == route and not r.get('ok'))}")
