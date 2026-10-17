"""Hybrid test on the Gold words: Sonnet 5.5 keeps the DECISION step; pass, haul review, sale review and word fix +
zoom go to a cheaper model (Claude Haiku 5.5, or Gemini Flash-Lite). Live 1.2.1 code and prompts, only the model of
each call changes. Ephemeral `modal run`; nothing deployed, nothing written."""
import json
import os
import time

import modal

app = modal.App("cutsell-engine-hybrid-gold")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", "/root/gold_v1.json")
)
RUNS = 2
SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"


def _kind(prompt):
    return ("pass" if "A PASS is one continuous run" in prompt else "decision" if "STRICT DEFINITION OF A RETAKE" in prompt
            else "haul_review" if "FIRST decide the TYPE of recording" in prompt else "sale_review" if "checking a FIRST CUT" in prompt
            else "fixwords" if "index:word" in prompt else "other")


def _anthropic(model, prompt):
    import requests
    body = {"model": model, "max_tokens": 16000, "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"].strip(), "anthropic-version": "2023-06-01", "content-type": "application/json"}
    last = None
    for attempt in range(4):
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=(20, 600))
        if r.status_code == 200:
            d = r.json()
            return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text"), dict(d.get("usage") or {})
        last = f"HTTP {r.status_code} {r.text[:200]}"
        if r.status_code in (400, 401, 403, 404):
            break
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Anthropic {model}: {last}")


def _gemini(model, prompt):
    import requests
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}}
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip(), "Content-Type": "application/json"}
    last = None
    for attempt in range(4):
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          json=body, headers=headers, timeout=(20, 600))
        if r.status_code == 200:
            d = r.json()
            parts = (((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            um = d.get("usageMetadata") or {}
            return text, {"input_tokens": um.get("promptTokenCount", 0),
                          "output_tokens": (um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0)}
        last = f"HTTP {r.status_code} {r.text[:200]}"
        if r.status_code in (400, 401, 403, 404):
            break
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini {model}: {last}")


@app.function(image=image, secrets=[secret], timeout=120)
def pick_models():
    import requests
    out = {}
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                     headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip()}, timeout=30)
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in (m.get("supportedGenerationMethods") or [])] if r.status_code == 200 else []
    lite = sorted([n for n in names if "flash-lite" in n and "preview" not in n and "exp" not in n])
    out["gemini_all_flash_lite"] = lite
    # 2.5 Flash-Lite answers 404 "no longer available to new users"; 3.5 Flash-Lite is the current one.
    out["gemini"] = next((n for n in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest") if n in lite), None)
    try:
        t, u = _anthropic(HAIKU, 'Return only {"ok":true}')
        out["haiku_ok"] = True
    except Exception as exc:
        out["haiku_ok"] = repr(exc)[:200]
    return out


@app.function(image=image, secrets=[secret], timeout=1500, max_containers=10, retries=1)
def one(item):
    vid, arm, cheap, run = item
    from cutsell_worker.simple_engine import engine as E
    video = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    calls = []

    def call(prompt):
        kind = _kind(prompt)
        model = SONNET if kind in ("decision", "pass") else cheap      # pass picks the backbone: Haiku broke Y08
        text, u = (_gemini if model.startswith("gemini") else _anthropic)(model, prompt)
        calls.append({"kind": kind, "model": model, "in": u.get("input_tokens"), "out": u.get("output_tokens"), "head": text[:200]})
        return text, u

    try:
        splits, info = E.decide_reviewed(words, float(video["duration"]), call, run)
        E.orphans(splits, words)
        shown, fixes, moments, _ = E.fix_words_and_moments(words, splits, call)
        return {"id": vid, "arm": arm, "run": run, "splits": splits, "calls": calls, "sale_review": info.get("sale_review"),
                "fixes": fixes, "moments": moments}
    except Exception as exc:
        return {"id": vid, "arm": arm, "run": run, "error": repr(exc)[:300], "calls": calls}


@app.local_entrypoint()
def main():
    models = pick_models.remote()
    print("::notice title=models::" + json.dumps(models)[:900])
    arms = []
    if models.get("haiku_ok") is True:
        arms.append(("haiku", HAIKU))
    if models.get("gemini"):
        arms.append(("gemini", models["gemini"]))
    ids = [v["id"] for v in json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))["videos"]]
    items = [(v, arm, m, r) for r in range(RUNS) for arm, m in arms for v in ids]
    results = [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in one.map(items, return_exceptions=True)]
    with open("dedup_eval.json", "w") as f:
        json.dump({"models": models, "gold": results}, f, ensure_ascii=False, default=str)
    for arm, _ in arms:
        cs = [c for g in results if g.get("arm") == arm for c in g.get("calls", [])]
        print(f"::notice title={arm}::in {sum(c['in'] or 0 for c in cs)} out {sum(c['out'] or 0 for c in cs)} "
              f"errors {sum(1 for g in results if g.get('arm') == arm and 'error' in g)}")
