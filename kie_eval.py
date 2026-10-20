"""Gold (26 videos, stored words only: no Deepgram, no render) through kie.ai vs Anthropic direct, plus kie.ai with a
1-hour prompt cache on every video. Ephemeral `modal run`; nothing deployed, nothing written. Report encrypted."""
import json
import os
import time

import modal

app = modal.App("cutsell-kie-gold")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", "/root/gold_v1.json")
)
PHASE = os.environ.get("KIE_PHASE", "smoke")
# Every key is set for every config (a warm container must not keep the previous video's settings).
CONFIGS = {"A": {"CUTSELL_LLM_PROVIDER": "anthropic", "CUTSELL_SIMPLE_ENGINE_CACHE_SCOPE": "", "CUTSELL_SIMPLE_ENGINE_CACHE_TTL": ""},
           "K": {"CUTSELL_LLM_PROVIDER": "kie", "CUTSELL_SIMPLE_ENGINE_CACHE_SCOPE": "", "CUTSELL_SIMPLE_ENGINE_CACHE_TTL": ""},
           "K1h": {"CUTSELL_LLM_PROVIDER": "kie", "CUTSELL_SIMPLE_ENGINE_CACHE_SCOPE": "all",
                   "CUTSELL_SIMPLE_ENGINE_CACHE_TTL": "1h"}}


def _count(text):
    import requests
    if not text:
        return 0
    body = {"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": text}]}
    h = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for _ in range(3):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages/count_tokens", json=body, headers=h, timeout=30)
            if r.status_code == 200:
                return max(0, int(r.json()["input_tokens"]) - 7)
        except Exception:
            pass
        time.sleep(2)
    return None


@app.function(image=image, secrets=[secret], timeout=900, max_containers=4, retries=0)
def one(item):
    vid, run, cfg = item
    os.environ.update(CONFIGS[cfg])
    from cutsell_worker.simple_engine import engine as E
    from cutsell_worker.simple_engine.llm import call_anthropic
    video = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    calls = []

    def call(prompt, **kw):
        t = time.time()
        text, u = call_anthropic(prompt, **kw)
        calls.append({"step": u.get("step"), "provider": u.get("provider"), "fallback": u.get("provider_fallback"),
                      "kie_dropped": u.get("kie_dropped"), "latency": round(time.time() - t, 2),
                      "in": u.get("input_tokens"), "out": u.get("output_tokens"),
                      "cw": u.get("cache_creation_input_tokens"), "cr": u.get("cache_read_input_tokens"),
                      "cc": u.get("cache_creation"), "credits": u.get("credits_consumed"),
                      "thinking_blocks": u.get("thinking_blocks"), "effort": u.get("effort"),
                      "answer_tokens": _count(text), "answer_chars": len(text), "json": u.get("json_format"),
                      "cache": u.get("cache")})
        return text, u

    t0 = time.time()
    try:
        splits, info = E.decide_reviewed(words, float(video["duration"]), call, run)
        E.orphans(splits, words)
        E.fix_words_and_moments(words, splits, call)
        return {"id": vid, "run": run, "cfg": cfg, "splits": splits, "calls": calls, "wall": round(time.time() - t0, 1),
                "labels": [l[0] for l in info.get("labels") or []]}
    except Exception as exc:
        return {"id": vid, "run": run, "cfg": cfg, "error": repr(exc)[:300], "calls": calls}


@app.local_entrypoint()
def main():
    ids = [v["id"] for v in json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))["videos"]]
    if PHASE == "smoke":
        items = [("B01", 0, "K"), ("B01", 0, "K1h"), ("B02", 0, "K1h"), ("B01", 0, "A")]
        results = []
        for it in items:                                   # in order: the second K1h video can read the cache
            results.append(one.remote(it))
    else:
        first = [(v, 0, "K1h") for v in ids[:1]]
        rest = [(v, r, "K") for r in range(2) for v in ids] + [(v, 0, "K1h") for v in ids[1:]]
        results = [one.remote(first[0])]
        for x in one.map(rest, return_exceptions=True, order_outputs=False):
            results.append(x if isinstance(x, dict) else {"error": repr(x)[:300]})
            with open("engine_check.json", "w") as f:            # partial results survive the 30-minute limit
                json.dump({"gold": results, "partial": True}, f, ensure_ascii=False, default=str)
    with open("engine_check.json", "w") as f:
        json.dump({"gold": results}, f, ensure_ascii=False, default=str)
    cs = [c for g in results for c in g.get("calls", [])]
    print(f"::notice title=kie::calls {len(cs)} fallbacks {sum(1 for c in cs if c['fallback'])} errors {sum(1 for g in results if 'error' in g)}")
