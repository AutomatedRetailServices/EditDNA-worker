"""Token measurement of the simple engine on the Gold words: every Claude call with its step, input/output/cache
tokens, max_tokens, thinking setting and the visible answer size (thinking = output - answer, counted with the free
token-counting endpoint). Ephemeral `modal run`; nothing deployed, nothing written to Redis or storage."""
import json
import os

import modal

app = modal.App("cutsell-engine-tokens")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", "/root/gold_v1.json")
)
LABEL = os.environ.get("TOK_LABEL", "base")
RUNS = int(os.environ.get("TOK_RUNS", "2"))


def _count(text):
    import requests
    if not text:
        return 0
    body = {"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": text}]}
    h = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for _ in range(4):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages/count_tokens", json=body, headers=h, timeout=30)
            if r.status_code == 200:
                return max(0, int(r.json()["input_tokens"]) - 7)
        except Exception:
            pass
        import time; time.sleep(2)
    return None


@app.function(image=image, secrets=[secret], timeout=1500, max_containers=8, retries=1)
def one(item):
    vid, run, label = item
    from cutsell_worker.simple_engine import engine as E
    from cutsell_worker.simple_engine.llm import call_anthropic
    video = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    calls = []

    def call(prompt, **kw):
        text, u = call_anthropic(prompt, **kw)
        calls.append({"step": u.get("step"), "in": u.get("input_tokens"), "out": u.get("output_tokens"),
                      "cache_write": u.get("cache_creation_input_tokens"), "cache_read": u.get("cache_read_input_tokens"),
                      "max_tokens": u.get("max_tokens"), "thinking": u.get("thinking"), "effort": u.get("effort"),
                      "stop": u.get("stop_reason"), "answer_tokens": _count(text), "prompt_chars": len(prompt) if isinstance(prompt, str) else None})
        return text, u

    try:
        splits, info = E.decide_reviewed(words, float(video["duration"]), call, run)
        E.orphans(splits, words)
        shown, fixes, moments, _ = E.fix_words_and_moments(words, splits, call)
        return {"id": vid, "run": run, "cfg": label, "splits": splits, "calls": calls, "review": info.get("review"),
                "sale_review": info.get("sale_review"), "labels": [l[0] for l in info.get("labels") or []],
                "fixes": fixes, "moments": moments}
    except Exception as exc:
        return {"id": vid, "run": run, "cfg": label, "error": repr(exc)[:300], "calls": calls}


@app.local_entrypoint()
def main():
    ids = [v["id"] for v in json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))["videos"]]
    items = [(v, r, LABEL) for r in range(RUNS) for v in ids]
    results = [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in one.map(items, return_exceptions=True)]
    with open("engine_check.json", "w") as f:
        json.dump({"gold": results}, f, ensure_ascii=False, default=str)
    cs = [c for g in results for c in g.get("calls", [])]
    print(f"::notice title=tokens::in {sum(c['in'] or 0 for c in cs)} out {sum(c['out'] or 0 for c in cs)} "
          f"errors {sum(1 for g in results if 'error' in g)}")
