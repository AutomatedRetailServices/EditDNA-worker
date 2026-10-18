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


PASS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["passes", "backbone"], "properties": {
    "passes": {"type": "array", "items": {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}}},
    "backbone": {"type": "array", "items": {"type": "integer"}}}}


@app.function(image=image, secrets=[secret], timeout=900, max_containers=8, retries=1)
def one(item):
    vid, run, cfg = item
    from cutsell_worker.simple_engine import engine as E, llm as L
    from cutsell_worker.simple_engine.llm import call_anthropic
    video = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    units = E.lines_of(words); text = E.transcript(words, units)
    p = L.Prompt(E.PASS_PROMPT + "\nTRANSCRIPT:\n", text, f"\n(run v3pass-{run})", schema=PASS_SCHEMA if cfg == "json" else None)
    try:
        with L.step("pass"):
            out, u = call_anthropic(p)
        d = E.parse(out)
        passes, backbone = d.get("passes", []), d.get("backbone")
        multi = (sum(1 for x in passes if len(x) > 2 and x[2]) >= 2 and isinstance(backbone, list) and len(backbone) == 2)
        return {"id": vid, "run": run, "cfg": cfg, "multi": multi, "backbone": backbone if multi else None,
                "in": u.get("input_tokens"), "out": u.get("output_tokens"), "cr": u.get("cache_read_input_tokens"),
                "cw": u.get("cache_creation_input_tokens"), "answer_tokens": _count(out)}
    except Exception as exc:
        return {"id": vid, "run": run, "cfg": cfg, "error": repr(exc)[:300]}


@app.local_entrypoint()
def main():
    ids = [v["id"] for v in json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))["videos"]]
    items = [(v, r, c) for c in ("text", "json") for r in range(3) for v in ids]
    results = [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in one.map(items, return_exceptions=True)]
    with open("engine_check.json", "w") as f:
        json.dump({"gold": results}, f, ensure_ascii=False, default=str)
