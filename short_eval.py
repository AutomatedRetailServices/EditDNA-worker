"""A/B on the Gold words: live 1.2.1 prompts vs 1.2.2 (JSON-only answers). Same code, only the prompt texts differ.
Ephemeral `modal run`; nothing deployed, nothing written."""
import json

import modal

app = modal.App("cutsell-engine-short-gold")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir("eval_live_prompts", remote_path="/root/eval_live_prompts")
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", "/root/gold_v1.json")
)
RUNS = 2
NAMES = {"PASS_PROMPT": "pass_v1", "DECISION_PROMPT": "decision_v1", "REVIEW_PROMPT": "review_v1",
         "SALE_REVIEW_PROMPT": "sale_review_v1", "FIXWORDS_PROMPT": "fixwords_v2"}


@app.function(image=image, secrets=[secret], timeout=1500, max_containers=10, retries=1)
def one(item):
    vid, version, run = item
    from cutsell_worker.simple_engine import engine as E
    from cutsell_worker.simple_engine.llm import call_anthropic
    if version == "live":
        for attr, name in NAMES.items():
            setattr(E, attr, open(f"/root/eval_live_prompts/{name}.txt", encoding="utf-8").read())
    video = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    calls = []

    def call(prompt):
        text, u = call_anthropic(prompt)
        kind = ("pass" if "A PASS is one continuous run" in prompt else "decision" if "STRICT DEFINITION OF A RETAKE" in prompt
                else "haul_review" if "FIRST decide the TYPE of recording" in prompt else "sale_review" if "checking a FIRST CUT" in prompt
                else "fixwords" if "index:word" in prompt else "other")
        calls.append({"kind": kind, "in": u.get("input_tokens"), "out": u.get("output_tokens"), "head": text[:160]})
        return text, u

    try:
        splits, info = E.decide_reviewed(words, float(video["duration"]), call, run)
        E.orphans(splits, words)
        E.fix_words_and_moments(words, splits, call)
        return {"id": vid, "version": version, "run": run, "splits": splits, "calls": calls,
                "sale_review": info.get("sale_review")}
    except Exception as exc:
        return {"id": vid, "version": version, "run": run, "error": repr(exc)[:300], "calls": calls}


@app.local_entrypoint()
def main():
    ids = [v["id"] for v in json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))["videos"]]
    items = [(v, ver, r) for r in range(RUNS) for ver in ("live", "short") for v in ids]
    results = [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in one.map(items, return_exceptions=True)]
    with open("dedup_eval.json", "w") as f:
        json.dump({"gold": results}, f, ensure_ascii=False, default=str)
    for ver in ("live", "short"):
        cs = [c for g in results if g.get("version") == ver for c in g.get("calls", [])]
        print(f"::notice title={ver} tokens::in {sum(c['in'] or 0 for c in cs)} out {sum(c['out'] or 0 for c in cs)} "
              f"errors {sum(1 for g in results if g.get('version') == ver and 'error' in g)}")
