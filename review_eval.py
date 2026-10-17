"""Gold measurement of 1.2.1 (1.2 + sale review): ephemeral `modal run`, nothing deployed, nothing written.
For each sales video and run: the 1.2 cut, then the sale review; the review's 3 answers are replayed (no extra cost)
with one kind of fix at a time, so each kind is measured on its own."""
import json
import os

import modal

app = modal.App("cutsell-engine-review-gold")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", "/root/gold_v1.json")
)
SALES = ["Y03", "Y04", "Y05", "Y06", "Y08", "Y09", "Y10", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
         "B09", "B10", "B11", "B14", "N01", "G01"]
RUNS = 2
VARIANTS = {"restart": ("restart",), "mixed": ("mixed",), "repeat": ("repeat",), "missing": ("missing",),
            "restart+mixed": ("restart", "mixed"), "all": ("restart", "mixed", "repeat", "missing")}


@app.function(image=image, secrets=[secret], timeout=1200, max_containers=10, retries=1)
def one(item):
    vid, run = item
    from cutsell_worker.simple_engine import engine as E
    from cutsell_worker.simple_engine.llm import call_anthropic
    video = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    duration = float(video["duration"])
    usage, review_texts = [], []

    def call(prompt):
        text, u = call_anthropic(prompt)
        usage.append(u)
        if "checking a FIRST CUT" in prompt:
            review_texts.append(text)
        return text, u

    try:
        base, info = E.decide(words, duration, call, run)
        units = E.lines_of(words)
        base_lab = info["labels"]
        _, _, srv = E.sale_review(words, units, base_lab, call, run)
        out = {"id": vid, "run": run, "base": base, "review": srv, "variants": {}, "usage": usage, "answers": review_texts}
        for name, kinds in VARIANTS.items():
            it = iter(review_texts)
            lab, _, sv = E.sale_review(words, units, base_lab, lambda p: (next(it), {}), run, kinds)
            out["variants"][name] = {"splits": E.splits_from_labels(words, [(st, why, False) for st, why in lab], duration),
                                     "applied": sv.get("applied"), "rejected": sv.get("rejected")}
        return out
    except Exception as exc:
        return {"id": vid, "run": run, "error": repr(exc)[:300], "usage": usage}


@app.local_entrypoint()
def main():
    items = [(v, r) for r in range(RUNS) for v in SALES]
    results = [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in one.map(items, return_exceptions=True)]
    with open("gold.json", "w") as f:
        json.dump({"gold": results}, f, ensure_ascii=False, default=str)
    tin = sum(int(u.get("input_tokens") or 0) for g in results for u in g.get("usage", []))
    tout = sum(int(u.get("output_tokens") or 0) for g in results for u in g.get("usage", []))
    print(f"::notice title=review tokens::in {tin} out {tout} errors {sum(1 for g in results if 'error' in g)}")
