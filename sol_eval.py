"""GPT-6.1 Sol on the Gold words: live 1.2.1 code and prompts, every engine call goes to OpenAI instead of Sonnet.
Step 1 lists the OpenAI models this key can call (names only) and picks Sol; no Sol -> nothing is run.
Ephemeral `modal run`; nothing deployed, nothing written."""
import json
import os
import re
import time

import modal

app = modal.App("cutsell-engine-sol-gold")
secret = modal.Secret.from_name("cutsell-worker")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.cutsell.worker.cpu.txt")
    .add_local_dir("cutsell_worker", remote_path="/root/cutsell_worker", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file("benchmarks/simple_engine_gold/gold_v1.json", "/root/gold_v1.json")
)
RUNS = 2


def _kind(prompt):
    return ("pass" if "A PASS is one continuous run" in prompt else "decision" if "STRICT DEFINITION OF A RETAKE" in prompt
            else "haul_review" if "FIRST decide the TYPE of recording" in prompt else "sale_review" if "checking a FIRST CUT" in prompt
            else "fixwords" if "index:word" in prompt else "other")


def _openai(model, prompt):
    import requests
    body = {"model": model, "input": prompt, "max_output_tokens": 16000}
    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"].strip(), "Content-Type": "application/json"}
    last = None
    for attempt in range(4):
        r = requests.post("https://api.openai.com/v1/responses", json=body, headers=headers, timeout=(20, 600))
        if r.status_code == 200:
            d = r.json()
            text = "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message"
                           for c in o.get("content", []) if c.get("type") == "output_text")
            u = d.get("usage") or {}
            return text, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                          "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
        last = f"HTTP {r.status_code} {r.text[:200]}"
        if r.status_code in (400, 401, 403, 404):
            break
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"OpenAI {model}: {last}")


@app.function(image=image, secrets=[secret], timeout=120)
def pick_model():
    import requests
    r = requests.get("https://api.openai.com/v1/models",
                     headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "").strip()}, timeout=30)
    if r.status_code != 200:
        return {"status": r.status_code, "models": [], "pick": None}
    names = sorted(m.get("id", "") for m in r.json().get("data", []))
    sol = [n for n in names if re.search(r"6\.1.*sol|sol.*6\.1", n, re.I)]
    plain = [n for n in sol if not re.search(r"preview|mini|nano|audio|realtime|search|\d{4}-\d\d-\d\d", n)]
    pick = (plain or sol or [None])[0]
    gpt6 = [n for n in names if n.startswith("gpt-6")]
    if pick:      # one tiny call: the key really reaches the model
        try:
            _openai(pick, 'Reply with {"ok":1}')
        except Exception as exc:
            return {"status": 200, "models": gpt6, "pick": None, "error": str(exc)[:200]}
    return {"status": 200, "models": gpt6, "pick": pick}


@app.function(image=image, secrets=[secret], timeout=1500, max_containers=8, retries=1)
def one(item):
    vid, run, model = item
    from cutsell_worker.simple_engine import engine as E
    video = next(v for v in json.load(open("/root/gold_v1.json"))["videos"] if v["id"] == vid)
    words = [w for w in video["words"] if float(w["e"]) > float(w["s"])]
    calls = []

    def call(prompt):
        text, u = _openai(model, prompt)
        calls.append({"kind": _kind(prompt), "in": u.get("input_tokens"), "out": u.get("output_tokens"),
                      "reasoning": u.get("reasoning"), "head": text[:120]})
        return text, u

    try:
        splits, info = E.decide_reviewed(words, float(video["duration"]), call, run)
        E.orphans(splits, words)
        E.fix_words_and_moments(words, splits, call)
        return {"id": vid, "run": run, "cfg": "sol", "model": model, "splits": splits, "calls": calls,
                "review": info.get("review"), "sale_review": info.get("sale_review")}
    except Exception as exc:
        return {"id": vid, "run": run, "cfg": "sol", "error": repr(exc)[:300], "calls": calls}


@app.local_entrypoint()
def main():
    found = pick_model.remote()
    print(f"::notice title=openai models::status {found['status']} pick {found['pick']} gpt6 {found['models'][:20]} "
          f"{found.get('error', '')}")
    results = []
    if found["pick"]:
        ids = [v["id"] for v in json.load(open("benchmarks/simple_engine_gold/gold_v1.json"))["videos"]]
        items = [(v, r, found["pick"]) for r in range(RUNS) for v in ids]
        results = [x if isinstance(x, dict) else {"error": repr(x)[:300]} for x in one.map(items, return_exceptions=True)]
    with open("dedup_eval.json", "w") as f:
        json.dump({"found": found, "gold": results}, f, ensure_ascii=False, default=str)
    cs = [c for g in results for c in g.get("calls", [])]
    print(f"::notice title=sol tokens::in {sum(c['in'] or 0 for c in cs)} out {sum(c['out'] or 0 for c in cs)} "
          f"reasoning {sum(c.get('reasoning') or 0 for c in cs)} errors {sum(1 for g in results if 'error' in g)}")
