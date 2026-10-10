"""CutSell - prueba del CORTE con Gemini (Flash-Lite y 3.8 Flash) en los 25 videos del Gold, desde Modal.
Mismo motor, mismas instrucciones, mismo texto: solo cambia la IA que decide qué se queda.
Usa las llaves que ya están guardadas en el secreto de Modal 'cutsell-worker'. No imprime llaves ni transcripciones.
Version para GitHub: lee payload_corte_gold.json y escribe resultado_corte_gold.json en la carpeta privada
de prueba de Amazon (prefijo --prefix), nunca en GitHub. El registro solo muestra video, modelo, segundos y ok/ERROR.
"""
import json, os, re, time, urllib.request
from pathlib import Path
import modal

app = modal.App("cutsell-cut-compare-gold")
image = modal.Image.debian_slim().pip_install("boto3")
HERE = Path(__file__).parent


def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def _parse(text):
    for cand in (text, re.sub(r"\bL(\d+)\b", r"\1", text)):
        best = None
        for m in re.finditer(r"\{", cand):
            try:
                obj = json.JSONDecoder().raw_decode(cand[m.start():])[0]
            except Exception:
                continue
            if isinstance(obj, dict) and ("hide" in obj or "passes" in obj or "trim" in obj):
                best = obj
        if best is not None:
            return best
    raise ValueError("no JSON")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=3600)
def run(payload):
    ok, gk = os.environ.get("OPENAI_API_KEY", "").strip(), os.environ.get("GEMINI_API_KEY", "").strip()
    om = gm = []
    if ok:
        d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": "Bearer " + ok}), timeout=60))
        om = sorted(m["id"] for m in d["data"])
    if gk:
        d = json.load(urllib.request.urlopen(urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": gk}), timeout=60))
        gm = [m["name"].split("/")[-1] for m in d["models"] if "generateContent" in m.get("supportedGenerationMethods", [])]

    def pick(lst, want):
        if want in lst:
            return want
        c = sorted(m for m in lst if m.startswith(want) and not any(x in m for x in ("preview", "audio", "image", "tts", "live", "search", "chat", "codex")))
        return c[-1] if c else None

    targets = []
    for prov, lst, want in (("gemini", gm, "gemini-3.5-flash-lite"), ("gemini", gm, "gemini-3.8-flash")):
        m = pick(lst, want)
        if m and m.endswith("-lite") is want.endswith("-lite"):
            targets.append((prov, m))

    def call(prov, model, prompt):
        if prov == "openai":
            r = _post("https://api.openai.com/v1/responses", {"model": model, "input": prompt}, {"Authorization": "Bearer " + ok})
            txt = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
            return txt, r.get("usage", {})
        r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                  {"contents": [{"parts": [{"text": prompt}]}]}, {"x-goog-api-key": gk})
        txt = "".join(p.get("text", "") for c in r.get("candidates", []) for p in c.get("content", {}).get("parts", []))
        return txt, r.get("usageMetadata", {})

    def ask(prov, model, tag, prompt, tries=3):
        last = None
        for i in range(tries):
            t = tag if i == 0 else f"{tag}-retry{i}"
            try:
                txt, usage = call(prov, model, prompt + f"\n(run {t})")
                _parse(txt)
                return txt, usage
            except Exception as e:
                last = e
        raise RuntimeError(str(last))

    out = {"targets": targets, "results": []}
    from concurrent.futures import ThreadPoolExecutor
    def one(job):
        v, (prov, model) = job
        res = []
        if True:
            t0 = time.time()
            try:
                ptxt, pu = ask(prov, model, "v3pass-0", payload["pass_prompt"] + "\nTRANSCRIPT:\n" + v["transcript"])
                p = _parse(ptxt)
                passes, backbone = p.get("passes", []), p.get("backbone")
                multi = (sum(1 for x in passes if len(x) > 2 and x[2]) >= 2 and isinstance(backbone, list) and len(backbone) == 2)
                mode = ("MULTI-PASS: the recording has several passes. The BACKBONE is fixed: lines L%d..L%d. Hide everything outside it with code \"p\", "
                        "except a whole sentence with a fact the backbone truly lacks." % (backbone[0], backbone[1])) if multi else (
                        "SINGLE PASS: the recording is one continuous story/pitch. When a sentence is retaken and each version has a concrete detail the other "
                        "lacks (a body part or place, a number, a name, an example, a cause), keep BOTH versions; when one version only rephrases the other with "
                        "nothing new, keep the most complete one.")
                dtxt, du = ask(prov, model, "v3dec-0", payload["decision_prompt"] + "\nFIXED DECISION FROM THE PASS STEP: " + mode + "\n\nTRANSCRIPT:\n" + v["transcript"])
                res.append({"video": v["video"], "provider": prov, "model": model, "secs": round(time.time() - t0, 1),
                                       "pass_text": ptxt, "decision_text": dtxt, "usage": [pu, du]})
            except Exception as e:
                body = getattr(e, "read", lambda: b"")()[:300].decode("utf-8", "replace")
                res.append({"video": v["video"], "provider": prov, "model": model, "error": (str(e) + " " + body)[:400]})
        return res
    jobs = [(v, t) for v in payload["videos"] for t in targets]
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one, jobs):
            out["results"].extend(res)
    return out


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=300)
def store(prefix, result):
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")
    s3.put_object(Bucket=os.environ["S3_BUCKET"], Key=f"{prefix}resultado_corte_gold.json", ServerSideEncryption="AES256",
                  Body=json.dumps(result, ensure_ascii=False, indent=1).encode(), ContentType="application/json")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=300)
def load(prefix):
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")
    return json.load(s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=f"{prefix}payload_corte_gold.json")["Body"])


@app.local_entrypoint()
def main(prefix: str = "pruebas/ai-compare/"):
    res = run.remote(load.remote(prefix))
    store.remote(prefix, res)
    print("modelos:", res["targets"])
    for r in res["results"]:
        print(r["video"], r["model"], r.get("secs"), "ERROR " + r["error"][:150] if "error" in r else "ok")
    errors = sum(1 for r in res["results"] if "error" in r)
    print(f"RESUMEN resultados={len(res['results'])} errores={errors} guardado={prefix}resultado_corte_gold.json")
