"""Run the SAME editing prompt on OpenAI and Gemini models from inside Modal (which can reach them).
Uses the keys stored in the Modal secret 'cutsell-worker' (OPENAI_API_KEY, GEMINI_API_KEY, and the
Amazon keys + S3_BUCKET of CutSell's private storage).

Privacy (this repository is public, and so are its workflow logs):
- the two payloads (video frames + transcript) are read from the private storage, never from git;
- the result (each model's edit plan) is written back to the private storage, never to git or logs;
- the log shows only model names, timings, token counts and error codes. No keys, no video text."""
import json, os, re, time, urllib.request
import modal
app = modal.App("cutsell-ai-compare")
image = modal.Image.debian_slim().pip_install("boto3")

def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))

def _s3():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")


@app.function(image=image, secrets=[modal.Secret.from_name("cutsell-worker")], timeout=1800)
def run(prefix):
    bucket = os.environ["S3_BUCKET"]
    s3 = _s3()
    payloads = [json.load(s3.get_object(Bucket=bucket, Key=f"{prefix}payload_{v}.json")["Body"]) for v in ("t09", "t05")]
    ok, gk = os.environ.get("OPENAI_API_KEY", "").strip(), os.environ.get("GEMINI_API_KEY", "").strip()
    out = {"has_openai": bool(ok), "has_gemini": bool(gk), "results": []}
    # discover current model ids
    # A provider whose key is refused is recorded (HTTP code only) and skipped; the other one still runs.
    out["provider_errors"] = {}
    om = []
    if ok:
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": "Bearer " + ok}), timeout=60))
            om = sorted(m["id"] for m in d["data"])
        except Exception as e:
            reason = ""
            try:  # OpenAI says why (e.g. invalid_api_key); keep only that code, never the message.
                reason = json.loads(e.read().decode("utf-8", "replace")).get("error", {}).get("code") or ""
            except Exception:
                pass
            out["provider_errors"]["openai"] = ("HTTP " + str(getattr(e, "code", type(e).__name__)) + " " + reason).strip()
    gm = []
    if gk:
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", headers={"x-goog-api-key": gk}), timeout=60))
            gm = [m["name"].split("/")[-1] for m in d["models"] if "generateContent" in m.get("supportedGenerationMethods", [])]
        except Exception as e:
            out["provider_errors"]["gemini"] = "HTTP " + str(getattr(e, "code", type(e).__name__))
    out["openai_models"] = [m for m in om if m.startswith("gpt-5")][:60]
    out["gemini_models"] = [m for m in gm if "flash" in m or "pro" in m][:60]
    def pick(lst, keys):
        for k in keys:
            c = [m for m in lst if k in m and "preview" not in m and "audio" not in m and "image" not in m and "tts" not in m and "live" not in m]
            if c: return sorted(c)[-1]
    targets = []
    # GPT-5.6 Luna and Terra by their official ids (limited preview: they may not appear in /v1/models
    # even when callable). If the key has no access, the call records OpenAI's error code.
    for m in ("gpt-5.6-luna", "gpt-5.6-terra"):
        if ok: targets.append(("openai", m))
    for k in (["gpt-5-mini", "gpt-5.1-mini"], ["gpt-5-nano"]):
        m = pick(om, k)
        if m and ("openai", m) not in targets: targets.append(("openai", m))
    # Gemini: the newest numbered stable Flash-Lite and Flash (e.g. gemini-3.8-flash-lite, gemini-3.8-flash),
    # never an alias like "-latest" or a special model (omni, image, tts, preview).
    import re
    def newest(pattern):
        found = [m for m in gm if re.fullmatch(pattern, m)]
        return max(found, key=lambda m: [int(x) for x in re.findall(r"\d+", m)]) if found else None
    for pattern in (r"gemini-\d+(?:\.\d+)?-flash-lite", r"gemini-\d+(?:\.\d+)?-flash"):
        m = newest(pattern)
        if m and ("gemini", m) not in targets: targets.append(("gemini", m))
    out["targets"] = targets
    for p in payloads:
        for prov, model in targets:
            t0 = time.time()
            try:
                if prov == "openai":
                    content = []
                    for f in p["frames"]:
                        content.append({"type": "input_text", "text": f"t={f['t']:.1f}s"})
                        content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + f["jpg"], "detail": "low"})
                    content.append({"type": "input_text", "text": p["text"]})
                    r = _post("https://api.openai.com/v1/responses", {"model": model, "input": [{"role": "user", "content": content}]},
                              {"Authorization": "Bearer " + ok})
                    txt = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
                    usage = r.get("usage", {})
                else:
                    parts = []
                    for f in p["frames"]:
                        parts.append({"text": f"t={f['t']:.1f}s"})
                        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": f["jpg"]}})
                    parts.append({"text": p["text"]})
                    r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              {"contents": [{"parts": parts}], "generationConfig": {"mediaResolution": "MEDIA_RESOLUTION_LOW"}},
                              {"x-goog-api-key": gk})
                    txt = "".join(pt.get("text", "") for c in r.get("candidates", []) for pt in c.get("content", {}).get("parts", []))
                    usage = r.get("usageMetadata", {})
                out["results"].append({"video": p["video"], "provider": prov, "model": model, "secs": round(time.time() - t0, 1), "text": txt, "usage": usage})
            except Exception as e:
                body = getattr(e, "read", lambda: b"")()[:500].decode("utf-8", "replace")
                out["results"].append({"video": p["video"], "provider": prov, "model": model, "error": str(e) + " " + body})
    key = f"{prefix}resultado_comparacion.json"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8"),
                  ContentType="application/json", ServerSideEncryption="AES256")
    # Only what is safe to show in a public log.
    summary = {"result_key": key, "has_openai": out["has_openai"], "has_gemini": out["has_gemini"],
               "provider_errors": out["provider_errors"],
               "targets": out["targets"], "results": []}
    for r in out["results"]:
        u = r.get("usage") or {}
        summary["results"].append({
            "video": r["video"], "model": r["model"], "secs": r.get("secs"),
            "ok": "error" not in r,
            "error_code": ((r["error"].split(":")[0][:40] + " " + " ".join(re.findall(r'"code":\s*"([a-z_]{3,40})"', r["error"])[:1])).strip()
                           if "error" in r else None),
            "tokens_in": u.get("input_tokens", u.get("promptTokenCount")),
            "tokens_out": u.get("output_tokens", u.get("candidatesTokenCount")),
            "tokens_thinking": (u.get("output_tokens_details") or {}).get("reasoning_tokens", u.get("thoughtsTokenCount")),
        })
    return summary

@app.local_entrypoint()
def main(prefix: str = "pruebas/ai-compare/"):
    res = run.remote(prefix)
    print("guardado en el almacenamiento privado:", res["result_key"])
    print("openai", res["has_openai"], "gemini", res["has_gemini"], "llaves rechazadas:", res["provider_errors"] or "ninguna")
    print("modelos:", res["targets"])
    for r in res["results"]:
        print(r["video"], r["model"], "ok" if r["ok"] else "ERROR " + str(r["error_code"]), f'{r["secs"]}s',
              "tokens entrada/salida/pensar:", r["tokens_in"], r["tokens_out"], r["tokens_thinking"])
