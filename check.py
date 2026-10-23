"""Security review: can the Modal worker's wake endpoint be called from outside without its token? Status codes only."""
import json, requests
U = "https://automatedretailservices--cutsell-worker-wake.modal.run"
out = {}
for name, kw in (("post_no_token", {"json": {}}), ("post_wrong_token", {"json": {"token": "wrong"}}),
                 ("post_non_ascii_token", {"json": {"token": "ñ"}}), ("get", {})):
    try:
        r = requests.request("GET" if name == "get" else "POST", U, timeout=60, **kw)
        out[name] = [r.status_code, r.text[:80]]
    except Exception as e:
        out[name] = ["error", e.__class__.__name__]
json.dump(out, open("engine_check.json", "w"), indent=1)
