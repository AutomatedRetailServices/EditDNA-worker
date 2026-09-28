"""Single-source, bounded diagnostic for an explicitly blocked V2 selection.

The control request is observed unchanged. If it is explicitly blocked, one
text-only request with the same schema tests whether the attached source video
is necessary for the block. Neither response becomes an editorial decision.
"""

import hashlib
import json
import os
from pathlib import Path

import requests


OUTPUT = Path("selection-block-diagnostic")
OUTPUT.mkdir(exist_ok=True)
report = {"source_key": "Yaskira/08.mp4", "control": "not_reached", "comparison": "not_run"}
original_post = requests.Session.post
observed = False


def post_with_diagnostic(self, url, *args, **kwargs):
    global observed
    body = kwargs.get("json")
    if (observed or not url.endswith(":generateContent") or
            not isinstance(body, dict) or not body.get("contents") or
            "cutsell_editorial_engine_v2" not in str(body["contents"][0]["parts"][0].get("text", ""))):
        return original_post(self, url, *args, **kwargs)
    observed = True
    parts = body["contents"][0]["parts"]
    prompt = parts[0]["text"]
    report.update({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "prompt_chars": len(prompt), "part_count": len(parts),
                   "has_av": len(parts) > 1})
    response = original_post(self, url, *args, **kwargs)
    raw = response.json() if response.ok else {}
    feedback = raw.get("promptFeedback") or {}
    report["control"] = {"http_status": response.status_code,
                         "block_reason": feedback.get("blockReason"),
                         "prompt_tokens": (raw.get("usageMetadata") or {}).get("promptTokenCount"),
                         "has_candidates": bool(raw.get("candidates"))}
    if feedback.get("blockReason") == "PROHIBITED_CONTENT" and len(parts) > 1:
        # Diagnostic only: never apply this partial-input output to the edit.
        compact = {"contents": [{"role": "user", "parts": [parts[0]]}],
                   "generationConfig": body["generationConfig"]}
        headers = kwargs.get("headers")
        endpoint = url.rsplit(":", 1)[0]
        count = original_post(self, endpoint + ":countTokens", headers=headers,
                              json={"contents": compact["contents"]}, timeout=60)
        count.raise_for_status()
        tokens = count.json().get("totalTokens")
        output = int(compact["generationConfig"].get("maxOutputTokens", 0))
        # Existing Selection pricing; stop before a comparison above $0.02.
        if type(tokens) is not int or tokens <= 0 or (tokens * .30 + output * 2.50) / 1_000_000 > .02:
            report["comparison"] = "budget_rejected"
        else:
            comparison = original_post(self, url, headers=headers, json=compact,
                                       timeout=kwargs.get("timeout", 120))
            other = comparison.json() if comparison.ok else {}
            report["comparison"] = {"input": "text_only_diagnostic",
                                    "http_status": comparison.status_code,
                                    "block_reason": (other.get("promptFeedback") or {}).get("blockReason"),
                                    "has_candidates": bool(other.get("candidates")),
                                    "prompt_tokens": (other.get("usageMetadata") or {}).get("promptTokenCount")}
    return response


requests.Session.post = post_with_diagnostic
try:
    from cutsell_worker.universal_clean_cut_validation import run_single_universal_clean_cut_validation
    result = run_single_universal_clean_cut_validation(
        "Yaskira/08.mp4", project_id="cutsell-v2-selection-block-diagnostic",
        preview_output=str(OUTPUT / "08.mp4"))
    report["pipeline"] = {"status": "complete", "selection_reasoner_status":
                          result.get("selection_reasoner_status")}
except Exception as exc:
    report["pipeline"] = {"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)[:300]}
finally:
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
