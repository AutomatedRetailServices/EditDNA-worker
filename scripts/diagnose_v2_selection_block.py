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
original_post = requests.post
observed = False


def post_with_diagnostic(url, *args, **kwargs):
    global observed
    body = kwargs.get("json")
    if (observed or not url.endswith(":generateContent") or
            not isinstance(body, dict) or not body.get("contents") or
            "cutsell_editorial_engine_v2" not in str(body["contents"][0]["parts"][0].get("text", ""))):
        return original_post(url, *args, **kwargs)
    observed = True
    parts = body["contents"][0]["parts"]
    prompt = parts[0]["text"]
    report.update({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "prompt_chars": len(prompt), "part_count": len(parts),
                   "has_av": len(parts) > 1})
    response = original_post(url, *args, **kwargs)
    raw = response.json() if response.ok else {}
    feedback = raw.get("promptFeedback") or {}
    report["control"] = {"http_status": response.status_code,
                         "block_reason": feedback.get("blockReason"),
                         "prompt_tokens": (raw.get("usageMetadata") or {}).get("promptTokenCount"),
                         "has_candidates": bool(raw.get("candidates"))}
    if feedback.get("blockReason") == "PROHIBITED_CONTENT" and len(parts) > 1:
        # This diagnostic does not feed a blocked answer into Selection. The
        # prompt evolves with candidate reconstruction, so a previous block
        # does not establish whether *this* request's AV attachment is the
        # trigger. A single text-only control identifies which side differs.
        text_body = {**body, "contents": [{**body["contents"][0], "parts": [parts[0]]}]}
        try:
            control = original_post(url, *args, **{**kwargs, "json": text_body})
            control_raw = control.json() if control.ok else {}
            report["comparison"] = {
                "status": "observed", "http_status": control.status_code,
                "block_reason": (control_raw.get("promptFeedback") or {}).get("blockReason"),
                "has_candidates": bool(control_raw.get("candidates")),
                "prompt_tokens": (control_raw.get("usageMetadata") or {}).get("promptTokenCount"),
                "editorial_output_used": False,
            }
        except requests.RequestException as exc:
            report["comparison"] = {"status": "request_failed", "error_type": type(exc).__name__}
    return response


requests.post = post_with_diagnostic
try:
    from cutsell_worker.universal_clean_cut_validation import run_single_universal_clean_cut_validation
    result = run_single_universal_clean_cut_validation(
        "Yaskira/08.mp4", project_id="cutsell-v2-selection-block-diagnostic",
        preview_output=str(OUTPUT / "08.mp4"))
    selected = [(float(row["start"]), float(row["end"]))
                for row in result.get("selected", ())]
    diagnostics = result.get("diagnostics") or {}
    reasoner = diagnostics.get("unified_selection_reasoner") or {}
    candidate_intervals = {
        row.get("clip_id"): row
        for row in reasoner.get("candidate_intervals") or ()
    }
    decisions = []
    for decision in reasoner.get("decisions") or ():
        row = candidate_intervals.get(decision.get("clip_id"), {})
        decisions.append({
            "clip_id": decision.get("clip_id"),
            "source_order": row.get("source_order"),
            "start": row.get("start"), "end": row.get("end"),
            "model_action": decision.get("model_action"),
            "effective_action": decision.get("effective_action"),
            "relation": decision.get("relation"),
            "reason_code": decision.get("reason_code"),
            "safety_override": decision.get("safety_override"),
        })
    competitions = []
    for contest in diagnostics.get("v2_take_competitions") or ():
        competitions.append({
            **contest,
            "winner_intervals": [candidate_intervals.get(key)
                                 for key in contest.get("winners") or ()],
            "covered_intervals": [candidate_intervals.get(key)
                                  for key in contest.get("covered") or ()],
            "material_unique_intervals": [candidate_intervals.get(key)
                                          for key in contest.get("material_unique") or ()],
        })
    gold_start, gold_end = 120.0, 147.0
    report["pipeline"] = {"status": "complete", "selection_reasoner_status":
                          result.get("selection_reasoner_status"),
                          "candidate_count": reasoner.get("candidate_count"),
                          "competition_review": diagnostics.get("competition_review"),
                          "take_competitions": competitions,
                          "decisions": decisions,
                          "selected_intervals": selected,
                          "gold_keep_retained_sec": round(sum(
                              max(0., min(end, gold_end) - max(start, gold_start))
                              for start, end in selected), 3),
                          "outside_gold_retained_sec": round(sum(
                              max(0., min(end, gold_start) - start) +
                              max(0., end - max(start, gold_end))
                              for start, end in selected), 3),
                          "render_qc": (result.get("live_render_qc") or {}).get("status")}
except Exception as exc:
    report["pipeline"] = {"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)[:300]}
finally:
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
