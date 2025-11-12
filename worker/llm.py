# worker/llm.py
import os
import traceback
from typing import Dict, Any, List

from openai import OpenAI

KEEP_MIN = float(os.getenv("KEEP_MIN", "0.28"))  # permissive to avoid empty outputs
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

client = OpenAI()

SYS_PROMPT = (
    "You are a JSON scorer for UGC sales clips. For each spoken clause, "
    'return JSON: {"keep_score": 0..1, "brief_reason": "..."} with no extra keys. '
    "Score higher if the text is on-message, persuasive, and coherent."
)

def _score_text_only(text: str) -> Dict[str, Any]:
    prompt = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Clause:\n{text}\n\nReturn JSON only."}
        ]},
    ]
    r = client.responses.create(
        model=OPENAI_VISION_MODEL,
        input=prompt,
        temperature=0.2,
    )
    out = r.output_text or "{}"
    try:
        import json
        data = json.loads(out)
    except Exception:
        data = {"keep_score": 0.35, "brief_reason": "parse-failed-softkeep"}
    keep = max(0.0, min(1.0, float(data.get("keep_score", 0.35))))
    return {"keep_score": keep, "brief_reason": data.get("brief_reason", "ok")}

def filter_segments_with_llm(asr_segments: List[Dict[str, Any]], media_path_or_none: str) -> Dict[str, Dict[str, Any]]:
    """
    Always-on scoring; on any LLM error we soft-keep to avoid zero-length outputs.
    Returns: {seg_id: {keep_score, brief_reason}}
    """
    decisions: Dict[str, Dict[str, Any]] = {}
    kept = 0
    for seg in asr_segments or []:
        sid = seg.get("id") or ""
        text = (seg.get("text") or "").strip()
        try:
            res = _score_text_only(text) if text else {"keep_score": 0.35, "brief_reason": "empty-text-softkeep"}
        except Exception:
            traceback.print_exc()
            res = {"keep_score": 0.35, "brief_reason": "llm-error-softkeep"}
        keep = float(res.get("keep_score", 0.0))
        if keep >= KEEP_MIN:
            kept += 1
        decisions[sid] = {"keep_score": max(0.0, min(1.0, keep)), "brief_reason": res.get("brief_reason", "")}
    print(f"[LLM] segments={len(asr_segments)} kept>={KEEP_MIN} => {kept}")
    return decisions
