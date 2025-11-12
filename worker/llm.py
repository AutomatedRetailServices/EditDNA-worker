import os
from typing import Optional, Tuple
from openai import OpenAI

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or ""

# Always-on LLM; raise if missing key so we don't silently degrade.
def _client() -> OpenAI:
    if not _OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=_OPENAI_API_KEY)

# Returns (keep_score: float in [0,1], reason: str).
def score_clause_multimodal(clause_text: str, data_uri_png: Optional[str], slot_hint: str) -> Tuple[float, str]:
    """
    Ask GPT-4o to judge if this clause belongs in a clean 30–60s product sales edit.
    Prefers: hook clarity, on-script features, natural CTA; rejects: re-takes, 'wait', filler, contradictions.
    """
    client = _client()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict ad editor for short UGC sales videos. "
                "Score if this clause should stay in a clean final cut. "
                "Penalize repeated re-takes, hesitations, contradictions, or off-topic filler. "
                "Reward clear hooks, tangible features/benefits, proof, and one concise CTA near the end."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": f"Slot hint: {slot_hint}. Clause: \"{clause_text}\".\n"
                         f"Return a JSON with fields: keep_score (0..1), brief_reason."}
            ]
        }
    ]

    # If we have a frame, make it multimodal
    if data_uri_png:
        messages[1]["content"].append({"type": "input_image", "image_url": {"url": data_uri_png}})

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content or "{}"
    # simple parse:
    keep = 0.0
    reason = "n/a"
    try:
        import json
        j = json.loads(txt)
        keep = float(j.get("keep_score", 0.0))
        reason = str(j.get("brief_reason", ""))
    except Exception:
        pass
    keep = max(0.0, min(1.0, keep))
    return keep, reason
