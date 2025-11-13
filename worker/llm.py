import os
import json
from typing import Optional, Tuple

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client: Optional[OpenAI] = None


def get_client() -> Optional[OpenAI]:
    global _client
    if _client is not None:
        return _client
    if not OPENAI_API_KEY:
        return None
    _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def score_clause_multimodal(
    text: str,
    frame_b64: Optional[str] = None,
    slot_hint: Optional[str] = None,
) -> Tuple[str, float, str]:
    """
    Ask GPT to:
    - classify the clause into one funnel slot
      (HOOK / PROBLEM / FEATURE / PROOF / CTA / STORY)
    - give a quality score 0–1
    - return a short reason

    If frame_b64 is provided, it is passed as an image_url (multimodal).
    """

    client = get_client()
    if client is None:
        # LLM disabled → fall back
        return (slot_hint or "STORY", 0.5, "LLM disabled (no OPENAI_API_KEY)")

    system_prompt = (
        "You are an assistant scoring short clauses from TikTok product videos.\n"
        "For each input, decide which funnel slot it belongs to:\n"
        " - HOOK: grabs attention, curiosity, bold opening\n"
        " - PROBLEM: describes pain, struggle, frustration\n"
        " - FEATURE: describes product features, ingredients, what it does\n"
        " - PROOF: testimonials, results, evidence, 'I use this', numbers\n"
        " - CTA: asks viewer to take action: buy, click, use code, shop now\n"
        " - STORY: neutral description that doesn't fit others strongly\n\n"
        "Return ONLY a JSON object with keys: slot, score, reason.\n"
        "score must be a number between 0.0 and 1.0."
    )

    user_text = f"Text: {text.strip()}"
    if slot_hint:
        user_text += f"\n\nOptional slot hint: {slot_hint}"

    content = [
        {"type": "text", "text": user_text}
    ]

    if frame_b64:
        # Multimodal – include frame as image_url
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                },
            }
        )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        # Fallback if model doesn't respect JSON
        return (slot_hint or "STORY", 0.5, f"Bad JSON from LLM: {raw[:100]}")

    slot = str(data.get("slot") or (slot_hint or "STORY")).upper().strip()
    if slot not in {"HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA", "STORY"}:
        slot = "STORY"

    try:
        score = float(data.get("score", 0.5))
    except Exception:
        score = 0.5

    reason = str(data.get("reason", ""))[:300]
    return slot, max(0.0, min(1.0, score)), reason
