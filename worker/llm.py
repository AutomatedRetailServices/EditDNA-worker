# worker/llm.py
import os
import json
from typing import Optional, Tuple
from openai import OpenAI, BadRequestError

_OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_APIKEY")
    or os.getenv("OPENAI_TOKEN")
    or ""
)

# Vision model that supports the Responses API multimodal input
_DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

def _client() -> OpenAI:
    if not _OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=_OPENAI_API_KEY)

def _parse_json(content: str) -> Tuple[float, str]:
    keep, reason = 0.0, "n/a"
    try:
        j = json.loads(content or "{}")
        keep = float(j.get("keep_score", 0.0))
        reason = str(j.get("brief_reason", ""))
    except Exception:
        # If the model returned non-JSON by mistake, clamp to 0
        pass
    # clamp 0..1
    keep = max(0.0, min(1.0, keep))
    return keep, reason

def _system_prompt() -> str:
    return (
        "You are a JSON-only scorer for short UGC sales videos.\n"
        "Task: Score whether this single spoken clause should be KEPT in a clean final cut.\n"
        "- Reward: clear hooks, concrete features/benefits, brief proof, one concise CTA near the end.\n"
        "- Penalize: re-takes, hesitations, contradictions, off-topic filler, repeated lines.\n"
        "Return JSON ONLY with exactly:\n"
        "{ \"keep_score\": <0..1 number>, \"brief_reason\": <string> }"
    )

def _user_text(slot_hint: str, clause_text: str) -> str:
    return (
        f"Slot hint: {slot_hint}\n"
        f"Clause: \"{clause_text}\"\n"
        "Decide keep_score objectively for final ad quality."
    )

# ---- Public API -------------------------------------------------------------

# Returns (keep_score: float in [0,1], reason: str)
def score_clause_multimodal(clause_text: str, data_uri_png: Optional[str], slot_hint: str) -> Tuple[float, str]:
    """
    ALWAYS-ON LLM scoring via the Responses API (multimodal).
    If no image or the server rejects image payload, retries text-only.
    """
    client = _client()

    # 1) Try multimodal (text + optional image)
    try:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _system_prompt()}],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "input_text", "text": _user_text(slot_hint, clause_text)}]
                    + (
                        [{"type": "input_image", "image_url": data_uri_png}]
                        if data_uri_png else []
                    )
                ),
            },
        ]

        # Responses API accepts input=[...] with content parts like input_text/input_image
        resp = client.responses.create(
            model=_DEFAULT_VISION_MODEL,
            input=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = resp.output_text or "{}"
        return _parse_json(content)

    except BadRequestError:
        # 2) Retry text-only with Responses API (no image)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _system_prompt()}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _user_text(slot_hint, clause_text)}
                ],
            },
        ]
        resp = client.responses.create(
            model=_DEFAULT_VISION_MODEL,
            input=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.output_text or "{}"
        return _parse_json(content)

    except Exception:
        # 3) Final safety: return low score; pipeline will keep other clauses
        return 0.0, "fallback-error"
