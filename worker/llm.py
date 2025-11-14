import os
import json
from typing import Optional, Tuple

from openai import OpenAI

# You must have OPENAI_API_KEY in env
client = OpenAI()

# Default model for judging clips
LLM_MODEL = os.getenv("EDITDNA_LLM_MODEL", "gpt-4.1-mini")


SYSTEM_PROMPT = """
You are an expert UGC ad editor.

You receive:
- A short sentence from a video (taken from a longer script)
- Optionally, a single video frame (image)
- Optionally, a funnel slot hint: HOOK, PROBLEM, FEATURE, PROOF, CTA

Your job:
1. Judge how good this sentence is for that slot in a TikTok-style ad.
2. Consider both the words and (if present) the visual frame.
3. Return a score between 0.0 and 1.0 (float) and a short reason.

Guidelines:
- 0.8–1.0 = very strong for this slot (clear, punchy, persuasive)
- 0.5–0.79 = usable but not amazing
- 0.2–0.49 = weak or boring
- 0.0–0.19 = unusable, off-topic, or confusing

Always respond with a JSON object: {"score": float, "reason": string}
""".strip()


def _build_user_content(
    text: str,
    slot_hint: Optional[str] = None,
    include_image: bool = False,
    frame_b64: Optional[str] = None,
):
    """
    Build the 'content' field for the user message.
    Uses correct OpenAI multimodal format:
    - [{"type": "text", "text": "..."}] + optional {"type": "image_url", ...}
    """
    slot_txt = slot_hint or "UNKNOWN"
    prompt = (
        f"Slot: {slot_txt}\n\n"
        f"Sentence:\n{text.strip()}\n\n"
        "Please rate this sentence for this slot and return JSON with keys "
        '"score" (0..1) and "reason".'
    )

    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    if include_image and frame_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}"
                },
            }
        )

    return content


def score_clause_multimodal(
    text: str,
    frame_b64: Optional[str] = None,
    slot_hint: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Main entry: used by pipeline.py
    Returns: (score, reason)
    Uses GPT-4o-style multimodal if frame is present, else text-only.
    """
    include_image = bool(frame_b64)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": _build_user_content(
                text=text,
                slot_hint=slot_hint,
                include_image=include_image,
                frame_b64=frame_b64,
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception as e:
        # If the LLM call completely fails, fall back to neutral score
        return 0.5, f"llm_error: {e}"

    raw = resp.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
    except Exception:
        # If parsing JSON fails, still don't crash the pipeline
        return 0.5, f"parse_error: {raw[:200]}"

    score = data.get("score", 0.5)
    reason = data.get("reason", "no_reason")

    try:
        score = float(score)
    except Exception:
        score = 0.5

    # Clamp 0..1
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    return score, str(reason)
