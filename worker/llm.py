import os
import json
from typing import Optional, Tuple, List, Union

from openai import OpenAI

# You must have OPENAI_API_KEY in env
client = OpenAI()

# Default model for judging clips
LLM_MODEL = os.getenv("EDITDNA_LLM_MODEL", "gpt-4.1-mini")

# Optional global defaults for tone / safety
DEFAULT_TONE_LEVEL = os.getenv("EDITDNA_TONE_LEVEL", "spicy")  # "pg", "casual", "spicy", "chaos"
DEFAULT_ALLOW_SLANG = os.getenv("EDITDNA_ALLOW_SLANG", "true").lower() == "true"
DEFAULT_BRAND_SAFETY_MODE = os.getenv("EDITDNA_BRAND_SAFETY_MODE", "relaxed")  # "strict", "relaxed", "off"


SYSTEM_PROMPT = """
You are an expert UGC ad editor for TikTok-style videos.

You receive:
- A short sentence from a video (taken from a longer script)
- Optionally, a single video frame (image)
- Optionally, a funnel slot hint: HOOK, PROBLEM, FEATURE, PROOF, CTA
- Optional creator tone controls:
  - tone_level: "pg", "casual", "spicy", or "chaos"
  - allow_slang: true/false
  - slang_whitelist: list of slang phrases that are explicitly allowed
  - brand_safety_mode: "strict", "relaxed", or "off"

Your job:
1. Judge how good this sentence is for that slot in a TikTok-style ad.
2. Consider BOTH the words and (if present) the visual frame.
3. Respect the creator's tone controls:
   - If allow_slang is true OR tone_level is "spicy" or "chaos":
       * Do NOT penalize slang, sexual innuendo, or edgy phrasing by itself.
       * Only penalize if the sentence is confusing, self-sabotaging, off-topic for selling, or completely incoherent.
   - If brand_safety_mode = "strict":
       * You may lower the score for content that is explicitly unsafe for broad brands (e.g. hate, violence, illegal activity).
       * BUT still do NOT penalize harmless TikTok-style slang if slang_whitelist includes that phrase.
   - If brand_safety_mode = "relaxed" or "off":
       * Ignore moral judgment. Focus ONLY on persuasive power, clarity, and fit for the slot.

4. Use slang_whitelist as a strong signal:
   - Any phrase that appears in slang_whitelist should NOT be penalized for edginess.
   - Only penalize it if the sentence becomes unreadable or destroys the funnel flow.

Scoring guidelines:
- 0.8–1.0 = very strong for this slot (clear, punchy, persuasive, on-brand)
- 0.5–0.79 = usable but not amazing
- 0.2–0.49 = weak or boring, or slightly confusing
- 0.0–0.19 = unusable, off-topic, or highly confusing

You are NOT a moral censor. You are a performance ad editor.
You care about:
- Does this help sell?
- Is it clear enough for TikTok viewers?
- Does the energy match the slot (HOOK, FEATURE, CTA, etc.)?

Always respond with a JSON object: {"score": float, "reason": string}
""".strip()


def _build_user_content(
    text: str,
    slot_hint: Optional[str] = None,
    include_image: bool = False,
    frame_b64: Optional[str] = None,
    tone_level: Optional[Union[str, int]] = None,
    allow_slang: Optional[bool] = None,
    slang_whitelist: Optional[List[str]] = None,
    brand_safety_mode: Optional[str] = None,
):
    """
    Build the 'content' field for the user message.
    Uses correct OpenAI multimodal format:
    - [{"type": "text", "text": "..."}] + optional {"type": "image_url", ...}
    """

    # Normalize tone_level if int-like
    if isinstance(tone_level, int):
        # Map rough integer levels to labels
        if tone_level <= 0:
            tone_str = "pg"
        elif tone_level == 1:
            tone_str = "casual"
        elif tone_level == 2:
            tone_str = "spicy"
        else:
            tone_str = "chaos"
    else:
        tone_str = (tone_level or DEFAULT_TONE_LEVEL).lower()

    allow_slang_flag = DEFAULT_ALLOW_SLANG if allow_slang is None else bool(allow_slang)
    brand_mode = (brand_safety_mode or DEFAULT_BRAND_SAFETY_MODE).lower()
    slang_whitelist = slang_whitelist or []

    slot_txt = slot_hint or "UNKNOWN"

    meta_lines = [
        f"Slot: {slot_txt}",
        f"Tone level: {tone_str}",
        f"Allow slang: {allow_slang_flag}",
        f"Brand safety mode: {brand_mode}",
        f"Slang whitelist: {', '.join(slang_whitelist) if slang_whitelist else '(none)'}",
        "",
        "Sentence:",
        text.strip(),
        "",
        "Please rate this sentence for this slot and return JSON with keys "
        '"score" (0..1) and "reason".',
    ]
    meta_text = "\n".join(meta_lines)

    content = [
        {
            "type": "text",
            "text": meta_text,
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
    tone_level: Optional[Union[str, int]] = None,
    allow_slang: Optional[bool] = None,
    slang_whitelist: Optional[List[str]] = None,
    brand_safety_mode: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Main entry: used by pipeline.py
    Returns: (score, reason)

    - Uses GPT-4.1-style multimodal if frame is present, else text-only.
    - Respects creator tone controls so TikTok/OF slang is not punished by default.
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
                tone_level=tone_level,
                allow_slang=allow_slang,
                slang_whitelist=slang_whitelist,
                brand_safety_mode=brand_safety_mode,
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
