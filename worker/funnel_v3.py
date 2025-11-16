"""
funnel_v3.py

Smarter funnel composer for EditDNA:
- Filters out obvious bad takes / fragments / outtakes.
- Avoids duplicates.
- Picks a clear HOOK, good FEATURE chain, and CTA.
- Keeps the same output shape as the old composer:
  result["composer"] and result["composer_human"].
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set


@dataclass
class Clip:
    id: str
    slot: str
    text: str
    score: float
    semantic_score: float
    visual_score: float
    start: float
    end: float
    llm_reason: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "Clip":
        """
        Accepts either:
        - a dict like the JSON in result["clips"], or
        - an object with attributes (id, slot, text, etc.).
        """

        def get(field: str, default=None):
            if isinstance(raw, dict):
                return raw.get(field, default)
            return getattr(raw, field, default)

        meta = get("meta", {}) or {}
        # Some pipelines store the "slot" in meta["slot"]
        slot = get("slot") or meta.get("slot") or ""

        return cls(
            id=get("id", ""),
            slot=str(slot or "").upper(),
            text=str(get("text", "") or "").strip(),
            score=float(
                meta.get("score", get("score", 0.0) or 0.0)
            ),
            semantic_score=float(
                meta.get("semantic_score", get("semantic_score", 0.0) or 0.0)
            ),
            visual_score=float(
                meta.get("visual_score", get("visual_score", 0.0) or 0.0)
            ),
            start=float(get("start", 0.0) or 0.0),
            end=float(get("end", 0.0) or 0.0),
            llm_reason=str(get("llm_reason", "") or ""),
        )


# ---------- Heuristics ----------


BAD_PHRASE_SNIPPETS = [
    "is that good",
    "is that funny",
    "that one good",
    "yeah.",
    "i think they're really good",
    "why can't i remember after that",
    "wait not moisture control",
    "oh no, i don't know why they're",
]

BAD_REASON_SNIPPETS = [
    "very vague",
    "vague and",
    "weak for a story slot",
    "too short",
    "extremely short",
    "incomplete and",
    "ends abruptly",
    "confusing and unclear",
    "does not engage the viewer",
    "does not effectively",
    "unusable for the story slot",
]

# Anything with these words is **probably** an outtake / blooper-ish.
OUTTAKE_HINTS = [
    "is that good",
    "is that funny",
    "that one good",
    "yeah.",
    "wait.",
    "wait not",
]


def _looks_like_fragment(text: str) -> bool:
    """
    Detect fragments like:
    - Very short 1–3 word lines.
    - No punctuation and generic phrasing.
    """
    t = text.strip()
    if not t:
        return True

    words = t.split()
    if len(words) <= 2:
        return True
    if len(words) <= 4 and t[-1] not in ".?!":
        return True

    return False


def _has_bad_phrase(text: str) -> bool:
    t = text.lower()
    return any(sn in t for sn in BAD_PHRASE_SNIPPETS)


def _looks_like_outtake(text: str, llm_reason: str) -> bool:
    t = text.lower()
    r = llm_reason.lower()
    if any(h in t for h in OUTTAKE_HINTS):
        return True
    if "outtake" in r:
        return True
    return False


def _llm_reason_flags_bad(llm_reason: str) -> bool:
    r = llm_reason.lower()
    return any(sn in r for sn in BAD_REASON_SNIPPETS)


def _is_bad_clip(c: Clip, min_score: float) -> bool:
    """
    Decide whether to drop a clip entirely from the funnel.
    """

    # Always drop clips under the global min score
    if c.score < min_score:
        return True

    # Outtakes, bloopers, clearly weak.
    if _looks_like_outtake(c.text, c.llm_reason):
        return True

    # Super short / fragment-like lines are bad for HOOK/FEATURE/PROBLEM
    if c.slot in {"HOOK", "FEATURE", "PROBLEM", "STORY"} and _looks_like_fragment(
        c.text
    ):
        return True

    # LLM explicitly calling it vague/weak/etc.
    if _llm_reason_flags_bad(c.llm_reason):
        return True

    return False


def _dedupe_by_text(clips: List[Clip]) -> List[Clip]:
    seen: Set[str] = set()
    out: List[Clip] = []
    for c in clips:
        key = c.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _pick_best_hook(
    all_clips: List[Clip],
    slots_raw: Dict[str, List[Any]],
    min_score: float,
) -> Optional[Clip]:
    """
    Strategy:
    1. Try HOOK slot first (if exists) and pass filters.
    2. Otherwise, fall back to a strong FEATURE/PROBLEM/STORY line.
    """

    # 1) HOOK slot candidates
    hook_candidates: List[Clip] = []
    for raw in slots_raw.get("HOOK", []):
        c = Clip.from_raw(raw)
        hook_candidates.append(c)

    hook_candidates = [
        c for c in hook_candidates if not _is_bad_clip(c, min_score)
    ]
    if hook_candidates:
        # Prefer highest score among clean hooks
        hook_candidates.sort(key=lambda c: (c.score, c.semantic_score), reverse=True)
        return hook_candidates[0]

    # 2) Fallback from FEATURE / PROBLEM / STORY
    fallback_candidates: List[Clip] = []
    for c in all_clips:
        if c.slot in {"FEATURE", "PROBLEM", "STORY"} and not _is_bad_clip(
            c, min_score
        ):
            fallback_candidates.append(c)

    if not fallback_candidates:
        return None

    # Prefer earlier in timeline, then higher semantic_score
    fallback_candidates.sort(
        key=lambda c: (-c.semantic_score, -c.score, c.start)
    )
    return fallback_candidates[0]


def _pick_features(
    all_clips: List[Clip],
    min_score: float,
    hook_id: Optional[str],
    cta_id: Optional[str],
    max_features: int = 7,
) -> List[Clip]:
    """
    Choose a chain of FEATURE/PROBLEM/STORY clips:
    - Pass bad-clip filter.
    - Not the HOOK or CTA.
    - De-duplicated by text.
    - Sorted by start time (so story flows).
    """

    candidates: List[Clip] = []
    for c in all_clips:
        if c.id == hook_id or c.id == cta_id:
            continue
        if c.slot not in {"FEATURE", "PROBLEM", "STORY"}:
            continue
        if _is_bad_clip(c, min_score):
            continue
        candidates.append(c)

    # Sort by start time so they appear in natural order
    candidates.sort(key=lambda c: c.start)

    # Remove duplicate lines like repeated “Worry no more…” lines
    candidates = _dedupe_by_text(candidates)

    # Keep only top N
    return candidates[:max_features]


def _pick_cta(
    all_clips: List[Clip],
    slots_raw: Dict[str, List[Any]],
    min_score: float,
) -> Optional[Clip]:
    """
    Prefer CTA slot clips, otherwise fallback to last strong line.
    """

    # 1) CTA slot
    cta_candidates: List[Clip] = []
    for raw in slots_raw.get("CTA", []):
        c = Clip.from_raw(raw)
        if not _is_bad_clip(c, min_score):
            cta_candidates.append(c)

    if cta_candidates:
        cta_candidates.sort(
            key=lambda c: (c.score, c.semantic_score), reverse=True
        )
        return cta_candidates[0]

    # 2) Fallback: strongest line near the end of video
    usable: List[Clip] = [
        c for c in all_clips if not _is_bad_clip(c, min_score)
    ]
    if not usable:
        return None

    usable.sort(key=lambda c: (c.end, c.score), reverse=True)
    return usable[0]


# ---------- Public entrypoint ----------


def compose_funnel_v3(
    slots_raw: Dict[str, List[Any]],
    clips_raw: List[Dict[str, Any]],
    min_score: float = 0.5,
) -> Dict[str, Any]:
    """
    Main function used by the worker pipeline.

    Parameters
    ----------
    slots_raw : dict
        The JSON-ish slots mapping (HOOK/FEATURE/PROBLEM/PROOF/CTA → list of items).
    clips_raw : list
        The JSON-ish list of clips (like result["clips"]).
    min_score : float
        Minimum score threshold for a clip to be considered.

    Returns
    -------
    dict with keys:
        - "composer"
        - "composer_human"
    """

    # Normalize all clips
    all_clips: List[Clip] = [Clip.from_raw(c) for c in clips_raw]

    # Pick CTA first (we don't want to accidentally reuse it as feature)
    cta = _pick_cta(all_clips, slots_raw, min_score)

    # Pick HOOK
    hook = _pick_best_hook(all_clips, slots_raw, min_score)

    # Pick feature chain
    hook_id = hook.id if hook else None
    cta_id = cta.id if cta else None
    features = _pick_features(
        all_clips,
        min_score=min_score,
        hook_id=hook_id,
        cta_id=cta_id,
    )

    # Build composer dict
    composer: Dict[str, Any] = {
        "hook_id": hook_id,
        "feature_ids": [c.id for c in features],
        "cta_id": cta_id,
        "used_clip_ids": [],
        "min_score": min_score,
    }

    ordered_ids: List[str] = []
    if hook_id:
        ordered_ids.append(hook_id)
    ordered_ids.extend([c.id for c in features])
    if cta_id:
        ordered_ids.append(cta_id)

    composer["used_clip_ids"] = ordered_ids

    # Human-readable string (for debugging)
    lines: List[str] = []
    lines.append("===== EDITDNA FUNNEL COMPOSER V3 =====")
    lines.append("")

    if hook:
        lines.append(f"HOOK ({hook.id}, score={hook.score:.2f}):")
        lines.append(f'  "{hook.text}"')
        lines.append("")

    if features:
        lines.append("FEATURES (kept):")
        for c in features:
            lines.append(
                f'  - [{c.id}] score={c.score:.2f} → "{c.text}"'
            )
        lines.append("")

    if cta:
        lines.append(f"CTA ({cta.id}, score={cta.score:.2f}):")
        lines.append(f'  "{cta.text}"')
        lines.append("")

    lines.append("")
    lines.append("FINAL ORDER TIMELINE:")
    idx = 1
    for cid in ordered_ids:
        clip = next((c for c in all_clips if c.id == cid), None)
        if not clip:
            continue
        lines.append(f'{idx}) {clip.id} → "{clip.text}"')
        idx += 1

    lines.append("")
    lines.append("=====================================")

    composer_human = "\n".join(lines)

    return {
        "composer": composer,
        "composer_human": composer_human,
    }
