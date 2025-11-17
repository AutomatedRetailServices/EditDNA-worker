# ========= EDITDNA V3 FUNNEL COMPOSER (TEXT + VISION + SLOTS) =========
# Drop this whole block into worker/pipeline.py and REMOVE old composer logic.

import re
from typing import Dict, List, Tuple, Optional

# ------------  HARD FILTER FOR BAD TAKES / MICRO-RETakes / BLOOPERS  ------------ #

BAD_PHRASE_PATTERNS = [
    r"\bwait[.,!?]?$",                      # "wait." / "wait?" at the end
    r"\bwait not\b",
    r"\bI don't know why\b",
    r"\bWhy can't I remember\b",
    r"\bIs that funny\??",
    r"\bIs that good\??",
    r"^Yeah[.?!]?$",
    r"^That one good\??",
]

BAD_REASON_KEYWORDS = [
    "incomplete",
    "vague",
    "confusing",
    "weak",
    "does not engage",
    "too short",
    "generic",
    "repetitive",
    "ends abruptly",
    "slip-up",
    "correction",
]

# slang we WANT to keep / boost (kutigei / coochie etc)
GOOD_SLANG_HOOK_KEYWORDS = [
    "kutigei",
    "coochie gang",
    "coochie",
]


def is_trash_line(text: str, llm_reason: str) -> bool:
    """
    Mark obvious retakes / bloopers / filler as trash so they NEVER go into the funnel,
    BUT do NOT kill a good slang line just because it's spicy.
    """
    t = (text or "").strip().lower()
    r = (llm_reason or "").lower()

    # 0) If it’s clearly one of our "good slang" candidates, never auto-trash just for length
    if any(kw in t for kw in GOOD_SLANG_HOOK_KEYWORDS):
        return False

    # 1) Super short, non-meaningful lines (except a few common words)
    word_count = len(t.split())
    if word_count <= 2 and t not in ("yes", "no", "okay", "ok", "sure"):
        return True

    # 2) Explicit bad / re-take phrases
    for pat in BAD_PHRASE_PATTERNS:
        if re.search(pat, t):
            return True

    # 3) LLM explicitly says it's bad
    for kw in BAD_REASON_KEYWORDS:
        if kw in r:
            return True

    return False


def combine_scores(semantic_score: float, visual_score: float) -> float:
    """
    Favor semantic score (what is said) over visual score (pretty face / stable scene).
    """
    sem = float(semantic_score or 0.0)
    vis = float(visual_score or 0.0)
    return 0.85 * sem + 0.15 * vis


def score_clip_for_funnel(clip: Dict) -> Optional[Dict]:
    """
    Take one raw clip dict with:
      - clip["text"]
      - clip["semantic_score"] or clip["meta"]["semantic_score"]
      - clip["visual_score"] or clip["meta"]["visual_score"]
      - clip["llm_reason"] or clip["meta"]["llm_reason"]
    Return a NEW dict with combined 'score' in meta.
    Return None if we consider this clip trash and don't want it in the funnel.
    """
    text = clip.get("text", "") or ""
    meta = dict(clip.get("meta", {}) or {})
    llm_reason = meta.get("llm_reason") or clip.get("llm_reason") or ""

    semantic_score = float(meta.get("semantic_score", clip.get("semantic_score", 0.0)) or 0.0)
    visual_score = float(meta.get("visual_score", clip.get("visual_score", 0.0)) or 0.0)

    # 1) Hard filter obvious bad takes / retakes / fillers
    if is_trash_line(text, llm_reason):
        return None

    # 2) Combine semantic + visual
    combined = combine_scores(semantic_score, visual_score)

    # 3) Tiny boost for “good slang” hooks so they don't get unfairly punished
    txt_lower = text.lower()
    if any(kw in txt_lower for kw in GOOD_SLANG_HOOK_KEYWORDS):
        combined += 0.08
        if combined > 1.0:
            combined = 1.0

    out = dict(clip)
    out_meta = meta
    out_meta["semantic_score"] = semantic_score
    out_meta["visual_score"] = visual_score
    out_meta["score"] = combined
    out["meta"] = out_meta
    return out


# ------------  MAIN COMPOSER: builds composer + composer_human  ------------ #

def build_funnel_composer(
    clips: List[Dict],
    slots: Dict[str, List[Dict]],
    min_feature_score: float = 0.70,
) -> Tuple[Dict, str]:
    """
    Build the EDITDNA funnel composer:
      - HOOK: best hook clip, strongly prefers 'kutigei/coochie' line if present.
      - FEATURES: best feature/story lines above min_feature_score, after trash filter.
      - CTA: best CTA line.
    Returns (composer_dict, composer_human_str).

    Expected shape (same as your JSON):
      composer = {
          "hook_id": "ASR0000_c1",
          "feature_ids": [...],
          "cta_id": "ASR0022_c1",
          "used_clip_ids": [...],
          "min_score": 0.7
      }
    """

    # Re-score all clips with the new logic (trash filtering + slang boost)
    id_to_scored: Dict[str, Dict] = {}
    for c in clips:
        scored = score_clip_for_funnel(c)
        if scored is not None:
            cid = scored.get("id")
            if cid:
                id_to_scored[cid] = scored

    # Helper to map a slot list to scored clips
    def scored_from_slot(slot_name: str) -> List[Dict]:
        raw_list = slots.get(slot_name, []) or []
        result: List[Dict] = []
        for c in raw_list:
            cid = c.get("id")
            if not cid:
                continue
            scored = id_to_scored.get(cid)
            if not scored:
                continue
            result.append(scored)
        return result

    hooks = scored_from_slot("HOOK")
    features = scored_from_slot("FEATURE")
    proofs = scored_from_slot("PROOF")
    ctas = scored_from_slot("CTA")

    # ---------------- HOOK PICKING ---------------- #

    hook_id: Optional[str] = None
    hook_clip: Optional[Dict] = None

    # 1) Try to find a slang hook (kutigei / coochie) in HOOK slot first
    slang_hooks = [
        c for c in hooks
        if any(kw in (c.get("text", "").lower()) for kw in GOOD_SLANG_HOOK_KEYWORDS)
    ]
    if slang_hooks:
        hook_clip = max(slang_hooks, key=lambda c: c["meta"].get("score", 0.0))
    elif hooks:
        # 2) Otherwise, best-scoring HOOK
        hook_clip = max(hooks, key=lambda c: c["meta"].get("score", 0.0))

    if hook_clip:
        hook_id = hook_clip["id"]

    # ---------------- FEATURE PICKING ---------------- #

    # Merge STORY/FEATURE + PROOF
    all_feat_candidates: List[Dict] = []
    all_feat_candidates.extend(features)
    all_feat_candidates.extend(proofs)

    # Filter by score
    good_feats = [
        c for c in all_feat_candidates
        if c["meta"].get("score", 0.0) >= float(min_feature_score)
    ]

    # Sort by score (descending)
    good_feats.sort(key=lambda c: c["meta"].get("score", 0.0), reverse=True)

    feature_ids = [c["id"] for c in good_feats]

    # ---------------- CTA PICKING ---------------- #

    cta_id: Optional[str] = None
    cta_clip: Optional[Dict] = None
    if ctas:
        cta_clip = max(ctas, key=lambda c: c["meta"].get("score", 0.0))
    if cta_clip:
        cta_id = cta_clip["id"]

    # ---------------- USED CLIP IDS ---------------- #

    used_clip_ids: List[str] = []
    if hook_id:
        used_clip_ids.append(hook_id)
    used_clip_ids.extend([cid for cid in feature_ids if cid not in used_clip_ids])
    if cta_id and cta_id not in used_clip_ids:
        used_clip_ids.append(cta_id)

    composer = {
        "hook_id": hook_id,
        "feature_ids": feature_ids,
        "cta_id": cta_id,
        "used_clip_ids": used_clip_ids,
        "min_score": float(min_feature_score),
    }

    # ---------------- HUMAN-READABLE SUMMARY ---------------- #

    def fmt_clip_line(c: Dict) -> str:
        cid = c.get("id", "?")
        text = c.get("text", "").strip()
        score = c.get("meta", {}).get("score", 0.0)
        return f'  - [{cid}] score={score:.2f} → "{text}"'

    lines: List[str] = []
    lines.append("===== EDITDNA FUNNEL COMPOSER =====\n")

    # HOOK
    if hook_clip:
        hscore = hook_clip["meta"].get("score", 0.0)
        lines.append(f'HOOK ({hook_clip["id"]}, score={hscore:.2f}):')
        lines.append(f'  "{hook_clip.get("text", "").strip()}"\n')
    else:
        lines.append("HOOK: <none>\n")

    # FEATURES
    lines.append("FEATURES (kept):")
    if good_feats:
        for c in good_feats:
            lines.append(fmt_clip_line(c))
    else:
        lines.append("  - <none>")
    lines.append("")

    # CTA
    if cta_clip:
        cscore = cta_clip["meta"].get("score", 0.0)
        lines.append(f'CTA ({cta_clip["id"]}, score={cscore:.2f}):')
        lines.append(f'  "{cta_clip.get("text", "").strip()}"\n')
    else:
        lines.append("CTA: <none>\n")

    # Final timeline (in play order)
    lines.append("FINAL ORDER TIMELINE:")
    order_idx = 1
    if hook_clip:
        lines.append(f'{order_idx}) {hook_clip["id"]} → "{hook_clip.get("text", "").strip()}"')
        order_idx += 1

    for cid in feature_ids:
        c = id_to_scored.get(cid)
        if not c:
            continue
        lines.append(f'{order_idx}) {cid} → "{c.get("text", "").strip()}"')
        order_idx += 1

    if cta_clip:
        lines.append(f'{order_idx}) {cta_clip["id"]} → "{cta_clip.get("text", "").strip()}"')

    lines.append("\n=====================================")
    composer_human = "\n".join(lines)

    return composer, composer_human
# ========= END EDITDNA V3 FUNNEL COMPOSER =========
