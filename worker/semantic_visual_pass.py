# -------- Slot tagging (keywords + heuristics) --------
# Scope says: HOOK → PROBLEM/BENEFITS → FEATURE → PROOF → CTA
# We keep the internal label "PROBLEM" to also mean BENEFITS.
SLOT_LABELS = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]

# Expanded phrase lists to avoid over-tagging everything as HOOK
KEYS = {
    "HOOK": [
        "imagine", "what if", "did you know", "stop scrolling", "listen", "real quick",
        "why not", "guess what", "you need to see", "before you buy", "no one tells you"
    ],
    # PROBLEM/BENEFITS combined:
    "PROBLEM": [
        # pain/problems
        "problem", "struggle", "hard", "annoying", "hate when", "tired of", "issue",
        # benefits/adjectives (common creator talk)
        "great quality", "lightweight", "comfortable", "soft", "smooth", "hydrating",
        "smells", "scent", "long lasting", "elevates any outfit", "holds a lot",
        "fits my phone", "waterproof", "durable", "affordable", "on sale"
    ],
    "FEATURE": [
        "feature", "comes with", "includes", "made of", "material", "fabric",
        "zipper", "pocket", "strap", "adjustable", "dimensions", "size", "capacity",
        "color options", "checker print", "compartment", "inside pocket", "outside pocket"
    ],
    "PROOF": [
        "it works", "i use it", "i use this", "i wear this", "every day", "daily",
        "demo", "watch", "test", "review", "my results", "my skin", "before and after",
        # light numeric proofs
        "for hours", "all day", "12 hours", "90%", "thousands", "5 star", "five star"
    ],
    "CTA": [
        "buy", "get", "grab", "shop", "use code", "link in bio", "check them out",
        "tap the link", "order now", "today", "right now", "don’t miss", "limited", "sale"
    ],
}

# Simple helpers
def _contains_any(txt: str, phrases):
    t = " " + txt.lower() + " "
    return any((" " + p + " ") in t for p in phrases)

def tag_slot(t: Take, outline_hint: Optional[Dict] = None) -> str:
    txt = (t.text or "").strip().lower()

    # 1) Strong signals first
    if _contains_any(txt, KEYS["CTA"]):     return "CTA"
    if _contains_any(txt, KEYS["FEATURE"]): return "FEATURE"
    if _contains_any(txt, KEYS["PROOF"]):   return "PROOF"
    if _contains_any(txt, KEYS["PROBLEM"]): return "PROBLEM"
    if _contains_any(txt, KEYS["HOOK"]):    return "HOOK"

    # 2) Heuristics
    #   - Questions are often hooks.
    if txt.endswith("?") or txt.startswith(("why ", "what ", "how ", "guess ")):
        return "HOOK"

    #   - Enumerations/parts likely FEATURE (zipper, strap, pocket, color)
    feature_terms = {"zipper", "pocket", "strap", "material", "fabric", "size", "dimensions", "color"}
    if any(w in txt for w in feature_terms):
        return "FEATURE"

    #   - First-person usage often PROOF
    if " i use " in f" {txt} " or " i wear " in f" {txt} ":
        return "PROOF"

    #   - Adjective-heavy with value words → BENEFITS (map to PROBLEM slot)
    benefit_terms = {"quality","lightweight","comfortable","soft","smooth","hydrating","scent","smells","elevates"}
    if any(w in txt for w in benefit_terms):
        return "PROBLEM"

    #   - Long descriptive lines → prefer FEATURE/PROOF over HOOK
    words = len(txt.split())
    if words >= 18:
        return "PROOF"
    if words >= 10:
        return "FEATURE"

    # 3) Fallback
    return "HOOK"
