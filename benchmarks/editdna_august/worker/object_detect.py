import os
from typing import List

# Cheap "product detection" stub:
# For now we mark clauses as has_product=True if their text contains
# any of the brand/product keywords. Later you can swap this for YOLO.

DEFAULT_PRODUCT_KEYWORDS = [
    "bag",
    "bum bag",
    "tote",
    "probiotic",
    "gummy",
    "gummies",
    "supplement",
    "serum",
    "cream",
    "tallow",
]

def _load_keywords():
    raw = os.getenv("EDITDNA_PRODUCT_KEYWORDS", "")
    if not raw.strip():
        return DEFAULT_PRODUCT_KEYWORDS
    extra = [w.strip().lower() for w in raw.split(",") if w.strip()]
    return DEFAULT_PRODUCT_KEYWORDS + extra


PRODUCT_KEYWORDS = _load_keywords()


def enrich_clauses_with_product_flags(clauses: List["Clause"]) -> None:
    """
    Mutates each Clause:
      - clause.has_product: bool
    """
    for c in clauses:
        text = (c.text or "").lower()
        has_prod = any(k in text for k in PRODUCT_KEYWORDS)
        setattr(c, "has_product", bool(has_prod))
