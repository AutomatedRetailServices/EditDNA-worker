from typing import List

# Cheap heuristic "OCR" for now:
# We just look at the transcript text and mark if it has CTA-ish phrases.
# Later, we can swap this for real OCR on frames without changing pipeline.

CTA_KEYWORDS = [
    "click the link",
    "link in bio",
    "get yours today",
    "shop now",
    "grab yours",
    "buy now",
    "check them out",
    "i left it for you down below",
    "use my code",
]


def enrich_clauses_with_ocr(clauses: List["Clause"]) -> None:
    """
    Mutates each Clause:
      - clause.ocr_hit: int (0 or 1 for now)
    """
    for c in clauses:
        text = (c.text or "").lower()
        hit = any(k in text for k in CTA_KEYWORDS)
        setattr(c, "ocr_hit", 1 if hit else 0)
