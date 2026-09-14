"""D-097 Priority D -- shared semantic-polarity vocabulary for meaning safety.

Run 34008386434 (Video00) rendered a clause with its meaning INVERTED: the
speaker paused for emphasis after a bare "No", ASR put the particle in its
own speech unit, the strict one-word bridge refused to rejoin it, and a
downstream cleanup deleted it as micro debris. The clause that followed was
then delivered as an affirmation. A polarity particle is never disposable
debris: whichever stage sees it alone must either reattach it to the clause
it negates (AttemptReconstructor/segmentation, the earliest authority) or
keep it (WHEN UNCERTAIN, KEEP) -- never delete it on its own.

ONE vocabulary, shared by `take_segmentation` (rejoin) and
`semantic_fragment_guard` (protection), so the two authorities can never
disagree about what a polarity particle is. Bilingual (Spanish/English)
general vocabulary; nothing here is a Video00 phrase.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ɏ']+")

# Bare polarity carriers: a token that, standing alone, flips or asserts the
# polarity of the clause it attaches to. Contractions are listed both with
# and without the apostrophe because ASR output is inconsistent about them.
POLARITY_PARTICLES = frozenset({
    # Spanish
    "no", "nunca", "jamas", "jamás", "tampoco", "ni", "nada", "nadie",
    # English
    "not", "never", "neither", "nor", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "cant", "can't", "cannot", "wont", "won't", "isnt",
    "isn't", "wasnt", "wasn't", "arent", "aren't", "werent", "weren't",
    "couldnt", "couldn't", "shouldnt", "shouldn't", "wouldnt", "wouldn't",
})

# A leading polarity particle is allowed one preceding/following discourse
# token ("y no", "pero no", "no, no") and still counts as a BARE particle
# unit: the unit carries polarity and nothing else that could stand alone.
_POLARITY_ATTENDANTS = frozenset({
    "y", "pero", "o", "e", "and", "but", "or", "so", "que", "pues", "eh", "um", "uh",
})


def polarity_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _TOKEN_RE.findall(str(text or "")))


def is_bare_polarity_unit(text: str, *, maximum_tokens: int = 3) -> bool:
    """True when the whole unit is a polarity particle plus at most attendant
    discourse tokens (e.g. "No", "no no", "pero no", "and not"). A unit that
    contains any other content word is NOT bare -- it may be a complete short
    answer ("No lo sé.") and is left to normal handling."""
    tokens = polarity_tokens(text)
    if not tokens or len(tokens) > maximum_tokens:
        return False
    if not any(token in POLARITY_PARTICLES for token in tokens):
        return False
    return all(token in POLARITY_PARTICLES or token in _POLARITY_ATTENDANTS for token in tokens)


def carries_polarity(text: str) -> bool:
    """True when the unit contains any polarity particle at all."""
    return any(token in POLARITY_PARTICLES for token in polarity_tokens(text))
