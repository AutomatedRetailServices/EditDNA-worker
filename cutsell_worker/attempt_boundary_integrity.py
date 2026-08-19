"""Refine reconstructed-attempt suffix boundaries without weakening bad-tail isolation.

A short incomplete suffix after a complete sentence is usually worth isolating so Hybrid
can remove only a broken final self-correction.  One important exception is a fragment
that *starts* with an explicit grammatical continuation (``because ...``, ``and ...``,
``porque ...``): that speech is structurally dependent on what came before and must stay
inside the same delivery attempt.  This keeps the new bad-tail protection conservative.
"""
from __future__ import annotations

_CONTINUATION_PREFIXES = frozenset({
    # English
    "and", "because", "but", "if", "or", "so", "that", "when", "where", "which",
    "while", "who", "whose", "with", "without",
    # Spanish
    "aunque", "como", "cuando", "donde", "mientras", "o", "pero", "porque", "que",
    "si", "y",
})


def install_attempt_boundary_integrity() -> None:
    from . import attempt_reconstruction as reconstruction

    original = reconstruction._short_incomplete_suffix
    if getattr(original, "_cutsell_continuation_prefix_guard", False):
        return

    def short_incomplete_suffix_with_continuation_guard(left, right):
        tokens = reconstruction._tokens(right.text)
        if tokens and tokens[0] in _CONTINUATION_PREFIXES:
            return False
        return original(left, right)

    short_incomplete_suffix_with_continuation_guard._cutsell_continuation_prefix_guard = True
    reconstruction._short_incomplete_suffix = short_incomplete_suffix_with_continuation_guard
