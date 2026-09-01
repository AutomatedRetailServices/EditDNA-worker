"""Refine reconstructed-attempt suffix boundaries without weakening bad-tail isolation.

A short incomplete suffix after a complete sentence is normally isolated so Hybrid can
remove only a broken final self-correction.  Two related cases must remain distinct:

* if more speech follows, an incomplete connector-led fragment (``because the ...``)
  should start the next attempt so it can merge forward into its continuation;
* if that connector-led fragment is the *terminal* speech in the source, there is no
  forward continuation to recover, so it remains attached to the preceding delivery and
  correctly makes that delivery incomplete.

This post-pass changes only that terminal ambiguity.  Hard session/reset/restart
boundaries and non-connector bad tails remain untouched.
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


def _terminal_connector_fragment(take, token_fn) -> bool:
    tokens = token_fn(take.text)
    return bool(tokens and not take.complete_idea and tokens[0] in _CONTINUATION_PREFIXES)


def install_attempt_boundary_integrity() -> None:
    from . import attempt_reconstruction as reconstruction

    original = reconstruction.reconstruct_delivery_attempts
    if getattr(original, "_cutsell_terminal_connector_guard", False):
        return

    def reconstruct_with_terminal_connector_guard(
        takes,
        context,
        *,
        max_continuation_gap_sec=1.20,
    ):
        take_tuple = tuple(takes)
        attempts, diagnostics = original(
            take_tuple,
            context,
            max_continuation_gap_sec=max_continuation_gap_sec,
        )
        if len(attempts) < 2 or not take_tuple:
            return attempts, diagnostics

        terminal = take_tuple[-1]
        if not _terminal_connector_fragment(terminal, reconstruction._tokens):
            return attempts, diagnostics

        boundaries = list(diagnostics.get("boundaries") or ())
        matching_index = None
        for index in range(len(boundaries) - 1, -1, -1):
            boundary = boundaries[index]
            if (
                boundary.get("before_clip_id") == terminal.clip_id
                and boundary.get("reason") == "short_incomplete_suffix"
            ):
                matching_index = index
                break
        if matching_index is None:
            return attempts, diagnostics

        # The exact boundary we are removing was created only by the short-incomplete
        # suffix rule.  Stronger source/session/reset/restart boundaries therefore retain
        # precedence and can never be erased here.
        merged_tail = reconstruction._merge_attempt((attempts[-2], attempts[-1]))
        repaired_attempts = tuple((*attempts[:-2], merged_tail))
        del boundaries[matching_index]

        repaired_diagnostics = dict(diagnostics)
        repaired_diagnostics["attempt_count"] = len(repaired_attempts)
        repaired_diagnostics["merged_fragment_count"] = max(0, len(take_tuple) - len(repaired_attempts))
        repaired_diagnostics["boundaries"] = boundaries

        attempt_rows = list(diagnostics.get("attempts") or ())
        if len(attempt_rows) >= 2:
            left_row = dict(attempt_rows[-2])
            right_row = dict(attempt_rows[-1])
            member_ids = list(left_row.get("member_clip_ids") or ()) + list(right_row.get("member_clip_ids") or ())
            repaired_row = {
                "clip_id": merged_tail.clip_id,
                "source_asset_id": merged_tail.source_asset_id,
                "start": round(merged_tail.start, 3),
                "end": round(merged_tail.end, 3),
                "duration_sec": round(merged_tail.duration_sec, 3),
                "member_clip_ids": member_ids,
                "member_count": len(member_ids),
                "complete_idea": merged_tail.complete_idea,
            }
            repaired_diagnostics["attempts"] = [*attempt_rows[:-2], repaired_row]
        return repaired_attempts, repaired_diagnostics

    reconstruct_with_terminal_connector_guard._cutsell_terminal_connector_guard = True
    reconstruction.reconstruct_delivery_attempts = reconstruct_with_terminal_connector_guard
