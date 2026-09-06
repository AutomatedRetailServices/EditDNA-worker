"""Install the source-audio boundary completion pass after Flow B draft construction."""
from __future__ import annotations

import sys

from .audio_boundary_completion import repair_audio_confirmed_boundary_completions


def install_audio_boundary_completion() -> None:
    from . import flow_b

    original = flow_b.process_local_sources
    if getattr(original, "_cutsell_audio_boundary_completion", False):
        return

    def process_with_audio_boundary_completion(*args, **kwargs):
        result = original(*args, **kwargs)
        request = args[0] if args else kwargs.get("request")
        local_paths = args[1] if len(args) > 1 else kwargs.get("local_paths")
        asr_provider = kwargs.get("asr_provider")
        if request is None or local_paths is None or asr_provider is None:
            return result
        return repair_audio_confirmed_boundary_completions(
            result,
            local_paths,
            asr_provider,
            language_hint=getattr(request, "language_hint", None),
        )

    process_with_audio_boundary_completion._cutsell_audio_boundary_completion = True
    flow_b.process_local_sources = process_with_audio_boundary_completion

    # If a caller module imported the function before this installer ran, keep that
    # reference synchronized without forcing a circular import.
    universal = sys.modules.get("cutsell_worker.universal_clean_cut")
    if universal is not None:
        universal.process_local_sources = process_with_audio_boundary_completion
