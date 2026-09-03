"""D-050C2: the Unified Realization Resolver's explicit, typed rollout mode.

See docs/CUTSELL_DECISIONS.md D-050C1 (shadow authority), D-050C1.5/D-050C1.6
(qualification), and D-050C2 (this module, the controlled cutover) for full
context.

Deliberately a typed 3-state mode -- never an ambiguous boolean or a pair of
booleans -- so "half on" is not a representable state:

    LEGACY        -- the current engine (DeliveryScorer, `_semantic_best_
                     take`, `deterministic_best_take_authority`,
                     ClaimCoverageBestTake, CompositeResolver) behaves
                     byte-for-byte as before D-050C1 ever existed. The
                     Semantic Ledger and Realization Resolver still build
                     (cheap, read-only, additive diagnostics only -- see
                     semantic_ledger.py/realization_resolver.py's own
                     shadow-mode contracts) but nothing about `DraftTimeline.
                     selected`/`discarded` changes.
    SHADOW        -- identical selection behavior to LEGACY. The resolver's
                     shadow output is additionally computed and diagnosed
                     (this is D-050C1's own steady state, and D-050C1.6's
                     qualification baseline) -- still never applied.
    AUTHORITATIVE -- `apply_authoritative_realization_resolution` (see
                     realization_resolver.py) overwrites `DraftTimeline.
                     selected`/`discarded` with the resolver's own decision
                     per semantic idea, and legacy modules' own conclusions
                     become evidence-only (see universal_clean_cut.py's
                     cutover-point comment for the exact list). Default
                     remains LEGACY; this mode is opt-in only, and rolling
                     back requires nothing but resetting one environment
                     variable -- no code revert.

Default is LEGACY. An unrecognized value is treated as LEGACY (fail-safe:
a typo or a stale/garbage env value can never accidentally enable a more
aggressive mode) -- never silently escalated toward AUTHORITATIVE.
"""
from __future__ import annotations

import os
from typing import Mapping

RESOLVER_MODE_LEGACY = "LEGACY"
RESOLVER_MODE_SHADOW = "SHADOW"
RESOLVER_MODE_AUTHORITATIVE = "AUTHORITATIVE"

_VALID_MODES = frozenset({RESOLVER_MODE_LEGACY, RESOLVER_MODE_SHADOW, RESOLVER_MODE_AUTHORITATIVE})

ENV_VAR_NAME = "CUTSELL_UNIFIED_REALIZATION_RESOLVER"


def resolve_resolver_mode(env: Mapping[str, str] | None = None) -> str:
    """Reads `CUTSELL_UNIFIED_REALIZATION_RESOLVER` (case-insensitive),
    defaulting to `RESOLVER_MODE_LEGACY`. `env` defaults to `os.environ`;
    pass an explicit mapping in tests rather than mutating process env.
    Never raises -- an unset, empty, or unrecognized value all fail safe
    to LEGACY."""
    source = os.environ if env is None else env
    raw = str(source.get(ENV_VAR_NAME) or "").strip().upper()
    if raw in _VALID_MODES:
        return raw
    return RESOLVER_MODE_LEGACY
