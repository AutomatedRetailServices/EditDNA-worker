from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Iterable

print("🧠 [semantic_visual_pass] Semantic pipeline active.", flush=True)

# ----- Data model -----
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

# ----- Helpers -----
def _as_list(x: Iterable) -> List[Any]:
    """Always return a list, even if x is a single object or a generator."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return [x]

# ----- Safe, simple behavior that won't error -----
def dedup_takes(takes) -> List[Take]:
    # No-op dedup (keeps input order)
    return _as_list(takes)

def can_merge(a: Take, b: Take) -> bool:
    # Disable auto-merge in fallback
    return False

def stitch_chain(takes) -> List[Take]:
    # Identity: keep same takes/ordering
    return _as_list(takes)

# Back-compat for jobs.py: must return a list of chains (list of lists of Take)
def continuity_chains(takes) -> List[List[Take]]:
    return [[t] for t in _as_list(takes)]
