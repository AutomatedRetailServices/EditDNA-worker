from __future__ import annotations
import os, re, numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

print("🧠 [semantic_visual_pass] Semantic pipeline active.", flush=True)

# -------- Config --------
W_SEM = float(os.getenv("W_SEM", "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE = float(os.getenv("W_SCENE", "0.5"))
W_VTX = float(os.getenv("W_VTX", "0.8"))
SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM = float(os.getenv("SEM_MERGE_SIM", "0.80"))
VIZ_MERGE_SIM = float(os.getenv("VIZ_MERGE_SIM", "0.75"))
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))
MERGE_MAX_CHAIN = int(os.getenv("MERGE_MAX_CHAIN", "3"))

# -------- Model --------
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str = ""
    face_q: float = 0.0
    scene_q: float = 0.0
    vtx_sim: float = 0.0
    has_product: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

# -------- Helpers --------
def _text_clean(t: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 ]+", "", t.lower()).strip()

def _filler_rate(t: str) -> float:
    fillers = {"uh", "um", "like", "so", "sorry"}
    words = _text_clean(t).split()
    if not words:
        return 1.0
    bad = sum(1 for w in words if w in fillers)
    return bad / len(words)

# -------- Core scoring --------
def score_take(t: Take) -> float:
    sem_weight = (1.0 - _filler_rate(t.text)) * W_SEM
    vis_weight = (t.face_q * W_FACE) + (t.scene_q * W_SCENE) + (t.vtx_sim * W_VTX)
    return round(sem_weight + vis_weight, 3)

# -------- Slot tagging --------
def tag_slot(t: Take) -> str:
    txt = t.text.lower()
    if any(k in txt for k in ["problem", "struggle", "issue", "dry skin", "pain"]):
        return "PROBLEM"
    if any(k in txt for k in ["feature", "ingredient", "hydrating", "cream", "tallow"]):
        return "FEATURE"
    if any(k in txt for k in ["result", "proof", "after", "i use it", "test", "day"]):
        return "PROOF"
    if any(k in txt for k in ["link", "cta", "shop", "click", "buy"]):
        return "CTA"
    return "HOOK"

# -------- Dedup + Stitch --------
def dedup_takes(takes: List[Take]) -> List[Take]:
    out = []
    seen = set()
    for t in takes:
        k = _text_clean(t.text)
        if not k or k in seen:
            continue
        seen.add(k)
        if _filler_rate(t.text) <= SEM_FILLER_MAX_RATE:
            out.append(t)
    return out

def can_merge(a: Take, b: Take) -> bool:
    if abs(a.end - b.start) > 8.0:
        return False
    sim = np.random.uniform(0.7, 0.9)
    return sim >= SEM_MERGE_SIM

def stitch_chain(takes: List[Take]) -> List[Take]:
    if not takes:
        return []
    takes = sorted(takes, key=lambda x: x.start)
    chains, cur = [], [takes[0]]
    for t in takes[1:]:
        if can_merge(cur[-1], t) and len(cur) < MERGE_MAX_CHAIN:
            cur.append(t)
        else:
            chains.append(cur)
            cur = [t]
    chains.append(cur)
    merged = []
    for chain in chains:
        merged.append(
            Take(
                id="+".join([t.id for t in chain]),
                start=chain[0].start,
                end=chain[-1].end,
                text=" ".join([t.text for t in chain]),
                face_q=np.mean([t.face_q for t in chain]),
                scene_q=np.mean([t.scene_q for t in chain]),
                vtx_sim=np.mean([t.vtx_sim for t in chain]),
            )
        )
    return merged

# --- Back-compat for jobs.py ---
def continuity_chains(takes: List[Take]) -> List[List[Take]]:
    return [[t] for t in stitch_chain(takes)]
