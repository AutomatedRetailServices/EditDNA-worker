# worker/semantic_visual_pass.py — semantic “tallow” pass
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import List, Optional, Any

# === Env flags & knobs ===
SEMANTICS_ENABLED = os.getenv("SEMANTICS_ENABLED", "0").lower() in ("1", "true", "yes", "on")
EMBEDDER = os.getenv("EMBEDDER", "local").lower()  # local|openai

# retry/dup + stitching
SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))  # cosine
SEM_MERGE_SIM     = float(os.getenv("SEM_MERGE_SIM", "0.80"))      # cosine
MERGE_MAX_CHAIN   = int(os.getenv("MERGE_MAX_CHAIN", "3"))         # max consecutive merges
MAX_GAP_SECONDS   = float(os.getenv("SEM_MAX_GAP_SEC", "2.0"))     # allow small gaps

# filler penalty
SEM_FILLER_LIST     = [w.strip() for w in os.getenv("SEM_FILLER_LIST", "um,uh,like,so,okay").split(",") if w.strip()]
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))

# coherence (not strictly needed for chaining, but useful to expose)
SLOT_W = {
    "HOOK":    float(os.getenv("SEM_W_HOOK", "1.0")),
    "PROBLEM": float(os.getenv("SEM_W_PROBLEM", "1.0")),
    "FEATURE": float(os.getenv("SEM_W_FEATURE", "1.2")),
    "PROOF":   float(os.getenv("SEM_W_PROOF", "1.4")),
    "CTA":     float(os.getenv("SEM_W_CTA", "1.2")),
}
W_FLOW = float(os.getenv("SEM_W_FLOW", "1.0"))

# === Regexes ===
FILLER_RX = re.compile(r"\b(" + "|".join(re.escape(w) for w in SEM_FILLER_LIST) + r")\b", flags=re.I) if SEM_FILLER_LIST else None
RETRY_RX  = re.compile(r"\b(wait|start\s*again|retry|take\s*two|no,\s?let\s?me|sorry|hold\s?on|actually|i mean)\b", re.I)

# ===== Data model =====
@dataclass
class Take:
    text: str
    start: float
    end: float
    score_base: float = 1.0
    slot: Optional[str] = None
    vec: Optional[List[float]] = None
    meta: Optional[dict] = None  # optional payload (e.g., original clip ref)

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ===== Embedding helpers =====
class _Emb:
    _model = None

    @classmethod
    def encode(cls, texts: List[str]) -> List[List[float]]:
        if EMBEDDER == "openai":
            # Only used if SEMANTICS_ENABLED=1 and EMBEDDER=openai
            from openai import OpenAI
            client = OpenAI()
            out = []
            for t in texts:
                r = client.embeddings.create(model="text-embedding-3-small", input=t or "")
                out.append(r.data[0].embedding)
            return out
        else:
            # Local, fast, CPU-friendly
            from sentence_transformers import SentenceTransformer
            if cls._model is None:
                # canonical short name for SentenceTransformers hub
                cls._model = SentenceTransformer("all-MiniLM-L6-v2")
            return cls._model.encode(texts, normalize_embeddings=True).tolist()

def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x, y in zip(a, b))
    da = (sum(x*x for x in a) ** 0.5) or 1e-9
    db = (sum(y*y for y in b) ** 0.5) or 1e-9
    v = num / (da * db)
    return max(-1.0, min(1.0, v))

# ===== Light slot tagger (heuristic) =====
def _tag_slots(texts: List[str]) -> List[str]:
    slots = []
    for tx in texts:
        t = (tx or "").lower()
        if any(k in t for k in ["stop", "wait", "before you", "attention", "did you know"]):
            slots.append("HOOK"); continue
        if any(k in t for k in ["problem", "struggle", "hard", "issue", "pain"]):
            slots.append("PROBLEM"); continue
        if any(k in t for k in ["proof", "testimonial", "results", "review", "customers say", "i use it every"]):
            slots.append("PROOF"); continue
        if any(k in t for k in ["buy", "get", "claim", "download", "book", "start", "link in bio", "shop now", "today"]):
            slots.append("CTA"); continue
        # default
        slots.append("FEATURE")
    return slots

# ===== Scoring & filters =====
def _filler_rate(t: str) -> float:
    if not t:
        return 0.0
    words = re.findall(r"\w+", t.lower())
    if not words:
        return 0.0
    hits = len(FILLER_RX.findall(t)) if FILLER_RX else 0
    return hits / max(1, len(words))

def score_takes(takes: List[Take]) -> List[Take]:
    """Apply filler penalty + compute embeddings + tag slots."""
    if not (SEMANTICS_ENABLED and takes):
        return takes or []
    # fill penalties
    scored = []
    for t in takes:
        fr = _filler_rate(t.text or "")
        penalty = (fr - SEM_FILLER_MAX_RATE) * 2.0 if fr > SEM_FILLER_MAX_RATE else 0.0
        scored.append(Take(
            text=t.text, start=t.start, end=t.end,
            score_base=max(0.0, t.score_base * (1.0 - max(0.0, penalty))),
            slot=t.slot, vec=t.vec, meta=t.meta
        ))
    # embeddings + slots
    vecs  = _Emb.encode([t.text or "" for t in scored])
    slots = _tag_slots([t.text or "" for t in scored])
    for t, v, s in zip(scored, vecs, slots):
        t.vec = v
        t.slot = s
    return scored

def dedupe_retries(takes: List[Take]) -> List[Take]:
    """Remove explicit retries and near-duplicates, keep best-scored version."""
    if not (SEMANTICS_ENABLED and takes):
        return takes or []
    texts = [t.text or "" for t in takes]
    vecs  = _Emb.encode(texts)
    kept: List[Take] = []
    for take, vec in zip(takes, vecs):
        # Drop obvious "restart" segments
        if RETRY_RX.search(take.text or ""):
            continue
        # Keep if not near-duplicate of already kept
        ok = True
        for k in kept:
            if k.vec and _cos(vec, k.vec) >= SEM_DUP_THRESHOLD:
                ok = False
                break
        if ok:
            take.vec = vec
            kept.append(take)
    return kept

def _can_merge(a: Take, b: Take) -> bool:
    if not (a.vec and b.vec):
        return False
    if (b.start - a.end) < 0.0 or (b.start - a.end) > MAX_GAP_SECONDS:
        return False
    return _cos(a.vec, b.vec) >= SEM_MERGE_SIM

def chain_continuity(takes: List[Take]) -> List[List[Take]]:
    """Greedy chain of adjacent takes by semantic continuity with max chain length."""
    if not (SEMANTICS_ENABLED and takes):
        # no semantics → each take stands alone
        return [[t] for t in (takes or [])]
    chains: List[List[Take]] = []
    cur: List[Take] = []
    for t in takes:
        if not cur:
            cur = [t]
            continue
        if len(cur) < MERGE_MAX_CHAIN and _can_merge(cur[-1], t):
            cur.append(t)
        else:
            chains.append(cur)
            cur = [t]
    if cur:
        chains.append(cur)
    return chains

def coherence_score(seq: List[Take]) -> float:
    if not seq:
        return 0.0
    flow, pairs = 0.0, 0
    for i in range(len(seq) - 1):
        if seq[i].vec and seq[i + 1].vec:
            flow += _cos(seq[i].vec, seq[i + 1].vec)
            pairs += 1
    flow = flow / max(1, pairs)
    slot_w = sum(SLOT_W.get((t.slot or "FEATURE"), 1.0) * (t.score_base or 1.0) for t in seq)
    return slot_w + W_FLOW * flow

# ===== Public entry used by jobs.py =====
def continuity_chains(takes: List[Any]) -> List[List[Any]]:
    """
    Flexible adapter:
      - If takes are of shape: {'text': str, 'clip': <Clip>, 'start': float, 'end': float}
        → returns List[List[Clip]] grouped by continuity.
      - If takes are Take objects → returns List[List[Take]].
      - If SEMANTICS_ENABLED=0 → returns [[item], ...]
    """
    if not takes:
        return []

    # If not enabled, behave as no-op
    if not SEMANTICS_ENABLED:
        return [[t["clip"]] for t in takes] if (isinstance(takes[0], dict) and "clip" in takes[0]) else [[t] for t in takes]

    # Normalize to Take list
    norm: List[Take] = []
    if isinstance(takes[0], dict):
        for t in takes:
            # tolerate missing fields
            norm.append(Take(
                text=t.get("text") or "",
                start=float(t.get("start", 0.0)),
                end=float(t.get("end", max(0.01, float(t.get("start", 0.0)) + 0.01))),
                score_base=float(t.get("score", 1.0)),
                slot=t.get("slot"),
                meta={"clip": t.get("clip")}
            ))
    else:
        # assume already Take objects
        norm = [t for t in takes if isinstance(t, Take)]

    # pipeline: score → dedupe → chain
    scored = score_takes(norm)
    filtered = dedupe_retries(scored)
    chains = chain_continuity(filtered)

    # Map back to clips if input had clips
    if isinstance(takes[0], dict) and "clip" in takes[0]:
        out: List[List[Any]] = []
        for ch in chains:
            clips = []
            for t in ch:
                c = (t.meta or {}).get("clip")
                if c is not None:
                    clips.append(c)
            if clips:
                out.append(clips)
        return out

    return chains

__all__ = [
    "Take",
    "score_takes",
    "dedupe_retries",
    "chain_continuity",
    "coherence_score",
    "continuity_chains",
]
