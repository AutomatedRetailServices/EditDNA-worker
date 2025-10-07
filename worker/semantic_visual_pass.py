# worker/semantic_visual_pass.py
from __future__ import annotations
import os, math, re
from dataclasses import dataclass
from typing import List, Optional

EMBEDDER = os.getenv("EMBEDDER", "local").lower()  # local|openai
SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_FILLER_LIST = [w.strip() for w in os.getenv("SEM_FILLER_LIST", "um,uh,like,so,okay").split(",") if w.strip()]
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))
SEM_MERGE_SIM = float(os.getenv("SEM_MERGE_SIM", "0.80"))
MERGE_MAX_CHAIN = int(os.getenv("MERGE_MAX_CHAIN", "3"))

SLOT_W = {
    "HOOK":   float(os.getenv("SEM_W_HOOK", "1.0")),
    "PROBLEM":float(os.getenv("SEM_W_PROBLEM", "1.0")),
    "FEATURE":float(os.getenv("SEM_W_FEATURE", "1.2")),
    "PROOF":  float(os.getenv("SEM_W_PROOF", "1.4")),
    "CTA":    float(os.getenv("SEM_W_CTA", "1.2")),
}
W_FLOW = float(os.getenv("SEM_W_FLOW", "1.0"))

class Emb:
    _model = None
    @classmethod
    def encode(cls, texts: List[str]) -> List[List[float]]:
        if EMBEDDER == "openai":
            from openai import OpenAI
            client = OpenAI()
            out = []
            for t in texts:
                r = client.embeddings.create(model="text-embedding-3-small", input=t or "")
                out.append(r.data[0].embedding)
            return out
        else:
            from sentence_transformers import SentenceTransformer
            if cls._model is None:
                cls._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return cls._model.encode(texts, normalize_embeddings=True).tolist()

def cos(a, b):
    num = sum(x*y for x,y in zip(a,b))
    da = (sum(x*x for x in a) ** 0.5) or 1e-9
    db = (sum(y*y for y in b) ** 0.5) or 1e-9
    return max(-1.0, min(1.0, num/(da*db)))

FILLER_RX = re.compile(r"\b(" + "|".join(re.escape(w) for w in SEM_FILLER_LIST) + r")\b", flags=re.I) if SEM_FILLER_LIST else None
RETRY_RX  = re.compile(r"\b(wait|start\s*again|retry|take\s*two|no,\s?let\s?me|sorry|hold\s?on|actually|I mean)\b", re.I)

@dataclass
class Take:
    text: str
    start: float
    end: float
    score_base: float = 1.0
    slot: Optional[str] = None
    vec: Optional[List[float]] = None
    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

def filler_rate(t: str) -> float:
    if not t: return 0.0
    words = re.findall(r"\w+", t.lower())
    if not words: return 0.0
    hits = len(FILLER_RX.findall(t)) if FILLER_RX else 0
    return hits / max(1, len(words))

def tag_slot(texts: List[str]) -> List[str]:
    slots = []
    for tx in texts:
        t = (tx or "").lower()
        if any(k in t for k in ["stop", "wait", "before you", "attention", "did you know"]):
            slots.append("HOOK"); continue
        if any(k in t for k in ["problem", "struggle", "hard", "issue", "pain"]):
            slots.append("PROBLEM"); continue
        if any(k in t for k in ["feature", "how it works", "includes", "works by", "made with"]):
            slots.append("FEATURE"); continue
        if any(k in t for k in ["proof", "testimonial", "results", "review", "customers say", "i use it every"]):
            slots.append("PROOF"); continue
        if any(k in t for k in ["buy", "get", "claim", "download", "book", "start", "link in bio", "shop now", "today"]):
            slots.append("CTA"); continue
        slots.append("FEATURE")
    return slots

def dedupe_retries(takes: List[Take]) -> List[Take]:
    if not takes: return []
    texts = [t.text or "" for t in takes]
    vecs = Emb.encode(texts)
    kept: List[Take] = []
    for take, vec in zip(takes, vecs):
        if RETRY_RX.search(take.text or ""):
            continue
        if all((k.vec is None) or (cos(vec, k.vec) < SEM_DUP_THRESHOLD) for k in kept):
            take.vec = vec
            kept.append(take)
    return kept

def can_merge(a: Take, b: Take) -> bool:
    if not a.vec or not b.vec:
        return False
    return (cos(a.vec, b.vec) >= SEM_MERGE_SIM) and (0.0 <= (b.start - a.end) <= 2.0)

def coherence_score(seq: List[Take]) -> float:
    if not seq: return 0.0
    flow, pairs = 0.0, 0
    for i in range(len(seq)-1):
        if seq[i].vec and seq[i+1].vec:
            flow += cos(seq[i].vec, seq[i+1].vec); pairs += 1
    flow = (flow / max(1, pairs))
    slot_w = sum(SLOT_W.get(t.slot or "FEATURE", 1.0) * t.score_base for t in seq)
    return slot_w + W_FLOW * flow

def score_takes(takes: List[Take]) -> List[Take]:
    if not takes: return []
    scored = []
    for t in takes:
        fr = filler_rate(t.text or "")
        penalty = max(0.0, (fr - SEM_FILLER_MAX_RATE) * 2.0) if fr > SEM_FILLER_MAX_RATE else 0.0
        scored.append(Take(text=t.text, start=t.start, end=t.end, score_base=max(0.0, 1.0 - penalty)))
    vecs = Emb.encode([t.text or "" for t in scored])
    slots = tag_slot([t.text or "" for t in scored])
    for t, v, s in zip(scored, vecs, slots):
        t.vec = v; t.slot = s
    return scored

def chain_continuity(takes: List[Take]) -> List[List[Take]]:
    if not takes: return []
    chains: List[List[Take]] = []
    cur: List[Take] = []
    for t in takes:
        if not cur:
            cur = [t]; continue
        if can_merge(cur[-1], t) and len(cur) < MERGE_MAX_CHAIN:
            cur.append(t)
        else:
            chains.append(cur); cur = [t]
    if cur: chains.append(cur)
    return chains
    
