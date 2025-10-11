# worker/semantic_visual_pass.py
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ---- Config from env ----
W_SEM = float(os.getenv("W_SEM", "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE = float(os.getenv("W_SCENE", "0.5"))
W_VTX  = float(os.getenv("W_VTX",  "0.8"))

SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM     = float(os.getenv("SEM_MERGE_SIM", "0.80"))
VIZ_MERGE_SIM     = float(os.getenv("VIZ_MERGE_SIM", "0.75"))
MERGE_MAX_CHAIN   = int(os.getenv("MERGE_MAX_CHAIN", "3"))
SEM_FILLER_MAX    = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))

SLOT_REQUIRE_PRODUCT = {s.strip() for s in (os.getenv("SLOT_REQUIRE_PRODUCT","").split(",")) if s.strip()}
SLOT_REQUIRE_OCR_CTA = {s.strip() for s in (os.getenv("SLOT_REQUIRE_OCR_CTA","").split(",")) if s.strip()}

RETRY_TOKENS = re.compile(
    r"\b(uh|um|wait|hold on|let me start again|start over|sorry|i mean|actually|no no|take two|redo)\b",
    re.I,
)

# ---- Embeddings (local model with TF-IDF fallback) ----
_embedder = None
def _load_embedder():
    global _embedder
    if _embedder is not None: return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        _embedder = "tfidf"
    return _embedder

def _emb(texts: List[str]):
    mdl = _load_embedder()
    if mdl == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(min_df=1).fit(texts)
        X = vec.transform(texts)
        return X, lambda A,B: float(cosine_similarity(A, B)[0,0])
    else:
        import numpy as np
        V = mdl.encode(texts, normalize_embeddings=True)
        return V, lambda A,B: float((A*B).sum())

def _cos_pair(a, b) -> float:
    try:
        import numpy as np
        if hasattr(a, "ndim"): a = a.reshape(1,-1)
        if hasattr(b, "ndim"): b = b.reshape(1,-1)
        return float((a*b).sum())
    except Exception:
        return 0.0

# ---- Data model ----
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    slot_hint: Optional[str] = None

    # visual/quality inputs filled by vision sampler or upstream
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0

    meta: Dict = field(default_factory=dict)

# ---- Slot rules ----
SLOT_LABELS = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]
KEYS = {
    "HOOK":    ["imagine", "what if", "did you know", "stop scrolling", "quick tip", "the secret"],
    "PROBLEM": ["problem", "struggle", "hard", "pain", "issue", "annoying"],
    "FEATURE": ["feature", "comes with", "includes", "made of", "ingredient", "formula"],
    "PROOF":   ["results", "testimonial", "i use it", "demo", "watch", "evidence"],
    "CTA":     ["buy", "get", "claim", "use code", "link in bio", "shop", "today", "now"]
}

def tag_slot(t: Take) -> str:
    txt = (t.text or "").lower()
    for slot in SLOT_LABELS:
        if any(k in txt for k in KEYS[slot]):
            return slot
    # backstops
    if len(txt.split()) > 25: return "PROOF"
    if "includes" in txt or "made of" in txt: return "FEATURE"
    return "HOOK"

# ---- Filters ----
def _is_retry_or_noise(text: str) -> bool:
    words = text.split()
    fillers = sum(1 for w in words if w.lower() in {"uh","um","like","so","okay","sorry"})
    rate = fillers / max(1,len(words))
    return bool(RETRY_TOKENS.search(text)) or (rate > SEM_FILLER_MAX)

def dedup_takes(takes: List[Take]) -> List[Take]:
    if not takes: return []
    texts = [t.text for t in takes]
    V, _ = _emb(texts)
    kept: List[Take] = []
    for i, t in enumerate(takes):
        if _is_retry_or_noise(t.text):
            t.meta["drop_reason"] = "retry_or_noise"; continue
        duplicate = False
        for k in kept:
            j = texts.index(k.text)
            s = _cos_pair(V[i], V[j])
            if s >= SEM_DUP_THRESHOLD:
                duplicate = True; break
        if not duplicate: kept.append(t)
        else: t.meta["drop_reason"] = "duplicate"
    return kept

# ---- Continuity ----
def can_merge(a: Take, b: Take) -> bool:
    V, _ = _emb([a.text, b.text])
    s_sem = _cos_pair(V[0], V[1])
    if s_sem < SEM_MERGE_SIM: return False
    s_viz = 0.5*(a.vtx_sim + b.vtx_sim) if (a.vtx_sim and b.vtx_sim) else min(a.scene_q, b.scene_q)
    if s_viz < VIZ_MERGE_SIM: return False
    if a.meta.get("scene_cut_next") or b.meta.get("scene_cut_prev"): return False
    return True

def stitch_chain(takes: List[Take]) -> List[Take]:
    if not takes: return []
    takes = sorted(takes, key=lambda x: (x.start, x.end))
    out: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (j+1 < len(takes)) and (len(chain) < MERGE_MAX_CHAIN) and can_merge(takes[j], takes[j+1]):
            chain.append(takes[j+1]); j += 1
        merged = chain[0]
        if len(chain) > 1:
            merged.meta["chain_ids"] = [c.id for c in chain]
            merged.end = chain[-1].end
        out.append(merged)
        i = j + 1
    return out

# ---- Scoring ----
def score_take(t: Take, slot: str) -> float:
    if slot in SLOT_REQUIRE_PRODUCT and not t.has_product: return -1.0
    if slot in SLOT_REQUIRE_OCR_CTA and t.ocr_hit < 1:    return -1.0
    sem = 1.0 - (0.5 if _is_retry_or_noise(t.text) else 0.0)
    return float(
        W_SEM*sem +
        W_FACE*float(t.face_q) +
        W_SCENE*float(t.scene_q) +
        W_VTX*float(t.vtx_sim)
    )

