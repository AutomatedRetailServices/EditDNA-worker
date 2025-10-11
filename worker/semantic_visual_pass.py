# overwrite with the full implementation
cat > /workspace/editdna/app/worker/semantic_visual_pass.py <<'PY'
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

print("🧠 [semantic_visual_pass] Semantic pipeline active.", flush=True)

# -------- Config (env overrides safe) --------
W_SEM = float(os.getenv("W_SEM", "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE = float(os.getenv("W_SCENE", "0.5"))
W_PROD = float(os.getenv("W_PROD", "0.0"))  # 0.0 if no product detector yet
W_OCR  = float(os.getenv("W_OCR",  "0.0"))  # 0.0 if no OCR yet
W_VTX  = float(os.getenv("W_VTX",  "0.8"))

SEM_DUP_THRESHOLD = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM     = float(os.getenv("SEM_MERGE_SIM", "0.80"))
VIZ_MERGE_SIM     = float(os.getenv("VIZ_MERGE_SIM", "0.75"))
MERGE_MAX_CHAIN   = int(os.getenv("MERGE_MAX_CHAIN", "3"))
SEM_FILLER_MAX    = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))

SLOT_REQUIRE_PRODUCT = set((os.getenv("SLOT_REQUIRE_PRODUCT","") or "").split(",")) - {""}
SLOT_REQUIRE_OCR_CTA = set((os.getenv("SLOT_REQUIRE_OCR_CTA","") or "").split(",")) - {""}

RETRY_TOKENS = re.compile(
    r"\b(uh|um|wait|hold on|let me start again|start over|sorry|i mean|actually|no no|take two|redo)\b",
    re.I,
)

# -------- Embeddings (SentenceTransformer with TF-IDF fallback) --------
_embedder = None
_use_tfidf = False

def _load_embedder():
    global _embedder, _use_tfidf
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _use_tfidf = False
    except Exception:
        _embedder = "tfidf"
        _use_tfidf = True
    return _embedder

def _emb(texts: List[str]):
    _load_embedder()
    if _use_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(min_df=1).fit(texts)
        X = vec.transform(texts)
        return X, lambda A, B: float(cosine_similarity(A, B)[0,0])
    else:
        import numpy as np
        V = _embedder.encode(texts, normalize_embeddings=True)  # type: ignore
        def _cos(a, b):
            a = a.reshape(1,-1); b = b.reshape(1,-1)
            return float((a*b).sum())
        return V, _cos

def _cos_pair(a, b):
    if hasattr(a, "shape"):
        a = a.reshape(1,-1)
    if hasattr(b, "shape"):
        b = b.reshape(1,-1)
    return float((a*b).sum())

# -------- Data model --------
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str = ""
    slot_hint: Optional[str] = None
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

# -------- Slot classifier --------
SLOT_LABELS = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]
KEYS = {
    "HOOK":    ["imagine","what if","did you know","stop scrolling","quick tip","the secret"],
    "PROBLEM": ["problem","struggle","hard","pain","issue","annoying","waste","dry skin"],
    "FEATURE": ["feature","comes with","includes","made of","ingredient","formula","tallow"],
    "PROOF":   ["results","testimonial","i use","demo","watch","evidence","it works"],
    "CTA":     ["buy","get","claim","use code","link in bio","shop","today","now"],
}

def tag_slot(t: Take, outline_hint: Optional[Dict]=None) -> str:
    txt = (t.text or "").lower()
    for slot in SLOT_LABELS:
        if any(k in txt for k in KEYS[slot]):
            return slot
    if len(txt.split()) > 25:
        return "PROOF"
    if "made of" in txt or "includes" in txt:
        return "FEATURE"
    return "HOOK"

# -------- Retry/filler + dedup --------
def _is_retry_or_noise(text: str) -> bool:
    words = (text or "").split()
    fillers = sum(1 for w in words if w.lower() in {"uh","um","like","so","okay","sorry"})
    rate = fillers / max(1, len(words))
    return bool(RETRY_TOKENS.search(text or "")) or rate > SEM_FILLER_MAX

def dedup_takes(takes: List[Take]) -> List[Take]:
    if not takes:
        return []
    texts = [t.text or "" for t in takes]
    V, _ = _emb(texts)
    kept: List[Take] = []
    for i, t in enumerate(takes):
        if _is_retry_or_noise(t.text):
            t.meta["drop_reason"] = "retry_or_noise"
            continue
        dup = False
        for k in kept:
            j = texts.index(k.text)
            s = _cos_pair(V[i], V[j]) if hasattr(V, "shape") else _
            if s >= SEM_DUP_THRESHOLD:
                dup = True
                break
        if not dup:
            kept.append(t)
        else:
            t.meta["drop_reason"] = "duplicate"
    return kept

# -------- Merge neighbors (semantic + visual continuity) --------
def can_merge(a: Take, b: Take) -> bool:
    V, _ = _emb([a.text or "", b.text or ""])
    s_sem = _cos_pair(V[0], V[1]) if hasattr(V, "shape") else 0.0
    if s_sem < SEM_MERGE_SIM:
        return False
    s_viz = 0.5*(a.vtx_sim + b.vtx_sim) if (a.vtx_sim and b.vtx_sim) else min(a.scene_q, b.scene_q)
    if s_viz < VIZ_MERGE_SIM:
        return False
    if a.meta.get("scene_cut_next") or b.meta.get("scene_cut_prev"):
        return False
    return True

def stitch_chain(takes: List[Take]) -> List[Take]:
    if not takes:
        return []
    takes = sorted(takes, key=lambda x: (x.start, x.end))
    out: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (j+1 < len(takes)) and (len(chain) < MERGE_MAX_CHAIN) and can_merge(takes[j], takes[j+1]):
            chain.append(takes[j+1]); j += 1
        merged = chain[0]
        merged.meta = dict(merged.meta)
        merged.meta["chain_ids"] = [c.id for c in chain]
        merged.end = chain[-1].end
        out.append(merged)
        i = j + 1
    return out

# -------- Scoring --------
def score_take(t: Take, slot: str) -> float:
    if slot in SLOT_REQUIRE_PRODUCT and not t.has_product:
        return -1.0
    if slot in SLOT_REQUIRE_OCR_CTA and t.ocr_hit < 1:
        return -1.0
    sem = 1.0
    if _is_retry_or_noise(t.text):
        sem -= 0.5
    score = (
        W_SEM*sem +
        W_FACE*float(t.face_q) +
        W_SCENE*float(t.scene_q) +
        W_PROD*(1.0 if t.has_product else 0.0) +
        W_OCR*min(1.0, float(t.ocr_hit)) +
        W_VTX*float(t.vtx_sim)
    )
    return float(score)
PY
