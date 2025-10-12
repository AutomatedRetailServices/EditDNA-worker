cat > /workspace/editdna/app/worker/semantic_visual_pass.py <<'PY'
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable

print("🧠 [semantic_visual_pass] Semantic pipeline active.", flush=True)

# -------- Config (env) --------
W_SEM  = float(os.getenv("W_SEM",  "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE= float(os.getenv("W_SCENE","0.5"))
W_PROD = float(os.getenv("W_PROD", "0.0"))
W_OCR  = float(os.getenv("W_OCR",  "0.0"))
W_VTX  = float(os.getenv("W_VTX",  "0.8"))

SEM_DUP_THRESHOLD   = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM       = float(os.getenv("SEM_MERGE_SIM",     "0.80"))
VIZ_MERGE_SIM       = float(os.getenv("VIZ_MERGE_SIM",     "0.75"))
MERGE_MAX_CHAIN     = int(os.getenv("MERGE_MAX_CHAIN",     "3"))
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE","0.08"))

SLOT_REQUIRE_PRODUCT = set((os.getenv("SLOT_REQUIRE_PRODUCT","") or "").split(",")) - {""}
SLOT_REQUIRE_OCR_CTA = set((os.getenv("SLOT_REQUIRE_OCR_CTA","") or "").split(",")) - {""}

RETRY_TOKENS = re.compile(
    r"\b(uh|um|wait|hold on|let me start again|start over|sorry|i mean|actually|no no|take two|redo)\b",
    re.I,
)

# -------- Embeddings (sentence-transformers w/ TF-IDF fallback) --------
_embedder = None
def _load_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
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
        return X, (lambda A,B: float(cosine_similarity(A, B)[0,0]))
    else:
        import numpy as np
        V = mdl.encode(texts, normalize_embeddings=True)
        return V, (lambda A,B: float((A*B).sum()))

def _cos(a, b):
    # a,b already normalized when using ST; TF-IDF uses provided similarity fn.
    raise NotImplementedError  # not used directly

# -------- Data model --------
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str = ""
    # visual/semantic meta (optional fields — jobs.py fills some)
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0
    slot_hint: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# -------- Utilities --------
def _as_list(x: Iterable) -> List[Any]:
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    try: return list(x)
    except TypeError: return [x]

# -------- Retry / filler detection --------
def is_retry_or_noise(text: str) -> bool:
    words = text.split()
    fillers = len([w for w in words if w.lower() in {"uh","um","like","so","sorry"}])
    rate = fillers / max(1, len(words))
    return bool(RETRY_TOKENS.search(text)) or rate > SEM_FILLER_MAX_RATE

# -------- Slot tagging (keywords + simple heuristics) --------
SLOT_LABELS = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]
KEYS = {
    "HOOK":    ["imagine","what if","did you know","stop scrolling","quick tip","the secret"],
    "PROBLEM": ["problem","struggle","hard","pain","issue","annoying","dry skin","waste"],
    "FEATURE": ["feature","comes with","includes","made of","ingredient","formula"],
    "PROOF":   ["results","testimonials","it works","i use it","demo","watch","evidence"],
    "CTA":     ["buy","get","claim","use code","link in bio","shop","today","now"],
}

def tag_slot(t: Take, outline_hint: Optional[Dict]=None) -> str:
    txt = (t.text or "").lower()
    for slot in SLOT_LABELS:
        if any(k in txt for k in KEYS[slot]):
            return slot
    # fallback heuristics
    if len(txt.split()) > 25:
        return "PROOF"
    if "made of" in txt or "includes" in txt:
        return "FEATURE"
    return "HOOK"

# -------- Dedup (keep best unique meanings; drop retries) --------
def dedup_takes(takes: List[Take]) -> List[Take]:
    if not takes: return []
    texts = [t.text for t in takes]
    V, simfun = _emb(texts)
    kept: List[Take] = []
    for i, t in enumerate(takes):
        if is_retry_or_noise(t.text):
            t.meta["drop_reason"] = "retry_or_noise"
            continue
        is_dup = False
        for k in kept:
            j = texts.index(k.text)
            s = simfun(V[i], V[j])
            if s >= SEM_DUP_THRESHOLD:
                is_dup = True
                t.meta["drop_reason"] = "duplicate"
                break
        if not is_dup:
            kept.append(t)
    return kept

# -------- Merge continuity checks --------
def can_merge(a: Take, b: Take) -> bool:
    V, simfun = _emb([a.text, b.text])
    s_sem = simfun(V[0], V[1])
    if s_sem < SEM_MERGE_SIM:
        return False
    # visual continuity proxy
    s_viz = 0.5*(a.vtx_sim + b.vtx_sim) if (a.vtx_sim and b.vtx_sim) else min(a.scene_q, b.scene_q)
    if s_viz < VIZ_MERGE_SIM:
        return False
    if a.meta.get("scene_cut_next") or b.meta.get("scene_cut_prev"):
        return False
    return True

def stitch_chain(takes: List[Take]) -> List[Take]:
    takes = sorted(_as_list(takes), key=lambda x: (x.start, x.end))
    out: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (j+1 < len(takes)) and (len(chain) < MERGE_MAX_CHAIN) and can_merge(takes[j], takes[j+1]):
            chain.append(takes[j+1]); j += 1
        merged = chain[0]
        merged.meta["chain_ids"] = [c.id for c in chain]
        merged.end = chain[-1].end
        out.append(merged)
        i = j + 1
    return out

# Back-compat for older jobs.py that expects list-of-lists
def continuity_chains(takes: List[Take]) -> List[List[Take]]:
    return [[t] for t in stitch_chain(takes)]

# -------- Scoring --------
def score_take(t: Take, slot: str) -> float:
    # hard constraints per slot
    if slot in SLOT_REQUIRE_PRODUCT and not t.has_product:
        return -1.0
    if slot in SLOT_REQUIRE_OCR_CTA and t.ocr_hit < 1:
        return -1.0

    # semantic base (1.0 minus penalty if retry/noise)
    sem_score = 1.0
    if is_retry_or_noise(t.text):
        sem_score -= 0.5

    score = (
        W_SEM  * sem_score +
        W_FACE * float(t.face_q) +
        W_SCENE* float(t.scene_q) +
        W_PROD * (1.0 if t.has_product else 0.0) +
        W_OCR  * min(1.0, float(t.ocr_hit)) +
        W_VTX  * float(t.vtx_sim)
    )
    return float(score)
PY
