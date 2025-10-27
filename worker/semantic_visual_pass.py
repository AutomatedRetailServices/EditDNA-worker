## worker/semantic_visual_pass.py
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterable

print("🧠 [semantic_visual_pass] Semantic pipeline active.", flush=True)

# =========================
# Config (all env-tunable)
# =========================
W_SEM   = float(os.getenv("W_SEM",  "1.2"))
W_FACE  = float(os.getenv("W_FACE", "0.8"))
W_SCENE = float(os.getenv("W_SCENE","0.5"))
W_PROD  = float(os.getenv("W_PROD", "0.0"))
W_OCR   = float(os.getenv("W_OCR",  "0.0"))
W_VTX   = float(os.getenv("W_VTX",  "0.8"))

SEM_DUP_THRESHOLD    = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM        = float(os.getenv("SEM_MERGE_SIM",     "0.80"))
VIZ_MERGE_SIM        = float(os.getenv("VIZ_MERGE_SIM",     "0.75"))
MERGE_MAX_CHAIN      = int(os.getenv("MERGE_MAX_CHAIN",     "3"))
SEM_FILLER_MAX_RATE  = float(os.getenv("SEM_FILLER_MAX_RATE","0.08"))

SLOT_REQUIRE_PRODUCT = set((os.getenv("SLOT_REQUIRE_PRODUCT","") or "").split(",")) - {""}
SLOT_REQUIRE_OCR_CTA = set((os.getenv("SLOT_REQUIRE_OCR_CTA","") or "").split(",")) - {""}

# phrases indicating retries / restarts
RETRY_TOKENS = re.compile(
    r"\b(uh|um|wait|hold on|let me start again|start over|sorry|i mean|actually|no no|take two|redo|restart)\b",
    re.I,
)

# ===================================
# Embeddings (ST w/ TF-IDF fallback)
# ===================================
_embedder = None
def _load_embedder():
    """
    Try sentence-transformers (CPU friendly). Fall back to TF-IDF if unavailable.
    """
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
    """
    Returns (Vectors, sim_fn). sim_fn(A,B) -> cosine similarity in [~0..1+].
    For ST, vectors are L2-normalized; for TF-IDF we use sklearn cosine.
    """
    mdl = _load_embedder()
    if mdl == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(min_df=1).fit(texts)
        X = vec.transform(texts)
        return X, (lambda A, B: float(cosine_similarity(A, B)[0, 0]))
    else:
        import numpy as np
        V = mdl.encode(texts, normalize_embeddings=True)
        # cosine for normalized vectors reduces to dot product
        return V, (lambda A, B: float((A * B).sum()) if hasattr(A, "ndim") else float((A @ B)))

# =============
# Data model
# =============
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str = ""
    # visual/semantic meta (optional — upstream may fill)
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0
    slot_hint: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# =================
# Small utilities
# =================
def _as_list(x: Iterable) -> List[Any]:
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    try: return list(x)
    except TypeError: return [x]

# ==========================
# Retry / filler detection
# ==========================
def is_retry_or_noise(text: str) -> bool:
    words = (text or "").split()
    fillers = len([w for w in words if w.lower() in {"uh","um","like","so","sorry","okay","ok"}])
    rate = fillers / max(1, len(words))
    return bool(RETRY_TOKENS.search(text or "")) or rate > SEM_FILLER_MAX_RATE

# =========================================
# Slot tagging (HOOK → PROBLEM/BENEFITS → FEATURE → PROOF → CTA)
# =========================================
SLOT_LABELS = ["HOOK", "PROBLEM", "FEATURE", "PROOF", "CTA"]

# Expanded phrases to improve recall and avoid over-tagging HOOK
KEYS = {
    "HOOK": [
        "imagine","what if","did you know","stop scrolling","listen","real quick",
        "why not","guess what","you need to see","before you buy","no one tells you"
    ],
    # PROBLEM includes BENEFITS per scope (“PROBLEM/BENEFITS” unified)
    "PROBLEM": [
        # pain/problems
        "problem","struggle","hard","annoying","hate when","tired of","issue",
        # benefits/value phrasing common in creator talk
        "great quality","lightweight","comfortable","soft","smooth","hydrating",
        "smells","scent","long lasting","elevates any outfit","holds a lot",
        "fits my phone","waterproof","durable","affordable","on sale","good price"
    ],
    "FEATURE": [
        "feature","comes with","includes","made of","material","fabric",
        "zipper","pocket","strap","adjustable","dimensions","size","capacity",
        "color options","checker print","compartment","inside pocket","outside pocket",
        "specs","details"
    ],
    "PROOF": [
        "it works","i use it","i use this","i wear this","every day","daily",
        "demo","watch","test","review","my results","before and after",
        "for hours","all day","12 hours","90%","thousands","5 star","five star"
    ],
    "CTA": [
        "buy","get","grab","shop","use code","link in bio","check them out",
        "tap the link","order now","today","right now","don’t miss","limited","sale",
        "add to cart","claim","download","book","start"
    ],
}

def _contains_any(txt: str, phrases) -> bool:
    t = f" {txt.lower()} "
    return any(f" {p} " in t for p in phrases)

def tag_slot(t: Take, outline_hint: Optional[Dict] = None) -> str:
    txt = (t.text or "").strip().lower()

    # 1) Strong lexical signals first
    if _contains_any(txt, KEYS["CTA"]):     return "CTA"
    if _contains_any(txt, KEYS["FEATURE"]): return "FEATURE"
    if _contains_any(txt, KEYS["PROOF"]):   return "PROOF"
    if _contains_any(txt, KEYS["PROBLEM"]): return "PROBLEM"
    if _contains_any(txt, KEYS["HOOK"]):    return "HOOK"

    # 2) Heuristics
    # Questions → often hooks
    if txt.endswith("?") or txt.startswith(("why ", "what ", "how ", "guess ")):
        return "HOOK"

    # Enumerations/parts → likely feature
    feature_terms = {"zipper","pocket","strap","material","fabric","size","dimensions","color","compartment"}
    if any(w in txt for w in feature_terms):
        return "FEATURE"

    # First-person usage → proof
    if " i use " in f" {txt} " or " i wear " in f" {txt} " or " i love " in f" {txt} ":
        return "PROOF"

    # Adjective/value heavy → benefits (map to PROBLEM)
    benefit_terms = {"quality","lightweight","comfortable","soft","smooth","hydrating","scent","smells","elevates"}
    if any(w in txt for w in benefit_terms):
        return "PROBLEM"

    # Long descriptive lines → prefer FEATURE/PROOF over HOOK
    words = len(txt.split())
    if words >= 18:
        return "PROOF"
    if words >= 10:
        return "FEATURE"

    # 3) Fallback
    return "HOOK"

# ===================
# Dedup / Retry drop
# ===================
def dedup_takes(takes: List[Take]) -> List[Take]:
    if not takes: return []
    texts = [t.text or "" for t in takes]
    V, simfun = _emb(texts)
    kept: List[Take] = []
    kept_idx: List[int] = []
    for i, t in enumerate(takes):
        if is_retry_or_noise(t.text or ""):
            t.meta["drop_reason"] = "retry_or_noise"
            continue
        is_dup = False
        for j, k in zip(kept_idx, kept):
            s = simfun(V[i], V[j])
            if s >= SEM_DUP_THRESHOLD:
                is_dup = True
                t.meta["drop_reason"] = "duplicate"
                break
        if not is_dup:
            kept.append(t)
            kept_idx.append(i)
    return kept

# ============================
# Continuity / Smart stitching
# ============================
def can_merge(a: Take, b: Take) -> bool:
    # Semantic continuity
    V, simfun = _emb([a.text or "", b.text or ""])
    s_sem = simfun(V[0], V[1])
    if s_sem < SEM_MERGE_SIM:
        return False
    # Visual continuity proxy (use vtx_sim if present, else scene_q)
    s_viz = 0.5 * (float(a.vtx_sim) + float(b.vtx_sim)) if (a.vtx_sim and b.vtx_sim) else min(float(a.scene_q), float(b.scene_q))
    if s_viz < VIZ_MERGE_SIM:
        return False
    # Avoid if a scene cut flag was set upstream
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
        while (j + 1 < len(takes)) and (len(chain) < MERGE_MAX_CHAIN) and can_merge(takes[j], takes[j + 1]):
            chain.append(takes[j + 1])
            j += 1
        merged = chain[0]
        merged.meta["chain_ids"] = [c.id for c in chain]
        merged.end = chain[-1].end
        out.append(merged)
        i = j + 1
    return out

# Back-compat helper (old code expects list-of-lists)
def continuity_chains(takes: List[Take]) -> List[List[Take]]:
    return [[t] for t in stitch_chain(takes)]

# ===========
# Scoring
# ===========
def score_take(t: Take, slot: str) -> float:
    # hard constraints per slot
    if slot in SLOT_REQUIRE_PRODUCT and not t.has_product:
        return -1.0
    if slot in SLOT_REQUIRE_OCR_CTA and int(t.ocr_hit or 0) < 1:
        return -1.0

    # semantic base score (light penalty for retry/noise)
    sem_score = 1.0
    if is_retry_or_noise(t.text or ""):
        sem_score -= 0.5

    score = (
        W_SEM   * float(sem_score) +
        W_FACE  * float(t.face_q) +
        W_SCENE * float(t.scene_q) +
        W_PROD  * (1.0 if t.has_product else 0.0) +
        W_OCR   * min(1.0, float(t.ocr_hit or 0)) +
        W_VTX   * float(t.vtx_sim or 0.0)
    )
    t.meta["score"] = float(score)
    t.meta["slot"]  = slot
    return float(score)
