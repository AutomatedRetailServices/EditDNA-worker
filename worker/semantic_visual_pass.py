from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

print("🧠 [semantic_visual_pass] Semantic pipeline active.", flush=True)

# ---------- thresholds (env or defaults) ----------
SEM_DUP_THRESHOLD   = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM       = float(os.getenv("SEM_MERGE_SIM", "0.80"))
VIZ_MERGE_SIM       = float(os.getenv("VIZ_MERGE_SIM", "0.75"))
MERGE_MAX_CHAIN     = int(os.getenv("MERGE_MAX_CHAIN", "3"))
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))

# ---------- retry/filler detection ----------
FILLERS = set([w.strip().lower() for w in (os.getenv("SEM_FILLER_LIST","um,uh,like,so,okay").split(",")) if w.strip()])
RETRY_RX = re.compile(
    r"\b("
    r"wait|hold\s*on|start\s*again|start\s*over|retry|take\s*two|"
    r"no,\s*let\s*me|sorry|actually|i\s*mean"
    r")\b",
    re.I
)

# ---------- Embedding helpers with graceful fallbacks ----------
def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (s or "").lower())

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / float(len(A | B))

_use_st = None
_use_sklearn = None

def _try_imports():
    global _use_st, _use_sklearn
    if _use_st is not None: return
    try:
        from sentence_transformers import SentenceTransformer  # noqa
        _use_st = True
        _use_sklearn = False
        return
    except Exception:
        _use_st = False
        try:
            import sklearn  # noqa
            _use_sklearn = True
        except Exception:
            _use_sklearn = False

_try_imports()

class _Embedder:
    _st_model = None
    _tfidf = None

    @classmethod
    def encode(cls, texts: List[str]):
        # 1) sentence-transformers (best, if present)
        if _use_st:
            if cls._st_model is None:
                from sentence_transformers import SentenceTransformer
                cls._st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                print("✅ Loaded ST all-MiniLM-L6-v2", flush=True)
            V = cls._st_model.encode(texts, normalize_embeddings=True)
            return V, lambda A,B: float((A*B).sum())

        # 2) sklearn Tfidf (good fallback)
        if _use_sklearn:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            if cls._tfidf is None:
                cls._tfidf = TfidfVectorizer(min_df=1, max_df=0.95)
            X = cls._tfidf.fit_transform(texts)
            return X, lambda A,B: float(cosine_similarity(A, B)[0,0])

        # 3) token jaccard (lightest fallback)
        toks = [_tokenize(t) for t in texts]
        class _TokWrap:
            def __init__(self, arr): self.arr = arr
        return _TokWrap(toks), lambda A,B: _jaccard(A.arr[0], B.arr[1])

def _cos_pair(text_a: str, text_b: str) -> float:
    V, simf = _Embedder.encode([text_a, text_b])
    return simf(V[0], V[1])

# ---------- Data ----------
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    has_product: bool = False
    ocr_hit: int = 0
    meta: Dict = field(default_factory=dict)

# ---------- Core logic ----------
def _filler_rate(t: str) -> float:
    if not t: return 0.0
    words = _tokenize(t)
    if not words: return 0.0
    hits = sum(1 for w in words if w in FILLERS)
    return hits / max(1, len(words))

def is_retry_or_noise(text: str) -> bool:
    return bool(RETRY_RX.search(text or "")) or (_filler_rate(text) > SEM_FILLER_MAX_RATE)

def dedup_takes(takes: List[Take]) -> List[Take]:
    if not takes: return []
    kept: List[Take] = []
    texts = [t.text or "" for t in takes]
    V, simf = _Embedder.encode(texts)
    for i, t in enumerate(takes):
        if is_retry_or_noise(t.text):
            t.meta["drop_reason"] = "retry_or_noise"
            continue
        is_dup = False
        for k in kept:
            j = texts.index(k.text)
            s = simf(V[i], V[j])
            if s >= SEM_DUP_THRESHOLD:
                is_dup = True
                t.meta["drop_reason"] = "duplicate"
                break
        if not is_dup:
            kept.append(t)
    return kept

def can_merge(a: Take, b: Take) -> bool:
    sem = _cos_pair(a.text or "", b.text or "")
    if sem < SEM_MERGE_SIM:
        return False
    # visual continuity proxy: prefer explicit vtx_sim if both present; else fall back to scene_q
    viz = 0.5*(a.vtx_sim + b.vtx_sim) if (a.vtx_sim and b.vtx_sim) else min(a.scene_q, b.scene_q)
    if viz < VIZ_MERGE_SIM:
        return False
    # gentle temporal gap check
    if (b.start - a.end) > 2.5:
        return False
    if a.meta.get("scene_cut_next") or b.meta.get("scene_cut_prev"):
        return False
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
        merged.meta = dict(merged.meta)
        merged.meta["chain_ids"] = [c.id for c in chain]
        merged.end = chain[-1].end
        out.append(merged)
        i = j + 1
    return out

# compatibility alias for older callers
def continuity_chains(takes: List[Take]) -> List[List[Take]]:
    chains = []
    stitched = stitch_chain(takes)
    for t in stitched:
        ids = t.meta.get("chain_ids", [t.id])
        chains.append([x for x in takes if x.id in ids])
    return chains
