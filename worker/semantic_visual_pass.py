# worker/semantic_visual_pass.py
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

print("🧠 [semantic_visual_pass] Semantic pipeline active (Torch optional).", flush=True)

# ---------------- Env knobs ----------------
W_SEM  = float(os.getenv("W_SEM",  "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE= float(os.getenv("W_SCENE","0.5"))
W_VTX  = float(os.getenv("W_VTX",  "0.8"))

SEM_DUP_THRESHOLD   = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_MERGE_SIM       = float(os.getenv("SEM_MERGE_SIM",     "0.80"))
VIZ_MERGE_SIM       = float(os.getenv("VIZ_MERGE_SIM",     "0.75"))
MERGE_MAX_CHAIN     = int(os.getenv("MERGE_MAX_CHAIN",     "3"))
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE","0.08"))

SLOT_REQUIRE_PRODUCT = set((os.getenv("SLOT_REQUIRE_PRODUCT","") or "").split(",")) - {""}
SLOT_REQUIRE_OCR_CTA = set((os.getenv("SLOT_REQUIRE_OCR_CTA","") or "").split(",")) - {""}

# -------------- Light slot rules --------------
SLOT_LABELS = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]
KEYS = {
    "HOOK":    ["imagine","what if","did you know","stop scrolling","quick tip","the secret"],
    "PROBLEM": ["problem","struggle","hard","pain","issue","annoying","waste","dry skin"],
    "FEATURE": ["feature","comes with","includes","made of","ingredient","formula","works by"],
    "PROOF":   ["results","testimonial","it works","i use it","demo","watch","evidence"],
    "CTA":     ["buy","get","claim","use code","link in bio","shop","today","now"],
}

FILLER_SET = {w.strip().lower() for w in (os.getenv("SEM_FILLER_LIST","um,uh,like,so,okay").split(","))}
RETRY_RX   = re.compile(r"\b(wait|start\s*again|retry|take\s*two|no,\s?let\s?me|sorry|hold\s?on|actually|i mean)\b", re.I)

# ---------------- Embeddings with fallback ----------------
class _Emb:
    _mode = None
    _st_model = None
    _tfidf_vec = None

    @classmethod
    def _ensure_mode(cls):
        if cls._mode is not None:
            return
        try:
            # Try sentence-transformers (Torch)
            from sentence_transformers import SentenceTransformer  # noqa: F401
            cls._mode = "st"
            print("🔤 [semantic] Using SentenceTransformers backend.", flush=True)
        except Exception as e:
            cls._mode = "tfidf"
            print(f"🔤 [semantic] Falling back to TF-IDF embeddings ({e.__class__.__name__}).", flush=True)

    @classmethod
    def encode(cls, texts: List[str]):
        """
        Returns (vectors, cosine_fn). Vectors may be numpy array or scipy sparse.
        """
        cls._ensure_mode()
        texts = [t or "" for t in texts]

        if cls._mode == "st":
            # Lazy-load model to avoid import until needed
            if cls._st_model is None:
                from sentence_transformers import SentenceTransformer
                cls._st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                print("✅ [semantic] Loaded ST model all-MiniLM-L6-v2.", flush=True)
            import numpy as np
            V = cls._st_model.encode(texts, normalize_embeddings=True)
            def _cos(a, b):
                # already normalized
                return float((a*b).sum())
            return V, _cos

        # TF-IDF fallback (no Torch)
        if cls._tfidf_vec is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            cls._tfidf_vec = TfidfVectorizer(min_df=1)
            # fit on the first batch; vectorizer persists for subsequent calls
            cls._tfidf_vec.fit(texts if texts else [""])
        X = cls._tfidf_vec.transform(texts)

        # cosine for sparse
        from sklearn.metrics.pairwise import cosine_similarity
        def _cos(a, b):
            import numpy as np
            # a,b expected as 1xN sparse matrices
            s = cosine_similarity(a, b)
            return float(s[0,0]) if s.size else 0.0

        return X, _cos

# ---------------- Data model ----------------
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
    # cache
    _vec: Optional[object] = None  # numpy vector or sparse row

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ---------------- Helpers ----------------
def _filler_rate(text: str) -> float:
    if not text: return 0.0
    words = re.findall(r"\w+", text.lower())
    if not words: return 0.0
    hits = sum(1 for w in words if w in FILLER_SET)
    return hits / max(1, len(words))

def tag_slot_text(txt: str) -> str:
    t = (txt or "").lower()
    for slot in SLOT_LABELS:
        if any(k in t for k in KEYS[slot]): return slot
    if len(t.split()) > 25: return "PROOF"
    return "FEATURE"

# ---------------- Core ops ----------------
def is_retry_or_noise(text: str) -> bool:
    return bool(RETRY_RX.search(text or "")) or (_filler_rate(text) > SEM_FILLER_MAX_RATE)

def dedup_takes(takes: List[Take]) -> List[Take]:
    if not takes: return []
    texts = [t.text for t in takes]
    V, cos = _Emb.encode(texts)
    kept: List[Take] = []
    for i, t in enumerate(takes):
        # mark vector row
        vec_i = V[i] if hasattr(V, "__getitem__") else V[i:i+1]
        if is_retry_or_noise(t.text):
            t.meta["drop_reason"] = "retry_or_noise"; continue
        is_dup = False
        for k in kept:
            # compare by vectors (ensure 1xN rows for sparse)
            v_k = k._vec if k._vec is not None else V[texts.index(k.text)]
            a = vec_i if getattr(vec_i, "shape", None) else vec_i
            b = v_k if getattr(v_k, "shape", None) else v_k
            if cos(a, b) >= SEM_DUP_THRESHOLD:
                is_dup = True; break
        if not is_dup:
            t._vec = vec_i
            kept.append(t)
        else:
            t.meta["drop_reason"] = "duplicate"
    return kept

def _semantic_sim(a: Take, b: Take) -> float:
    V, cos = _Emb.encode([a.text, b.text])
    a_vec = V[0] if hasattr(V, "__getitem__") else V[0:1]
    b_vec = V[1] if hasattr(V, "__getitem__") else V[1:2]
    return cos(a_vec, b_vec)

def can_merge(a: Take, b: Take) -> bool:
    # semantic continuity
    s_sem = _semantic_sim(a, b)
    if s_sem < SEM_MERGE_SIM:
        return False
    # visual continuity proxy
    s_viz = 0.5*(float(a.vtx_sim) + float(b.vtx_sim)) if (a.vtx_sim and b.vtx_sim) else min(float(a.scene_q), float(b.scene_q))
    if s_viz < VIZ_MERGE_SIM:
        return False
    # require a small natural gap (≤ 2.0s)
    if not (0.0 <= (b.start - a.end) <= 2.0):
        return False
    # block hard scene cut flags if provided
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
        merged.meta["chain_ids"] = [c.id for c in chain]
        merged.end = chain[-1].end
        out.append(merged)
        i = j + 1
    return out

def score_take(t: Take, slot: str) -> float:
    # slot constraints
    if slot in SLOT_REQUIRE_PRODUCT and not t.has_product:
        return -1.0
    if slot in SLOT_REQUIRE_OCR_CTA and t.ocr_hit < 1:
        return -1.0
    # semantic penalty if noisy
    penalty = 0.5 if is_retry_or_noise(t.text) else 0.0
    score = (
        W_SEM*(1.0 - penalty) +
        W_FACE*float(t.face_q) +
        W_SCENE*float(t.scene_q) +
        W_VTX*float(t.vtx_sim)
    )
    return float(score)

# ---------- Back-compat shim used by jobs.py ----------
def continuity_chains(takes_like: List[object]) -> List[List[object]]:
    """
    Accepts list of objects with .start, .end, .text (optional).
    Returns grouped chains of coherent neighbors (no mutation of inputs).
    """
    proxy: List[Take] = []
    for i, t in enumerate(takes_like):
        proxy.append(
            Take(
                id=getattr(t, "id", f"t{i}"),
                start=float(getattr(t, "start", 0.0)),
                end=float(getattr(t, "end", 0.0)),
                text=str(getattr(t, "text", "")),
            )
        )
    stitched = stitch_chain(proxy)
    # Map stitched groups back to original-like objects by id matching
    id_to_obj = {getattr(t, "id", f"t{i}"):t for i,t in enumerate(takes_like)}
    chains: List[List[object]] = []
    for m in stitched:
        chain_ids = m.meta.get("chain_ids", [m.id])
        chains.append([id_to_obj.get(cid, id_to_obj.get(str(cid), None)) for cid in chain_ids if cid in id_to_obj])
    return chains
