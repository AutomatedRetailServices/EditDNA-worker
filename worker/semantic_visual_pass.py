# worker/semantic_visual_pass.py
from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

print("🧠 [semantic_visual_pass] Semantic pipeline active.", flush=True)

# ---------------- Env knobs (safe defaults) ----------------
EMBEDDER = os.getenv("EMBEDDER", "local").lower()  # local|openai

# durations (used by composer elsewhere; here for context)
MAX_TAKE_SEC = float(os.getenv("MAX_TAKE_SEC", "20"))
MIN_TAKE_SEC = float(os.getenv("MIN_TAKE_SEC", "1.5"))

# fusion + thresholds
W_SEM  = float(os.getenv("W_SEM",  "1.2"))
W_FACE = float(os.getenv("W_FACE", "0.8"))
W_SCENE= float(os.getenv("W_SCENE","0.5"))
W_PROD = float(os.getenv("W_PROD","0.0"))  # keep 0 if not doing product detect yet
W_OCR  = float(os.getenv("W_OCR", "0.0"))  # keep 0 if OCR not enabled
W_VTX  = float(os.getenv("W_VTX", "0.8"))

SEM_DUP_THRESHOLD   = float(os.getenv("SEM_DUP_THRESHOLD", "0.88"))
SEM_FILLER_MAX_RATE = float(os.getenv("SEM_FILLER_MAX_RATE", "0.08"))
SEM_MERGE_SIM       = float(os.getenv("SEM_MERGE_SIM", "0.80"))
VIZ_MERGE_SIM       = float(os.getenv("VIZ_MERGE_SIM", "0.75"))
MERGE_MAX_CHAIN     = int(os.getenv("MERGE_MAX_CHAIN", "3"))

# slot rules
SLOT_REQUIRE_PRODUCT = set((os.getenv("SLOT_REQUIRE_PRODUCT","") or "").split(",")) - {""}
SLOT_REQUIRE_OCR_CTA = set((os.getenv("SLOT_REQUIRE_OCR_CTA","") or "").split(",")) - {""}

# retry / filler patterns
SEM_FILLER_LIST = [w.strip() for w in os.getenv("SEM_FILLER_LIST","um,uh,like,so,okay").split(",") if w.strip()]
FILLER_RX = re.compile(r"\b(" + "|".join(re.escape(w) for w in SEM_FILLER_LIST) + r")\b", flags=re.I) if SEM_FILLER_LIST else None
RETRY_RX  = re.compile(r"\b(wait|start\s*again|retry|take\s*two|start\s*over|no,\s?let\s?me|sorry|hold\s?on|actually|i mean)\b", re.I)

# ---------------- Embeddings ----------------
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
                print("✅ [semantic_visual_pass] Loaded local SentenceTransformer.", flush=True)
            return cls._model.encode(texts, normalize_embeddings=True).tolist()

def cos(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = (sum(x*x for x in a) ** 0.5) or 1e-9
    db = (sum(y*y for y in b) ** 0.5) or 1e-9
    return max(-1.0, min(1.0, num/(da*db)))

# ---------------- Data model ----------------
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    # visual / quality features (optional; default neutral)
    face_q: float = 1.0         # 0..1
    scene_q: float = 1.0        # 0..1 (stability/exposure proxy)
    vtx_sim: float = 0.0        # 0..1 (vision<->text similarity if you compute it)
    has_product: bool = False
    ocr_hit: int = 0
    # semantic features
    slot_hint: Optional[str] = None
    vec: Optional[List[float]] = None
    score_base: float = 1.0
    meta: Dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)

# ---------------- Helpers ----------------
def filler_rate(t: str) -> float:
    if not t: return 0.0
    words = re.findall(r"\w+", t.lower())
    if not words: return 0.0
    hits = len(FILLER_RX.findall(t)) if FILLER_RX else 0
    return hits / max(1, len(words))

SLOT_LABELS = ["HOOK","PROBLEM","FEATURE","PROOF","CTA"]
KEYS = {
    "HOOK":    ["imagine","what if","did you know","stop scrolling","quick tip","the secret","attention"],
    "PROBLEM": ["problem","struggle","hard","issue","pain","annoying"],
    "FEATURE": ["feature","comes with","includes","made of","made with","formula","works by"],
    "PROOF":   ["results","testimonial","it works","i use it","demo","review","customers say"],
    "CTA":     ["buy","get","claim","use code","link in bio","shop","today","now","download","book","start"],
}

def tag_slot_text(txt: str) -> str:
    t = (txt or "").lower()
    for slot in SLOT_LABELS:
        if any(k in t for k in KEYS[slot]):
            return slot
    if len(t.split()) > 25:
        return "PROOF"
    return "FEATURE"

def is_retry_or_noise(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    fillers = len([w for w in words if w in {w.lower() for w in SEM_FILLER_LIST}])
    rate = fillers / max(1,len(words))
    return bool(RETRY_RX.search(text)) or rate > SEM_FILLER_MAX_RATE

# ---------------- Main ops ----------------
def dedup_takes(takes: List[Take]) -> List[Take]:
    """Drop retries and near-duplicates by cosine similarity."""
    if not takes: return []
    texts = [t.text or "" for t in takes]
    vecs = Emb.encode(texts)
    kept: List[Take] = []
    for tk, v in zip(takes, vecs):
        if is_retry_or_noise(tk.text or ""):
            tk.meta["drop_reason"] = "retry_or_noise"
            continue
        dup = False
        for kk in kept:
            if kk.vec is None:  # shouldn’t happen if we set below, but be safe
                continue
            if cos(v, kk.vec) >= SEM_DUP_THRESHOLD:
                dup = True
                tk.meta["drop_reason"] = "duplicate"
                break
        if not dup:
            tk.vec = v
            kept.append(tk)
    return kept

def score_take(t: Take, slot: str) -> float:
    """Slot-aware scoring combining semantic & visual features."""
    # slot constraints
    if slot in SLOT_REQUIRE_PRODUCT and not t.has_product:
        return -1.0
    if slot in SLOT_REQUIRE_OCR_CTA and t.ocr_hit < 1:
        return -1.0

    # base semantic penalty for filler/retry
    if is_retry_or_noise(t.text or ""):
        base_sem = 0.5
    else:
        base_sem = 1.0

    score = (
        W_SEM  * base_sem +
        W_FACE * float(t.face_q) +
        W_SCENE* float(t.scene_q) +
        W_PROD * (1.0 if t.has_product else 0.0) +
        W_OCR  * min(1.0, float(t.ocr_hit)) +
        W_VTX  * float(t.vtx_sim)
    )
    return float(score)

def can_merge(a: Take, b: Take) -> bool:
    """Smart stitch: semantic + visual continuity, no hard scene cut."""
    if not a.vec or not b.vec:
        # compute on the fly if missing
        a.vec = a.vec or Emb.encode([a.text or ""])[0]
        b.vec = b.vec or Emb.encode([b.text or ""])[0]
    s_sem = cos(a.vec, b.vec)
    if s_sem < SEM_MERGE_SIM:
        return False
    s_viz = 0.5*(a.vtx_sim + b.vtx_sim) if (a.vtx_sim and b.vtx_sim) else min(a.scene_q, b.scene_q)
    if s_viz < VIZ_MERGE_SIM:
        return False
    if a.meta.get("scene_cut_next") or b.meta.get("scene_cut_prev"):
        return False
    # keep small gap tolerance (≤2s) so joins sound natural
    return 0.0 <= (b.start - a.end) <= 2.0

def stitch_chain(takes: List[Take]) -> List[Take]:
    """Build merged chains up to MERGE_MAX_CHAIN; extend end time of head."""
    if not takes: return []
    takes = sorted(takes, key=lambda x: (x.start, x.end))
    out: List[Take] = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        while (j+1 < len(takes)) and (len(chain) < MERGE_MAX_CHAIN) and can_merge(takes[j], takes[j+1]):
            chain.append(takes[j+1]); j += 1
        head = chain[0]
        head.meta["chain_ids"] = [c.id for c in chain]
        head.end = chain[-1].end
        out.append(head)
        i = j + 1
    return out

# --------------- Backward-compat API ---------------
# Some older code expects a function named `continuity_chains(takes)`
# that returns List[List[TakeLike]]. We map to our stitch logic but keep signature.
def continuity_chains(takes: List[Take]) -> List[List[Take]]:
    """Return chains (lists) for compatibility; each chain is what stitch_chain would have merged."""
    if not takes: return []
    takes = sorted(takes, key=lambda x: (x.start, x.end))
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

__all__ = [
    "Take",
    "dedup_takes",
    "score_take",
    "can_merge",
    "stitch_chain",
    "continuity_chains",
    "tag_slot_text",
]
