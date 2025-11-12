# worker/vision.py
from typing import Dict, Any, List
from .utils import ensure_float

def analyze_frames(media_path: str, asr_segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Very light placeholder vision scoring.
    Returns per-segment scores so the pipeline can rank segments without crashing.
    You can replace this later with real face/scene quality & brand-match.
    """
    out: Dict[str, Dict[str, float]] = {}
    for seg in asr_segments or []:
        sid = seg.get("id") or ""
        dur = max(0.0, ensure_float(seg.get("end", 0.0)) - ensure_float(seg.get("start", 0.0)))
        # Heuristic: longer readable segments get a tiny bump on scene_q
        scene_q = 0.45 + min(dur / 60.0, 0.25)  # 0.45 .. 0.70
        out[sid] = {
            "face_q": 0.50,   # neutral
            "scene_q": float(scene_q),
            "vtx_sim": 0.00,  # not computed in placeholder
        }
    return out
