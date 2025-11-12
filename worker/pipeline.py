import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from worker import utils, asr, vision, llm

# ---------- Config (no hard caps) ----------
MIN_CLAUSE_WORDS   = int(os.getenv("MIN_CLAUSE_WORDS", "3"))
LLM_KEEP_THRESHOLD = float(os.getenv("LLM_KEEP_THRESHOLD", "0.58"))  # tweak to be stricter/looser
ALLOW_CTA_EARLY    = os.getenv("ALLOW_CTA_EARLY", "0").strip() == "1"

@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    slot_hint: str = "FEATURE"  # HOOK/FEATURE/PROOF/CTA heuristic

    @property
    def dur(self) -> float:
        return self.end - self.start

def _split_into_clauses(text: str) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    # naive split on .?! and " and " / " but "
    tmp: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in ".?!":
            tmp.append(buf.strip())
            buf = ""
    if buf.strip():
        tmp.append(buf.strip())
    out: List[str] = []
    for piece in tmp:
        low = piece.lower()
        piece = piece.replace(" and ", "|S|").replace(" but ", "|S|")
        for part in piece.split("|S|"):
            p = part.strip(" ,.;")
            if p and len(p.split()) >= MIN_CLAUSE_WORDS:
                out.append(p)
    return out

def _assign_times(seg_start: float, seg_end: float, clauses: List[str]) -> List[Tuple[float, float, str]]:
    dur = max(0.05, seg_end - seg_start)
    total_len = sum(len(c) for c in clauses) or 1
    t = 0
    spans = []
    cursor = seg_start
    for c in clauses:
        frac = len(c) / total_len
        length = dur * frac
        spans.append((cursor, cursor + length, c))
        cursor += length
    return spans

def _heuristic_slot_hint(c: str, is_last: bool) -> str:
    low = c.lower()
    if any(w in low for w in ["why not", "listen", "stop scrolling", "okay", "let me show"]):
        return "HOOK"
    if not is_last and any(w in low for w in ["it has", "comes with", "it's actually", "feature", "material", "zipper", "strap", "pocket", "benefit"]):
        return "FEATURE"
    if "because" in low or "proof" in low or "reviews" in low:
        return "PROOF"
    if ("click" in low or "grab" in low or "get yours" in low or "i left it" in low
        or low.startswith("if you want to")):
        return "CTA"
    return "FEATURE"

def _filter_bad_phrases(c: str) -> bool:
    low = c.lower()
    bad = ["wait", "hold on", "lemme start", "let me start", "start over", "no no", "redo", "sorry"]
    for b in bad:
        if b in low:
            return False
    return True

def _load_asr_takes(local_path: str) -> List[Take]:
    segs = asr.transcribe_segments(local_path) or []
    takes: List[Take] = []
    for i, seg in enumerate(segs, 1):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        clauses = _split_into_clauses(txt)
        if not clauses:
            continue
        spans = _assign_times(float(seg["start"]), float(seg["end"]), clauses)
        for j, (s, e, clause) in enumerate(spans, 1):
            if not _filter_bad_phrases(clause):
                continue
            takes.append(
                Take(
                    id=f"ASR{i:04d}_c{j}",
                    start=s, end=e, text=clause
                )
            )
    # Slot hints (rough): last clause tends to CTA permission
    if takes:
        last_id = takes[-1].id
        for idx, t in enumerate(takes):
            is_last = (t.id == last_id)
            t.slot_hint = _heuristic_slot_hint(t.text, is_last)
            if t.slot_hint == "CTA" and not is_last and not ALLOW_CTA_EARLY:
                # demote early CTA to FEATURE unless allowed
                t.slot_hint = "FEATURE"
    return takes

def run_pipeline(
    *,
    session_id: str,
    file_urls: List[str],
    portrait: bool,
    funnel_counts: Dict[str, int],
    max_duration: Optional[float],
    s3_prefix: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    if not file_urls:
        return {"ok": False, "error": "no input files"}

    # Download first file (single-file MVP)
    import tempfile, requests, os as _os
    fd, local_path = tempfile.mkstemp(suffix=".mp4")
    _os.close(fd)
    with requests.get(file_urls[0], stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk:
                    f.write(chunk)

    real_dur = utils.ffprobe_duration(local_path)
    cap = float(max_duration) if (max_duration and max_duration > 0) else real_dur

    # 1) ASR → clause takes
    all_takes = _load_asr_takes(local_path)

    # 2) Multimodal LLM scoring (always on; raises if no key)
    keepers: List[Take] = []
    chosen_spans: List[Tuple[float, float]] = []
    reasons: Dict[str, str] = {}
    total = 0.0

    for idx, t in enumerate(all_takes):
        # sample midframe for the clause
        frame_b64 = vision.extract_midframe_b64(local_path, t.start, t.end)
        score, reason = llm.score_clause_multimodal(t.text, frame_b64, t.slot_hint)
        reasons[t.id] = reason
        if score >= LLM_KEEP_THRESHOLD:
            # accept; but don't exceed cap if provided
            if cap and total + t.dur > cap:
                # stop once we fill the requested cap
                break
            keepers.append(t)
            chosen_spans.append((t.start, t.end))
            total += t.dur

    # If nothing passed (too strict), relax threshold once and retry quickly
    if not keepers and all_takes:
        for t in all_takes:
            frame_b64 = vision.extract_midframe_b64(local_path, t.start, t.end)
            score, reason = llm.score_clause_multimodal(t.text, frame_b64, t.slot_hint)
            reasons[t.id] = f"(relaxed) {reason}"
            if score >= max(0.35, LLM_KEEP_THRESHOLD - 0.2):
                if cap and total + t.dur > cap:
                    break
                keepers.append(t)
                chosen_spans.append((t.start, t.end))
                total += t.dur
        # Still empty? keep a 5s safety slice so users see output
        if not keepers:
            sl = min(5.0, real_dur)
            chosen_spans = [(0.0, sl)]
            keepers = [Take(id="FALLBACK", start=0.0, end=sl, text="")]

    # 3) Export
    final_path = utils.ffmpeg_cut_concat(local_path, chosen_spans)
    up = utils.upload_to_s3(final_path, s3_prefix=s3_prefix)

    # 4) Slots (light auto-placement: first→HOOK, mids→FEATURE, last→CTA if CTA-ish; else FEATURE)
    clips = []
    slots = {"HOOK": [], "PROBLEM": [], "FEATURE": [], "PROOF": [], "CTA": []}

    def _trim(txt: str, n: int = 220) -> str:
        txt = txt.strip()
        return txt if len(txt) <= n else txt[:n].rstrip() + "..."

    if keepers:
        for t in keepers:
            clips.append({
                "id": t.id, "slot": "STORY", "start": t.start, "end": t.end,
                "score": 2.5, "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0,
                "chain_ids": [t.id], "text": _trim(t.text)
            })

        # simple slotting
        first = keepers[0]
        slots["HOOK"].append({
            "id": first.id, "start": first.start, "end": first.end, "text": _trim(first.text),
            "meta": {"slot": "HOOK", "score": 2.5, "chain_ids": [first.id]},
            "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0, "has_product": False, "ocr_hit": 0
        })

        for mid in keepers[1:-1]:
            dest = "CTA" if mid.slot_hint == "CTA" and ALLOW_CTA_EARLY else "FEATURE"
            slots[dest].append({
                "id": mid.id, "start": mid.start, "end": mid.end, "text": _trim(mid.text),
                "meta": {"slot": dest, "score": 2.0, "chain_ids": [mid.id]},
                "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0, "has_product": False, "ocr_hit": 0
            })

        if len(keepers) >= 2:
            last = keepers[-1]
            last_dest = "CTA" if (last.slot_hint == "CTA" or "grab" in last.text.lower() or "click" in last.text.lower() or "i left it" in last.text.lower()) else "FEATURE"
            slots[last_dest].append({
                "id": last.id, "start": last.start, "end": last.end, "text": _trim(last.text),
                "meta": {"slot": last_dest, "score": 2.0, "chain_ids": [last.id]},
                "face_q": 1.0, "scene_q": 1.0, "vtx_sim": 0.0, "has_product": False, "ocr_hit": 0
            })

    return {
        "ok": True,
        "session_id": session_id,
        "input_local": local_path,
        "duration_sec": utils.ffprobe_duration(final_path),
        "s3_key": up["s3_key"],
        "s3_url": up["s3_url"],
        "https_url": up["https_url"],
        "clips": clips,
        "slots": slots,
        "asr": True,
        "semantic": True,
        "vision": True
    }
