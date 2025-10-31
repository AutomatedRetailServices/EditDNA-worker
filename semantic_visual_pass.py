import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict


# ---- dataclass to track candidate clips / "takes" ----
@dataclass
class Take:
    id: str
    start: float
    end: float
    text: str
    face_q: float = 1.0
    scene_q: float = 1.0
    vtx_sim: float = 0.0
    chain_ids: list[str] | None = None

    @property
    def dur(self) -> float:
        return float(self.end) - float(self.start)


def _is_fillerish(txt: str, filler_tokens: list[str], max_rate: float) -> bool:
    """
    Drop absolute garbage like 'wait wait no redo'
    or giant filler rate like "um uh like like so um".
    """
    low = txt.lower()
    # instant kill words that clearly mean restart
    restart_markers = [
        "wait", "hold on", "sorry", "let me start again",
        "no no no", "ok ok restart", "cut cut cut"
    ]
    for m in restart_markers:
        if m in low:
            return True

    words = low.split()
    if not words:
        return True
    filler_count = sum(1 for w in words if w.strip(",.?!") in filler_tokens)
    rate = filler_count / max(1, len(words))
    return rate > max_rate


def _dedupe_semantic_greedy(takes: List[Take], dup_threshold: float) -> List[Take]:
    """
    You load sentence-transformers/all-MiniLM-L6-v2 in the pod.
    For now, keep a dumb greedy dedupe:
    if normalized text is repeated, drop later ones.
    """
    seen_norm = []
    out = []
    for t in takes:
        norm = "".join(
            c.lower() for c in t.text if (c.isalnum() or c.isspace())
        ).strip()
        is_dup = any(norm == s for s in seen_norm)
        if not is_dup:
            seen_norm.append(norm)
            out.append(t)
    return out


def _merge_adjacent(
    takes: List[Take],
    viz_merge_sim: float,
    merge_max_chain: int
) -> List[Take]:
    """
    Merge nearby takes into longer thoughts for smoother story flow.
    We loosen the rule but still keep cuts reasonable.
    """
    if not takes:
        return []

    # Sort by time
    takes = sorted(takes, key=lambda t: t.start)
    merged = []
    i = 0
    while i < len(takes):
        chain = [takes[i]]
        j = i
        # Greedy extend forward
        while j + 1 < len(takes) and len(chain) < merge_max_chain:
            a = chain[-1]
            b = takes[j + 1]

            # If huge gap, stop chain
            if b.start - a.end > 1.0:
                break

            # Scene quality gate (rough safety)
            if min(a.scene_q, b.scene_q) < 0.4:
                break

            chain.append(b)
            j += 1

        first = chain[0]
        last = chain[-1]
        combo_text = " ".join([c.text.strip() for c in chain])

        merged.append(
            Take(
                id=f"{first.id}_to_{last.id}",
                start=first.start,
                end=last.end,
                text=combo_text,
                face_q=min(c.face_q for c in chain),
                scene_q=min(c.scene_q for c in chain),
                vtx_sim=max(c.vtx_sim for c in chain),
                chain_ids=[c.id for c in chain],
            )
        )
        i = j + 1

    return merged


def _select_story_in_order(
    merged: List[Take],
    max_duration: float,
    fallback_min_sec: float
) -> List[Take]:
    """
    Naive 'best cut':
    - Keep takes in chronological order
    - Skip tiny 'uh okay wait' lines
    - Stop at max_duration
    - If total is somehow too tiny (< fallback_min_sec), just take first long chain
    """
    story: List[Take] = []
    total = 0.0
    for t in merged:
        clean_words = t.text.strip().split()
        # skip 1-word garbage like "okay" or "wait" alone
        if len(clean_words) <= 2 and any(w.lower() in ("okay", "wait", "yeah") for w in clean_words):
            continue

        dur = t.dur
        if total + dur > max_duration:
            break

        story.append(t)
        total += dur

    # protection: make sure we don't return 2 seconds total
    if total < fallback_min_sec and merged:
        # pick the longest merged take as fallback
        longest = max(merged, key=lambda x: x.dur)
        story = [longest]

    return story


def build_funnel_slots(story: List[Take]) -> Dict[str, list[Dict[str, Any]]]:
    """
    For now:
    - We’re not doing full marketing slot classification (HOOK/PROBLEM/FEATURE/PROOF/CTA)
      because that's a future layer.
    - We just dump everything under HOOK so frontend still works.

    NOTE:
    Your earlier successful responses had slots["HOOK"] full of takes.
    We'll keep that shape so your web UI doesn't break.
    """
    slot_list = []
    for t in story:
        slot_list.append({
            "id": t.id,
            "start": t.start,
            "end": t.end,
            "text": t.text,
            "meta": {
                "slot": "HOOK",
                "score": 2.5
            },
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "has_product": False,
            "ocr_hit": 0
        })

    return {
        "HOOK": slot_list,
        "PROBLEM": [],
        "FEATURE": [],
        "PROOF": [],
        "CTA": [],
    }


def semantic_visual_pass(
    *,
    raw_segments: List[Dict[str, Any]],
    bin_sec: float,
    min_take_sec: float,
    max_take_sec: float,
    veto_min_score: float,
    filler_tokens: list[str],
    filler_max_rate: float,
    sem_merge_sim: float,
    viz_merge_sim: float,
    merge_max_chain: int,
    max_duration: float,
    fallback_min_sec: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[Take]]:
    """
    1. Convert raw_segments -> Take list
    2. Filter fillers / retries
    3. Dedupe
    4. Merge chains
    5. Choose best storyline
    6. Build (clips, slots)

    Returns (clips_list, slots_dict, final_story_takes)
    """

    # 1) Build initial takes
    takes: List[Take] = []
    for i, seg in enumerate(raw_segments, start=1):
        try:
            start = float(seg["start"])
            end   = float(seg["end"])
        except Exception:
            continue

        txt = str(seg.get("text", "")).strip()
        if not txt:
            continue

        face_q = float(seg.get("face_q", 1.0))
        scene_q = float(seg.get("scene_q", 1.0))
        vtx_sim = float(seg.get("vtx_sim", 0.0))

        # length clamp
        dur = end - start
        if dur < min_take_sec:
            continue
        if dur > max_take_sec:
            end = start + max_take_sec
            dur = max_take_sec

        # basic veto: drop trash scored below veto_min_score
        # we don't have a global 'score' yet so we just use face/scene fallback
        rough_score = min(face_q, scene_q)
        if rough_score < veto_min_score:
            continue

        # filler cleanup (um/uh/etc + restart noise)
        if _is_fillerish(txt, filler_tokens=filler_tokens, max_rate=filler_max_rate):
            continue

        takes.append(
            Take(
                id=f"T{i:04d}",
                start=start,
                end=end,
                text=txt,
                face_q=face_q,
                scene_q=scene_q,
                vtx_sim=vtx_sim,
                chain_ids=[f"T{i:04d}"],
            )
        )

    # 2) dedupe similar wording
    deduped = _dedupe_semantic_greedy(takes, dup_threshold=0.88)

    # 3) merge adjacent chunks into bigger thoughts
    merged = _merge_adjacent(
        deduped,
        viz_merge_sim=viz_merge_sim,
        merge_max_chain=merge_max_chain,
    )

    # 4) pick storyline respecting max_duration
    story = _select_story_in_order(
        merged,
        max_duration=max_duration,
        fallback_min_sec=fallback_min_sec,
    )

    # 5) Build 'clips' list like your API logs showed
    clips_list: List[Dict[str, Any]] = []
    for t in story:
        clips_list.append({
            "id": t.id,
            "slot": "HOOK",
            "start": t.start,
            "end": t.end,
            "score": 2.5,
            "face_q": t.face_q,
            "scene_q": t.scene_q,
            "vtx_sim": t.vtx_sim,
            "chain_ids": t.chain_ids or [],
        })

    # 6) Build slots dict (HOOK/PROBLEM/FEATURE/PROOF/CTA lists)
    slots_dict = build_funnel_slots(story)

    return clips_list, slots_dict, story
