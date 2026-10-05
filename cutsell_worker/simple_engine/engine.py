"""CutSell simple engine v1.0 - decision + finishing, calibrated on the 25-video Human Gold set.

words (Deepgram, word times) -> pass step -> decision step -> loose-piece rule -> silence refine
-> shot-change snap -> splits (visible / hidden + reason) + captions on the timeline of the cut.

Two LLM calls per video, one vote. Everything after the decision is code (no AI cost).
Safety net: a malformed answer is asked again (up to 3 tries); an answer that hides almost every
word is asked again once and the less destructive answer wins.

The numeric constants and the two prompts were tuned together against the Gold set. Change them
only with a full re-run of the evaluation (scripts/simple_engine_eval.py).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from . import llm as _llm

VERSION = "1.0"
_HERE = Path(__file__).parent
PASS_PROMPT = (_HERE / "prompts" / "pass_v1.txt").read_text(encoding="utf-8")
DECISION_PROMPT = (_HERE / "prompts" / "decision_v1.txt").read_text(encoding="utf-8")

OV = 0.08                                   # audio crossfade assumed at every junction (s)
WHY = {"r": "retoma", "a": "frase abandonada", "n": "ruido", "p": "otra pasada"}
# splits_from_labels
MAX_INNER_PAUSE, INNER_PAUSE_KEEP, PAD_BEFORE, PAD_AFTER, MERGE_GAP = 0.60, 0.25, 0.08, 0.15, 0.12
MAX_WORD, WORD_CLAMP = 1.2, 0.9
# loose-piece rule
ORPHAN_MAX_WORDS, ORPHAN_MAX_SEC = 1, 2.0
# refine
SR, HOP = 16000, 160                        # 10 ms frames
RF_MAX_PAUSE, RF_KEEP_EACH = 0.30, 0.12
RF_LOOK_BACK, RF_LOOK_FWD, RF_QUIET_MIN = 0.30, 0.45, 0.05
RF_MERGE_GAP = 0.10
# visual snap
VS_WIN, VS_TH, VS_FR = 0.6, 0.20, 1 / 30

LLMCall = Callable[[str], "tuple[str, dict]"]


def _norm(text: str) -> str:
    return re.sub(r"[^\w']", "", text.lower())


# ---------------------------------------------------------------- decision

def parse(text: str) -> dict:
    """The model sometimes writes notes before the JSON, or L6 instead of 6: take the last valid object."""
    for candidate in (text, re.sub(r"\bL(\d+)\b", r"\1", text)):
        best = None
        for match in re.finditer(r"\{", candidate):
            try:
                obj = json.JSONDecoder().raw_decode(candidate[match.start():])[0]
            except Exception:
                continue
            if isinstance(obj, dict) and ("hide" in obj or "passes" in obj or "trim" in obj):
                best = obj
        if best is not None:
            return best
    raise ValueError("no JSON in answer")


def lines_of(words):
    units, cur = [], []
    for k, w in enumerate(words):
        if cur and w["s"] - words[k - 1]["e"] >= 0.5:
            units.append(cur); cur = []
        cur.append(k)
        t = w["w"]
        if t.endswith((".", "?", "!")) or (len(cur) >= 12 and t.endswith(",")):
            units.append(cur); cur = []
    if cur:
        units.append(cur)
    return units


def transcript(words, units) -> str:
    out = []
    for n, u in enumerate(units):
        if n:
            gap = words[u[0]]["s"] - words[u[0] - 1]["e"]
            if gap >= 0.7:
                out.append(f"[pause {gap:.1f}s]")
        out.append(f"L{n + 1}| " + " ".join(words[k]["w"] for k in u))
    return "\n".join(out)


def _ask(call: LLMCall, tag: str, prompt: str, tries: int = 3):
    """One LLM call with the safety net for malformed answers. Returns (decision, [usage])."""
    last = None
    for i in range(tries):
        run_tag = tag if i == 0 else f"{tag}-retry{i}"
        try:
            text, usage = call(prompt + f"\n(run {run_tag})")
            return parse(text), [usage]
        except Exception as exc:                      # malformed answer or transport error: ask again
            last = exc
    raise RuntimeError(f"engine answer unreadable after {tries} tries: {last}")


def labels(dec, words, units):
    n = len(units)
    lab = [("keep", "se queda")] * len(words)
    for h in dec.get("hide", []):
        try:
            a, b = int(h[0]), int(h[1])
        except Exception:
            continue
        for ln in range(max(1, a), min(n, b) + 1):
            for k in units[ln - 1]:
                lab[k] = ("hide", WHY.get(h[2] if len(h) > 2 else "", "ruido"))
    for t in dec.get("trim", []):
        try:
            ln = int(t[0])
            target = [_norm(x) for x in str(t[1]).split() if _norm(x)]
        except Exception:
            continue
        if not (1 <= ln <= n) or not target:
            continue
        u = units[ln - 1]
        toks = [_norm(words[k]["w"]) for k in u]
        for i in range(len(toks) - len(target) + 1):
            if toks[i:i + len(target)] == target:
                if len(target) < len(toks):
                    for k in u[i:i + len(target)]:
                        lab[k] = ("hide", "tropiezo")
                break
    return lab


def _timing_words(words):
    """Timing-only copy: a mid-sentence word stretched over a hesitation keeps only its last WORD_CLAMP s."""
    out = []
    for w in words:
        w = dict(w)
        if w["e"] - w["s"] > MAX_WORD and not w["w"].rstrip().endswith((".", "?", "!", "...", ",")):
            w["s"] = w["e"] - WORD_CLAMP
        out.append(w)
    return out


def splits_from_labels(words, final, duration):
    """Contiguous runs -> splits. Visible splits get silences trimmed into hidden 'silencio' splits."""
    words = _timing_words(words)
    runs = []
    for i, (st, why, doubt) in enumerate(final):
        if runs and runs[-1]["state"] == st and runs[-1]["why"] == why:
            runs[-1]["b"] = i; runs[-1]["doubt"] |= doubt
        else:
            runs.append({"state": st, "why": why, "a": i, "b": i, "doubt": doubt})
    vis = []
    for r in runs:
        if r["state"] != "keep":
            continue
        s = words[r["a"]]["s"]
        for k in range(r["a"], r["b"]):
            if words[k + 1]["s"] - words[k]["e"] > MAX_INNER_PAUSE:
                vis.append([s, words[k]["e"] + INNER_PAUSE_KEEP / 2, r["why"], r["doubt"]])
                s = words[k + 1]["s"] - INNER_PAUSE_KEEP / 2
        vis.append([s, words[r["b"]]["e"], r["why"], r["doubt"]])
    vis = [[max(0, a - PAD_BEFORE), min(duration, b + PAD_AFTER), w, d] for a, b, w, d in vis]
    merged = []
    for v in sorted(vis):
        if merged and v[0] - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], v[1])
        else:
            merged.append(v)
    hidden_reason = lambda a, b: next(
        (why for (st, why, _), w in zip(final, words) if st == "hide" and w["s"] >= a - 0.05 and w["e"] <= b + 0.05),
        "silencio")
    splits, t = [], 0.0
    for a, b, why, doubt in merged:
        if a - t > 0.05:
            splits.append({"start": round(t, 3), "end": round(a, 3), "visible": False, "reason": hidden_reason(t, a)})
        splits.append({"start": round(a, 3), "end": round(b, 3), "visible": True, "reason": why, "doubt": doubt})
        t = b
    if duration - t > 0.05:
        splits.append({"start": round(t, 3), "end": round(duration, 3), "visible": False, "reason": hidden_reason(t, duration)})
    return splits


def decide(words, duration, call: LLMCall, run: int = 0):
    """Pass step + decision step -> raw splits (before refine)."""
    units = lines_of(words)
    text = transcript(words, units)
    usage = []
    p, u = _ask(call, f"v3pass-{run}", PASS_PROMPT + "\nTRANSCRIPT:\n" + text); usage += u
    passes = p.get("passes", [])
    backbone = p.get("backbone")
    multi = (sum(1 for x in passes if len(x) > 2 and x[2]) >= 2
             and isinstance(backbone, list) and len(backbone) == 2)
    mode = ("MULTI-PASS: the recording has several passes. The BACKBONE is fixed: lines L%d..L%d. Hide everything outside it with code \"p\", "
            "except a whole sentence with a fact the backbone truly lacks." % (backbone[0], backbone[1])) if multi else (
            "SINGLE PASS: the recording is one continuous story/pitch. When a sentence is retaken and each version has a concrete detail the other "
            "lacks (a body part or place, a number, a name, an example, a cause), keep BOTH versions; when one version only rephrases the other with "
            "nothing new, keep the most complete one.")
    prompt = DECISION_PROMPT + "\nFIXED DECISION FROM THE PASS STEP: " + mode + "\n\nTRANSCRIPT:\n" + text
    d, u = _ask(call, f"v3dec-{run}", prompt); usage += u
    lab = labels(d, words, units)
    kept = sum(1 for st, _ in lab if st == "keep")
    if len(words) >= 30 and kept < 0.05 * len(words):   # hides almost everything: ask once more, keep the gentler answer
        d2, u = _ask(call, f"v3dec-{run}-net", prompt); usage += u
        lab2 = labels(d2, words, units)
        if sum(1 for st, _ in lab2 if st == "keep") > kept:
            lab = lab2
    splits = splits_from_labels(words, [(st, why, False) for st, why in lab], duration)
    return splits, {"passes": len(passes) if multi else 1, "usage": usage}


# ---------------------------------------------------------------- finishing (no AI)

def orphans(splits, words):
    """Loose-piece rule: a kept part with 0-1 words and under 2 s is hidden. Must run BEFORE refine."""
    log = []
    for s in splits:
        if not s["visible"]:
            continue
        ws = [w for w in words if s["start"] - 0.05 <= (w["s"] + w["e"]) / 2 <= s["end"] + 0.05]
        if len(ws) <= ORPHAN_MAX_WORDS and s["end"] - s["start"] < ORPHAN_MAX_SEC:
            s["visible"] = False; s["reason"] = "pedazo suelto"
            log.append([s["start"], s["end"], " ".join(w["w"] for w in ws)])
    return log


def _load_audio(path):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def _energy_db(x):
    import numpy as np
    n = len(x) // HOP
    frames = x[: n * HOP].reshape(n, HOP)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-6)


def _threshold(db):
    import numpy as np
    floor, loud = np.percentile(db, 15), np.percentile(db, 90)
    return floor + 0.30 * (loud - floor)


def refine(splits, audio_path, duration, words):
    """Snap every visible split to the real audio. Word-aware: edges only move inside the silence next to the
    kept words (never into a hidden word); inner pauses are removed only in real gaps BETWEEN words."""
    db = _energy_db(_load_audio(audio_path))
    if len(db) == 0:
        return splits
    q = db < _threshold(db)
    f = lambda t: int(max(0, min(len(q) - 1, round(t * SR / HOP))))
    t = lambda i: i * HOP / SR
    need = max(1, int(RF_QUIET_MIN * SR / HOP))

    def snap_start(lo, w0):          # latest quiet moment in [lo, w0]; else a small fixed pad
        for j in range(f(w0), f(lo) - 1, -1):
            if all(q[max(0, j - need):j + 1]):
                return max(lo, t(j) - 0.02)
        return max(lo, w0 - 0.08)

    def snap_end(w1, hi):            # earliest quiet moment in [w1, hi]; else a small fixed pad
        for j in range(f(w1), f(hi) + 1):
            if all(q[j:j + need]):
                return min(hi, t(j) + 0.04)
        return min(hi, w1 + 0.12)

    vis = []
    for s in splits:
        if not s["visible"]:
            continue
        kept = [i for i, w in enumerate(words) if w["s"] >= s["start"] - 0.05 and w["e"] <= s["end"] + 0.05]
        if not kept:
            vis.append([s["start"], s["end"], s["reason"], s.get("doubt", False)]); continue
        i0, i1 = kept[0], kept[-1]
        lo = max(0.0, words[i0 - 1]["e"] + 0.02 if i0 > 0 else 0.0, words[i0]["s"] - RF_LOOK_BACK)
        hi = min(duration, words[i1 + 1]["s"] - 0.02 if i1 + 1 < len(words) else duration, words[i1]["e"] + RF_LOOK_FWD)
        a, b = snap_start(lo, words[i0]["s"]), snap_end(words[i1]["e"], hi)
        cur = a
        for k in range(i0, i1):                       # inner pauses: only between two kept words
            ge, gs = words[k]["e"], words[k + 1]["s"]
            if gs - ge > RF_MAX_PAUSE:
                vis.append([cur, snap_end(ge, ge + RF_KEEP_EACH + 0.1), s["reason"], s.get("doubt", False)])
                cur = snap_start(gs - RF_KEEP_EACH - 0.1, gs)
        vis.append([cur, b, s["reason"], s.get("doubt", False)])
    vis.sort()
    merged = []
    for v in vis:
        if merged and (v[0] - merged[-1][1] <= RF_MERGE_GAP or v[0] < merged[-1][1]):
            merged[-1][1] = max(merged[-1][1], v[1])
        else:
            merged.append(v)
    hidden = [s for s in splits if not s["visible"]]

    def reason_at(a, b):
        best, overlap = "silencio", 0
        for h in hidden:
            o = min(b, h["end"]) - max(a, h["start"])
            if o > overlap:
                best, overlap = h["reason"], o
        return best

    out, cur = [], 0.0
    for a, b, why, doubt in merged:
        a, b = float(a), float(b)
        if a - cur > 0.03:
            out.append({"start": round(cur, 3), "end": round(a, 3), "visible": False, "reason": reason_at(cur, a)})
        out.append({"start": round(a, 3), "end": round(b, 3), "visible": True, "reason": why, "doubt": bool(doubt)})
        cur = b
    if duration - cur > 0.03:
        out.append({"start": round(cur, 3), "end": round(duration, 3), "visible": False, "reason": reason_at(cur, duration)})
    return out


def scene_cuts(video, th: float = VS_TH):
    """Hard visual cuts already present in the source: [(first_frame_time, last_frame_time)]."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='gt(scene,{th})',metadata=print:file=-",
                        "-an", "-f", "null", "-"], capture_output=True, text=True)
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", r.stdout + r.stderr)]
    groups = []                                   # flashes of 1-3 frames -> one cut, placed at its LAST frame
    for t in times:
        if groups and t - groups[-1][-1] <= 0.12:
            groups[-1].append(t)
        else:
            groups.append([t])
    return [(g[0], g[-1]) for g in groups]


def snap(splits, words, cuts):
    """Move a kept edge onto a shot change right next to it, when no whole word lives in the leftover."""
    has_word = lambda a, b: any(w["s"] >= a - 0.05 and w["e"] <= b + 0.05 for w in words)
    log = []
    for s in splits:
        if not s["visible"]:
            continue
        for first, last in cuts:
            if s["start"] < first and last - s["start"] <= VS_WIN and last < s["end"] and not has_word(s["start"], last):
                log.append(["inicio", s["start"], round(last, 3)]); s["start"] = round(last, 3)
            if (first - VS_FR < s["end"] and s["end"] - first <= VS_WIN and first > s["start"]
                    and not has_word(first, s["end"]) and s["end"] > first):
                log.append(["final", s["end"], round(first, 3)]); s["end"] = round(first, 3)
    return log


# ---------------------------------------------------------------- captions

def captions(words, splits):
    """Words that survive, with their times ON THE CUT (what the phone draws over the exported video)."""
    out, t0 = [], 0.0
    vis = [s for s in splits if s["visible"]]
    for i, s in enumerate(vis):
        for w in words:
            mid = (w["s"] + w["e"]) / 2
            if s["start"] <= mid <= s["end"]:
                out.append({"w": w["w"], "s": round(t0 + max(0, w["s"] - s["start"]), 3),
                            "e": round(t0 + min(s["end"], w["e"]) - s["start"], 3)})
        t0 += s["end"] - s["start"] - (OV if i < len(vis) - 1 else 0)
    return out, round(t0, 3)


def caption_groups(caps, max_words: int = 3, max_gap: float = 0.45):
    """Captions as the editor shows them: groups of about 3 words with every word's time, so the app can light up
    the word being said. A group closes at max_words, at punctuation, or at a pause / a junction of the cut."""
    out, cur = [], []

    def close():
        if cur:
            out.append({"text": " ".join(w["w"] for w in cur), "s": cur[0]["s"], "e": cur[-1]["e"], "words": list(cur)})
        cur.clear()

    for w in caps:
        if cur and (w["s"] - cur[-1]["e"] > max_gap or w["s"] < cur[-1]["e"] - 0.05):
            close()
        cur.append(w)
        if len(cur) >= max_words or w["w"].endswith((".", "?", "!", ",")):
            close()
    close()
    return out


# ---------------------------------------------------------------- entry point

def process(words, duration, audio_path, video_path=None, run: int = 0, scene_cut_list=None,
            llm: Optional[LLMCall] = None) -> dict:
    """words: [{"w","s","e"}] from Deepgram. Returns splits (full non-destructive timeline) + captions."""
    if not words:
        raise ValueError("simple engine: no words to edit")
    call = llm or _llm.call_anthropic
    splits, info = decide(words, float(duration), call, run)
    log = {"sueltos": orphans(splits, words)}
    splits = refine(splits, audio_path, float(duration), words)
    if scene_cut_list is None and video_path:
        scene_cut_list = scene_cuts(video_path)
    log["tomas"] = snap(splits, words, scene_cut_list) if scene_cut_list else []
    caps, cut_duration = captions(words, splits)
    return {"engine": VERSION, "duration": float(duration), "cut_duration": cut_duration, "passes": info["passes"],
            "splits": splits, "captions": caps, "caption_groups": caption_groups(caps), "log": log, "usage": info["usage"]}
