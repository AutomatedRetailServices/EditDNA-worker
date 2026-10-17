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
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from . import llm as _llm

VERSION = "1.2.1"
_HERE = Path(__file__).parent
PASS_PROMPT = (_HERE / "prompts" / "pass_v1.txt").read_text(encoding="utf-8")
DECISION_PROMPT = (_HERE / "prompts" / "decision_v1.txt").read_text(encoding="utf-8")
REVIEW_PROMPT = (_HERE / "prompts" / "review_v1.txt").read_text(encoding="utf-8")
FIXWORDS_PROMPT = (_HERE / "prompts" / "fixwords_v2.txt").read_text(encoding="utf-8")
SALE_REVIEW_PROMPT = (_HERE / "prompts" / "sale_review_v1.txt").read_text(encoding="utf-8")
SALE_REVIEW_KINDS = ("restart", "mixed", "repeat", "missing")
ZOOM_KINDS = ("hook", "product", "price", "cta")
BREATH_AFTER, BREATH_BEFORE = 0.30, 0.15         # natural pause kept around each kept phrase (s), silence only

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
            "nothing new, keep the most complete one. A RESTARTED SECTION is different: see the RESTARTED SECTION rule.")
    prompt = DECISION_PROMPT + "\nFIXED DECISION FROM THE PASS STEP: " + mode + "\n\nTRANSCRIPT:\n" + text
    d, u = _ask(call, f"v3dec-{run}", prompt); usage += u
    lab = labels(d, words, units)
    kept = sum(1 for st, _ in lab if st == "keep")
    if len(words) >= 30 and kept < 0.05 * len(words):   # hides almost everything: ask once more, keep the gentler answer
        d2, u = _ask(call, f"v3dec-{run}-net", prompt); usage += u
        lab2 = labels(d2, words, units)
        if sum(1 for st, _ in lab2 if st == "keep") > kept:
            lab = lab2
    lab, u, rv = review(words, units, lab, call, run); usage += u
    splits = splits_from_labels(words, [(st, why, False) for st, why in lab], duration)
    return splits, {"passes": len(passes) if multi else 1, "usage": usage, "review": rv, "labels": lab}


# ---------------------------------------------------------------- sale review (1.2.1): understanding as a reviewer
# The 1.2 decision stays the editor. A senior editor reads that cut and may fix only three known problems in a sale:
# a mixed attempt, an idea said twice, a missing hook / price / CTA / closer. A fix is applied only when 2 of 3
# independent readings propose it, never removes a number/price the cut would lose, and never takes out more than a
# quarter of the kept words. Anything else, or any doubt: the 1.2 cut stands.

def _json_obj(text: str, key: str) -> dict:
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            obj = json.JSONDecoder().raw_decode(text[match.start():])[0]
        except Exception:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    raise ValueError(f"no JSON with {key!r}")


def _line_numbers(xs, n):
    out = []
    for x in xs if isinstance(xs, (list, tuple)) else [xs]:
        nums = [int(m) for m in re.findall(r"\d+", str(x))]
        if len(nums) == 2 and re.search(r"[-–]\s*L?\d", str(x)):
            nums = list(range(nums[0], nums[1] + 1))
        out += [v for v in nums if 1 <= v <= n]
    return out


_DIGITS = re.compile(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|hundred|"
                     r"uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|veinte|treinta|cuarenta|cincuenta|cien|mil)\b", re.I)


def _numbers(text: str) -> set:
    return {_norm(m.group(0)) for m in re.finditer(r"\$?\d[\d.,]*|\b\w+\b", text.lower()) if _DIGITS.search(m.group(0))}


def sale_review(words, units, lab, call: LLMCall, run: int = 0, kinds=SALE_REVIEW_KINDS):
    n = len(units)
    kept = _line_kept(lab, units)
    prompt = SALE_REVIEW_PROMPT + "\nFIRST CUT:\n" + review_text(words, units, kept)
    answers, usage = [], []
    for r in range(3):
        answer = None
        for attempt in range(2):
            try:
                text, u = call(prompt + f"\n(sale-review {run}-{r}" + (f"-retry{attempt})" if attempt else ")")); usage.append(u)
                answer = _json_obj(text, "fixes")
                break
            except Exception:
                continue
        answers.append(answer or {"type": "error", "fixes": []})
        first = answers[0]
        if r == 0 and (str(first.get("type", "")).lower() != "sale" or not first.get("fixes")):
            break
    sale = sum(1 for a in answers if str(a.get("type", "")).lower() == "sale") >= (2 if len(answers) > 1 else 1)
    info = {"type": "sale" if sale else (answers[0].get("type") or "other"), "calls": len(answers), "applied": [],
            "proposed": answers[0].get("fixes") if answers else []}
    if not sale or len(answers) < 3:
        return lab, usage, info
    votes = {}
    for a in answers:
        seen = set()
        for f in a.get("fixes", []) or []:
            kind = str(f.get("kind", "")).lower()
            if kind not in kinds:
                continue
            for act in ("hide", "unhide"):
                for ln in _line_numbers(f.get(act, []), n):
                    seen.add((kind, act, ln))
        for key in seen:
            votes[key] = votes.get(key, 0) + 1
    chosen = sorted(k for k, c in votes.items() if c >= 2)
    new = list(lab)
    for kind, act, ln in chosen:
        if act == "hide" and kept[ln - 1]:
            for k in units[ln - 1]:
                new[k] = ("hide", {"mixed": "revisión: otro intento", "repeat": "revisión: repetido",
                                   "restart": "revisión: venta reiniciada"}.get(kind, "revisión"))
        elif act == "unhide" and not kept[ln - 1]:
            for k in units[ln - 1]:
                new[k] = ("keep", "revisión: faltaba" if kind == "missing" else "se queda")
    before = " ".join(w["w"] for w, l in zip(words, lab) if l[0] == "keep")
    after = " ".join(w["w"] for w, l in zip(words, new) if l[0] == "keep")
    removed = sum(1 for a, b in zip(lab, new) if a[0] == "keep" and b[0] == "hide")
    if _numbers(before) - _numbers(after):
        info["rejected"] = "a number/price would disappear"
        return lab, usage, info
    # a restarted pitch can legitimately take out a whole first run (about a third of the cut)
    limit = 0.40 if any(k[0] == "restart" for k in chosen) else 0.25
    if removed > limit * max(1, sum(1 for l in lab if l[0] == "keep")):
        info["rejected"] = "too much removed"
        return lab, usage, info
    info["applied"] = [list(k) for k in chosen]
    return new, usage, info


def decide_reviewed(words, duration, call: LLMCall, run: int = 0, kinds=SALE_REVIEW_KINDS):
    """1.2 decision, then the sale review on top of it."""
    splits, info = decide(words, duration, call, run)
    units = lines_of(words)
    lab, u, srv = sale_review(words, units, info["labels"], call, run, kinds)
    info["usage"] = info["usage"] + u
    info["sale_review"] = srv
    if srv.get("applied"):
        splits = splits_from_labels(words, [(st, why, False) for st, why in lab], duration)
    info["labels"] = lab
    return splits, info


# ---------------------------------------------------------------- editor review (hauls / try-ons only)

def _line_kept(lab, units):
    return [sum(1 for k in u if lab[k][0] == "keep") * 2 >= len(u) for u in units]


def review_text(words, units, kept) -> str:
    out = []
    for n, u in enumerate(units):
        if n:
            gap = words[u[0]]["s"] - words[u[0] - 1]["e"]
            if gap >= 0.7:
                out.append(f"[pause {gap:.1f}s]")
        out.append(f"L{n + 1} [{'KEEP' if kept[n] else 'CUT'}]| " + " ".join(words[k]["w"] for k in u))
    return "\n".join(out)


def _parse_review(text: str) -> dict:
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            obj = json.JSONDecoder().raw_decode(text[match.start():])[0]
        except Exception:
            continue
        if isinstance(obj, dict) and ("hide" in obj or "unhide" in obj or "type" in obj):
            return obj
    raise ValueError("no review JSON")


def _line_no(x):
    try:
        return int(x[0]) if isinstance(x, (list, tuple)) else int(x)
    except Exception:
        return None


def review(words, units, lab, call: LLMCall, run: int = 0):
    """An editor reads the finished cut. Only for a haul / try-on (several different items) it may hide a repeated
    announcement and fix mixed takes. A change is applied only when 2 of 3 independent answers agree; when the first
    answer says "other" or proposes nothing, no confirmation calls are made (cheap for ordinary pitches)."""
    kept = _line_kept(lab, units)
    prompt = REVIEW_PROMPT + "\nFIRST CUT:\n" + review_text(words, units, kept)
    answers, usage = [], []
    errors = 0
    for r in range(3):
        answer = None
        for attempt in range(3):                     # transport errors / unreadable answers: ask again, never guess
            try:
                text, u = call(prompt + f"\n(review {run}-{r}" + (f"-retry{attempt})" if attempt else ")")); usage.append(u)
                answer = _parse_review(text)
                break
            except Exception:
                errors += 1
        answers.append(answer or {"type": "error", "hide": [], "unhide": []})
        first = answers[0]
        if r == 0 and (first.get("type") != "haul" or not (first.get("hide") or first.get("unhide"))):
            break
    haul = sum(1 for a in answers if a.get("type") == "haul") >= 2

    def votes(key):
        count = {}
        for a in answers:
            for ln in {_line_no(x) for x in a.get(key, []) or []} - {None}:
                count[ln] = count.get(ln, 0) + 1
        return {ln for ln, c in count.items() if c >= 2 and 1 <= ln <= len(units)}

    hide, unhide = (votes("hide"), votes("unhide")) if haul else (set(), set())
    lab = list(lab)
    for ln in hide:
        if kept[ln - 1]:
            for k in units[ln - 1]:
                lab[k] = ("hide", "retoma")
    for ln in unhide:
        if not kept[ln - 1]:
            for k in units[ln - 1]:
                lab[k] = ("keep", "se queda")
    return lab, usage, {"haul": haul, "hide": sorted(hide), "unhide": sorted(unhide), "calls": len(answers), "errors": errors}


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


def breathe(splits, words, duration):
    """Leave a short natural pause after (and a shorter one before) each kept part, taken only from the silence
    next to it, never over a hidden word. The cut sounds like an editor, not like phrases glued end to end."""
    out = [dict(s) for s in splits]
    has_word = lambda a, b: any(a < (w["s"] + w["e"]) / 2 < b for w in words)
    for i, s in enumerate(out):
        if not s["visible"]:
            continue
        if i + 1 < len(out) and not out[i + 1]["visible"]:
            n = out[i + 1]; ext = min(BREATH_AFTER, (n["end"] - n["start"]) * 0.5)
            while ext > 0.02 and has_word(s["end"], s["end"] + ext):
                ext -= 0.05
            if ext > 0.02:
                s["end"] = round(s["end"] + ext, 3); n["start"] = s["end"]
        if i > 0 and not out[i - 1]["visible"]:
            p = out[i - 1]; ext = min(BREATH_BEFORE, (p["end"] - p["start"]) * 0.5)
            while ext > 0.02 and has_word(s["start"] - ext, s["start"]):
                ext -= 0.05
            if ext > 0.02:
                s["start"] = round(s["start"] - ext, 3); p["end"] = s["start"]
    out = [s for s in out if s["end"] - s["start"] > 0.001]
    out[0]["start"], out[-1]["end"] = 0.0, duration
    return out


def fix_words(words, splits, call: LLMCall):
    """Caption text only: an LLM fixes clearly misheard words in the kept part. Edit decisions never change.
    Returns (words for captions, [[index, old, new]], usage). On repeated failure the captions keep the raw words."""
    shown, fixes, _moments, usage = fix_words_and_moments(words, splits, call)
    return shown, fixes, usage


def _parse_fix_answer(text: str):
    """v2 answer {"fix": [...], "zoom": [...]}; a bare v1 list [[i, "w"], ...] is still accepted."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return list(data.get("fix") or []), list(data.get("zoom") or [])
        except ValueError:
            pass
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON in the word-fix answer")
    return json.loads(m.group(0)), []


def fix_words_and_moments(words, splits, call: LLMCall):
    """The word-fix call (already made for every video) also marks the key moments for Auto Zoom AI:
    hook, product, price, call to action -- no extra call, no extra cost, and the edit decisions are
    never touched. Returns (shown words, fixes, moments [{"i", "kind"}], usage)."""
    vis = [(s["start"], s["end"]) for s in splits if s["visible"]]
    idx = [i for i, w in enumerate(words) if any(a <= (w["s"] + w["e"]) / 2 <= b for a, b in vis)]
    if not idx:
        return words, [], [], []
    prompt = FIXWORDS_PROMPT + "\n" + " ".join(f"{i}:{words[i]['w']}" for i in idx)
    usage = []
    pairs, marks = None, []
    for attempt in range(3):
        try:
            text, u = call(prompt + (f"\n(retry {attempt})" if attempt else "")); usage.append(u)
            pairs, marks = _parse_fix_answer(text)
            break
        except Exception:
            pairs = None
    allowed = set(idx)
    moments = []
    for mark in marks or []:
        try:
            i, kind = int(mark[0]), str(mark[1]).strip().lower()
        except Exception:
            continue
        if i in allowed and kind in ZOOM_KINDS and not any(m["i"] == i for m in moments):
            moments.append({"i": i, "kind": kind})
    if not pairs:
        return words, ([] if pairs == [] else [["error"]]), moments, usage
    shown = [dict(w) for w in words]; fixes = []
    for p in pairs:
        try:
            i, new = int(p[0]), str(p[1]).strip()
        except Exception:
            continue
        if i in allowed and new and new != shown[i]["w"] and len(new) <= 3 * len(shown[i]["w"]) + 8:
            fixes.append([i, shown[i]["w"], new]); shown[i]["w"] = new
    return shown, fixes, moments, usage


def keep_spoken_words(splits, words):
    """A part hidden as silence (short gap, reason 'silencio') must never contain a spoken word: quiet words such as
    a soft "set de short" would otherwise vanish. Adjacent visible parts are merged back."""
    for s in splits:
        if not s["visible"] and s.get("reason") in ("silencio", None) and s["end"] - s["start"] < 3.0:
            if any(s["start"] < (w["s"] + w["e"]) / 2 < s["end"] for w in words):
                s["visible"] = True; s["reason"] = "se queda"
    out = []
    for s in splits:
        if out and out[-1]["visible"] and s["visible"]:
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


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
            llm: Optional[LLMCall] = None, mode: Optional[str] = None) -> dict:
    """words: [{"w","s","e"}] from Deepgram. Returns splits (full non-destructive timeline) + captions.
    mode "rules" = the 1.2 decision only (default); "review" = 1.2 + the sale review (CUTSELL_SIMPLE_ENGINE_MODE)."""
    if not words:
        raise ValueError("simple engine: no words to edit")
    call = llm or _llm.call_anthropic
    mode = (mode or os.environ.get("CUTSELL_SIMPLE_ENGINE_MODE", "") or "rules").strip().lower()
    if mode == "review":
        splits, info = decide_reviewed(words, float(duration), call, run)
    else:
        splits, info = decide(words, float(duration), call, run)
    log = {"sueltos": orphans(splits, words)}
    splits = keep_spoken_words(refine(splits, audio_path, float(duration), words), words)
    if scene_cut_list is None and video_path:
        scene_cut_list = scene_cuts(video_path)
    log["tomas"] = snap(splits, words, scene_cut_list) if scene_cut_list else []
    splits = breathe(splits, words, float(duration))
    shown, fixes, moments, u = fix_words_and_moments(words, splits, call); info["usage"] += u
    caps, cut_duration = captions(shown, splits)
    log["revision"] = info.get("review"); log["palabras"] = fixes
    if info.get("sale_review"):
        log["revision_venta"] = info["sale_review"]
    return {"engine": VERSION, "duration": float(duration), "cut_duration": cut_duration, "passes": info["passes"],
            "splits": splits, "captions": caps, "caption_groups": caption_groups(caps), "log": log, "usage": info["usage"],
            "moments": [{"i": m["i"], "kind": m["kind"], "s": words[m["i"]]["s"]} for m in moments]}
