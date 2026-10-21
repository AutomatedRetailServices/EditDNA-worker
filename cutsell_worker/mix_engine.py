"""Mix: one product video in parts (Hook, Body, Benefit, CTA, ...), several recorded versions of each part, and every
combination (one clip of each part, in order) made into its own video.

The cut engine (cutsell_worker.simple_engine) and its prompts are NOT changed: this module only reuses its finishing
functions (line split, labels -> splits, loose pieces, silence refine, captions grouping). The AI part is one separate
"mix_clip" call for ALL the clips of a Mix together (the fixed instructions are paid once, not once per clip).

    words per clip (Deepgram, once) -> ONE mix_clip call -> per clip: hide/trim labels -> splits -> silence refine
    -> clean clip (ffmpeg: kept parts, common canvas / 30 fps, -16 LUFS, 30 ms fades at every edge)
    -> each combination = the clean clips joined in order (no AI, video copied, audio encoded once)
    -> captions / Auto Zoom moments of each clip moved onto the combination's timeline (no new transcription)
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from .simple_engine import engine as _engine

_LOG = logging.getLogger("cutsell.mix")
MIX_VERSION = "mix-1.0.1"
PROMPT = (Path(__file__).parent / "mix_prompts" / "mix_clip_v2.txt").read_text(encoding="utf-8")
API_URL = "https://api.anthropic.com/v1/messages"
# Measured choice (see docs / the Mix report): CUTSELL_MIX_MODEL overrides without a redeploy.
DEFAULT_MODEL = "claude-sonnet-5-5"   # Product Owner, 21 Oct: Sonnet for mix_clip
PRICES = {  # USD per million tokens (Anthropic price list), <= 200k context
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
}

MAX_CLIPS_PER_SECTION, MAX_SECTIONS, MAX_COMBOS = 10, 6, 100
FADE_SEC = 0.03                 # audio fade at every edge of a kept part: no click at a join
TARGET_LUFS, TARGET_TP, TARGET_LRA = -16.0, -1.5, 11.0
FPS = 30
CANVAS_PORTRAIT, CANVAS_LANDSCAPE = (1080, 1920), (1920, 1080)
ZOOM_KINDS = ("hook", "product", "price", "cta")

LLMCall = Callable[[dict], "tuple[str, dict]"]


# ---------------------------------------------------------------- combinations (no AI)

def section_codes(names: list[str]) -> list[str]:
    """Short code per section: first letter, longer when two sections share it (Body/Benefit -> B, Be)."""
    out: list[str] = []
    for name in names:
        clean = re.sub(r"[^A-Za-z0-9]", "", name or "") or "S"
        code = clean[0].upper()
        k = 1
        while code in out and k < len(clean):
            k += 1
            code = clean[0].upper() + clean[1:k].lower()
        while code in out:
            code += "x"
        out.append(code)
    return out


def combo_code(picks: list[int], codes: list[str]) -> str:
    return " · ".join(f"{c}{p + 1}" for c, p in zip(codes, picks))


def check_limits(counts: list[int]) -> None:
    if not counts or len(counts) > MAX_SECTIONS:
        raise ValueError(f"a Mix has 1 to {MAX_SECTIONS} sections")
    if any(c < 1 or c > MAX_CLIPS_PER_SECTION for c in counts):
        raise ValueError(f"each section has 1 to {MAX_CLIPS_PER_SECTION} clips")


def total_combos(counts: list[int]) -> int:
    total = 1
    for c in counts:
        total *= c
    return total


def plan(counts: list[int], how_many: Optional[int] = None) -> list[list[int]]:
    """The combinations to make, in the order they are shown. Every clip of a section is used a similar number of
    times (the first and last parts weigh most: each hook / each CTA), and two videos in a row never share the same
    start or the same end when that can be avoided (never both). All of them when how_many is None."""
    check_limits(counts)
    every = [list(c) for c in itertools.product(*[range(n) for n in counts])]
    n = len(every) if how_many is None else max(1, min(int(how_many), len(every)))
    if n > MAX_COMBOS:
        raise ValueError(f"at most {MAX_COMBOS} videos per Mix")
    last = len(counts) - 1
    weight = [3.0 if i in (0, last) else 1.0 for i in range(len(counts))]
    used = [[0] * c for c in counts]
    pairs: dict[tuple, int] = {}
    together: dict[tuple, int] = {}               # (section i, clip, section j, clip) used in the same video
    chosen: list[list[int]] = []
    left = list(every)
    for _ in range(n):
        prev = chosen[-1] if chosen else None

        def score(combo):
            s = sum(weight[i] * used[i][p] for i, p in enumerate(combo))
            s += 2.0 * pairs.get((combo[0], combo[last]), 0)
            s += 0.5 * sum(together.get((i, combo[i], j, combo[j]), 0)
                           for i in range(len(combo)) for j in range(i + 1, len(combo)))
            if prev is not None:
                if combo[0] == prev[0] and counts[0] > 1:
                    s += 4.0
                if combo[last] == prev[last] and counts[last] > 1:
                    s += 4.0
            return s

        best = min(left, key=lambda c: (score(c), every.index(c)))
        left.remove(best)
        chosen.append(best)
        for i, p in enumerate(best):
            used[i][p] += 1
        pairs[(best[0], best[last])] = pairs.get((best[0], best[last]), 0) + 1
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                together[(i, best[i], j, best[j])] = together.get((i, best[i], j, best[j]), 0) + 1
    return chosen


# ---------------------------------------------------------------- the mix_clip call (one for the whole Mix)

def mix_transcript(clips: list[dict]) -> tuple[str, list[list[list[int]]]]:
    """clips: [{"section": "Hook", "words": [...]}] -> text for the model + lines (word indexes) per clip."""
    out, all_units = [], []
    for n, clip in enumerate(clips, start=1):
        words = clip["words"]
        units = _engine.lines_of(words) if words else []
        all_units.append(units)
        out.append(f"=== CLIP {n} · part: {clip.get('section') or '?'} ===")
        out.append(_engine.transcript(words, units) if units else "(no speech)")
    return "\n".join(out), all_units


_ROW = {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["clips", "fix", "zoom"],
    "properties": {
        "clips": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["clip", "hide", "trim"],
            "properties": {"clip": {"type": "integer"}, "hide": {"type": "array", "items": _ROW},
                           "trim": {"type": "array", "items": _ROW}}}},
        "fix": {"type": "array", "items": _ROW},
        "zoom": {"type": "array", "items": _ROW},
    },
}


def model_name() -> str:
    return os.environ.get("CUTSELL_MIX_MODEL", "").strip() or DEFAULT_MODEL


def cost_usd(model: str, usage: dict) -> float:
    price = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    return (int(usage.get("input_tokens") or 0) * price["input"]
            + int(usage.get("output_tokens") or 0) * price["output"]) / 1e6


def call_anthropic(body: dict, *, attempts: int = 4) -> tuple[str, dict]:
    """One request (the body is built by clean_all). Never logs the prompt or the key."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(API_URL, json=body, headers=headers, timeout=(20, 600))
            if response.status_code == 200:
                data = response.json()
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                return text, dict(data.get("usage") or {})
            if response.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"Anthropic HTTP {response.status_code}")
            last = RuntimeError(f"Anthropic HTTP {response.status_code}")
        except requests.RequestException as exc:
            last = exc
        if attempt < attempts - 1:
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Anthropic call failed: {last}")


def _parse(text: str) -> dict:
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            obj = json.JSONDecoder().raw_decode(text[match.start():])[0]
        except ValueError:
            continue
        if isinstance(obj, dict) and "clips" in obj:
            return obj
    raise ValueError("no mix_clip JSON")


def ask(clips: list[dict], *, call: Optional[LLMCall] = None, model: Optional[str] = None,
        effort: str = "medium") -> tuple[dict, list[list[list[int]]], dict]:
    """The one AI call of a Mix. Returns (answer, lines per clip, usage {model, tokens, cost_usd, calls})."""
    text, units = mix_transcript(clips)
    model = model or model_name()
    body = {"model": model, "max_tokens": 16000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT + "\nCLIPS:\n"},
                                                      {"type": "text", "text": text}]}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}
    call = call or call_anthropic
    usage = {"model": model, "input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0}
    last = None
    for attempt in range(3):                       # an unreadable answer is asked again, never guessed
        try:
            answer_text, u = call(body)
            usage["calls"] += 1
            usage["input_tokens"] += int(u.get("input_tokens") or 0)
            usage["output_tokens"] += int(u.get("output_tokens") or 0)
            usage["cost_usd"] = round(cost_usd(model, usage), 6)
            return _parse(answer_text), units, usage
        except Exception as exc:
            last = exc
            if attempt == 1 and "format" in body.get("output_config", {}):
                body["output_config"].pop("format")   # the JSON format refused twice: free text (the parser reads it)
    raise RuntimeError(f"mix_clip answer unreadable: {last}")


def clip_labels(answer: dict, index: int, words: list[dict], units: list[list[int]]) -> list[tuple[str, str]]:
    """Labels for clip number index+1. A clip the answer would empty completely keeps everything (WHEN UNCERTAIN, KEEP)."""
    own = next((c for c in answer.get("clips") or [] if _int(c.get("clip")) == index + 1), None) or {}
    lab = _engine.labels({"hide": own.get("hide") or [], "trim": own.get("trim") or []}, words, units)
    if words and not any(st == "keep" for st, _ in lab):
        lab = [("keep", "se queda")] * len(words)
    return lab


def _int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_word(words, units, line, text):
    """Index of `text` inside line `line` (1-based) of a clip, or None."""
    if not (1 <= line <= len(units)):
        return None
    target = _engine._norm(str(text).split()[0]) if str(text).split() else ""
    for k in units[line - 1]:
        if _engine._norm(words[k]["w"]) == target:
            return k
    return None


def fixes_and_moments(answer: dict, index: int, words: list[dict], units, kept: set[int]):
    """Caption fixes [k, old, new] and zoom moments [{"i", "kind"}] of clip index+1, only on kept words."""
    fixes, moments = [], []
    for row in answer.get("fix") or []:
        if len(row) >= 4 and _int(row[0]) == index + 1:
            k = _find_word(words, units, _int(row[1]) or 0, row[2])
            new = str(row[3]).strip()
            if k is not None and k in kept and new and new != words[k]["w"] and len(new) <= 3 * len(words[k]["w"]) + 8:
                fixes.append([k, words[k]["w"], new])
    for row in answer.get("zoom") or []:
        if len(row) >= 4 and _int(row[0]) == index + 1:
            k = _find_word(words, units, _int(row[1]) or 0, row[2])
            kind = str(row[3]).strip().lower()
            if k is not None and k in kept and kind in ZOOM_KINDS and not moments:
                moments.append({"i": k, "kind": kind})
    return fixes, moments


def finish_clip(words: list[dict], lab, duration: float, audio_path: str) -> list[dict]:
    """Labels -> kept parts, with the cut engine's own finishing (loose pieces, silence refine, spoken words kept,
    natural breath). No shot-change snap: a clip is one take."""
    splits = _engine.splits_from_labels(words, [(st, why, False) for st, why in lab], duration)
    _engine.orphans(splits, words)
    if not any(s["visible"] for s in splits):          # never lose a whole clip to the loose-piece rule
        splits = _engine.splits_from_labels(words, [("keep", "se queda", False)] * len(words), duration)
    splits = _engine.keep_spoken_words(_engine.refine(splits, audio_path, duration, words), words)
    return _engine.breathe(splits, words, duration)


EDGE_PAD_SEC = 0.05              # silence left before the first / after the last word of a clean clip: ~0.1 s per join
EDGE_SEARCH_SEC = (0.30, 0.40)   # how far the voice may start before / end after its word's timestamps


def _frames_db(audio_path: str, rate: int = 16000, hop: float = 0.01):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio_path), "-vn", "-ac", "1", "-ar", str(rate),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    n = int(rate * hop)
    if len(x) < n:
        return np.zeros(0), hop
    frames = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-6), hop


def tighten_edges(parts: list[tuple[float, float]], words: list[dict], audio_path: str, *,
                  pad: float = EDGE_PAD_SEC) -> list[tuple[float, float]]:
    """Shorten the silence before the clip's first kept word and after its last one to `pad`, so a join between
    two clean clips is a ~0.1 s pause. The edge follows the VOICE, not only the word timestamps: the sound around a
    word is measured and the cut never goes inside it (a word's tail or a soft first consonant is kept). Only ever
    shortens; inner parts are untouched."""
    if not parts:
        return parts
    kept = [w for w in words if any(a <= (w["s"] + w["e"]) / 2 <= b for a, b in parts)]
    if not kept:
        return parts
    try:
        db, hop = _frames_db(audio_path)
    except Exception:
        return parts
    if len(db) < 10:
        return parts
    import numpy as np
    speech = float(np.percentile(db, 95))
    floor = float(np.percentile(db, 10))
    thr = min(max(speech - 30.0, floor + 8.0), speech - 6.0)    # no real silence in the clip: still finds the voice
    loud = lambda t: 0 <= int(t / hop) < len(db) and db[int(t / hop)] > thr
    first_s, last_e = float(kept[0]["s"]), float(kept[-1]["e"])
    onset = first_s
    while onset - hop >= max(0.0, first_s - EDGE_SEARCH_SEC[0]) and loud(onset - hop):
        onset -= hop
    offset = last_e
    while offset + hop <= last_e + EDGE_SEARCH_SEC[1] and loud(offset):
        offset += hop
    out = [list(p) for p in parts]
    # still voice at the end of the search: the edge is not where the words say -- leave that edge as it was
    if not loud(onset - hop) or onset - hop < 0:
        out[0][0] = max(out[0][0], min(onset - pad, first_s - pad))
    if not loud(offset):
        out[-1][1] = min(out[-1][1], max(offset + pad, last_e + pad))
    out = [(round(a, 3), round(b, 3)) for a, b in out if b - a > 0.05]
    return out or parts


def kept_parts(splits) -> list[tuple[float, float]]:
    return [(float(s["start"]), float(s["end"])) for s in splits if s["visible"] and s["end"] - s["start"] > 0.05]


def clip_captions(words: list[dict], parts: list[tuple[float, float]]) -> list[dict]:
    """Kept words with their times on the CLEAN clip (parts joined end to end, no overlap)."""
    out, t0 = [], 0.0
    for a, b in parts:
        for w in words:
            mid = (w["s"] + w["e"]) / 2
            if a <= mid <= b:
                out.append({"w": w["w"], "s": round(t0 + max(0.0, w["s"] - a), 3),
                            "e": round(t0 + min(b, w["e"]) - a, 3)})
        t0 += b - a
    return out


def clip_time(t: float, parts: list[tuple[float, float]]) -> Optional[float]:
    """A source time of the clip -> its time on the clean clip (None if not kept)."""
    t0 = 0.0
    for a, b in parts:
        if a <= t <= b:
            return round(t0 + t - a, 3)
        t0 += b - a
    return None


# ---------------------------------------------------------------- media (ffmpeg, no AI)

def probe(path: str) -> dict:
    raw = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(raw)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    rotation = 0
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(float(side["rotation"]))
    rotation = rotation or int(float((video.get("tags") or {}).get("rotate", 0) or 0))
    if abs(rotation) % 180 == 90:
        w, h = h, w
    return {"width": w, "height": h, "duration": float((data.get("format") or {}).get("duration") or 0.0),
            "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", []))}


def canvas_for(shapes: list[dict]) -> tuple[int, int]:
    """The Mix's frame: portrait unless most clips are landscape."""
    landscape = sum(1 for s in shapes if s["width"] > s["height"])
    return CANVAS_LANDSCAPE if landscape * 2 > len(shapes) else CANVAS_PORTRAIT


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)


def render_clean_clip(video_path: str, parts: list[tuple[float, float]], canvas: tuple[int, int], out_path: str,
                      *, has_audio: bool = True) -> dict:
    """The kept parts of one clip, on the Mix's canvas at 30 fps, 30 ms fades at every edge, at -16 LUFS.
    Every clean clip has the same encoding, so a combination is a plain join (video copied)."""
    w, h = canvas
    chains, labels = [], []
    for i, (a, b) in enumerate(parts):
        d = max(0.001, b - a)
        chains.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,"
                      f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                      f"setsar=1,fps={FPS},format=yuv420p[v{i}]")
        if has_audio:
            chains.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                          f"aformat=sample_rates=48000:channel_layouts=stereo,"
                          f"afade=t=in:d={FADE_SEC},afade=t=out:st={max(0.0, d - FADE_SEC):.3f}:d={FADE_SEC}[a{i}]")
        else:
            chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.3f}[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    chains.append(f"{''.join(labels)}concat=n={len(parts)}:v=1:a=1[v][a]")
    tmp = str(out_path) + ".raw.mp4"
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-filter_complex", ";".join(chains),
          "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high",
          "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", str(2 * FPS), "-video_track_timescale", "15360",
          "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-f", "mov", tmp])
    measured = loudness(tmp)
    if measured is None or measured["input_i"] < -70:          # silent clip: no gain to apply
        af = "anull"
    else:
        af = (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:measured_I={measured['input_i']}:"
              f"measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}:"
              f"measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true,"
              f"aresample=48000")
    _run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-map", "0:v", "-map", "0:a", "-c:v", "copy", "-af", af,
          "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", "-f", "mov", str(out_path)])
    Path(tmp).unlink(missing_ok=True)
    return {"path": str(out_path), "duration": probe(out_path)["duration"], "loudness_in": measured}


def loudness(path: str) -> Optional[dict]:
    """Integrated loudness (EBU R128, ffmpeg loudnorm first pass)."""
    r = subprocess.run(["ffmpeg", "-v", "info", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
                        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=600)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return None
    data = json.loads(m.group(0))
    try:
        return {k: float(data[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    except (KeyError, ValueError):
        return None


def join_clips(clean_paths: list[str], out_path: str) -> dict:
    """A combination: the clean clips one after the other. Video copied (same encoding everywhere); the sound is
    joined as PCM and encoded to AAC once, so no gap or click is added at a join."""
    listing = Path(str(out_path) + ".txt")
    listing.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in clean_paths))
    _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "copy",
          "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", str(out_path)])
    listing.unlink(missing_ok=True)
    return {"path": str(out_path), "duration": probe(out_path)["duration"]}


# ---------------------------------------------------------------- a combination's timeline

def combo_timeline(clips: list[dict]) -> dict:
    """clips (in order): [{"duration", "captions", "moments" [{"t", "kind"}], "section"}] ->
    parts [(start, end)], captions and zoom moments on the combination's timeline."""
    parts, caps, moments, t0 = [], [], [], 0.0
    for clip in clips:
        d = float(clip["duration"])
        parts.append((round(t0, 3), round(t0 + d, 3)))
        caps += [{"w": c["w"], "s": round(c["s"] + t0, 3), "e": round(min(c["e"], d) + t0, 3)} for c in clip["captions"]]
        moments += [{"s": round(m["t"] + t0, 3), "kind": m["kind"]} for m in clip.get("moments") or []]
        t0 += d
    return {"parts": parts, "captions": caps, "caption_groups": _engine.caption_groups(caps), "moments": moments,
            "duration": round(t0, 3)}


# ---------------------------------------------------------------- join check (QA, no AI)

def join_report(path: str, joins: list[float], *, window: float = 0.4) -> list[dict]:
    """Per join: loudness just before / after (dB RMS of the voiced part) and the largest sample jump within 5 ms
    (a click shows as a jump far above the signal around it)."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    sr = 48000
    out = []
    for t in joins:
        i = int(t * sr)
        before = x[max(0, i - int(window * sr)):i]
        after = x[i:i + int(window * sr)]

        def level(seg):
            if len(seg) == 0:
                return None
            frames = seg[: len(seg) // 480 * 480].reshape(-1, 480)
            rms = np.sqrt((frames ** 2).mean(1)) + 1e-9
            voiced = rms[rms > np.percentile(rms, 50)]
            return round(float(20 * np.log10(voiced.mean())), 1) if len(voiced) else None

        near = x[max(0, i - 240):i + 240]
        jump = float(np.abs(np.diff(near)).max()) if len(near) > 1 else 0.0
        context = x[max(0, i - sr):i + sr]
        typical = float(np.percentile(np.abs(np.diff(context)), 99)) if len(context) > 1 else 0.0
        out.append({"t": round(t, 3), "db_before": level(before), "db_after": level(after),
                    "max_jump": round(jump, 4), "typical_jump_p99": round(typical, 4),
                    "click": bool(jump > max(0.05, 3 * typical))})
    return out
