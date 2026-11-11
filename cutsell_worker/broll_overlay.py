"""B-roll over the voice (PO 9-10 Nov 2026).

The iPhone sends ONE project with several sources: the talking part (`metadata.clip_type = "primary"`, or no mark)
and N B-roll clips (`metadata.clip_type = "broll"`; gallery clips marked B-roll carry the same mark).

- B-roll is never cut: it never goes through speech-to-text, the cut decision, silence refine or the loose-piece
  rule, and it never enters the voice timeline (`selected` / `alternates`). The voice is edited exactly as before.
- B-roll goes ON TOP of the voice as `draft.media_overlays` items (kind "video", layout "full", fill, own sound
  muted), the representation the iOS editor already draws, moves, trims and deletes. The voice is never cut,
  shortened or moved because of B-roll.
- Matching (one extra small Claude call, `match_with_claude`): 3 frames of each B-roll clip + the kept phrases with
  their times -> which phrase each clip fits by meaning, which of its frames fits best, where the closing call to
  action starts. The answer is a SUGGESTION: every rule below is enforced in code (`plan_placements`).
- Rules: no B-roll in the first HOOK_SEC (face for the hook) nor over the closing call to action; each block lasts
  MIN_SEC..MAX_SEC and starts on a phrase start / ends on a phrase or word end; at most MAX_COVERAGE of the video
  is B-roll; each clip at most once, using its steadiest window (near the frame the model chose); a clip the
  model matched to no phrase goes where the product is mentioned, else the clips are spread evenly.
- Deterministic fallback: no answer / an invalid answer / CUTSELL_BROLL_AI=0 -> the same rules without the
  suggestions (product mentions, then even spread). Nothing here can fail the cut: `place_broll_overlays`
  catches everything and records why in the diagnostics.

Switches (names only): CUTSELL_BROLL_OVERLAY ("auto" default = on for a request that has B-roll sources; "0" = off:
every source is treated as a voice source, exactly as before this module), CUTSELL_BROLL_AI ("1" default; "0" = no
model call, deterministic placement only).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .contracts import MediaOverlay

_LOG = logging.getLogger("cutsell.broll_overlay")

CLIP_TYPE_KEY = "clip_type"
BROLL = "broll"

HOOK_SEC = 3.0            # the face shows for the hook
CTA_MIN_TAIL_SEC = 3.0    # the face shows for at least the last 3 s even when no call to action was found
MIN_SEC, MAX_SEC, TARGET_SEC = 2.0, 4.0, 3.0
MAX_COVERAGE = 0.5
PHRASE_GAP_SEC = 0.30     # a pause this long between two words starts a new phrase
FRAMES_PER_CLIP = 3
FRAME_LONG_SIDE = 512     # px: ~200 image tokens per frame
MOTION_FPS = 4
WINDOW_STEP_SEC = 0.25
OVERLAY_ID_PREFIX = "ai-broll-"

_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")


# ------------------------------------------------------------------ switches / detection

def overlay_mode(env: Mapping[str, str] | None = None) -> str:
    raw = str((env if env is not None else os.environ).get("CUTSELL_BROLL_OVERLAY", "auto")).strip().lower()
    return "off" if raw in ("0", "off", "false", "no") else "auto"


def ai_enabled(env: Mapping[str, str] | None = None) -> bool:
    return str((env if env is not None else os.environ).get("CUTSELL_BROLL_AI", "1")).strip() != "0"


def is_broll(source) -> bool:
    meta = getattr(source, "metadata", None)
    return isinstance(meta, Mapping) and str(meta.get(CLIP_TYPE_KEY) or "").strip().lower() == BROLL


def split_sources(sources: Sequence, env: Mapping[str, str] | None = None) -> tuple[list, list]:
    """(voice sources, B-roll sources). With the switch off, or when EVERY source is marked B-roll (there would be
    no voice to put them over), every source is a voice source, exactly as before."""
    sources = list(sources)
    if overlay_mode(env) == "off":
        return sources, []
    broll = [s for s in sources if is_broll(s)]
    voice = [s for s in sources if not is_broll(s)]
    if not broll or not voice:
        return sources, []
    return voice, broll


# ------------------------------------------------------------------ the voice timeline

@dataclass(frozen=True)
class Phrase:
    index: int
    start: float          # final-timeline seconds
    end: float
    text: str
    word_ends: tuple = ()


def _clip_length(clip) -> float:
    return max(0.0, float(clip.end) - float(clip.start)) / (float(getattr(clip, "speed", 1.0) or 1.0))


def timeline_duration(selected: Sequence) -> float:
    return sum(_clip_length(c) for c in selected)


def timeline_words(selected: Sequence) -> list[dict]:
    """Every kept word on the final timeline (the same sum-of-clips timeline the overlays live on)."""
    out: list[dict] = []
    offset = 0.0
    for clip in selected:
        speed = float(getattr(clip, "speed", 1.0) or 1.0)
        length = _clip_length(clip)
        for word in getattr(clip, "words", ()) or ():
            s = offset + (float(word.start) - float(clip.start)) / speed
            e = offset + (float(word.end) - float(clip.start)) / speed
            s, e = max(offset, s), min(offset + length, e)
            if e > s:
                out.append({"w": str(word.text), "s": round(s, 3), "e": round(e, 3), "clip": clip.clip_id})
        offset += length
    return out


def phrases_of(words: Sequence[dict]) -> list[Phrase]:
    """Phrases: split at sentence punctuation, at a pause of PHRASE_GAP_SEC or more, and at every clip join."""
    groups: list[list[dict]] = []
    for word in words:
        if groups:
            last = groups[-1][-1]
            if (_SENTENCE_END.search(last["w"]) or word["s"] - last["e"] >= PHRASE_GAP_SEC
                    or word.get("clip") != last.get("clip")):
                groups.append([word])
                continue
            groups[-1].append(word)
        else:
            groups.append([word])
    return [Phrase(index=i + 1, start=g[0]["s"], end=g[-1]["e"], text=" ".join(w["w"] for w in g),
                   word_ends=tuple(w["e"] for w in g)) for i, g in enumerate(groups)]


def source_time_to_timeline(selected: Sequence, source_asset_id: str, t: float) -> Optional[float]:
    offset = 0.0
    for clip in selected:
        length = _clip_length(clip)
        if clip.source_asset_id == source_asset_id and float(clip.start) <= t <= float(clip.end):
            return offset + (t - float(clip.start)) / float(getattr(clip, "speed", 1.0) or 1.0)
        offset += length
    return None


# ------------------------------------------------------------------ the B-roll clips (local media only)

def _ffmpeg_frame(path: str, at: float, out: str) -> bool:
    scale = f"scale='if(gt(iw,ih),{FRAME_LONG_SIDE},-2)':'if(gt(iw,ih),-2,{FRAME_LONG_SIDE})'"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, at):.3f}", "-i", path, "-frames:v", "1",
                        "-vf", scale, "-q:v", "5", out], check=True, capture_output=True, timeout=60)
        return Path(out).exists() and Path(out).stat().st_size > 0
    except Exception:
        return False


def frame_times(duration: float, count: int = FRAMES_PER_CLIP) -> list[float]:
    """Evenly inside the clip, never on its very first/last frame: 1/6, 3/6, 5/6 of it for three frames."""
    return [round(duration * (2 * i + 1) / (2 * count), 3) for i in range(count)] if duration > 0 else []


def sample_frames(path: str, duration: float, count: int = FRAMES_PER_CLIP) -> list[dict]:
    """[{"t": seconds, "jpeg_b64": ...}] for the frames that could be read (none for an audio-only file)."""
    out = []
    with tempfile.TemporaryDirectory(prefix="cutsell-broll-") as directory:
        for i, t in enumerate(frame_times(duration, count)):
            target = str(Path(directory) / f"f{i}.jpg")
            if _ffmpeg_frame(path, t, target):
                out.append({"t": t, "jpeg_b64": base64.b64encode(Path(target).read_bytes()).decode("ascii")})
    return out


def motion_profile(path: str, *, fps: int = MOTION_FPS) -> list[float]:
    """Picture change between consecutive tiny grey frames (0..255), one value per 1/fps s. [] when unreadable."""
    side = 32
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={side}:{side},format=gray",
                              "-f", "rawvideo", "-"], check=True, capture_output=True, timeout=120).stdout
    except Exception:
        return []
    size = side * side
    frames = [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]
    return [sum(abs(a - b) for a, b in zip(x, y)) / size for x, y in zip(frames, frames[1:])]


def best_window(duration: float, length: float, *, motion: Sequence[float] = (), near: Optional[float] = None,
                fps: int = MOTION_FPS) -> float:
    """In-point of the steadiest `length`-second window of the clip; when `near` is given (the frame the model picked)
    only windows containing it compete. Without motion data: centred on `near`, else the middle of the clip."""
    span = max(0.0, duration - length)
    if span <= 1e-6:
        return 0.0
    starts = [round(i * WINDOW_STEP_SEC, 3) for i in range(int(span / WINDOW_STEP_SEC) + 1)]
    if starts[-1] < span - 1e-6:
        starts.append(round(span, 3))
    if near is not None:
        containing = [s for s in starts if s <= near <= s + length]
        starts = containing or starts
    if not motion:
        centre = near if near is not None else duration / 2
        return min(starts, key=lambda s: abs(s + length / 2 - centre))

    def score(s: float) -> float:
        a, b = int(s * fps), max(int(s * fps) + 1, int((s + length) * fps))
        window = motion[a:b] or motion[-1:]
        return sum(window) / len(window)

    return min(starts, key=lambda s: (round(score(s), 6), abs(s + length / 2 - (near if near is not None else duration / 2))))


@dataclass
class BrollClip:
    source: object
    duration: float
    path: str = ""
    video: bool = True
    frames: list = field(default_factory=list)
    motion: list = field(default_factory=list)

    @property
    def source_asset_id(self) -> str:
        return str(self.source.source_asset_id)


# ------------------------------------------------------------------ the one extra model call

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {"type": "array", "items": {
            "type": "object",
            "properties": {"clip": {"type": "string"}, "shows": {"type": "string"},
                           "phrase": {"type": ["integer", "null"]}, "frame": {"type": ["integer", "null"]}},
            "required": ["clip", "shows", "phrase", "frame"], "additionalProperties": False}},
        "cta_phrase": {"type": ["integer", "null"]},
    },
    "required": ["clips", "cta_phrase"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You place B-roll over a short vertical selling video. The creator's voice is the main track and is \
never changed. Below are the kept phrases of her voice (number, time on the final video, words) and, for each B-roll \
clip, three frames of it.
For each B-roll clip:
- "shows": what the clip shows, in at most 12 words (in the language of the phrases);
- "phrase": the number of the ONE phrase whose meaning this clip illustrates best (e.g. "look at the texture" -> \
cream spread on a hand), or null when no phrase fits;
- "frame": 1, 2 or 3, the frame that fits that phrase best (null if none).
Never pick a phrase that starts in the first 3 seconds. Use each phrase for at most one clip.
"cta_phrase": the number of the phrase where the closing call to action starts (buy it, link, follow, comment...), \
or null when there is none.
Answer with JSON only: {"clips": [{"clip": "B1", "shows": "...", "phrase": 7, "frame": 2}], "cta_phrase": 12}"""


def build_content(phrases: Sequence[Phrase], clips: Sequence[BrollClip], *, product_hint: str = "") -> list[dict]:
    lines = "\n".join(f"P{p.index} [{p.start:.1f}-{p.end:.1f}s] {p.text}" for p in phrases)
    text = INSTRUCTIONS + (f"\nPRODUCT: {product_hint}" if product_hint else "") + "\n\nPHRASES:\n" + lines + "\n"
    content: list[dict] = [{"type": "text", "text": text}]
    for i, clip in enumerate(clips, start=1):
        content.append({"type": "text", "text": f"\nB{i} ({clip.duration:.1f} s): frames at "
                                                + ", ".join(f"{f['t']:.1f} s" for f in clip.frames)})
        for frame in clip.frames:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": frame["jpeg_b64"]}})
    return content


def match_with_claude(content: list[dict], *, post=None, max_tokens: int = 2000) -> tuple[str, dict]:
    """One Anthropic call (same model and key as the cut engine; effort low; JSON only). Never logs the prompt."""
    import requests
    from .simple_engine import llm as _llm

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    body = {"model": _llm.model_name(), "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    response = (post or requests.post)(_llm.API_URL, json=body, headers=headers, timeout=(20, 180))
    if response.status_code != 200:
        raise RuntimeError(f"Anthropic HTTP {response.status_code}")
    data = response.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = dict(data.get("usage") or {})
    usage["step"] = "broll_match"
    _LOG.info("broll_match_call %s", json.dumps({k: usage.get(k) for k in ("input_tokens", "output_tokens")}))
    return text, usage


def parse_answer(text: str, clip_count: int, phrase_count: int) -> dict:
    """{"by_clip": {0-based clip: {"phrase": n|None, "frame": 0-based|None, "shows": str}}, "cta_phrase": n|None}.
    Raises ValueError on an answer that is not that JSON."""
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON in the answer")
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        raise ValueError("answer has no clips list")

    def number(value, low, high):
        return value if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high else None

    by_clip: dict[int, dict] = {}
    for item in data["clips"]:
        if not isinstance(item, dict):
            continue
        match = re.fullmatch(r"\s*B?(\d+)\s*", str(item.get("clip", "")))
        index = int(match.group(1)) - 1 if match else -1
        if not 0 <= index < clip_count or index in by_clip:
            continue
        frame = number(item.get("frame"), 1, FRAMES_PER_CLIP)
        by_clip[index] = {"phrase": number(item.get("phrase"), 1, phrase_count),
                          "frame": frame - 1 if frame else None, "shows": str(item.get("shows") or "")[:120]}
    return {"by_clip": by_clip, "cta_phrase": number(data.get("cta_phrase"), 1, phrase_count)}


# ------------------------------------------------------------------ rules (pure)

@dataclass(frozen=True)
class Placement:
    clip_index: int
    start: float
    end: float
    source_start: float
    reason: str           # "matched" | "product" | "spread"
    phrase: Optional[int] = None


def _end_candidates(phrases: Sequence[Phrase]) -> list[float]:
    ends = set()
    for p in phrases:
        ends.add(round(p.end, 3))
        ends.update(round(e, 3) for e in p.word_ends)
    return sorted(ends)


def _inside_word(t: float, words: Sequence[dict]) -> bool:
    return any(w["s"] + 1e-3 < t < w["e"] - 1e-3 for w in words)


def _pick_end(start: float, limit: float, max_len: float, phrases: Sequence[Phrase], words: Sequence[dict]) -> Optional[float]:
    """An end in [start+MIN_SEC, start+max_len] and <= limit: a phrase end first, else a word end, else a moment
    that is not inside a word. Closest to TARGET_SEC."""
    low, high = start + MIN_SEC, min(start + max_len, limit)
    if high < low - 1e-6:
        return None
    target = start + TARGET_SEC
    phrase_ends = [p.end for p in phrases if low - 1e-6 <= p.end <= high + 1e-6]
    if phrase_ends:
        return round(min(phrase_ends, key=lambda e: abs(e - target)), 3)
    word_ends = [e for e in _end_candidates(phrases) if low - 1e-6 <= e <= high + 1e-6]
    if word_ends:
        return round(min(word_ends, key=lambda e: abs(e - target)), 3)
    for candidate in sorted({min(max(target, low), high), low, high}, key=lambda e: abs(e - target)):
        if not _inside_word(candidate, words):
            return round(candidate, 3)
    return None


def protected_tail_start(total: float, phrases: Sequence[Phrase], cta_times: Iterable[float] = (),
                         cta_phrase: Optional[int] = None) -> float:
    """Where the face must come back for the closing call to action: the start of the phrase holding the LAST call
    to action found (the model's cta_phrase or the engine's "cta" moments), never later than the last CTA_MIN_TAIL_SEC
    of the video nor than the start of the last phrase. A call to action found in the first half is not "closing"."""
    tail = total - CTA_MIN_TAIL_SEC
    if phrases:
        tail = min(tail, phrases[-1].start)
    candidates = [t for t in cta_times if t is not None and t >= total / 2]
    if cta_phrase is not None and 1 <= cta_phrase <= len(phrases) and phrases[cta_phrase - 1].start >= total / 2:
        candidates.append(phrases[cta_phrase - 1].start)
    for t in candidates:
        holder = next((p for p in phrases if p.start - 1e-6 <= t <= p.end + 1e-6), None)
        tail = min(tail, holder.start if holder else t)
    return max(0.0, round(tail, 3))


def plan_placements(
    *, total: float, phrases: Sequence[Phrase], words: Sequence[dict], clips: Sequence[BrollClip],
    suggestions: Optional[Mapping[int, Mapping]] = None, cta_start: float, product_times: Sequence[float] = (),
) -> tuple[list[Placement], dict[int, str]]:
    """Places each clip at most once, obeying every rule. Returns (placements by start, why each unplaced clip was
    not placed)."""
    suggestions = suggestions or {}
    budget = MAX_COVERAGE * total
    placed: list[Placement] = []
    used_phrases: set[int] = set()
    unplaced: dict[int, str] = {}
    lo, hi = HOOK_SEC, cta_start
    starts = [p for p in phrases if lo - 1e-6 <= p.start and p.start + MIN_SEC <= hi + 1e-6]

    def used() -> float:
        return sum(p.end - p.start for p in placed)

    def free(a: float, b: float) -> bool:
        return all(b <= p.start + 1e-6 or a >= p.end - 1e-6 for p in placed)

    def try_at(index: int, phrase: Phrase, reason: str, near_frame: Optional[float] = None) -> bool:
        clip = clips[index]
        room = min(MAX_SEC, clip.duration, budget - used())
        if room < MIN_SEC - 1e-6:
            return False
        nxt = min([p.start for p in placed if p.start >= phrase.start - 1e-6] + [hi])
        end = _pick_end(phrase.start, nxt, room, phrases, words)
        if end is None or not free(phrase.start, end):
            return False
        length = round(end - phrase.start, 3)
        source_start = best_window(clip.duration, length, motion=clip.motion, near=near_frame)
        placed.append(Placement(index, round(phrase.start, 3), end, round(source_start, 3), reason, phrase.index))
        placed.sort(key=lambda p: p.start)
        used_phrases.add(phrase.index)
        return True

    def frame_time(index: int) -> Optional[float]:
        frame = (suggestions.get(index) or {}).get("frame")
        frames = clips[index].frames
        return frames[frame]["t"] if frame is not None and 0 <= frame < len(frames) else None

    order = list(range(len(clips)))
    # 1. the model's matches, in timeline order
    matched = sorted((i for i in order if (suggestions.get(i) or {}).get("phrase")),
                     key=lambda i: suggestions[i]["phrase"])
    for i in matched:
        n = suggestions[i]["phrase"]
        phrase = phrases[n - 1] if 1 <= n <= len(phrases) else None
        if phrase is None or phrase not in starts or n in used_phrases:
            continue
        try_at(i, phrase, "matched", frame_time(i))
    rest = [i for i in order if all(p.clip_index != i for p in placed)]
    # 2. where the product is mentioned
    for t in sorted(product_times):
        if not rest:
            break
        candidates = [p for p in starts if p.index not in used_phrases and p.start - 1e-6 <= t <= p.end + 1e-6]
        candidates += [p for p in starts if p.index not in used_phrases and p.start >= t and p not in candidates][:1]
        for phrase in candidates:
            if try_at(rest[0], phrase, "product", frame_time(rest[0])):
                rest.pop(0)
                break
    # 3. spread evenly over what is left
    if rest and starts:
        span, count = hi - lo, len(rest)
        for k, i in enumerate(list(rest)):
            target = lo + span * (k + 0.5) / count
            for phrase in sorted((p for p in starts if p.index not in used_phrases), key=lambda p: abs(p.start - target)):
                if try_at(i, phrase, "spread", frame_time(i)):
                    rest.remove(i)
                    break
    for i in order:
        if all(p.clip_index != i for p in placed):
            clip = clips[i]
            unplaced[i] = ("clip_shorter_than_min" if clip.duration < MIN_SEC
                           else "coverage_limit" if budget - used() < MIN_SEC else "no_room")
    return placed, unplaced


def to_overlays(placements: Sequence[Placement], clips: Sequence[BrollClip]) -> tuple[MediaOverlay, ...]:
    out = []
    for p in placements:
        clip = clips[p.clip_index]
        length = round(p.end - p.start, 3)
        out.append(MediaOverlay(
            overlay_id=f"{OVERLAY_ID_PREFIX}{int(clip.source.source_order):03d}", kind="video", uri=str(clip.source.uri),
            start=p.start, end=p.end, x=0.5, y=0.5, width=1.0,
            source_start=p.source_start, source_end=round(min(clip.duration, p.source_start + length), 3),
            mute_audio=True, layout="full", fill=True,
        ))
    return tuple(out)


# ------------------------------------------------------------------ the whole step

def product_hint_of(sources: Sequence) -> str:
    """What she is promoting: the app's metadata brand / product (Promoting choice), else the Photos album name."""
    for source in sources:
        meta = getattr(source, "metadata", None)
        if not isinstance(meta, Mapping):
            continue
        named = " ".join(str(meta.get(k) or "").strip() for k in ("brand", "product")).strip()
        if named:
            return named[:80]
    for source in sources:
        meta = getattr(source, "metadata", None)
        if isinstance(meta, Mapping) and str(meta.get("album_hint") or "").strip():
            return str(meta["album_hint"]).strip()[:80]
    return ""


def _product_mentions(phrases: Sequence[Phrase], hint: str) -> list[float]:
    tokens = {t for t in re.findall(r"\w+", hint.lower()) if len(t) >= 3}
    if not tokens:
        return []
    return [p.start for p in phrases if tokens & set(re.findall(r"\w+", p.text.lower()))]


def place_broll_overlays(
    *, selected: Sequence, broll_sources: Sequence, local_paths: Mapping[str, str], durations: Mapping[str, float],
    voice_sources: Sequence = (), moments: Sequence[Mapping] = (), audio_only: Optional[set] = None,
    ask: Optional[Callable[[list], "tuple[str, dict]"]] = None,
    frames: Optional[Callable[[str, float], list]] = None, motion: Optional[Callable[[str], list]] = None,
) -> tuple[tuple[MediaOverlay, ...], dict]:
    """(media overlays, diagnostics). Never raises. `ask`, `frames`, `motion` are injectable for tests.
    `moments` = [{"source_asset_id", "s" (source time), "kind"}] from the cut engine (cta / product moments)."""
    diag: dict = {"mode": overlay_mode(), "ai": {"used": False}, "sources": []}
    try:
        total = timeline_duration(selected)
        words = timeline_words(selected)
        phrases = phrases_of(words)
        frames = frames or sample_frames
        motion = motion or motion_profile
        clips: list[BrollClip] = []
        for source in sorted(broll_sources, key=lambda s: int(s.source_order)):
            sid = str(source.source_asset_id)
            path = str(local_paths.get(sid) or "")
            video = bool(path) and not (audio_only and sid in audio_only)
            clip = BrollClip(source=source, duration=float(durations.get(sid) or source.duration_sec or 0.0),
                             path=path, video=video)
            if video:
                try:
                    clip.frames = list(frames(path, clip.duration) or [])
                except Exception:
                    clip.frames = []
                try:
                    clip.motion = list(motion(path) or [])
                except Exception:
                    clip.motion = []
            clips.append(clip)

        def timeline_of(kind: str) -> list[float]:
            out = []
            for m in moments:
                if str(m.get("kind")) == kind:
                    t = source_time_to_timeline(selected, str(m.get("source_asset_id")), float(m.get("s", -1)))
                    if t is not None:
                        out.append(round(t, 3))
            return out

        hint = product_hint_of(list(voice_sources) + list(broll_sources))
        suggestions: dict[int, dict] = {}
        cta_phrase = None
        with_frames = [i for i, c in enumerate(clips) if c.frames]
        if ai_enabled() and with_frames and phrases:
            asked = [clips[i] for i in with_frames]
            try:
                text, usage = (ask or match_with_claude)(build_content(phrases, asked, product_hint=hint))
                answer = parse_answer(text, len(asked), len(phrases))
                suggestions = {with_frames[k]: v for k, v in answer["by_clip"].items()}
                cta_phrase = answer["cta_phrase"]
                diag["ai"] = {"used": True, "usage": usage, "clips_asked": len(asked),
                              "frames_sent": sum(len(c.frames) for c in asked)}
            except Exception as exc:          # missing / invalid answer -> the deterministic rules alone
                diag["ai"] = {"used": False, "error": exc.__class__.__name__}
        elif not ai_enabled():
            diag["ai"] = {"used": False, "reason": "off"}
        else:
            diag["ai"] = {"used": False, "reason": "no_frames" if not with_frames else "no_phrases"}

        cta_start = protected_tail_start(total, phrases, timeline_of("cta"), cta_phrase)
        product_times = sorted(set(timeline_of("product")) | set(_product_mentions(phrases, hint)))
        placements, unplaced = plan_placements(total=total, phrases=phrases, words=words, clips=clips,
                                               suggestions=suggestions, cta_start=cta_start,
                                               product_times=product_times)
        overlays = to_overlays(placements, clips)
        by_clip = {p.clip_index: p for p in placements}
        for i, clip in enumerate(clips):
            p = by_clip.get(i)
            diag["sources"].append({
                "source_asset_id": clip.source_asset_id, "duration_sec": round(clip.duration, 3),
                "video_available": clip.video, "frames": len(clip.frames),
                "shows": (suggestions.get(i) or {}).get("shows", ""),
                "suggested_phrase": (suggestions.get(i) or {}).get("phrase"),
                "placed": p is not None,
                **({"start": p.start, "end": p.end, "source_start": p.source_start, "reason": p.reason,
                    "phrase": p.phrase} if p else {"why_not": unplaced.get(i, "no_room")}),
            })
        covered = sum(o.end - o.start for o in overlays)
        diag.update({"timeline_sec": round(total, 3), "hook_sec": HOOK_SEC, "cta_start": cta_start,
                     "coverage_sec": round(covered, 3), "coverage_ratio": round(covered / total, 4) if total else 0.0,
                     "product_hint": bool(hint), "placed": len(overlays)})
        return overlays, diag
    except Exception as exc:                  # never fail the cut because of B-roll
        _LOG.warning("broll_overlay_failed %s", exc.__class__.__name__)
        diag.update({"error": exc.__class__.__name__, "placed": 0})
        return (), diag
