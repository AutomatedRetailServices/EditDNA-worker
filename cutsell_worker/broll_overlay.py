"""B-roll over the voice (PO 9-10 Nov 2026; v2 11 Nov 2026).

The iPhone sends ONE project with several sources (all values strings, in the free-form `metadata` dict):
- `clip_type: "primary"` (or no mark): a talking video. There can be SEVERAL; they are joined in source_order and cut
  as one talking video (exactly like Combine: each is cut by the engine, the kept parts follow one another).
- `clip_type: "broll"`: a B-roll clip, `broll_kind: "video" | "photo"` (a photo is a JPEG source), `broll_order:
  "1".."N"` (the creator's order) and `broll_keep_order: "1" | "0"` (same on every B-roll source).

- B-roll is never cut: it never goes through speech-to-text, the cut decision, silence refine or the loose-piece
  rule, and it never enters the voice timeline (`selected` / `alternates`). The voice is edited exactly as before.
- B-roll goes ON TOP of the voice as `draft.media_overlays` items (layout "full", fill, own sound muted): a video clip
  is kind "video"; a photo is kind "photo" with `motion: "ken_burns"` (a slow centred zoom 1.0 -> 1.08, cropped to
  9:16), which the server export draws (media_overlay_render.ken_burns_chain).
- Matching (one extra small call to Claude HAIKU, `match_with_claude`, model `CUTSELL_BROLL_MATCH_MODEL`, read at call
  time): frames of each clip (3 for a clip under 10 s, else one every ~2.5 s, at most 10; one image per photo) + the
  kept phrases with their times -> which phrase(s) each clip fits by meaning and WHERE in the clip (seconds) that
  content is. A clip of 10 s or more may be used more than once, on different (never overlapping) parts of it. The
  answer is a SUGGESTION: every rule below is enforced in code (`plan_placements`). The cut's own model
  (CUTSELL_SIMPLE_ENGINE_MODEL) and calls are untouched.
- Rules: no B-roll in the first HOOK_SEC (face for the hook) nor over the closing call to action; each block lasts
  MIN_SEC..MAX_SEC and starts on a phrase start / ends on a phrase or word end; at most MAX_COVERAGE of the video
  is B-roll; a video piece is the steadiest MIN..MAX window around the second the model named.
- Order: `broll_keep_order "1"` -> the shots appear in the creator's order (1 before 2 before 3); the model's picks
  are kept when they respect it, otherwise the best order-preserving assignment is used (deterministic). "0" -> the
  model places freely (build-79 behaviour); a clip it matched to nothing goes where the product is mentioned, else
  the clips are spread evenly.
- Deterministic fallback: no answer / an invalid answer / CUTSELL_BROLL_AI=0 -> the same rules without the
  suggestions (product mentions, then even spread; in the creator's order when keep_order is on). Nothing here can
  fail the cut: `place_broll_overlays` catches everything and records why in the diagnostics.

Switches (names only): CUTSELL_BROLL_OVERLAY ("auto" default = on for a request that has B-roll sources; "0" = off:
every non-photo source is treated as a voice source, exactly as before this module), CUTSELL_BROLL_AI ("1" default;
"0" = no model call, deterministic placement only), CUTSELL_BROLL_MATCH_MODEL (default claude-haiku-5-5).
"""
from __future__ import annotations

import base64
import json
import logging
import math
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
BROLL_KIND_KEY, BROLL_ORDER_KEY, KEEP_ORDER_KEY = "broll_kind", "broll_order", "broll_keep_order"
PHOTO = "photo"
PHOTO_EXTENSIONS = (".jpg", ".jpeg")      # = uploads.ALLOWED_PHOTO_SOURCE_EXTENSIONS

HOOK_SEC = 3.0            # the face shows for the hook
CTA_MIN_TAIL_SEC = 3.0    # the face shows for at least the last 3 s even when no call to action was found
MIN_SEC, MAX_SEC, TARGET_SEC = 2.0, 4.0, 3.0
MAX_COVERAGE = 0.5
PHRASE_GAP_SEC = 0.30     # a pause this long between two words starts a new phrase
FRAMES_PER_CLIP = 3       # a clip under LONG_CLIP_SEC
LONG_CLIP_SEC = 10.0
LONG_FRAME_EVERY_SEC = 2.5
MAX_FRAMES_PER_CLIP = 10
MAX_USES_PER_CLIP = 3     # a long clip: different parts, never the same / an overlapping one
USE_EVERY_SEC = 5.0       # ... one use per 5 s of clip (10 s -> 2, 15 s+ -> 3)
SAME_CLIP_GAP_SEC = 4.0   # PO 11 Nov: between two pieces of the SAME clip, at least this much face / another shot
FRAME_LONG_SIDE = 512     # px: ~200 image tokens per frame
MOTION_FPS = 4
WINDOW_STEP_SEC = 0.25
OVERLAY_ID_PREFIX = "ai-broll-"
KEN_BURNS, KEN_BURNS_ZOOM_FROM, KEN_BURNS_ZOOM_TO = "ken_burns", 1.0, 1.08

MATCH_MODEL_ENV = "CUTSELL_BROLL_MATCH_MODEL"
DEFAULT_MATCH_MODEL = "claude-haiku-5-5"
# USD per million tokens. Haiku 5.5 and Sonnet 5.5 as in the repo's existing tables (post_captions.py,
# product_tagging.py, mix_engine.py, simple_engine/llm.py: "Anthropic price list"); not re-read from the live price
# page for this change. An unknown model is costed at the Sonnet price (never under-reported).
PRICES = {"claude-haiku-5-5": {"input": 0.10, "output": 0.50},
          "claude-sonnet-5-5": {"input": 2.00, "output": 10.00}}


def match_model(env: Mapping[str, str] | None = None) -> str:
    """The B-roll matcher's model, read at every call (a new value needs no redeploy)."""
    return str((env if env is not None else os.environ).get(MATCH_MODEL_ENV, "")).strip() or DEFAULT_MATCH_MODEL


def call_cost_usd(usage: Mapping, model: str) -> float:
    price = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    return round((int(usage.get("input_tokens") or 0) * price["input"]
                  + int(usage.get("output_tokens") or 0) * price["output"]) / 1e6, 6)


_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")


# ------------------------------------------------------------------ switches / detection

def overlay_mode(env: Mapping[str, str] | None = None) -> str:
    raw = str((env if env is not None else os.environ).get("CUTSELL_BROLL_OVERLAY", "auto")).strip().lower()
    return "off" if raw in ("0", "off", "false", "no") else "auto"


def ai_enabled(env: Mapping[str, str] | None = None) -> bool:
    return str((env if env is not None else os.environ).get("CUTSELL_BROLL_AI", "1")).strip() != "0"


def _meta(source) -> Mapping:
    meta = getattr(source, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def is_broll(source) -> bool:
    return str(_meta(source).get(CLIP_TYPE_KEY) or "").strip().lower() == BROLL


def is_photo_source(source) -> bool:
    """A B-roll photo: marked B-roll AND a JPEG source (the only photo type the source upload accepts). The
    `broll_kind` mark alone does not make a video file a photo (the overlay kind must match the stored object)."""
    if not is_broll(source):
        return False
    names = (str(getattr(source, "uri", "") or ""), str(getattr(source, "original_name", "") or ""))
    return any(Path(n).suffix.lower() in PHOTO_EXTENSIONS for n in names if n)


def broll_order_of(source) -> Optional[int]:
    try:
        value = int(str(_meta(source).get(BROLL_ORDER_KEY, "")).strip())
    except ValueError:
        return None
    return value if value >= 1 else None


def keep_order_of(sources: Sequence) -> bool:
    """broll_keep_order "1" on the B-roll sources (the app sends the same value on each; any "1" turns it on)."""
    return any(str(_meta(s).get(KEEP_ORDER_KEY, "")).strip() == "1" for s in sources if is_broll(s))


def ordered_broll(sources: Sequence) -> list:
    """B-roll in the creator's order (broll_order 1..N), then source_order for any clip without a valid number."""
    return sorted(sources, key=lambda s: (broll_order_of(s) is None, broll_order_of(s) or 0, int(s.source_order)))


def split_sources(sources: Sequence, env: Mapping[str, str] | None = None) -> tuple[list, list]:
    """(voice sources, B-roll sources). A photo is never a voice source (never sent to speech-to-text). With the
    switch off, or when EVERY source is marked B-roll (there would be no voice to put them over), every non-photo
    source is a voice source, exactly as before (a request without photos is unchanged)."""
    sources = list(sources)
    no_photos = [s for s in sources if not is_photo_source(s)]
    if overlay_mode(env) == "off":
        return no_photos, []
    broll = [s for s in sources if is_broll(s)]
    voice = [s for s in sources if not is_broll(s)]
    if not broll or not voice:
        return no_photos, []
    return voice, broll


# ------------------------------------------------------------------ B-roll still uploading (PO 11 Nov 2026)

UPLOAD_WAIT_ENV = "CUTSELL_BROLL_UPLOAD_WAIT_SEC"
DEFAULT_UPLOAD_WAIT_SEC = 60.0
MAX_UPLOAD_WAIT_SEC = 300.0
UPLOAD_POLL_SEC = 3.0


class UploadRejected(Exception):
    """The file arrived but can never be used (unreadable photo, format gate, too long): stop polling for it."""


def _now() -> float:            # module-level so tests swap in a fake clock (no real waiting in tests)
    import time
    return time.monotonic()


def _pause(seconds: float) -> None:
    import time
    time.sleep(seconds)


def upload_wait_sec(env: Mapping[str, str] | None = None) -> float:
    """Total seconds a cut may wait for ALL its still-uploading B-roll files together (0 = never wait)."""
    raw = str((env if env is not None else os.environ).get(UPLOAD_WAIT_ENV, "")).strip()
    try:
        value = float(raw) if raw else DEFAULT_UPLOAD_WAIT_SEC
    except ValueError:
        value = DEFAULT_UPLOAD_WAIT_SEC
    return max(0.0, min(MAX_UPLOAD_WAIT_SEC, value))


def wait_for_uploads(fetchers: Mapping[str, Callable[[], Optional[str]]], *, cap: float,
                     interval: float = UPLOAD_POLL_SEC, clock: Optional[Callable[[], float]] = None,
                     sleep: Optional[Callable[[float], None]] = None) -> tuple[dict, dict]:
    """Polls every still-missing file at once, then every `interval` s, until all are here or `cap` seconds have
    passed IN TOTAL (one budget shared by all the job's files, not per file). A fetcher returns the local path when
    the file is here, None (or raises) while it is not, and raises UploadRejected when it is here but unusable.
    -> ({source id: local path}, diagnostics). Never raises."""
    clock = clock or _now
    sleep = sleep or _pause
    started = clock()
    arrived: dict[str, str] = {}
    rejected: list[str] = []
    waiting = sorted(fetchers)
    polls = 0
    while waiting:
        polls += 1
        for sid in list(waiting):
            try:
                path = fetchers[sid]()
            except UploadRejected:
                rejected.append(sid)
                waiting.remove(sid)
                continue
            except Exception:
                path = None
            if path:
                arrived[sid] = path
                waiting.remove(sid)
        left = cap - (clock() - started)
        if not waiting or left <= 1e-9:
            break
        sleep(min(interval, left))
    return arrived, {"cap_sec": round(cap, 3), "waited_sec": round(clock() - started, 3), "polls": polls,
                     "arrived": sorted(arrived), "rejected": sorted(rejected), "still_missing": sorted(waiting)}


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

def _ffmpeg_frame(path: str, at: Optional[float], out: str) -> bool:
    """One frame at `at` seconds; `at` None = the first frame without seeking (a still photo: seeking drops it)."""
    scale = f"scale='if(gt(iw,ih),{FRAME_LONG_SIDE},-2)':'if(gt(iw,ih),-2,{FRAME_LONG_SIDE})'"
    seek = [] if at is None else ["-ss", f"{max(0.0, at):.3f}"]
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", *seek, "-i", path, "-frames:v", "1",
                        "-vf", scale, "-q:v", "5", out], check=True, capture_output=True, timeout=60)
        return Path(out).exists() and Path(out).stat().st_size > 0
    except Exception:
        return False


def frame_count(duration: float) -> int:
    """3 frames for a clip under LONG_CLIP_SEC; longer: one every ~LONG_FRAME_EVERY_SEC, at most MAX_FRAMES_PER_CLIP."""
    if duration <= 0:
        return 0
    if duration < LONG_CLIP_SEC:
        return FRAMES_PER_CLIP
    return max(FRAMES_PER_CLIP, min(MAX_FRAMES_PER_CLIP, int(round(duration / LONG_FRAME_EVERY_SEC))))


def frame_times(duration: float, count: Optional[int] = None) -> list[float]:
    """Evenly inside the clip, never on its very first/last frame: 1/6, 3/6, 5/6 of it for three frames; for a long
    clip the centres of equal slices (20 s -> 8 frames at 1.25, 3.75, ... 18.75 s)."""
    count = frame_count(duration) if count is None else count
    return [round(duration * (2 * i + 1) / (2 * count), 3) for i in range(count)] if duration > 0 and count else []


def sample_frames(path: str, duration: float, count: Optional[int] = None) -> list[dict]:
    """[{"t": seconds, "jpeg_b64": ...}] for the frames that could be read (none for an audio-only file)."""
    out = []
    with tempfile.TemporaryDirectory(prefix="cutsell-broll-") as directory:
        for i, t in enumerate(frame_times(duration, count)):
            target = str(Path(directory) / f"f{i}.jpg")
            if _ffmpeg_frame(path, t, target):
                out.append({"t": t, "jpeg_b64": base64.b64encode(Path(target).read_bytes()).decode("ascii")})
    return out


def sample_photo(path: str) -> list[dict]:
    """One photo = one image for the model, scaled to FRAME_LONG_SIDE on its long side."""
    with tempfile.TemporaryDirectory(prefix="cutsell-broll-") as directory:
        target = str(Path(directory) / "photo.jpg")
        if _ffmpeg_frame(path, None, target):
            return [{"t": 0.0, "jpeg_b64": base64.b64encode(Path(target).read_bytes()).decode("ascii")}]
    return []


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
                fps: int = MOTION_FPS, exclude: Sequence[tuple] = ()) -> Optional[float]:
    """In-point of the steadiest `length`-second window of the clip; when `near` is given (the second the model named)
    only windows containing it compete. Without motion data: centred on `near`, else the middle of the clip.
    `exclude` = (in, out) parts of the clip already used: a window overlapping one never competes; None when no
    window is left (with `near`: when every window containing it overlaps a used part)."""
    span = max(0.0, duration - length)
    if span <= 1e-6:
        starts = [0.0]
    else:
        starts = [round(i * WINDOW_STEP_SEC, 3) for i in range(int(span / WINDOW_STEP_SEC) + 1)]
        if starts[-1] < span - 1e-6:
            starts.append(round(span, 3))
    if near is not None:
        containing = [s for s in starts if s - 1e-6 <= near <= s + length + 1e-6]
        starts = containing or starts
    starts = [s for s in starts if all(s + length <= a + 1e-6 or s >= b - 1e-6 for a, b in exclude)]
    if not starts:
        return None
    if len(starts) == 1:
        return starts[0]
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
    video: bool = True              # its picture can be looked at
    frames: list = field(default_factory=list)
    motion: list = field(default_factory=list)
    kind: str = "video"             # "video" | "photo"
    order: int = 0                  # the creator's order (1..N), 0 = unknown

    @property
    def source_asset_id(self) -> str:
        return str(self.source.source_asset_id)

    @property
    def is_photo(self) -> bool:
        return self.kind == PHOTO

    @property
    def max_uses(self) -> int:
        """A photo or a short clip once; a clip of LONG_CLIP_SEC or more once per USE_EVERY_SEC (at most 3)."""
        if self.is_photo or self.duration < LONG_CLIP_SEC:
            return 1
        return max(1, min(MAX_USES_PER_CLIP, int(self.duration // USE_EVERY_SEC)))


# ------------------------------------------------------------------ the one extra model call (Haiku)

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "clip": {"type": "string"}, "shows": {"type": "string"},
                "uses": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"phrase": {"type": "integer"}, "at": {"type": ["number", "null"]}},
                    "required": ["phrase", "at"], "additionalProperties": False}},
            },
            "required": ["clip", "shows", "uses"], "additionalProperties": False}},
        "cta_phrase": {"type": ["integer", "null"]},
    },
    "required": ["clips", "cta_phrase"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You place B-roll over a short vertical selling video. The creator's voice is the main track and is \
never changed. Below are the kept phrases of her voice (number, time on the final video, words) and, for each B-roll \
clip, frames of it with the second of the clip each frame is at (a PHOTO is one image).
For each B-roll clip:
- "shows": what the clip shows, in at most 12 words (in the language of the phrases);
- "uses": the phrase(s) whose meaning this clip illustrates best (e.g. "look at the texture" -> cream spread on a \
hand), each as {"phrase": number, "at": the second of the clip where that content is (read it from the frame times; \
null for a photo)}. One use per clip, except a clip marked "may be used N times": up to N uses, each on a DIFFERENT \
part of the clip (at least 4 s apart) and a different phrase. An empty list when no phrase fits.
Never pick a phrase that starts in the first 3 seconds. Use each phrase for at most one clip.
"cta_phrase": the number of the phrase where the closing call to action starts (buy it, link, follow, comment...), \
or null when there is none.
Answer with JSON only: {"clips": [{"clip": "B1", "shows": "...", "uses": [{"phrase": 7, "at": 2.5}]}], \
"cta_phrase": 12}"""

KEEP_ORDER_NOTE = ("\nORDER: the creator wants the clips shown in the order B1, B2, B3...: B1's phrase must come "
                   "before B2's, B2's before B3's, and so on.")


def build_content(phrases: Sequence[Phrase], clips: Sequence[BrollClip], *, product_hint: str = "",
                  keep_order: bool = False) -> list[dict]:
    lines = "\n".join(f"P{p.index} [{p.start:.1f}-{p.end:.1f}s] {p.text}" for p in phrases)
    text = (INSTRUCTIONS + (KEEP_ORDER_NOTE if keep_order else "")
            + (f"\nPRODUCT: {product_hint}" if product_hint else "") + "\n\nPHRASES:\n" + lines + "\n")
    content: list[dict] = [{"type": "text", "text": text}]
    for i, clip in enumerate(clips, start=1):
        if clip.is_photo:
            label = f"\nB{i}: PHOTO"
        else:
            label = (f"\nB{i} ({clip.duration:.1f} s"
                     + (f", may be used {clip.max_uses} times" if clip.max_uses > 1 else "") + "): frames at "
                     + ", ".join(f"{f['t']:.1f} s" for f in clip.frames))
        content.append({"type": "text", "text": label})
        for frame in clip.frames:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": frame["jpeg_b64"]}})
    return content


def match_with_claude(content: list[dict], *, post=None, max_tokens: int = 2000) -> tuple[str, dict]:
    """One Anthropic call with the B-roll matcher's own model (CUTSELL_BROLL_MATCH_MODEL, default Haiku, read now);
    effort low; JSON only. Same key and endpoint as the cut engine. Never logs the prompt."""
    import requests
    from .simple_engine import llm as _llm

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    model = match_model()
    body = {"model": model, "max_tokens": max_tokens,
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
    usage["model"] = model
    usage["cost_usd"] = call_cost_usd(usage, model)
    _LOG.info("broll_match_call %s", json.dumps({k: usage.get(k) for k in ("model", "input_tokens", "output_tokens")}))
    return text, usage


def parse_answer(text: str, clips: Sequence[BrollClip], phrase_count: int) -> dict:
    """{"by_clip": {0-based clip: {"uses": [{"phrase": n, "at": seconds|None}], "shows": str}}, "cta_phrase": n|None}.
    Keeps at most clip.max_uses uses per clip, each a different phrase; `at` only inside the clip (None for a photo).
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
        if not 0 <= index < len(clips) or index in by_clip:
            continue
        clip = clips[index]
        uses: list[dict] = []
        for use in item.get("uses") if isinstance(item.get("uses"), list) else ():
            if not isinstance(use, dict):
                continue
            phrase = number(use.get("phrase"), 1, phrase_count)
            if phrase is None or any(u["phrase"] == phrase for u in uses):
                continue
            at = use.get("at")
            at = (float(at) if isinstance(at, (int, float)) and not isinstance(at, bool)
                  and not clip.is_photo and 0.0 <= float(at) <= clip.duration else None)
            uses.append({"phrase": phrase, "at": at})
            if len(uses) >= clip.max_uses:
                break
        by_clip[index] = {"uses": uses, "shows": str(item.get("shows") or "")[:120]}
    return {"by_clip": by_clip, "cta_phrase": number(data.get("cta_phrase"), 1, phrase_count)}


# ------------------------------------------------------------------ rules (pure)

@dataclass(frozen=True)
class Placement:
    clip_index: int
    start: float
    end: float
    source_start: float
    reason: str           # "matched" | "product" | "spread" | "reordered" (keep_order moved the model's pick)
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


def best_ordered_assignment(requests: Sequence[Mapping], starts: Sequence[Phrase], *, lo: float, hi: float,
                            total: float, phrases: Sequence[Phrase]) -> list[tuple[int, int]]:
    """keep_order: [(request position, position in `starts`)] with strictly increasing phrase times (each next shot
    starts at least MIN_SEC after the previous one), choosing the most requests, then the lowest cost: a request the
    model matched to phrase q costs 0 on q and 1 + distance elsewhere (so every order-respecting pick of the model is
    kept), a request without a pick costs its distance to its even-spread target (< 1). Deterministic."""
    m, n = len(requests), len(starts)
    if not m or not n:
        return []
    span = max(hi - lo, 1e-6)
    scale = max(total, 1e-6)

    def cost(i: int, j: int) -> float:
        pref = requests[i].get("phrase")
        here = starts[j]
        if pref is not None and 1 <= pref <= len(phrases):
            return 0.0 if here.index == pref else 1.0 + abs(here.start - phrases[pref - 1].start) / scale
        target = lo + span * (i + 0.5) / m
        return 0.5 * abs(here.start - target) / span

    memo: dict[tuple, tuple] = {}

    def best(i: int, last: int, last_clip=None) -> tuple:
        """(-placed count, cost, choices) for requests i.. after the shot at starts[last] (-1 = none yet) of clip
        `last_clip`. A next piece of the SAME clip starts at least MIN_SEC + SAME_CLIP_GAP_SEC later (its first piece
        lasts >= MIN_SEC, then >= SAME_CLIP_GAP_SEC of face)."""
        if i == m:
            return (0, 0.0, ())
        key = (i, last, last_clip)
        if key in memo:
            return memo[key]
        result = best(i + 1, last, last_clip)                            # skip this request
        gap = MIN_SEC + (SAME_CLIP_GAP_SEC if last_clip is not None and requests[i].get("clip") == last_clip else 0.0)
        floor = starts[last].start + gap - 1e-6 if last >= 0 else -1.0
        for j in range(last + 1, n):
            if starts[j].start < floor:
                continue
            count, total_cost, choices = best(i + 1, j, requests[i].get("clip"))
            option = (count - 1, round(total_cost + cost(i, j), 9), ((i, j),) + choices)
            if option[:2] < result[:2]:
                result = option
        memo[key] = result
        return result

    return list(best(0, -1)[2])


def plan_placements(
    *, total: float, phrases: Sequence[Phrase], words: Sequence[dict], clips: Sequence[BrollClip],
    suggestions: Optional[Mapping[int, Mapping]] = None, cta_start: float, product_times: Sequence[float] = (),
    keep_order: bool = False,
) -> tuple[list[Placement], dict[int, str]]:
    """Places the clips obeying every rule. Returns (placements by start, why each unplaced clip was not placed).
    keep_order False (build-79): the model's matches, then product mentions, then an even spread; a long clip may get
    several of its model-given pieces. keep_order True: the shots in clip order (`clip.order`, then list order),
    best_ordered_assignment over the model's picks (or an even spread in order without them)."""
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

    def usable(index: int) -> bool:
        clip = clips[index]
        return clip.is_photo or clip.duration >= MIN_SEC - 1e-6

    def try_at(index: int, phrase: Phrase, reason: str, near: Optional[float] = None, *,
               limit: Optional[float] = None, reserve: float = 0.0) -> bool:
        clip = clips[index]
        if not usable(index):
            return False
        room = min(MAX_SEC, MAX_SEC if clip.is_photo else clip.duration, budget - used() - reserve)
        if room < MIN_SEC - 1e-6:
            return False
        same = [p for p in placed if p.clip_index == index]
        # PO 11 Nov: two pieces of the same clip never closer than SAME_CLIP_GAP_SEC (end of one to start of the next)
        if any(p.end - 1e-6 < phrase.start < p.end + SAME_CLIP_GAP_SEC - 1e-6 for p in same):
            return False
        nxt = min([p.start for p in placed if p.start >= phrase.start - 1e-6] + [hi] + ([limit] if limit else [])
                  + [p.start - SAME_CLIP_GAP_SEC for p in same if p.start >= phrase.start - 1e-6])
        end = _pick_end(phrase.start, nxt, room, phrases, words)
        if end is None or not free(phrase.start, end):
            return False
        if any(not (phrase.start >= p.end + SAME_CLIP_GAP_SEC - 1e-6 or end <= p.start - SAME_CLIP_GAP_SEC + 1e-6)
               for p in same):
            return False                      # never violated: the piece is dropped
        length = round(end - phrase.start, 3)
        if clip.is_photo:
            source_start = 0.0
        else:
            taken = [(p.source_start, p.source_start + (p.end - p.start)) for p in placed if p.clip_index == index]
            source_start = best_window(clip.duration, length, motion=clip.motion, near=near, exclude=taken)
            if source_start is None:
                return False                  # never the same / an overlapping part of a clip twice
        placed.append(Placement(index, round(phrase.start, 3), end, round(source_start, 3), reason, phrase.index))
        placed.sort(key=lambda p: p.start)
        used_phrases.add(phrase.index)
        return True

    def uses_of(index: int) -> list[dict]:
        return list((suggestions.get(index) or {}).get("uses") or [])

    def first_at(index: int) -> Optional[float]:
        return next((u.get("at") for u in uses_of(index) if u.get("at") is not None), None)

    order = list(range(len(clips)))
    if keep_order:
        sequence = sorted(order, key=lambda i: (clips[i].order <= 0, clips[i].order, i))
        requests: list[dict] = []
        for i in sequence:
            if not usable(i):
                continue
            uses = sorted(uses_of(i), key=lambda u: u["phrase"]) or [{"phrase": None, "at": None}]
            requests.extend({"clip": i, "phrase": u["phrase"], "at": u.get("at"), "first": k == 0}
                            for k, u in enumerate(uses))
        cap = int((budget + 1e-6) // MIN_SEC)
        if len(requests) > cap:              # first pieces of the clips first, in order; extra pieces of long clips last
            keep = [r for r in requests if r["first"]][:cap]
            keep += [r for r in requests if not r["first"]][:max(0, cap - len(keep))]
            requests = [r for r in requests if any(r is k for k in keep)]
        chosen = best_ordered_assignment(requests, starts, lo=lo, hi=hi, total=total, phrases=phrases)
        for position, (i, j) in enumerate(chosen):
            request, phrase = requests[i], starts[j]
            following = starts[chosen[position + 1][1]].start if position + 1 < len(chosen) else None
            if following is not None and requests[chosen[position + 1][0]]["clip"] == request["clip"]:
                following -= SAME_CLIP_GAP_SEC   # piece A, then >= 4 s of face, then piece B of the same clip
            reason = ("matched" if request["phrase"] == phrase.index
                      else "reordered" if request["phrase"] is not None else "spread")
            near = request["at"] if request["phrase"] is not None else None
            try_at(request["clip"], phrase, reason, near, limit=following,
                   reserve=MIN_SEC * (len(chosen) - position - 1))
    else:
        # 1. the model's matches, in timeline order (several pieces of one long clip allowed, never overlapping)
        matched = sorted(((u["phrase"], i, u.get("at")) for i in order for u in uses_of(i)),
                         key=lambda item: (item[0], item[1]))
        for n, i, at in matched:
            phrase = phrases[n - 1] if 1 <= n <= len(phrases) else None
            if phrase is None or phrase not in starts or n in used_phrases:
                continue
            try_at(i, phrase, "matched", at)
        rest = [i for i in order if all(p.clip_index != i for p in placed)]
        # 2. where the product is mentioned
        for t in sorted(product_times):
            if not rest:
                break
            candidates = [p for p in starts if p.index not in used_phrases and p.start - 1e-6 <= t <= p.end + 1e-6]
            candidates += [p for p in starts if p.index not in used_phrases and p.start >= t and p not in candidates][:1]
            for phrase in candidates:
                if try_at(rest[0], phrase, "product", first_at(rest[0])):
                    rest.pop(0)
                    break
        # 3. spread evenly over what is left
        if rest and starts:
            span, count = hi - lo, len(rest)
            for k, i in enumerate(list(rest)):
                target = lo + span * (k + 0.5) / count
                for phrase in sorted((p for p in starts if p.index not in used_phrases),
                                     key=lambda p: abs(p.start - target)):
                    if try_at(i, phrase, "spread", first_at(i)):
                        rest.remove(i)
                        break
    for i in order:
        if all(p.clip_index != i for p in placed):
            unplaced[i] = ("clip_shorter_than_min" if not usable(i)
                           else "coverage_limit" if budget - used() < MIN_SEC else "no_room")
    return placed, unplaced


def to_overlays(placements: Sequence[Placement], clips: Sequence[BrollClip]) -> tuple[MediaOverlay, ...]:
    """A video piece: kind "video", its in/out inside the clip. A photo: kind "photo", no trim, motion Ken Burns.
    The first piece of a clip keeps the id `ai-broll-<source_order>`; a second piece of a long clip is `...-2`."""
    out = []
    seen: dict[int, int] = {}
    for p in placements:
        clip = clips[p.clip_index]
        seen[p.clip_index] = seen.get(p.clip_index, 0) + 1
        overlay_id = f"{OVERLAY_ID_PREFIX}{int(clip.source.source_order):03d}"
        if seen[p.clip_index] > 1:
            overlay_id += f"-{seen[p.clip_index]}"
        length = round(p.end - p.start, 3)
        common = dict(overlay_id=overlay_id, uri=str(clip.source.uri), start=p.start, end=p.end, x=0.5, y=0.5,
                      width=1.0, mute_audio=True, layout="full", fill=True)
        if clip.is_photo:
            out.append(MediaOverlay(kind="photo", source_start=0.0, source_end=None, motion=KEN_BURNS,
                                    motion_zoom_from=KEN_BURNS_ZOOM_FROM, motion_zoom_to=KEN_BURNS_ZOOM_TO, **common))
        else:
            out.append(MediaOverlay(kind="video", source_start=p.source_start,
                                    source_end=round(min(clip.duration, p.source_start + length), 3), **common))
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
    photo_frames: Optional[Callable[[str], list]] = None,
) -> tuple[tuple[MediaOverlay, ...], dict]:
    """(media overlays, diagnostics). Never raises. `ask`, `frames`, `motion`, `photo_frames` are injectable for
    tests. `moments` = [{"source_asset_id", "s" (source time), "kind"}] from the cut engine (cta / product moments)."""
    diag: dict = {"mode": overlay_mode(), "ai": {"used": False}, "sources": []}
    try:
        total = timeline_duration(selected)
        words = timeline_words(selected)
        phrases = phrases_of(words)
        frames = frames or sample_frames
        motion = motion or motion_profile
        photo_frames = photo_frames or sample_photo
        keep_order = keep_order_of(broll_sources)
        clips: list[BrollClip] = []
        for position, source in enumerate(ordered_broll(broll_sources), start=1):
            sid = str(source.source_asset_id)
            path = str(local_paths.get(sid) or "")
            photo = is_photo_source(source)
            video = bool(path) and not (audio_only and sid in audio_only)
            clip = BrollClip(source=source, duration=0.0 if photo else float(durations.get(sid) or source.duration_sec or 0.0),
                             path=path, video=video, kind=PHOTO if photo else "video",
                             order=broll_order_of(source) or position)
            if video:
                try:
                    clip.frames = list((photo_frames(path) if photo else frames(path, clip.duration)) or [])
                except Exception:
                    clip.frames = []
                if not photo:
                    try:
                        clip.motion = list(motion(path) or [])
                    except Exception:
                        clip.motion = []
            if photo and video and not clip.frames:
                continue                     # a photo that is here but cannot be read is never shown (recorded below)
            clips.append(clip)
        unreadable = [str(s.source_asset_id) for s in broll_sources
                      if is_photo_source(s) and all(c.source is not s for c in clips)]

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
                text, usage = (ask or match_with_claude)(build_content(phrases, asked, product_hint=hint,
                                                                       keep_order=keep_order))
                answer = parse_answer(text, asked, len(phrases))
                suggestions = {with_frames[k]: v for k, v in answer["by_clip"].items()}
                cta_phrase = answer["cta_phrase"]
                diag["ai"] = {"used": True, "usage": usage, "model": usage.get("model") or match_model(),
                              "cost_usd": usage.get("cost_usd", call_cost_usd(usage, usage.get("model") or match_model())),
                              "clips_asked": len(asked), "frames_sent": sum(len(c.frames) for c in asked)}
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
                                               product_times=product_times, keep_order=keep_order)
        overlays = to_overlays(placements, clips)
        for i, clip in enumerate(clips):
            pieces = [p for p in placements if p.clip_index == i]
            first = pieces[0] if pieces else None
            diag["sources"].append({
                "source_asset_id": clip.source_asset_id, "kind": clip.kind, "order": clip.order,
                "duration_sec": round(clip.duration, 3), "video_available": clip.video, "frames": len(clip.frames),
                "shows": (suggestions.get(i) or {}).get("shows", ""),
                "suggested": [dict(u) for u in (suggestions.get(i) or {}).get("uses") or []],
                "placed": first is not None,
                **({"start": first.start, "end": first.end, "source_start": first.source_start,
                    "reason": first.reason, "phrase": first.phrase,
                    "pieces": [{"start": p.start, "end": p.end, "source_start": p.source_start, "reason": p.reason,
                                "phrase": p.phrase} for p in pieces]}
                   if first else {"why_not": unplaced.get(i, "no_room")}),
            })
        for sid in unreadable:
            diag["sources"].append({"source_asset_id": sid, "kind": PHOTO, "placed": False, "why_not": "unreadable"})
        covered = sum(o.end - o.start for o in overlays)
        diag.update({"timeline_sec": round(total, 3), "hook_sec": HOOK_SEC, "cta_start": cta_start,
                     "keep_order": keep_order, "coverage_sec": round(covered, 3),
                     "coverage_ratio": round(covered / total, 4) if total else 0.0,
                     "product_hint": bool(hint), "placed": len(overlays)})
        return overlays, diag
    except Exception as exc:                  # never fail the cut because of B-roll
        _LOG.warning("broll_overlay_failed %s", exc.__class__.__name__)
        diag.update({"error": exc.__class__.__name__, "placed": 0})
        return (), diag
