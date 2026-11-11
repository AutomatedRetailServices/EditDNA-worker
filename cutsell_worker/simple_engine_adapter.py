"""Adapter: run the simple engine (cutsell_worker.simple_engine) for a Flow B request and return the
SAME ProcessingResult / DraftTimeline contract the legacy engine returns, so the API, the draft store,
the mobile editor and the export job keep working unchanged.

Selected by CUTSELL_ENGINE=simple (see config.py, worker_job.py). No GPU, no Whisper, no Gemini:
Deepgram Nova-3 for word times + two Claude calls per source + local ffmpeg.

Mapping
- every VISIBLE split        -> one `selected` DraftClip (source in/out = split start/end)
- every HIDDEN split that contains spoken words -> one `alternates` DraftClip with take_group_id=None,
  so the existing restore endpoint (draft_edits.restore_clip) can bring it back
- hidden silence             -> nothing (it is simply not on the timeline)
- word-level captions, caption groups, reasons, token usage -> draft.diagnostics["simple_engine"]
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Mapping, Optional

from .contracts import (
    SCHEMA_VERSION, DraftClip, DraftTimeline, EditStrategy, JobState, ProcessingRequest, ProcessingResult, Word,
    ZoomKeyframe,
)
from . import broll_overlay as _broll
from .media_probe import probe_media
from .simple_engine import VERSION as ENGINE_VERSION
from .simple_engine import asr as _asr
from .simple_engine import caption_groups as _caption_groups
from .simple_engine import llm as _llm
from .simple_engine import cache_policy as _cache_policy
from .simple_engine import process as _process
from .simple_engine.engine import OV as _JUNCTION_OVERLAP
from .usage_limits import check_processing_allowance


# 22 Oct (IMG_9661): Deepgram's language detection heard a quiet Spanish clip as Japanese (one word) and Slovak.
# CutSell creators speak Spanish or English: when the detected language is neither, the clip is heard again in Spanish
# and in English and the version with more words is kept. Clips detected as es / en (every Gold video) never change.
CREATOR_LANGUAGES = ("es", "en")
# Fewer words than this per minute of video = nobody is talking to the camera (5:22 with 9 stray words).
MIN_WORDS_PER_MINUTE = 15.0


def _heard_in_her_language(heard: dict, audio_path: str, transcribe, language_hint) -> dict:
    language = str(heard.get("language") or "").lower()[:2]
    if language_hint or language in CREATOR_LANGUAGES:
        return heard
    best = heard
    for code in CREATOR_LANGUAGES:
        try:
            again = transcribe(audio_path, language_hint=code)
        except Exception:
            continue
        if len(again.get("words") or []) > len(best.get("words") or []):
            best = {**again, "language": code, "language_detected": language or None}
    return best


# PO 22 Oct (IMG_9661): a quiet voice in the background (another person, a TV) with only a few words is not her talking
# to the camera, whatever the clip's length: no cut is made from it.
BACKGROUND_MEAN_DB = -25.0
BACKGROUND_MAX_WORDS = 15


def _mean_volume_db(audio_path: str) -> float | None:
    import subprocess
    try:
        err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio_path), "-af", "volumedetect", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=120).stderr
    except Exception:
        return None
    for line in err.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0])
            except ValueError:
                return None
    return None


def _only_background_voice(words: list, audio_path: str, *, mean_db=None) -> bool:
    if len(words) >= BACKGROUND_MAX_WORDS:
        return False                     # a real talk has more words than this: never measured, never changed
    level = _mean_volume_db(audio_path) if mean_db is None else mean_db
    return level is not None and level < BACKGROUND_MEAN_DB


def _nobody_talking(words: list, duration: float) -> bool:
    if duration < 20:
        return False          # a short clip: any words count
    return len(words) / (duration / 60.0) < MIN_WORDS_PER_MINUTE


class NoSpeechError(ValueError):
    """Nobody can be heard in the video (22 Oct). The app says so in plain words ("We couldn't hear anyone talking in
    this video") and offers the video for Mix, instead of "This cut needs another try"."""

ENGINE_NAME = "simple"
ProgressCallback = Callable[[str, int], None]


def _words_inside(words, start: float, end: float):
    return [w for w in words if start <= (w["s"] + w["e"]) / 2 <= end]


# Overlap audio: each side of a join may borrow up to this much of the clip's own source audio,
# stopping this far short of any other spoken word (so no other phrase is ever heard in a join).
ROLL_MAX_SEC = 0.06
ROLL_WORD_MARGIN_SEC = 0.02
# Auto Zoom AI: soft push-ins on key moments.
# 21 Oct: zoom in ~0.4 s, hold, zoom out ~0.4 s; 110-125 %, centred (keyframes.zoom_to_keyframes).
ZOOM_SCALE = {"hook": 1.12, "product": 1.18, "price": 1.20, "cta": 1.12, "look": 1.22, "benefit": 1.15}
ZOOM_LEAD_SEC, ZOOM_MAX_SEC, ZOOM_MIN_SEC = 0.10, 2.2, 0.9
ZOOM_EDGE_SEC = 0.30          # never during a join
ZOOM_MIN_SPACING_SEC = 3.0


def _rolls(split: Mapping, words, duration: float) -> tuple[float, float]:
    """Silence available right before / after a kept part in its source, capped for a short crossfade."""
    start, end = float(split["start"]), float(split["end"])
    before = [float(w["e"]) for w in words if float(w["e"]) <= start + 1e-6]
    after = [float(w["s"]) for w in words if float(w["s"]) >= end - 1e-6]
    room_before = start - (max(before) + ROLL_WORD_MARGIN_SEC if before else 0.0)
    room_after = (min(after) - ROLL_WORD_MARGIN_SEC if after else duration) - end
    pre = max(0.0, min(ROLL_MAX_SEC, room_before, start))
    post = max(0.0, min(ROLL_MAX_SEC, room_after, duration - end))
    return round(pre, 3), round(post, 3)


def _zoom_keyframes(source, moments, splits, log: list | None = None) -> list[ZoomKeyframe]:
    """Key moments -> zooms inside kept parts, away from joins, at most one every 3 s. A moment near the end
    of its part starts its zoom earlier (still inside the part) instead of being dropped. `log` (optional)
    gets one row per moment: the word, its kind and what became of it (diagnostics: why a cut has N zooms)."""
    visible = [(float(s["start"]), float(s["end"])) for s in splits if s["visible"]]
    out: list[ZoomKeyframe] = []

    def note(moment, result):
        if log is not None:
            log.append({"s": round(float(moment["s"]), 3), "word": str(moment.get("w", "")), "kind": str(moment["kind"]),
                        "zoom": result})

    for moment in sorted(moments, key=lambda m: float(m["s"])):
        t = float(moment["s"]) - ZOOM_LEAD_SEC
        part = next(((a, b) for a, b in visible if a <= float(moment["s"]) <= b), None)
        if part is None:
            note(moment, "cut_out")
            continue
        a, b = part
        t = max(t, a + ZOOM_EDGE_SEC)
        if b - ZOOM_EDGE_SEC - t < ZOOM_MIN_SEC:
            t = max(a + ZOOM_EDGE_SEC, b - ZOOM_EDGE_SEC - ZOOM_MIN_SEC)
        length = min(ZOOM_MAX_SEC, b - ZOOM_EDGE_SEC - t)
        if length < ZOOM_MIN_SEC:
            note(moment, "part_too_short")
            continue
        if out and t - out[-1].source_time < ZOOM_MIN_SPACING_SEC:
            note(moment, "too_close")
            continue
        note(moment, "added")
        kind = str(moment["kind"])
        out.append(ZoomKeyframe(
            keyframe_id=f"z-{source.source_order:03d}-{len(out) + 1:02d}",
            source_asset_id=source.source_asset_id, source_time=round(t, 3), duration=round(length, 3),
            scale=ZOOM_SCALE.get(kind, 1.08), kind=kind, origin="ai",
        ))
    return out


def _clip(*, clip_id: str, source, split: Mapping, words, selected: bool,
          rolls: tuple[float, float] = (0.0, 0.0)) -> DraftClip:
    text = " ".join(w["w"] for w in words)
    return DraftClip(
        clip_id=clip_id,
        source_asset_id=source.source_asset_id,
        source_order=int(source.source_order),
        start=float(split["start"]),
        end=float(split["end"]),
        text=text,
        caption_text=text,
        words=tuple(Word(w["w"], float(w["s"]), float(w["e"]), None) for w in words),
        take_group_id=None,
        selected=selected,
        audio_pre_roll=rolls[0],
        audio_post_roll=rolls[1],
    )


def is_bulk(sources) -> bool:
    """A video sent in a Bulk of several (the app puts metadata.bulk_size on each source; iOS build 26+)."""
    for source in sources:
        meta = getattr(source, "metadata", None)
        try:
            if isinstance(meta, Mapping) and int(str(meta.get("bulk_size", "0"))) > 1:
                return True
        except ValueError:
            continue
    return False


def process_with_simple_engine(
    request: ProcessingRequest,
    local_paths: Mapping[str, str],
    *,
    progress: Optional[ProgressCallback] = None,
    transcribe: Optional[Callable[..., dict]] = None,
    llm: Optional[Callable[[str], "tuple[str, dict]"]] = None,
    audio_only: Optional[set] = None,
    broll_ask: Optional[Callable[[list], "tuple[str, dict]"]] = None,
    broll_frames: Optional[Callable[[str, float], list]] = None,
    broll_motion: Optional[Callable[[str], list]] = None,
    pending_broll: Optional[set] = None,
    wait_for_broll: Optional[Callable[[set], "tuple[dict, dict]"]] = None,
) -> ProcessingResult:
    """`transcribe` and `llm` are injectable for tests; production uses Deepgram and Anthropic.
    B-roll (PO 9-10 Nov 2026, broll_overlay.py): sources marked metadata.clip_type "broll" are never cut and never
    enter the voice timeline; they are placed over the voice as `media_overlays`. `broll_*` are injectable for tests.
    A request with no B-roll source runs exactly as before.
    B-roll v2: a B-roll photo (JPEG) counts 0 s for the usage meter and is never heard. `pending_broll` = B-roll
    sources whose file is not on storage yet (audio-first flow: the app sends no audio for B-roll): no local file is
    needed for them. After the voice is cut, `wait_for_broll(pending ids) -> ({id: local path}, diagnostics)` gives
    them a last chance (the worker polls storage, CUTSELL_BROLL_UPLOAD_WAIT_SEC in total): one that arrived is looked
    at normally; one still missing is placed from its declared length without being looked at."""
    notify = progress or (lambda stage, percent: None)
    transcribe = transcribe or _asr.transcribe_words
    sources = sorted(request.sources, key=lambda item: int(item.source_order))
    if not sources:
        raise ValueError("at least one source is required")
    pending_broll = set(pending_broll or ())

    durations: dict[str, float] = {}
    for source in sources:
        if _broll.is_photo_source(source) or (source.source_asset_id in pending_broll and _broll.is_broll(source)):
            # never probed for length: a photo counts 0 minutes; a pending clip is measured when it arrives
            durations[source.source_asset_id] = 0.0 if _broll.is_photo_source(source) else float(source.duration_sec or 0.0)
            continue
        path = local_paths.get(source.source_asset_id)
        if not path:
            raise ValueError(f"missing local path for source {source.source_asset_id}")
        if not Path(path).exists():
            raise FileNotFoundError(path)
        durations[source.source_asset_id] = float(probe_media(path).duration_sec)

    usage = check_processing_allowance(user_id=request.user_id, durations_sec=[
        0.0 if source.source_asset_id in pending_broll else durations[source.source_asset_id] for source in sources])
    if not usage.allowed:
        raise ValueError(f"processing denied: {usage.reason}")

    bulk = is_bulk(sources)
    # B-roll is never cut: only the voice sources go through the engine below (all of them when there is no B-roll).
    voice_sources, broll_sources = _broll.split_sources(sources)
    if not voice_sources:
        raise ValueError("at least one talking video is required")
    engine_moments: list = []
    selected: list[DraftClip] = []
    alternates: list[DraftClip] = []
    per_source: list[dict] = []
    timeline_captions: list[dict] = []
    zoom_keyframes: list[ZoomKeyframe] = []
    timeline_offset = 0.0
    spoken: list = []              # (source, words, duration) of every source that has speech
    total = len(voice_sources)

    # Prompt cache for every video (PO 2026-10-22): 1 hour while few videos are being cut, else 5 minutes.
    cache_ttl, videos_last_hour = _cache_policy.note_video()
    with tempfile.TemporaryDirectory(prefix="cutsell-simple-engine-") as directory:
        for index, source in enumerate(voice_sources):
            path = str(local_paths[source.source_asset_id])
            duration = durations[source.source_asset_id]
            base = 12 + int(index * 78 / total)
            notify("transcribing", base)
            audio_path = _asr.extract_audio(path, str(Path(directory) / f"{source.source_order:03d}.mp3"))
            heard = transcribe(audio_path, language_hint=request.language_hint)
            heard = _heard_in_her_language(heard, audio_path, transcribe, request.language_hint)
            words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
            if not words:
                raise NoSpeechError(f"no speech found in source {source.source_asset_id}")
            spoken.append((source, words, duration))
            # PO 2026-10-29: a video with only a little speech (a few words, quiet, or the rest silent) is no longer
            # refused: the cut keeps the words that were said and drops the rest. Only a video with no words at all
            # is "no speech". The two old checks are kept as a note in the diagnostics.
            little_speech = _nobody_talking(words, duration) or _only_background_voice(words, audio_path)

            notify("analyzing", base + int(30 / total))
            # The silence refine reads the audio decoded from the source file itself (what it was calibrated on).
            # A voice-only source (audio first) has no picture yet: no shot-change snap for its first cut.
            voice_only = bool(audio_only) and source.source_asset_id in audio_only
            # Bulk (PO 2026-10-23): the calls go through the Batch API (50 %), never more than 30 min of waiting.
            with _llm.cache_scope(True, ttl=cache_ttl), _llm.batch_scope(bulk), _llm.memory_owner(request.user_id):
                out = _process(words, duration, path, video_path=None if voice_only else path,
                               scene_cut_list=[] if voice_only else None, llm=llm)
            notify("composing", base + int(70 / total))

            hidden_index = 0
            source_selected = 0
            for split in out["splits"]:
                inside = _words_inside(words, float(split["start"]), float(split["end"]))
                if split["visible"]:
                    source_selected += 1
                    selected.append(_clip(
                        clip_id=f"se-{source.source_order:03d}-{source_selected:04d}",
                        source=source, split=split, words=inside, selected=True,
                        rolls=_rolls(split, words, duration),
                    ))
                elif inside:
                    hidden_index += 1
                    alternates.append(_clip(
                        clip_id=f"se-{source.source_order:03d}-h{hidden_index:04d}",
                        source=source, split=split, words=inside, selected=False,
                    ))

            zoom_log: list = []
            zoom_keyframes.extend(_zoom_keyframes(source, out.get("moments") or [], out["splits"], zoom_log))
            if broll_sources:
                engine_moments.extend({**dict(m), "source_asset_id": source.source_asset_id}
                                      for m in out.get("moments") or [] if isinstance(m, Mapping))

            # captions are on the cut timeline of THIS source; shift them onto the whole draft timeline
            shifted = [{"w": c["w"], "s": round(c["s"] + timeline_offset, 3), "e": round(c["e"] + timeline_offset, 3)}
                       for c in out["captions"]]
            timeline_captions.extend(shifted)
            if source_selected:
                timeline_offset += float(out["cut_duration"]) - (_JUNCTION_OVERLAP if index < total - 1 else 0.0)

            per_source.append({
                "source_asset_id": source.source_asset_id,
                "source_order": int(source.source_order),
                "language": heard.get("language"),
                "duration_sec": round(duration, 3),
                "cut_duration_sec": out["cut_duration"],
                "passes": out["passes"],
                "word_count": len(words),
                "little_speech": bool(little_speech),
                "splits": out["splits"],
                "log": out["log"],
                "usage": out["usage"],
                "prompt_cache": _llm.cache_report(out["usage"], ttl=cache_ttl, videos_last_hour=videos_last_hour),
                "zoom_moments": zoom_log,
            })

    kept_spoken_part = False
    if not selected and spoken:
        # PO 5 Nov 2026 (a 2 s clip came back as "No pudimos terminar este corte"): when the engine keeps nothing of
        # a video that HAS speech, the cut keeps the part where she talks (first word to last word) instead of
        # failing -- "cut where the audio ends, never an error because there is audio" (PO 29 Oct). WHEN UNCERTAIN,
        # KEEP. A video with no words at all is still "no speech" (raised above).
        kept_spoken_part = True
        offset = 0.0
        for source, words, duration in spoken:
            start = max(0.0, float(words[0]["s"]) - 0.15)
            end = min(float(duration), float(words[-1]["e"]) + 0.35) if duration else float(words[-1]["e"]) + 0.35
            if end <= start:
                continue
            selected.append(_clip(clip_id=f"se-{source.source_order:03d}-0001", source=source,
                                  split={"start": start, "end": end}, words=words, selected=True))
            timeline_captions.extend({"w": w["w"], "s": round(float(w["s"]) - start + offset, 3),
                                      "e": round(float(w["e"]) - start + offset, 3)} for w in words)
            offset += end - start
        for item in per_source:
            item["little_speech"] = True
            item["kept_spoken_part"] = True
    if not selected:
        raise ValueError("simple engine kept no content")

    cut_duration = round(sum(max(0.0, clip.end - clip.start) for clip in selected), 3)
    diagnostics = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        # The app shows "Almost no voice in this video, check the cut" when this is true (PO 2026-10-29).
        "little_speech": any(item.get("little_speech") for item in per_source),
        "kept_spoken_part": kept_spoken_part,
        "simple_engine": {
            "model": _llm.model_name(),
            "asr": "deepgram-nova-3",
            "cut_duration_sec": cut_duration,
            "captions": timeline_captions,
            "caption_groups": _caption_groups(timeline_captions),
            "sources": per_source,
        },
    }
    media_overlays: tuple = ()
    upload_wait = None
    if broll_sources and pending_broll and wait_for_broll is not None:
        # PO 11 Nov: B-roll still uploading -- polled AFTER the voice cut, so the cut's own time is not added.
        try:
            arrived, upload_wait = wait_for_broll(set(pending_broll))
        except Exception as exc:              # never fail the cut because of B-roll
            arrived, upload_wait = {}, {"error": exc.__class__.__name__}
        local_paths = dict(local_paths)
        for sid, path in (arrived or {}).items():
            source = next((s for s in broll_sources if s.source_asset_id == sid), None)
            if source is None or not path or not Path(path).exists():
                continue
            local_paths[sid] = path
            if not _broll.is_photo_source(source):
                try:
                    durations[sid] = float(probe_media(path).duration_sec)
                except Exception:
                    local_paths.pop(sid, None)
                    continue
            pending_broll.discard(sid)
    if broll_sources:
        # Over the finished voice cut: the voice (selected) is never changed by this step, and it never raises.
        media_overlays, diagnostics["broll_overlay"] = _broll.place_broll_overlays(
            selected=selected, broll_sources=broll_sources, voice_sources=voice_sources, local_paths=local_paths,
            durations=durations, moments=engine_moments, audio_only=audio_only,
            ask=broll_ask, frames=broll_frames, motion=broll_motion,
        )
        if upload_wait is not None:
            diagnostics["broll_overlay"]["upload_wait"] = upload_wait
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        strategy=EditStrategy.MIXED,
        selected=tuple(selected),
        alternates=tuple(alternates),
        discarded=(),
        diagnostics=diagnostics,
        audio_overlap_enabled=bool(request.dialogue_overlap_enabled),
        auto_zoom_enabled=bool(request.auto_zoom_enabled),
        zoom_keyframes=tuple(zoom_keyframes),
        media_overlays=tuple(media_overlays),
    )
    return ProcessingResult(
        schema_version=SCHEMA_VERSION,
        project_id=request.project_id,
        state=JobState.DRAFT_READY,
        draft=draft,
        stage_status={
            "engine": ENGINE_NAME,
            "engine_version": ENGINE_VERSION,
            "asr": "deepgram-nova-3",
            "decision": "applied",
            "selected_count": len(selected),
            "hidden_spoken_count": len(alternates),
        },
    )
