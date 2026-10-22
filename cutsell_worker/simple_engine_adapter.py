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
from .media_probe import probe_media
from .simple_engine import VERSION as ENGINE_VERSION
from .simple_engine import asr as _asr
from .simple_engine import caption_groups as _caption_groups
from .simple_engine import llm as _llm
from .simple_engine import cache_policy as _cache_policy
from .simple_engine import process as _process
from .simple_engine.engine import OV as _JUNCTION_OVERLAP
from .usage_limits import check_processing_allowance

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
) -> ProcessingResult:
    """`transcribe` and `llm` are injectable for tests; production uses Deepgram and Anthropic."""
    notify = progress or (lambda stage, percent: None)
    transcribe = transcribe or _asr.transcribe_words
    sources = sorted(request.sources, key=lambda item: int(item.source_order))
    if not sources:
        raise ValueError("at least one source is required")

    durations: dict[str, float] = {}
    for source in sources:
        path = local_paths.get(source.source_asset_id)
        if not path:
            raise ValueError(f"missing local path for source {source.source_asset_id}")
        if not Path(path).exists():
            raise FileNotFoundError(path)
        durations[source.source_asset_id] = float(probe_media(path).duration_sec)

    usage = check_processing_allowance(user_id=request.user_id, durations_sec=list(durations.values()))
    if not usage.allowed:
        raise ValueError(f"processing denied: {usage.reason}")

    bulk = is_bulk(sources)
    selected: list[DraftClip] = []
    alternates: list[DraftClip] = []
    per_source: list[dict] = []
    timeline_captions: list[dict] = []
    zoom_keyframes: list[ZoomKeyframe] = []
    timeline_offset = 0.0
    total = len(sources)

    # Prompt cache for every video (PO 2026-10-22): 1 hour while few videos are being cut, else 5 minutes.
    cache_ttl, videos_last_hour = _cache_policy.note_video()
    with tempfile.TemporaryDirectory(prefix="cutsell-simple-engine-") as directory:
        for index, source in enumerate(sources):
            path = str(local_paths[source.source_asset_id])
            duration = durations[source.source_asset_id]
            base = 12 + int(index * 78 / total)
            notify("transcribing", base)
            audio_path = _asr.extract_audio(path, str(Path(directory) / f"{source.source_order:03d}.mp3"))
            heard = transcribe(audio_path, language_hint=request.language_hint)
            words = [w for w in heard.get("words", []) if float(w["e"]) > float(w["s"])]
            if not words:
                raise ValueError(f"no speech found in source {source.source_asset_id}")

            notify("analyzing", base + int(30 / total))
            # The silence refine reads the audio decoded from the source file itself (what it was calibrated on).
            # A voice-only source (audio first) has no picture yet: no shot-change snap for its first cut.
            voice_only = bool(audio_only) and source.source_asset_id in audio_only
            with _llm.cache_scope(True, ttl=cache_ttl):
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
                "splits": out["splits"],
                "log": out["log"],
                "usage": out["usage"],
                "prompt_cache": _llm.cache_report(out["usage"], ttl=cache_ttl, videos_last_hour=videos_last_hour),
                "zoom_moments": zoom_log,
            })

    if not selected:
        raise ValueError("simple engine kept no content")

    cut_duration = round(sum(max(0.0, clip.end - clip.start) for clip in selected), 3)
    diagnostics = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "simple_engine": {
            "model": _llm.model_name(),
            "asr": "deepgram-nova-3",
            "cut_duration_sec": cut_duration,
            "captions": timeline_captions,
            "caption_groups": _caption_groups(timeline_captions),
            "sources": per_source,
        },
    }
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
