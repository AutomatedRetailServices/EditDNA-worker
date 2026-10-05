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
)
from .media_probe import probe_media
from .simple_engine import VERSION as ENGINE_VERSION
from .simple_engine import asr as _asr
from .simple_engine import caption_groups as _caption_groups
from .simple_engine import llm as _llm
from .simple_engine import process as _process
from .simple_engine.engine import OV as _JUNCTION_OVERLAP
from .usage_limits import check_processing_allowance

ENGINE_NAME = "simple"
ProgressCallback = Callable[[str, int], None]


def _words_inside(words, start: float, end: float):
    return [w for w in words if start <= (w["s"] + w["e"]) / 2 <= end]


def _clip(*, clip_id: str, source, split: Mapping, words, selected: bool) -> DraftClip:
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
    )


def process_with_simple_engine(
    request: ProcessingRequest,
    local_paths: Mapping[str, str],
    *,
    progress: Optional[ProgressCallback] = None,
    transcribe: Optional[Callable[..., dict]] = None,
    llm: Optional[Callable[[str], "tuple[str, dict]"]] = None,
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

    selected: list[DraftClip] = []
    alternates: list[DraftClip] = []
    per_source: list[dict] = []
    timeline_captions: list[dict] = []
    timeline_offset = 0.0
    total = len(sources)

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
            out = _process(words, duration, path, video_path=path, llm=llm)
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
                    ))
                elif inside:
                    hidden_index += 1
                    alternates.append(_clip(
                        clip_id=f"se-{source.source_order:03d}-h{hidden_index:04d}",
                        source=source, split=split, words=inside, selected=False,
                    ))

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
