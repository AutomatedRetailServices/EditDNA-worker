"""OpenAI image+transcript adapter for whole-video CutSell context."""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from .contracts import SourceAsset, TranscriptSegment
from .openai_json import parse_json_object
from .providers import ProviderStatus
from .source_sampling import SourceFrameSample
from .whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


@dataclass
class OpenAIWholeVideoProvider:
    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    @staticmethod
    def _image_url(path: str) -> str:
        raw = Path(path).read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _score(value) -> float:
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError("whole-video confidence outside 0..1")
        return score

    def analyze(
        self,
        sources: Tuple[SourceAsset, ...],
        transcripts: Tuple[TranscriptSegment, ...],
        samples: Tuple[SourceFrameSample, ...],
    ) -> WholeVideoContext:
        transcript_by_source: dict[str, list[TranscriptSegment]] = {}
        for segment in transcripts:
            transcript_by_source.setdefault(segment.source_asset_id, []).append(segment)
        samples_by_source: dict[str, list[SourceFrameSample]] = {}
        for sample in samples:
            samples_by_source.setdefault(sample.source_asset_id, []).append(sample)

        content = [{
            "type": "input_text",
            "text": (
                "You are CutSell Whole-Video Watch + Listen. Observe each entire raw TikTok Shop/UGC source before editing. "
                "Use the timestamped transcript and ordered frames to understand creator intent, story/demo flow, retries, visual demonstrations, "
                "obvious recording fumbles/restarts, reactions, product-use moments, and transitions. This stage NEVER deletes or reorders clips. "
                "Do not label normal personal mannerisms as mistakes. Return JSON only as "
                "{\"sources\":[{\"source_asset_id\":...,\"summary\":...,\"dominant_style\":...,\"creator_intent\":...,"
                "\"events\":[{\"start\":0.0,\"end\":0.0,\"kind\":...,\"confidence\":0..1,\"description\":...}]}]}. "
                "Useful event kinds include warmup, retry, false_start, visual_fumble, product_demo, story_beat, proof, reaction, transition, cta, valid_delivery. "
                "Timestamps are approximate observations, not cut boundaries. Include every source exactly once."
            ),
        }]

        for source in sorted(sources, key=lambda item: item.source_order):
            segments = sorted(transcript_by_source.get(source.source_asset_id, ()), key=lambda item: (item.start, item.end))
            frames = sorted(samples_by_source.get(source.source_asset_id, ()), key=lambda item: item.timestamp)
            content.append({
                "type": "input_text",
                "text": json.dumps({
                    "source_asset_id": source.source_asset_id,
                    "source_order": source.source_order,
                    "duration_sec": source.duration_sec,
                    "transcript": [
                        {"start": round(item.start, 3), "end": round(item.end, 3), "text": item.text}
                        for item in segments
                    ],
                    "frame_count": len(frames),
                }, ensure_ascii=False),
            })
            for frame in frames:
                content.append({
                    "type": "input_text",
                    "text": json.dumps({
                        "source_asset_id": source.source_asset_id,
                        "timestamp_sec": round(frame.timestamp, 3),
                        "relative_position": round(frame.relative_position, 4),
                    }),
                })
                content.append({
                    "type": "input_image",
                    "image_url": self._image_url(frame.path),
                    "detail": "low",
                })

        response = self._client().responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
        )
        data = parse_json_object(response.output_text)
        items = data.get("sources")
        if not isinstance(items, list):
            raise ValueError("whole-video provider returned invalid payload")

        contexts = []
        known = {source.source_asset_id for source in sources}
        seen = set()
        source_duration = {source.source_asset_id: source.duration_sec for source in sources}
        for item in items:
            source_id = str(item.get("source_asset_id") or "")
            if source_id not in known or source_id in seen:
                raise ValueError("whole-video provider returned invalid source id")
            events_raw = item.get("events") or []
            if not isinstance(events_raw, list):
                raise ValueError("whole-video provider returned invalid events")
            events = []
            duration = max(0.0, float(source_duration[source_id]))
            for event in events_raw[:100]:
                start = max(0.0, min(duration, float(event.get("start") or 0.0)))
                end = max(start, min(duration, float(event.get("end") or start)))
                events.append(TemporalEvent(
                    source_asset_id=source_id,
                    start=start,
                    end=end,
                    kind=str(event.get("kind") or "observation")[:80],
                    confidence=self._score(event.get("confidence", 0.5)),
                    description=str(event.get("description") or "")[:300],
                ))
            contexts.append(SourceVideoContext(
                source_asset_id=source_id,
                summary=str(item.get("summary") or "")[:800],
                dominant_style=str(item.get("dominant_style") or "mixed")[:80],
                creator_intent=str(item.get("creator_intent") or "")[:500],
                events=tuple(events),
            ))
            seen.add(source_id)
        if seen != known:
            raise ValueError("whole-video provider omitted source")
        return WholeVideoContext(
            sources=tuple(contexts),
            status=ProviderStatus("openai", True, True, "applied"),
        )
