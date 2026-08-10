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
                "You are CutSell Whole-Video Watch + Listen. FIRST understand each complete raw recording before any edit decision. "
                "Treat the ordered frames as a temporal performance sequence and combine them with timestamped speech. Infer what the creator is "
                "actually trying to communicate, how the performance changes over time, and what narrative exists in the footage. "
                "Classify edit_mode as sales, natural, or mixed. Sales means the footage genuinely tries to sell/show a product or service; natural "
                "means storytime, yapping, talking-head, routine/lifestyle, commentary, education, vlog or personal update without a sales goal. "
                "Never force non-sales footage into a funnel. For sales, understand the product and available sales story without forcing a rigid "
                "HOOK->PROBLEM->BENEFIT->PROOF->CTA sequence. For natural footage, understand topic, setup, development, payoff/conclusion and useful personality. "
                "Identify visual hooks, verbal hooks, combined visual_verbal_hook moments and later rehook moments. A silent visual hook can be valuable. "
                "Observe camera engagement, eye contact, face/expression changes, body/hand movement, product handling, speech/body congruency and delivery. "
                "Timestamp recording-process failures: false_start, wrong_take, verbal_fumble, visual_fumble, body_reset, retry_setup, frustration, "
                "breaking_character, recording_joke, accidental_laughter, camera_adjustment, product_handling_mistake, searching_for_words, "
                "unintentional_dead_air. Also timestamp retry, valid_delivery, product_demo, story_beat, proof, reaction, transition and cta when relevant. "
                "Do NOT mark an authentic joke/reaction that belongs in the actual story as a recording_joke. Do NOT mark normal individual mannerisms as errors. "
                "Long silence with no speech AND no visual/story value is unintentional_dead_air; an intentional dramatic/emphasis/reveal pause is meaningful_pause. "
                "Events describe what happened with the tightest defensible timestamps. This stage does not itself delete or reorder. "
                "Ignore burned-in captions/stickers/text as a primary editing signal; understand the actual performance and spoken/narrative content. "
                "Return JSON only as {\"sources\":[{\"source_asset_id\":...,\"summary\":...,\"dominant_style\":...,"
                "\"creator_intent\":...,\"edit_mode\":\"sales|natural|mixed\",\"sales_intent\":0..1,\"main_topic\":...,"
                "\"product_or_subject\":...,\"story_logic\":...,\"events\":[{\"start\":0.0,\"end\":0.0,\"kind\":...,"
                "\"confidence\":0..1,\"description\":...}]}]}. Include every source exactly once."
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
            for event in events_raw[:240]:
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
            mode = str(item.get("edit_mode") or "natural").strip().lower()
            if mode not in {"sales", "natural", "mixed"}:
                mode = "natural"
            contexts.append(SourceVideoContext(
                source_asset_id=source_id,
                summary=str(item.get("summary") or "")[:1000],
                dominant_style=str(item.get("dominant_style") or "mixed")[:80],
                creator_intent=str(item.get("creator_intent") or "")[:600],
                events=tuple(events),
                edit_mode=mode,
                sales_intent=self._score(item.get("sales_intent", 0.0)),
                main_topic=str(item.get("main_topic") or "")[:500],
                product_or_subject=str(item.get("product_or_subject") or "")[:500],
                story_logic=str(item.get("story_logic") or "")[:1200],
            ))
            seen.add(source_id)
        if seen != known:
            raise ValueError("whole-video provider omitted source")
        return WholeVideoContext(
            sources=tuple(contexts),
            status=ProviderStatus("openai", True, True, "applied"),
        )
