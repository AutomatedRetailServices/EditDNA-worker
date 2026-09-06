"""Whole-video context boundary for CutSell Watch + Listen."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Tuple

from .contracts import SourceAsset, TranscriptSegment
from .providers import ProviderStatus
from .source_sampling import SourceFrameSample


@dataclass(frozen=True)
class TemporalEvent:
    source_asset_id: str
    start: float
    end: float
    kind: str
    confidence: float
    description: str


@dataclass(frozen=True)
class SourceVideoContext:
    source_asset_id: str
    summary: str
    dominant_style: str
    creator_intent: str
    events: Tuple[TemporalEvent, ...] = ()
    # Narrative routing is inferred from the whole source, never forced per clip.
    edit_mode: str = "natural"  # sales | natural | mixed
    sales_intent: float = 0.0
    main_topic: str = ""
    product_or_subject: str = ""
    story_logic: str = ""


@dataclass(frozen=True)
class WholeVideoContext:
    sources: Tuple[SourceVideoContext, ...]
    status: ProviderStatus

    def compact_text(self) -> str:
        """Return compact semantic context without flooding later LLM stages.

        Dense local MediaPipe/OpenCV events intentionally use *_candidate names.
        Hundreds of those measurements are valuable for local/take-level fusion, but
        serializing every one into grouping/composer/reviewer prompts bloats context and
        can cause provider failures.  Keep authoritative whole-video events verbatim and
        summarize dense candidates by kind/count instead.
        """
        parts = []
        for source in self.sources:
            parts.append(
                f"source={source.source_asset_id}; mode={source.edit_mode}; "
                f"sales_intent={source.sales_intent:.2f}; style={source.dominant_style}; "
                f"intent={source.creator_intent}; topic={source.main_topic}; "
                f"product_or_subject={source.product_or_subject}; story_logic={source.story_logic}; "
                f"summary={source.summary}"
            )
            candidate_counts = Counter()
            for event in source.events:
                if str(event.kind).endswith("_candidate"):
                    candidate_counts[event.kind] += 1
                    continue
                parts.append(
                    f"{event.source_asset_id}@{event.start:.2f}-{event.end:.2f} "
                    f"{event.kind}: {event.description}"
                )
            if candidate_counts:
                summary = ", ".join(
                    f"{kind}={count}" for kind, count in sorted(candidate_counts.items())
                )
                parts.append(f"dense_local_candidates[{source.source_asset_id}]: {summary}")
        return "\n".join(parts)[:20000]

    @property
    def dominant_edit_mode(self) -> str:
        if not self.sources:
            return "natural"
        counts = {"sales": 0, "natural": 0, "mixed": 0}
        for source in self.sources:
            mode = source.edit_mode if source.edit_mode in counts else "natural"
            counts[mode] += 1
        return max(counts, key=lambda key: counts[key])


class WholeVideoProvider(Protocol):
    def analyze(
        self,
        sources: Tuple[SourceAsset, ...],
        transcripts: Tuple[TranscriptSegment, ...],
        samples: Tuple[SourceFrameSample, ...],
    ) -> WholeVideoContext: ...


def safe_whole_video_analyze(
    provider: WholeVideoProvider | None,
    sources: Tuple[SourceAsset, ...],
    transcripts: Tuple[TranscriptSegment, ...],
    samples: Tuple[SourceFrameSample, ...],
) -> WholeVideoContext:
    if provider is None:
        return WholeVideoContext((), ProviderStatus("none", False, False, "not_requested"))
    try:
        result = provider.analyze(sources, transcripts, samples)
        known = {source.source_asset_id for source in sources}
        seen = set()
        for source in result.sources:
            if source.source_asset_id not in known or source.source_asset_id in seen:
                raise ValueError("whole-video provider returned invalid source id")
            if source.edit_mode not in {"sales", "natural", "mixed"}:
                raise ValueError("whole-video provider returned invalid edit mode")
            if not 0.0 <= source.sales_intent <= 1.0:
                raise ValueError("whole-video provider returned invalid sales intent")
            seen.add(source.source_asset_id)
            for event in source.events:
                if event.source_asset_id != source.source_asset_id:
                    raise ValueError("whole-video event crossed source identity")
                if event.end < event.start:
                    raise ValueError("whole-video event has invalid time range")
        if seen != known:
            raise ValueError("whole-video provider omitted source")
        return result
    except Exception as exc:
        detail = str(exc).strip()
        reason = exc.__class__.__name__
        if detail:
            reason = f"{reason}: {detail[:180]}"
        return WholeVideoContext(
            (),
            ProviderStatus(
                provider=provider.__class__.__name__,
                requested=True,
                available=False,
                status="provider_error",
                reason=reason,
            ),
        )
