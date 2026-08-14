"""RunPod-local whole-video context for Universal Clean Cut.

This provider never calls an external API. It creates the per-source context shell
needed by dense local MediaPipe/OpenCV evidence and performance confirmation, using
only source metadata and local ASR transcripts produced inside the worker.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Tuple

from .contracts import SourceAsset, TranscriptSegment
from .providers import ProviderStatus
from .source_sampling import SourceFrameSample
from .whole_video_analysis import SourceVideoContext, WholeVideoContext


def _compact_transcript(segments: tuple[TranscriptSegment, ...], *, limit: int = 700) -> str:
    text = " ".join(str(segment.text or "").strip() for segment in segments if str(segment.text or "").strip())
    return " ".join(text.split())[:limit]


class RunPodLocalWholeVideoProvider:
    """Create source context locally so dense visual events remain first-class evidence."""

    def analyze(
        self,
        sources: Tuple[SourceAsset, ...],
        transcripts: Tuple[TranscriptSegment, ...],
        samples: Tuple[SourceFrameSample, ...],
    ) -> WholeVideoContext:
        by_source: dict[str, list[TranscriptSegment]] = defaultdict(list)
        for segment in transcripts:
            by_source[segment.source_asset_id].append(segment)

        contexts = []
        for source in sources:
            source_segments = tuple(sorted(by_source.get(source.source_asset_id, ()), key=lambda item: (item.start, item.end)))
            compact = _compact_transcript(source_segments)
            contexts.append(SourceVideoContext(
                source_asset_id=source.source_asset_id,
                summary=compact,
                dominant_style="creator_raw",
                creator_intent="recording_clean_cut",
                events=(),
                edit_mode="natural",
                sales_intent=0.0,
                main_topic="",
                product_or_subject="",
                story_logic="preserve natural source order; remove recording mistakes only",
            ))

        return WholeVideoContext(
            tuple(contexts),
            ProviderStatus(
                provider="runpod_local_whole_video",
                requested=True,
                available=True,
                status="applied",
                reason="local_asr_context",
            ),
        )
