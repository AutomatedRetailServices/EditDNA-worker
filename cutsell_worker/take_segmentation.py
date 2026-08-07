"""Source-safe conversion of ASR segments into candidate takes."""
from __future__ import annotations

from typing import Iterable, Mapping, Tuple

from .contracts import CandidateTake, MediaSignals, SourceAsset, TranscriptSegment
from .silence_analysis import SilenceGap, silence_ratio
from .source_identity import stable_clip_id


def _audio_quality(segment: TranscriptSegment, silence: float) -> float:
    confidences = [word.confidence for word in segment.words if word.confidence is not None]
    speech_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    duration = max(0.001, segment.end - segment.start)
    words_per_second = len(segment.words) / duration if segment.words else 0.0
    # Broad creator-friendly pace band: do not punish expressive/fast TikTok delivery heavily.
    if 1.4 <= words_per_second <= 4.2:
        pace_quality = 1.0
    elif words_per_second > 0:
        pace_quality = 0.7
    else:
        pace_quality = 0.5
    score = 0.60 * speech_confidence + 0.25 * (1.0 - silence) + 0.15 * pace_quality
    return round(max(0.0, min(1.0, score)), 4)


def segment_takes(
    segments: Iterable[TranscriptSegment],
    sources: Iterable[SourceAsset],
    gaps: Iterable[SilenceGap] = (),
) -> Tuple[CandidateTake, ...]:
    source_map: Mapping[str, SourceAsset] = {source.source_asset_id: source for source in sources}
    gap_tuple = tuple(gaps)
    output = []
    for segment in segments:
        source = source_map.get(segment.source_asset_id)
        if source is None:
            raise ValueError("transcript source is not registered in processing request")
        start = max(0.0, float(segment.start))
        end = min(float(source.duration_sec), max(start, float(segment.end))) if source.duration_sec > 0 else max(start, float(segment.end))
        text = segment.text.strip()
        if not text or end <= start:
            continue
        ratio = silence_ratio(start, end, gap_tuple, source.source_asset_id)
        signals = MediaSignals(
            source_asset_id=source.source_asset_id,
            start=start,
            end=end,
            silence_ratio=ratio,
            audio_quality=_audio_quality(segment, ratio),
        )
        output.append(CandidateTake(
            clip_id=stable_clip_id(source.source_asset_id, start, end, text),
            source_asset_id=source.source_asset_id,
            source_order=source.source_order,
            start=start,
            end=end,
            text=text,
            words=segment.words,
            signals=signals,
            complete_idea=True,
        ))
    return tuple(sorted(output, key=lambda take: (take.source_order, take.start, take.end, take.clip_id)))
