"""Deterministic silence/dead-air analysis from ASR word timing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .contracts import TranscriptSegment, Word


@dataclass(frozen=True)
class SilenceGap:
    source_asset_id: str
    start: float
    end: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end - self.start)


def word_silence_gaps(segments: Iterable[TranscriptSegment], *, min_gap_sec: float = 0.45) -> Tuple[SilenceGap, ...]:
    by_source: dict[str, list[Word]] = {}
    for segment in segments:
        by_source.setdefault(segment.source_asset_id, []).extend(segment.words)

    gaps = []
    for source_asset_id, words in by_source.items():
        ordered = sorted(words, key=lambda word: (word.start, word.end))
        for left, right in zip(ordered, ordered[1:]):
            start = max(0.0, left.end)
            end = max(start, right.start)
            if end - start >= min_gap_sec:
                gaps.append(SilenceGap(source_asset_id, start, end))
    return tuple(sorted(gaps, key=lambda gap: (gap.source_asset_id, gap.start, gap.end)))


def silence_ratio(start: float, end: float, gaps: Iterable[SilenceGap], source_asset_id: str) -> float:
    duration = max(0.001, end - start)
    overlap = 0.0
    for gap in gaps:
        if gap.source_asset_id != source_asset_id:
            continue
        overlap += max(0.0, min(end, gap.end) - max(start, gap.start))
    return min(1.0, overlap / duration)
