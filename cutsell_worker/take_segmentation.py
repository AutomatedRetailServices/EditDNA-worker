"""Source-safe conversion of ASR segments into candidate takes."""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Mapping, Tuple

from .contracts import CandidateTake, MediaSignals, SourceAsset, TranscriptSegment, Word
from .silence_analysis import SilenceGap, silence_ratio
from .source_identity import stable_clip_id


def _audio_quality(segment: TranscriptSegment, silence: float) -> float:
    confidences = [word.confidence for word in segment.words if word.confidence is not None]
    speech_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    duration = max(0.001, segment.end - segment.start)
    words_per_second = len(segment.words) / duration if segment.words else 0.0
    if 1.4 <= words_per_second <= 4.2:
        pace_quality = 1.0
    elif words_per_second > 0:
        pace_quality = 0.7
    else:
        pace_quality = 0.5
    score = 0.60 * speech_confidence + 0.25 * (1.0 - silence) + 0.15 * pace_quality
    return round(max(0.0, min(1.0, score)), 4)


def _speech_units(segment: TranscriptSegment, *, split_gap_sec: float = 0.75) -> Tuple[TranscriptSegment, ...]:
    """Split one ASR segment only at strong word-timestamp gaps."""
    words = tuple(sorted(segment.words, key=lambda word: (word.start, word.end)))
    if len(words) < 2:
        return (segment,)

    chunks: list[list[Word]] = [[]]
    for index, word in enumerate(words):
        if index:
            previous = words[index - 1]
            if word.start - previous.end >= split_gap_sec:
                chunks.append([])
        chunks[-1].append(word)
    if len(chunks) == 1:
        return (segment,)

    output = []
    for chunk in chunks:
        if not chunk:
            continue
        output.append(TranscriptSegment(
            source_asset_id=segment.source_asset_id,
            start=chunk[0].start,
            end=chunk[-1].end,
            text=" ".join(word.text.strip() for word in chunk if word.text.strip()),
            words=tuple(chunk),
        ))
    return tuple(output) or (segment,)


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w'’-]+", text, flags=re.UNICODE))


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?][\"'”’)]*\s*$", text.strip()))


def _merge_signals(left: CandidateTake, right: CandidateTake) -> MediaSignals | None:
    if left.signals is None and right.signals is None:
        return None
    a = left.signals or MediaSignals(left.source_asset_id, left.start, left.end)
    b = right.signals or MediaSignals(right.source_asset_id, right.start, right.end)
    left_duration = max(0.001, left.duration_sec)
    right_duration = max(0.001, right.duration_sec)
    total = left_duration + right_duration

    def weighted(x: float, y: float) -> float:
        return (x * left_duration + y * right_duration) / total

    return MediaSignals(
        source_asset_id=left.source_asset_id,
        start=left.start,
        end=right.end,
        silence_ratio=weighted(a.silence_ratio, b.silence_ratio),
        audio_quality=weighted(a.audio_quality, b.audio_quality),
        face_visibility=weighted(a.face_visibility, b.face_visibility),
        eye_contact=weighted(a.eye_contact, b.eye_contact),
        framing_quality=weighted(a.framing_quality, b.framing_quality),
        product_visibility=weighted(a.product_visibility, b.product_visibility),
        motion_stability=weighted(a.motion_stability, b.motion_stability),
        continuity=weighted(a.continuity, b.continuity),
        visual_fumble=max(a.visual_fumble, b.visual_fumble),
    )


def _repair_boundary_fragments(
    takes: Iterable[CandidateTake],
    *,
    max_fragment_sec: float = 0.45,
    max_join_gap_sec: float = 0.12,
) -> Tuple[CandidateTake, ...]:
    """Reattach obvious tiny ASR tail fragments to the prior spoken idea.

    Example: ``"for working"`` followed immediately by ``"out."`` should be one
    candidate. This never crosses a source and deliberately does not merge a short
    standalone utterance after a completed sentence (for example ``"Wow!"``).
    """
    ordered = sorted(takes, key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    repaired: list[CandidateTake] = []
    for take in ordered:
        if repaired:
            previous = repaired[-1]
            gap = take.start - previous.end
            tiny_tail = take.duration_sec <= max_fragment_sec and _word_count(take.text) <= 1
            same_source = previous.source_asset_id == take.source_asset_id
            prior_is_open = not _ends_sentence(previous.text)
            if same_source and tiny_tail and prior_is_open and -0.02 <= gap <= max_join_gap_sec:
                text = f"{previous.text.rstrip()} {take.text.lstrip()}".strip()
                merged = CandidateTake(
                    clip_id=stable_clip_id(previous.source_asset_id, previous.start, take.end, text),
                    source_asset_id=previous.source_asset_id,
                    source_order=previous.source_order,
                    start=previous.start,
                    end=take.end,
                    text=text,
                    words=tuple(previous.words) + tuple(take.words),
                    signals=_merge_signals(previous, take),
                    complete_idea=previous.complete_idea and take.complete_idea,
                )
                repaired[-1] = merged
                continue
        repaired.append(take)
    return tuple(repaired)


def segment_takes(
    segments: Iterable[TranscriptSegment],
    sources: Iterable[SourceAsset],
    gaps: Iterable[SilenceGap] = (),
) -> Tuple[CandidateTake, ...]:
    source_map: Mapping[str, SourceAsset] = {source.source_asset_id: source for source in sources}
    gap_tuple = tuple(gaps)
    output = []
    for original_segment in segments:
        if original_segment.source_asset_id not in source_map:
            raise ValueError("transcript source is not registered in processing request")
        for segment in _speech_units(original_segment):
            source = source_map[segment.source_asset_id]
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
    return _repair_boundary_fragments(output)
