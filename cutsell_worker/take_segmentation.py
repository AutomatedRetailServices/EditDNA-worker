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
        expression_naturalness=weighted(a.expression_naturalness, b.expression_naturalness),
        gesture_naturalness=weighted(a.gesture_naturalness, b.gesture_naturalness),
        delivery_energy=weighted(a.delivery_energy, b.delivery_energy),
        distraction_risk=max(a.distraction_risk, b.distraction_risk),
    )


def _join_takes(left: CandidateTake, right: CandidateTake) -> CandidateTake:
    text = f"{left.text.rstrip()} {right.text.lstrip()}".strip()
    return CandidateTake(
        clip_id=stable_clip_id(left.source_asset_id, left.start, right.end, text),
        source_asset_id=left.source_asset_id,
        source_order=left.source_order,
        start=left.start,
        end=right.end,
        text=text,
        words=tuple(left.words) + tuple(right.words),
        signals=_merge_signals(left, right),
        complete_idea=left.complete_idea and right.complete_idea,
    )


def _repair_boundary_fragments(
    takes: Iterable[CandidateTake],
    *,
    max_fragment_sec: float = 1.5,
    max_fragment_words: int = 3,
    max_join_gap_sec: float = 0.16,
) -> Tuple[CandidateTake, ...]:
    """Reattach obvious contiguous ASR fragments without deleting real short lines.

    Two conservative repairs are allowed before Best Take/grouping:
    1. a tiny suffix may close an unfinished previous phrase, even if that suffix
       itself carries punctuation (for example an ASR-split final word);
    2. an open tiny lead-in may join immediately forward into contiguous speech.

    Real pauses, source boundaries, and short utterances that are already complete
    remain separate.
    """
    ordered = sorted(takes, key=lambda take: (take.source_order, take.start, take.end, take.clip_id))
    repaired: list[CandidateTake] = []
    index = 0
    while index < len(ordered):
        take = ordered[index]
        is_micro = take.duration_sec <= max_fragment_sec and _word_count(take.text) <= max_fragment_words
        is_open_micro = is_micro and not _ends_sentence(take.text)

        # Tiny suffix: the previous phrase is unfinished and this adjacent fragment
        # completes it. The suffix may legitimately end in punctuation.
        if is_micro and repaired:
            previous = repaired[-1]
            gap = take.start - previous.end
            same_source = previous.source_asset_id == take.source_asset_id
            if same_source and not _ends_sentence(previous.text) and -0.02 <= gap <= max_join_gap_sec:
                repaired[-1] = _join_takes(previous, take)
                index += 1
                continue

        # Tiny lead-in: attach forward only across an effectively contiguous
        # boundary. Never bridge a real pause just to create a longer sentence.
        if is_open_micro and index + 1 < len(ordered):
            following = ordered[index + 1]
            gap = following.start - take.end
            if following.source_asset_id == take.source_asset_id and -0.02 <= gap <= max_join_gap_sec:
                repaired.append(_join_takes(take, following))
                index += 2
                continue

        repaired.append(take)
        index += 1
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
