"""D-097.5 (R6) -- the AttemptReconstructor must hear the source, not only
read the ASR clock.

RAW 34040848026 (head 1d10799; the first Video00 run with a deliverable
MP4): `reconstruct_delivery_attempts` fused "La biopsia confirmó que era
un cáncer papilar de tiroides." with the next sentence ("Síntomas que
tuve. Según yo era sintomática ...") into ONE 14 s delivery attempt
although the source carried 2.96 s of measured dead air between them
(`audio_silence` interval 138.196-141.158 s, already in the whole-video
context via D-097 Priority C). Whisper had padded both segments into that
silence, so the ASR gap fell under the 1.2 s continuation ceiling. The
fused take then lost its family to the clean "Síntomas que no me parecían
sospechosos" retry (Hybrid label `failed` 0.9 for the second half's
resets), the D-089 waiver downgraded the first half's CRITICAL diagnosis
claim because its only source was a "failed" realization, and the
story's pivotal fact left the edit with Freeze passing clean.

The owning authority is the AttemptReconstructor: a measured silence that
reaches the transition and is at least as long as the continuation ceiling
is a real speech pause, whatever the ASR timestamps say. Nothing else in
the boundary ladder changes; without measurement the ASR gap rule stands
alone exactly as before.
"""
from __future__ import annotations

from cutsell_worker.attempt_reconstruction import reconstruct_delivery_attempts
from cutsell_worker.contracts import CandidateTake
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext

SRC = "src_test"


def _take(clip_id: str, start: float, end: float, text: str, order: int) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id, source_asset_id=SRC, source_order=order, start=start, end=end, text=text,
        complete_idea=True,
    )


def _context(silences: list[tuple[float, float]], *, confidence: float = 1.0) -> WholeVideoContext:
    events = tuple(
        TemporalEvent(source_asset_id=SRC, start=s, end=e, kind="audio_silence_interval", confidence=confidence,
                      description="test silence")
        for s, e in silences
    )
    return WholeVideoContext(
        sources=(SourceVideoContext(source_asset_id=SRC, summary="", dominant_style="", creator_intent="", events=events),),
        status=ProviderStatus("test", True, True, "applied"),
    )


DIAGNOSIS = _take("c_diag", 10.0, 13.9, "The biopsy confirmed it was papillary thyroid cancer.", 0)
# Whisper padded the next segment's start back into the silence: ASR gap 0.3 s.
SYMPTOMS = _take("c_sym", 14.2, 20.0, "Symptoms I had, I thought I was asymptomatic but there were signs looking back.", 1)


def test_measured_dead_air_at_the_transition_splits_a_padded_asr_gap():
    context = _context([(11.9, 14.6)])  # 2.7 s of real silence reaching the ASR boundary
    attempts, diagnostics = reconstruct_delivery_attempts((DIAGNOSIS, SYMPTOMS), context)
    assert [a.text for a in attempts] == [DIAGNOSIS.text, SYMPTOMS.text]
    assert diagnostics["boundaries"][0]["reason"] == "measured_dead_air_pause"


def test_a_short_measured_pause_still_merges_the_delivery():
    context = _context([(13.7, 14.4)])  # 0.7 s: an ordinary breath, below the 1.2 s ceiling
    attempts, diagnostics = reconstruct_delivery_attempts((DIAGNOSIS, SYMPTOMS), context)
    assert len(attempts) == 1
    assert diagnostics["boundaries"] == []


def test_a_long_silence_away_from_the_transition_does_not_split():
    context = _context([(10.2, 12.4)])  # inside the first take, ends 1.5 s before the boundary
    attempts, _diagnostics = reconstruct_delivery_attempts((DIAGNOSIS, SYMPTOMS), context)
    assert len(attempts) == 1


def test_low_confidence_relaxed_floor_silence_is_not_a_boundary_on_its_own():
    context = _context([(11.9, 14.6)], confidence=0.6)
    attempts, _diagnostics = reconstruct_delivery_attempts((DIAGNOSIS, SYMPTOMS), context)
    assert len(attempts) == 1


def test_without_measurement_the_asr_gap_rule_is_unchanged():
    attempts, diagnostics = reconstruct_delivery_attempts((DIAGNOSIS, SYMPTOMS), None)
    assert len(attempts) == 1
    far = _take("c_far", 16.0, 20.0, SYMPTOMS.text, 1)  # ASR gap 2.1 s > 1.2 s
    attempts, diagnostics = reconstruct_delivery_attempts((DIAGNOSIS, far), None)
    assert len(attempts) == 2
    assert diagnostics["boundaries"][0]["reason"] == "real_speech_pause"


def test_measured_pause_counts_only_the_silence_inside_the_pair_span():
    # A silence that starts at the boundary but runs far past the second
    # take's end is mostly not "between" the pair; only the part inside
    # the span counts (here 0.4 s -> no split).
    short_second = _take("c_short", 14.2, 14.6, "Okay.", 1)
    context = _context([(14.0, 19.0)])
    attempts, _diagnostics = reconstruct_delivery_attempts((DIAGNOSIS, short_second), context)
    assert len(attempts) == 1
