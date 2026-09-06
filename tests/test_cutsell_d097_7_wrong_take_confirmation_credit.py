"""D-097.7 (R8) -- a take the multimodal performance layer rejected as a
`wrong_take` (and that the clean-cut stage removed for exactly that reason)
was lost BY DECISION; its coarse vocabulary loss must not block Freeze.

RAW 34043247473 (head d59b94f): "Tuve problemas de estómago en una
temporada, en 2023, hay que voltar." was removed by clean_cut_decisions
(`whole_video_bad_take:wrong_take`, 0.97: facial-expression shift + hand
reset at its end, a restart of the same opening 0.74 s later). Both QA
references remove that take. StoryValidator still blocked Freeze
(UNIQUE_FACT_LOST, coverage 0.43, only a CONTEXTUAL year missing) because
its pre-group credit trusts only the arbiter, which answered "the retry
adds details" (not same idea) -- a human-review block with no repair
strategy for a delivery the creator had visibly thrown away. The
confirmation is the recording-process evidence class D-097.A ranks above
the semantic judge; here it suppresses ONLY the coarse vocabulary signal,
exactly like the arbiter credit. Atoms are never touched.
"""
from __future__ import annotations

from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
from cutsell_worker.final_story_coherence_validation import (
    PRE_GROUP_WRONG_TAKE_CONFIRMATION,
    apply_final_story_coherence_validation,
)

DELIVERY = (
    "Tuve problemas de digestión en donde me hicieron un estudio y tenía gastritis, "
    "nada severo, pero tenía gastritis y me mandaron tres meses con pastillas."
)
WRONG_TAKE = "Tuve problemas de estómago en una temporada, en 2023, hay que voltar."


def _clip(clip_id, text, *, selected, start, end, attempt_id, source_asset_id="src"):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0, start=start, end=end,
        text=text, caption_text=text, selected=selected, attempt_id=attempt_id,
    )


def _draft(selected, discarded, diagnostics):
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=tuple(selected), alternates=(), discarded=tuple(discarded), diagnostics=dict(diagnostics),
    )


def _confirmation(kind="wrong_take", confidence=0.9686, take_id="c_wrong"):
    return {
        "source_asset_id": "src", "take_id": take_id, "retry_take_id": "c_restart",
        "retry_similarity": 0.687, "gap_sec": 0.74,
        "candidate_event_kinds": ["facial_expression_shift_candidate", "hand_motion_reset_candidate"],
        "confirmed_kind": kind, "confidence": confidence,
    }


def _clean_cut(keep=False, reason="whole_video_bad_take:wrong_take"):
    return {"clip_id": "c_wrong", "keep": keep, "reason": reason, "confidence": 0.9686}


def _scenario(*, confirmation=None, clean_cut=None, delivery_start=6.0, wrong_text=WRONG_TAKE):
    wrong = _clip("c_wrong", wrong_text, selected=False, start=0.0, end=5.0, attempt_id="att_1")
    delivery = _clip("c_delivery", DELIVERY, selected=True, start=delivery_start, end=delivery_start + 9.0, attempt_id="att_2")
    diagnostics = {
        "performance_confirmation": [confirmation] if confirmation is not None else [],
        "clean_cut_decisions": [clean_cut] if clean_cut is not None else [],
    }
    return _draft([delivery], [wrong], diagnostics)


def _row(validated):
    return next(f for f in validated.diagnostics["final_story_coherence_validation"]["lost_semantic_atoms"] if f["clip_id"] == "c_wrong")


def test_a_confirmed_wrong_take_removed_by_clean_cut_is_credited_without_an_arbiter():
    validated = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(), clean_cut=_clean_cut()))
    row = _row(validated)
    assert row["blocking"] is False
    assert row["content_loss_suppressed_by"] == PRE_GROUP_WRONG_TAKE_CONFIRMATION
    consult = row["pre_group_restart_consultations"][0]
    assert consult["relation"] == "wrong_take_confirmation" and consult["neighbour_clip_id"] == "c_delivery"
    assert consult["provider"] == "deterministic"
    assert row["atom_classifications"][0]["importance"] == "CONTEXTUAL"  # atoms untouched
    assert validated.diagnostics["final_story_coherence_validation"]["freeze_blocked"] is False


def test_a_lone_retry_setup_is_not_enough():
    validated = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(kind="retry_setup", confidence=0.86), clean_cut=_clean_cut()))
    assert _row(validated)["blocking"] is True


def test_a_confirmation_below_the_floor_is_not_enough():
    validated = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(confidence=0.80), clean_cut=_clean_cut()))
    assert _row(validated)["blocking"] is True


def test_the_clean_cut_stage_must_have_removed_the_take_for_that_reason():
    kept = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(), clean_cut=_clean_cut(keep=True)))
    assert _row(kept)["blocking"] is True
    other = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(), clean_cut=_clean_cut(reason="whole_video_bad_take:other")))
    assert _row(other)["blocking"] is True
    none = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(), clean_cut=None))
    assert _row(none)["blocking"] is True


def test_the_retry_adjacency_to_a_selected_delivery_is_still_required():
    far = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(), clean_cut=_clean_cut(), delivery_start=40.0))
    assert _row(far)["blocking"] is True


def test_a_critical_atom_is_never_credited_by_the_confirmation():
    critical = "Tuve problemas de estómago y me mandaron 6 meses con pastillas, hay que voltar."
    validated = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(), clean_cut=_clean_cut(), wrong_text=critical))
    row = _row(validated)
    assert row["blocking"] is True and "6" in row["missing_critical_atoms"]


def test_the_confirmation_must_name_this_clip():
    validated = apply_final_story_coherence_validation(_scenario(confirmation=_confirmation(take_id="c_other"), clean_cut=_clean_cut()))
    assert _row(validated)["blocking"] is True
