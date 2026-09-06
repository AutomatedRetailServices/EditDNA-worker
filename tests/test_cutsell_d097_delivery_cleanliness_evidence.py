"""D-097 (PO adjustment §2) -- MEASURED CLEANLINESS EVIDENCE IN BEST TAKE.

"A missing signal is not a confirmed clean take." Two takes carrying the
same message: the one with a REAL delivery failure (accidental interior
dead air >= 1.2 s, the post-render QC's own threshold, or a multimodal
reset -- strong body/hand reset AND an independent disengagement/face break
inside the take) must lose to the clean one. Negative controls: a single
gesture, a single glance, a natural pause under the threshold, silence at
the take's edges, or an event from another source never penalise.
Generic fixtures only.
"""
from cutsell_worker.contracts import CandidateTake, ProcessingRequest, RankedTake, SourceAsset
from cutsell_worker.pipeline import build_flow_b_draft
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.take_judge import apply_delivery_cleanliness_evidence, delivery_cleanliness_evidence, rank_takes
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext

TEXT = "This serum cleared my skin in two weeks and I will keep using it every night."


def _take(clip_id, start, text=TEXT, source="src"):
    return CandidateTake(clip_id, source, 0, start, start + 8.0, text)


def _event(kind, start, end, confidence=0.95, source="src"):
    return TemporalEvent(source_asset_id=source, start=start, end=end, kind=kind, confidence=confidence, description="")


def test_accidental_interior_dead_air_penalises_the_dirty_take():
    dirty, clean = _take("dirty", 0.0), _take("clean", 20.0)
    events = [_event("audio_silence_interval", 2.0, 3.6, 1.0)]
    baseline = rank_takes((dirty, clean))
    assert baseline[0].score == baseline[1].score  # same message, same baseline
    ranked, rows = apply_delivery_cleanliness_evidence(baseline, (dirty, clean), events)
    assert ranked[0].clip_id == "clean"
    dirty_row = next(r for r in rows if r["clip_id"] == "dirty")
    assert dirty_row["interior_dead_air_count"] == 1 and "interior_dead_air_penalty" in dirty_row["reasons"]
    assert "interior_dead_air_penalty" in next(r for r in ranked if r.clip_id == "dirty").reason


def test_multimodal_reset_inside_the_take_penalises_it():
    dirty, clean = _take("dirty", 0.0), _take("clean", 20.0)
    events = [
        _event("body_reset_candidate", 3.0, 3.4, 0.92),
        _event("facial_expression_shift_candidate", 3.5, 3.9, 0.80),
    ]
    ranked, rows = apply_delivery_cleanliness_evidence(rank_takes((dirty, clean)), (dirty, clean), events)
    assert ranked[0].clip_id == "clean"
    assert next(r for r in rows if r["clip_id"] == "dirty")["multimodal_reset"] is True


def test_negative_controls_never_penalise():
    take = _take("t", 0.0)
    controls = {
        "single_gesture": [_event("hand_motion_reset_candidate", 3.0, 3.3, 0.95)],
        "single_glance": [_event("camera_disengagement_candidate", 3.0, 3.3, 0.95)],
        "natural_pause_under_threshold": [_event("audio_silence_interval", 3.0, 4.1, 1.0)],
        "silence_at_the_entry_edge": [_event("audio_silence_interval", 0.05, 1.5, 1.0)],
        "silence_at_the_exit_edge": [_event("audio_silence_interval", 6.6, 7.95, 1.0)],
        "other_source": [_event("audio_silence_interval", 2.0, 4.0, 1.0, source="other")],
        "weak_reset_and_break": [_event("body_reset_candidate", 3.0, 3.4, 0.70), _event("facial_expression_shift_candidate", 3.5, 3.9, 0.80)],
    }
    for name, events in controls.items():
        row = delivery_cleanliness_evidence(take, events)
        assert row["penalty"] == 0.0 and row["reasons"] == [], name


def test_baseline_reason_and_order_are_preserved_without_evidence():
    a, b = _take("a", 0.0), _take("b", 20.0, text="A shorter different line about the price.")
    baseline = rank_takes((a, b))
    ranked, rows = apply_delivery_cleanliness_evidence(baseline, (a, b), [])
    assert ranked == baseline
    assert all(r["penalty"] == 0.0 for r in rows)


def test_penalty_is_capped_and_visible():
    take = _take("t", 0.0)
    events = [_event("audio_silence_interval", 1.0, 2.5, 1.0), _event("audio_silence_interval", 3.0, 4.5, 1.0),
              _event("audio_silence_interval", 5.0, 6.5, 1.0)]
    row = delivery_cleanliness_evidence(take, events)
    assert row["interior_dead_air_count"] == 3 and row["penalty"] == 0.24


def test_pipeline_family_ranking_uses_the_evidence_and_records_it():
    source = SourceAsset(source_asset_id="src", project_id="p", user_id="u", original_name="raw.mp4",
                         source_order=0, duration_sec=60.0, uri="s3://b/raw.mp4")
    dirty, clean = _take("dirty", 0.0), _take("clean", 20.0)
    context = WholeVideoContext(
        sources=(SourceVideoContext("src", "", "", "", events=(_event("audio_silence_interval", 2.0, 3.6, 1.0),)),),
        status=ProviderStatus("whole_video", True, True, "applied", None),
    )
    request = ProcessingRequest(project_id="p", user_id="u", sources=(source,))
    result = build_flow_b_draft(request, (dirty, clean), whole_video_context=context)
    assert [c.clip_id for c in result.draft.selected] == ["clean"]
    group = result.draft.diagnostics["take_judge_groups"][0]
    assert any(r["clip_id"] == "dirty" and r["penalty"] > 0 for r in group["delivery_cleanliness"])
    assert "interior_dead_air_penalty" in next(r for r in group["ranked"] if r["clip_id"] == "dirty")["reason"]
