from cutsell_worker.contracts import CandidateTake
from cutsell_worker.superseded_attempt_cleanup import remove_superseded_attempts


def _take(clip_id: str, start: float, end: float, text: str, *, complete_idea: bool) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete_idea,
    )


def test_complete_concise_setup_is_not_removed_just_because_later_take_is_longer():
    setup = _take(
        "setup",
        10.0,
        15.5,
        "También me salían espinillas. Era como un rash, una alergia.",
        complete_idea=True,
    )
    later = _take(
        "later",
        16.0,
        28.7,
        "También me salían espinillas detrás de la oreja y por el cuello; yo pensaba que era alergia, pero parecía hormonal.",
        complete_idea=True,
    )

    kept, removed, diagnostics = remove_superseded_attempts((setup, later))

    assert [take.clip_id for take in kept] == ["setup", "later"]
    assert removed == ()
    assert diagnostics == ()


def test_incomplete_short_attempt_can_still_yield_to_full_later_retake():
    partial = _take(
        "partial",
        10.0,
        13.0,
        "También me salían espinillas como una alergia",
        complete_idea=False,
    )
    later = _take(
        "later",
        14.0,
        24.0,
        "También me salían espinillas como una alergia detrás de la oreja y por el cuello durante ciertas temporadas.",
        complete_idea=True,
    )

    kept, removed, diagnostics = remove_superseded_attempts((partial, later))

    assert [take.clip_id for take in kept] == ["later"]
    assert [take.clip_id for take in removed] == ["partial"]
    assert diagnostics and diagnostics[0]["reason"] == "superseded_partial_attempt"
