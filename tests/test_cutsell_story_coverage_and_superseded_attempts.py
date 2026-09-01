from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.story_coverage_guard import restore_unique_story_coverage
from cutsell_worker.superseded_attempt_cleanup import remove_superseded_attempts


def _take(clip_id, start, end, text, *, complete=True, signals=None):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        complete_idea=complete,
        signals=signals or MediaSignals("src", float(start), float(end)),
    )


def test_unique_long_story_paragraph_is_restored_when_no_retry_or_strong_failure_exists():
    unique = _take(
        "unique",
        10.0,
        19.0,
        "I changed doctors and asked for every test she could imagine because my metabolism had slowed down",
    )
    other = _take(
        "other",
        20.0,
        28.0,
        "The ultrasound later showed a suspicious thyroid nodule that needed a biopsy",
    )
    kept, discarded, diagnostics = restore_unique_story_coverage(
        (other,), (unique,), (unique, other), None
    )
    assert {item.clip_id for item in kept} == {"unique", "other"}
    assert discarded == ()
    assert diagnostics[0]["reason"] == "restore_unique_story_coverage"


def test_long_retry_is_not_restored_when_same_idea_has_competing_peer():
    failed = _take(
        "failed",
        10.0,
        19.0,
        "At the end of my contract I asked my doctor for every possible test because I wanted answers",
    )
    retry = _take(
        "retry",
        22.0,
        33.0,
        "At the end of my contract I changed doctors and asked my doctor for every possible test because I wanted real answers",
    )
    kept, discarded, diagnostics = restore_unique_story_coverage(
        (retry,), (failed,), (failed, retry), None
    )
    assert kept == (retry,)
    assert discarded == (failed,)
    assert diagnostics == ()


def test_partial_attempt_is_removed_when_later_fuller_delivery_covers_same_content():
    partial = _take(
        "partial",
        10.0,
        14.0,
        "only two a day helps get those nutrients",
        complete=False,
    )
    full = _take(
        "full",
        25.0,
        36.0,
        "only two of these a day helps them get those nutrients their bodies are missing with no artificial flavors",
    )
    kept, removed, diagnostics = remove_superseded_attempts((partial, full))
    assert kept == (full,)
    assert removed == (partial,)
    assert diagnostics[0]["superseding_clip_id"] == "full"


def test_unique_short_audience_line_fails_open_when_no_fuller_retry_exists():
    line = _take("line", 10.0, 13.0, "this video was so much fun", complete=True)
    next_line = _take("next", 16.0, 22.0, "people could tell we had real chemistry on camera", complete=True)
    kept, removed, diagnostics = remove_superseded_attempts((line, next_line))
    assert kept == (line, next_line)
    assert removed == ()
    assert diagnostics == ()


def test_open_micro_fragment_can_drop_before_nearby_full_delivery():
    fragment = _take("fragment", 0.0, 1.4, "worried if", complete=False)
    full = _take(
        "full",
        6.0,
        13.0,
        "your kids are picky eaters and you are worried they are missing important nutrients",
        complete=True,
    )
    kept, removed, diagnostics = remove_superseded_attempts((fragment, full))
    assert kept == (full,)
    assert removed == (fragment,)
    assert diagnostics[0]["reason"] == "superseded_partial_attempt"
