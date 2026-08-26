from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_session_cleanup import _later_semantic_retry_replacement


def _take(clip_id, start, end, text, *, complete_idea=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete_idea,
    )


def test_complete_topic_continuation_is_not_mistaken_for_retake():
    failed = _take(
        "failed",
        10.0,
        14.0,
        "a hacer sonografia de tiroides y otras sonografias",
        complete_idea=True,
    )
    continuation = _take(
        "continuation",
        18.0,
        24.0,
        "en la sonografia de tiroides aparecio un nodulo sospechoso de 3 centimetros que se mando a biopsia",
        complete_idea=True,
    )
    decisions = {
        "failed": ("failed", 0.85),
        "continuation": ("winner", 0.95),
    }

    replacement, overlap = _later_semantic_retry_replacement(
        failed,
        (failed, continuation),
        decisions,
    )

    assert overlap == 0.0
    assert replacement is None


def test_complete_same_delivery_can_still_be_superseded_by_strong_retake():
    failed = _take(
        "failed",
        10.0,
        14.0,
        "ahi fue cuando me mandaron a hacer sonografias de tiroides y otros",
        complete_idea=True,
    )
    retake = _take(
        "retake",
        18.0,
        23.0,
        "a hacer sonografia de tiroides y otras sonografias",
        complete_idea=True,
    )
    decisions = {
        "failed": ("failed", 0.85),
        "retake": ("winner", 0.95),
    }

    replacement, overlap = _later_semantic_retry_replacement(
        failed,
        (failed, retake),
        decisions,
    )

    assert replacement is retake
    assert overlap >= 0.64


def test_complete_retry_cannot_drop_existing_numeric_fact():
    failed = _take(
        "failed",
        10.0,
        14.0,
        "el nodulo media 3 centimetros y se mando a biopsia",
        complete_idea=True,
    )
    retake_without_number = _take(
        "retake",
        18.0,
        23.0,
        "el nodulo se mando a biopsia porque era sospechoso",
        complete_idea=True,
    )
    decisions = {
        "failed": ("failed", 0.90),
        "retake": ("winner", 0.95),
    }

    replacement, overlap = _later_semantic_retry_replacement(
        failed,
        (failed, retake_without_number),
        decisions,
    )

    assert overlap == 0.0
    assert replacement is None


def test_incomplete_retry_keeps_looser_partial_match_behavior():
    failed = _take(
        "failed",
        10.0,
        12.0,
        "me mandaron hacer sonografia tiroides",
        complete_idea=False,
    )
    retake = _take(
        "retake",
        14.0,
        19.0,
        "ahi fue cuando me mandaron a hacer sonografia de tiroides completa",
        complete_idea=True,
    )
    decisions = {
        "failed": ("failed", 0.85),
        "retake": ("winner", 0.95),
    }

    replacement, overlap = _later_semantic_retry_replacement(
        failed,
        (failed, retake),
        decisions,
    )

    assert replacement is retake
    assert overlap >= 0.50
