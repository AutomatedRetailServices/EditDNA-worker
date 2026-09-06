"""D-097.6 (R7) -- the D-097 §3 usability rule holds at the legacy pre-resolver
guards too: a hybrid `failed` label at or above the Resolver's unusable floor
(0.85) is never restored on lexical "unique tail" evidence and is never a
composite member.

RAW 34042123557 (head 18a2707): the clean retry "al terminar mi contrato
cambié de ginecóloga ..." was labelled `winner` 0.95, the abandoned full
attempt `alternate` 0.85 and the abandoned start "Al terminar mi contrato le
pedía a mi ginecóloga" `failed` 0.95. `hybrid_cross_group_retry_integrity`
correctly deleted both abandoned attempts as covered by the winner; then
`hybrid_complementary_delivery_guard` RESTORED both because inflected forms
("pedía" vs "pedí", "imaginarse" vs "imaginar", "hablé" vs "cambié") counted
as a unique tail, and `hybrid_composite_best_take` composed the two abandoned
attempts to REPLACE the winner. The clean delivery left the edit, 10.45 s of
abandoned attempts played, and nothing downstream could contest it because
the winner was gone before IdeaClusterer ever ran.
"""
from __future__ import annotations

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_complementary_delivery_guard import _restore_complementary_cross_group_deletions
from cutsell_worker.hybrid_composite_best_take import _choose_composite_replacements, _restore_performance_only_unique_deliveries


def take(clip_id: str, start: float, end: float, text: str, *, complete: bool = True) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id, source_asset_id="src", source_order=0, start=start, end=end, text=text,
        complete_idea=complete,
    )


ABANDONED_FULL = take("abandoned_full", 82.8, 90.6, "When my contract ended I spoke with my gynecologist and asked for all the tests she could imagine or indicate.")
ABANDONED_START = take("abandoned_start", 91.2, 94.3, "When my contract ended I was asking my gynecologist")
CLEAN_RETRY = take("clean_retry", 95.5, 104.3, "When my contract ended I changed gynecologist and asked her for a test of everything she could imagine and indicate.")


def test_a_failed_label_at_the_unusable_floor_is_not_restored_on_a_unique_tail():
    semantic = {
        "abandoned_full": ("alternate", 0.85),
        "abandoned_start": ("failed", 0.95),
        "clean_retry": ("winner", 0.95),
    }
    restored, rows = _restore_complementary_cross_group_deletions(
        (CLEAN_RETRY,), (ABANDONED_FULL, ABANDONED_START), semantic, {"abandoned_full", "abandoned_start"},
    )
    assert "abandoned_start" not in restored
    assert all(row["clip_id"] != "abandoned_start" for row in rows)


def test_a_low_confidence_failed_label_is_still_restorable():
    semantic = {"abandoned_start": ("failed", 0.70), "clean_retry": ("winner", 0.95)}
    # Lengthen the candidate so the guard's own duration/content floors apply as before.
    candidate = take("abandoned_start", 91.2, 94.6, "When my contract ended I was asking my gynecologist for every test possible")
    restored, _rows = _restore_complementary_cross_group_deletions((CLEAN_RETRY,), (candidate,), semantic, {"abandoned_start"})
    assert restored == {"abandoned_start"}


def test_a_composite_never_includes_a_failed_member_at_the_unusable_floor():
    semantic = {
        "abandoned_full": ("alternate", 0.85),
        "abandoned_start": ("failed", 0.95),
        "clean_retry": ("winner", 0.95),
    }
    restored_rows = [
        {"clip_id": "abandoned_full", "peer_clip_id": "clean_retry"},
        {"clip_id": "abandoned_start", "peer_clip_id": "clean_retry"},
    ]
    suppressed, split_ids, rows = _choose_composite_replacements(
        (ABANDONED_FULL, ABANDONED_START, CLEAN_RETRY), semantic, restored_rows,
    )
    assert suppressed == set()
    assert split_ids == set()
    assert rows == []


def test_the_performance_only_rescue_stops_at_the_unusable_floor():
    winner = take("winner", 20.0, 30.0, "I also had bumps behind my ear and on my neck that looked like an allergy.")
    candidate = take("candidate", 32.0, 38.0, "I also had bumps behind my ear and neck like an allergy and they came in seasons.")
    decisions = {"candidate": {
        "clip_id": "candidate", "applied_delete": True, "reason_code": "",
        "delete_basis": "semantic_failed_plus_local_performance",
        "local_failure_reasons": ["dense_physical_reset:6", "visual_fumble:0.85"],
    }}
    restored, _rows = _restore_performance_only_unique_deliveries(
        (winner,), (candidate,), {"winner": ("winner", 0.93), "candidate": ("failed", 0.90)}, decisions,
    )
    assert restored == set()
    restored, _rows = _restore_performance_only_unique_deliveries(
        (winner,), (candidate,), {"winner": ("winner", 0.93), "candidate": ("failed", 0.80)}, decisions,
    )
    assert restored == {"candidate"}
