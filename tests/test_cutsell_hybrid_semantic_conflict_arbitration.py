from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_semantic_conflict_arbitration import conflict_restore_ids


def take(clip_id, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=0.0,
        end=8.0,
        text="complete delivery",
        complete_idea=complete,
    )


def test_stronger_winner_conflict_restores_complete_deleted_clip():
    rows = (("c", "winner", 0.95), ("c", "failed", 0.90))
    assert conflict_restore_ids((take("c"),), {"c"}, rows) == {"c"}


def test_failed_only_does_not_restore():
    rows = (("c", "failed", 0.95),)
    assert conflict_restore_ids((take("c"),), {"c"}, rows) == set()


def test_incomplete_clip_does_not_restore_even_when_conflicted():
    rows = (("c", "winner", 0.95), ("c", "failed", 0.90))
    assert conflict_restore_ids((take("c", complete=False),), {"c"}, rows) == set()
