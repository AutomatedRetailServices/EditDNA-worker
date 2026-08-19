from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
from cutsell_worker.hybrid_story_guard import restore_hybrid_story_coverage
from cutsell_worker.semantic_fragment_guard import remove_semantic_fragment_debris
from cutsell_worker.session_boundaries import safe_group_takes_by_sessions


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        complete_idea=complete,
        signals=MediaSignals("src", float(start), float(end)),
    )


def test_medium_confidence_hybrid_failure_cannot_erase_unique_long_story():
    story = _take(
        "story",
        6.56,
        18.02,
        "There is no internal pump. The magic happens from this little doodad, which is magnetic, and the mechanism works without a traditional pump.",
    )
    result = HybridSessionCleanupResult(
        kept=(),
        deleted=(story,),
        requested_chunk_count=1,
        available_chunk_count=1,
        diagnostics=(),
        semantic_decisions=(("story", "failed", 0.88),),
    )

    repaired = restore_hybrid_story_coverage((story,), result, None)

    assert repaired.kept == (story,)
    assert repaired.deleted == ()
    assert repaired.diagnostics[-1]["restored_ids"] == ["story"]


def test_very_high_confidence_hybrid_failure_remains_deleted():
    story = _take(
        "story-hard-fail",
        6.56,
        18.02,
        "There is no internal pump. The magic happens from this little doodad, which is magnetic, and the mechanism works without a traditional pump.",
    )
    result = HybridSessionCleanupResult(
        kept=(),
        deleted=(story,),
        requested_chunk_count=1,
        available_chunk_count=1,
        diagnostics=(),
        semantic_decisions=(("story-hard-fail", "failed", 0.99),),
    )

    repaired = restore_hybrid_story_coverage((story,), result, None)

    assert repaired.kept == ()
    assert repaired.deleted == (story,)


def test_failed_repeated_compound_is_removed_without_visual_corroboration():
    broken = _take(
        "non-gmo-loop",
        16.84,
        22.92,
        "non-gmo non-gmo non-gmo gluten-free and be they're not eating if they're not eating",
        complete=False,
    )

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (broken,),
        (("non-gmo-loop", "failed", 0.75),),
    )

    assert kept == ()
    assert removed == (broken,)
    assert diagnostics[0]["reason"] == "semantic_failed_repetition_pathology"


def test_failed_longer_repeated_phrase_is_removed_without_visual_corroboration():
    broken = _take(
        "eating-loop",
        22.92,
        27.54,
        "if they're not eating if they're not eating health already worried that their kid",
        complete=False,
    )

    kept, removed, diagnostics = remove_semantic_fragment_debris(
        (broken,),
        (("eating-loop", "failed", 0.80),),
    )

    assert kept == ()
    assert removed == (broken,)
    assert diagnostics[0]["reason"] == "semantic_failed_repetition_pathology"


def test_session_scoped_path_uses_local_reformulated_retry_reconciliation():
    first = _take(
        "first",
        82.74,
        90.04,
        "Al terminar mi contrato hablé con mi ginecóloga, le pedí todos los test que ella pudiera imaginarse o que me pudiera indicar.",
    )
    retry = _take(
        "retry",
        95.60,
        104.32,
        "Al terminar mi contrato, cambié de ginecóloga y le pedí que me hiciera un test de todo lo que ella se pudiera imaginar y me pudiese indicar.",
    )

    result = safe_group_takes_by_sessions(None, (first, retry), None)

    assert len(result.groups) == 1
    assert set(result.groups[0]) == {"first", "retry"}
