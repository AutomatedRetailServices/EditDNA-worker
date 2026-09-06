from cutsell_worker.contracts import CandidateTake, MediaSignals
from cutsell_worker.hybrid_failed_continuation_integrity import collapse_failed_split_retry_continuations
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
from cutsell_worker.hybrid_story_guard import restore_hybrid_story_coverage


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=(),
        signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def test_video00_hybrid_story_guard_cannot_resurrect_corroborated_failed_retry():
    failed = _take(
        "failed",
        25.60,
        32.42,
        "Nunca se nos ocurrió hacer un chequeo de sonografía de la tiroides pues porque cada año que me hacía mínimo dos estados",
    )
    winner = _take(
        "winner",
        35.46,
        45.54,
        "Nunca se nos ocurrió hacer un chequeo de la tiroides por sonografía porque siempre en mis exámenes la tiroides salía funcionando perfectamente",
    )
    result = HybridSessionCleanupResult(
        kept=(winner,),
        deleted=(failed,),
        requested_chunk_count=1,
        available_chunk_count=1,
        diagnostics=({
            "decisions": [{
                "clip_id": "failed",
                "label": "failed",
                "confidence": 0.88,
                "applied_delete": True,
                "delete_basis": "semantic_failed_plus_local_performance",
            }],
        },),
        semantic_decisions=(("failed", "failed", 0.88), ("winner", "winner", 0.94)),
    )

    guarded = restore_hybrid_story_coverage((failed, winner), result, None)

    assert tuple(t.clip_id for t in guarded.kept) == ("winner",)
    assert tuple(t.clip_id for t in guarded.deleted) == ("failed",)


def test_video00_failed_hereditary_split_take_removes_prefix_and_unjudged_continuation():
    winner = _take(
        "winner",
        295.36,
        314.60,
        "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer por eso no creo que los cánceres sean hereditarios más bien solo un 5-10% son de carácter hereditario",
    )
    failed = _take(
        "failed",
        319.38,
        334.24,
        "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene carcinoma papilar ni sufre de la tiroides así que estoy convencida que solo un 5 -10 % de los",
        complete=False,
    )
    continuation = _take(
        "continuation",
        335.88,
        346.54,
        "cánceres son hereditarios soy la única que tiene este tipo de cáncer",
    )

    kept, removed, diagnostics = collapse_failed_split_retry_continuations(
        (winner, failed, continuation),
        (("winner", "winner", 0.96), ("failed", "failed", 0.82)),
    )

    assert tuple(t.clip_id for t in kept) == ("winner",)
    assert {t.clip_id for t in removed} == {"failed", "continuation"}
    assert diagnostics[0]["winner_clip_id"] == "winner"
    assert diagnostics[0]["critical_preserved"] is True
