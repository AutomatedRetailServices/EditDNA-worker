from cutsell_worker.contracts import CandidateTake, MediaSignals, Word
from cutsell_worker.hybrid_gold_reconciliation import reconcile_human_gold_hybrid
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
from cutsell_worker.providers import ProviderStatus
from cutsell_worker.whole_video_analysis import SourceVideoContext, TemporalEvent, WholeVideoContext


def _take(clip_id, start, end, text, *, complete=True):
    tokens = text.split()
    step = max(0.08, (end - start) / max(1, len(tokens)))
    words = []
    cursor = float(start)
    for token in tokens:
        words.append(Word(token, cursor, min(float(end), cursor + step), 0.97))
        cursor += step
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=float(start),
        end=float(end),
        text=text,
        words=tuple(words),
        signals=MediaSignals("src", float(start), float(end)),
        complete_idea=complete,
    )


def _context(*events):
    return WholeVideoContext(
        sources=(SourceVideoContext(
            source_asset_id="src",
            summary="raw talking head",
            dominant_style="talking_head",
            creator_intent="recording clean cut",
            events=tuple(events),
        ),),
        status=ProviderStatus("test", True, True, "applied"),
    )


def test_round3_restores_clean_retake_and_removes_locally_failed_previous_attempt():
    broken = _take(
        "broken",
        108.56,
        111.86,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides",
        complete=True,
    )
    clean = _take(
        "clean",
        120.11,
        124.15,
        "a hacer sonografías de tiroides y otras sonografías",
        complete=True,
    )
    following = _take(
        "following",
        128.16,
        134.22,
        "En la sonografía de tiroides apareció un nódulo sospechoso de tres centímetros",
        complete=True,
    )
    result = HybridSessionCleanupResult(
        kept=(broken, following),
        deleted=(clean,),
        requested_chunk_count=1,
        available_chunk_count=1,
        diagnostics=({
            "hybrid_retry_completion_integrity": [{
                "clip_id": "clean",
                "reason": "semantic_short_alternate_covered_by_neighbors",
            }],
        },),
        semantic_decisions=(("broken", "keep", 0.85), ("clean", "alternate", 0.85), ("following", "keep", 0.90)),
    )
    context = _context(
        TemporalEvent("src", 112.01, 112.34, "retry_setup", 0.86, "creator resets and retries same delivery"),
    )

    fixed = reconcile_human_gold_hybrid(result, (broken, clean, following), context)

    assert {take.clip_id for take in fixed.kept} == {"clean", "following"}
    assert "broken" in {take.clip_id for take in fixed.deleted}
    assert "clean" not in {take.clip_id for take in fixed.deleted}
    repairs = fixed.diagnostics[-1]["hybrid_gold_reconciliation"]
    assert repairs[0]["reason"] == "restore_clean_retake_remove_failed_previous"


def test_round4_restores_clean_retake_at_actual_overlap_window_confidence():
    """Exact Round 4 integration regression from the real Video 00 report.

    The clean 120.11-124.15 retake was alternate=0.75 in one overlapping Hybrid
    window and keep=0.80 in another. The semantic reducer retained the higher-priority
    alternate label, so the prior 0.80 Gold threshold prevented reconciliation and the
    clean retake was deleted as short-alternate debris while 108.56-111.86 survived.
    """
    broken = _take(
        "broken",
        108.56,
        111.86,
        "Ahí fue cuando me mandaron a hacer sonografías de tiroides",
        complete=True,
    )
    clean = _take(
        "clean",
        120.11,
        124.15,
        "a hacer sonografías de tiroides y otras sonografías",
        complete=True,
    )
    following = _take(
        "following",
        128.16,
        134.22,
        "En la sonografía de tiroides apareció un nódulo sospechoso de tres centímetros",
        complete=True,
    )
    result = HybridSessionCleanupResult(
        kept=(broken, following),
        deleted=(clean,),
        requested_chunk_count=3,
        available_chunk_count=3,
        diagnostics=({
            "hybrid_retry_completion_integrity": [{
                "clip_id": "clean",
                "reason": "semantic_short_alternate_covered_by_neighbors",
            }],
        },),
        semantic_decisions=(("broken", "alternate", 0.80), ("clean", "alternate", 0.75), ("following", "winner", 0.95)),
    )
    context = _context(
        TemporalEvent("src", 112.009, 112.34, "retry_setup", 0.86, "creator resets and retries same delivery"),
    )

    fixed = reconcile_human_gold_hybrid(result, (broken, clean, following), context)

    assert {take.clip_id for take in fixed.kept} == {"clean", "following"}
    assert {take.clip_id for take in fixed.deleted} == {"broken"}
    repair = fixed.diagnostics[-1]["hybrid_gold_reconciliation"][0]
    assert repair["reason"] == "restore_clean_retake_remove_failed_previous"
    assert repair["semantic_label"] == "alternate"
    assert repair["semantic_confidence"] == 0.75
    assert repair["retry_setup_confidence"] == 0.86


def test_round3_removes_orphan_continuation_after_deleted_incomplete_alternate():
    winner = _take(
        "winner",
        295.36,
        314.60,
        "Esta es mi experiencia soy la única en mi familia que tiene este tipo de cáncer por eso no creo que los cánceres son hereditarios más bien solo un 5 10 por ciento son de carácter hereditario mayormente son nuestras elecciones de vida así que cuídate",
        complete=True,
    )
    deleted_prefix = _take(
        "prefix",
        319.38,
        334.24,
        "Soy la primera en mi familia con este tipo de cáncer nadie en mi familia tiene un carcinoma papilar en la tiroides ni sufre de la tiroides así que estoy convencida que solo un 5 10 por ciento de los",
        complete=False,
    )
    continuation = _take(
        "continuation",
        335.88,
        346.54,
        "cánceres son hereditarios soy la única que tiene este tipo de cáncer",
        complete=True,
    )
    result = HybridSessionCleanupResult(
        kept=(winner, continuation),
        deleted=(deleted_prefix,),
        requested_chunk_count=1,
        available_chunk_count=1,
        diagnostics=({
            "hybrid_alternate_integrity": [{
                "clip_id": "prefix",
                "reason": "semantic_alternate_incomplete_retry_after_winner",
                "winner_clip_id": "winner",
            }],
        },),
        semantic_decisions=(("winner", "winner", 0.93), ("prefix", "alternate", 0.80)),
    )

    fixed = reconcile_human_gold_hybrid(result, (winner, deleted_prefix, continuation), _context())

    assert {take.clip_id for take in fixed.kept} == {"winner"}
    assert {take.clip_id for take in fixed.deleted} == {"prefix", "continuation"}
    repairs = fixed.diagnostics[-1]["hybrid_gold_reconciliation"]
    assert repairs[0]["reason"] == "remove_orphan_continuation_of_deleted_incomplete_alternate"
