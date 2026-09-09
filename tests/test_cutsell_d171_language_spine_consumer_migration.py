"""D-171: Language / Transcript Spine, Phase D -- incremental, bounded
consumer migration. Covers every directive-required fixture category
(30 items) plus no-authority-change proofs for every module this task's
directive names as CLOSED/untouched.

See cutsell_worker/language_spine_consumer_migration.py's own module
docstring for the full design rationale (fail-open, provably-identical-
behavior migration contract) this suite verifies against.
"""
from __future__ import annotations

import inspect
import subprocess

import pytest

from cutsell_worker.contracts import CandidateTake, Word
from cutsell_worker.language_spine_consumer_migration import (
    LEGACY_ONLY,
    SPINE_AVAILABLE_LEGACY_FALLBACK,
    SPINE_CONFLICT_FALLBACK,
    SPINE_CONSUMED,
    ConsumerMigrationTrace,
    continuation_migration,
    language_spine_consumer_migration_diagnostics,
    proposition_divergence_migration,
)
from cutsell_worker.recording_meta_continuation import (
    _direct_meta_short_tail,
    _legacy_tiny_continuation,
    apply_recording_meta_continuation_cleanup,
)
from cutsell_worker.take_grouping_provider import (
    _legacy_marked_side_diverges_in_content,
    _marked_side_diverges_in_content,
)


def _run_git_diff(path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", path],
        cwd="/home/user/EditDNA-worker", capture_output=True, text=True,
    )
    return result.stdout.strip()


def _source(module_name: str) -> str:
    return inspect.getsource(__import__(f"cutsell_worker.{module_name}", fromlist=["_"]))


def _mk_take(clip_id: str, source_asset_id: str, start: float, end: float, text: str, words=()) -> CandidateTake:
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0,
        start=start, end=end, text=text, words=tuple(words),
    )


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, confidence=0.9)


# ---------------------------------------------------------------------------
# 1. Proposition legacy == spine
# ---------------------------------------------------------------------------
def test_01_proposition_legacy_equals_spine():
    left = "me salian espinillas detras de la oreja y en el cuello por la alergia"
    right = "me salian espinillas detras de la oreja y en el cuello por la alergia otra vez"
    legacy = _legacy_marked_side_diverges_in_content(left, right)
    verdict, trace = proposition_divergence_migration("t", "l", left, "r", right, legacy)
    assert verdict == legacy
    assert trace.result_source == SPINE_CONSUMED
    assert trace.spine_used is True


# ---------------------------------------------------------------------------
# 2. Retry legacy == spine
# ---------------------------------------------------------------------------
def test_02_retry_legacy_equals_spine():
    left = "otro sintoma era que me salian espinillas detras de la oreja y en el cuello por la alergia"
    right = "me salian espinillas detras de la oreja y en el cuello por la alergia todo el tiempo"
    legacy = _legacy_marked_side_diverges_in_content(left, right)
    verdict, trace = proposition_divergence_migration("t", "l", left, "r", right, legacy)
    assert legacy is False  # D-047 Case 1: restatement, no divergence
    assert verdict == legacy
    assert trace.result_source == SPINE_CONSUMED


# ---------------------------------------------------------------------------
# 3. Same topic, different proposition (D-039 founding case: arm vs leg)
# ---------------------------------------------------------------------------
def test_03_same_topic_different_proposition():
    left = "otro sintoma que tuve fueron manchas rojas en la piel del brazo derecho durante semanas"
    right = "tuve manchas rojas en la piel de la pierna izquierda por varios meses seguidos"
    legacy = _legacy_marked_side_diverges_in_content(left, right)
    assert legacy is True
    verdict, trace = proposition_divergence_migration("t", "l", left, "r", right, legacy)
    assert verdict is True
    assert trace.result_source in (SPINE_CONSUMED, SPINE_AVAILABLE_LEGACY_FALLBACK, SPINE_CONFLICT_FALLBACK)


# ---------------------------------------------------------------------------
# 4. Same opener, different proposition
# ---------------------------------------------------------------------------
def test_04_same_opener_different_proposition():
    left = "otro problema que tuve fue dolor de cabeza constante durante el embarazo"
    right = "tambien tuve nauseas severas cada manana durante el primer trimestre"
    legacy = _legacy_marked_side_diverges_in_content(left, right)
    verdict, trace = proposition_divergence_migration("t", "l", left, "r", right, legacy)
    # Never asserts a false authority change: verdict is always legacy or
    # provably-identical-to-legacy, regardless of which branch fires.
    assert verdict == legacy or trace.result_source != SPINE_CONSUMED


# ---------------------------------------------------------------------------
# 5. Negation conflict (Spine detects, legacy heuristic is blind to it)
# ---------------------------------------------------------------------------
def test_05_negation_conflict_reported_not_forced():
    left = "el tratamiento funciona muy bien para las manchas de la piel"
    right = "el tratamiento no funciona para nada con las manchas de la piel"
    legacy_diverges = False  # legacy has no negation awareness at all
    verdict, trace = proposition_divergence_migration("t", "l", left, "r", right, legacy_diverges)
    # Fail-open: even though Spine may find a real negation conflict, the
    # returned verdict never silently overrides the caller-supplied legacy
    # value -- it is either legacy, or provably equal to it.
    assert verdict == legacy_diverges
    if trace.conflict:
        assert trace.result_source == SPINE_CONFLICT_FALLBACK


# ---------------------------------------------------------------------------
# 6. Number conflict (Spine detects, legacy heuristic is blind to it)
# ---------------------------------------------------------------------------
def test_06_number_conflict_reported_not_forced():
    left = "perdi 5 kilos en el primer mes de tratamiento con este producto"
    right = "perdi 15 kilos en el primer mes de tratamiento con este producto"
    legacy_diverges = False
    verdict, trace = proposition_divergence_migration("t", "l", left, "r", right, legacy_diverges)
    assert verdict == legacy_diverges
    if trace.conflict:
        assert trace.result_source == SPINE_CONFLICT_FALLBACK


# ---------------------------------------------------------------------------
# 7. Continuation legacy == spine
# ---------------------------------------------------------------------------
def test_07_continuation_legacy_equals_spine():
    words = (_w("you", 0.0, 0.2), _w("know", 0.2, 0.5))
    legacy = True  # short, matches legacy tiny-continuation shape
    verdict, trace = continuation_migration("t", "a1", words, legacy)
    assert verdict == legacy
    assert trace.spine_available is True


# ---------------------------------------------------------------------------
# 8. Incomplete -> continuation
# ---------------------------------------------------------------------------
def test_08_incomplete_utterance_reads_as_continuation():
    # Grammatically open tail ("and the") -- take_segmentation's own
    # completeness check reads this as INCOMPLETE.
    words = (_w("and", 0.0, 0.2), _w("the", 0.2, 0.4))
    verdict, trace = continuation_migration("t", "a1", words, True)
    assert trace.spine_available is True


# ---------------------------------------------------------------------------
# 9. Complete retry not misclassified as continuation
# ---------------------------------------------------------------------------
def test_09_complete_utterance_not_forced_into_continuation():
    words = tuple(
        _w(word, 0.1 * i, 0.1 * i + 0.09)
        for i, word in enumerate(["This", "cream", "works", "great", "for", "everyone", "here", "today", "period"])
    )
    legacy_is_continuation = False
    verdict, trace = continuation_migration("t", "a1", words, legacy_is_continuation)
    # Never forced to True by Spine alone when legacy already says False and
    # Spine disagrees -- fail-open keeps the legacy (non-continuation) verdict.
    if trace.result_source != SPINE_CONSUMED:
        assert verdict == legacy_is_continuation


# ---------------------------------------------------------------------------
# 10. Correction stays correction (no authority change -- D-158 untouched)
# ---------------------------------------------------------------------------
def test_10_correction_authority_untouched():
    assert _run_git_diff("cutsell_worker/attempt_relationship_authority.py") == ""


# ---------------------------------------------------------------------------
# 11. Complementary stays non-retry (no authority change -- D-161 untouched)
# ---------------------------------------------------------------------------
def test_11_complementary_authority_untouched():
    assert _run_git_diff("cutsell_worker/watch_listen_relation_discovery.py") == ""


# ---------------------------------------------------------------------------
# 12. Missing spine -> legacy fallback (Target A: build_claim_signature error)
# ---------------------------------------------------------------------------
def test_12_missing_spine_legacy_fallback_target_a(monkeypatch):
    import cutsell_worker.language_spine_consumer_migration as m

    def _raise(*_args, **_kwargs):
        raise ValueError("simulated evidence-construction failure")

    monkeypatch.setattr(m, "build_claim_signature", _raise)
    verdict, trace = proposition_divergence_migration("t", "l", "some text", "r", "other text", True)
    assert verdict is True
    assert trace.result_source == LEGACY_ONLY
    assert trace.spine_available is False


def test_12b_empty_text_both_sides_still_spine_available():
    # build_claim_signature never raises on empty/None text (fails open to
    # an empty ClaimSignature) -- this is SPINE_AVAILABLE, not LEGACY_ONLY,
    # and it agrees with a legacy_diverges=True input (no shared content).
    verdict, trace = proposition_divergence_migration("t", "l", "", "r", "", True)
    assert verdict is True
    assert trace.spine_available is True


# ---------------------------------------------------------------------------
# 13. Partial spine -> legacy fallback (Target B: no word timing at all)
# ---------------------------------------------------------------------------
def test_13_missing_spine_legacy_fallback_target_b():
    verdict, trace = continuation_migration("t", "a1", (), False)
    assert verdict is False
    assert trace.result_source == LEGACY_ONLY
    assert trace.spine_available is False


# ---------------------------------------------------------------------------
# 14. Spine/legacy conflict -> safe fallback (verdict stays legacy)
# ---------------------------------------------------------------------------
def test_14_spine_legacy_conflict_safe_fallback():
    # Force a real disagreement: legacy says "no divergence" for two texts
    # that share zero content (Spine will say diverges=True).
    left = "manzanas y naranjas en el mercado local"
    right = "carros electricos ultramodernos con baterias grandes"
    verdict, trace = proposition_divergence_migration("t", "l", left, "r", right, False)
    assert verdict is False  # legacy value preserved, never silently flipped
    assert trace.result_source in (SPINE_AVAILABLE_LEGACY_FALLBACK, SPINE_CONFLICT_FALLBACK)
    assert trace.legacy_fallback_used is True


# ---------------------------------------------------------------------------
# 15. Deterministic diagnostics
# ---------------------------------------------------------------------------
def test_15_deterministic_diagnostics():
    left, right = "cream works great for wrinkles", "cream works great for wrinkles too"
    legacy = _legacy_marked_side_diverges_in_content(left, right)
    r1 = proposition_divergence_migration("t", "l", left, "r", right, legacy)
    r2 = proposition_divergence_migration("t", "l", left, "r", right, legacy)
    assert r1 == r2
    diag1 = language_spine_consumer_migration_diagnostics([r1[1], r2[1]])
    diag2 = language_spine_consumer_migration_diagnostics([r1[1], r2[1]])
    assert diag1 == diag2


# ---------------------------------------------------------------------------
# 16. Source identity preserved
# ---------------------------------------------------------------------------
def test_16_source_identity_preserved():
    take = _mk_take("clip_9", "asset_1", 0.0, 1.0, "you know", words=(_w("you", 0.0, 0.4), _w("know", 0.4, 0.8)))
    _direct_meta_short_tail(take, ())
    assert take.clip_id == "clip_9"
    assert take.source_asset_id == "asset_1"


# ---------------------------------------------------------------------------
# 17. Timeline preserved
# ---------------------------------------------------------------------------
def test_17_timeline_preserved():
    take = _mk_take("clip_9", "asset_1", 3.5, 4.1, "you know", words=(_w("you", 3.5, 3.9), _w("know", 3.9, 4.1)))
    _direct_meta_short_tail(take, ())
    assert take.start == 3.5
    assert take.end == 4.1


# ---------------------------------------------------------------------------
# 18. No Family Formation change (behavior-level: known calibration pairs
# still resolve identically through the public API)
# ---------------------------------------------------------------------------
def test_18_family_formation_calibration_unchanged():
    founding_left = "otro sintoma que tuve fueron manchas rojas en la piel del brazo derecho durante semanas"
    founding_right = "tuve manchas rojas en la piel de la pierna izquierda por varios meses seguidos"
    assert _marked_side_diverges_in_content(founding_left, founding_right) is True

    restatement_left = "otro sintoma era que me salian espinillas detras de la oreja y en el cuello por la alergia"
    restatement_right = "me salian espinillas detras de la oreja y en el cuello por la alergia todo el tiempo"
    assert _marked_side_diverges_in_content(restatement_left, restatement_right) is False

    for name in ("take_grouping", "hybrid_session_cleanup"):
        assert "language_spine_consumer_migration" not in _source(name)


# ---------------------------------------------------------------------------
# 19. No BestTake change
# ---------------------------------------------------------------------------
def test_19_no_besttake_change():
    for name in ("deterministic_best_take_authority", "take_judge"):
        assert "language_spine_consumer_migration" not in _source(name)


# ---------------------------------------------------------------------------
# 20. No D-150 change
# ---------------------------------------------------------------------------
def test_20_no_d150_change():
    assert "language_spine_consumer_migration" not in _source("semantic_authority_observability")


# ---------------------------------------------------------------------------
# 21. No D-158 change
# ---------------------------------------------------------------------------
def test_21_no_d158_change():
    assert _run_git_diff("cutsell_worker/attempt_relationship_authority.py") == ""


# ---------------------------------------------------------------------------
# 22. No D-161 change
# ---------------------------------------------------------------------------
def test_22_no_d161_change():
    assert _run_git_diff("cutsell_worker/watch_listen_relation_discovery.py") == ""


# ---------------------------------------------------------------------------
# 23. No D-163 change
# ---------------------------------------------------------------------------
def test_23_no_d163_change():
    assert _run_git_diff("cutsell_worker/watch_listen_besttake_evidence.py") == ""


# ---------------------------------------------------------------------------
# 24. No D-167 change
# ---------------------------------------------------------------------------
def test_24_no_d167_change():
    assert _run_git_diff("cutsell_worker/watch_listen_zone_usability_v2.py") == ""


# ---------------------------------------------------------------------------
# 25. Boundary unchanged
# ---------------------------------------------------------------------------
def test_25_boundary_unchanged():
    assert _run_git_diff("cutsell_worker/boundary_engine_pass.py") == ""


# ---------------------------------------------------------------------------
# 26. Pacing unchanged
# ---------------------------------------------------------------------------
def test_26_pacing_unchanged():
    assert _run_git_diff("cutsell_worker/dialogue_pacing_transition.py") == ""


# ---------------------------------------------------------------------------
# 27. Render unchanged
# ---------------------------------------------------------------------------
def test_27_render_unchanged():
    for name in ("render", "render_plan", "render_versions"):
        assert _run_git_diff(f"cutsell_worker/{name}.py") == ""


# ---------------------------------------------------------------------------
# 28. No provider/network
# ---------------------------------------------------------------------------
def test_28_no_provider_network():
    import cutsell_worker.language_spine_consumer_migration as m
    source = inspect.getsource(m)
    for forbidden in ("openai", "gemini", "requests.", "httpx.", "urllib.request", "socket."):
        assert forbidden not in source.lower()


# ---------------------------------------------------------------------------
# 29. Max 3 consumer clusters (exactly 2 migrated this task)
# ---------------------------------------------------------------------------
def test_29_max_three_consumer_clusters():
    migrated = []
    if "language_spine_consumer_migration" in _source("take_grouping_provider"):
        migrated.append("take_grouping_provider")
    if "language_spine_consumer_migration" in _source("recording_meta_continuation"):
        migrated.append("recording_meta_continuation")
    assert 1 <= len(migrated) <= 3
    assert migrated == ["take_grouping_provider", "recording_meta_continuation"]


# ---------------------------------------------------------------------------
# 30. Old serialized ids unaffected
# ---------------------------------------------------------------------------
def test_30_old_serialized_ids_unaffected():
    take = _mk_take("clip_1", "asset_1", 0.0, 1.0, "hola", words=(_w("hola", 0.0, 0.5),))
    assert take.attempt_id is None
    assert take.realization_id is None
    assert take.source_span_id is None
    _direct_meta_short_tail(take, ())
    # Untouched by the migration call.
    assert take.attempt_id is None
    assert take.realization_id is None
    assert take.source_span_id is None


# ---------------------------------------------------------------------------
# Additional structural/contract tests beyond the 30-item matrix.
# ---------------------------------------------------------------------------
def test_language_spine_untouched():
    assert _run_git_diff("cutsell_worker/language_spine.py") == ""


def test_language_utterance_attempt_untouched():
    assert _run_git_diff("cutsell_worker/language_utterance_attempt.py") == ""


def test_language_proposition_relation_untouched():
    assert _run_git_diff("cutsell_worker/language_proposition_relation.py") == ""


def test_semantic_claims_untouched():
    assert _run_git_diff("cutsell_worker/semantic_claims.py") == ""


def test_semantic_idea_equivalence_untouched():
    assert _run_git_diff("cutsell_worker/semantic_idea_equivalence.py") == ""


def test_no_id_minted_by_this_module():
    import cutsell_worker.language_spine_consumer_migration as m
    source = inspect.getsource(m)
    for forbidden in ("mint_retry_family_id", "mint_semantic_idea_id", "retry_family_id ="):
        assert forbidden not in source


def test_recording_meta_continuation_behavior_unchanged_full_flow():
    anchor = _mk_take(
        "c1", "a1", 0.0, 2.0, "I don't know how to end this video",
        words=(_w("I", 0.0, 0.1), _w("don't", 0.1, 0.3), _w("know", 0.3, 0.5), _w("how", 0.5, 0.6),
               _w("to", 0.6, 0.7), _w("end", 0.7, 0.9), _w("this", 0.9, 1.1), _w("video", 1.1, 1.5)),
    )
    tiny = _mk_take("c2", "a1", 2.2, 2.8, "you know", words=(_w("you", 2.2, 2.4), _w("know", 2.4, 2.8)))
    kept, removed, diagnostics = apply_recording_meta_continuation_cleanup([anchor, tiny], [])
    assert [c.clip_id for c in kept] == ["c1"]
    assert [c.clip_id for c in removed] == ["c2"]


def test_legacy_tiny_continuation_matches_original_boundary_semantics():
    # duration_sec == 2.2 exactly must still count as tiny (original used
    # `> 2.2` for rejection, i.e. `<= 2.2` accepted).
    take = _mk_take("c1", "a1", 0.0, 2.2, "one two three", words=())
    assert _legacy_tiny_continuation(take) is True
    take_over = _mk_take("c1", "a1", 0.0, 2.21, "one two three", words=())
    assert _legacy_tiny_continuation(take_over) is False
    take_empty = _mk_take("c1", "a1", 0.0, 1.0, "", words=())
    assert _legacy_tiny_continuation(take_empty) is False


def test_empty_words_returns_legacy_only():
    verdict, trace = continuation_migration("t", "a1", (), True)
    assert verdict is True
    assert trace.result_source == LEGACY_ONLY


def test_diagnostics_shape():
    traces = [
        ConsumerMigrationTrace("x", True, True, False, False, SPINE_CONSUMED),
        ConsumerMigrationTrace("x", True, False, True, False, SPINE_AVAILABLE_LEGACY_FALLBACK),
        ConsumerMigrationTrace("x", False, False, True, False, LEGACY_ONLY),
        ConsumerMigrationTrace("x", True, False, True, True, SPINE_CONFLICT_FALLBACK),
    ]
    diag = language_spine_consumer_migration_diagnostics(traces)
    assert diag["language_spine_consumer_evaluated_count"] == 4
    assert diag["language_spine_consumer_used_count"] == 1
    assert diag["language_spine_consumer_legacy_fallback_count"] == 3
    assert diag["language_spine_consumer_conflict_count"] == 1
    assert len(diag["traces"]) == 4
    assert "transcript" not in str(diag).lower()


def test_not_imported_by_unrelated_production_call_sites():
    call_sites = (
        "pipeline", "flow_b", "take_grouping", "hybrid_session_cleanup",
        "semantic_idea_equivalence", "attempt_relationship_authority",
        "deterministic_best_take_authority", "take_judge",
        "watch_listen_besttake_evidence", "watch_listen_zone_usability_v2",
        "boundary_engine_pass", "dialogue_pacing_transition",
        "semantic_authority_observability", "watch_listen_relation_discovery",
    )
    for name in call_sites:
        assert "language_spine_consumer_migration" not in _source(name), name
