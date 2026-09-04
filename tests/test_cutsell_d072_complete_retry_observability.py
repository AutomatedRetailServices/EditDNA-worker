"""D-072: additive observability for complete_retry_identity_guard.py.

Proves, all at once:
  - the guard's DECISION (the (candidate, overlap) tuple it returns) is
    byte-for-byte unchanged from before D-072 for every founding safety
    case (see test_cutsell_complete_retry_identity_guard.py, still green,
    untouched by this directive);
  - the new diagnostic side channel correctly explains every distinct
    rejection/acceptance shape with a bounded reason;
  - the ContextVar side channel is read-once, cleared-on-read, and does
    not leak between clips, decisions, or concurrent contexts.

No semantic PATH B. No threshold change. No orphan/Freeze policy change.
"""
import threading

from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_session_cleanup import _later_semantic_retry_replacement
from cutsell_worker.complete_retry_identity_guard import (
    INCOMPLETE_RETRY_LOOSER_MATCH,
    LEXICAL_REPLACEMENT_VERIFIED,
    NO_CANDIDATE,
    NUMBER_PRESERVATION_FAILED,
    SEQUENCE_IDENTITY_BELOW_THRESHOLD,
    SEMANTIC_OVERLAP_BELOW_THRESHOLD,
    _consume_replacement_guard_diagnostic,
)


def _take(clip_id, start, end, text, *, source="src", complete_idea=True):
    return CandidateTake(
        clip_id=clip_id, source_asset_id=source, source_order=0,
        start=start, end=end, text=text, complete_idea=complete_idea,
    )


# --- candidate accepted: LEXICAL_REPLACEMENT_VERIFIED -----------------------

def test_diagnostic_explains_lexical_replacement_verified():
    failed = _take("failed", 10.0, 14.0, "ahi fue cuando me mandaron a hacer sonografias de tiroides y otros")
    retake = _take("retake", 18.0, 23.0, "a hacer sonografia de tiroides y otras sonografias")
    decisions = {"failed": ("failed", 0.85), "retake": ("winner", 0.95)}

    replacement, overlap = _later_semantic_retry_replacement(failed, (failed, retake), decisions)
    diag = _consume_replacement_guard_diagnostic()

    assert replacement is retake
    assert overlap >= 0.64
    assert diag is not None
    assert diag.replacement_candidate_clip_id_before_guard == "retake"
    assert diag.semantic_overlap == overlap
    assert diag.lexical_identity_passed is True
    assert diag.sequence_identity is not None and diag.sequence_identity >= diag.sequence_identity_threshold
    assert diag.replacement_rejection_reason == LEXICAL_REPLACEMENT_VERIFIED


# --- candidate rejected: sequence identity (the exact D-070/D-071 shape) ---

def test_diagnostic_explains_sequence_identity_rejection_d070_regression():
    """D-070's own proven anomaly: overlap=1.0, replacement=None. D-072
    must explain it instead of leaving it looking contradictory."""
    failed = _take(
        "clip_6a0741140525fc819886", 91.22, 93.52,
        "Al terminar mi contrato, le pedí a mi ginecóloga.",
        source="src_f4f7bd1056de0a371195",
    )
    candidate = _take(
        "clip_6c2c1403d9b83045bf08", 95.58, 104.02,
        "Al terminar mi contrato, cambié de ginecóloga y le pedí que me hiciera "
        "un test de todo lo que ella se pudiera imaginar y me pudiese indicar.",
        source="src_f4f7bd1056de0a371195",
    )
    decisions = {"clip_6c2c1403d9b83045bf08": ("winner", 0.95)}

    replacement, overlap = _later_semantic_retry_replacement(failed, (failed, candidate), decisions)
    diag = _consume_replacement_guard_diagnostic()

    # The decision itself: unchanged from D-070's own live-reproduced result.
    assert replacement is None
    assert overlap == 1.0

    # The diagnostic now explains it fully.
    assert diag is not None
    assert diag.replacement_candidate_clip_id_before_guard == "clip_6c2c1403d9b83045bf08"
    assert diag.semantic_overlap == 1.0
    assert round(diag.sequence_identity, 4) == 0.4108
    assert diag.sequence_identity_threshold == 0.52
    assert diag.lexical_identity_passed is False
    assert diag.replacement_rejection_reason == SEQUENCE_IDENTITY_BELOW_THRESHOLD


# --- candidate rejected: number preservation --------------------------------

def test_diagnostic_explains_number_preservation_failure():
    failed = _take("failed", 10.0, 14.0, "el nodulo media 3 centimetros y se mando a biopsia")
    retake_without_number = _take("retake", 18.0, 23.0, "el nodulo se mando a biopsia porque era sospechoso")
    decisions = {"failed": ("failed", 0.90), "retake": ("winner", 0.95)}

    replacement, overlap = _later_semantic_retry_replacement(
        failed, (failed, retake_without_number), decisions,
    )
    diag = _consume_replacement_guard_diagnostic()

    assert replacement is None
    assert overlap == 0.0
    assert diag is not None
    assert diag.replacement_candidate_clip_id_before_guard is None
    assert diag.replacement_rejection_reason == NUMBER_PRESERVATION_FAILED


# --- no candidate at all -----------------------------------------------------

def test_diagnostic_explains_no_candidate_wrong_source_asset():
    failed = _take(
        "failed", 10.0, 14.0,
        "el paciente presento sintomas leves durante la consulta", source="srcA",
    )
    other_source = _take(
        "other", 18.0, 23.0,
        "el paciente presento sintomas leves durante la consulta", source="srcB",
    )
    decisions = {"failed": ("failed", 0.85), "other": ("winner", 0.95)}

    replacement, overlap = _later_semantic_retry_replacement(failed, (failed, other_source), decisions)
    diag = _consume_replacement_guard_diagnostic()

    assert replacement is None
    assert overlap == 0.0
    assert diag is not None
    assert diag.replacement_candidate_clip_id_before_guard is None
    assert diag.replacement_rejection_reason == NO_CANDIDATE


def test_diagnostic_explains_semantic_overlap_below_threshold():
    failed = _take("failed", 10.0, 14.0, "el paciente presento sintomas leves durante la consulta")
    low_overlap = _take("lowover", 18.0, 23.0, "compre pan y leche en la tienda de la esquina")
    decisions = {"failed": ("failed", 0.85), "lowover": ("winner", 0.95)}

    replacement, overlap = _later_semantic_retry_replacement(failed, (failed, low_overlap), decisions)
    diag = _consume_replacement_guard_diagnostic()

    assert replacement is None
    assert overlap == 0.0
    assert diag is not None
    assert diag.replacement_rejection_reason == SEMANTIC_OVERLAP_BELOW_THRESHOLD


# --- incomplete retry: existing looser behavior unchanged, now explained ---

def test_diagnostic_explains_incomplete_retry_looser_match_found():
    failed = _take("failed", 10.0, 12.0, "me mandaron hacer sonografia tiroides", complete_idea=False)
    retake = _take("retake", 14.0, 19.0, "ahi fue cuando me mandaron a hacer sonografia de tiroides completa")
    decisions = {"failed": ("failed", 0.85), "retake": ("winner", 0.95)}

    replacement, overlap = _later_semantic_retry_replacement(failed, (failed, retake), decisions)
    diag = _consume_replacement_guard_diagnostic()

    assert replacement is retake
    assert overlap >= 0.50
    assert diag is not None
    assert diag.replacement_candidate_clip_id_before_guard == "retake"
    assert diag.sequence_identity is None  # guard's sequence check never applies here
    assert diag.lexical_identity_passed is None
    assert diag.replacement_rejection_reason == INCOMPLETE_RETRY_LOOSER_MATCH


def test_diagnostic_explains_incomplete_retry_no_candidate():
    failed = _take("failed", 10.0, 12.0, "me mandaron hacer sonografia tiroides", complete_idea=False)
    unrelated = _take("unrelated", 14.0, 19.0, "compre pan y leche en la tienda")
    decisions = {"failed": ("failed", 0.85), "unrelated": ("winner", 0.95)}

    replacement, overlap = _later_semantic_retry_replacement(failed, (failed, unrelated), decisions)
    diag = _consume_replacement_guard_diagnostic()

    assert replacement is None
    assert diag is not None
    assert diag.replacement_rejection_reason == NO_CANDIDATE


# --- ContextVar isolation / reset -------------------------------------------

def test_diagnostic_is_cleared_after_single_consumption():
    failed = _take("failed", 10.0, 14.0, "ahi fue cuando me mandaron a hacer sonografias de tiroides y otros")
    retake = _take("retake", 18.0, 23.0, "a hacer sonografia de tiroides y otras sonografias")
    decisions = {"failed": ("failed", 0.85), "retake": ("winner", 0.95)}

    _later_semantic_retry_replacement(failed, (failed, retake), decisions)
    first = _consume_replacement_guard_diagnostic()
    second = _consume_replacement_guard_diagnostic()

    assert first is not None
    assert second is None


def test_diagnostic_returns_none_when_guard_never_invoked():
    # Drain any diagnostic left behind by an earlier test in this same
    # process/context (ContextVars persist across sequential calls in one
    # context, same as production's own per-request context) so this
    # assertion is about a genuinely empty channel, not test ordering.
    _consume_replacement_guard_diagnostic()
    assert _consume_replacement_guard_diagnostic() is None


def test_diagnostic_does_not_leak_between_successive_clips():
    """Simulates hybrid_session_cleanup.py's own per-decision loop: one
    'failed' clip whose guard search is rejected, immediately followed by
    another whose search succeeds -- the second consumption must reflect
    ONLY the second call, never a stale carryover from the first."""
    failed_1 = _take("failed1", 10.0, 14.0, "el paciente presento sintomas leves durante la consulta")
    low_overlap = _take("lowover", 18.0, 23.0, "compre pan y leche en la tienda de la esquina")
    decisions_1 = {"failed1": ("failed", 0.85), "lowover": ("winner", 0.95)}
    _later_semantic_retry_replacement(failed_1, (failed_1, low_overlap), decisions_1)
    diag_1 = _consume_replacement_guard_diagnostic()

    failed_2 = _take("failed2", 10.0, 14.0, "ahi fue cuando me mandaron a hacer sonografias de tiroides y otros")
    retake_2 = _take("retake2", 18.0, 23.0, "a hacer sonografia de tiroides y otras sonografias")
    decisions_2 = {"failed2": ("failed", 0.85), "retake2": ("winner", 0.95)}
    _later_semantic_retry_replacement(failed_2, (failed_2, retake_2), decisions_2)
    diag_2 = _consume_replacement_guard_diagnostic()

    assert diag_1.replacement_rejection_reason == SEMANTIC_OVERLAP_BELOW_THRESHOLD
    assert diag_2.replacement_rejection_reason == LEXICAL_REPLACEMENT_VERIFIED
    assert diag_2.replacement_candidate_clip_id_before_guard == "retake2"


def test_diagnostic_request_isolation_across_threads():
    """ContextVars are per-context (per-thread here, per-async-task in
    production) -- two threads each running their OWN guard call must
    each see only their own diagnostic, never the other's, even without
    any lock or explicit synchronization beyond a barrier for timing."""
    results = {}
    barrier = threading.Barrier(2)

    def worker(name, failed_text, candidate_text, expect_reason):
        failed = _take(f"failed_{name}", 10.0, 14.0, failed_text)
        candidate = _take(f"cand_{name}", 18.0, 23.0, candidate_text)
        decisions = {f"failed_{name}": ("failed", 0.85), f"cand_{name}": ("winner", 0.95)}
        barrier.wait()  # maximize the chance of true overlap between threads
        _later_semantic_retry_replacement(failed, (failed, candidate), decisions)
        diag = _consume_replacement_guard_diagnostic()
        results[name] = diag

    t1 = threading.Thread(
        target=worker,
        args=(
            "a",
            "ahi fue cuando me mandaron a hacer sonografias de tiroides y otros",
            "a hacer sonografia de tiroides y otras sonografias",
            LEXICAL_REPLACEMENT_VERIFIED,
        ),
    )
    t2 = threading.Thread(
        target=worker,
        args=(
            "b",
            "el paciente presento sintomas leves durante la consulta",
            "compre pan y leche en la tienda de la esquina",
            SEMANTIC_OVERLAP_BELOW_THRESHOLD,
        ),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["a"].replacement_rejection_reason == LEXICAL_REPLACEMENT_VERIFIED
    assert results["a"].replacement_candidate_clip_id_before_guard == "cand_a"
    assert results["b"].replacement_rejection_reason == SEMANTIC_OVERLAP_BELOW_THRESHOLD
    assert results["b"].replacement_candidate_clip_id_before_guard is None
