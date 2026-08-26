from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
from cutsell_worker import hybrid_session_cleanup
from cutsell_worker.hybrid_unavailable_retry_fallback import install_hybrid_unavailable_retry_fallback


def _take(clip_id, start, end, text, *, complete=True):
    return CandidateTake(
        clip_id=clip_id,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def _install_over_fake(monkeypatch, fake_result):
    def fake_apply(takes, *args, **kwargs):
        return fake_result

    monkeypatch.setattr(hybrid_session_cleanup, "apply_hybrid_session_cleanup", fake_apply)
    install_hybrid_unavailable_retry_fallback()
    return hybrid_session_cleanup.apply_hybrid_session_cleanup


def test_unavailable_window_removes_only_undecided_incomplete_retry_covered_later(monkeypatch):
    failed = _take(
        "failed",
        10.0,
        14.0,
        "me mandaron a hacer sonografia de tiroides",
        complete=False,
    )
    winner = _take(
        "winner",
        18.0,
        23.0,
        "me mandaron a hacer sonografia de tiroides y otras sonografias",
        complete=True,
    )
    result = HybridSessionCleanupResult(
        kept=(failed, winner),
        deleted=(),
        requested_chunk_count=5,
        available_chunk_count=2,
        diagnostics=(),
        semantic_decisions=(("winner", "winner", 0.96),),
    )
    wrapped = _install_over_fake(monkeypatch, result)
    out = wrapped((failed, winner), None, object())
    assert [take.clip_id for take in out.kept] == ["winner"]
    assert [take.clip_id for take in out.deleted] == ["failed"]
    assert any("hybrid_unavailable_retry_fallback" in row for row in out.diagnostics)


def test_unavailable_window_keeps_unique_or_complete_undecided_speech(monkeypatch):
    unique_incomplete = _take(
        "unique",
        10.0,
        14.0,
        "mi hermana vivia conmigo durante ese periodo",
        complete=False,
    )
    complete_repeat = _take(
        "complete",
        16.0,
        20.0,
        "me mandaron a hacer sonografia de tiroides",
        complete=True,
    )
    later = _take(
        "later",
        22.0,
        27.0,
        "me mandaron a hacer sonografia de tiroides y otras sonografias",
        complete=True,
    )
    result = HybridSessionCleanupResult(
        kept=(unique_incomplete, complete_repeat, later),
        deleted=(),
        requested_chunk_count=4,
        available_chunk_count=1,
        diagnostics=(),
        semantic_decisions=(("later", "winner", 0.95),),
    )
    wrapped = _install_over_fake(monkeypatch, result)
    out = wrapped((unique_incomplete, complete_repeat, later), None, object())
    assert [take.clip_id for take in out.kept] == ["unique", "complete", "later"]
    assert not out.deleted


def test_fallback_is_inactive_when_all_hybrid_windows_are_available(monkeypatch):
    incomplete = _take("incomplete", 1.0, 3.0, "hacer sonografia tiroides", complete=False)
    later = _take("later", 5.0, 8.0, "hacer sonografia de tiroides completa", complete=True)
    result = HybridSessionCleanupResult(
        kept=(incomplete, later),
        deleted=(),
        requested_chunk_count=2,
        available_chunk_count=2,
        diagnostics=(),
        semantic_decisions=(("later", "winner", 0.95),),
    )
    wrapped = _install_over_fake(monkeypatch, result)
    out = wrapped((incomplete, later), None, object())
    assert [take.clip_id for take in out.kept] == ["incomplete", "later"]
