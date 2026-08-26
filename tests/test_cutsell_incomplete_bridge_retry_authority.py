from cutsell_worker.contracts import CandidateTake
from cutsell_worker.hybrid_session_cleanup import HybridSessionCleanupResult
import cutsell_worker.hybrid_session_cleanup as cleanup
from cutsell_worker.incomplete_bridge_retry_authority import install_incomplete_bridge_retry_authority


def _take(cid, start, end, text, complete=True):
    return CandidateTake(
        clip_id=cid,
        source_asset_id="src",
        source_order=0,
        start=start,
        end=end,
        text=text,
        complete_idea=complete,
    )


def test_later_complete_retry_wins_when_incomplete_bridge_proves_restart(monkeypatch):
    early = _take("early", 10.0, 14.0, "They sent me to do a thyroid ultrasound and other scans")
    bridge = _take("bridge", 16.0, 18.0, "They sent me to do", complete=False)
    late = _take("late", 20.0, 24.5, "to do a thyroid ultrasound and other scans")

    def baseline(takes, context, editorial_judge, **kwargs):
        return HybridSessionCleanupResult(
            kept=(early,),
            deleted=(bridge, late),
            requested_chunk_count=1,
            available_chunk_count=1,
            diagnostics=(),
            semantic_decisions=((early.clip_id, "winner", 0.9), (bridge.clip_id, "failed", 0.9), (late.clip_id, "failed", 0.85)),
        )

    monkeypatch.setattr(cleanup, "apply_hybrid_session_cleanup", baseline)
    install_incomplete_bridge_retry_authority()
    result = cleanup.apply_hybrid_session_cleanup((early, bridge, late), None, None)
    assert [t.clip_id for t in result.kept] == ["late"]
    assert {t.clip_id for t in result.deleted} == {"early", "bridge"}
    decisions = {cid: (label, conf) for cid, label, conf in result.semantic_decisions}
    assert decisions["late"][0] == "winner"


def test_numeric_conflict_prevents_supersession(monkeypatch):
    early = _take("early", 10.0, 14.0, "The nodule measured 3 centimeters")
    bridge = _take("bridge", 16.0, 18.0, "The nodule measured", complete=False)
    late = _take("late", 20.0, 24.5, "The nodule measured 2 centimeters")

    def baseline(takes, context, editorial_judge, **kwargs):
        return HybridSessionCleanupResult(
            kept=(early,),
            deleted=(bridge, late),
            requested_chunk_count=1,
            available_chunk_count=1,
            diagnostics=(),
            semantic_decisions=(),
        )

    monkeypatch.setattr(cleanup, "apply_hybrid_session_cleanup", baseline)
    install_incomplete_bridge_retry_authority()
    result = cleanup.apply_hybrid_session_cleanup((early, bridge, late), None, None)
    assert [t.clip_id for t in result.kept] == ["early"]
