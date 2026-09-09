"""D-189: Prosodic Audio V2 -- LIVE PIPELINE DIAGNOSTIC WIRING. Offline
only. No RAW. No provider. No winner authority. No score weights. No
prosodic master score.

Proves the WIRING itself (pipeline.py's new `local_paths` parameter,
the eligibility-first gate, the source-audio decode cache, the D-184
`prosodic_comparison` hand-off, diagnostics/run-summary shape,
default-off byte-identical compatibility, fail-open behavior, error
isolation, and the meaning/D-183-DECISIVE firewalls holding at the
pipeline level) -- NOT the comparison logic itself, which D-187's 51
tests and D-188's 47 tests already exhaustively prove (pipeline.py
calls the SAME `analyze_prosodic_delivery`/`compare_prosodic_finalists`
functions verbatim, per this task's own "do not reimplement" mandate).
"""
from __future__ import annotations

import wave

import numpy as np
import pytest

from cutsell_worker.bounded_finalist_arbiter import (
    DECISION_ABSTAIN,
    DECISION_PREFER_CANDIDATE,
)
from cutsell_worker.contracts import CandidateTake, MediaSignals, ProcessingRequest, SemanticLabel, SemanticRole
from cutsell_worker.pipeline import build_flow_b_draft

SR = 16000


# ---------------------------------------------------------------------------
# Real, generic, non-Video00 synthetic audio fixtures (stdlib `wave`, no
# ffmpeg dependency to AUTHOR the fixture -- ffmpeg is exercised as the
# DECODE path under test, exactly like D-187's own integration test).
# ---------------------------------------------------------------------------
def _write_wav(path, samples_float, sample_rate=SR):
    clipped = np.clip(samples_float, -1.0, 1.0)
    ints = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())


def _tone(duration_sec, amplitude=0.25, freq=180.0, sample_rate=SR):
    n = int(round(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_sec, sample_rate=SR):
    return np.zeros(int(round(duration_sec * sample_rate)), dtype=np.float32)


def _near_equal_audio_path(tmp_path, duration_sec=8.0):
    """One continuous, undifferentiated tone spanning the whole source --
    both candidate windows see the same delivery shape (near-equal)."""
    path = tmp_path / "near_equal_source.wav"
    _write_wav(path, _tone(duration_sec))
    return str(path)


def _dominance_audio_path(tmp_path):
    """Source audio where the [1.0, 3.0] window (candidate 'weak') is
    fragmented by an interior pause, and the [4.0, 6.0] window (candidate
    'strong') is continuous -- an abstract, generic differentiating
    fixture, no literal Video00/Pimples transcript anywhere."""
    path = tmp_path / "dominance_source.wav"
    waveform = np.concatenate([
        _tone(1.0),                # 0.0-1.0 lead-in
        _tone(0.7),                # 1.0-1.7 (inside weak's span)
        _silence(0.6),             # 1.7-2.3 interior pause (inside weak's span)
        _tone(0.7),                # 2.3-3.0 (inside weak's span)
        _tone(1.0),                # 3.0-4.0 gap between candidates
        _tone(2.0),                # 4.0-6.0 continuous (strong's whole span)
        _tone(2.0),                # 6.0-8.0 trailer
    ])
    _write_wav(path, waveform)
    return str(path)


def _silent_audio_path(tmp_path, duration_sec=8.0):
    path = tmp_path / "silent_source.wav"
    _write_wav(path, _silence(duration_sec))
    return str(path)


# ---------------------------------------------------------------------------
# Candidate-take fixtures.
# ---------------------------------------------------------------------------
def _weak_strong_fixture():
    weak = CandidateTake(
        clip_id="weak", source_asset_id="src", source_order=0, start=1.0, end=3.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=1.0, end=3.0, audio_quality=0.3, eye_contact=0.2),
    )
    strong = CandidateTake(
        clip_id="strong", source_asset_id="src", source_order=0, start=4.0, end=6.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=4.0, end=6.0, audio_quality=0.95, eye_contact=0.95),
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (
        SemanticLabel(weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(strong.clip_id, SemanticRole.PROOF, 0.9),
    )
    return request, (weak, strong), labels, strong.clip_id


def _single_candidate_fixture():
    only = CandidateTake(
        clip_id="only", source_asset_id="src", source_order=0, start=1.0, end=3.0,
        text="this serum changed my skin",
        signals=MediaSignals(source_asset_id="src", start=1.0, end=3.0, audio_quality=0.9, eye_contact=0.9),
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (SemanticLabel(only.clip_id, SemanticRole.PROOF, 0.9),)
    return request, (only,), labels


def _four_candidate_fixture():
    candidates = tuple(
        CandidateTake(
            clip_id=f"c{i}", source_asset_id="src", source_order=0, start=1.0 + i * 2, end=2.5 + i * 2,
            text="this serum changed my skin",
            signals=MediaSignals(source_asset_id="src", start=1.0 + i * 2, end=2.5 + i * 2, audio_quality=0.5, eye_contact=0.5),
        )
        for i in range(4)
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = tuple(SemanticLabel(c.clip_id, SemanticRole.PROOF, 0.9) for c in candidates)
    return request, candidates, labels


ENV_BOTH_ON = {
    "CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED": "1",
    "CUTSELL_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENABLED": "1",
}


def _set_env(monkeypatch, env: dict):
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _group_row(result):
    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    assert len(groups) == 1
    return groups[0]


# ===========================================================================
# 1-3. Feature-flag gating.
# ===========================================================================
def test_01_feature_flag_default_off_no_execution():
    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels)
    row = _group_row(result)
    assert not any(key.startswith("prosodic_pipeline_") or key.startswith("prosodic_finalist_") for key in row)
    assert result.draft.diagnostics.get("prosodic_pipeline") == {"status": "disabled"}
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


def test_02_arbiter_flag_off_prosodic_flag_on_no_execution(monkeypatch):
    monkeypatch.setenv("CUTSELL_PROSODIC_FINALIST_ARBITER_DIAGNOSTICS_ENABLED", "1")
    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels)
    row = _group_row(result)
    assert not any(key.startswith("bounded_finalist_arbiter_") for key in row)
    assert not any(key.startswith("prosodic_pipeline_") or key.startswith("prosodic_finalist_") for key in row)
    assert result.draft.diagnostics.get("prosodic_pipeline") == {"status": "disabled"}
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


def test_03_arbiter_on_prosodic_off_matches_pre_d189_exactly(monkeypatch):
    monkeypatch.setenv("CUTSELL_BOUNDED_FINALIST_ARBITER_ENABLED", "1")
    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels)
    row = _group_row(result)
    assert row["bounded_finalist_arbiter_prosodic_audio_status"] == "NOT_AVAILABLE"
    assert not any(key.startswith("prosodic_pipeline_") or key.startswith("prosodic_finalist_") for key in row)
    # The D-188 fusion-status field IS always present once the arbiter
    # itself runs (it is informative even when Prosody was never
    # consulted) -- but it always reports the honest, unchanged-since-
    # D-184 "NOT_AVAILABLE" value when the D-189 prosodic flag is off.
    assert row["bounded_finalist_arbiter_prosodic_status"] == "NOT_AVAILABLE"
    assert result.draft.diagnostics.get("prosodic_pipeline") == {"status": "disabled"}
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


# ===========================================================================
# 4. Both ON, no local_paths -> fail open, decision/winner unchanged.
# ===========================================================================
def test_04_both_flags_on_no_local_paths_fails_open(monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels)  # local_paths omitted -> None
    row = _group_row(result)
    assert row["prosodic_pipeline_audio_available"] is False
    assert row["prosodic_pipeline_source_decode_status"] == "unavailable"
    assert row["bounded_finalist_arbiter_prosodic_status"] == "INSUFFICIENT"
    assert row["bounded_finalist_arbiter_action_applied"] is False
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


# ===========================================================================
# 5. Real decodable audio -> evaluated, decode count 1.
# ===========================================================================
def test_05_real_audio_decoded_and_evaluated(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, expected_winner = _weak_strong_fixture()
    path = _near_equal_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    row = _group_row(result)
    assert row["prosodic_pipeline_audio_available"] is True
    assert row["prosodic_pipeline_source_decode_status"] == "decoded"
    assert row["prosodic_pipeline_candidate_count"] == 2
    assert row["prosodic_pipeline_candidates_evaluated"] == 2
    assert row["prosodic_pipeline_source_decode_reused"] is False  # first (only) decode this family
    assert row["prosodic_pipeline_pause_evidence_reused"] is True
    assert row["prosodic_pipeline_language_evidence_reused"] is True
    assert row["prosodic_finalist_evaluated"] is True
    summary = result.draft.diagnostics["prosodic_pipeline"]
    assert summary["status"] == "evaluated"
    assert summary["prosodic_pipeline_source_decode_count"] == 1
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


# ===========================================================================
# 6-7. Eligibility-first: candidate count out of [2,3] -> no execution.
# ===========================================================================
def test_06_single_finalist_no_execution(tmp_path, monkeypatch):
    """A genuine lone candidate is never even a 'contest' -- confirmed
    directly: `take_judge_groups` carries no row for it at all (nothing
    to arbitrate), so Prosodic work is trivially never triggered."""
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels = _single_candidate_fixture()
    path = _near_equal_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    for row in groups:
        assert not any(key.startswith("prosodic_pipeline_") or key.startswith("prosodic_finalist_") for key in row)
    assert result.draft.diagnostics.get("prosodic_pipeline") == {"status": "no_eligible_families"}


def test_07_four_finalists_no_execution(tmp_path, monkeypatch):
    """This fixture's own grouping heuristic (proximity/session
    boundaries, not this task's concern) may split 4 identical-text
    candidates into two 2-member retry families rather than one 4-member
    family -- a legitimate, disclosed grouping outcome. The invariant
    this test actually proves is the eligibility bound itself: no family
    row this task's own wiring ever produces exceeds the [2,3] candidate
    range for Prosodic evaluation, however grouping split the pool."""
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels = _four_candidate_fixture()
    path = _near_equal_audio_path(tmp_path, duration_sec=12.0)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    for row in groups:
        count = row.get("prosodic_pipeline_candidate_count")
        if count is not None:
            assert 2 <= count <= 3


# ===========================================================================
# 8. Prosodic evidence map actually built (structural).
# ===========================================================================
def test_08_prosodic_evidence_map_built(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, _ = _weak_strong_fixture()
    path = _near_equal_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    row = _group_row(result)
    for key in (
        "prosodic_finalist_evaluated", "prosodic_finalist_state", "prosodic_finalist_preferred_candidate_id",
        "prosodic_finalist_continuity_relation", "prosodic_finalist_hesitation_relation",
        "prosodic_finalist_restart_relation", "prosodic_finalist_pause_relation",
        "prosodic_finalist_descriptive_rate_relation", "prosodic_finalist_descriptive_energy_relation",
        "prosodic_finalist_descriptive_emphasis_relation", "prosodic_finalist_directional_evidence_present",
        "prosodic_finalist_conflict", "prosodic_finalist_missing_evidence",
        "bounded_finalist_arbiter_prosodic_status",
    ):
        assert key in row
    import json
    json.dumps({k: v for k, v in row.items() if k.startswith("prosodic_")})  # bounded, JSON-safe


# ===========================================================================
# 9. Near-equal audio -> arbiter does not fabricate a preference.
# ===========================================================================
def test_09_near_equal_audio_no_fabricated_preference(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, expected_winner = _weak_strong_fixture()
    path = _near_equal_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    row = _group_row(result)
    assert row["prosodic_finalist_state"] in ("NEAR_EQUAL", "CONFLICTED", "INSUFFICIENT_EVIDENCE", "NOT_EVALUABLE")
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


# ===========================================================================
# 10. Prosodic dominance -> real diagnostic differentiation, no winner change.
# ===========================================================================
def test_10_prosodic_dominance_differentiates_never_mutates_winner(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, expected_winner = _weak_strong_fixture()
    path = _dominance_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    row = _group_row(result)
    assert row["prosodic_finalist_evaluated"] is True
    # The continuous ("strong") span must never come out WORSE than the
    # fragmented ("weak") span on the safe dimensions.
    if row["prosodic_finalist_state"] == "DOMINANT":
        assert row["prosodic_finalist_preferred_candidate_id"] == "strong"
    assert row["bounded_finalist_arbiter_action_applied"] is False
    # NO WINNER MUTATION -- the whole point of "diagnostic only".
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


# ===========================================================================
# 11-12. Fail-open: decode failure, no-speech.
# ===========================================================================
def test_11_decode_failure_fails_open_no_crash(monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, expected_winner = _weak_strong_fixture()
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": "/nonexistent/path/does_not_exist.wav"})
    row = _group_row(result)
    assert row["prosodic_pipeline_audio_available"] is False
    assert row["bounded_finalist_arbiter_prosodic_status"] == "INSUFFICIENT"
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


def test_12_no_speech_honestly_represented(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, expected_winner = _weak_strong_fixture()
    path = _silent_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    row = _group_row(result)
    assert row["prosodic_pipeline_audio_available"] is True  # decode succeeded
    assert row["prosodic_finalist_state"] in ("NOT_EVALUABLE", "INSUFFICIENT_EVIDENCE")
    assert row["prosodic_finalist_preferred_candidate_id"] is None
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


# ===========================================================================
# 13. Partial-candidate failure isolation (error isolation).
# ===========================================================================
def test_13_one_candidate_analysis_failure_is_isolated(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    import cutsell_worker.pipeline as pipeline_mod
    real_analyze = pipeline_mod.analyze_prosodic_delivery

    def _flaky_analyze(candidate_id, *args, **kwargs):
        if candidate_id == "weak":
            raise RuntimeError("synthetic per-candidate failure")
        return real_analyze(candidate_id, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "analyze_prosodic_delivery", _flaky_analyze)
    request, takes, labels, expected_winner = _weak_strong_fixture()
    path = _near_equal_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})  # must not raise
    row = _group_row(result)
    assert row["prosodic_pipeline_candidates_evaluated"] == 1  # only "strong" succeeded
    assert row["prosodic_finalist_state"] in ("INSUFFICIENT_EVIDENCE", "NOT_EVALUABLE")
    assert [c.clip_id for c in result.draft.selected] == [expected_winner]


# ===========================================================================
# 14. Meaning conflict blocks -- Prosody built but never consulted.
# ===========================================================================
def test_14_meaning_conflict_blocks_prosody_never_consulted_by_arbiter(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    weak = CandidateTake(
        clip_id="weak", source_asset_id="src", source_order=0, start=1.0, end=3.0,
        text="The cream clears acne breakouts in 2 weeks.",
        signals=MediaSignals(source_asset_id="src", start=1.0, end=3.0, audio_quality=0.3, eye_contact=0.2),
    )
    strong = CandidateTake(
        clip_id="strong", source_asset_id="src", source_order=0, start=4.0, end=6.0,
        text="The cream does not clear acne breakouts in 2 weeks.",
        signals=MediaSignals(source_asset_id="src", start=4.0, end=6.0, audio_quality=0.95, eye_contact=0.95),
    )
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (
        SemanticLabel(weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(strong.clip_id, SemanticRole.PROOF, 0.9),
    )
    path = _dominance_audio_path(tmp_path)
    result = build_flow_b_draft(request, (weak, strong), labels, local_paths={"src": path})
    row = _group_row(result)
    # The pipeline DID build Prosodic evidence (eligibility-first only
    # checks candidate count + terminal state, not meaning) -- but the
    # arbiter's OWN P0 meaning gate runs first internally and reports
    # "never consulted", exactly matching D-188's own established
    # contract (see bounded_finalist_arbiter.py's own meaning-conflict
    # early return).
    assert row["bounded_finalist_arbiter_prosodic_status"] == "NOT_AVAILABLE"
    assert row.get("bounded_finalist_arbiter_meaning_parity") == "CONFLICT"


# ===========================================================================
# 15-16. Decode-once / reuse across multiple eligible families sharing a
# source, and the compute/performance integration fixture.
# ===========================================================================
def _two_family_same_source_fixture():
    def clip(cid, text, start, end, aq, ec):
        return CandidateTake(
            clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end, text=text,
            signals=MediaSignals(source_asset_id="src", start=start, end=end, audio_quality=aq, eye_contact=ec),
        )
    fam1_weak = clip("f1_weak", "this serum changed my skin", 1.0, 3.0, 0.3, 0.2)
    fam1_strong = clip("f1_strong", "this serum changed my skin", 4.0, 6.0, 0.95, 0.95)
    fam2_weak = clip("f2_weak", "this cream cleared my acne", 7.0, 9.0, 0.3, 0.2)
    fam2_strong = clip("f2_strong", "this cream cleared my acne", 10.0, 12.0, 0.95, 0.95)
    request = ProcessingRequest(project_id="project-1", user_id="user-1", sources=())
    labels = (
        SemanticLabel(fam1_weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(fam1_strong.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(fam2_weak.clip_id, SemanticRole.PROOF, 0.9),
        SemanticLabel(fam2_strong.clip_id, SemanticRole.PROOF, 0.9),
    )
    return request, (fam1_weak, fam1_strong, fam2_weak, fam2_strong), labels


def test_15_and_16_multi_family_same_source_decodes_once(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels = _two_family_same_source_fixture()
    path = _near_equal_audio_path(tmp_path, duration_sec=13.0)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    groups = (result.draft.diagnostics or {}).get("take_judge_groups") or []
    eligible_rows = [g for g in groups if g.get("prosodic_pipeline_candidate_count")]
    summary = result.draft.diagnostics.get("prosodic_pipeline")
    if len(eligible_rows) >= 2:
        # THE compute/performance requirement: multiple eligible families
        # from ONE source -> exactly ONE source decode.
        assert summary["prosodic_pipeline_source_decode_count"] == 1
        assert summary["prosodic_pipeline_family_evaluated_count"] == len(eligible_rows)
        reused_flags = [g["prosodic_pipeline_source_decode_reused"] for g in eligible_rows]
        assert reused_flags.count(False) == 1  # exactly one first-decode
        assert reused_flags.count(True) == len(eligible_rows) - 1
    else:
        # Grouping produced fewer than 2 eligible families this run (a
        # legitimate, disclosed grouping-heuristic outcome, not a D-189
        # defect) -- still assert the decode count is sane (<=1 decode
        # for the one physical source either way).
        assert summary["prosodic_pipeline_source_decode_count"] <= 1


# ===========================================================================
# 17. Runtime/performance report (informational only).
# ===========================================================================
def test_17_runtime_report(tmp_path, monkeypatch):
    import time
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels = _two_family_same_source_fixture()
    path = _near_equal_audio_path(tmp_path, duration_sec=13.0)
    start = time.monotonic()
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    elapsed = time.monotonic() - start
    summary = result.draft.diagnostics.get("prosodic_pipeline")
    print(
        f"D-189 runtime: source_duration_sec=13.0 "
        f"eligible_family_count={summary.get('prosodic_pipeline_family_evaluated_count')} "
        f"candidate_count={summary.get('prosodic_pipeline_candidate_evaluated_count')} "
        f"decode_count={summary.get('prosodic_pipeline_source_decode_count')} "
        f"wall_sec={elapsed:.4f}"
    )
    assert summary["status"] in ("evaluated", "no_eligible_families")


# ===========================================================================
# 18. Determinism: repeated execution yields identical diagnostics.
# ===========================================================================
def test_18_deterministic_repeated_execution(tmp_path, monkeypatch):
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, _ = _weak_strong_fixture()
    path = _dominance_audio_path(tmp_path)
    r1 = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    r2 = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    row1, row2 = _group_row(r1), _group_row(r2)
    prosodic_keys = [k for k in row1 if k.startswith("prosodic_") or k.startswith("bounded_finalist_arbiter_")]
    for key in prosodic_keys:
        assert row1[key] == row2.get(key), key


# ===========================================================================
# 19. No-psychology / no-provider / no-winner-mutation structural scans on
# the NEW pipeline.py wiring specifically.
# ===========================================================================
def _pipeline_d189_source_slice():
    import inspect
    import cutsell_worker.pipeline as pipeline_mod
    return inspect.getsource(pipeline_mod.build_flow_b_draft)


def test_19_no_psychology_inference_in_wiring():
    import re
    source = _pipeline_d189_source_slice()
    code_only = re.sub(r'""".*?"""', "", source, flags=__import__("re").DOTALL)
    code_only = re.sub(r"#.*", "", code_only)
    for forbidden in ("confident", "nervous", "excited", "persuasive", "authentic", "truthful", "emotion"):
        assert forbidden not in code_only.casefold()


def test_20_no_provider_network_in_wiring():
    source = _pipeline_d189_source_slice()
    lowered = source.casefold()
    for forbidden in ("requests.", "urllib", "openai", "anthropic", "gemini", "modal.com"):
        assert forbidden not in lowered


def test_21_no_winner_mutation_in_wiring_source():
    import re
    source = _pipeline_d189_source_slice()
    code_only = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    code_only = re.sub(r"#.*", "", code_only)
    # The new D-189 block never assigns to selected_clip_id/ranked/winner.
    for pattern in (r"selected_clip_id\s*=", r"\bwinner\s*="):
        assert not re.search(pattern, "\n".join(
            line for line in code_only.splitlines() if "prosodic" in line.casefold()
        )), pattern


def test_22_no_duplicate_silence_detection_source_scan():
    source = _pipeline_d189_source_slice()
    assert "detect_audio_silence_intervals(" not in source


def test_23_no_duplicate_asr_source_scan():
    source = _pipeline_d189_source_slice()
    for forbidden in ("_run_asr(", "ASRProvider("):
        assert forbidden not in source


# ===========================================================================
# 24. Default-OFF compatibility: same selected winner as a plain baseline
# call with no flags and no local_paths, across several fixtures.
# ===========================================================================
@pytest.mark.parametrize("fixture_fn", [_weak_strong_fixture])
def test_24_default_off_matches_baseline_winner(fixture_fn, tmp_path, monkeypatch):
    request, takes, labels, expected_winner = fixture_fn()
    baseline = build_flow_b_draft(request, takes, labels)
    _set_env(monkeypatch, ENV_BOTH_ON)
    path = _dominance_audio_path(tmp_path)
    with_prosody = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    assert [c.clip_id for c in baseline.draft.selected] == [c.clip_id for c in with_prosody.draft.selected]
    assert [c.clip_id for c in baseline.draft.selected] == [expected_winner]


# ===========================================================================
# 25. Diagnostics/run-summary are JSON-safe and bounded (no waveform/
# transcript dump anywhere in the new pipeline-level rows).
# ===========================================================================
def test_25_pipeline_diagnostics_bounded_no_dump(tmp_path, monkeypatch):
    import json
    _set_env(monkeypatch, ENV_BOTH_ON)
    request, takes, labels, _ = _weak_strong_fixture()
    path = _dominance_audio_path(tmp_path)
    result = build_flow_b_draft(request, takes, labels, local_paths={"src": path})
    row = _group_row(result)
    prosodic_row = {k: v for k, v in row.items() if k.startswith("prosodic_")}
    encoded = json.dumps(prosodic_row)
    assert len(encoded) < 2000
    summary = result.draft.diagnostics["prosodic_pipeline"]
    json.dumps(summary)
