"""D-097.13 (bounded encargo) -- CLEAN RAW Watch+Listen checkpoint.

Makes the D-097.12 CLEAN_RAW_SELECTION observable on rendered media: a
bounded diagnostic render of the CLEAN_RAW_SELECTION through the SAME
canonical physical authorities production uses
(`render_plan.build_render_plan`, `live_render_qc.render_with_post_render_
qc`, `perceptual_watch_listen.review_rendered_candidate`), plus an explicit
audio-correlation membership proof (reusing `video00_quality_ladder.
verify_render_against_plan`'s own template-matching, not a second
implementation) that a clip claimed present is PHYSICALLY in the file and
a clip claimed absent is PHYSICALLY nowhere in it.

The RAW media here is a SYNTHETIC, speech-shaped, three-region proxy (same
deterministic-signal technique `test_cutsell_d097_4_join_probe_and_repair_
edge.py` already uses for offline QC verification) standing in for the
persisted Video00 stomach-family footage: this sandboxed environment's AWS
credentials are not valid for the real S3 bucket the production RAW media
lives in (verified: `ListBuckets`/`GetObject` both fail with
`InvalidAccessKeyId`), so the real Video00 footage is unreachable from
here without new paid infrastructure this task's authorization forbids.
The checkpoint mechanism itself -- render, technical QC, perceptual
Watch+Listen, audio-correlation membership proof, StageVerdict reduction
-- is exercised end-to-end against real ffmpeg-rendered media using the
production functions; only the SOURCE FOOTAGE identity is a stand-in.
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.clean_raw_checkpoint import (
    ROUTE_BOUNDARY_ENGINE,
    ROUTE_RENDERER,
    ROUTE_TECHNICAL_QC,
    ROUTE_UNKNOWN_CAPABILITY,
    STAGE_ERROR,
    STAGE_FAIL,
    STAGE_PASS,
    STAGE_UNCERTAIN,
    CleanRawCheckpointResult,
    ExpectedAbsence,
    ExpectedIndependence,
    ExpectedPresence,
    _membership_check,
    _reduce_verdict,
    run_clean_raw_checkpoint,
)
from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

SR = 22_050


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


def _voice(seconds: float, *, seed: int, f0: float = 118.0, syllable: float = 4.5, phase_offset: float = 0.0, sample_rate: int = SR) -> np.ndarray:
    """Deterministic speech-shaped signal (glottal pulse train + formants +
    syllabic envelope + plosive bursts). `video00_quality_ladder`'s audio
    correlation matches on RMS-energy/zero-crossing ENVELOPE shape (the
    feature real speech content varies by: different words, different
    pause/phrase timing), NOT raw pitch -- so `syllable` (distinct per
    region) is what makes three regions of one file independently
    identifiable, exactly the property real distinct spoken deliveries
    have; `f0` alone (pitch) is not discriminative for this matcher.
    `phase_offset` shifts where in the syllable cycle t=0 falls, without
    changing the envelope's shape/rate -- used to land a render join's
    t=0 in the MIDDLE of a quiet trough (comfortably >60ms of true quiet
    before the next voiced onset) rather than right on its rising edge."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    freq = f0 + 10.0 * np.sin(2 * np.pi * 0.6 * t)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    voiced = np.zeros(n)
    for k in range(1, 20):
        f = k * f0
        formant = np.exp(-((f - 600.0) / 250.0) ** 2) + 0.6 * np.exp(-((f - 1800.0) / 400.0) ** 2)
        voiced += (formant / k ** 0.5) * np.sin(k * phase)
    voiced /= np.max(np.abs(voiced))
    env = np.clip(np.sin(2 * np.pi * syllable * t + phase_offset) * 1.6 - 0.3, 0.0, 1.0)
    burst = np.zeros(n)
    onsets = np.flatnonzero(np.diff((env > 0.0).astype(int)) == 1)
    for onset in onsets:
        length = int(0.010 * sample_rate)
        burst[onset:onset + length] += rng.standard_normal(length) * np.linspace(1.0, 0.0, length)
    signal = 0.55 * voiced * env + 0.35 * burst
    signal += 0.0015 * rng.standard_normal(n)
    return np.clip(signal * 32767.0, -32767, 32767).astype(np.int16)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int = SR) -> str:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())
    return str(path)


def _mux(directory: Path, name: str, wav_path: str, seconds: float) -> str:
    """`testsrc` (a moving test pattern), not `color` (a frozen solid
    frame) -- see `test_cutsell_d097_4_join_probe_and_repair_edge.py`'s
    same convention. A frozen `color` source is itself a real physical
    defect (FROZEN_OR_REPEATED_FRAME / DEAD_BLACK_FRAME) to the reused
    production technical QC, which correctly flagged it as such the
    first time this fixture used `color` -- a fixture-authenticity bug,
    not a false positive in the reused detector."""
    mp4 = str(directory / name)
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=30:duration={seconds}",
        "-i", wav_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", mp4,
    ])
    return mp4


@pytest.fixture(scope="module")
def raw_source(tmp_path_factory):
    """One synthetic 'RAW' file, three regions, each a distinguishable
    synthetic voice: [0,4) abandoned, [5,9) aside, [10,16) clean."""
    directory = tmp_path_factory.mktemp("d097_13_raw")
    silence_a = np.zeros(int(1.0 * SR), dtype=np.int16)
    # `_voice`'s envelope is exactly 0 at t=0, and this region's render
    # join lands exactly at its t=0 (build_render_plan extracts straight
    # from the RAW span with no offset). At the default phase, t=0 sits
    # right on the trough's RISING edge, so the first voiced onset (and
    # its 10ms plosive burst) arrives only ~8ms in -- squarely inside the
    # 60ms cut-adjacent-energy window `perceptual_watch_listen` measures,
    # a real, correctly-detected UNCERTAIN (possible clipped word), not a
    # checkpoint bug. `phase_offset=3*pi/2` instead lands t=0 in the
    # MIDDLE of a trough, giving ~76ms of genuine quiet before the next
    # onset -- past the 60ms window -- so this fixture demonstrates a
    # truly clean join, the honest precondition for a PASS verdict.
    pcm = np.concatenate([
        _voice(4.0, seed=1, f0=110.0, syllable=2.2), silence_a,
        _voice(4.0, seed=2, f0=140.0, syllable=5.5), silence_a,
        _voice(6.0, seed=3, f0=170.0, syllable=3.7, phase_offset=3 * np.pi / 2),
    ])
    wav = _write_wav(directory / "raw.wav", pcm)
    return _mux(directory, "raw.mp4", wav, seconds=len(pcm) / SR)


def _clip(clip_id, source_asset_id, start, end, text, *, selected):
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0, start=start, end=end,
        text=text, caption_text=text, selected=selected,
    )


def _clean_raw_draft():
    abandoned = _clip("abandoned", "src", 0.0, 4.0, "abandoned attempt", selected=False)
    aside = _clip("aside", "src", 5.0, 9.0, "independent aside", selected=True)
    clean = _clip("clean", "src", 10.0, 16.0, "clean retry", selected=True)
    # `_reset_debris_at_edges_source_evidence` needs REAL persisted local
    # performance evidence (A-5 reset/break events) per source_asset_id to
    # evaluate at all -- with none it correctly reports UNCERTAIN rather
    # than silently PASSing. This synthetic proxy has no such evidence
    # file, but the underlying D-097.12 stomach fixture never recorded a
    # reset/break event on either "aside" or "clean" either, so an empty
    # (evaluated, zero-findings) evidence record for "src" is an honest
    # stand-in, not a fabricated result.
    diagnostics = {"whole_video_context": {"sources": [{"source_asset_id": "src", "events": []}]}}
    return DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="clean-raw-stomach", strategy=EditStrategy.STORYTELLING,
        selected=(aside, clean), alternates=(), discarded=(abandoned,), diagnostics=diagnostics,
    )


def _run_identity():
    return {"raw_identity": "synthetic-stomach-proxy-v1", "git_head": "b5b2ffe", "stage": "CLEAN_RAW"}


# --------------------------------------------------------------- end to end

def test_pass_end_to_end_through_the_real_render_qc_and_perceptual_authorities(tmp_path, raw_source):
    draft = _clean_raw_draft()
    before_selected, before_discarded = draft.selected, draft.discarded
    result = run_clean_raw_checkpoint(
        draft, {"src": raw_source},
        raw_media_path=raw_source, output_path=str(tmp_path / "diagnostic.mp4"),
        family="stomach", run_identity=_run_identity(),
        expected_present=[ExpectedPresence("aside", "independent aside"), ExpectedPresence("clean", "clean retry")],
        expected_absent=[ExpectedAbsence("abandoned", "src", 0.0, 4.0, "abandoned attempt")],
        expected_independent=[ExpectedIndependence("aside", "independent aside")],
    )
    assert isinstance(result, CleanRawCheckpointResult)
    assert result.verdict == STAGE_PASS, result.diagnostic_record.get("verdict_reason")
    record = result.diagnostic_record
    assert record["technical_qc"]["status"] == "PASS"
    assert record["physical_membership"]["present_all_confirmed"] is True
    assert record["physical_membership"]["absent_all_confirmed"] is True
    assert all(r["status"] == "PRESERVED_AS_OWN_SELECTION" for r in record["physical_membership"]["independent"])
    assert record["diagnostic_mp4_path"] and Path(record["diagnostic_mp4_path"]).exists()
    # required diagnostic-record fields
    for field_name in ("run_identity", "stage", "family", "selected_ids", "discarded_ids", "source_ranges",
                       "final_rendered_ranges", "diagnostic_mp4_path", "technical_qc", "watch_listen",
                       "unsupported_capabilities", "routed_owner", "final_stage_verdict"):
        assert field_name in record, field_name
    assert record["selected_ids"] == ["aside", "clean"]
    assert record["discarded_ids"] == ["abandoned"]
    assert record["routed_owner"] is None  # PASS carries no routed owner
    # Watch+Listen never mutates selection.
    assert draft.selected == before_selected and draft.discarded == before_discarded


def test_checkpoint_errors_cleanly_when_the_source_media_cannot_be_located(tmp_path, raw_source):
    draft = _clean_raw_draft()
    result = run_clean_raw_checkpoint(
        draft, {},  # no local_paths entry for "src" -> build_render_plan raises
        raw_media_path=raw_source, output_path=str(tmp_path / "diagnostic.mp4"),
        family="stomach", run_identity=_run_identity(),
    )
    assert result.verdict == STAGE_ERROR
    assert "exception" in result.diagnostic_record


# ------------------------------------------------------- membership proof

def test_membership_check_confirms_presence_and_absence_on_a_clean_render(tmp_path, raw_source):
    directory = tmp_path
    silence = np.zeros(int(0.5 * SR), dtype=np.int16)
    pcm = np.concatenate([_voice(4.0, seed=2, f0=140.0, syllable=5.5), silence, _voice(6.0, seed=3, f0=170.0, syllable=3.7, phase_offset=3 * np.pi / 2)])
    wav = _write_wav(directory / "good.wav", pcm)
    good_render = _mux(directory, "good.mp4", wav, seconds=len(pcm) / SR)

    membership = _membership_check(
        raw_media_path=raw_source, render_path=good_render,
        present=[ExpectedPresence("aside"), ExpectedPresence("clean")],
        absent=[ExpectedAbsence("abandoned", "src", 0.0, 4.0)],
        independent=[ExpectedIndependence("aside")],
        selected_spans={"aside": (5.0, 9.0), "clean": (10.0, 16.0)},
    )
    assert membership["present_all_confirmed"] is True
    assert membership["absent_all_confirmed"] is True
    assert membership["absent_violation"] is False
    assert membership["absent"][0]["status"] == "CONFIRMED_ABSENT"


def test_membership_check_detects_a_physically_reintroduced_discarded_clip(tmp_path, raw_source):
    """The defect this checkpoint exists to catch: a renderer/physical-plan
    bug that re-splices discarded material into the output even though the
    frozen plan never asked for it. No production code path can currently
    produce this (the renderer only ever renders `draft.selected`), so the
    'bad' render is constructed directly to prove the DETECTION works."""
    directory = tmp_path
    silence = np.zeros(int(0.5 * SR), dtype=np.int16)
    pcm = np.concatenate([
        _voice(4.0, seed=2, f0=140.0, syllable=5.5), silence, _voice(6.0, seed=3, f0=170.0, syllable=3.7, phase_offset=3 * np.pi / 2), silence,
        _voice(4.0, seed=1, f0=110.0, syllable=2.2),  # the abandoned attempt's audio, spliced back in
    ])
    wav = _write_wav(directory / "bad.wav", pcm)
    bad_render = _mux(directory, "bad.mp4", wav, seconds=len(pcm) / SR)

    membership = _membership_check(
        raw_media_path=raw_source, render_path=bad_render,
        present=[ExpectedPresence("aside"), ExpectedPresence("clean")],
        absent=[ExpectedAbsence("abandoned", "src", 0.0, 4.0)],
        independent=[],
        selected_spans={"aside": (5.0, 9.0), "clean": (10.0, 16.0)},
    )
    assert membership["absent_violation"] is True
    assert membership["absent"][0]["status"] == "PHYSICALLY_PRESENT"


# ------------------------------------------------------------ verdict reduction

class _QC:
    def __init__(self, status="PASS"):
        self.status = status


def _clean_membership():
    return {"absent_violation": False, "present_all_confirmed": True, "present": [{"status": "CONFIRMED"}],
            "absent": [{"status": "CONFIRMED_ABSENT"}], "absent_all_confirmed": True}


def test_reduce_verdict_pass_ignores_not_implemented_capabilities():
    review = {"status": "UNCERTAIN", "routing": {},
             "capability_status_counts": {"EVALUATED_PASS": 2, "EVALUATED_FAIL": 0, "UNCERTAIN": 0, "NOT_IMPLEMENTED": 4, "ERROR": 0}}
    verdict, owner, _reason = _reduce_verdict(_QC("PASS"), review, _clean_membership())
    assert verdict == STAGE_PASS and owner is None


def test_reduce_verdict_uncertain_when_a_capability_could_not_evaluate():
    review = {"status": "UNCERTAIN", "routing": {},
             "capability_status_counts": {"EVALUATED_PASS": 1, "EVALUATED_FAIL": 0, "UNCERTAIN": 1, "NOT_IMPLEMENTED": 4, "ERROR": 0}}
    verdict, owner, _reason = _reduce_verdict(_QC("PASS"), review, _clean_membership())
    assert verdict == STAGE_UNCERTAIN and owner == ROUTE_UNKNOWN_CAPABILITY


def test_reduce_verdict_fail_when_technical_qc_is_not_pass():
    review = {"status": "UNCERTAIN", "routing": {}, "capability_status_counts": {}}
    verdict, owner, _reason = _reduce_verdict(_QC("NEEDS_HUMAN_REVIEW"), review, _clean_membership())
    assert verdict == STAGE_FAIL and owner == ROUTE_TECHNICAL_QC


def test_reduce_verdict_fail_routes_a_perceptual_finding_to_its_named_owner():
    review = {"status": "FAIL", "routing": {"BoundaryEngine": 1}, "capability_status_counts": {"EVALUATED_FAIL": 1}}
    verdict, owner, _reason = _reduce_verdict(_QC("PASS"), review, _clean_membership())
    assert verdict == STAGE_FAIL and owner == ROUTE_BOUNDARY_ENGINE


def test_reduce_verdict_fail_when_an_expected_absent_clip_physically_appears():
    membership = dict(_clean_membership())
    membership["absent_violation"] = True
    review = {"status": "UNCERTAIN", "routing": {}, "capability_status_counts": {}}
    verdict, owner, _reason = _reduce_verdict(_QC("PASS"), review, membership)
    assert verdict == STAGE_FAIL and owner == ROUTE_RENDERER


def test_reduce_verdict_fail_when_an_expected_present_clip_is_not_found():
    membership = {"absent_violation": False, "present_all_confirmed": False,
                 "present": [{"status": "NOT_FOUND"}], "absent": [], "absent_all_confirmed": True}
    review = {"status": "UNCERTAIN", "routing": {}, "capability_status_counts": {}}
    verdict, owner, _reason = _reduce_verdict(_QC("PASS"), review, membership)
    assert verdict == STAGE_FAIL and owner == ROUTE_RENDERER


def test_only_the_four_allowed_verdicts_are_accepted():
    for value in (STAGE_PASS, STAGE_FAIL, STAGE_UNCERTAIN, STAGE_ERROR):
        CleanRawCheckpointResult(verdict=value, routed_owner=None, diagnostic_record={})
    with pytest.raises(ValueError):
        CleanRawCheckpointResult(verdict="MAYBE", routed_owner=None, diagnostic_record={})
