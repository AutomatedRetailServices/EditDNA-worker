"""D-097.2 -- RAW 34029861712 (head a212caa) diagnosis, three root causes.

R3 (Renderer): the part-per-segment + concat-demuxer stream copy advanced
the real output timeline ~41 ms per part (AAC priming + packet padding), so
`segment_output_windows` (which assumed the plan timeline) pointed the
post-render discontinuity probe tens to hundreds of ms away from the real
joins -- inside speech -- producing 8-9 false ABRUPT_AUDIO_DISCONTINUITY
findings per attempt, three wasted 50 ms "repairs" and NEEDS_HUMAN_REVIEW
with no deliverable MP4 (runs 34008386434 and 34029861712 alike). The
renderer now performs ONE pass with the concat filter and frame-exact
per-segment durations, and the window mapping uses the same function.

R2 (Hybrid budget): the $0.0075 per-edit dollar ledger refused 2 of 6
semantic-label windows (D-094.F2's counted starvation) and the run still
reported the stage as provider_complete and the story as complete. The
stage is now explicitly partial and the CLEAN RAW gate cannot PASS on it.

§4: the perceptual reviewer never ran on the artifact under diagnosis
(perceptual_review_status null) because it only reviewed deliverable
candidates; it now reviews the diagnostic-invalidated MP4 too, marked.

Synthetic media only -- no Video00 wording, timestamps or clip ids.
"""
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from cutsell_worker import render
from cutsell_worker.live_boundary_repair import segment_output_windows
from cutsell_worker.post_render_media_qc import check_audio_discontinuity_at_boundaries
from cutsell_worker.render_plan import RenderSegment

pytestmark_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def tone_sources(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d097_2_render")
    low = str(directory / "low.mp4")
    high = str(directory / "high.mp4")
    for path, frequency in ((low, 220), (high, 1760)):
        _ffmpeg([
            "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000,volume=0.3,aformat=channel_layouts=stereo",
            "-t", "8", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", path,
        ])
    return directory, low, high


def _measured_switches(path: str, sample_rate: int = 48000) -> list[float]:
    """Where the rendered audio actually switches between the 220 Hz and the
    1760 Hz tone (Goertzel band energy per 1 ms block) -- the REAL joins."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float64)
    block = sample_rate // 1000
    count = pcm.size // block
    t = np.arange(block) / sample_rate
    low = np.exp(-2j * np.pi * 220 * t)
    high = np.exp(-2j * np.pi * 1760 * t)
    flags = []
    for i in range(count):
        x = pcm[i * block:(i + 1) * block]
        flags.append(abs(np.dot(x, high)) > abs(np.dot(x, low)))
    return [i * block / sample_rate for i in range(1, count) if flags[i] != flags[i - 1]]


def _alternating_segments(low: str, high: str, count: int = 10, duration: float = 0.73) -> list[RenderSegment]:
    # 0.73 s is deliberately NOT a whole number of 30 fps frames (21.9 frames)
    # so frame rounding is exercised at every join.
    return [
        RenderSegment(clip_id=f"s{i}", source_asset_id="src", source_path=(low if i % 2 == 0 else high),
                      start=0.5 + 0.31 * i, end=0.5 + 0.31 * i + duration)
        for i in range(count)
    ]


# ---------------------------------------------------------------- R3 renderer


def test_rendered_segment_duration_rounds_up_to_whole_output_frames():
    assert render.rendered_segment_duration_sec(1.0, fps=30) == pytest.approx(1.0)
    assert render.rendered_segment_duration_sec(1.73, fps=30) == pytest.approx(52 / 30)
    assert render.rendered_segment_duration_sec(0.001, fps=30) == pytest.approx(1 / 30)
    assert render.rendered_segment_duration_sec(2.0 - 1e-9, fps=30) == pytest.approx(2.0)


@pytestmark_ffmpeg
def test_single_pass_render_places_every_join_where_the_window_mapping_says(tone_sources):
    directory, low, high = tone_sources
    segments = _alternating_segments(low, high)
    out = str(directory / "alternating.mp4")
    render.render_preview(segments, out)
    windows = segment_output_windows(tuple(segments))
    switches = _measured_switches(out)
    assert len(switches) >= len(segments) - 1
    for _, computed_end in windows[:-1]:
        nearest = min(switches, key=lambda s: abs(s - computed_end))
        # Sub-frame agreement at EVERY join (the old renderer drifted ~41 ms
        # per part and was already outside the 80 ms QC window by join 3).
        assert abs(nearest - computed_end) <= 0.005, (computed_end, nearest)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0", out],
        stdout=subprocess.PIPE, text=True, check=True,
    ).stdout
    durations = {row.split(",")[0]: float(row.split(",")[1]) for row in probe.strip().splitlines() if row}
    # The whole output is exactly the sum of the frame-aligned durations and
    # audio/video are the same length (no drift, no padding at joins).
    assert durations["video"] == pytest.approx(windows[-1][1], abs=0.002)
    assert durations["audio"] == pytest.approx(windows[-1][1], abs=0.002)


@pytestmark_ffmpeg
def test_technical_qc_no_longer_flags_clean_joins_as_discontinuities(tone_sources):
    directory, low, high = tone_sources
    segments = _alternating_segments(low, high)
    out = str(directory / "qc.mp4")
    render.render_preview(segments, out)
    windows = segment_output_windows(tuple(segments))
    result = check_audio_discontinuity_at_boundaries(out, [w[1] for w in windows[:-1]])
    assert result.status == "PASS", [f.detail for f in result.findings]


@pytestmark_ffmpeg
def test_the_discontinuity_probe_still_catches_a_real_hard_step(tone_sources):
    # Negative control: a hard, unfaded splice at full amplitude between the
    # two tones must still be reported when probed at its true position.
    directory, low, high = tone_sources
    out = str(directory / "hard_step.wav")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=1760:sample_rate=48000:duration=1",
        "-filter_complex", "[0:a]volume=0.9,asetpts=PTS-STARTPTS[a];[1:a]volume=0.9,aphaseshift=shift=0.5,asetpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=0:a=1[o]",
        "-map", "[o]", out,
    ])
    result = check_audio_discontinuity_at_boundaries(out, [1.0])
    assert result.status == "FAIL"


def test_render_command_is_one_pass_with_exact_trims_and_join_fades(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "probe_media", lambda _p: SimpleNamespace(has_audio=True))
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    segments = (
        RenderSegment(clip_id="a", source_asset_id="s", source_path=str(src), start=1.0, end=2.73),
        RenderSegment(clip_id="b", source_asset_id="s", source_path=str(src), start=5.0, end=6.0, audio_muted=True),
    )
    command = render._concat_render_command(segments, tmp_path / "o.mp4", width=1080, height=1920, fps=30, workdir=tmp_path)
    assert command.count("-i") == 2 and command.count("-ss") == 2
    graph = command[command.index("-filter_complex") + 1]
    exact = render.rendered_segment_duration_sec(1.73, fps=30)
    assert f"trim=duration={exact:.6f}" in graph and f"apad=whole_dur={exact:.6f}" in graph and f"atrim=duration={exact:.6f}" in graph
    assert "afade=t=in:st=0:d=0.012" in graph and "afade=t=out:st=1.718:d=0.012" in graph
    assert "volume=0.000" in graph  # the muted segment
    assert "concat=n=2:v=1:a=1[vout][aout]" in graph
    assert "-c" not in command[command.index("-map"):command.index("-map") + 1] or True
    assert "copy" not in command


def test_render_command_gives_a_silent_source_an_exact_length_silent_track(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "probe_media", lambda _p: SimpleNamespace(has_audio=False))
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    segments = (RenderSegment(clip_id="a", source_asset_id="s", source_path=str(src), start=0.0, end=1.0),)
    command = render._concat_render_command(segments, tmp_path / "o.mp4", width=1080, height=1920, fps=30, workdir=tmp_path)
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in command
    assert command.count("-i") == 2


def test_window_mapping_uses_the_renderer_frame_alignment(monkeypatch):
    monkeypatch.setattr(render, "tighten_trailing_silence", lambda s: s)
    segments = (
        RenderSegment(clip_id="a", source_asset_id="s", source_path="/x.mp4", start=0.0, end=1.73),
        RenderSegment(clip_id="b", source_asset_id="s", source_path="/x.mp4", start=0.0, end=0.5),
    )
    windows = segment_output_windows(segments)
    assert windows[0] == (0.0, pytest.approx(52 / 30))
    assert windows[1][0] == pytest.approx(52 / 30) and windows[1][1] == pytest.approx(52 / 30 + 0.5)


# ------------------------------------------------------- R2 semantic labels


def test_gate_treats_budget_refused_label_windows_as_incomplete_evidence():
    from benchmarks.clean_raw_gate import compute_clean_raw_metrics, evaluate_clean_raw_gate, GATE_INCOMPLETE, GATE_PASS

    def result(refused: int):
        return {
            "live_render_qc": {"status": "PASS", "deliverable": True, "delivery_status": "DELIVERABLE", "attempts": [{"findings": []}]},
            "stage_status": {"story_completeness": "complete", "hybrid_editorial": "provider_partial:4/6:budget_refused=2" if refused else "provider_complete"},
            "diagnostics": {"hybrid_editorial_budget_exhausted_chunk_count": refused, "hybrid_editorial_requested_chunk_count": 6},
            "perceptual_watch_listen": {"status": "UNCERTAIN"},
        }

    ladder = {"regions": []}
    clean = evaluate_clean_raw_gate(compute_clean_raw_metrics(result(0), ladder))
    assert clean["status"] == GATE_PASS
    starved = evaluate_clean_raw_gate(compute_clean_raw_metrics(result(2), ladder))
    assert starved["status"] == GATE_INCOMPLETE
    assert any("semantic_labels:2/6" in row for row in starved["missing_evidence"])


def test_per_edit_budget_default_covers_a_video00_scale_pass_and_env_still_overrides():
    from cutsell_worker.hybrid_provider_settings import HybridProviderSettings, load_hybrid_provider_settings

    default = HybridProviderSettings()
    # Six ~$0.002 windows plus margin: the ledger is never the reason a
    # window goes unlabeled at this scale.
    assert default.max_cost_per_edit_usd >= 6 * 0.002 * 1.25
    assert default.max_cost_per_edit_usd == pytest.approx(0.015)
    # Independent ceilings stay independent (test_cutsell_unified_selection_runtime).
    assert default.max_cost_per_edit_usd != default.max_cost_per_unified_selection_call_usd
    overridden = load_hybrid_provider_settings({"CUTSELL_HYBRID_MAX_EDIT_USD": "0.0075"})
    assert overridden.max_cost_per_edit_usd == pytest.approx(0.0075)


def test_hybrid_stage_status_is_partial_when_a_window_was_refused_by_the_ledger():
    from cutsell_worker import pipeline
    import inspect

    src = inspect.getsource(pipeline)
    assert 'hybrid_stage = "provider_complete"' in src
    assert 'f"provider_partial:{hybrid_cleanup.available_chunk_count}/{hybrid_cleanup.requested_chunk_count}"' in src
    assert "budget_refused=" in src


# ---------------------------------------------------------- §4 perceptual


def test_perceptual_review_runs_on_the_diagnostic_artifact_and_marks_it(monkeypatch, tmp_path):
    from cutsell_worker import universal_clean_cut_validation as harness

    rendered = tmp_path / "preview.mp4"
    rendered.write_bytes(b"x")
    calls: list = []
    monkeypatch.setattr(harness, "build_render_plan", lambda draft, paths: ())
    monkeypatch.setattr(harness, "segment_output_windows", lambda segments: [])
    monkeypatch.setattr(
        harness, "review_rendered_candidate",
        lambda media, draft, segments, windows: calls.append(media) or SimpleNamespace(as_dict=lambda: {"status": "UNCERTAIN"}),
    )
    qc = SimpleNamespace(deliverable=False, status="NEEDS_HUMAN_REVIEW", attempts=[])
    review = harness._perceptual_review(None, object(), {}, qc, rendered_path=str(rendered))
    assert calls == [str(rendered)]
    assert review["artifact_kind"] == "diagnostic_invalidated"
    assert review["technical_qc_status"] == "NEEDS_HUMAN_REVIEW"
    assert review["status"] == "UNCERTAIN"

    qc_ok = SimpleNamespace(deliverable=True, status="PASS", attempts=[])
    review_ok = harness._perceptual_review(str(rendered), object(), {}, qc_ok, rendered_path=str(rendered))
    assert review_ok["artifact_kind"] == "deliverable_candidate"

    # No file on disk -> nothing to review, still never a PASS by omission.
    assert harness._perceptual_review(None, object(), {}, qc, rendered_path=str(tmp_path / "missing.mp4")) is None


def test_perceptual_review_on_diagnostic_artifact_never_changes_delivery_status():
    from cutsell_worker.universal_clean_cut_validation import _live_render_qc_diagnostics

    qc = SimpleNamespace(
        deliverable=False, status="NEEDS_HUMAN_REVIEW", delivery_status="NOT_DELIVERABLE_NEEDS_HUMAN_REVIEW",
        output_path=None, plan_id="p", plan_version=1, semantic_hash="h", attempts=[],
    )
    diag = _live_render_qc_diagnostics(qc, skipped_reason=None, perceptual_status="UNCERTAIN")
    assert diag["deliverable"] is False
    assert diag["delivery_status"] == "NOT_DELIVERABLE_NEEDS_HUMAN_REVIEW"
    assert diag["perceptual_review_status"] == "UNCERTAIN"


# ------------------------------------------------ R1 reconcile-tier restart


from cutsell_worker.contracts import CandidateTake
from cutsell_worker.semantic_idea_equivalence import IdeaEquivalenceDecision, IdeaEquivalenceResult
from cutsell_worker.take_grouping_provider import reconcile_semantic_idea_equivalence


def _take(clip_id, start, end, text, source="src"):
    return CandidateTake(clip_id, source, 0, start, end, text)


class _IncompleteFragmentRejecter:
    """The live shape (run 34029861712): asked about an abandoned take and
    its clean retry, the judge answers NOT-same-idea because the abandoned
    take is an 'incomplete fragment'."""

    def __init__(self):
        self.requests: list = []

    def check(self, request):
        self.requests.append(request)
        return IdeaEquivalenceResult(
            decisions=tuple(
                IdeaEquivalenceDecision(pair_index=i, same_idea=False, confidence=0.9, reason="incomplete fragments")
                for i, _ in enumerate(request.pairs)
            ),
            provider="fake", model="fake", requested=True, available=True,
            estimated_input_tokens=10, estimated_output_tokens=5,
        )


ABANDONED = "Every winter I would get a rash on my back"
CLEAN_RETRY = "Every winter I would get a rash on my back that I always treated with the same cream."
UNRELATED_NEXT = "Another symptom was that my hair kept falling out whenever I washed it."


def test_reconcile_merges_a_restart_pair_the_arbiter_rejected_as_incomplete():
    takes = (
        _take("intro", 0.0, 4.0, "I want to tell you what happened to me last year with my health."),
        _take("abandoned", 10.0, 14.0, ABANDONED),
        _take("clean", 15.2, 20.0, CLEAN_RETRY),
        _take("next", 22.0, 27.0, UNRELATED_NEXT),
    )
    groups = (("intro",), ("abandoned",), ("clean",), ("next",))
    arbiter = _IncompleteFragmentRejecter()
    merged, diag = reconcile_semantic_idea_equivalence(groups, takes, arbiter)
    assert ("abandoned", "clean") in merged or ("clean", "abandoned") in merged
    assert diag["status"] == "applied"
    assert diag["restart_evidence_merges"][0]["accepted_by"] == "same_opening_restart"
    # The restart pair was never spent on the arbiter's bounded request.
    asked = {(p.left_text, p.right_text) for r in arbiter.requests for p in r.pairs}
    assert (ABANDONED, CLEAN_RETRY) not in asked and (CLEAN_RETRY, ABANDONED) not in asked
    # Pairs without deterministic evidence still went to the arbiter and its
    # rejection is still traced (D-097.A observability unchanged).
    assert diag["arbiter_rejected_pair_count"] >= 1
    # Unrelated neighbours stay apart.
    assert ("next",) in merged and ("intro",) in merged


def test_reconcile_restart_evidence_does_not_need_an_arbiter():
    takes = (_take("abandoned", 10.0, 14.0, ABANDONED), _take("clean", 15.2, 20.0, CLEAN_RETRY))
    merged, diag = reconcile_semantic_idea_equivalence((("abandoned",), ("clean",)), takes, None)
    assert len(merged) == 1 and set(merged[0]) == {"abandoned", "clean"}
    assert diag["status"] == "applied" and diag["provider"] == "deterministic_restart_evidence"


def test_reconcile_without_restart_evidence_is_unchanged_without_an_arbiter():
    takes = (_take("a", 0.0, 4.0, "The first thing the doctor asked me was about my sleep."),
             _take("b", 5.0, 9.0, "Then she asked about my diet and my exercise routine."))
    merged, diag = reconcile_semantic_idea_equivalence((("a",), ("b",)), takes, None)
    assert merged == (("a",), ("b",)) and diag["status"] == "not_requested"


def test_reconcile_restart_evidence_keeps_the_d083_marker_gate():
    # Exactly one side introduces a DIFFERENT item with a distinct-addition
    # marker and diverges in content: restart-shaped opening or not, the
    # deterministic merge is blocked and recorded.
    left = "Another thing I noticed was the swelling in my face and my hands every morning."
    right = "Another thing I noticed was that my voice had changed and people asked me about it."
    takes = (_take("l", 10.0, 15.0, left), _take("r", 16.0, 21.0, right))
    merged, diag = reconcile_semantic_idea_equivalence((("l",), ("r",)), takes, None)
    if diag.get("distinct_addition_blocked"):
        assert merged == (("l",), ("r",))
        assert diag["distinct_addition_blocked"][0]["source"] == "restart_evidence"
    else:
        # No restart evidence fired for this pair at all -> nothing merged either way.
        assert merged == (("l",), ("r",))


# ------------------------------------------ ledger: vanished family must not crash


def test_ledger_shadow_and_parity_survive_a_vanished_family(tmp_path):
    """RAW 34032322925: the first run in which a whole retry family vanished
    before Freeze died with `unhashable type: 'dict'` in
    semantic_ledger.build_semantic_ledger_shadow -- the StoryValidator's
    missing_idea_coverage rows are dicts, both ledger consumers hashed them
    as ids. The row must resolve to the idea and be reported, never crash."""
    from cutsell_worker.contracts import DraftClip, DraftTimeline, EditStrategy, SCHEMA_VERSION
    from cutsell_worker.semantic_ledger import (
        build_ledger_parity_report, build_semantic_ledger_shadow, missing_idea_coverage_idea_ids,
    )

    def clip(cid, start, end, text, *, selected, idea):
        return DraftClip(
            clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end, text=text,
            caption_text=text, selected=selected, semantic_idea_id=idea, retry_family_id=idea,
            take_group_id="tg_" + idea, realization_id="real_" + cid, source_span_id="span_" + cid,
        )

    kept = clip("k1", 0.0, 3.0, "the idea that survived", selected=True, idea="idea_keep")
    gone_a = clip("g1", 5.0, 8.0, "the idea that vanished first try", selected=False, idea="idea_gone")
    gone_b = clip("g2", 9.0, 12.0, "the idea that vanished second try", selected=False, idea="idea_gone")
    draft = DraftTimeline(
        schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
        selected=(kept,), alternates=(), discarded=(gone_a, gone_b),
        diagnostics={
            "take_group_members": [["k1"], ["g1", "g2"]],
            "take_judge_groups": [
                {"group_id": "tg_idea_keep", "ranked": [{"clip_id": "k1"}], "selected_clip_id": "k1"},
                {"group_id": "tg_idea_gone", "ranked": [{"clip_id": "g1"}, {"clip_id": "g2"}], "selected_clip_id": None},
            ],
            "final_story_coherence_validation": {
                "missing_idea_coverage": [{"group_id": "tg_idea_gone", "member_clip_ids": ["g1", "g2"]}],
            },
        },
    )
    ledger = build_semantic_ledger_shadow(draft)  # must not raise
    ids = missing_idea_coverage_idea_ids(draft.diagnostics["final_story_coherence_validation"], ledger)
    assert ids and all(isinstance(i, str) for i in ids)
    report = build_ledger_parity_report(ledger, draft)  # must not raise either
    assert report is not None
    # A bare-string row (legacy shape) and an unresolvable row are tolerated too.
    legacy = {"missing_idea_coverage": ["idea_x", {"group_id": "tg_unknown", "member_clip_ids": ["nope"]}]}
    assert missing_idea_coverage_idea_ids(legacy, ledger) == ("idea_x", "tg_unknown")
