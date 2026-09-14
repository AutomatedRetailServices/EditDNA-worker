"""D-094.3 -- run 33995806350 (df3946e): first live Freeze PASS + render.

F14 renderer: hard cuts between segments left step discontinuities (8 of 22
    joins flagged ABRUPT_AUDIO_DISCONTINUITY). Per-segment 12 ms edge fades.
F13 QC loop: gave up on the FIRST unrepairable physical finding while later
    findings were repairable. Now tries each finding in QC order.
F9  placement: a restored clip with no idea context was appended after the
    CTA; now inserted by recording position.
F8  labels: the per-clip cross-window merge let a "winner" from a window
    that never saw the better sibling tie with the family-complete window's
    verdict. Family-complete windows now decide.
F4b dedup: two same-number claims with low token overlap were never put to
    the claim-equivalence arbiter; now consulted from a lower floor.

Fixtures are generic; no Video00 clip ids.
"""
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace as NS

import pytest

from cutsell_worker import live_render_qc, render
from cutsell_worker.canonical_edit_plan import build_canonical_edit_plan  # noqa: F401  (import parity with live QC tests)
from cutsell_worker.contracts import SCHEMA_VERSION, CandidateTake, DraftClip, DraftTimeline, EditStrategy
from cutsell_worker.pipeline import _semantic_best_take, family_scoped_semantic_decisions
from cutsell_worker.post_render_media_qc import check_audio_discontinuity_at_boundaries
from cutsell_worker.post_render_watch_listen_qc import LINGERING_ACCIDENTAL_SILENCE, PostRenderFinding, PostRenderQCResult
from cutsell_worker.realization_resolver import (
    PLACEMENT_UNIT_NO_ANCHOR_APPEND,
    PLACEMENT_UNIT_SOURCE_ORDER_INSERTION,
    _claims_dedup_equivalent,
    _place_restored_clips_at_story_position,
    build_requirement_groups,
)
from cutsell_worker.render_plan import RenderSegment
from cutsell_worker.semantic_ledger import CanonicalClaimRecord

ffmpeg_available = shutil.which("ffmpeg") is not None


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


# ---------------------------------------------------------------------------
# F14: audio join fades
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_dc_sources(tmp_path_factory):
    """Two 1.5 s sources whose audio is a constant at opposite polarity: a
    hard cut between them is the worst-case step a splice can produce."""
    d = tmp_path_factory.mktemp("d094_3_render")
    paths = []
    for name, level in (("pos", "0.8"), ("neg", "-0.8")):
        path = str(d / f"{name}.mp4")
        _ffmpeg([
            "-f", "lavfi", "-i", f"aevalsrc={level}:d=1.5:s=48000",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=1.5",
            "-c:v", "libx264", "-c:a", "pcm_s16le", "-shortest", "-y", path.replace(".mp4", ".mov"),
        ])
        paths.append(path.replace(".mp4", ".mov"))
    return d, paths


def _seg(cid, path, start, end):
    return RenderSegment(clip_id=cid, source_asset_id="src", source_path=path, start=start, end=end)


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not available")
def test_f14_join_fades_remove_the_splice_step_and_keep_boundaries(monkeypatch, two_dc_sources):
    d, (pos, neg) = two_dc_sources
    segments = (_seg("a", pos, 0.0, 1.0), _seg("b", neg, 0.0, 1.0))
    # Red shape: no fades -> the splice at 1.0 s is a hard step.
    monkeypatch.setattr(render, "_AUDIO_JOIN_FADE_SEC", 0.0)
    out_hard = str(d / "hard.mp4")
    render.render_preview(segments, out_hard, width=64, height=64, fps=25)
    hard = check_audio_discontinuity_at_boundaries(out_hard, [1.0], sample_rate=48_000)
    assert hard.status == "FAIL", "fixture must reproduce the live click at the join"
    # Green: default 12 ms fades -> click-free, and the boundary itself is unchanged.
    monkeypatch.setattr(render, "_AUDIO_JOIN_FADE_SEC", 0.012)
    out_soft = str(d / "soft.mp4")
    render.render_preview(segments, out_soft, width=64, height=64, fps=25)
    soft = check_audio_discontinuity_at_boundaries(out_soft, [1.0], sample_rate=48_000)
    assert soft.status == "PASS", soft.findings
    probe_hard = render.probe_media(out_hard)
    probe_soft = render.probe_media(out_soft)
    assert abs(float(probe_soft.duration_sec) - float(probe_hard.duration_sec)) < 0.05


def test_f14_fades_are_skipped_for_very_short_segments_and_never_negative():
    assert render._audio_join_fade_filters(0.10) == []
    filters = render._audio_join_fade_filters(2.0)
    assert filters == ["afade=t=in:st=0:d=0.012", "afade=t=out:st=1.988:d=0.012"]


# ---------------------------------------------------------------------------
# F13: the QC loop tries every physical finding before declaring UNREPAIRABLE
# ---------------------------------------------------------------------------

def _clip(cid, start, end, text):
    return DraftClip(clip_id=cid, source_asset_id="src", source_order=0, start=start, end=end, text=text, caption_text=text, selected=True)


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("d094_3_qc")
    path = str(d / "source.mp4")
    _ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=8",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", path,
    ])
    return path


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not available")
def test_f13_unrepairable_first_finding_no_longer_hides_a_repairable_later_one(monkeypatch, tmp_path, source_video):
    a, b = _clip("a", 0.0, 3.0, "first idea"), _clip("b", 3.0, 6.0, "second idea")
    draft = DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                          selected=(a, b), alternates=(), discarded=(), diagnostics={})
    segments = (_seg("a", source_video, 0.0, 3.0), _seg("b", source_video, 3.0, 6.0))
    mid_segment = PostRenderFinding(kind=LINGERING_ACCIDENTAL_SILENCE, start=1.0, end=1.8, detail={}, routes_to="BoundaryEngine")
    trailing_edge = PostRenderFinding(kind=LINGERING_ACCIDENTAL_SILENCE, start=2.7, end=3.0, detail={}, routes_to="BoundaryEngine")
    calls = []

    def fake_qc(media_path, **kwargs):
        calls.append(media_path)
        if len(calls) == 1:
            return PostRenderQCResult(status="FAIL", findings=(mid_segment, trailing_edge))
        return PostRenderQCResult(status="PASS", findings=())
    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", fake_qc)

    result = live_render_qc.render_with_post_render_qc(draft, segments, str(tmp_path / "out.mp4"), max_attempts=3)
    assert result.status == "PASS"
    first = result.attempts[0]
    assert first.status == "PHYSICAL_FAIL_REPAIRED"
    assert first.unrepairable_finding_count == 1
    assert first.repair_target["start"] == 2.7 and first.repair_applied["clip_id"] == "a"


@pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not available")
def test_f13_all_unrepairable_is_still_unrepairable_and_bounded(monkeypatch, tmp_path, source_video):
    a, b = _clip("a", 0.0, 3.0, "first idea"), _clip("b", 3.0, 6.0, "second idea")
    draft = DraftTimeline(schema_version=SCHEMA_VERSION, project_id="p", strategy=EditStrategy.STORYTELLING,
                          selected=(a, b), alternates=(), discarded=(), diagnostics={})
    segments = (_seg("a", source_video, 0.0, 3.0), _seg("b", source_video, 3.0, 6.0))
    mids = tuple(PostRenderFinding(kind=LINGERING_ACCIDENTAL_SILENCE, start=s, end=s + 0.5, detail={}, routes_to="BoundaryEngine") for s in (1.0, 4.0))
    monkeypatch.setattr(live_render_qc, "run_post_render_media_qc", lambda *a, **k: PostRenderQCResult(status="FAIL", findings=mids))
    result = live_render_qc.render_with_post_render_qc(draft, segments, str(tmp_path / "out.mp4"), max_attempts=3)
    assert result.status == "NEEDS_HUMAN_REVIEW"
    assert len(result.attempts) == 1 and result.attempts[0].status == "PHYSICAL_FAIL_UNREPAIRABLE"
    assert result.attempts[0].unrepairable_finding_count == 2 and result.attempts[0].repair_target is None


# ---------------------------------------------------------------------------
# F9: source-order insertion
# ---------------------------------------------------------------------------

def _ns(cid, start, idea):
    return NS(clip_id=cid, start=start, realization_id=f"real_{cid}", semantic_idea_id=idea)


def _place(kept, restored, legacy, ideas, **kw):
    log = []
    out = _place_restored_clips_at_story_position(list(kept), list(restored), tuple(legacy), ideas_by_realization=ideas, placement_log=log, **kw)
    return [c.clip_id for c in out], log


def test_f9_restored_clip_without_idea_context_lands_at_its_recording_position():
    hook, x, y, cta = _ns("hook", 0.0, "i_hook"), _ns("x", 82.0, "i_x"), _ns("y", 120.0, "i_y"), _ns("cta", 357.0, "i_cta")
    legacy = [hook, y, cta]  # x was never selected originally: no departed sibling, no anchor
    ideas = {"real_hook": "i_hook", "real_x": "i_x", "real_y": "i_y", "real_cta": "i_cta"}
    ids, log = _place([hook, y, cta], [x], legacy, ideas)
    assert ids == ["hook", "x", "y", "cta"]
    assert log[-1]["unit_type"] == PLACEMENT_UNIT_SOURCE_ORDER_INSERTION
    assert log[-1]["placement_reason"] == "source_order_before_first_later_start" and log[-1]["successor_anchor"] == "y"


def test_f9_true_tail_clip_is_still_appended():
    hook, cta, tail = _ns("hook", 0.0, "i_hook"), _ns("cta", 357.0, "i_cta"), _ns("tail", 400.0, "i_tail")
    ideas = {"real_hook": "i_hook", "real_cta": "i_cta", "real_tail": "i_tail"}
    ids, log = _place([hook, cta], [tail], [hook, cta], ideas)
    assert ids == ["hook", "cta", "tail"]
    assert log[-1]["unit_type"] == PLACEMENT_UNIT_NO_ANCHOR_APPEND


def test_f9_insertion_never_splits_a_placed_composite_block():
    hook, a, b, later, cta = _ns("hook", 0.0, "i_hook"), _ns("A", 100.0, "i_ab"), _ns("B", 110.0, "i_ab"), _ns("later", 200.0, "i_l"), _ns("cta", 300.0, "i_cta")
    x = _ns("x", 105.0, "i_x")  # recorded between A and B
    legacy = [hook, a, later, cta]
    ideas = {"real_hook": "i_hook", "real_A": "i_ab", "real_B": "i_ab", "real_later": "i_l", "real_cta": "i_cta", "real_x": "i_x"}
    ids, log = _place([hook, a, later, cta], [b, x], legacy, ideas, composite_order_by_idea={"i_ab": ("real_A", "real_B")})
    assert ids == ["hook", "x", "A", "B", "later", "cta"]  # block head, never inside A|B


# ---------------------------------------------------------------------------
# F8: family-complete window labels
# ---------------------------------------------------------------------------

def _take(cid, start):
    return CandidateTake(cid, "src", 0, start, start + 5.0, f"delivery {cid}")


def _window(chunk_index, member_ids, labels):
    return {"chunk_index": chunk_index, "member_ids": list(member_ids),
            "decisions": [{"clip_id": c, "label": l, "confidence": k} for c, (l, k) in labels.items()]}


def test_f8_family_complete_window_overrides_the_cross_window_merge():
    aside, mono, later = _take("aside", 10.0), _take("mono", 20.0), _take("later", 30.0)
    members = (aside, mono, later)
    global_merge = {"aside": ("alternate", 0.85), "mono": ("winner", 0.96), "later": ("winner", 0.95)}
    windows = [
        _window(2, ["x1", "aside", "mono"], {"aside": ("alternate", 0.8), "mono": ("winner", 0.96)}),  # never saw `later`
        _window(3, ["aside", "mono", "later", "x2"], {"aside": ("alternate", 0.85), "mono": ("alternate", 0.88), "later": ("winner", 0.95)}),
        _window(4, ["later", "x3"], {"later": ("winner", 0.95)}),
    ]
    scoped, source = family_scoped_semantic_decisions(members, global_merge, windows)
    assert source["family_complete_window_chunk_indices"] == [3]
    assert scoped["mono"] == ("alternate", 0.88) and scoped["later"] == ("winner", 0.95)
    # Red shape (global merge): two winners -> not decisive.
    sel_global, _, reason_global = _semantic_best_take(members, global_merge, "mono", ())
    assert reason_global != "single_semantic_winner"
    # Green: the family window's single winner decides.
    sel, preferred, reason = _semantic_best_take(members, scoped, "mono", ())
    assert (sel, preferred, reason) == ("later", "later", "single_semantic_winner")


def test_f8_without_a_family_complete_window_the_merge_is_unchanged():
    members = (_take("a", 1.0), _take("b", 2.0), _take("c", 3.0))
    global_merge = {"a": ("winner", 0.9), "b": ("alternate", 0.7), "c": ("keep", 0.5)}
    windows = [_window(0, ["a", "b"], {"a": ("winner", 0.9), "b": ("alternate", 0.7)}), _window(1, ["b", "c"], {"b": ("alternate", 0.7), "c": ("keep", 0.5)})]
    scoped, source = family_scoped_semantic_decisions(members, global_merge, windows)
    assert scoped == global_merge and source is None


def test_f8_two_complete_windows_merge_by_priority_among_themselves_only():
    members = (_take("a", 1.0), _take("b", 2.0))
    global_merge = {"a": ("winner", 0.99), "b": ("winner", 0.9)}  # polluted by a partial window
    windows = [
        _window(0, ["a", "z"], {"a": ("winner", 0.99)}),
        _window(1, ["a", "b"], {"a": ("alternate", 0.8), "b": ("winner", 0.9)}),
        _window(2, ["a", "b", "y"], {"a": ("alternate", 0.85), "b": ("winner", 0.92)}),
    ]
    scoped, source = family_scoped_semantic_decisions(members, global_merge, windows)
    assert source["family_complete_window_chunk_indices"] == [1, 2]
    assert scoped["a"] == ("alternate", 0.85) and scoped["b"] == ("winner", 0.92)


# ---------------------------------------------------------------------------
# F4b: same-number claims reach the arbiter from a lower floor
# ---------------------------------------------------------------------------

def _claim(cid, text, tokens, *, rid):
    return CanonicalClaimRecord(
        canonical_claim_id=cid, claim_type="MEASUREMENT_QUANTITY", content_tokens=frozenset(tokens), importance="CRITICAL",
        source_realization_ids=(rid,), covered_by_realization_ids=(), coverage_state="unresolved", text=text, negation_role="",
    )


class _YesArbiter:
    def __init__(self):
        self.asked = []

    def claim_covered(self, left, right):
        self.asked.append((left, right))
        return True, 0.92, "same hereditary statistic"


FULL = _claim("cc_full", "solo un 5-10 % son de carácter hereditario.", {"5", "10", "solo", "carácter", "hereditario", "bien"}, rid="real_full")
TRUNC = _claim("cc_trunc", "estoy convencida, y la ciencia lo avala, que solo un 5-10 % de los", {"5", "10", "solo", "convencida", "ciencia", "avala"}, rid="real_trunc")
OTHER = _claim("cc_other", "solo un 3 % de los casos.", {"3", "solo", "casos"}, rid="real_other")


def test_f4b_same_numbers_low_overlap_now_consult_the_arbiter():
    arbiter = _YesArbiter()
    log = []
    assert _claims_dedup_equivalent(FULL, TRUNC, claim_equivalence_arbiter=arbiter, arbiter_log=log) is True
    assert len(arbiter.asked) == 1 and log[0]["verdict"] is True
    groups = build_requirement_groups((FULL, TRUNC), claim_equivalence_arbiter=arbiter)
    assert len(groups) == 1 and set(groups[0].member_claim_ids) == {"cc_full", "cc_trunc"}


def test_f4b_without_an_arbiter_same_numbers_still_fail_open_to_distinct():
    assert _claims_dedup_equivalent(FULL, TRUNC) is False
    assert len(build_requirement_groups((FULL, TRUNC))) == 2


def test_f4b_different_numbers_never_reach_the_arbiter():
    arbiter = _YesArbiter()
    assert _claims_dedup_equivalent(FULL, OTHER, claim_equivalence_arbiter=arbiter) is False
    assert arbiter.asked == []
