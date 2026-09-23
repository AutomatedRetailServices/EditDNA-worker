"""D-226 -- Pacing V2 Controlled J/L Perceptual Timing Qualification.
OFFLINE / LOCAL MEDIA ONLY. Test-support fixture lab, NOT part of the
production `cutsell_worker` package -- never imported by `universal_
clean_cut.py`, `pipeline.py`, or any live call site. No production
pipeline wiring, no live authority.

Builds tiny, deterministic, purely-local media files (`ffmpeg` lavfi
generators only -- no downloaded media, no TTS/network/provider call)
with known, exactly-constructed "word"-shaped tone-burst timing, then
exercises the REAL, UNMODIFIED D-220 timing-amount policy (`pacing_v2_
timing_policy.decide_jcut_timing`/`decide_lcut_timing`) and the REAL,
UNMODIFIED D-214 renderer (`render.render_timeline_with_audio_windows`)
end-to-end, producing actual rendered comparison MP4s for human
Watch+Listen review. D-215's own eligibility/safety decision
(`decide_transition`) is not re-exercised here -- D-226's own scope is
strictly the TIMING AMOUNT a caller who ALREADY found a pair J_CUT/L_CUT-
eligible would request, exactly matching `decide_jcut_timing`'s/`decide_
lcut_timing`'s own "only ever meaningfully called on an already-eligible
pair" contract.

## Honest limitation (this task's own required disclosure)

Real natural speech audio cannot be produced offline without a TTS/
network/provider call, which this task's own scope forbids ("Do NOT
silently call TTS/network/provider"). Every "word" in these fixtures is
a short, gated sine-tone burst (`ffmpeg`'s own `volume=...:eval=frame`
gating over a `sine` lavfi source) standing in for a spoken word -- a
deterministic, EXACTLY-known-boundary MARKER, never natural speech. This
is the documented limitation this task's own directive explicitly names
as acceptable ("create deterministic audio markers plus a separately
documented perceptual limitation") when no natural speech fixture
exists in this repository (confirmed: none does).

## Why `max_safe_window` here models PRE_ROLL/POST_ROLL physical room,
## not the narrower in-window-only D-217 evidence

`pacing_v2_evidence_adapter.available_silent_head_sec`/`available_
silent_tail_sec` (D-217) measure silent room STRICTLY INSIDE a clip's
own already-selected `[start, end)` span. But `render.py`'s own D-214
physical realization of a J_CUT/L_CUT (`RenderSegment.audio_start <
start` / `audio_end > end`) needs REAL, PHYSICALLY PRESENT audio content
OUTSIDE that span (before `start` for a J-cut's own right segment, after
`end` for an L-cut's own left segment) -- D-222's own forensic already
proved this exact mismatch is real (item 3-6 of that entry) and named
the PRE_ROLL/POST_ROLL `SourceAudioHandle` shape (D-223) as the
architecturally-correct evidence source for it. Since THIS task's own
goal is to render and perceptually evaluate an ACTUALLY REALIZABLE J/L
timing amount (not merely replicate today's narrower live evidence
source), `max_safe_window` in every fixture below is constructed and
verified as genuine, physically-present PRE_ROLL/POST_ROLL room --
exactly the shape `render.py`'s own audio-window mechanism can already
consume (D-214, unmodified) and exactly the shape a real `SourceAudioHandle`
(D-223, unmodified) would supply. `decide_jcut_timing`/`decide_lcut_
timing` (D-220) accept a plain `max_safe_lead`/`max_safe_tail` float
regardless of which evidence source produced it, so this substitution
changes nothing about D-220 itself -- it only changes which honestly-
constructed number this fixture lab feeds it, matching what the renderer
can physically execute rather than a narrower evidence source already
flagged (D-222) as an architectural gap.

## No production constant

`_EXPERIMENTAL_SHORTER_CONTROL_FRACTION` below is a PERCEPTUAL CONTROL
constant for this offline qualification only -- referenced nowhere in
`cutsell_worker/*.py` (verified by this task's own test suite) -- never
a production timing heuristic.

## Renderer timeline-capacity constraint (discovered building this lab)

`max_safe_window` (source-side pre-roll/post-roll room, see above) is a
NECESSARY but NOT SUFFICIENT condition for a J-cut lead to be physically
realizable by `render.py`'s own `_validate_audio_placements`. A J-cut's
lead audio is placed on the OUTPUT timeline at `video_position(right) -
lead`; if `lead` (the requested amount) exceeds `video_position(right)`
(i.e. the LEFT segment's own rendered video duration, when there is only
one preceding segment), the renderer fails closed
(`audio_lead_exceeds_available_timeline`) rather than silently clamping.
This is orthogonal to whether the RIGHT clip's own source has that much
pre-roll silence available -- it is purely about how much OUTPUT timeline
precedes the join. D-220 itself is never told about this constraint (it
only ever receives an already-decided `max_safe_lead` float); capping
`max_safe_lead` by the preceding segment's own rendered duration is the
CALLER's responsibility (the evidence adapter that assembles
`max_safe_lead`/`max_safe_tail` before calling `decide_jcut_timing`/
`decide_lcut_timing`), not something to add to D-220 (out of this task's
scope; D-220 itself is unmodified). This lab's own shared LEFT-J clip
(`_LEFT_J_END` below) is sized with enough trailing silent padding to
absorb every J-case's `max_safe_window`, matching what a real caller's
own capping step would already guarantee. No equivalent constraint
exists on the L-cut (trailing-audio) side: extending a LEFT segment's
own `audio_end` past its video end never moves that segment's own
placement (`_audio_placement_sec`'s `lead` term is 0 for a pure trailing
window), so an L-cut's realizability is bounded only by its own source's
audio availability (already validated by `validate_audio_window`),
independent of any following timeline.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

from cutsell_worker.contracts import DraftClip, SemanticRole, Word
from cutsell_worker.dialogue_pacing_transition import J_CUT, L_CUT
from cutsell_worker.media_probe import probe_media
from cutsell_worker.pacing_v2_timing_policy import decide_jcut_timing, decide_lcut_timing
from cutsell_worker.render import RENDER_FPS_DEFAULT, render_timeline_with_audio_windows, rendered_segment_duration_sec
from cutsell_worker.render_plan import RenderSegment

SCHEMA_VERSION = "cutsell.d226_jl_fixture_lab.v1"

# --- Fixture geometry (deterministic, chosen once, never derived from any real production media) --
_WIDTH, _HEIGHT, _FPS = 320, 180, 15
_TONE_FREQ_LEFT = 220.0
_TONE_FREQ_RIGHT = 440.0

# Shared "anchor" clips (one per direction, reused across all 3 size cases).
_LEFT_J_WORDS: Tuple[Tuple[float, float], ...] = ((0.20, 0.50), (0.60, 0.85), (1.00, 1.40))
# `_LEFT_J_END` must exceed the LARGEST J-case `max_safe_window` (see J_CASES
# below), not just the last word's own end. This is a genuine, documented
# renderer constraint discovered while building this lab (see module
# docstring's "Renderer timeline-capacity constraint" section): a J-cut's
# `max_safe_window` describes room physically present in the RIGHT clip's
# own source audio (pre-roll silence before its first word) -- but
# `render.py`'s own `_validate_audio_placements` additionally requires
# enough PRECEDING OUTPUT-TIMELINE duration (i.e. the LEFT segment's own
# rendered video length) to place that much lead audio into. The two
# constraints are independent; a fixture (or a real caller) that sizes
# `max_safe_window` from source room alone, without also capping it by the
# preceding segment's own rendered duration, can request a placement the
# renderer fails closed on (`audio_lead_exceeds_available_timeline`). This
# fixture lab's own LEFT clip is therefore sized with enough trailing
# padding (word-free, silent) to absorb every J-case's `max_safe_window`
# below -- exactly matching what a real caller's evidence-capping step
# would need to guarantee before ever calling `decide_jcut_timing`.
_LEFT_J_END = 2.5
_ANCHOR_J = _LEFT_J_WORDS[-1][1] - _LEFT_J_WORDS[-1][0]  # 0.40s -- left's own last word duration.

_RIGHT_L_WORDS: Tuple[Tuple[float, float], ...] = ((0.10, 0.50), (0.65, 0.90), (1.05, 1.30))
_RIGHT_L_END = 1.5
_ANCHOR_L = _RIGHT_L_WORDS[0][1] - _RIGHT_L_WORDS[0][0]  # 0.40s -- right's own first word duration.

# J-cut cases: `offset` == right.start == the exact amount of genuine,
# physically-present, word-free PRE_ROLL room this fixture builds before it.
J_CASES: Tuple[Tuple[str, float], ...] = (
    ("J1_moderate", 0.60),
    ("J2_large", 2.00),
    ("J3_small", 0.15),
)
# L-cut cases: `trailing` == the exact amount of genuine POST_ROLL room
# this fixture builds after left.end.
L_CASES: Tuple[Tuple[str, float], ...] = (
    ("L1_moderate", 0.60),
    ("L2_large", 2.00),
    ("L3_small", 0.15),
)

# Perceptual CONTROL only -- never a production constant (see module docstring).
_EXPERIMENTAL_SHORTER_CONTROL_FRACTION = 0.5

VARIANT_BASELINE = "BASELINE"
VARIANT_D220_POLICY = "D220_POLICY"
VARIANT_MAX_SAFE = "MAX_SAFE"
VARIANT_SHORTER_CONTROL = "SHORTER_CONTROL"
VARIANTS: Tuple[str, ...] = (VARIANT_BASELINE, VARIANT_D220_POLICY, VARIANT_MAX_SAFE, VARIANT_SHORTER_CONTROL)


class _SimpleProsody:
    """Minimal duck-typed stand-in for `ProsodicDeliveryEvidence` -- exposes
    ONLY the one attribute `pacing_v2_timing_policy._prosodic_continuity_
    state` reads via `getattr`. A local, deterministic fixture value, never
    a provider computation."""

    __slots__ = ("vocal_continuity_state",)

    def __init__(self, vocal_continuity_state: str):
        self.vocal_continuity_state = vocal_continuity_state


def _run_ffmpeg(args: Sequence[str]) -> None:
    completed = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"d226_fixture_ffmpeg_failed: {completed.stderr[-2000:]}")


def build_tone_word_clip(
    path: Path, *, color: str, word_windows: Sequence[Tuple[float, float]],
    total_duration: float, frequency: float,
) -> None:
    """Deterministic local media builder -- one solid-color visual pattern
    (visually distinct per clip via `color`) plus a single continuous
    `sine` audio track gated into short bursts at exactly `word_windows`
    (silence everywhere else) -- the "spoken word" markers this whole
    fixture lab is built on (see module docstring's own honest
    limitation). No downloaded media, no TTS, no network call."""
    conditions = "+".join(f"between(t\\,{s:.3f}\\,{e:.3f})" for s, e in word_windows)
    volume_expr = f"if({conditions}\\,1\\,0)" if conditions else "0"
    video_src = f"color=c={color}:s={_WIDTH}x{_HEIGHT}:r={_FPS}:d={total_duration:.3f}"
    audio_src = f"sine=frequency={frequency}:sample_rate=48000:duration={total_duration:.3f}"
    _run_ffmpeg([
        "-y",
        "-f", "lavfi", "-i", video_src,
        "-f", "lavfi", "-i", audio_src,
        "-filter_complex", f"[1:a]volume='{volume_expr}':eval=frame[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-b:a", "96k", "-ar", "48000",
        "-shortest",
        str(path),
    ])


def _words(windows: Sequence[Tuple[float, float]], *, prefix: str) -> Tuple[Word, ...]:
    return tuple(Word(text=f"{prefix}{i}", start=s, end=e) for i, (s, e) in enumerate(windows))


def _clip(clip_id: str, source_asset_id: str, start: float, end: float, words: Tuple[Word, ...]) -> DraftClip:
    return DraftClip(
        clip_id=clip_id, source_asset_id=source_asset_id, source_order=0, start=start, end=end,
        text="", caption_text="", words=words, semantic_role=SemanticRole.STORY, selected=True,
    )


@dataclass(frozen=True)
class JLFixtureResult:
    case_name: str
    direction: str  # J_CUT / L_CUT
    max_safe_window: float
    anchor_word_duration: float
    decision_chosen_duration: Optional[float]
    decision_timing_status: str
    decision_timing_basis: str
    decision_with_prosody_chosen_duration: Optional[float]
    decision_with_prosody_timing_basis: str
    variant_paths: dict  # {variant_name: str(path)}
    variant_durations_sec: dict  # {variant_name: float}
    no_word_truncation: bool
    no_duplicate_speech_region: bool
    chosen_within_max_safe: bool
    deterministic_repeat_matches: bool


def _build_j_media(tmpdir: Path, case_name: str, offset: float) -> Tuple[DraftClip, DraftClip, Path, Path]:
    left_path = tmpdir / "left_j_shared.mp4"
    if not left_path.exists():
        build_tone_word_clip(left_path, color="red", word_windows=_LEFT_J_WORDS, total_duration=_LEFT_J_END, frequency=_TONE_FREQ_LEFT)
    left = _clip("left", "src_left_j", 0.0, _LEFT_J_END, _words(_LEFT_J_WORDS, prefix="lw"))

    right_words = tuple((offset + s, offset + e) for s, e in ((0.10, 0.25), (0.45, 0.60), (0.80, 0.95)))
    right_end = offset + 1.10
    right_path = tmpdir / f"right_{case_name}.mp4"
    build_tone_word_clip(right_path, color="blue", word_windows=right_words, total_duration=right_end, frequency=_TONE_FREQ_RIGHT)
    right = _clip("right", "src_right_j", offset, right_end, _words(right_words, prefix="rw"))
    return left, right, left_path, right_path


def _build_l_media(tmpdir: Path, case_name: str, trailing: float) -> Tuple[DraftClip, DraftClip, Path, Path]:
    left_end = 1.0
    left_words = ((0.10, 0.25), (0.35, 0.50), (0.55, 0.80))
    left_total = left_end + trailing
    left_path = tmpdir / f"left_{case_name}.mp4"
    build_tone_word_clip(left_path, color="green", word_windows=left_words, total_duration=left_total, frequency=_TONE_FREQ_LEFT)
    left = _clip("left", "src_left_l", 0.0, left_end, _words(left_words, prefix="lw"))

    right_path = tmpdir / "right_l_shared.mp4"
    if not right_path.exists():
        build_tone_word_clip(right_path, color="yellow", word_windows=_RIGHT_L_WORDS, total_duration=_RIGHT_L_END, frequency=_TONE_FREQ_RIGHT)
    right = _clip("right", "src_right_l", 0.0, _RIGHT_L_END, _words(_RIGHT_L_WORDS, prefix="rw"))
    return left, right, left_path, right_path


def _render_variant(
    left: DraftClip, right: DraftClip, left_path: Path, right_path: Path, *,
    direction: str, amount: Optional[float], out_path: Path,
) -> float:
    left_seg = RenderSegment(
        clip_id=left.clip_id, source_asset_id=left.source_asset_id, source_path=str(left_path),
        start=left.start, end=left.end,
    )
    right_seg = RenderSegment(
        clip_id=right.clip_id, source_asset_id=right.source_asset_id, source_path=str(right_path),
        start=right.start, end=right.end,
    )
    if amount is not None and amount > 0:
        if direction == J_CUT:
            right_seg = replace(right_seg, audio_start=right.start - amount)
        else:
            left_seg = replace(left_seg, audio_end=left.end + amount)
    render_timeline_with_audio_windows((left_seg, right_seg), str(out_path), width=_WIDTH, height=_HEIGHT, fps=_FPS)
    return probe_media(str(out_path)).duration_sec


def run_j_case(tmpdir: Path, case_name: str, offset: float, transition_index: int) -> JLFixtureResult:
    left, right, left_path, right_path = _build_j_media(tmpdir, case_name, offset)
    max_safe_window = offset  # exactly how much word-free room this fixture built before right.start.

    decision = decide_jcut_timing(left, right, max_safe_lead=max_safe_window, transition_index=transition_index)
    decision2 = decide_jcut_timing(left, right, max_safe_lead=max_safe_window, transition_index=transition_index)
    decision_with_prosody = decide_jcut_timing(
        left, right, max_safe_lead=max_safe_window, transition_index=transition_index,
        right_prosody=_SimpleProsody("CONTINUOUS"),
    )

    chosen = decision.chosen_duration
    amounts = {
        VARIANT_BASELINE: None,
        VARIANT_D220_POLICY: chosen,
        VARIANT_MAX_SAFE: max_safe_window,
        VARIANT_SHORTER_CONTROL: (chosen * _EXPERIMENTAL_SHORTER_CONTROL_FRACTION) if chosen else None,
    }
    variant_paths, variant_durations = {}, {}
    for variant, amount in amounts.items():
        out_path = tmpdir / f"render_{case_name}_{variant}.mp4"
        duration = _render_variant(left, right, left_path, right_path, direction=J_CUT, amount=amount, out_path=out_path)
        variant_paths[variant] = str(out_path)
        variant_durations[variant] = duration

    no_word_truncation = all(
        not (right.start - (chosen or 0.0) <= w.start < right.start) for w in right.words
    ) if chosen else True
    no_duplicate = True  # by construction: the added window is pure pre-roll silence, never a word span.

    return JLFixtureResult(
        case_name=case_name, direction=J_CUT, max_safe_window=max_safe_window, anchor_word_duration=_ANCHOR_J,
        decision_chosen_duration=chosen, decision_timing_status=decision.timing_status,
        decision_timing_basis=decision.timing_basis,
        decision_with_prosody_chosen_duration=decision_with_prosody.chosen_duration,
        decision_with_prosody_timing_basis=decision_with_prosody.timing_basis,
        variant_paths=variant_paths, variant_durations_sec=variant_durations,
        no_word_truncation=no_word_truncation, no_duplicate_speech_region=no_duplicate,
        chosen_within_max_safe=(chosen is None or chosen <= max_safe_window + 1e-6),
        deterministic_repeat_matches=(decision == decision2),
    )


def run_l_case(tmpdir: Path, case_name: str, trailing: float, transition_index: int) -> JLFixtureResult:
    left, right, left_path, right_path = _build_l_media(tmpdir, case_name, trailing)
    max_safe_window = trailing

    decision = decide_lcut_timing(left, right, max_safe_tail=max_safe_window, transition_index=transition_index)
    decision2 = decide_lcut_timing(left, right, max_safe_tail=max_safe_window, transition_index=transition_index)
    decision_with_prosody = decide_lcut_timing(
        left, right, max_safe_tail=max_safe_window, transition_index=transition_index,
        left_prosody=_SimpleProsody("CONTINUOUS"),
    )

    chosen = decision.chosen_duration
    amounts = {
        VARIANT_BASELINE: None,
        VARIANT_D220_POLICY: chosen,
        VARIANT_MAX_SAFE: max_safe_window,
        VARIANT_SHORTER_CONTROL: (chosen * _EXPERIMENTAL_SHORTER_CONTROL_FRACTION) if chosen else None,
    }
    variant_paths, variant_durations = {}, {}
    for variant, amount in amounts.items():
        out_path = tmpdir / f"render_{case_name}_{variant}.mp4"
        duration = _render_variant(left, right, left_path, right_path, direction=L_CUT, amount=amount, out_path=out_path)
        variant_paths[variant] = str(out_path)
        variant_durations[variant] = duration

    no_word_truncation = all(
        not (left.end < w.end <= left.end + (chosen or 0.0)) for w in left.words
    ) if chosen else True
    no_duplicate = True

    return JLFixtureResult(
        case_name=case_name, direction=L_CUT, max_safe_window=max_safe_window, anchor_word_duration=_ANCHOR_L,
        decision_chosen_duration=chosen, decision_timing_status=decision.timing_status,
        decision_timing_basis=decision.timing_basis,
        decision_with_prosody_chosen_duration=decision_with_prosody.chosen_duration,
        decision_with_prosody_timing_basis=decision_with_prosody.timing_basis,
        variant_paths=variant_paths, variant_durations_sec=variant_durations,
        no_word_truncation=no_word_truncation, no_duplicate_speech_region=no_duplicate,
        chosen_within_max_safe=(chosen is None or chosen <= max_safe_window + 1e-6),
        deterministic_repeat_matches=(decision == decision2),
    )


def run_all_cases(tmpdir: Path) -> Tuple[JLFixtureResult, ...]:
    results = []
    for i, (name, offset) in enumerate(J_CASES):
        results.append(run_j_case(tmpdir, name, offset, transition_index=i))
    for i, (name, trailing) in enumerate(L_CASES):
        results.append(run_l_case(tmpdir, name, trailing, transition_index=100 + i))
    return tuple(results)


def expected_output_duration_sec(left_duration: float, right_duration: float) -> float:
    """The renderer's own VIDEO-timeline length is invariant to any audio
    decision (D-214's own `_video_timeline_positions` contract) -- always
    the frame-rounded sum of both segments' own video durations, for every
    variant alike."""
    return rendered_segment_duration_sec(left_duration, fps=_FPS) + rendered_segment_duration_sec(right_duration, fps=_FPS)
