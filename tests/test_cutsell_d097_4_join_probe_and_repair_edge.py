"""D-097.4 -- the technical post-render QC must judge the JOIN, not the
speech next to it; a trailing repair must trim the edge the output really
carries.

RAW 34034507983 (head 19cec48) was the first Video00 run where Selection
Freeze passed AND the D-097.2 single-pass renderer placed every join at the
frame-exact position `segment_output_windows` predicts (verified from the
run's own repair records: joins at 21.133 / 29.733 / 48.633 ... s are the
cumulative frame-rounded segment durations to the millisecond). The
technical QC still reported 9 of 21 joins as ABRUPT_AUDIO_DISCONTINUITY
(peak 579-707 vs "typical" 26-30) and the bounded repair loop burnt its
three attempts on them: no deliverable MP4, NEEDS_HUMAN_REVIEW.

Two root causes, both reproduced here with REAL rendered audio:

1. `check_audio_discontinuity_at_boundaries` compared the largest
   sample-to-sample jump anywhere in a +/-80 ms window against that
   window's median. A Boundary cut lands tightly on a word onset, so the
   window always contains a speech transient tens of milliseconds from
   the join, while the 12 ms join fades keep the join itself near-silent
   and drag the median down. The fixed probe looks for ONE isolated jump
   within a few milliseconds of the join whose immediate neighbours on
   both sides are ordinary -- the real signature of a splice click -- and
   ignores speech transients elsewhere in the window.

2. `repair_segment_for_finding` trimmed a trailing edge from the PLAN's
   segment end. The renderer had already removed a 0.286 s silent tail
   from that segment; after the 50 ms "repair" the remaining silent tail
   (0.236 s) fell under the tightener's 0.28 s minimum, the tightening
   vanished, and the rendered segment came out 0.236 s LONGER, moving
   every later join by +0.233 s. A trailing repair now trims from the
   renderer-tightened edge, so a repair can only ever shorten the output.

The synthetic voice below is deterministic (fixed seed): a glottal pulse
train with formant-shaped harmonics, syllabic envelopes, plosive bursts at
syllable onsets, and a room-noise floor -- the jump statistics of real
speech (bursts of consecutive large jumps at onsets, quiet floors between
words) without depending on a TTS engine being present in ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from cutsell_worker import render
from cutsell_worker.live_boundary_repair import repair_segment_for_finding, segment_output_windows
from cutsell_worker.post_render_media_qc import ABRUPT_AUDIO_DISCONTINUITY, _extract_pcm_window, check_audio_discontinuity_at_boundaries
from cutsell_worker.post_render_watch_listen_qc import PostRenderFinding
from cutsell_worker.render_plan import RenderSegment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on this runner")

SR = 22_050


def _ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], check=True)


def synthetic_voice(seconds: float, *, seed: int = 7, sample_rate: int = SR) -> np.ndarray:
    """Speech-shaped test signal as int16 mono PCM."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 118.0 + 12.0 * np.sin(2 * np.pi * 0.7 * t)  # slow pitch drift
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    voiced = np.zeros(n)
    for k in range(1, 24):  # harmonics with two formant peaks (~600 Hz and ~1.8 kHz)
        freq = k * 118.0
        formant = np.exp(-((freq - 600.0) / 250.0) ** 2) + 0.6 * np.exp(-((freq - 1800.0) / 400.0) ** 2)
        voiced += (formant / k ** 0.5) * np.sin(k * phase)
    voiced /= np.max(np.abs(voiced))
    # syllables: 4.5 per second, 55 % voiced duty, smooth attack/decay
    syllable = 4.5
    env = np.clip(np.sin(2 * np.pi * syllable * t) * 1.6 - 0.3, 0.0, 1.0)
    # plosive bursts (10 ms white noise) at every syllable onset
    burst = np.zeros(n)
    onsets = np.flatnonzero(np.diff((env > 0.0).astype(int)) == 1)
    for onset in onsets:
        length = int(0.010 * sample_rate)
        burst[onset:onset + length] += rng.standard_normal(length) * np.linspace(1.0, 0.0, length)
    # short silent phrase breaks every ~1.3 s
    phrase = (np.mod(t, 1.3) > 0.25).astype(float)
    signal = (0.55 * voiced * env + 0.35 * burst) * phrase
    signal += 0.0015 * rng.standard_normal(n)  # room floor
    return np.clip(signal * 32767.0, -32767, 32767).astype(np.int16)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int = SR) -> str:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())
    return str(path)


@pytest.fixture(scope="module")
def voice_source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d097_4_voice")
    pcm = synthetic_voice(14.0)
    wav = _write_wav(directory / "voice.wav", pcm)
    mp4 = str(directory / "voice.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
        "-i", wav, "-t", "14", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-shortest", mp4,
    ])
    return directory, mp4, pcm


def _window_wide_ratio(path: str, timestamp: float, window_sec: float = 0.08) -> float:
    """The D-030..D-097.3 formulation (max jump anywhere in the window over
    the window-wide median) -- kept here only to prove the fixture
    reproduces the RAW's false positive, never as a production check."""
    pcm = _extract_pcm_window(path, center_sec=timestamp, window_sec=window_sec, sample_rate=SR)
    deltas = np.abs(np.diff(pcm.astype(np.int64)))
    return float(np.max(deltas)) / (float(np.median(deltas)) + 1.0)


def _speech_cuts(pcm: np.ndarray) -> list[tuple[float, float]]:
    """Segment windows that start and end INSIDE speech (tight Boundary cuts)."""
    env = np.abs(pcm.astype(np.float64))
    block = SR // 100  # 10 ms
    loud = np.array([env[i:i + block].mean() > 1500 for i in range(0, env.size - block, block)])
    loud_times = [i * block / SR for i, flag in enumerate(loud) if flag]
    anchors = [1.0, 3.6, 6.3, 9.0, 11.7]
    cuts = []
    for a, b in zip(anchors, anchors[1:]):
        start = min(loud_times, key=lambda x: abs(x - a))
        end = min(loud_times, key=lambda x: abs(x - (b - 0.35)))
        cuts.append((start, end))
    return cuts


# ------------------------------------------------------------ 1. the probe


def test_clean_faded_joins_of_tightly_cut_speech_pass_the_join_probe(voice_source):
    directory, mp4, pcm = voice_source
    cuts = _speech_cuts(pcm)
    segments = tuple(
        RenderSegment(clip_id=f"s{i}", source_asset_id="src", source_path=mp4, start=a, end=b)
        for i, (a, b) in enumerate(cuts)
    )
    out = str(directory / "clean.mp4")
    render.render_preview(list(segments), out)
    joins = [w[1] for w in segment_output_windows(segments)[:-1]]

    # The fixture reproduces the RAW's failure mode under the old formula...
    old_flags = [_window_wide_ratio(out, j) >= 6.0 for j in joins]
    assert any(old_flags), "fixture no longer contains a speech transient near a join"
    # ...and the join-instant probe judges the real joins clean.
    result = check_audio_discontinuity_at_boundaries(out, joins)
    assert result.status == "PASS", [f.detail for f in result.findings]


def test_a_real_hard_splice_in_speech_is_still_reported_at_the_join(voice_source):
    directory, _mp4, pcm = voice_source
    x = pcm.astype(np.int64)
    a = x[int(0.4 * SR):int(2.4 * SR)]
    b = x[int(3.2 * SR):int(5.2 * SR)]
    ia = int(np.argmax(a[-2000:])) + a.size - 2000
    ib = int(np.argmin(b[:2000]))
    step = int(a[ia - 1] - b[ib])
    assert step > 5000, step
    bad = np.concatenate([a[:ia], b[ib:]]).astype(np.int16)
    wav = _write_wav(directory / "bad_join.wav", bad)
    join = ia / SR
    result = check_audio_discontinuity_at_boundaries(wav, [join])
    assert result.status == "FAIL"
    finding = result.findings[0]
    assert finding.kind == ABRUPT_AUDIO_DISCONTINUITY
    assert finding.routes_to == "BoundaryEngine"
    assert abs(finding.detail["offset_ms"]) <= 1.0
    assert finding.detail["ratio"] >= 6.0

    # ...and it survives the delivery codec.
    m4a = str(directory / "bad_join.m4a")
    _ffmpeg(["-y", "-i", wav, "-c:a", "aac", "-b:a", "160k", "-ar", "48000", m4a])
    assert check_audio_discontinuity_at_boundaries(m4a, [join]).status == "FAIL"


def test_a_speech_onset_near_but_not_at_the_join_is_not_a_join_defect(voice_source):
    directory, _mp4, pcm = voice_source
    # 300 ms of room floor, then an abrupt (unfaded) voice onset 40 ms after
    # the probed join: a transient inside the old +/-80 ms window, but not
    # at the join instant.
    rng = np.random.default_rng(3)
    floor = (rng.standard_normal(int(0.3 * SR)) * 25).astype(np.int64)
    x = pcm.astype(np.int64)
    block = SR // 10
    loudest = max(range(0, x.size - SR, block), key=lambda i: float(np.abs(x[i:i + SR]).mean()))
    voice = x[loudest:loudest + SR]
    signal = np.concatenate([floor, voice]).astype(np.int16)
    wav = _write_wav(directory / "onset.wav", signal)
    onset = 0.3
    assert _window_wide_ratio(wav, onset - 0.04) >= 6.0  # the old formula flagged this shape
    assert check_audio_discontinuity_at_boundaries(wav, [onset - 0.04]).status == "PASS"


def test_probe_skips_a_boundary_without_enough_samples(voice_source):
    directory, mp4, _pcm = voice_source
    assert check_audio_discontinuity_at_boundaries(mp4, [0.0]).status == "PASS"


# ----------------------------------------------- 2. repair from the tightened edge


@pytest.fixture(scope="module")
def silent_tail_source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("d097_4_tail")
    voice = synthetic_voice(2.0, seed=11)
    silence = np.zeros(int(0.8 * SR), dtype=np.int16)
    pcm = np.concatenate([voice, silence])
    wav = _write_wav(directory / "tail.wav", pcm)
    mp4 = str(directory / "tail.mp4")
    _ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
        "-i", wav, "-t", "2.8", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-shortest", mp4,
    ])
    return directory, mp4


def test_trailing_repair_trims_from_the_renderer_tightened_edge(silent_tail_source):
    _directory, mp4 = silent_tail_source
    seg = RenderSegment(clip_id="c", source_asset_id="src", source_path=mp4, start=0.0, end=2.8)
    tightened = render.tighten_trailing_silence(seg)
    assert tightened.end < seg.end - 0.5, "fixture must carry a tightenable silent tail"
    segments = (seg,)
    before = segment_output_windows(segments)
    join = before[0][1]
    finding = PostRenderFinding(kind=ABRUPT_AUDIO_DISCONTINUITY, start=join, end=join, detail={}, routes_to="BoundaryEngine")

    repaired = repair_segment_for_finding(segments, finding)
    assert repaired is not None
    new_segments, attempt = repaired
    assert attempt.edge == "trailing"
    assert attempt.tightened_end == pytest.approx(tightened.end)
    assert attempt.repaired_end == pytest.approx(tightened.end - attempt.trim_sec)
    assert attempt.original_end == pytest.approx(2.8)

    after = segment_output_windows(new_segments)
    # A repair only ever SHORTENS the output (RAW 34034507983: +0.233 s).
    assert after[0][1] <= before[0][1]
    assert after[0][1] == pytest.approx(before[0][1] - attempt.trim_sec, abs=1 / 30)


def test_leading_repair_is_unchanged_by_the_tightened_edge(silent_tail_source):
    _directory, mp4 = silent_tail_source
    seg = RenderSegment(clip_id="c", source_asset_id="src", source_path=mp4, start=0.0, end=2.8)
    finding = PostRenderFinding(kind=ABRUPT_AUDIO_DISCONTINUITY, start=0.0, end=0.0, detail={}, routes_to="BoundaryEngine")
    repaired = repair_segment_for_finding((seg,), finding)
    assert repaired is not None
    _new, attempt = repaired
    assert attempt.edge == "leading"
    assert attempt.tightened_end is None
    assert attempt.repaired_start == pytest.approx(attempt.trim_sec)
