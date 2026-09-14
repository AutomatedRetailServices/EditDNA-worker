"""General guardrails for complete retries and speech-safe post-take visual slack.

This module is intentionally small and installed at package bootstrap so it can harden
existing cleanup layers without hardcoding any benchmark timestamps.
"""
from __future__ import annotations

from typing import Any


def _install_complete_idea_alternate_guard() -> None:
    from . import hybrid_retry_completion_integrity as integrity

    original = integrity._safe_short_alternate_debris
    if getattr(original, "_cutsell_complete_idea_guard", False):
        return

    def protected(take, previous, following, semantic):
        # A semantically complete delivery is never recording debris merely because it
        # is short or because neighboring clips happen to cover similar vocabulary.
        # Keep it alive for downstream retry grouping / Best Take authority.
        if bool(getattr(take, "complete_idea", False)):
            return False
        return original(take, previous, following, semantic)

    protected._cutsell_complete_idea_guard = True
    integrity._safe_short_alternate_debris = protected


def _install_long_speech_safe_visual_slack_detector() -> None:
    from . import speech_visual_microtrim as microtrim

    original = microtrim.detect_speech_safe_visual_microtrims
    if getattr(original, "_cutsell_long_visual_slack_v2", False):
        return

    def detect_v2(
        path: str,
        *,
        asr_model: str = "medium",
        language_hint: str | None = None,
        max_total_trim_sec: float = 2.0,
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        """Detect speech-safe reset slack, including clearly non-narrative longer gaps.

        Gaps up to 1.25s are eligible, but gaps beyond the original 0.62s window use a
        stricter acoustic threshold and still require persistent visual-reset evidence.
        Speech envelopes remain absolute authority on both sides of every cut.
        """
        transcript = microtrim.FasterWhisperASR(model_name=asr_model).transcribe(
            path,
            source_asset_id="rendered-output",
            language_hint=language_hint,
        )
        words = sorted(
            [word for segment in transcript for word in segment.words],
            key=lambda word: (float(word.start), float(word.end)),
        )
        silences = microtrim._silences(path)
        if len(words) < 2 or not silences:
            return (), {
                "speech_lock_ok": True,
                "candidate_count": 0,
                "reason": "insufficient_evidence",
                "detector_version": "v2_long_visual_slack",
            }

        cuts: list[dict[str, Any]] = []
        candidate_diagnostics: list[dict[str, Any]] = []
        candidates = 0
        long_candidates = 0
        total_trim = 0.0

        for left, right in zip(words, words[1:]):
            left_end = float(left.end)
            right_start = float(right.start)
            raw_gap = right_start - left_end
            if raw_gap < 0.12 or raw_gap > 1.25:
                continue

            long_gap = raw_gap > 0.62
            safe_start = left_end + 0.035
            safe_end = right_start - 0.045
            if safe_end - safe_start < 0.075:
                continue

            quiet_ratio = microtrim._quiet_ratio(silences, safe_start, safe_end)
            quiet_threshold = 0.86 if long_gap else 0.72
            if quiet_ratio < quiet_threshold:
                continue

            candidates += 1
            if long_gap:
                long_candidates += 1

            onset, visual = microtrim._visual_reset_onset(path, left_end, safe_start, safe_end)
            diagnostic = {
                "left_word": str(left.text),
                "right_word": str(right.text),
                "left_word_end": round(left_end, 3),
                "right_word_start": round(right_start, 3),
                "raw_gap_sec": round(raw_gap, 3),
                "quiet_ratio": round(quiet_ratio, 3),
                "long_gap": long_gap,
                "visual_reason": str((visual or {}).get("reason", "")),
            }
            if onset is None:
                diagnostic["decision"] = "keep_no_persistent_visual_reset"
                if len(candidate_diagnostics) < 24:
                    candidate_diagnostics.append(diagnostic)
                continue

            cut_start = max(safe_start, float(onset))
            cut_end = safe_end
            cut_duration = cut_end - cut_start
            max_cut = 0.85 if long_gap else 0.42
            if cut_duration < 0.075 or cut_duration > max_cut:
                diagnostic["decision"] = "keep_cut_duration_outside_safe_window"
                diagnostic["proposed_cut_duration_sec"] = round(cut_duration, 3)
                if len(candidate_diagnostics) < 24:
                    candidate_diagnostics.append(diagnostic)
                continue

            if total_trim + cut_duration > max_total_trim_sec:
                diagnostic["decision"] = "keep_total_trim_budget"
                if len(candidate_diagnostics) < 24:
                    candidate_diagnostics.append(diagnostic)
                break

            if cut_start <= left_end + 0.02 or cut_end >= right_start - 0.02:
                diagnostic["decision"] = "keep_speech_guard"
                if len(candidate_diagnostics) < 24:
                    candidate_diagnostics.append(diagnostic)
                continue

            cuts.append({
                "start": round(cut_start, 3),
                "end": round(cut_end, 3),
                "duration_sec": round(cut_duration, 3),
                "reason": (
                    "auto_speech_safe_long_post_take_visual_slack"
                    if long_gap
                    else "auto_speech_safe_visual_reset_microtrim"
                ),
                "left_word": str(left.text),
                "right_word": str(right.text),
                "left_word_end": round(left_end, 3),
                "right_word_start": round(right_start, 3),
                "quiet_ratio": round(quiet_ratio, 3),
                "visual_evidence": visual,
            })
            total_trim += cut_duration
            diagnostic["decision"] = "trim"
            diagnostic["trim_duration_sec"] = round(cut_duration, 3)
            if len(candidate_diagnostics) < 24:
                candidate_diagnostics.append(diagnostic)

        return tuple(cuts), {
            "speech_lock_ok": True,
            "word_count": len(words),
            "candidate_count": candidates,
            "long_candidate_count": long_candidates,
            "auto_microtrim_count": len(cuts),
            "auto_microtrim_duration_sec": round(total_trim, 3),
            "frame_aware": True,
            "visual_channels": ["face_head", "pose_wrist_gesture"],
            "detector_version": "v2_long_visual_slack",
            "candidate_diagnostics": candidate_diagnostics,
            "rule": (
                "word_end_plus_acoustic_guard_then_persistent_visual_reset_"
                "with_stricter_long_gap_silence_until_next_word_guard"
            ),
        }

    detect_v2._cutsell_long_visual_slack_v2 = True
    microtrim.detect_speech_safe_visual_microtrims = detect_v2


def install_editorial_guardrails_v2() -> None:
    _install_complete_idea_alternate_guard()
    _install_long_speech_safe_visual_slack_detector()
