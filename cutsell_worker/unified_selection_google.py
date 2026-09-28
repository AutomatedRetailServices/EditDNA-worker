Warning: truncated output (original token count: 16279)
Total output lines: 1060

"""Gemini-backed whole-video Selection reasoner for CutSell.

Unlike the legacy bounded Hybrid judge, this authority sees the complete candidate
universe for one source in a single request.  It asks the model to form idea/retry
families, recognize composites and continuations, and assign final semantic membership
before Selection freeze.  Boundary ownership remains elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import base64
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import requests

from .contracts import DraftTimeline
from .hybrid_google_transport import DollarBudgetLedger
from .hybrid_payload import estimate_tokens_from_chars
from .hybrid_provider_settings import HybridProviderSettings
from .unified_selection_reasoner import (
    UnifiedSelectionDecision,
    UnifiedSelectionPlan,
    UnifiedTakeCompetition,
)

_ACTIONS = ["select", "swap", "discard"]
_RELATIONS = [
    "independent",
    "retry_winner",
    "retry_alternate",
    "composite_piece",
    "continuation",
    "failed",
    "bts",
    "uncertain",
]
_REASON_CODES = [
    "best_complete_take",
    "independent_story_coverage",
    "composite_best_take_piece",
    "necessary_continuation",
    "usable_alternate",
    "redundant_retry",
    "failed_delivery",
    "recording_process_bts",
    "uncertain_preserve",
]


class UnifiedSelectionUnreliableResponseError(ValueError):
    """The provider response could not be trusted as one complete decision per
    candidate: truncated/malformed JSON, a missing field, or a decision count
    that does not match the candidate universe. Always raised instead of ever
    treating a partial or malformed response as an applied editorial result --
    apply_unified_selection_reasoner()'s fail-open path is the only place a
    response like this may take effect, and it discards the response entirely.
    """


class UnifiedSelectionProviderBlockedError(RuntimeError):
    """A provider explicitly declined the input; repeating it wastes budget."""


def _candidate_universe(draft: DraftTimeline) -> list[dict[str, Any]]:
    buckets: dict[str, str] = {}
    clips = {}
    for bucket_name, bucket in (
        ("discard", draft.discarded),
        ("swap", draft.alternates),
        ("select", draft.selected),
    ):
        for clip in bucket:
            clips.setdefault(clip.clip_id, clip)
            buckets[clip.clip_id] = bucket_name

    hybrid_votes: dict[str, list[dict[str, Any]]] = {}
    for chunk in (draft.diagnostics or {}).get("hybrid_editorial_chunks") or ():
        if not isinstance(chunk, Mapping):
            continue
        for row in chunk.get("decisions") or ():
            if not isinstance(row, Mapping) or not row.get("clip_id"):
                continue
            try:
                confidence = round(float(row.get("confidence") or 0.0), 3)
            except (TypeError, ValueError):
                confidence = 0.0
            hybrid_votes.setdefault(str(row["clip_id"]), []).append({
                "label": str(row.get("label") or ""),
                "confidence": confidence,
            })

    rows = []
    for clip in sorted(
        clips.values(),
        key=lambda item: (item.source_order, float(item.start), float(item.end), item.clip_id),
    ):
        row: dict[str, Any] = {
            "clip_id": clip.clip_id,
            "current_bucket": buckets.get(clip.clip_id, "swap"),
            "source_order": int(clip.source_order),
            "start": round(float(clip.start), 3),
            "end": round(float(clip.end), 3),
            "duration": round(max(0.0, float(clip.end) - float(clip.start)), 3),
            "take_group_id": clip.take_group_id,
            "text": " ".join(str(clip.text or "").split())[:1800],
            "hybrid_votes": hybrid_votes.get(clip.clip_id, [])[:6],
        }
        if (draft.diagnostics or {}).get("editorial_engine_v2_request"):
            row["source_asset_id"] = clip.source_asset_id
            if clip.audio_muted and not clip.words and not clip.text.strip():
                observed = next((item for item in
                    (draft.diagnostics or {}).get("v2_focused_visual_action_candidates", ())
                    if item.get("clip_id") == clip.clip_id), None)
                if observed:
                    row["visual_action_only"] = observed
            row["aligned_word_texts"] = [word.text for word in clip.words]
            row["aligned_words"] = [[i, word.text, round(word.start, 3), round(word.end, 3)]
                                    for i, word in enumerate(clip.words)]
        # Local face/pose/motion evidence (local_performance.py), when the
        # upstream take was analyzed. Higher visual_fumble/distraction_risk
        # and lower expression/gesture naturalness indicate a visible reset,
        # stumble, or camera-disengagement moment -- transcript text alone
        # cannot see this. Omitted entirely (not zeroed) when unavailable, so
        # the reasoner never mistakes "no evidence" for "confirmed clean".
        if clip.signals is not None:
            row["visual_evidence"] = {
                "face_visibility": round(float(clip.signals.face_visibility), 3),
                "eye_contact": round(float(clip.signals.eye_contact), 3),
                "motion_stability": round(float(clip.signals.motion_stability), 3),
                "visual_fumble": round(float(clip.signals.visual_fumble), 3),
                "expression_naturalness": round(float(clip.signals.expression_naturalness), 3),
                "gesture_naturalness": round(float(clip.signals.gesture_naturalness), 3),
                "distraction_risk": round(float(clip.signals.distraction_risk), 3),
            }
        rows.append(row)
    return rows


def _take_group_summaries(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose upstream take grouping as inspectable evidence, never authority.

    V2 previously sent ``take_group_id`` only on isolated candidate rows.  That
    made a delivery split by pauses look like unrelated clips and encouraged
    the whole-video reasoner to compare fragments instead of reconstructed
    attempts.  The summary keeps every original interval and gap visible.  It
    deliberately does not call a group a retry family or an idea: those are
    separate editorial judgments the reasoner must still make.
    """
    grouped: dict[tuple[str, tuple[str, str]], list[tuple[int, dict[str, Any]]]] = {}
    for candidate_index, row in enumerate(candidates):
        source_asset_id = str(row.get("source_asset_id") or "")
        raw_group_id = row.get("take_group_id")
        # Missing grouping evidence is always a singleton.  In particular,
        # unrelated null-ID candidates must never collapse into one mega-take.
        # The tagged tuple cannot collide with any real upstream string ID.
        group_key = (("present", str(raw_group_id)) if raw_group_id
                     else ("missing", str(candidate_index)))
        grouped.setdefault((source_asset_id, group_key), []).append((candidate_index, row))

    summaries = []
    for (source_asset_id, group_key), members in grouped.items():
        members.sort(key=lambda item: (float(item[1]["start"]), float(item[1]["end"]), item[0]))
        intervals = []
        combined_text = []
        previous_end = None
        for candidate_index, row in members:
            start = float(row["start"])
            end = float(row["end"])
            gap_before = None if previous_end is None else round(max(0.0, start - previous_end), 3)
            intervals.append({
                "candidate_index": candidate_index,
                "clip_id": row["clip_id"],
                "start": round(start, 3),
                "end": round(end, 3),
                "gap_before": gap_before,
                # Full text remains authoritative in candidates[].  A short
                # excerpt makes the grouped view readable without duplicating
                # the whole transcript and exhausting V2's input budget.
                "text_excerpt": str(row.get("text") or "")[:160],
            })
            separator = "[START]" if gap_before is None else f"[GAP {gap_before:.3f}s]"
            combined_text.append(
                f"{separator} candidate {candidate_index}: {str(row.get('text') or '')[:160]}"
            )
            previous_end = max(previous_end or end, end)
        summaries.append({
            # Source qualification keeps equal upstream IDs from looking like
            # one cross-source attempt to the model.
            "take_evidence_id": f"{source_asset_id}:{group_key[0]}:{group_key[1]}",
            "upstream_take_group_id": members[0][1].get("take_group_id"),
            "source_asset_id": source_asset_id,
            "candidate_indices": [index for index, _ in members],
            "intervals": intervals,
            "combined_text_with_gaps": " ".join(combined_text),
            "evidence_only": True,
        })
    return summaries


def _source_context(draft: DraftTimeline, *, include_audiovisual: bool = False) -> dict[str, Any]:
    raw = (draft.diagnostics or {}).get("whole_video_context") or {}
    if not isinstance(raw, Mapping):
        return {}
    sources = raw.get("sources") or []
    compact_sources = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        row = {
            "source_asset_id": source.get("source_asset_id"),
            "summary": str(source.get("summary") or "")[:3000],
            "creator_intent": str(source.get("creator_intent") or "")[:600],
            "main_topic": str(source.get("main_topic") or "")[:500],
            "story_logic": str(source.get("story_logic") or "")[:1000],
            "dominant_style": str(source.get("dominant_style") or "")[:300],
            "edit_mode": str(source.get("edit_mode") or "")[:80],
        }
        if include_audiovisual:
            evidence = str(source.get("audiovisual_evidence") or "")
            if not evidence:
                raise ValueError("Editorial Engine V2 source missing audiovisual evidence")
            row["audiovisual_evidence"] = evidence[:12000]
        compact_sources.append(row)
    return {
        "dominant_edit_mode": raw.get("dominant_edit_mode"),
        "sources": compact_sources,
    }


def build_unified_selection_payload(draft: DraftTimeline) -> dict[str, Any]:
    candidates = _candidate_universe(draft)
    if not candidates:
        raise ValueError("unified selection requires at least one candidate")
    v2_request = bool((draft.diagnostics or {}).get("editorial_engine_v2_request"))
    if v2_request:
        # Provisional local buckets and Hybrid votes are past decisions, not
        # source evidence. V19 mostly repeated their labels, keeping the
        # selected early patchwork while calling the discarded final take
        # failed. Preserve those labels in draft diagnostics for audit; do
        # not anchor the V2 whole-video authority with them.
        candidates = [
            {key: value for key, value in row.items()
             if key not in {"current_bucket", "hybrid_votes"}}
            for row in candidates
        ]
    contract = [
        "Understand the full creator message before deciding any individual take.",
        "First infer idea families and retry relationships across the entire timeline.",
        "A genuine retry family produces exactly ONE SELECT: the cleanest complete delivery.",
        "SELECT independent valid story coverage, the winning retry, necessary continuations, and every clean composite piece.",
        "DISCARD only recording-process BTS, failed/abandoned delivery, or an inferior retry with no unique audience-facing information.",
        "Use candidate visual_evidence as real performance evidence, not decoration.",
        "Do not prefer a monolithic take merely because it is longer; a clean composite may be better.",
        "When one continuous complete take covers ideas otherwise spread across several earlier fragments, prefer the continuous take when the take, together with its immediately adjacent continuation candidates, covers the same ideas with better continuity and performance. Preserve earlier fragments that add unique audience information, conditions, or useful visual actions, and reject the continuous take when audiovisual evidence shows it failed. Do not call the complete take redundant merely because its ideas recur across a patchwork of selected attempts.",
        "Before assigning actions, reconstruct whole delivery attempts: adjacent candidates without an audiovisual restart belong to one take even when a pause split the sentence. Compare complete takes as units. When one take wins, do not SELECT covered alternate hooks, paraphrases, examples, or feature wording as independent beats. Preserve an earlier piece only for a material fact, number, condition, personality moment, or useful visual action that the winning take truly lacks.",
        "Do not treat adjacent valid statements as retries merely because they share topic words.",
        "Preserve numbers, negations, names, causal claims, and genuinely new story facts.",
        "Audience-directed speech is not automatically an independent story beat: compare every later complete delivery against the UNION of earlier fragments, even when local groups or wording differ.",
        "A compact later take may be the single retry winner over several earlier fragments when it cleanly covers their combined hook, benefits, proof, and CTA; discard those earlier fragments only after verifying that no unique audience-facing fact is lost.",
        "Never leave a retry family with only retry_alternate decisions and no selected winner. Re-evaluate the full timeline and select its cleanest complete delivery unless audiovisual evidence proves the entire family unusable.",
        "Do not compress the creator's story by deleting a unique hook, claim, example, transition, or CTA merely because another selected clip shares the topic.",
        "WHEN UNCERTAIN, preserve content rather than destructively deleting it.",
    ]
    if v2_request:
        contract.extend([
            "Use complete audiovisual observations as primary behavioral/performance evidence with the aligned transcript.",
            "A clean high-confidence audiovisual audience region contradicts failed_delivery unless that candidate itself contains an observed reset/stumble or its transcript is clearly abandoned; explain the conflict through the chosen relation and reason code.",
            "Assign every candidate one unique sequence_index. Preserve chronology by default, but reorder complete valid story beats when it clearly improves comprehension, hook, demonstration, payoff, or coherence without inventing speech.",
            "Return one final KEEP/DISCARD-equivalent plan: SELECT the final story and DISCARD every non-winner; never return SWAP.",
            "A candidate with visual_action_only is source-observed product action during objectively measured silence, with no invented speech or captions. Compare its visual story contribution to the spoken takes and SELECT useful unique product demonstration even without words; DISCARD redundant, failed, or production-only action. Place selected action in source order near its spoken context and do not substitute it for speech.",
            "For a SELECT containing a clean delivery followed by a short explicit recording-process aside, optionally return trailing_recording_word_count (1..8) counted from aligned_word_texts and trailing_recording_confidence (0..1) for that suffix classification independently of whole-take selection confidence. Return 0 or omit when uncertain, words are unavailable, or the ending is audience content. Never trim a disclaimer, offer, qualification, gratitude, humor, reaction, number, negation or product fact. Do not discard th…8279 tokens truncated…             json={"contents": contents, "generationConfig": {"temperature": 0,
                            "responseMimeType": "application/json", "responseJsonSchema": schema,
                            "maxOutputTokens": 650}}, timeout=self.timeout_sec)
                    response.raise_for_status()
                    raw = response.json()
                    if raw["candidates"][0].get("finishReason") != "STOP":
                        continue
                    parts = raw["candidates"][0]["content"]["parts"]
                    observation = json.loads("".join(p.get("text", "") for p in parts))
                    usage = raw.get("usageMetadata") or {}
                    actual = self.settings.estimate_cost_usd(
                        input_tokens=int(usage.get("promptTokenCount") or tokens),
                        output_tokens=int(usage.get("candidatesTokenCount") or 650), escalation=False)
                    if actual < estimated:
                        self.ledger.release(estimated - actual)
                    confirmed = (observation.get("linked") is True and
                            observation.get("restart_observed") is False and
                            observation.get("uncertainty") == "low" and
                            len(str(observation.get("audio_evidence") or "")) > 15 and
                            len(str(observation.get("visual_evidence") or "")) > 15)
                    evidence.append({"source_asset_id": left.source_asset_id,
                        "left_clip_id": left.clip_id, "right_clip_id": right.clip_id,
                        "source_window": [round(start, 3), round(start + length, 3)],
                        "linked": confirmed, "restart_observed": observation.get("restart_observed"),
                        "uncertainty": str(observation.get("uncertainty"))[:24],
                        "audio_evidence": str(observation.get("audio_evidence"))[:280],
                        "visual_evidence": str(observation.get("visual_evidence"))[:280],
                        "usage": {k: usage.get(k) for k in
                                  ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")},
                        "estimated_cost_usd": round(estimated, 7),
                        "actual_cost_usd": round(actual, 7)})
                    if confirmed:
                        links.append((left.clip_id, right.clip_id))
                except Exception:
                    # A generate request may have reached the provider before a
                    # timeout or parse failure. Keep its reservation in that case.
                    if not generation_started:
                        self.ledger.release(estimated)
                    raise
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError,
                    OSError, RuntimeError, subprocess.SubprocessError) as exc:
                evidence.append({"source_asset_id": left.source_asset_id,
                    "left_clip_id": left.clip_id, "right_clip_id": right.clip_id,
                    "status": "optional_probe_failed", "error_type": type(exc).__name__})
                continue
        return tuple(links), tuple(evidence)

    def _call_once(
        self,
        payload: Mapping[str, Any],
        candidate_rows: list[dict[str, Any]],
        *,
        output_tokens_requested: int,
    ) -> tuple[list[UnifiedSelectionDecision], tuple[UnifiedTakeCompetition, ...], int]:
        body = build_unified_selection_request(payload, max_output_tokens=output_tokens_requested)
        body["contents"][0]["parts"].extend(self.audiovisual_parts)
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        response = self.session.post(
            endpoint,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, Mapping):
            raise UnifiedSelectionUnreliableResponseError("Gemini unified HTTP response must be an object")
        raw_decisions, output_tokens, finish_reason = parse_unified_selection_response(raw)
        if len(raw_decisions) != len(candidate_rows):
            raise UnifiedSelectionUnreliableResponseError(
                "unified Selection ordered decision count mismatch "
                f"(expected {len(candidate_rows)}, got {len(raw_decisions)}, finishReason={finish_reason!r})"
            )

        # candidate_index catches more than a short response: a reordered or
        # duplicated one still has the right length but would otherwise apply
        # the wrong decision to the wrong clip with no error at all. Naming
        # the exact index(es) involved here is also what RAW #120 lacked --
        # a failed attempt previously captured no decisions array at all.
        mismatches = [
            (i, item.get("candidate_index") if isinstance(item, Mapping) else "<malformed>")
            for i, item in enumerate(raw_decisions)
            if not isinstance(item, Mapping) or item.get("candidate_index") != i
        ]
        if mismatches:
            raise UnifiedSelectionUnreliableResponseError(
                "unified Selection decision candidate_index mismatch (expected sequential "
                f"0..{len(candidate_rows) - 1}, mismatches={mismatches[:5]}, finishReason={finish_reason!r})"
            )

        # V2 uses sequence_index as the global story-order authority. A
        # schema-valid response can still repeat an integer, which would make
        # the plan ambiguous and previously failed only after the transport's
        # retry seam. Reject it here so the ordinary bounded provider retry
        # gets one chance to return a complete, unambiguous ordering.
        normalized_sequence: dict[int, int] = {}
        if payload.get("engine_version") == "v2":
            sequence = [item.get("sequence_index") for item in raw_decisions]
            if any(index is None or int(index) < 0 for index in sequence):
                raise UnifiedSelectionUnreliableResponseError(
                    "unified Selection V2 sequence_index missing or negative"
                )
            # Structured output constrains each value but cannot express
            # array-wide uniqueness. Gemini can repeatedly return ties even
            # after a paid retry. Resolve a tie without inventing editorial
            # order: stable-rank by the model's requested sequence first and
            # candidate/source order second.
            ranked = sorted(range(len(sequence)), key=lambda i: (int(sequence[i]), i))
            normalized_sequence = {candidate_index: rank for rank, candidate_index in enumerate(ranked)}

        decisions = []
        for candidate, item in zip(candidate_rows, raw_decisions):
            replacement_index = item.get("trailing_replacement_candidate_index")
            if replacement_index is not None and (type(replacement_index) is not int or not 0 <= replacement_index < len(candidate_rows)):
                raise UnifiedSelectionUnreliableResponseError("invalid trailing replacement candidate index")
            decisions.append(UnifiedSelectionDecision(
                clip_id=str(candidate["clip_id"]),
                action=str(item.get("action") or ""),
                relation=str(item.get("relation") or ""),
                confidence=float(item.get("confidence", -1.0)),
                family_index=int(item.get("family_index", -1)),
                reason_code=str(item.get("reason_code") or ""),
                trailing_recording_word_count=item.get("trailing_recording_word_count", 0),
                trailing_recording_confidence=item.get("trailing_recording_confidence"),
                trailing_recording_kind=item.get("trailing_recording_kind", "recording_aside"),
                trailing_replacement_clip_id=(str(candidate_rows[replacement_index]["clip_id"])
                                              if replacement_index is not None else None),
                sequence_index=(
                    normalized_sequence.get(len(decisions), int(item.get("sequence_index")))
                    if item.get("sequence_index") is not None else None
                ),
            ))
        competitions = _parse_take_competitions(raw, candidate_rows) if payload.get("engine_version") == "v2" else ()
        return decisions, competitions, output_tokens

    def _max_affordable_output_tokens(self, input_tokens: int) -> int:
        """The largest output budget a fresh call at this input size could
        reserve right now, given the ledger's remaining balance. Used to cap
        a retry's bumped reserve so growing the schema (or the bump itself)
        can never be the reason a genuinely retryable failure never gets a
        second attempt -- see the RAW #121 note in reason() below."""
        input_cost = self.settings.estimate_cost_usd(input_tokens=input_tokens, output_tokens=0, escalation=False)
        budget_for_output = max(0.0, self.ledger.remaining_usd - input_cost)
        rate = self.settings.primary_output_per_million_usd
        if rate <= 0:
            return self.max_output_tokens
        return int(budget_for_output / (rate / 1_000_000.0))

    def reason(self, draft: DraftTimeline) -> UnifiedSelectionPlan:
        if not self.api_key:
            raise ValueError("Gemini API key required")
        if not self.settings.enabled or self.settings.provider != "google":
            raise RuntimeError("unified Selection paid transport is disabled")
        if self.model not in {self.settings.primary_model, self.settings.escalation_model}:
            raise ValueError("Gemini model not approved by provider policy")

        payload = build_unified_selection_payload(draft)
        candidate_rows = payload["candidates"]
        payload_chars = len(json.dumps(payload, ensure_ascii=False))
        input_tokens = estimate_tokens_from_chars(payload_chars)
        if self.audiovisual_parts:
            body = build_unified_selection_request(payload, max_output_tokens=self.max_output_tokens)
            body["contents"][0]["parts"].extend(self.audiovisual_parts)
            response = self.session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:countTokens",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json={"contents": body["contents"]}, timeout=self.timeout_sec,
            )
            response.raise_for_status()
            input_tokens = response.json().get("totalTokens")
            if type(input_tokens) is not int or input_tokens <= 0:
                raise ValueError("V2 audiovisual selection token preflight unavailable")
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                'unified Selection input token budget exceeded '
                f'(input_tokens={input_tokens}, max_input_tokens={self.max_input_tokens}, '
                f'candidate_count={len(candidate_rows)})'
            )

        # Exact worst-case budget for the schema actually sent, not a guessed
        # constant -- see output_token_reserve()/_worst_case_decision_json_chars().
        output_reserve = output_token_reserve(len(candidate_rows), ceiling=self.max_output_tokens,
                                             v2=payload.get("engine_version") == "v2")

        for attempt in range(self.max_retries + 1):
            estimated_cost = self.settings.estimate_cost_usd(
                input_tokens=input_tokens,
                output_tokens=output_reserve,
                escalation=False,
            )
            if not self.settings.allows_estimated_session_cost(estimated_cost):
                raise RuntimeError("unified Selection session cost cap exceeded")
            if not self.ledger.reserve(estimated_cost):
                raise RuntimeError("unified Selection edit dollar budget exhausted")

            try:
                decisions, competitions, output_tokens = self._call_once(
                    payload, candidate_rows, output_tokens_requested=output_reserve,
                )
            except UnifiedSelectionProviderBlockedError:
                # The provider processed input tokens before refusing output;
                # retain the reservation because the charged amount is unknown.
                raise
            except (requests.RequestException, UnifiedSelectionUnreliableResponseError):
                # A failed attempt bills no real generation tokens, so give the
                # preflight reservation back rather than leaking it -- otherwise
                # the retry (or a later call in the same session) could be
                # starved by budget locked up for a call that produced nothing.
                self.ledger.release(estimated_cost)
                if attempt < self.max_retries:
                    # RAW #121: adding candidate_index (needed to fix RAW
                    # #120's undercount) grew the schema just enough that this
                    # naive 1.5x bump alone exceeded the tiny default per-edit
                    # cost cap ($0.0075), so a genuinely retryable failure --
                    # e.g. a candidate_index mismatch, which has nothing to do
                    # with token budget -- could never get a second attempt at
                    # all: the retry died on "budget exhausted" before ever
                    # making a call. Cap the bump at what the ledger can
                    # actually afford right now, and give up cleanly (surface
                    # the original failure) rather than loop with a reserve
                    # too small to even repeat the failed attempt.
                    bumped = max(output_reserve, int(output_reserve * 1.5))
                    affordable = self._max_affordable_output_tokens(input_tokens)
                    next_reserve = min(self.max_output_tokens, bumped, affordable)
                    if next_reserve < output_reserve:
                        raise
                    output_reserve = next_reserve
                    continue
                raise
            else:
                actual_cost = self.settings.estimate_cost_usd(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    escalation=False,
                )
                if actual_cost < estimated_cost:
                    self.ledger.release(estimated_cost - actual_cost)
                competition_review = None
                if payload.get("engine_version") == "v2":
                    if competitions:
                        competition_review = {"status": "first_pass_present",
                                              "competition_count": len(competitions)}
                    else:
                        additional, competition_review = self._review_missing_competitions(
                            payload, candidate_rows, decisions,
                        )
                        competitions = tuple((*competitions, *additional))
                links, evidence = self._verify_adjacent_continuations(draft, decisions)
                decisions = _reconcile_verified_continuation_order(decisions, links)
                return UnifiedSelectionPlan(
                    decisions=tuple(decisions),
                    provider="google",
                    model=self.model,
                    requested=True,
                    available=True,
                    estimated_input_tokens=input_tokens,
                    estimated_output_tokens=output_tokens,
                    take_competitions=competitions,
                    continuation_links=links,
                    continuation_evidence=evidence,
                    competition_review=competition_review,
                )
