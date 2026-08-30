"""Gemini-backed whole-video Selection reasoner for CutSell.

Unlike the legacy bounded Hybrid judge, this authority sees the complete candidate
universe for one source in a single request.  It asks the model to form idea/retry
families, recognize composites and continuations, and assign final semantic membership
before Selection freeze.  Boundary ownership remains elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import requests

from .contracts import DraftTimeline
from .hybrid_google_transport import DollarBudgetLedger
from .hybrid_payload import estimate_tokens_from_chars
from .hybrid_provider_settings import HybridProviderSettings
from .unified_selection_reasoner import (
    UnifiedSelectionDecision,
    UnifiedSelectionPlan,
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


def _source_context(draft: DraftTimeline) -> dict[str, Any]:
    raw = (draft.diagnostics or {}).get("whole_video_context") or {}
    if not isinstance(raw, Mapping):
        return {}
    sources = raw.get("sources") or []
    compact_sources = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        compact_sources.append({
            "source_asset_id": source.get("source_asset_id"),
            "summary": str(source.get("summary") or "")[:3000],
            "creator_intent": str(source.get("creator_intent") or "")[:600],
            "main_topic": str(source.get("main_topic") or "")[:500],
            "story_logic": str(source.get("story_logic") or "")[:1000],
            "dominant_style": str(source.get("dominant_style") or "")[:300],
            "edit_mode": str(source.get("edit_mode") or "")[:80],
        })
    return {
        "dominant_edit_mode": raw.get("dominant_edit_mode"),
        "sources": compact_sources,
    }


def build_unified_selection_payload(draft: DraftTimeline) -> dict[str, Any]:
    candidates = _candidate_universe(draft)
    if not candidates:
        raise ValueError("unified selection requires at least one candidate")
    return {
        "task": "cutsell_unified_whole_video_selection",
        "source_context": _source_context(draft),
        "editorial_contract": [
            "Understand the full creator message before deciding any individual take.",
            "First infer idea families and retry relationships across the entire timeline.",
            "A genuine retry family (competing takes of the same moment, relation retry_winner/retry_alternate) produces exactly ONE SELECT: the single cleanest complete delivery. Every other candidate in that same contest is a SWAP, never an additional SELECT, no matter how usable it is on its own.",
            "SELECT independent valid story coverage, the one winning retry per family, necessary continuations, and every clean piece needed for a composite best take.",
            "SWAP a usable alternative or redundant delivery that should not play by default but remains useful for manual replacement -- this is the correct action any time your own reason for keeping a clip is that it is merely a usable/redundant alternative, never SELECT.",
            "DISCARD only recording-process BTS, failed/abandoned delivery, or an inferior retry with no unique audience-facing information -- if you judge a clip's delivery failed or was abandoned, it must never be SELECT either.",
            "When visual_evidence is present for a candidate, use it as real evidence, not decoration: low motion_stability/expression_naturalness/gesture_naturalness or high visual_fumble/distraction_risk are signs of a visible reset, stumble, or camera-disengagement moment within that take. An incomplete, stumbled, or visually-reset take must not beat a cleaner, complete competing retry in the same family unless it has clearly stronger evidence (better visual_evidence AND a genuinely more complete delivery) -- being merely present or first is not evidence.",
            "Do not prefer a monolithic take merely because it is longer; a human-quality composite of cleaner micro-deliveries may be better.",
            "Do not treat adjacent valid statements as retries just because they share topic words.",
            "Preserve numbers, negations, names, causal claims, and genuinely new story facts.",
            "Natural source story order is authoritative; do not reorder candidates.",
            "WHEN UNCERTAIN, preserve content rather than destructively deleting it -- prefer SWAP over SELECT when uncertain which retry is best.",
        ],
        "candidates": candidates,
    }


def unified_selection_response_schema(candidate_count: int) -> dict[str, Any]:
    # `candidate_count` is accepted for call-site/API compatibility but deliberately
    # NOT encoded as an exact minItems==maxItems array bound: an isolation probe
    # (scripts/isolate_unified_selection_schema.py, see
    # docs/claude-handoff/CUTSELL_COMPLETE_HANDOFF.md) proved Gemini's structured
    # -output validator rejects an exact-length array bound at whole-video scale
    # (works at 5 candidates, 400s at 90) -- even with this same model and even with
    # the smaller/simpler schema that cutsell-hybrid-llm-bakeoff.yml already proved
    # works. A second isolation probe (scripts/isolate_unified_selection_
    # cardinality.py) confirmed this holds even for a LOOSE band (minItems=N-2,
    # maxItems=N+2), not just an exact bound: 8/8 trials 400'd at the real
    # Video00 candidate count (32) either way. No length constraint of any kind
    # belongs in this schema. `reason()` below already raises on any
    # decision-count or candidate_index mismatch after the response comes
    # back, so dropping the schema-level bound loses no correctness guarantee.
    del candidate_count
    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        # RAW #120 saw a normal-STOP response return 31
                        # decisions for 32 candidates -- no truncation, the
                        # model just undercounted. candidate_index requires
                        # the model to state which candidate each decision is
                        # for; the same isolation probe found this alone gets
                        # 8/8 trials with the exactly right count, and it lets
                        # _call_once() catch (with the specific index named)
                        # not just a short response but also a reordered or
                        # duplicated one that a bare length check would miss.
                        "candidate_index": {"type": "integer", "minimum": 0},
                        "action": {"type": "string", "enum": _ACTIONS},
                        "relation": {"type": "string", "enum": _RELATIONS},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "family_index": {"type": "integer", "minimum": 0},
                        "reason_code": {"type": "string", "enum": _REASON_CODES},
                    },
                    "required": [
                        "candidate_index", "action", "relation", "confidence", "family_index", "reason_code",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _worst_case_decision_json_chars() -> int:
    """Exact worst-case serialized length of one decision object, derived from
    the real enum values rather than a guessed constant. RAW #119 truncated
    (finishReason MAX_TOKENS -> malformed JSON) because the previous output
    token reserve (36 chars/candidate, chosen without reference to the actual
    schema) under-provisioned relative to this: the longest reason_code alone
    ("independent_story_coverage") is 27 characters, and that is also the
    reason_code the model chooses most often in practice."""
    sample = {
        "candidate_index": 999,
        "action": max(_ACTIONS, key=len),
        "relation": max(_RELATIONS, key=len),
        "confidence": 0.95,
        "family_index": 999,
        "reason_code": max(_REASON_CODES, key=len),
    }
    return len(json.dumps(sample, separators=(",", ":")))


# +1 char reserves the trailing comma between array items; this constant is
# derived from the schema above, so it can never silently drift out of date
# the way a hand-picked "tokens per candidate" guess could.
_TOKENS_PER_DECISION = estimate_tokens_from_chars(_worst_case_decision_json_chars() + 1)
_DECISION_ARRAY_OVERHEAD_TOKENS = estimate_tokens_from_chars(len('{"decisions":[]}') + 8)


def output_token_reserve(candidate_count: int, *, ceiling: int) -> int:
    """Worst-case output token budget for `candidate_count` decisions, capped
    at `ceiling`. Every field in the schema is bounded (enums, a 0-1 float,
    and a small integer), so this is a true upper bound, not a heuristic --
    the model cannot need more tokens than this to state one complete,
    schema-valid decision for every candidate."""
    return min(
        ceiling,
        max(640, _TOKENS_PER_DECISION * max(0, int(candidate_count)) + _DECISION_ARRAY_OVERHEAD_TOKENS),
    )


def build_unified_selection_request(payload: Mapping[str, Any], *, max_output_tokens: int) -> dict[str, Any]:
    candidate_count = len(payload.get("candidates") or ())
    prompt = (
        "You are CutSell's final human-style Selection editor for ONE complete raw creator video. "
        "Do not make isolated clip decisions. Read every candidate first, reconstruct the intended story, "
        "form same-idea retry families, distinguish continuations from retries, and identify when the best "
        "human edit is a composite assembled from multiple clean sub-deliveries. Current buckets, local groups, "
        "and Hybrid votes are evidence only and may be overturned. Return one decision for every candidate in "
        "the exact supplied order. family_index must be the same integer for genuine competing retries or "
        "composite pieces of one idea; use a different family for independent story beats. SELECT means it plays "
        "in the default edit. SWAP means it remains available but does not play. DISCARD is destructive and is "
        "reserved for failed/BTS/inferior duplicate material with no unique audience-facing value. Never delete "
        "information only because wording overlaps. When uncertain, preserve rather than delete. Do not echo IDs "
        "or timestamps. Output only the requested JSON schema. "
        f"You MUST return exactly {candidate_count} decisions, one per candidate, in the same order as the "
        "candidates array. Each decision's candidate_index must equal its zero-based position in that order "
        "(0, 1, 2, ...). Never merge two candidates into one decision and never omit any candidate, even if two "
        "candidates look nearly identical -- they still each need their own decision with their own "
        "candidate_index.\n\n"
        + json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": int(max_output_tokens),
            "thinkingConfig": {"thinkingLevel": "low"},
            "responseMimeType": "application/json",
            "responseJsonSchema": unified_selection_response_schema(candidate_count),
        },
    }


def parse_unified_selection_response(raw: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int, str]:
    """Return (decisions, output_tokens, finish_reason).

    Raises UnifiedSelectionUnreliableResponseError for any shape the response
    could take that must never be treated as a complete editorial result:
    a missing candidate/content/parts/decisions field, or JSON that failed to
    parse -- the latter always names finishReason so a MAX_TOKENS truncation
    is distinguishable from a genuinely malformed generation at a glance.
    """
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise UnifiedSelectionUnreliableResponseError("Gemini unified response missing candidates")
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise UnifiedSelectionUnreliableResponseError("Gemini unified candidate malformed")
    finish_reason = str(first.get("finishReason") or "")
    content = first.get("content")
    if not isinstance(content, Mapping):
        raise UnifiedSelectionUnreliableResponseError(
            f"Gemini unified response missing content (finishReason={finish_reason!r})"
        )
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise UnifiedSelectionUnreliableResponseError(
            f"Gemini unified response missing parts (finishReason={finish_reason!r})"
        )
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, Mapping))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UnifiedSelectionUnreliableResponseError(
            f"Gemini unified response was not valid JSON (finishReason={finish_reason!r}): {exc}"
        ) from exc
    decisions = parsed.get("decisions") if isinstance(parsed, Mapping) else None
    if not isinstance(decisions, list):
        raise UnifiedSelectionUnreliableResponseError(
            f"Gemini unified response missing decisions (finishReason={finish_reason!r})"
        )
    usage = raw.get("usageMetadata") or {}
    try:
        output_tokens = max(0, int(usage.get("candidatesTokenCount") or 0)) if isinstance(usage, Mapping) else 0
    except (TypeError, ValueError):
        output_tokens = 0
    return decisions, output_tokens, finish_reason


@dataclass
class GoogleUnifiedSelectionReasoner:
    api_key: str
    model: str
    settings: HybridProviderSettings
    ledger: DollarBudgetLedger
    timeout_sec: float = 90.0
    session: Any = requests
    max_input_tokens: int = 20_000
    max_output_tokens: int = 4_096
    # RAW #119: Gemini returned a MAX_TOKENS-truncated, unparseable response
    # once, with no retry available -- the pipeline fails open on the very
    # first provider hiccup. One retry, with a larger output token reserve in
    # case truncation was the cause, is the smallest general reliability
    # improvement that does not touch editorial Selection rules at all: it
    # only changes how many attempts a request gets and how large a response
    # budget it asks for. If both attempts still fail, `reason()` still raises
    # and apply_unified_selection_reasoner() still fails open exactly as
    # before -- a real, observable failure, never a partial result applied as
    # if it were complete.
    max_retries: int = 1

    def _call_once(
        self,
        payload: Mapping[str, Any],
        candidate_rows: list[dict[str, Any]],
        *,
        output_tokens_requested: int,
    ) -> tuple[list[UnifiedSelectionDecision], int]:
        body = build_unified_selection_request(payload, max_output_tokens=output_tokens_requested)
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

        decisions = []
        for candidate, item in zip(candidate_rows, raw_decisions):
            decisions.append(UnifiedSelectionDecision(
                clip_id=str(candidate["clip_id"]),
                action=str(item.get("action") or ""),
                relation=str(item.get("relation") or ""),
                confidence=float(item.get("confidence", -1.0)),
                family_index=int(item.get("family_index", -1)),
                reason_code=str(item.get("reason_code") or ""),
            ))
        return decisions, output_tokens

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
        if input_tokens > self.max_input_tokens:
            raise ValueError("unified Selection input token budget exceeded")

        # Exact worst-case budget for the schema actually sent, not a guessed
        # constant -- see output_token_reserve()/_worst_case_decision_json_chars().
        output_reserve = output_token_reserve(len(candidate_rows), ceiling=self.max_output_tokens)

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
                decisions, output_tokens = self._call_once(
                    payload, candidate_rows, output_tokens_requested=output_reserve,
                )
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
                return UnifiedSelectionPlan(
                    decisions=tuple(decisions),
                    provider="google",
                    model=self.model,
                    requested=True,
                    available=True,
                    estimated_input_tokens=input_tokens,
                    estimated_output_tokens=output_tokens,
                )
