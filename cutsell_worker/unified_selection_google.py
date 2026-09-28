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
        if (draft.diagnostics or {}).get("editorial_engine_v2_request"):
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
    contract = [
        "Understand the full creator message before deciding any individual take.",
        "First infer idea families and retry relationships across the entire timeline.",
        "A genuine retry family produces exactly ONE SELECT: the cleanest complete delivery.",
        "SELECT independent valid story coverage, the winning retry, necessary continuations, and every clean composite piece.",
        "DISCARD only recording-process BTS, failed/abandoned delivery, or an inferior retry with no unique audience-facing information.",
        "Use candidate visual_evidence as real performance evidence, not decoration.",
        "Do not prefer a monolithic take merely because it is longer; a clean composite may be better.",
        "When one continuous complete take covers ideas otherwise spread across several earlier fragments, prefer the continuous take when the take, together with its immediately adjacent continuation candidates, covers the same ideas with better continuity and performance. Preserve earlier fragments that add unique audience information, conditions, or useful visual actions, and reject the continuous take when audiovisual evidence shows it failed. Do not call the complete take redundant merely because its ideas recur across a patchwork of selected attempts.",
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
            "For a SELECT containing a clean delivery followed by a short explicit recording-process aside, optionally return trailing_recording_word_count (1..8) counted from aligned_word_texts and trailing_recording_confidence (0..1) for that suffix classification independently of whole-take selection confidence. Return 0 or omit when uncertain, words are unavailable, or the ending is audience content. Never trim a disclaimer, offer, qualification, gratitude, humor, reaction, number, negation or product fact. Do not discard the whole useful take because only its ending is recording talk.",
            "If actual source video/audio is attached, WATCH AND LISTEN to the exact indexed words before classifying a suffix. Broad earlier AV regions are advisory and can miss short defects. A useful take ending in an abandoned restart must be SELECT, with only its failed suffix proposed for removal: trailing_recording_kind='abandoned_restart', trailing_recording_word_count and trailing_replacement_candidate_index pointing to a SELECT that completes the same attempt. Do not discard its unique useful head. Use kind='recording_aside' for explicit off-audience recording talk; preserve intentional audience reactions. Count aligned word ENTRIES, including every connector belonging to the rejected suffix; never invent timestamps or alter transcript words. Inspect the end of EVERY selected take for these two defects. Omit proposals unless the actual performance makes the defect clear.",
        ])
    else:
        contract.extend([
            "Natural source story order is authoritative; do not reorder candidates.",
            "SWAP a usable alternative that should not play by default.",
        ])
    return {
        "task": "cutsell_editorial_engine_v2" if v2_request else "cutsell_unified_whole_video_selection",
        "engine_version": "v2" if v2_request else "legacy",
        "source_context": _source_context(draft, include_audiovisual=v2_request),
        "editorial_contract": contract,
        "candidates": candidates,
    }


def unified_selection_response_schema(candidate_count: int, *, v2: bool = False) -> dict[str, Any]:
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
                        **({"sequence_index": {"type": "integer", "minimum": 0}} if v2 else {}),
                        **({"trailing_recording_word_count": {"type": "integer", "minimum": 0, "maximum": 8}} if v2 else {}),
                        **({"trailing_recording_confidence": {"type": "number", "minimum": 0, "maximum": 1}} if v2 else {}),
                        **({"trailing_recording_kind": {"type": "string", "enum": ["recording_aside", "abandoned_restart"]},
                            "trailing_replacement_candidate_index": {"type": "integer", "minimum": 0}} if v2 else {}),
                    },
                    "required": [
                        "candidate_index", "action", "relation", "confidence", "family_index", "reason_code",
                        *(["sequence_index"] if v2 else []),
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def _worst_case_decision_json_chars(*, v2: bool = False) -> int:
    """Exact worst-case marginal cost, in characters, of one additional
    decision object as it actually appears embedded in Gemini's real
    generated response -- derived by diffing a 1-item and 2-item pretty-
    printed `{"decisions": [...]}` array (indent=2), not by guessing an
    indentation depth or assuming compact serialization.

    RAW #119 truncated because the original reserve (36 chars/candidate,
    chosen without reference to the actual schema) under-provisioned. The
    fix for that used this same real-enum-values technique but assumed a
    COMPACT serialization (`separators=(",", ":")`). That assumption was
    itself wrong: an isolation probe (scripts/isolate_unified_selection_
    output_budget.py) proved Gemini's real structured-output responses are
    pretty-printed with indentation and newlines, not compact -- the same
    probe reproduced RAW run 33316711594's MAX_TOKENS truncation (both
    attempts, head da8bd80) with an UNMODIFIED pre-fix prompt too, ruling
    out that fix's prompt/payload growth as the cause and pointing squarely
    at this compact-vs-pretty mismatch instead. The observed truncation
    point (~193-196 lines, ~1622 output tokens for 32 candidates never
    completing) matches a ~25/32-decision partial response at pretty-print
    sizing almost exactly, and is inconsistent with the old compact-based
    budget ever having been a true upper bound for this provider's actual
    output format."""
    sample = {
        "candidate_index": 999,
        "action": max(_ACTIONS, key=len),
        "relation": max(_RELATIONS, key=len),
        "confidence": 0.95,
        "family_index": 999,
        "reason_code": max(_REASON_CODES, key=len),
        "sequence_index": 999,
        "trailing_recording_word_count": 8,
        **({"trailing_recording_confidence": 0.999999} if v2 else {}),
        **({"trailing_recording_kind": "abandoned_restart", "trailing_replacement_candidate_index": 999} if v2 else {}),
    }
    one = json.dumps({"decisions": [sample]}, indent=2)
    two = json.dumps({"decisions": [sample, sample]}, indent=2)
    return len(two) - len(one)


# A margin on top of the exact pretty-printed marginal-cost measurement above,
# not a replacement for it: char-counting one observed formatting convention
# cannot promise the provider's tokenizer or exact whitespace/indentation
# style will never drift (e.g. a different indent width, or extra newlines).
# Getting caught by that gap once (compact vs. pretty) is the whole reason
# this constant exists instead of trusting the bare character count alone.
_JSON_FORMATTING_SAFETY_MARGIN = 1.20
_TOKENS_PER_DECISION = estimate_tokens_from_chars(
    int(_worst_case_decision_json_chars() * _JSON_FORMATTING_SAFETY_MARGIN)
)
_V2_TOKENS_PER_DECISION = estimate_tokens_from_chars(
    int(_worst_case_decision_json_chars(v2=True) * _JSON_FORMATTING_SAFETY_MARGIN)
)
_DECISION_ARRAY_OVERHEAD_TOKENS = estimate_tokens_from_chars(len('{"decisions":[]}') + 8)


def output_token_reserve(candidate_count: int, *, ceiling: int, v2: bool = False) -> int:
    """Worst-case output token budget for `candidate_count` decisions, capped
    at `ceiling`. Every field in the schema is bounded (enums, a 0-1 float,
    and a small integer), so this is a true upper bound, not a heuristic --
    the model cannot need more tokens than this to state one complete,
    schema-valid decision for every candidate."""
    return min(
        ceiling,
        max(640, (_V2_TOKENS_PER_DECISION if v2 else _TOKENS_PER_DECISION) * max(0, int(candidate_count)) + _DECISION_ARRAY_OVERHEAD_TOKENS),
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
            "responseJsonSchema": unified_selection_response_schema(
                candidate_count, v2=payload.get("engine_version") == "v2"
            ),
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
    audiovisual_parts: tuple = ()

    def _call_once(
        self,
        payload: Mapping[str, Any],
        candidate_rows: list[dict[str, Any]],
        *,
        output_tokens_requested: int,
    ) -> tuple[list[UnifiedSelectionDecision], int]:
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
            raise ValueError("unified Selection input token budget exceeded")

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
