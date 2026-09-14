"""D-136 (D-127 Phase 2) -- concrete OpenAI-backed `MultimodalBestTakeArbiter`.

docs/CUTSELL_DECISIONS.md D-136, docs/CUTSELL_MULTIMODAL_FALLBACK_ARBITER_
FORENSIC_D127.md Sections 17/21 (Phase 2). Bounded, OFFLINE-EVAL-ONLY
concrete implementation of the `MultimodalBestTakeArbiter` Protocol
(`multimodal_besttake_arbiter.py`) -- mirrors `visual_openai.
OpenAIVisualProvider`'s exact provider pattern (same `OpenAI().responses.
create(...)` call shape, same `input_text`/`input_image` content blocks,
same `client_factory` injection point every `*_openai.py` provider in this
repo already uses for tests). The Protocol interface itself
(`arbitrate(request) -> response`) is UNCHANGED -- no concrete
incompatibility was found that would require touching it.

THIS CLASS IS NEVER IMPORTED BY THE LIVE PIPELINE. `pipeline.py` imports
only `detect_class_b_trigger`/`fallback_trigger_diagnostics` from
`multimodal_besttake_fallback.py`; nothing in `pipeline.py`, `flow_b.py`,
or any production call path imports this module, `multimodal_besttake_
arbiter.py`, or `NullMultimodalBestTakeArbiter` -- confirmed structurally
by `tests/test_cutsell_d136_multimodal_besttake_openai_phase2.py::
test_no_live_pipeline_import_or_instantiation`.

AUDIO PERCEPTION: NOT AVAILABLE. This provider sends TEXT (transcript,
timing, CASE B aggregate evidence, semantic/DeliveryScorer summaries) and
sampled IMAGE frames only -- exactly like the existing `OpenAIVisualProvider`
it mirrors. No audio bytes are ever constructed, read, encoded, or sent to
the provider anywhere in this module. This task never claims the arbiter
"hears" a take, only that it "sees the sampled frames and reads the
transcript/metadata."
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .multimodal_besttake_arbiter import (
    MultimodalBestTakeRequest,
    MultimodalBestTakeResponse,
)
from .openai_json import parse_json_object

#: D-138 PROMPT CONTRACT (Phase 2B decision-contract hardening, general --
#: never Video00/fixture-specific, never a Human Gold/Cut.ai reference leak).
#:
#: D-137 ran this arbiter for real (CI run 34205717833) and found the OLD
#: (D-136) instruction below systematically defaulted to a confident
#: BEST_TAKE even on cases designed to require EQUIVALENT/UNCERTAIN, and
#: once reversed an already-resolved semantic-vs-DeliveryScorer disagreement
#: (D-123/D-128's own protected territory) -- see docs/CUTSELL_DECISIONS.md
#: D-137. This D-138 instruction is a GENERAL decision-contract hardening in
#: response: it teaches the canonical decision hierarchy (meaning
#: sufficiency -> relationship check -> performance quality -> editability
#: -> abstention), states explicitly that BEST_TAKE is not the default, and
#: names the structural signal shape (semantic_label vs deliveryscore_
#: summary disagreement) that means an upstream conflict is already
#: resolved and not this arbiter's to re-open. Every instruction here
#: refers only to structural field names/shapes present in every request
#: (meaning_sufficient, semantic_label, deliveryscore_summary, case_b_*,
#: boundary_editability_note) -- never a fixture transcript, phrase, or
#: Video00-specific detail, and never a reference/oracle value (this
#: arbiter is never given an expected outcome; scoring against one happens
#: entirely outside this request, in the eval harness, after the fact).
_OLD_D136_INSTRUCTION = (
    "You are a bounded BestTake arbiter comparing a small number of candidate "
    "takes of the same short marketing/UGC video segment. For each candidate you "
    "are given its transcript, timing, structured metadata, and one or more "
    "sampled visual frames spanning the take (not the full video). Reason as "
    "follows, in order: "
    "(1) every supplied candidate has ALREADY passed a meaning-sufficiency check "
    "upstream -- do not re-judge whether its spoken content is adequate, only "
    "compare delivery usability; "
    "(2) compare the candidates' delivery/performance quality, not their content; "
    "(3) consider any visible fumble, reset, or camera-disengagement moment; "
    "(4) consider performance continuity across the whole take, not one frame; "
    "(5) do not reward maximum motion or energy for its own sake; "
    "(6) consider how editable each candidate is -- a take with a clean core and a "
    "trimmable entry or exit issue is not the same as a take that is unusable "
    "throughout; "
    "(7) do not penalize a removable ENTRY- or EXIT-only defect as a whole-take "
    "failure -- prefer GOOD_TAKE_TRIM_ENTRY or GOOD_TAKE_TRIM_EXIT for that shape; "
    "(8) choose exactly one supplied candidate id as BEST_TAKE, or EQUIVALENT if "
    "genuinely tied, or UNCERTAIN if the evidence does not support a confident "
    "choice. Never name a candidate id that was not supplied to you. "
    'Return JSON only, no prose: {"outcome":"BEST_TAKE|EQUIVALENT|'
    'GOOD_TAKE_TRIM_ENTRY|GOOD_TAKE_TRIM_EXIT|UNCERTAIN",'
    '"best_take_candidate_id":"<id or null>","confidence":0.0,'
    '"reason":"<one short sentence>"}.'
)

INSTRUCTION = (
    "You are a bounded BestTake arbiter comparing a small number of candidate "
    "takes of the same short marketing/UGC video segment. For each candidate you "
    "are given its transcript, timing, structured metadata, and one or more "
    "sampled visual frames spanning the take (not the full video). "
    "You are NOT required to pick a winner. Two of the five valid outcomes "
    "(EQUIVALENT, UNCERTAIN) exist precisely because a forced choice is "
    "sometimes the wrong answer -- do not default to BEST_TAKE merely "
    "because candidates were supplied to compare. "
    "Reason in this fixed order: "
    "STEP 1 MEANING SUFFICIENCY -- every supplied candidate has ALREADY "
    "passed a meaning-sufficiency check upstream (see each candidate's own "
    "meaning_sufficient field); never re-judge whether its spoken content "
    "is adequate, and never select a candidate for content reasons alone. "
    "STEP 2 RELATIONSHIP CHECK -- before comparing performance, decide what "
    "relationship the candidates actually have: (a) equivalent alternative "
    "realizations of the same message, (b) complementary material where "
    "each candidate carries different required or additional audience-"
    "facing information, (c) genuinely ambiguous or insufficient evidence "
    "to tell, (d) the same valid take differing only by a removable edge "
    "defect, or (e) genuine performance competitors for the exact same "
    "complete message. If the candidates look like shape (b) -- carrying "
    "different required or additional information rather than true "
    "alternative realizations of one message -- you must return UNCERTAIN: "
    "never collapse complementary content into a single winner, never "
    "invent a composite, and never decide which unique fact could be "
    "dropped; that decision belongs to a different, upstream authority. "
    "STEP 2B STRUCTURED-CONFLICT CHECK -- each candidate's semantic_label "
    "and deliveryscore_summary come from two separate, already-existing "
    "upstream evaluators. If one candidate is semantic_label=\"winner\" "
    "while a DIFFERENT candidate has the higher deliveryscore_summary, that "
    "shape means an upstream structural disagreement between those two "
    "evaluators has ALREADY been evaluated and is not yours to re-open: "
    "respect the semantic_label=\"winner\" candidate, or return UNCERTAIN, "
    "unless the sampled frames show direct, overwhelming visual evidence "
    "that the semantic-winner candidate is genuinely unusable -- not merely "
    "that you subjectively prefer the other candidate's performance. "
    "STEP 3 PERFORMANCE QUALITY -- only once the candidates are confirmed "
    "genuine performance competitors for the same complete message "
    "(relationship (e), with no unresolved conflict from step 2B) does "
    "delivery/performance quality decide the outcome. Consider performance "
    "continuity across the whole take, not one frame; consider any visible "
    "fumble, reset, camera disengagement, broken character, or unstable "
    "delivery DURING the required spoken content as a legitimate defect. "
    "Ordinary hand movement, expressive gesture, personality, and higher or "
    "lower energy are NOT automatically defects -- do not equate more "
    "motion with worse or more energy with better; judge coherence, "
    "confidence, natural delivery, performance continuity, and audience "
    "usability, never raw activity level. Each candidate's case_b_* fields "
    "are FACTUAL counts and durations from a structured evidence layer, "
    "never a pre-computed score -- a higher event count does not by itself "
    "mean a candidate is worse; visually judge what those events actually "
    "represent before treating them as meaningful. "
    "STEP 4 EDITABILITY / BOUNDARY -- a removable defect that exists only "
    "before or after the required spoken content (see each candidate's own "
    "boundary_editability_note, when present) does NOT make an otherwise-"
    "good take globally worse. If a candidate's core delivery is good and "
    "the only issue is trimmable entry or exit material, prefer "
    "GOOD_TAKE_TRIM_ENTRY or GOOD_TAKE_TRIM_EXIT over BEST_TAKE or "
    "rejecting that candidate -- never collapse a removable edge issue "
    "into a whole-take failure verdict. "
    "STEP 5 ABSTENTION -- when the evidence does not clearly and safely "
    "establish one candidate as genuinely superior for the same complete "
    "message, UNCERTAIN or EQUIVALENT is the CORRECT and preferred "
    "outcome, not a failure to decide. Confidence measures how strong the "
    "evidence is, never how decisive you should sound -- a confident "
    "UNCERTAIN (high confidence that the case is genuinely ambiguous) is a "
    "valid, safe, successful outcome, and manufacturing certainty on tied, "
    "mixed, or insufficient evidence is a mistake, not a virtue. "
    "You have vision (sampled frames) and text (transcript, timing, "
    "structured metadata) only -- you cannot hear tone, cadence, "
    "pronunciation, or audio clipping, and must never judge or claim to "
    "judge anything you were not actually given a way to perceive. "
    "BEST_TAKE is valid ONLY when ALL of the following hold: the "
    "candidates sufficiently communicate the same complete message; one "
    "candidate has clearly superior audience-facing delivery, visible in "
    "the supplied evidence; the difference is not merely a removable "
    "entry/exit issue; and the evidence is strong enough that choosing one "
    "candidate is safer than abstaining. If any of these do not clearly "
    "hold, do not return BEST_TAKE. Return EQUIVALENT when both candidates "
    "sufficiently communicate the same message and any performance "
    "difference is minor, non-material, or merely stylistic -- do not "
    "force a preference. Never name a candidate id that was not supplied "
    "to you. "
    "In your \"reason\", identify which general basis applies -- "
    "CLEAR_PERFORMANCE_SUPERIORITY, EQUIVALENT_PERFORMANCE, "
    "REMOVABLE_ENTRY_DEFECT, REMOVABLE_EXIT_DEFECT, "
    "COMPLEMENTARY_OR_RELATIONSHIP_AMBIGUITY, MIXED_PERFORMANCE_EVIDENCE, "
    "INSUFFICIENT_VISUAL_EVIDENCE, or STRUCTURED_CONFLICT_UNRESOLVED -- "
    "then add one short supporting sentence. "
    'Return JSON only, no prose: {"outcome":"BEST_TAKE|EQUIVALENT|'
    'GOOD_TAKE_TRIM_ENTRY|GOOD_TAKE_TRIM_EXIT|UNCERTAIN",'
    '"best_take_candidate_id":"<id or null>","confidence":0.0,'
    '"reason":"<BASIS_LABEL: one short sentence>"}.'
)


@dataclass
class OpenAIMultimodalBestTakeArbiter:
    """Concrete, OFFLINE-EVAL-ONLY `MultimodalBestTakeArbiter` implementation.

    `model` defaults to a vision-capable OpenAI model (matching `visual_
    openai.OpenAIVisualProvider`'s own default). `client_factory` mirrors
    every other `*_openai.py` provider in this repo for test injection --
    the eval harness's own bounded live run leaves it `None` (real
    `openai.OpenAI()`, reading `OPENAI_API_KEY` from the environment);
    every targeted test always injects a fake client.

    After a successful `.arbitrate(...)` call, `self.last_call_metadata`
    holds a small dict (`latency_ms`, `request_candidate_count`,
    `frame_count`, `usage`) for the eval harness's own cost/latency
    observability -- kept OUTSIDE the `MultimodalBestTakeResponse` return
    value so the Protocol/dataclass interface stays byte-for-byte
    unchanged (per this task's "do not modify the interface contract
    unless a concrete incompatibility is proven" instruction)."""

    model: str = "gpt-4o-mini"
    client_factory: Callable[[], object] | None = None
    last_call_metadata: dict | None = None

    def _client(self):
        if self.client_factory is not None:
            return self.client_factory()
        from openai import OpenAI
        return OpenAI()

    @staticmethod
    def _image_url(path: str) -> str:
        raw = Path(path).read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    def _content_for(self, request: MultimodalBestTakeRequest) -> list[dict]:
        content: list[dict] = [{"type": "input_text", "text": INSTRUCTION}]
        content.append({
            "type": "input_text",
            "text": json.dumps({
                "family_id": request.family_id,
                "proposition_context": request.proposition_context,
                "candidate_ids": [f.candidate_id for f in request.finalists],
            }, ensure_ascii=False),
        })
        for finalist in request.finalists:
            content.append({
                "type": "input_text",
                "text": json.dumps({
                    "candidate_id": finalist.candidate_id,
                    "transcript": finalist.transcript_text,
                    "source_asset_id": finalist.source_asset_id,
                    "source_start_sec": finalist.source_start,
                    "source_end_sec": finalist.source_end,
                    # D-136: already-computed upstream fact, never re-judged.
                    "meaning_sufficient": finalist.meaning_sufficient,
                    "semantic_label": finalist.semantic_label,
                    "semantic_confidence": finalist.semantic_confidence,
                    "deliveryscore_summary": finalist.deliveryscore_summary,
                    "boundary_editability_note": finalist.boundary_editability_note,
                    # D-122 CASE B factual aggregates only -- never a score.
                    "case_b_delivery_event_count": finalist.case_b_evidence.delivery_event_count,
                    "case_b_delivery_event_duration_total": finalist.case_b_evidence.delivery_event_duration_total,
                    "case_b_count_by_kind": dict(finalist.case_b_evidence.count_by_kind),
                    "case_b_duration_by_kind": dict(finalist.case_b_evidence.duration_by_kind),
                    "case_b_event_density": finalist.case_b_evidence.event_density,
                    "sampled_frame_count": len(finalist.sampled_frame_references),
                }, ensure_ascii=False),
            })
            for frame_path in finalist.sampled_frame_references:
                content.append({
                    "type": "input_image",
                    "image_url": self._image_url(frame_path),
                    "detail": "low",
                })
        return content

    def arbitrate(self, request: MultimodalBestTakeRequest) -> MultimodalBestTakeResponse:
        content = self._content_for(request)
        client = self._client()
        started = time.monotonic()
        response = client.responses.create(model=self.model, input=[{"role": "user", "content": content}])
        latency_ms = round((time.monotonic() - started) * 1000, 1)

        usage = getattr(response, "usage", None)
        if usage is not None and not isinstance(usage, dict):
            usage = getattr(usage, "model_dump", lambda: None)() or getattr(usage, "__dict__", None)
        self.last_call_metadata = {
            "latency_ms": latency_ms,
            "request_candidate_count": len(request.finalists),
            "frame_count": sum(len(f.sampled_frame_references) for f in request.finalists),
            "usage": usage,
        }

        data = parse_json_object(response.output_text)
        outcome = str(data.get("outcome") or "")
        raw_candidate_id = data.get("best_take_candidate_id")
        confidence_value = data.get("confidence")
        reason = str(data.get("reason") or "")

        return MultimodalBestTakeResponse(
            family_id=request.family_id,
            outcome=outcome,
            best_take_candidate_id=(str(raw_candidate_id) if raw_candidate_id else None),
            confidence=float(confidence_value) if confidence_value is not None else 0.0,
            reason=reason,
            provider="openai",
            model=self.model,
            requested=True,
            available=True,
        )
